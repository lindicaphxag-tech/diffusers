"""CPU regressions: SD3 cached VAE latents must follow sample IDs, not steps."""
import ast
from pathlib import Path

import pytest
import torch

from indexed_latent_cache import cache_latent_distributions, sample_cached_latents


class ToyGaussian:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.deterministic = deterministic
        self.mean, self.logvar = parameters.chunk(2, dim=1)

    def sample(self):
        if self.deterministic:
            return self.mean
        return self.mean + torch.exp(self.logvar / 2) * torch.randn_like(self.mean)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("prior", [False, True])
def test_shuffled_identity_and_non_divisible_batch(batch_size, prior):
    cache = {}
    for start in range(0, 7, batch_size):
        group = [6, 1, 4, 3, 0, 5, 2][start : start + batch_size]
        keys = [2 * i for i in group]
        if prior:
            keys += [2 * i + 1 for i in group]
        means = torch.tensor(keys, dtype=torch.float32).view(-1, 1)
        cache_latent_distributions(cache, keys, ToyGaussian(torch.cat([means, torch.zeros_like(means)], dim=1)))
    for epoch_order in ([2, 0, 6, 3, 1, 4, 5], [0, 1, 2, 3, 4, 5, 6]):
        for start in range(0, 7, batch_size):
            group = epoch_order[start : start + batch_size]
            keys = [2 * i for i in group]
            if prior:
                keys += [2 * i + 1 for i in group]
            torch.manual_seed(17)
            actual = sample_cached_latents(cache, keys, ToyGaussian)
            torch.manual_seed(17)
            expected = torch.tensor(keys, dtype=torch.float32).view(-1, 1) + torch.randn(len(keys), 1)
            torch.testing.assert_close(actual, expected)
    assert len(cache) == 7 * (2 if prior else 1)


def test_duplicates_missing_and_empty_fail_closed():
    cache = {}
    params = ToyGaussian(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="unique"):
        cache_latent_distributions(cache, [1, 1], params)
    cache_latent_distributions(cache, [1, 2], params)
    with pytest.raises(ValueError, match="duplicate"):
        cache_latent_distributions(cache, [1, 3], params)
    with pytest.raises(ValueError, match="missing"):
        sample_cached_latents(cache, [1, 3], ToyGaussian)
    with pytest.raises(ValueError, match="empty"):
        sample_cached_latents(cache, [], ToyGaussian)


def test_cached_parameters_detached_but_each_step_resamples():
    params = torch.randn(2, 2, requires_grad=True)
    cache = {}
    cache_latent_distributions(cache, [20, 21], ToyGaussian(params))
    assert not cache[20][0].requires_grad
    torch.manual_seed(19)
    a = sample_cached_latents(cache, [20, 21], ToyGaussian)
    torch.manual_seed(23)
    b = sample_cached_latents(cache, [20, 21], ToyGaussian)
    assert not torch.equal(a, b)


def test_identity_keys_follow_collate_pixel_and_caption_order():
    # Extract the actual production collate function without importing huge
    # SD3 checkpoints or optional training dependencies.
    file = Path(__file__).resolve().parent / "train_dreambooth_lora_sd3.py"
    root = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    function = next(node for node in root.body if isinstance(node, ast.FunctionDef) and node.name == "collate_fn")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(file), "exec"), namespace)
    collate = namespace["collate_fn"]
    examples = [
        {"sample_index": 4, "instance_images": torch.full((1, 2, 2), 4.0),
         "class_images": torch.full((1, 2, 2), 40.0), "instance_prompt": "caption-4", "class_prompt": "class"},
        {"sample_index": 1, "instance_images": torch.full((1, 2, 2), 1.0),
         "class_images": torch.full((1, 2, 2), 10.0), "instance_prompt": "caption-1", "class_prompt": "class"},
    ]
    batch = collate(examples, with_prior_preservation=True)
    assert batch["sample_keys"].tolist() == [8, 2, 9, 3]
    assert batch["prompts"] == ["caption-4", "caption-1", "class", "class"]
    torch.testing.assert_close(batch["pixel_values"][:, 0, 0, 0], torch.tensor([4.0, 1.0, 40.0, 10.0]))
    plain = collate(examples, with_prior_preservation=False)
    assert plain["sample_keys"].tolist() == [8, 2]



def test_old_step_cache_mispairs_captions_under_permutation_but_identity_cache_does_not():
    n = 11
    batch_size = 3
    # Mimic the original step-indexed cache traversal: batches occupy slots
    # 0..num_batches-1 and are reused by step in a different shuffled epoch.
    first_order = torch.randperm(n, generator=torch.Generator().manual_seed(11)).tolist()
    old_step_cache = [first_order[i : i + batch_size] for i in range(0, n, batch_size)]
    keyed_cache = {}
    for original_id in first_order:
        params = torch.tensor([[float(original_id), 0.0]])
        cache_latent_distributions(keyed_cache, [2 * original_id], ToyGaussian(params, deterministic=True))
    mismatched_old_pairs = 0
    for seed in range(100):
        epoch_order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 200)).tolist()
        for batch_step, start in enumerate(range(0, n, batch_size)):
            actual_ids = epoch_order[start : start + batch_size]
            restored = sample_cached_latents(keyed_cache, [2 * i for i in actual_ids], ToyGaussian)
            torch.testing.assert_close(restored[:, 0], torch.tensor(actual_ids, dtype=torch.float32))
            stale_ids = old_step_cache[batch_step]
            mismatched_old_pairs += sum(a != b for a, b in zip(actual_ids, stale_ids))
    assert mismatched_old_pairs > 0


def test_deterministic_distribution_preserves_mode_and_rejects_misaligned_batch():
    cache = {}
    with pytest.raises(ValueError, match="batch-aligned"):
        cache_latent_distributions(cache, [1], ToyGaussian(torch.zeros(2, 2)))
    cache_latent_distributions(cache, [9], ToyGaussian(torch.tensor([[2.0, 0.0]]), deterministic=True))
    torch.testing.assert_close(sample_cached_latents(cache, [9, 9], ToyGaussian), torch.tensor([[2.0], [2.0]]))
