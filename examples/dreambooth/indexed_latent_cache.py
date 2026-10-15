"""Identity-keyed VAE latent cache for shuffled DreamBooth training.

Persist latent *distribution parameters*, not sampled latents: later passes
are free to shuffle data and each training step still samples fresh noise.
"""
import torch


def cache_latent_distributions(cache, sample_keys, latent_dist):
    keys = [int(key) for key in sample_keys]
    parameters = latent_dist.parameters
    if len(keys) != len(set(keys)) or len(keys) != parameters.shape[0]:
        raise ValueError("latent cache keys must be unique and batch-aligned")
    if any(key in cache for key in keys):
        raise ValueError("duplicate latent cache sample key")
    for idx, key in enumerate(keys):
        cache[key] = (parameters[idx].detach().clone(), latent_dist.deterministic)


def sample_cached_latents(cache, sample_keys, distribution_class):
    keys = [int(key) for key in sample_keys]
    if not keys:
        raise ValueError("cannot draw empty cached latent minibatch")
    try:
        entries = [cache[key] for key in keys]
    except KeyError as err:
        raise ValueError(f"missing cached latent for sample key {err.args[0]}") from err
    determinism = {entry[1] for entry in entries}
    if len(determinism) != 1:
        raise ValueError("cached VAE distributions disagree on deterministic flag")
    parameters = torch.stack([entry[0] for entry in entries], dim=0)
    return distribution_class(parameters, deterministic=entries[0][1]).sample()
