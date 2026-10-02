# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from diffusers import CogVideoXTransformer3DModel, TeaCacheConfig, apply_tea_cache
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.models.cache_utils import CacheMixin


class DualStreamBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states, temb):
        self.calls += 1
        return hidden_states + 1.0, encoder_hidden_states + 2.0


class DualStreamTransformer(torch.nn.Module, CacheMixin):
    def __init__(self, num_blocks=2):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DualStreamBlock() for _ in range(num_blocks)])

    def forward(self, hidden_states, encoder_hidden_states, temb):
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )
        return hidden_states, encoder_hidden_states


@pytest.fixture(autouse=True)
def register_dual_stream_block():
    TransformerBlockRegistry.register(
        DualStreamBlock,
        TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=1),
    )


def _head_state(model):
    hook = model.transformer_blocks[0]._diffusers_hook.get_hook("tea_cache_leader_block")
    return hook.state_manager.get_state()


def test_tea_cache_validation():
    with pytest.raises(ValueError, match="non-empty"):
        TeaCacheConfig(coefficients=[])
    with pytest.raises(ValueError, match="non-negative"):
        TeaCacheConfig(coefficients=[1.0], threshold=-0.1)
    with pytest.raises(ValueError, match="at least 2"):
        TeaCacheConfig(coefficients=[1.0], num_inference_steps=1)


def test_tea_cache_dual_stream_accumulation_and_forced_endpoints():
    model = DualStreamTransformer()
    config = TeaCacheConfig(
        coefficients=(1.0, 0.0),
        threshold=0.15,
        num_inference_steps=5,
    )
    apply_tea_cache(model, config)

    hidden = torch.tensor([[[10.0]]])
    encoder = torch.tensor([[[20.0]]])

    with model.cache_context("trajectory"):
        # Step 0: first step always computes. Two blocks add +2 / +4.
        out_h, out_e = model(hidden, encoder, torch.tensor([[1.0]]))
        torch.testing.assert_close(out_h, hidden + 2.0)
        torch.testing.assert_close(out_e, encoder + 4.0)

        # Step 1: relative change = 0.05, so reuse the full-stack residual.
        out_h, out_e = model(hidden + 1.0, encoder + 1.0, torch.tensor([[1.05]]))
        torch.testing.assert_close(out_h, hidden + 3.0)
        torch.testing.assert_close(out_e, encoder + 5.0)

        # Step 2: compares against the cached step's 1.05 indicator, not step 0's 1.0.
        model(hidden + 2.0, encoder + 2.0, torch.tensor([[1.10]]))
        state = _head_state(model)
        torch.testing.assert_close(state.previous_indicator, torch.tensor([[1.10]]))
        assert state.accumulated_distance == pytest.approx(0.05 + (0.05 / 1.05), rel=1e-5)

        # Step 3 crosses the accumulated threshold and therefore computes fully.
        model(hidden + 3.0, encoder + 3.0, torch.tensor([[1.30]]))

        # Step 4 is the final step and is always computed, irrespective of the gate.
        model(hidden + 4.0, encoder + 4.0, torch.tensor([[1.31]]))

    assert [block.calls for block in model.transformer_blocks] == [3, 3]


def test_tea_cache_contexts_keep_independent_trajectories():
    model = DualStreamTransformer()
    apply_tea_cache(
        model,
        TeaCacheConfig(coefficients=(0.0,), threshold=1.0, num_inference_steps=3),
    )
    hidden = torch.zeros(1, 1, 1)
    encoder = torch.zeros(1, 1, 1)
    temb = torch.ones(1, 1)

    with model.cache_context("cond"):
        model(hidden, encoder, temb)
    with model.cache_context("uncond"):
        model(hidden, encoder, temb)

    # Each context sees its own first step, so both calls execute fully.
    assert [block.calls for block in model.transformer_blocks] == [2, 2]


def test_tea_cache_cache_mixin_lifecycle():
    model = DualStreamTransformer()
    config = TeaCacheConfig(coefficients=(0.0,), threshold=1.0, num_inference_steps=3)

    model.enable_cache(config)
    assert model.is_cache_enabled
    assert model.transformer_blocks[0]._diffusers_hook.get_hook("tea_cache_leader_block") is not None
    assert model.transformer_blocks[1]._diffusers_hook.get_hook("tea_cache_block") is not None

    model.disable_cache()
    assert not model.is_cache_enabled
    assert model.transformer_blocks[0]._diffusers_hook.get_hook("tea_cache_leader_block") is None
    assert model.transformer_blocks[1]._diffusers_hook.get_hook("tea_cache_block") is None


@torch.no_grad()
def test_tea_cache_runs_on_tiny_cogvideox():
    model = CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        time_embed_dim=2,
        text_embed_dim=8,
        num_layers=2,
        sample_width=8,
        sample_height=8,
        sample_frames=8,
        patch_size=2,
        patch_size_t=None,
        temporal_compression_ratio=4,
        max_text_seq_length=8,
    ).eval()
    model.enable_cache(
        TeaCacheConfig(
            coefficients=(0.0,),
            threshold=1.0,
            num_inference_steps=3,
        )
    )

    generator = torch.Generator("cpu").manual_seed(0)
    hidden_states = torch.randn((1, 1, 4, 8, 8), generator=generator)
    encoder_hidden_states = torch.randn((1, 8, 8), generator=generator)

    with model.cache_context("trajectory"):
        first = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=torch.tensor([900]),
            return_dict=False,
        )[0]
        second = model(
            hidden_states=hidden_states + 0.01,
            encoder_hidden_states=encoder_hidden_states,
            timestep=torch.tensor([500]),
            return_dict=False,
        )[0]

        head_hook = model.transformer_blocks[0]._diffusers_hook.get_hook("tea_cache_leader_block")
        state = head_hook.state_manager.get_state()
        assert state.step_index == 2
        assert state.should_compute is False

        third = model(
            hidden_states=hidden_states + 0.02,
            encoder_hidden_states=encoder_hidden_states,
            timestep=torch.tensor([100]),
            return_dict=False,
        )[0]

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.isfinite(third).all()
    assert state.step_index == 0
    assert state.previous_hidden_residual is None


def test_tea_cache_requires_one_multi_block_stack():
    with pytest.raises(ValueError, match="at least two"):
        apply_tea_cache(
            DualStreamTransformer(num_blocks=1),
            TeaCacheConfig(coefficients=(0.0,), num_inference_steps=2),
        )
