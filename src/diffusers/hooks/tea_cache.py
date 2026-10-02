# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import inspect
import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ..utils import logging
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from .hooks import BaseState, CacheContext, HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_TEA_CACHE_LEADER_BLOCK_HOOK = "tea_cache_leader_block"
_TEA_CACHE_BLOCK_HOOK = "tea_cache_block"


COGVIDEOX_2B_TEACACHE_COEFFICIENTS = (
    -3.10658903e01,
    2.54732368e01,
    -5.92380459e00,
    1.75769064e00,
    -3.61568434e-03,
)
COGVIDEOX_5B_TEACACHE_COEFFICIENTS = (
    -1.53880483e03,
    8.43202495e02,
    -1.34363087e02,
    7.97131516e00,
    -5.23162339e-02,
)
COGVIDEOX_5B_I2V_TEACACHE_COEFFICIENTS = COGVIDEOX_5B_TEACACHE_COEFFICIENTS
COGVIDEOX1_5_5B_TEACACHE_COEFFICIENTS = (
    2.50210439e02,
    -1.65061612e02,
    3.57804877e01,
    -7.81551492e-01,
    3.58559703e-02,
)
COGVIDEOX1_5_5B_I2V_TEACACHE_COEFFICIENTS = (
    1.22842302e02,
    -1.04088754e02,
    2.62981677e01,
    -3.06009921e-01,
    3.71213220e-02,
)


@dataclass
class TeaCacheConfig:
    """
    Configuration for TeaCache (https://huggingface.co/papers/2411.19108).

    The CogVideoX formulation compares the timestep embedding between consecutive denoising steps, maps the relative
    L1 change through model-specific calibration coefficients, and accumulates the predicted output change. While the
    accumulated change stays below threshold, the transformer block stack is replaced by the residual cached from the
    latest full execution.

    Args:
        coefficients: Polynomial coefficients ordered from highest to lowest degree and calibrated for the checkpoint.
        threshold: Accumulated predicted-change budget. Larger values cache more aggressively.
        num_inference_steps: Number of transformer calls in one denoising trajectory. First and last always run fully.
    """

    coefficients: Sequence[float]
    threshold: float = 0.2
    num_inference_steps: int = 50

    def __post_init__(self):
        self.coefficients = tuple(float(value) for value in self.coefficients)
        if not self.coefficients or not all(math.isfinite(value) for value in self.coefficients):
            raise ValueError("TeaCache coefficients must be a non-empty sequence of finite values.")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError(f"TeaCache threshold must be non-negative, got {self.threshold}.")
        if isinstance(self.num_inference_steps, bool) or not isinstance(self.num_inference_steps, int):
            raise TypeError("TeaCache num_inference_steps must be an integer.")
        if self.num_inference_steps < 2:
            raise ValueError("TeaCache num_inference_steps must be at least 2.")


class TeaCacheState(BaseState):
    def __init__(self):
        self.previous_indicator: torch.Tensor | None = None
        self.previous_hidden_residual: torch.Tensor | None = None
        self.previous_encoder_residual: torch.Tensor | None = None
        self.accumulated_distance = 0.0
        self.step_index = 0
        self.should_compute = True
        self.head_hidden_input: torch.Tensor | None = None
        self.head_encoder_input: torch.Tensor | None = None

    def reset(self):
        self.previous_indicator = None
        self.previous_hidden_residual = None
        self.previous_encoder_residual = None
        self.accumulated_distance = 0.0
        self.step_index = 0
        self.should_compute = True
        self.head_hidden_input = None
        self.head_encoder_input = None


def _get_inputs(
    metadata: TransformerBlockMetadata, args: tuple, kwargs: dict
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden_states = metadata._get_parameter_from_args_kwargs(metadata.hidden_states_argument_name, args, kwargs)
    encoder_hidden_states = metadata._get_parameter_from_args_kwargs(
        metadata.encoder_hidden_states_argument_name, args, kwargs
    )
    return hidden_states, encoder_hidden_states


def _get_outputs(
    metadata: TransformerBlockMetadata, output: torch.Tensor | tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        return (
            output[metadata.return_hidden_states_index],
            output[metadata.return_encoder_hidden_states_index]
            if metadata.return_encoder_hidden_states_index is not None
            else None,
        )
    return output, None


def _build_output(
    metadata: TransformerBlockMetadata,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if metadata.return_encoder_hidden_states_index is None:
        return hidden_states

    output = [None] * (max(metadata.return_hidden_states_index, metadata.return_encoder_hidden_states_index) + 1)
    output[metadata.return_hidden_states_index] = hidden_states
    output[metadata.return_encoder_hidden_states_index] = encoder_hidden_states
    return tuple(output)


def _polyval(coefficients: Sequence[float], value: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


class TeaCacheHeadHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager, config: TeaCacheConfig):
        self.state_manager = state_manager
        self.config = config
        self._metadata = None

    def initialize_hook(self, module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._context is None:
            self.state_manager.set_context(CacheContext(name="inference"))

        hidden_states, encoder_hidden_states = _get_inputs(self._metadata, args, kwargs)
        indicator = self._metadata._get_parameter_from_args_kwargs("temb", args, kwargs)
        state: TeaCacheState = self.state_manager.get_state()
        state.head_hidden_input = hidden_states
        state.head_encoder_input = encoder_hidden_states

        forced_compute = state.step_index == 0 or state.step_index == self.config.num_inference_steps - 1
        cache_ready = (
            state.previous_indicator is not None
            and state.previous_hidden_residual is not None
            and (encoder_hidden_states is None or state.previous_encoder_residual is not None)
        )

        should_compute = True
        if not forced_compute and cache_ready:
            compatible = (
                indicator.shape == state.previous_indicator.shape
                and indicator.device == state.previous_indicator.device
                and indicator.dtype == state.previous_indicator.dtype
                and hidden_states.shape == state.previous_hidden_residual.shape
                and (
                    encoder_hidden_states is None
                    or encoder_hidden_states.shape == state.previous_encoder_residual.shape
                )
            )
            if compatible:
                current = indicator.detach().float()
                previous = state.previous_indicator.float()
                relative_l1 = (current - previous).abs().mean() / (previous.abs().mean() + 1e-16)
                distance = _polyval(self.config.coefficients, float(relative_l1.cpu()))
                candidate = state.accumulated_distance + distance
                if math.isfinite(candidate) and candidate < self.config.threshold:
                    state.accumulated_distance = candidate
                    should_compute = False
                else:
                    state.accumulated_distance = 0.0
            else:
                state.accumulated_distance = 0.0

        if forced_compute:
            state.accumulated_distance = 0.0

        state.previous_indicator = indicator.detach()
        state.should_compute = should_compute

        if should_compute:
            return self.fn_ref.original_forward(*args, **kwargs)

        cached_hidden = hidden_states + state.previous_hidden_residual.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        cached_encoder = encoder_hidden_states
        if encoder_hidden_states is not None:
            cached_encoder = encoder_hidden_states + state.previous_encoder_residual.to(
                device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype
            )
        return _build_output(self._metadata, cached_hidden, cached_encoder)

    def reset_state(self, module):
        self.state_manager.reset()
        return module


class TeaCacheBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, config: TeaCacheConfig, is_tail: bool = False):
        self.state_manager = state_manager
        self.config = config
        self.is_tail = is_tail
        self._metadata = None

    def initialize_hook(self, module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: TeaCacheState = self.state_manager.get_state()

        if not state.should_compute:
            hidden_states, encoder_hidden_states = _get_inputs(self._metadata, args, kwargs)
            if self.is_tail:
                self._advance_step(state)
            return _build_output(self._metadata, hidden_states, encoder_hidden_states)

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_tail:
            hidden_states, encoder_hidden_states = _get_outputs(self._metadata, output)
            if state.head_hidden_input is None:
                raise RuntimeError("TeaCache head input was not recorded before the tail block.")
            state.previous_hidden_residual = (hidden_states - state.head_hidden_input).detach()
            if encoder_hidden_states is not None:
                if state.head_encoder_input is None:
                    raise RuntimeError("TeaCache encoder input was not recorded before the tail block.")
                state.previous_encoder_residual = (encoder_hidden_states - state.head_encoder_input).detach()
            else:
                state.previous_encoder_residual = None
            self._advance_step(state)

        return output

    def _advance_step(self, state: TeaCacheState):
        state.step_index += 1
        state.head_hidden_input = None
        state.head_encoder_input = None
        if state.step_index >= self.config.num_inference_steps:
            state.reset()

    def reset_state(self, module):
        self.state_manager.reset()
        return module


def apply_tea_cache(module: torch.nn.Module, config: TeaCacheConfig) -> None:
    """Apply TeaCache to a transformer with one registered transformer-block stack."""

    HookRegistry.check_if_exists_or_initialize(module)
    stacks = []
    for name, submodule in module.named_children():
        if name in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS and isinstance(submodule, torch.nn.ModuleList):
            stacks.append((name, submodule))

    if len(stacks) != 1:
        raise ValueError(
            "TeaCache currently supports models with exactly one transformer block stack; "
            f"found {len(stacks)} compatible stacks."
        )

    stack_name, blocks = stacks[0]
    if len(blocks) < 2:
        raise ValueError("TeaCache currently requires at least two transformer blocks.")

    state_manager = StateManager(TeaCacheState, (), {})
    head = blocks[0]
    tail = blocks[-1]

    for block in blocks:
        metadata = TransformerBlockRegistry.get(unwrap_module(block).__class__)
        if metadata.return_encoder_hidden_states_index is None:
            raise ValueError(
                "TeaCache currently requires a dual-stream transformer block that returns both hidden and encoder states."
            )
        if "temb" not in inspect.signature(metadata._cls.forward).parameters:
            raise ValueError("TeaCache requires transformer blocks with a timestep embedding argument named 'temb'.")

    head_registry = HookRegistry.check_if_exists_or_initialize(head)
    head_registry.register_hook(TeaCacheHeadHook(state_manager, config), _TEA_CACHE_LEADER_BLOCK_HOOK)

    for index, block in enumerate(blocks[1:], start=1):
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(
            TeaCacheBlockHook(state_manager, config, is_tail=block is tail),
            _TEA_CACHE_BLOCK_HOOK,
        )
        logger.debug(f"TeaCache attached to {stack_name}.{index}.")
