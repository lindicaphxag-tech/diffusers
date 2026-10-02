import gc

import pytest
import torch
import torch.nn as nn

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    is_gguf,
    is_gguf_available,
    is_quantization,
    nightly,
    require_accelerate,
    require_accelerator,
    require_gguf_version_greater_or_equal,
    require_kernels_version_greater_or_equal,
    torch_device,
)


if is_gguf_available():
    import gguf

    from diffusers.quantizers.gguf.utils import GGUFParameter

enable_full_determinism()


@is_quantization
@is_gguf
class TestGGMLQuantizationKernelDispatch:
    class _FakeKernel:
        MAX_GEMV_ROWS = 8

        def __init__(self, quant_type):
            self.GEMV_TYPES = {int(quant_type)}

        @staticmethod
        def mul_mat_vec(weight, inputs, quant_type, out_features):
            from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

            dense_weight = dequantize_gguf_tensor(weight).to(inputs.dtype)
            return (inputs @ dense_weight.T).float()

    def test_shape_aware_gemv_dispatch(self, monkeypatch):
        import diffusers.quantizers.gguf.utils as gguf_utils
        from diffusers.quantizers.gguf.utils import GGUFLinear

        quant_type = gguf.GGMLQuantizationType.Q4_0
        monkeypatch.setattr(gguf_utils, "ggml_quantization_ops", self._FakeKernel(quant_type))

        in_features, out_features = 32, 16
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        packed_cols = in_features // block_size * type_size

        linear = GGUFLinear(in_features, out_features, compute_dtype=torch.float32, device="meta")
        linear.weight = GGUFParameter(
            torch.empty(out_features, packed_cols, dtype=torch.uint8, device="meta"),
            quant_type=quant_type,
        )

        assert linear._can_use_ggml_quantization_gemv(torch.empty(2, 4, in_features, device="meta"))
        assert not linear._can_use_ggml_quantization_gemv(torch.empty(1, 9, in_features, device="meta"))

    def test_packed_gemv_matches_native_shape_bias_and_dtype(self, monkeypatch):
        import diffusers.quantizers.gguf.utils as gguf_utils
        from diffusers.quantizers.gguf.utils import GGUFLinear

        quant_type = gguf.GGMLQuantizationType.Q4_0
        monkeypatch.setattr(gguf_utils, "ggml_quantization_ops", self._FakeKernel(quant_type))

        in_features, out_features = 32, 16
        torch.manual_seed(0)
        float_weight = torch.randn(out_features, in_features, dtype=torch.float32)
        packed = torch.from_numpy(gguf.quants.quantize(float_weight.numpy(), quant_type))
        weight = GGUFParameter(packed, quant_type=quant_type)

        linear = GGUFLinear(in_features, out_features, bias=True, compute_dtype=torch.float32)
        linear.weight = weight
        linear.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        inputs = torch.randn(2, 4, in_features, dtype=torch.float32)

        expected = linear.forward_native(inputs)
        actual = linear.forward_ggml_quantization(inputs)

        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected)


@is_quantization
@is_gguf
@nightly
@require_accelerator
@require_gguf_version_greater_or_equal("0.10.0")
@require_kernels_version_greater_or_equal("0.17.0")
class TestGGMLQuantizationKernels:
    @pytest.mark.parametrize("quant_name", ["Q4_0", "Q4_K"])
    @pytest.mark.parametrize("rows", [1, 8])
    def test_packed_gemv_vs_native(self, monkeypatch, quant_name, rows):
        from kernels import get_kernel

        import diffusers.quantizers.gguf.utils as gguf_utils
        from diffusers.quantizers.gguf.utils import GGUFLinear

        kernel = get_kernel("ggml-org/ggml-quantization", version=1)
        quant_type = getattr(gguf.GGMLQuantizationType, quant_name)
        if int(quant_type) not in kernel.GEMV_TYPES:
            pytest.skip(f"{quant_name} has no GEMV kernel on {torch_device}")
        monkeypatch.setattr(gguf_utils, "ggml_quantization_ops", kernel)

        in_features, out_features = 512, 256
        torch.manual_seed(0)
        float_weight = torch.randn(out_features, in_features, dtype=torch.float32)
        packed = torch.from_numpy(gguf.quants.quantize(float_weight.numpy(), quant_type)).to(torch_device)
        weight = GGUFParameter(packed, quant_type=quant_type)
        inputs = torch.randn(rows, in_features, dtype=torch.bfloat16, device=torch_device)

        linear = GGUFLinear(in_features, out_features, bias=True, compute_dtype=torch.bfloat16)
        linear.weight = weight
        linear.bias = nn.Parameter(torch.randn(out_features, dtype=torch.bfloat16, device=torch_device))
        linear = linear.to(torch_device)

        expected = linear.forward_native(inputs).float()
        actual = linear(inputs).float()

        assert linear._can_use_ggml_quantization_gemv(inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2 * expected.abs().max())


# Model-level GGUF tests live in `tests/models/testing_utils/quantization.py` and pipeline-level
# ones in `tests/pipelines/testing_utils/quantization.py`. This module covers backend behavior that
# fits neither: CUDA kernel correctness.
@is_quantization
@is_gguf
@nightly
@require_accelerate
@require_accelerator
@require_gguf_version_greater_or_equal("0.10.0")
@require_kernels_version_greater_or_equal("0.9.0")
class TestGGUFCudaKernels:
    @pytest.fixture(autouse=True)
    def _setup_cuda_kernels(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_cuda_kernels_vs_native(self):
        if torch_device != "cuda":
            pytest.skip("CUDA kernels test requires CUDA device")

        from diffusers.quantizers.gguf.utils import GGUFLinear, can_use_cuda_kernels

        if not can_use_cuda_kernels:
            pytest.skip("CUDA kernels not available (compute capability < 7 or kernels not installed)")

        test_quant_types = ["Q4_0", "Q4_K"]
        test_shape = (1, 64, 512)  # batch, seq_len, hidden_dim
        compute_dtype = torch.bfloat16

        for quant_type in test_quant_types:
            qtype = getattr(gguf.GGMLQuantizationType, quant_type)
            in_features, out_features = 512, 512

            torch.manual_seed(42)
            float_weight = torch.randn(out_features, in_features, dtype=torch.float32)
            quantized_data = gguf.quants.quantize(float_weight.numpy(), qtype)
            weight_data = torch.from_numpy(quantized_data).to(device=torch_device)
            weight = GGUFParameter(weight_data, quant_type=qtype)

            x = torch.randn(test_shape, dtype=compute_dtype, device=torch_device)

            linear = GGUFLinear(in_features, out_features, bias=True, compute_dtype=compute_dtype)
            linear.weight = weight
            linear.bias = nn.Parameter(torch.randn(out_features, dtype=compute_dtype))
            linear = linear.to(torch_device)

            with torch.no_grad():
                output_native = linear.forward_native(x)
                output_cuda = linear.forward_cuda(x)

            assert torch.allclose(output_native, output_cuda, 1e-2), (
                f"GGUF CUDA Kernel Output is different from Native Output for {quant_type}"
            )
