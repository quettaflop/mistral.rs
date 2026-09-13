// Quantizing reshape-and-cache: K -> e4m3, V -> packed e2m1, both with
// per-16-dim e4m3 block scales. Mirrors candle-ext's nvfp4/fp8 roundtrip sims.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

namespace {

__device__ __forceinline__ float e4m3_to_float_q(uint8_t b) {
  return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// e2m1 encode matching the sim's threshold rounding (round half up at
// 0.25/0.75/1.25/1.75/2.5/3.5/5.0); code index c is (e<<1)|m.
__device__ __forceinline__ uint8_t e2m1_encode(float q) {
  const float a = fabsf(q);
  const uint32_t c = uint32_t(a > 0.25f) + uint32_t(a > 0.75f) + uint32_t(a > 1.25f) +
                     uint32_t(a > 1.75f) + uint32_t(a > 2.5f) + uint32_t(a > 3.5f) +
                     uint32_t(a > 5.0f);
  return uint8_t((q < 0.f ? 8u : 0u) | c);
}

// One thread per 16-dim block; blockIdx.x = token.
template <typename DType>
__global__ void reshape_and_cache_quant_kernel(
    const DType* __restrict__ key, const DType* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    uint8_t* __restrict__ key_scales, uint8_t* __restrict__ value_scales,
    const int64_t* __restrict__ slot_mapping, int32_t num_heads, int32_t head_size,
    int32_t block_size, int32_t key_stride, int32_t value_stride) {
  const int64_t slot = slot_mapping[blockIdx.x];
  if (slot < 0) {
    return;
  }
  const int64_t page_idx = slot / block_size;
  const int64_t page_off = slot % block_size;
  const int32_t bph = head_size / 16;
  const int32_t head_idx = threadIdx.x / bph;
  if (head_idx >= num_heads) {
    return;
  }
  const int32_t blk = threadIdx.x % bph;
  const int64_t row = (page_idx * num_heads + head_idx) * block_size + page_off;

  {
    const DType* src = key + int64_t(blockIdx.x) * key_stride + head_idx * head_size + blk * 16;
    float x[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      x[i] = float(src[i]);
      amax = fmaxf(amax, fabsf(x[i]));
    }
    const float s = fminf(fmaxf(amax * (1.0f / 448.0f), 0x1p-9f), 448.0f);
    const uint8_t sb = __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
    const float sq = e4m3_to_float_q(sb);
    uint8_t out[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      out[i] = __nv_cvt_float_to_fp8(x[i] / sq, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint4*>(key_cache + row * head_size + blk * 16) =
        *reinterpret_cast<const uint4*>(out);
    key_scales[row * bph + blk] = sb;
  }
  {
    const DType* src =
        value + int64_t(blockIdx.x) * value_stride + head_idx * head_size + blk * 16;
    float x[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      x[i] = float(src[i]);
      amax = fmaxf(amax, fabsf(x[i]));
    }
    const float s = fminf(fmaxf(amax * (1.0f / 6.0f), 0x1p-9f), 448.0f);
    const uint8_t sb = __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
    const float sq = e4m3_to_float_q(sb);
    uint8_t packed[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      packed[i] = e2m1_encode(x[2 * i] / sq) | uint8_t(e2m1_encode(x[2 * i + 1] / sq) << 4);
    }
    *reinterpret_cast<uint2*>(value_cache + row * (head_size / 2) + blk * 8) =
        *reinterpret_cast<const uint2*>(packed);
    value_scales[row * bph + blk] = sb;
  }
}

}  // namespace

extern "C" void reshape_and_cache_quant(
    void* key, void* value, void* key_cache, void* value_cache, void* key_scales,
    void* value_scales, int64_t* slot_mapping, int32_t num_tokens, int32_t num_heads,
    int32_t head_size, int32_t block_size, int32_t key_stride, int32_t value_stride,
    uint32_t dtype, cudaStream_t stream) {
  const int32_t threads = num_heads * (head_size / 16);
  if (threads > 1024 || head_size % 16 != 0) {
    fprintf(stderr, "reshape_and_cache_quant: unsupported heads %d head_size %d\n", num_heads,
            head_size);
    return;
  }
  dim3 grid(num_tokens);
  dim3 block(threads);
  if (dtype == 0) {
    reshape_and_cache_quant_kernel<__half><<<grid, block, 0, stream>>>(
        static_cast<const __half*>(key), static_cast<const __half*>(value),
        static_cast<uint8_t*>(key_cache), static_cast<uint8_t*>(value_cache),
        static_cast<uint8_t*>(key_scales), static_cast<uint8_t*>(value_scales), slot_mapping,
        num_heads, head_size, block_size, key_stride, value_stride);
  } else if (dtype == 1) {
    reshape_and_cache_quant_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key), static_cast<const __nv_bfloat16*>(value),
        static_cast<uint8_t*>(key_cache), static_cast<uint8_t*>(value_cache),
        static_cast<uint8_t*>(key_scales), static_cast<uint8_t*>(value_scales), slot_mapping,
        num_heads, head_size, block_size, key_stride, value_stride);
  } else {
    fprintf(stderr, "reshape_and_cache_quant received unsupported dtype %u\n", dtype);
  }
}
