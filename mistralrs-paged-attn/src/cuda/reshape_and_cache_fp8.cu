#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

namespace {

template <typename DType>
__global__ void reshape_and_cache_fp8_kernel(
    const DType* __restrict__ key, const DType* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
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
  const int64_t dst = row * head_size + blk * 16;

  {
    const DType* src = key + int64_t(blockIdx.x) * key_stride + head_idx * head_size + blk * 16;
    uint8_t out[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      out[i] = __nv_cvt_float_to_fp8(float(src[i]), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint4*>(key_cache + dst) = *reinterpret_cast<const uint4*>(out);
  }
  {
    const DType* src =
        value + int64_t(blockIdx.x) * value_stride + head_idx * head_size + blk * 16;
    uint8_t out[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      out[i] = __nv_cvt_float_to_fp8(float(src[i]), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint4*>(value_cache + dst) = *reinterpret_cast<const uint4*>(out);
  }
}

}  // namespace

extern "C" void reshape_and_cache_fp8(
    void* key, void* value, void* key_cache, void* value_cache, int64_t* slot_mapping,
    int32_t num_tokens, int32_t num_heads, int32_t head_size, int32_t block_size,
    int32_t key_stride, int32_t value_stride, uint32_t dtype, cudaStream_t stream) {
  const int32_t threads = num_heads * (head_size / 16);
  if (threads > 1024 || head_size % 16 != 0) {
    fprintf(stderr, "reshape_and_cache_fp8: unsupported heads %d head_size %d\n", num_heads,
            head_size);
    return;
  }
  dim3 grid(num_tokens);
  dim3 block(threads);
  if (dtype == 0) {
    reshape_and_cache_fp8_kernel<__half><<<grid, block, 0, stream>>>(
        static_cast<const __half*>(key), static_cast<const __half*>(value),
        static_cast<uint8_t*>(key_cache), static_cast<uint8_t*>(value_cache), slot_mapping,
        num_heads, head_size, block_size, key_stride, value_stride);
  } else if (dtype == 1) {
    reshape_and_cache_fp8_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key), static_cast<const __nv_bfloat16*>(value),
        static_cast<uint8_t*>(key_cache), static_cast<uint8_t*>(value_cache), slot_mapping,
        num_heads, head_size, block_size, key_stride, value_stride);
  } else {
    fprintf(stderr, "reshape_and_cache_fp8 received unsupported dtype %u\n", dtype);
  }
}
