#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "flashinfer/attention/decode_quant.cuh"

using namespace flashinfer;
using flashinfer::quant::BatchDecodeQuantParams;
using flashinfer::quant::BatchDecodeQuantDispatched;

namespace {

template <typename DType>
int32_t run_quant(
    void* q, void* key_cache, void* value_cache, void* key_scales, void* value_scales,
    const int32_t* kv_indptr, const int32_t* kv_indices, const int32_t* kv_last_page_len,
    const int32_t* request_indices, const int32_t* kv_tile_indices, const int32_t* o_indptr,
    const int32_t* kv_chunk_size_ptr, const bool* block_valid_mask, void* o, void* tmp_v,
    void* tmp_s, int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, cudaStream_t stream) {
  if (head_size != 128) {
    fprintf(stderr, "flashinfer_decode_quant: unsupported head_size %d (want 128)\n", head_size);
    return cudaErrorInvalidValue;
  }
  using Params = BatchDecodeQuantParams<DType, DType, int32_t>;
  using AttentionVariant = DefaultAttention<false, false, false, false>;

  Params params{};
  params.q = static_cast<DType*>(q);
  params.k_data = static_cast<uint8_t*>(key_cache);
  params.v_data = static_cast<uint8_t*>(value_cache);
  params.k_scales = static_cast<uint8_t*>(key_scales);
  params.v_scales = static_cast<uint8_t*>(value_scales);
  params.indices = const_cast<int32_t*>(kv_indices);
  params.indptr = const_cast<int32_t*>(kv_indptr);
  params.last_page_len = const_cast<int32_t*>(kv_last_page_len);
  params.o = static_cast<DType*>(o);
  params.lse = nullptr;
  params.batch_size = batch_size;
  params.padded_batch_size = padded_batch_size;
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.page_size = uint_fastdiv(page_size);
  params.q_stride_n = q_stride_n;
  params.q_stride_h = q_stride_h;
  params.window_left = -1;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = sm_scale;
  params.request_indices = const_cast<int32_t*>(request_indices);
  params.kv_tile_indices = const_cast<int32_t*>(kv_tile_indices);
  params.o_indptr = const_cast<int32_t*>(o_indptr);
  params.kv_chunk_size_ptr = const_cast<int32_t*>(kv_chunk_size_ptr);
  params.block_valid_mask = const_cast<bool*>(block_valid_mask);
  params.partition_kv = tmp_v != nullptr;

  return BatchDecodeQuantDispatched<128, AttentionVariant, Params>(
      params, static_cast<DType*>(tmp_v), static_cast<float*>(tmp_s), stream);
}

}  // namespace

extern "C" int32_t flashinfer_decode_quant(
    void* q, void* key_cache, void* value_cache, void* key_scales, void* value_scales,
    const int32_t* kv_indptr, const int32_t* kv_indices, const int32_t* kv_last_page_len,
    const int32_t* request_indices, const int32_t* kv_tile_indices, const int32_t* o_indptr,
    const int32_t* kv_chunk_size_ptr, const bool* block_valid_mask, void* o, void* tmp_v,
    void* tmp_s, int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, uint32_t dtype, cudaStream_t stream) {
  if (dtype == 0) {
    return run_quant<__half>(q, key_cache, value_cache, key_scales, value_scales, kv_indptr,
                             kv_indices, kv_last_page_len, request_indices, kv_tile_indices,
                             o_indptr, kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s,
                             batch_size, padded_batch_size, num_qo_heads, num_kv_heads, head_size,
                             page_size, q_stride_n, q_stride_h, sm_scale, stream);
  } else if (dtype == 1) {
    return run_quant<__nv_bfloat16>(q, key_cache, value_cache, key_scales, value_scales, kv_indptr,
                                    kv_indices, kv_last_page_len, request_indices, kv_tile_indices,
                                    o_indptr, kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s,
                                    batch_size, padded_batch_size, num_qo_heads, num_kv_heads,
                                    head_size, page_size, q_stride_n, q_stride_h, sm_scale,
                                    stream);
  }
  fprintf(stderr, "flashinfer_decode_quant received unsupported dtype %u\n", dtype);
  return cudaErrorInvalidValue;
}
