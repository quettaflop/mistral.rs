/*
 * Batch decode attention for quantized paged KV cache:
 *   K: e4m3 (1B/elem), V: packed e2m1 (0.5B/elem), per-16-dim e4m3 block scales.
 *
 * Layouts (HND):
 *   k_data:   [num_pages, num_kv_heads, page_size, head_dim]      u8
 *   v_data:   [num_pages, num_kv_heads, page_size, head_dim / 2]  u8
 *   k_scales: [num_pages, num_kv_heads, page_size, head_dim / 16] u8 (e4m3)
 *   v_scales: [num_pages, num_kv_heads, page_size, head_dim / 16] u8 (e4m3)
 *
 * Thread geometry mirrors the f16 path (vec_size=8 lanes of f32 math per
 * thread): each thread's 8 dims sit inside one 16-dim scale block, so one
 * scale per (token, thread) suffices for both the qk dot and the v mac.
 */
#ifndef FLASHINFER_DECODE_QUANT_CUH_
#define FLASHINFER_DECODE_QUANT_CUH_

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../math.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "state.cuh"
#include "variants.cuh"

namespace flashinfer {
namespace quant {

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

__device__ __forceinline__ float e2m1_to_float(uint32_t n) {
  const uint32_t e = (n >> 1) & 3, m = n & 1;
  const float mag = e ? (1.0f + 0.5f * float(m)) * float(1 << (e - 1)) : 0.5f * float(m);
  return (n & 8) ? -mag : mag;
}

template <typename DTypeQ_, typename DTypeO_, typename IdType_>
struct BatchDecodeQuantParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = uint8_t;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  uint8_t* k_data;
  uint8_t* v_data;
  uint8_t* k_scales;
  uint8_t* v_scales;
  IdType* indices;
  IdType* indptr;
  IdType* last_page_len;
  DTypeO* o;
  float* lse;
  uint32_t batch_size;
  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint_fastdiv page_size;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;

  IdType* request_indices;
  IdType* kv_tile_indices;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;

  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t) const { return 1; }
  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * uint32_t(page_size) +
           last_page_len[batch_idx];
  }
};

template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename AttentionVariant,
          typename Params>
__device__ __forceinline__ void compute_qk_quant(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx,
    const uint8_t* k_smem, const uint8_t* ks_smem, const vec_t<float, vec_size>& q_vec,
    uint32_t kv_idx_base, uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx,
    uint32_t kv_head_idx, float* s, state_t<vec_size>& st, const uint32_t tx, const uint32_t tz) {
  constexpr uint32_t head_dim = bdx * vec_size;
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<half, vec_size> k_h;
    k_h.cast_load(reinterpret_cast<const __nv_fp8_e4m3*>(k_smem + j * head_dim + tx * vec_size));
    vec_t<float, vec_size> k_vec;
    k_vec.cast_from(k_h);
    const float ks = e4m3_to_float(ks_smem[j * (head_dim / 16) + (tx * vec_size) / 16]);
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
    s[j] *= ks;
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }
    bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
                                   kv_head_idx);
    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  if constexpr (variant.use_softmax) {
    const float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}

template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size>
__device__ __forceinline__ void update_local_state_quant(const uint8_t* v_smem,
                                                         const uint8_t* vs_smem,
                                                         const float2* e2m1_lut, const float* s,
                                                         state_t<vec_size>& st,
                                                         const uint32_t tx) {
  constexpr uint32_t head_dim = bdx * vec_size;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    const uint32_t packed =
        *reinterpret_cast<const uint32_t*>(v_smem + j * (head_dim / 2) + tx * (vec_size / 2));
    const float w = s[j] * e4m3_to_float(vs_smem[j * (head_dim / 16) + (tx * vec_size) / 16]);
#pragma unroll
    for (uint32_t i = 0; i < vec_size / 2; ++i) {
      const float2 pair = e2m1_lut[(packed >> (8 * i)) & 0xFF];
      st.o[2 * i] += w * pair.x;
      st.o[2 * i + 1] += w * pair.y;
    }
  }
}

// Copy of sync_state from decode.cuh (anonymous namespace there).
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant>
__device__ __forceinline__ void sync_state_quant(AttentionVariant variant, state_t<vec_size>& st,
                                                 float* smem, float* smem_md, const uint32_t tx,
                                                 const uint32_t ty, const uint32_t tz) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    if constexpr (variant.use_softmax) {
      smem_md[(tz * bdy + ty) * 2] = st.m;
      smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
        st.merge(oz, mz, dz);
      }
    } else {
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          st.o[i] += oz[i];
        }
      }
    }
  }
}

template <uint32_t num_stages_smem, uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename AttentionVariant, typename Params>
__device__ __forceinline__ void BatchDecodeQuantDevice(
    const Params& params, uint8_t smem[], const uint32_t bx = blockIdx.x,
    const uint32_t by = blockIdx.y, const uint32_t tx = threadIdx.x,
    const uint32_t ty = threadIdx.y, const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  constexpr uint32_t head_dim = bdx * vec_size;
  constexpr uint32_t k_row = head_dim;       // e4m3 bytes
  constexpr uint32_t v_row = head_dim / 2;   // packed e2m1 bytes
  constexpr uint32_t s_row = head_dim / 16;  // e4m3 scale bytes
  constexpr uint32_t rows_per_stage = tile_size_per_bdx * bdy * bdz;
  constexpr uint32_t stage_k = rows_per_stage * k_row;
  constexpr uint32_t stage_v = rows_per_stage * v_row;
  constexpr uint32_t stage_s = rows_per_stage * s_row;

  const uint32_t batch_idx = params.request_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;
  // With CUDA graphs more blocks than the actual batch may be launched.
  if (params.block_valid_mask && !params.block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *params.kv_chunk_size_ptr;
  const uint32_t kv_len = params.get_kv_len(batch_idx);
  const uint32_t max_chunk_size = params.partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = params.partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      params.partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  uint8_t* k_smem = smem;
  uint8_t* v_smem = k_smem + num_stages_smem * stage_k;
  uint8_t* ks_smem = v_smem + num_stages_smem * stage_v;
  uint8_t* vs_smem = ks_smem + num_stages_smem * stage_s;
  float2* e2m1_lut = reinterpret_cast<float2*>(vs_smem + num_stages_smem * stage_s);
  uint8_t* quant_end = reinterpret_cast<uint8_t*>(e2m1_lut + 256);
  // Aliased: page offsets during the pipeline, o/md floats during the final sync.
  size_t* kv_offset_smem = reinterpret_cast<size_t*>(quant_end);
  float* smem_o = reinterpret_cast<float*>(quant_end);
  float* smem_md = smem_o + bdy * bdz * head_dim;

  // byte -> (lo nibble, hi nibble) e2m1 values
  const uint32_t tid = (tz * bdy + ty) * bdx + tx;
#pragma unroll
  for (uint32_t b = tid; b < 256; b += bdx * bdy * bdz) {
    e2m1_lut[b] = make_float2(e2m1_to_float(b & 15), e2m1_to_float(b >> 4));
  }

  AttentionVariant variant(params, batch_idx, smem);

  vec_t<float, vec_size> q_vec;
  q_vec.cast_load(params.q + batch_idx * params.q_stride_n + qo_head_idx * params.q_stride_h +
                  tx * vec_size);
  block.sync();

  const IdType last_indptr = params.indptr[params.batch_size];
  const uint32_t packed_page_iter_base =
      params.indptr[batch_idx] * uint32_t(params.page_size) + chunk_start;

  // kv_offset_smem[i] = K-element offset (== byte offset) of token i of the
  // current bdx-iteration group, at dim 0. V byte offset = off/2, scale byte
  // offset = off/16 (head_dim is a multiple of 16).
  static_assert(num_stages_smem <= bdx);
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    params.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        (q < last_indptr)
            ? (size_t(params.indices[q]) * params.num_kv_heads + kv_head_idx) *
                      (uint32_t(params.page_size) * head_dim) +
                  r * head_dim
            : 0;
  }
  block.sync();

  auto produce_k = [&](uint32_t stage_idx, uint32_t iter_p) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      const size_t off =
          kv_offset_smem[(((iter_p % bdx) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j];
      const bool pred =
          ((iter_p * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size;
      const uint32_t row = ((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j;
      cp_async::pred_load_128b_from_64b<PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          reinterpret_cast<uint64_t*>(k_smem + row * k_row + tx * vec_size),
          reinterpret_cast<const uint64_t*>(params.k_data + off + tx * vec_size), pred);
      if (tx == 0) {
        cp_async::pred_load_128b_from_64b<PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
            reinterpret_cast<uint64_t*>(ks_smem + row * s_row),
            reinterpret_cast<const uint64_t*>(params.k_scales + off / 16), pred);
      }
    }
    cp_async::commit_group();
  };

  auto produce_v = [&](uint32_t stage_idx, uint32_t iter_p) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      const size_t off =
          kv_offset_smem[(((iter_p % bdx) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j];
      const bool pred =
          ((iter_p * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size;
      const uint32_t row = ((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j;
      cp_async::pred_load_32b<SharedMemFillMode::kFillZero>(
          reinterpret_cast<uint32_t*>(v_smem + row * v_row + tx * (vec_size / 2)),
          reinterpret_cast<const uint32_t*>(params.v_data + off / 2 + tx * (vec_size / 2)),
          pred);
      if (tx == 0) {
        cp_async::pred_load_128b_from_64b<PrefetchMode::kPrefetch,
                                          SharedMemFillMode::kFillZero>(
            reinterpret_cast<uint64_t*>(vs_smem + row * s_row),
            reinterpret_cast<const uint64_t*>(params.v_scales + off / 16), pred);
      }
    }
    cp_async::commit_group();
  };

  uint32_t stage_idx = 0;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    produce_k(iter, iter);
    produce_v(iter, iter);
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, rows_per_stage); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        params.page_size.divmod(
            packed_page_iter_base + (iter + num_stages_smem) * rows_per_stage +
                ((j * bdz + tz) * bdy + ty) * bdx + tx,
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            (q < last_indptr)
                ? (size_t(params.indices[q]) * params.num_kv_heads + kv_head_idx) *
                          (uint32_t(params.page_size) * head_dim) +
                      r * head_dim
                : 0;
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk_quant<vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * k_row,
        ks_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * s_row, q_vec,
        chunk_start + iter * rows_per_stage, iter * rows_per_stage, chunk_size, qo_head_idx,
        kv_head_idx, s, st, tx, tz);
    block.sync();

    // load next k tile into the just-consumed stage
    produce_k(stage_idx, iter + num_stages_smem);

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state_quant<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * v_row,
        vs_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * s_row, e2m1_lut, s, st, tx);
    block.sync();

    // load next v tile
    produce_v(stage_idx, iter + num_stages_smem);
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  sync_state_quant<vec_size, bdx, bdy, bdz>(variant, st, smem_o, smem_md, tx, ty, tz);
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    st.o[i] = variant.OutputTransform(params, st.o[i], bx, /*qo_idx=*/0, qo_head_idx, st.m, st.d,
                                      /*scale=*/1.0f);
  }

  if (tz == 0) {
    st.o.cast_store(params.o + (bx * params.num_qo_heads + qo_head_idx) * head_dim +
                    tx * vec_size);
    if (params.lse != nullptr) {
      params.lse[bx * params.num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
}

template <uint32_t num_stages_smem, uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename AttentionVariant, typename Params>
__global__ void BatchDecodeQuantKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchDecodeQuantDevice<num_stages_smem, tile_size_per_bdx, vec_size, bdx, bdy, bdz,
                         AttentionVariant>(params, smem);
}

template <uint32_t HEAD_DIM, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeQuantDispatched(Params params, typename Params::DTypeO* tmp_v, float* tmp_s,
                                       cudaStream_t stream) {
  using DTypeO = typename Params::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size = 8;
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  static_assert(HEAD_DIM % 16 == 0);
  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? 4U : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      constexpr uint32_t rows = tile_size_per_bdx * bdy * bdz;
      const uint32_t quant_bytes =
          NUM_STAGES_SMEM * rows * (HEAD_DIM + HEAD_DIM / 2 + 2 * (HEAD_DIM / 16)) +
          256 * sizeof(float2);
      const uint32_t smem_size =
          quant_bytes + std::max(tile_size_per_bdx * num_threads * sizeof(size_t),
                                 bdy * bdz * HEAD_DIM * sizeof(float) +
                                     2 * bdy * bdz * sizeof(float));
      auto kernel = BatchDecodeQuantKernel<NUM_STAGES_SMEM, tile_size_per_bdx, vec_size, bdx, bdy,
                                           bdz, AttentionVariant, Params>;
      FLASHINFER_SET_MAX_DYNAMIC_SMEM(kernel, smem_size, stream);
      dim3 nblks(padded_batch_size, num_kv_heads);
      dim3 nthrs(bdx, bdy, bdz);
      if (tmp_v == nullptr) {
        params.partition_kv = false;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        static_assert(AttentionVariant::use_softmax);
        FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
            tmp_v, tmp_s, params.o_indptr, o, lse, params.batch_size, nullptr, num_qo_heads,
            HEAD_DIM, /*enable_pdl=*/false, stream));
      }
    });
  });
  return cudaSuccess;
}

}  // namespace quant
}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_QUANT_CUH_
