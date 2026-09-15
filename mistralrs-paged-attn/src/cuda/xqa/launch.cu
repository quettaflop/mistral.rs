#include "mha.h"

#include <cuda_runtime.h>

extern "C" int xqa_fp8_decode(
    int num_kv_heads,
    float q_scale,
    void* output,
    void const* q,
    void* k_cache,
    void* v_cache,
    int const* page_table,
    int max_seq_len,
    unsigned int const* seq_lens,
    int batch_size,
    float kv_scale,
    unsigned int* semaphores,
    void* scratch,
    unsigned long long kv_stride_page,
    unsigned long long kv_stride_token,
    unsigned long long kv_stride_head,
    int enable_pdl,
    void* stream)
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return 1;
    }
    int sm_count = 0;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
        return 1;
    }
    try {
        launchMHAFlashInfer(
            static_cast<uint32_t>(sm_count),
            static_cast<uint32_t>(num_kv_heads),
            0,
            q_scale,
            nullptr,
            reinterpret_cast<OutputHead*>(output),
            reinterpret_cast<InputHead const*>(q),
            nullptr,
            reinterpret_cast<GMemCacheHead*>(k_cache),
            reinterpret_cast<GMemCacheHead*>(v_cache),
            reinterpret_cast<KVCachePageIndex const*>(page_table),
            static_cast<uint32_t>(max_seq_len),
            seq_lens,
            static_cast<uint32_t>(batch_size),
            kv_scale,
            nullptr,
            semaphores,
            scratch,
            enable_pdl != 0,
            kv_stride_page,
            kv_stride_token,
            kv_stride_head,
            static_cast<cudaStream_t>(stream));
        return 0;
    } catch (...) {
        return 1;
    }
}
