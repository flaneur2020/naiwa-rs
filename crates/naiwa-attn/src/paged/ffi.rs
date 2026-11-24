//! FFI bindings for `naiwa-attn`'s PagedAttention CUDA kernels (`naiwa_`-prefixed).

#![allow(dead_code)]

use core::ffi::{c_int, c_long, c_void};

use candle_core::cuda::cudarc::driver::sys::CUstream;

extern "C" {
    #[link_name = "naiwa_reshape_and_cache"]
    pub fn reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const c_long,
        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        stream: CUstream,
        dtype: u32,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
    );

    #[link_name = "naiwa_gather_kv_cache"]
    pub fn gather_kv_cache(
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_out: *const c_void,
        v_out: *const c_void,
        k_scale: *const f32,
        v_scale: *const f32,
        block_table: *const c_int,
        cu_seq_lens: *const c_int,
        num_tokens: c_int,
        num_seqs: c_int,
        block_size: c_int,
        block_table_stride: c_int,
        num_kv_heads: c_int,
        head_size: c_int,
        x: c_int,
        stream: CUstream,
        out_dtype: u32,
        cache_dtype: u32,
    );

    #[link_name = "naiwa_paged_attention_v1_f16"]
    pub fn paged_attention_v1_f16(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: CUstream,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    );

    #[link_name = "naiwa_paged_attention_v1_bf16"]
    pub fn paged_attention_v1_bf16(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: CUstream,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    );

    #[link_name = "naiwa_paged_attention_v1_f32"]
    pub fn paged_attention_v1_f32(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: CUstream,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    );

    #[link_name = "naiwa_paged_attention_v2_f16"]
    pub fn paged_attention_v2_f16(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: CUstream,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    );

    #[link_name = "naiwa_paged_attention_v2_bf16"]
    pub fn paged_attention_v2_bf16(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: CUstream,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    );

    #[link_name = "naiwa_paged_attention_v2_f32"]
    pub fn paged_attention_v2_f32(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: CUstream,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    );

    #[link_name = "naiwa_copy_blocks_bf16"]
    pub fn copy_blocks_bf16(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const i64,
        num_layers: c_int,
        num_pairs: c_int,
        numel_per_block_key: c_int,
        numel_per_block_value: c_int,
        stream: CUstream,
    );

    #[link_name = "naiwa_copy_blocks_f16"]
    pub fn copy_blocks_f16(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const i64,
        num_layers: c_int,
        num_pairs: c_int,
        numel_per_block_key: c_int,
        numel_per_block_value: c_int,
        stream: CUstream,
    );

    #[link_name = "naiwa_copy_blocks_f32"]
    pub fn copy_blocks_f32(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const i64,
        num_layers: c_int,
        num_pairs: c_int,
        numel_per_block_key: c_int,
        numel_per_block_value: c_int,
        stream: CUstream,
    );
}
