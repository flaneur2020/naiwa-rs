//! Paged KV block allocator and `PagedAttnMeta` builder.

use candle_core::{DType, Device, Result, Tensor};
use naiwa_attn::PagedAttnMeta;
use naiwa_models::Qwen4ExpConfig;

/// Default paged KV block size on CUDA (divisible by QSA compress_ratio=4).
pub const DEFAULT_BLOCK_SIZE: usize = 32;

/// Physical block allocator for PagedAttention KV caches.
pub struct BlockManager {
    free_blocks: Vec<u32>,
    num_blocks: usize,
    block_size: usize,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let free_blocks = (0..num_blocks as u32).rev().collect();
        Self {
            free_blocks,
            num_blocks,
            block_size,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn free_count(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn used_count(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Ensure `block_table` holds enough physical blocks for `total_tokens`.
    /// Returns `false` if there are not enough free blocks (leaving `block_table` unchanged).
    pub fn ensure_capacity(&mut self, block_table: &mut Vec<u32>, total_tokens: usize) -> bool {
        let needed_blocks = total_tokens.div_ceil(self.block_size);
        if needed_blocks <= block_table.len() {
            return true;
        }
        let extra = needed_blocks - block_table.len();
        if extra > self.free_blocks.len() {
            return false;
        }
        for _ in 0..extra {
            block_table.push(self.free_blocks.pop().unwrap());
        }
        true
    }

    /// Shrink `block_table` to hold `keep_tokens`, returning any trailing blocks to the free pool.
    pub fn truncate(&mut self, block_table: &mut Vec<u32>, keep_tokens: usize) {
        let keep_blocks = keep_tokens.div_ceil(self.block_size);
        while block_table.len() > keep_blocks {
            if let Some(blk) = block_table.pop() {
                self.free_blocks.push(blk);
            }
        }
    }

    /// Free all blocks belonging to a finished sequence.
    pub fn free_all(&mut self, block_table: &mut Vec<u32>) {
        self.free_blocks.extend(block_table.drain(..));
    }

    /// Physical slot index for logical token position `pos` in `block_table`.
    #[inline]
    pub fn slot_for(&self, block_table: &[u32], pos: usize) -> i64 {
        let blk = block_table[pos / self.block_size] as i64;
        blk * self.block_size as i64 + (pos % self.block_size) as i64
    }

    /// Build a [`PagedAttnMeta`] tensor bundle for a batch of sequences.
    ///
    /// Each sequence `i` appends `query_lens[i]` new tokens starting at logical position
    /// `context_lens[i] - query_lens[i]`.
    pub fn build_meta(
        &self,
        block_tables_host: &[&[u32]],
        context_lens_host: &[usize],
        query_lens_host: &[usize],
        is_decode: bool,
        device: &Device,
    ) -> Result<PagedAttnMeta> {
        let n_seqs = block_tables_host.len();
        let max_blocks = block_tables_host
            .iter()
            .map(|t| t.len())
            .max()
            .unwrap_or(1)
            .max(1);

        let mut bt_flat = vec![0u32; n_seqs * max_blocks];
        let mut slots = Vec::new();
        let mut cu_ctx = Vec::with_capacity(n_seqs + 1);
        cu_ctx.push(0u32);
        let mut running_ctx = 0u32;

        for seq_idx in 0..n_seqs {
            let table = block_tables_host[seq_idx];
            bt_flat[seq_idx * max_blocks..seq_idx * max_blocks + table.len()]
                .copy_from_slice(table);
            let kv_len = context_lens_host[seq_idx];
            let q_len = query_lens_host[seq_idx];
            let start_pos = kv_len - q_len;
            for pos in start_pos..kv_len {
                slots.push(self.slot_for(table, pos));
            }
            running_ctx += kv_len as u32;
            cu_ctx.push(running_ctx);
        }

        let max_context_len = context_lens_host.iter().copied().max().unwrap_or(0);
        let ctx_u32: Vec<u32> = context_lens_host.iter().map(|&l| l as u32).collect();
        let total_q: usize = query_lens_host.iter().sum();

        // For small multi-token batches (MTP verify/refresh, total_q <= 8), prebuild per-row
        // virtual block tables & causal context lengths so PagedAttention::forward runs fused
        // paged_attention_decode directly without gather_kv_cache!
        let (row_block_tables, row_context_lens) = if !is_decode && total_q > 0 && total_q <= 8 && device.is_cuda() {
            let mut rbt = vec![0u32; total_q * max_blocks];
            let mut rcl = Vec::with_capacity(total_q);
            let mut row = 0usize;
            for seq_idx in 0..n_seqs {
                let table = block_tables_host[seq_idx];
                let kv_len = context_lens_host[seq_idx];
                let q_len = query_lens_host[seq_idx];
                let start_pos = kv_len - q_len;
                for q in 0..q_len {
                    rbt[row * max_blocks..row * max_blocks + table.len()].copy_from_slice(table);
                    rcl.push((start_pos + q + 1) as u32);
                    row += 1;
                }
            }
            (
                Some(Tensor::from_vec(rbt, (total_q, max_blocks), device)?),
                Some(Tensor::from_vec(rcl, (total_q,), device)?),
            )
        } else {
            (None, None)
        };

        Ok(PagedAttnMeta {
            slot_mapping: Tensor::from_vec(slots, (total_q,), device)?,
            block_tables: Tensor::from_vec(bt_flat, (n_seqs, max_blocks), device)?,
            context_lens: Tensor::from_vec(ctx_u32, (n_seqs,), device)?,
            cu_context_lens: Tensor::from_vec(cu_ctx, (n_seqs + 1,), device)?,
            host_context_lens: context_lens_host.to_vec(),
            host_query_lens: query_lens_host.to_vec(),
            max_context_len,
            is_decode,
            row_block_tables,
            row_context_lens,
        })
    }
}

/// Compute the number of physical KV blocks to allocate from CLI flags and model geometry.
pub fn compute_num_blocks(
    cfg: &Qwen4ExpConfig,
    has_mtp: bool,
    context_len: Option<usize>,
    memory_mb: Option<usize>,
    memory_fraction: Option<f32>,
    block_size: usize,
    dtype: DType,
    free_vram_bytes: Option<usize>,
) -> usize {
    if let Some(ctx) = context_len {
        return ctx.div_ceil(block_size).max(16);
    }
    let kv_layers = cfg.num_full_attention_layers() + usize::from(has_mtp);
    // K + V caches + QSA aux cache per token
    let elem_bytes = dtype.size_in_bytes();
    let kv_bytes_per_token = kv_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * elem_bytes;
    let aux_bytes_per_token =
        kv_layers * cfg.qsa.aux_cache_elements_per_token(cfg.rotary_dim) * elem_bytes;
    let bytes_per_block = (kv_bytes_per_token + aux_bytes_per_token) * block_size;

    let budget_bytes = if let Some(mb) = memory_mb {
        mb * 1024 * 1024
    } else if let Some(frac) = memory_fraction {
        let free = free_vram_bytes.unwrap_or(2 * 1024 * 1024 * 1024);
        ((free as f64) * (frac as f64).clamp(0.05, 0.95)) as usize
    } else {
        // Default: use 40% of remaining free VRAM (capped to 4096 tokens on 12 GB cards so
        // expert cache + prefill buffers keep plenty of headroom).
        let free = free_vram_bytes.unwrap_or(1024 * 1024 * 1024);
        let default_ctx_bytes = 4096 * (kv_bytes_per_token + aux_bytes_per_token);
        ((free as f64) * 0.40).min(default_ctx_bytes as f64) as usize
    };
    (budget_bytes / bytes_per_block.max(1)).max(16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_manager_alloc_truncate_free() {
        let mut bm = BlockManager::new(8, 16);
        let mut bt = Vec::new();
        assert!(bm.ensure_capacity(&mut bt, 33)); // 3 blocks
        assert_eq!(bt.len(), 3);
        assert_eq!(bm.free_count(), 5);
        bm.truncate(&mut bt, 16); // 1 block
        assert_eq!(bt.len(), 1);
        assert_eq!(bm.free_count(), 7);
        bm.free_all(&mut bt);
        assert_eq!(bm.free_count(), 8);
    }
}
