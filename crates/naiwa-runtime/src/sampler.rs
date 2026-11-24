//! Token sampling (fast GPU argmax greedy + deterministic CPU temperature / top-k / top-p).

use candle_core::{DType, Result, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Per-request generation & sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: Option<u64>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            max_tokens: 256,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            ignore_eos: false,
        }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-6
    }
}

/// Stateful per-sequence sampler (holds deterministic RNG when stochastic).
pub struct Sampler {
    params: SamplingParams,
    rng: StdRng,
}

impl Sampler {
    pub fn new(params: SamplingParams, default_seed: u64) -> Self {
        let seed = params.seed.unwrap_or(default_seed);
        Self {
            params,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    pub fn is_greedy(&self) -> bool {
        self.params.is_greedy()
    }

    /// Sample one token ID from a single logit row `[vocab]` (or `[1, vocab]`).
    ///
    /// When `temperature == 0.0`, performs argmax directly on device (zero 1 MB vocab copy to CPU!).
    pub fn sample_row(&mut self, logits: &Tensor) -> Result<u32> {
        let row = if logits.rank() == 2 {
            logits.squeeze(0)?
        } else {
            logits.clone()
        };
        if self.params.is_greedy() {
            return row.argmax(0)?.to_scalar::<u32>();
        }
        let v = row.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(self.sample_from_f32(&v))
    }

    /// Sample a batch of rows `[B, vocab]` when all rows are greedy (`argmax(1)` on GPU in 1 call).
    pub fn sample_batch_greedy(logits: &Tensor) -> Result<Vec<u32>> {
        logits.argmax(1)?.to_vec1::<u32>()
    }

    /// Sample one token from F32 logits on CPU using temperature + top_k + top_p.
    pub fn sample_from_f32(&mut self, logits: &[f32]) -> u32 {
        if self.params.is_greedy() {
            return argmax_f32(logits);
        }
        let probs = logits_to_filtered_probs(
            logits,
            self.params.temperature,
            self.params.top_k,
            self.params.top_p,
        );
        sample_categorical(&probs, &mut self.rng)
    }

    /// Sample a uniform `[0, 1)` float from the sequence's deterministic RNG (for MTP rejection).
    pub fn next_uniform(&mut self) -> f32 {
        self.rng.random::<f32>()
    }
}

pub fn argmax_f32(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Convert raw F32 logits into normalized probabilities after temperature, top-k, and top-p.
pub fn logits_to_filtered_probs(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Vec<(u32, f32)> {
    let inv_t = 1.0f32 / temperature.max(1e-6);
    let max_l = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut pairs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, ((l - max_l) * inv_t).exp()))
        .collect();

    if top_k > 0 && top_k < pairs.len() {
        pairs.select_nth_unstable_by(top_k, |a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(top_k);
    }

    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let sum: f32 = pairs.iter().map(|(_, p)| *p).sum::<f32>().max(1e-30);
    for (_, p) in &mut pairs {
        *p /= sum;
    }

    if top_p > 0.0 && top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut keep = pairs.len();
        for (idx, (_, p)) in pairs.iter().enumerate() {
            cum += *p;
            if cum >= top_p {
                keep = idx + 1;
                break;
            }
        }
        pairs.truncate(keep.max(1));
        let renorm: f32 = pairs.iter().map(|(_, p)| *p).sum::<f32>().max(1e-30);
        for (_, p) in &mut pairs {
            *p /= renorm;
        }
    }
    pairs
}

pub fn sample_categorical(probs: &[(u32, f32)], rng: &mut StdRng) -> u32 {
    let r: f32 = rng.random();
    let mut acc = 0.0f32;
    for &(id, p) in probs {
        acc += p;
        if r <= acc {
            return id;
        }
    }
    probs.first().map_or(0, |&(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_seeded_sampling_are_deterministic() {
        let logits = vec![-1.0f32, 0.5, 2.3, 1.1, 2.3];
        assert_eq!(argmax_f32(&logits), 2);

        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 4,
            seed: Some(42),
            ..Default::default()
        };
        let mut s1 = Sampler::new(params.clone(), 0);
        let mut s2 = Sampler::new(params, 0);
        let a: Vec<u32> = (0..16).map(|_| s1.sample_from_f32(&logits)).collect();
        let b: Vec<u32> = (0..16).map(|_| s2.sample_from_f32(&logits)).collect();
        assert_eq!(a, b);
    }
}
