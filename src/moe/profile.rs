//! strata's expert-cache profile: every `(layer, expert)` pair, ranked by how often routing picked it.
//!
//! The file (`data/expert-profile-coder.bin`, written by strata's `tools/make_profile.py`) is:
//!
//! ```text
//! "STRP"                                  magic
//! <5I: version, layers, experts, slots, n> little-endian, version 1
//! n * <HH: layer, expert>                 the ranking, best first
//! layers * experts * <i                   table[layer][expert] = rank, or -1 (unused here)
//! ```
//!
//! strata uses it to decide which experts are resident *before* the first token (`expert cache 1028 slots,
//! pre-filled 1028 of 1028 slots from the profile in 0.2 s`). We use it for the same reason: our slots are
//! per layer, so each layer takes its own highest-ranked experts, in rank order, until it runs out of slots.
//!
//! Only the ranking is read. The trailing table is redundant (it is the ranking inverted) and strata's own reader
//! (`make_profile.read_profile`) ignores it too.

use std::path::Path;

/// A parsed profile: for each layer, its experts in descending rank order.
pub struct Profile {
    per_layer: Vec<Vec<usize>>,
    /// The file it came from, for the log line.
    pub source: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ProfileError {
    /// Not 4 bytes of `STRP`.
    Magic,
    /// A version this reader does not know.
    Version(u32),
    /// The pair table is shorter than its own header claims.
    Truncated { need: usize, have: usize },
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Magic => write!(f, "not a strata expert profile (no STRP magic)"),
            Self::Version(v) => write!(f, "profile version {v}, this reader knows 1"),
            Self::Truncated { need, have } => {
                write!(f, "profile truncated: {need} bytes of ranking, {have} in the file")
            }
        }
    }
}

impl std::error::Error for ProfileError {}

impl Profile {
    /// `num_layers` layers, and only experts below `num_experts` (a profile can be written for 512 while a pruned
    /// model keeps 256, which is exactly the coder pack's case: it ranks 48 x 256).
    pub fn read(
        path: &Path,
        num_layers: usize,
        num_experts: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let blob = std::fs::read(path)?;
        if blob.len() < 24 || &blob[..4] != b"STRP" {
            return Err(ProfileError::Magic.into());
        }
        let word = |i: usize| {
            u32::from_le_bytes([blob[4 + 4 * i], blob[5 + 4 * i], blob[6 + 4 * i], blob[7 + 4 * i]]) as usize
        };
        let (version, layers, experts, _slots, n) = (word(0), word(1), word(2), word(3), word(4));
        if version != 1 {
            return Err(ProfileError::Version(version as u32).into());
        }
        let need = 24 + 4 * n;
        if blob.len() < need {
            return Err(ProfileError::Truncated { need, have: blob.len() }.into());
        }
        // The ranking is global, its layers interleaved: keep this layer's pairs in the order they appear, which
        // is their rank order.
        let mut per_layer = vec![Vec::new(); num_layers];
        let mut seen = vec![vec![false; num_experts]; num_layers];
        for i in 0..n {
            let off = 24 + 4 * i;
            let layer = u16::from_le_bytes([blob[off], blob[off + 1]]) as usize;
            let expert = u16::from_le_bytes([blob[off + 2], blob[off + 3]]) as usize;
            if layer >= num_layers || expert >= num_experts || seen[layer][expert] {
                continue;
            }
            seen[layer][expert] = true;
            per_layer[layer].push(expert);
        }
        Ok(Self {
            per_layer,
            source: format!("{} ({layers}x{experts}, {n} ranked pairs)", path.display()),
        })
    }

    /// This layer's experts, best first.
    pub fn layer(&self, layer: usize) -> &[usize] {
        self.per_layer.get(layer).map_or(&[], |v| v.as_slice())
    }

    /// Total pairs read, after the layer/expert filter.
    pub fn pairs(&self) -> usize {
        self.per_layer.iter().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(pairs: &[(u16, u16)], layers: u32, experts: u32) -> Vec<u8> {
        let mut b = b"STRP".to_vec();
        for v in [1u32, layers, experts, pairs.len() as u32, pairs.len() as u32] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for (l, e) in pairs {
            b.extend_from_slice(&l.to_le_bytes());
            b.extend_from_slice(&e.to_le_bytes());
        }
        b
    }

    #[test]
    fn reads_the_ranking_per_layer_in_rank_order() {
        std::env::set_var("TMPDIR", "/tmp");
        let path = std::env::temp_dir().join("naiwa-profile-order.bin");
        // Interleaved layers, as the real file is: (1,5) ranks above (0,9) above (1,2).
        std::fs::write(&path, write(&[(1, 5), (0, 9), (1, 2), (0, 7)], 2, 16)).unwrap();
        let p = Profile::read(&path, 2, 16).unwrap();
        assert_eq!(p.layer(0), &[9, 7]);
        assert_eq!(p.layer(1), &[5, 2]);
        assert_eq!(p.pairs(), 4);
        // A layer the file does not mention, and one past the model's count, are both empty rather than a panic.
        assert_eq!(p.layer(1), &[5, 2]);
        assert!(p.layer(2).is_empty());
    }

    #[test]
    fn drops_pairs_outside_the_model() {
        let path = std::env::temp_dir().join("naiwa-profile-filter.bin");
        // Expert 300 does not exist in a 256-expert model; the duplicate is kept once.
        std::fs::write(&path, write(&[(0, 3), (0, 300), (0, 3), (9, 1)], 2, 256)).unwrap();
        let p = Profile::read(&path, 2, 256).unwrap();
        assert_eq!(p.layer(0), &[3]);
        assert_eq!(p.pairs(), 1);
    }

    #[test]
    fn rejects_what_it_cannot_read() {
        let path = std::env::temp_dir().join("naiwa-profile-bad.bin");
        std::fs::write(&path, b"NOPE").unwrap();
        assert!(Profile::read(&path, 2, 16).is_err());
        std::fs::write(&path, write(&[(0, 1)], 2, 16)[..26].to_vec()).unwrap(); // header says 1 pair, no room
        assert!(matches!(
            Profile::read(&path, 2, 16).map(|_| ()),
            Err(e) if e.to_string().contains("truncated")
        ));
    }
}
