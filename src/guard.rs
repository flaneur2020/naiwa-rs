//! Restricts every command to Qwen3.8-Flash-Next.
//!
//! The architecture is read from the same local artifacts the mistral.rs loaders use: the
//! (overridden) `config.json` for safetensors/UQFF checkpoints, or `general.architecture` in
//! the GGUF header.

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use mistralrs_core::HfConfigOverrides;

const HF_ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";
const GGUF_ARCHITECTURE: &str = "qwen4exp";

fn unsupported(what: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "qwen-fn-rs only runs Qwen3.8-Flash-Next ({HF_ARCHITECTURE}); {what} is not supported"
    )
}

/// Fail unless `model_dir/config.json`, after `hf_overrides`, declares Qwen3.8-Flash-Next.
pub fn ensure_flash_next_config(
    model_dir: &Path,
    hf_overrides: Option<&HfConfigOverrides>,
) -> Result<()> {
    let path = model_dir.join("config.json");
    let mut config = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read `{}`", path.display()))?;
    if let Some(overrides) = hf_overrides {
        config = overrides.apply(&config)?;
    }
    let config: serde_json::Value = serde_json::from_str(&config)
        .with_context(|| format!("failed to parse `{}`", path.display()))?;
    let architectures: Vec<&str> = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .map(|values| values.iter().filter_map(|value| value.as_str()).collect())
        .unwrap_or_default();
    if architectures.contains(&HF_ARCHITECTURE) {
        return Ok(());
    }
    Err(unsupported(&format!(
        "`{}` declares architectures {architectures:?}, which",
        model_dir.display()
    )))
}

/// Fail unless the GGUF file at `path` has `general.architecture = qwen4exp`.
pub fn ensure_flash_next_gguf(path: &Path) -> Result<()> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open `{}`", path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .with_context(|| format!("failed to read GGUF header of `{}`", path.display()))?;
    let Some(architecture) = content.metadata.get("general.architecture") else {
        bail!("`{}` has no `general.architecture`", path.display());
    };
    let architecture = architecture.to_string()?;
    if architecture.eq_ignore_ascii_case(GGUF_ARCHITECTURE) {
        return Ok(());
    }
    Err(unsupported(&format!(
        "GGUF `{}` has architecture `{architecture}`, which",
        path.display()
    )))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;

    use candle_core::quantized::gguf_file::Value;

    use super::*;

    pub(crate) fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("qwen-fn-rs-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub(crate) fn write_config(dir: &Path, architecture: &str) {
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"architectures":["{architecture}"]}}"#),
        )
        .unwrap();
    }

    pub(crate) fn write_gguf(path: &Path, architecture: &str) {
        let mut file = std::fs::File::create(path).unwrap();
        let architecture = Value::String(architecture.to_string());
        gguf_file::write(&mut file, &[("general.architecture", &architecture)], &[]).unwrap();
    }

    #[test]
    fn accepts_flash_next_config() {
        let dir = temp_dir("accept-config");
        write_config(&dir, HF_ARCHITECTURE);
        ensure_flash_next_config(&dir, None).unwrap();
    }

    #[test]
    fn rejects_other_configs() {
        let dir = temp_dir("reject-config");
        write_config(&dir, "Qwen3_5ForConditionalGeneration");
        let err = ensure_flash_next_config(&dir, None).unwrap_err();
        assert!(err.to_string().contains("only runs Qwen3.8-Flash-Next"));
    }

    #[test]
    fn hf_overrides_cannot_bypass_the_check() {
        let dir = temp_dir("override-config");
        write_config(&dir, HF_ARCHITECTURE);
        let overrides: HfConfigOverrides =
            r#"{"architectures":["LlamaForCausalLM"]}"#.parse().unwrap();
        assert!(ensure_flash_next_config(&dir, Some(&overrides)).is_err());
    }

    #[test]
    fn checks_gguf_architecture() {
        let dir = temp_dir("gguf");
        let good = dir.join("good.gguf");
        let bad = dir.join("bad.gguf");
        write_gguf(&good, GGUF_ARCHITECTURE);
        write_gguf(&bad, "llama");
        ensure_flash_next_gguf(&good).unwrap();
        assert!(ensure_flash_next_gguf(&bad).is_err());
    }
}
