//! Turns local model arguments into the mistral.rs loader selection and server builder.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use mistralrs_core::{
    IsqOrganization, ModelSelected, MultimodalLoaderType, UqffWriteConfig,
};
use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;

use crate::args::{
    ensure_local_dir, expand_home_in_list, split_path_list, DeviceOptions, GlobalOptions,
    ModelOptions, ModelSource,
};
use crate::guard;

const MEBIBYTE_BYTES: usize = 1024 * 1024;

/// Quantization inputs that shape the loader selection.
pub struct QuantSelection {
    pub isq_organization: Option<IsqOrganization>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub from_uqff: Option<String>,
    pub write_uqff: Option<UqffWriteConfig>,
}

/// A checked Qwen3.8-Flash-Next checkpoint on the local filesystem.
pub struct LocalModel {
    pub selected: ModelSelected,
    /// Human-readable identifier (the model directory or GGUF file).
    pub display_id: String,
}

/// Resolve local paths, verify the architecture, and build the loader selection.
pub fn select_model(
    source: &ModelSource,
    device: &DeviceOptions,
    quant: QuantSelection,
) -> Result<LocalModel> {
    let topology = device
        .topology
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned());
    let tokenizer_json = source
        .tokenizer
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned());

    let Some(raw_files) = source.quantized_file.as_deref() else {
        if source.mmproj.is_some() {
            bail!("`--mmproj` requires GGUF weights passed with `-f`");
        }
        let model_dir = source
            .model_id
            .clone()
            .context("pass `-m <model-dir>` or `-f <model.gguf>`")?;
        ensure_local_dir(&model_dir, "Model directory")?;
        guard::ensure_flash_next_config(&model_dir, source.hf_overrides.as_ref())?;
        let display_id = model_dir.to_string_lossy().into_owned();
        // The Qwen4-Exp loader is selected explicitly; no auto-detection takes place.
        let selected = ModelSelected::MultimodalPlain {
            model_id: display_id.clone(),
            tokenizer_json,
            arch: Some(MultimodalLoaderType::Qwen4Exp),
            dtype: source.dtype,
            topology,
            write_uqff: quant.write_uqff,
            from_uqff: quant.from_uqff,
            max_edge: source.multimodal.max_edge,
            calibration_file: quant.calibration_file,
            imatrix: quant.imatrix,
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            max_num_images: source.multimodal.max_num_images.unwrap_or(1),
            max_image_length: source.multimodal.max_image_length.unwrap_or(1024),
            hf_cache_path: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            organization: quant.isq_organization,
        };
        return Ok(LocalModel {
            selected,
            display_id,
        });
    };

    if quant.from_uqff.is_some() {
        bail!("`--from-uqff` loads safetensors-derived UQFF files; it cannot be combined with `-f`");
    }
    let raw_files = expand_home_in_list(raw_files);
    let files = split_path_list(&raw_files)?;
    if let Some(file) = files.iter().find(|file| !file.to_ascii_lowercase().ends_with(".gguf")) {
        bail!("`-f` only accepts GGUF files; `{file}` is not a `.gguf` file");
    }
    let (root, filenames) = gguf_root(source.model_id.as_deref(), &files)?;
    let first = root.join(&filenames[0]);
    guard::ensure_flash_next_gguf(&first)?;

    let mmproj = source
        .mmproj
        .as_deref()
        .map(|raw| -> Result<String> {
            let raw = expand_home_in_list(raw);
            let paths = split_path_list(&raw)?
                .into_iter()
                .map(|entry| local_file(&root, entry))
                .collect::<Result<Vec<_>>>()?;
            Ok(paths
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(";"))
        })
        .transpose()?;
    if let Some(tok_dir) = &source.tok_model_id {
        ensure_local_dir(tok_dir, "`--tok-model-id`")?;
    }

    let selected = ModelSelected::GGUF {
        tok_model_id: source
            .tok_model_id
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
        quantized_model_id: root.to_string_lossy().into_owned(),
        quantized_filename: filenames.join(";"),
        tokenizer_json,
        mmproj_filename: mmproj,
        lora_adapters: Vec::new(),
        lora_runtime_config: None,
        dtype: source.dtype,
        topology,
        organization: quant.isq_organization,
        write_uqff: quant.write_uqff,
        imatrix: quant.imatrix,
        calibration_file: quant.calibration_file,
        max_edge: source.multimodal.max_edge,
        max_seq_len: device.max_seq_len,
        max_batch_size: device.max_batch_size,
        max_num_images: source.multimodal.max_num_images,
        max_image_length: source.multimodal.max_image_length,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };
    Ok(LocalModel {
        selected,
        display_id: first.to_string_lossy().into_owned(),
    })
}

/// The directory GGUF shards are loaded from, plus each shard's path relative to it.
fn gguf_root(model_dir: Option<&Path>, files: &[&str]) -> Result<(PathBuf, Vec<String>)> {
    if let Some(model_dir) = model_dir {
        ensure_local_dir(model_dir, "Model directory")?;
        for file in files {
            local_file(model_dir, file)?;
        }
        let filenames = files.iter().map(|file| file.to_string()).collect();
        return Ok((model_dir.to_path_buf(), filenames));
    }

    let mut root: Option<PathBuf> = None;
    let mut filenames = Vec::with_capacity(files.len());
    for file in files {
        let path = Path::new(file);
        if !path.is_file() {
            bail!("GGUF file `{}` does not exist", path.display());
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if root.as_deref().is_some_and(|current| current != parent) {
            bail!("GGUF shards must share one directory when `-m` is omitted");
        }
        root = Some(parent.to_path_buf());
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("invalid GGUF path `{file}`"))?;
        filenames.push(name.to_string());
    }
    Ok((root.unwrap_or_else(|| PathBuf::from(".")), filenames))
}

/// Resolve `entry` against `root`, then the current directory, requiring an existing file.
fn local_file(root: &Path, entry: &str) -> Result<PathBuf> {
    let path = Path::new(entry);
    let candidates = if path.is_absolute() {
        vec![path.to_path_buf()]
    } else {
        vec![root.join(path), std::env::current_dir()?.join(path)]
    };
    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .with_context(|| format!("file `{entry}` does not exist under `{}`", root.display()))
}

/// A builder carrying the model plus the device, quantization, and cache settings shared by
/// every inference command.
pub fn base_builder(
    model: &ModelOptions,
    global: &GlobalOptions,
) -> Result<(MistralRsForServerBuilder, String)> {
    let local = select_model(
        &model.source,
        &model.device,
        QuantSelection {
            isq_organization: model.quantization.isq_organization,
            imatrix: model.quantization.imatrix.clone(),
            calibration_file: model.quantization.calibration_file.clone(),
            from_uqff: model.quantization.from_uqff.clone(),
            write_uqff: None,
        },
    )?;
    let paged = &model.paged_attn;
    let mut builder = MistralRsForServerBuilder::new()
        .with_model(local.selected)
        .with_interactive_mode(false)
        .with_cpu(model.device.cpu)
        .with_num_device_layers_optional(model.device.device_layers.clone())
        .with_in_situ_quant_optional(model.quantization.in_situ_quant.clone())
        .with_max_model_len_optional(model.source.max_model_len)
        .with_hf_config_overrides_optional(model.source.hf_overrides.clone())
        .with_seed_optional(global.seed)
        .with_log_optional(global.log.as_ref().map(|p| p.to_string_lossy().into_owned()))
        .set_paged_attn(paged.enabled())
        .with_paged_attn_gpu_mem_optional(paged.memory_mb)
        .with_paged_attn_gpu_mem_usage_optional(paged.memory_fraction)
        .with_paged_ctxt_len_optional(paged.context_len)
        .with_paged_attn_block_size_optional(paged.block_size)
        .with_paged_attn_cache_type(paged.cache_type);
    if let Some(memory_mb) = model.source.multimodal.encoder_cache_memory_mb {
        let bytes = memory_mb
            .get()
            .checked_mul(MEBIBYTE_BYTES)
            .context("encoder cache memory capacity overflow")?;
        builder = builder.with_encoder_cache_memory_bytes(bytes);
    }
    Ok((builder, local.display_id))
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::args::Cli;
    use crate::guard::tests::{temp_dir, write_config, write_gguf};

    fn model_options(args: &[&str]) -> ModelOptions {
        let mut argv = vec!["qwen-fn-rs", "tune"];
        argv.extend_from_slice(args);
        match Cli::try_parse_from(argv).unwrap().command {
            crate::args::Command::Tune { model, .. } => model,
            _ => unreachable!(),
        }
    }

    fn select(model: &ModelOptions) -> Result<LocalModel> {
        select_model(
            &model.source,
            &model.device,
            QuantSelection {
                isq_organization: None,
                imatrix: None,
                calibration_file: None,
                from_uqff: None,
                write_uqff: None,
            },
        )
    }

    #[test]
    fn safetensors_dir_forces_the_qwen4exp_loader() {
        let dir = temp_dir("select-plain");
        write_config(&dir, "Qwen4ExpForConditionalGeneration");
        let model = model_options(&["-m", dir.to_str().unwrap()]);
        let local = select(&model).unwrap();
        assert!(matches!(
            local.selected,
            ModelSelected::MultimodalPlain {
                arch: Some(MultimodalLoaderType::Qwen4Exp),
                ..
            }
        ));
    }

    #[test]
    fn gguf_without_model_dir_uses_the_file_directory() {
        let dir = temp_dir("select-gguf");
        write_gguf(&dir.join("flash-next-Q4_K_M.gguf"), "qwen4exp");
        let file = dir.join("flash-next-Q4_K_M.gguf");
        let model = model_options(&["-f", file.to_str().unwrap()]);
        let local = select(&model).unwrap();
        let ModelSelected::GGUF {
            quantized_model_id,
            quantized_filename,
            ..
        } = local.selected
        else {
            panic!("expected a GGUF selection");
        };
        assert_eq!(Path::new(&quantized_model_id), dir.as_path());
        assert_eq!(quantized_filename, "flash-next-Q4_K_M.gguf");
    }

    #[test]
    fn rejects_other_gguf_architectures() {
        let dir = temp_dir("select-gguf-other");
        let file = dir.join("llama.gguf");
        write_gguf(&file, "llama");
        let model = model_options(&["-f", file.to_str().unwrap()]);
        assert!(select(&model).is_err());
    }

    #[test]
    fn rejects_missing_model_dirs() {
        let model = model_options(&["-m", "Qwen/Qwen3.8-Flash-Next"]);
        let err = select(&model).err().unwrap();
        assert!(err.to_string().contains("not a local directory"));
    }
}
