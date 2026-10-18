//! Turns local model arguments into the mistral.rs loader selection and server builder.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use mistralrs_core::{
    HfConfigOverrides, IsqOrganization, ModelSelected, MultimodalLoaderType, UqffWriteConfig,
};
use mistralrs_quant::GgufArchive;
use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;

use crate::args::{
    ensure_local_dir, expand_home_in_list, split_path_list, DeviceOptions, GlobalOptions,
    ModelOptions, ModelSource,
};
use crate::offload::{install_cpu_moe, ALL_LAYERS_ON_DEVICE};

const MEBIBYTE_BYTES: usize = 1024 * 1024;

/// The architecture this server runs on, across both on-disk representations.
const HF_ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";
const GGUF_ARCHITECTURE: &str = "qwen4exp";

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
        ensure_flash_next_config(&model_dir, source.hf_overrides.as_ref())?;
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
        bail!(
            "`--from-uqff` loads safetensors-derived UQFF files; it cannot be combined with `-f`"
        );
    }
    let raw_files = expand_home_in_list(raw_files);
    let files = split_path_list(&raw_files)?;
    if let Some(file) = files
        .iter()
        .find(|file| !file.to_ascii_lowercase().ends_with(".gguf"))
    {
        bail!("`-f` only accepts GGUF files; `{file}` is not a `.gguf` file");
    }
    let (root, filenames) = gguf_root(source.model_id.as_deref(), &files)?;
    let shards: Vec<PathBuf> = filenames.iter().map(|name| root.join(name)).collect();
    ensure_flash_next_gguf(&shards)?;

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
        display_id: shards[0].to_string_lossy().into_owned(),
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
    let device_layers = device_layers(model);
    if model.cpu_moe {
        // The fused CPU MoE decode's parallel passes use the fork's rayon default: the v2
        // configuration is experts-on-rayon, predictions-on-dedicated-threads (§6.3), fixed here
        // rather than switched by an environment variable (§6.4).
        install_cpu_moe(
            model.expert_cache_gb,
            model
                .prefill_stream
                .then_some(model.prefill_stream_min_tokens.get()),
            model.pcie_frac,
            model.lookahead_swaps,
            !model.no_lookahead,
        )?;
    }
    let paged = &model.paged_attn;
    let mut builder = MistralRsForServerBuilder::new()
        .with_model(local.selected)
        .with_interactive_mode(false)
        .with_cpu(model.device.cpu)
        .with_num_device_layers_optional(device_layers)
        .with_in_situ_quant_optional(model.quantization.in_situ_quant.clone())
        .with_max_model_len_optional(model.source.max_model_len)
        .with_hf_config_overrides_optional(model.source.hf_overrides.clone())
        .with_seed_optional(global.seed)
        .with_log_optional(
            global
                .log
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        )
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

/// `--device-layers` as given; with `--cpu-moe` and no explicit map, every layer on the device.
fn device_layers(model: &ModelOptions) -> Option<Vec<String>> {
    match &model.device.device_layers {
        None if model.cpu_moe => Some(vec![ALL_LAYERS_ON_DEVICE.to_string()]),
        layers => layers.clone(),
    }
}

fn unsupported(what: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "naiwa-rs only runs Qwen3.8-Flash-Next ({HF_ARCHITECTURE}); {what} is not supported"
    )
}

/// Fail unless `model_dir/config.json`, after `hf_overrides`, declares Qwen3.8-Flash-Next.
fn ensure_flash_next_config(
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

/// Fail unless the GGUF model (every shard of a split file) has `general.architecture = qwen4exp`
/// and stores its tensors in types candle can run.
fn ensure_flash_next_gguf(shards: &[PathBuf]) -> Result<()> {
    let path = &shards[0];
    let archive = GgufArchive::open(shards)
        .with_context(|| format!("failed to read GGUF header of `{}`", path.display()))?;
    let Some(architecture) = archive.metadata_value("general.architecture") else {
        bail!("`{}` has no `general.architecture`", path.display());
    };
    let architecture = architecture.to_string()?;
    if !architecture.eq_ignore_ascii_case(GGUF_ARCHITECTURE) {
        return Err(unsupported(&format!(
            "GGUF `{}` has architecture `{architecture}`, which",
            path.display()
        )));
    }
    // The fork reads the PLE n-gram table itself (IQ4_NL included); every other tensor goes through candle.
    let mut unrunnable = BTreeMap::<&str, usize>::new();
    for info in archive.tensors().values() {
        if info.name() != "per_layer_token_embd.weight" && info.dtype().candle_dtype().is_err() {
            *unrunnable.entry(info.dtype().name()).or_default() += 1;
        }
    }
    if unrunnable.is_empty() {
        return Ok(());
    }
    let types = unrunnable
        .iter()
        .map(|(name, count)| format!("{name} x{count}"))
        .collect::<Vec<_>>()
        .join(", ");
    bail!(
        "`{}` stores tensors in quantization types candle cannot run ({types}); \
         re-quantize them to K-quants first",
        path.display()
    )
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_core::{Device, Tensor};
    use clap::Parser;

    use super::*;
    use crate::args::Cli;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("naiwa-rs-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &Path, architecture: &str) {
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"architectures":["{architecture}"]}}"#),
        )
        .unwrap();
    }

    fn write_gguf(path: &Path, architecture: &str) {
        let mut file = std::fs::File::create(path).unwrap();
        let architecture = gguf_file::Value::String(architecture.to_string());
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
        ensure_flash_next_gguf(&[good]).unwrap();
        assert!(ensure_flash_next_gguf(&[bad]).is_err());
    }

    /// A one-shard GGUF holding a single Q4_0 `blk.0.ffn_down_exps.weight`, then retyped to the
    /// GGUF dtype `id` in place. Q4_0's block layout (32 elements, 18 bytes) is shared with the IQ
    /// types used below, so the file's byte count stays consistent.
    fn retyped_gguf(name: &str, id: u32) -> PathBuf {
        let path = temp_dir(name).join("iq.gguf");
        let architecture = gguf_file::Value::String(GGUF_ARCHITECTURE.to_string());
        let weights = Tensor::zeros((2, 32), candle_core::DType::F32, &Device::Cpu).unwrap();
        let q4_0 = QTensor::quantize(&weights, GgmlDType::Q4_0).unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        gguf_file::write(
            &mut file,
            &[("general.architecture", &architecture)],
            &[("blk.0.ffn_down_exps.weight", &q4_0)],
        )
        .unwrap();
        drop(file);

        let mut bytes = std::fs::read(&path).unwrap();
        let name = b"blk.0.ffn_down_exps.weight";
        let at = bytes.windows(name.len()).position(|w| w == name).unwrap();
        let dtype = at + name.len() + 4 + 2 * 8; // n_dims: u32, dims: 2 x u64
        assert_eq!(bytes[dtype..dtype + 4], 2u32.to_le_bytes()); // Q4_0
        bytes[dtype..dtype + 4].copy_from_slice(&id.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn accepts_gguf_tensors_candle_can_run() {
        // IQ4_NL: native since the fork's IQ formats landed, so the loader must accept it. (This
        // test asserted the opposite until the engine's own test suite compiled again.)
        ensure_flash_next_gguf(&[retyped_gguf("gguf-iq4nl", 20)]).unwrap();
    }

    #[test]
    fn rejects_gguf_tensors_candle_cannot_run() {
        // Dtype id 4 is a retired legacy type with no candle mapping: rejected, and the error names
        // every offending type with its tensor count.
        let err = ensure_flash_next_gguf(&[retyped_gguf("gguf-unknown", 4)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("UNKNOWN x1"), "{err}");
        assert!(err.contains("candle cannot run"), "{err}");
    }

    fn model_options(args: &[&str]) -> ModelOptions {
        let mut argv = vec!["naiwa-rs", "tune"];
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
    fn cpu_moe_keeps_every_layer_on_the_device() {
        assert_eq!(device_layers(&model_options(&["-m", "x"])), None);
        assert_eq!(
            device_layers(&model_options(&["-m", "x", "--cpu-moe"])),
            Some(vec![usize::MAX.to_string()])
        );
        assert_eq!(
            device_layers(&model_options(&["-m", "x", "--cpu-moe", "-n", "0:10;1:20"])),
            Some(vec!["0:10".to_string(), "1:20".to_string()])
        );
    }

    #[test]
    fn rejects_missing_model_dirs() {
        let model = model_options(&["-m", "Qwen/Qwen3.8-Flash-Next"]);
        let err = select(&model).err().unwrap();
        assert!(err.to_string().contains("not a local directory"));
    }
}
