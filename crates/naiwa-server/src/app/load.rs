//! Local GGUF model & MoE offloading provider loader (zero `mistral.rs` dependency).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Result};
#[cfg(feature = "cuda")]
use anyhow::Context;
use candle_core::{DType, Device};
use naiwa_models::Qwen4ExpModel;
#[cfg(feature = "cuda")]
use naiwa_offload::pipeline::prefill::PrefillStreams;
#[cfg(feature = "cuda")]
use naiwa_offload::pipeline::GpuCacheMoeProvider;
use naiwa_quant::RoutedExpertsProvider;
use naiwa_runtime::GgufTokenizer;
use tracing::info;

use crate::app::args::{
    ensure_local_dir, expand_home_in_list, split_path_list, ModelOptions, ModelSource,
};

#[cfg(feature = "cuda")]
const PROFILE_LAYERS: usize = 128;
#[cfg(feature = "cuda")]
const PROFILE_EXPERTS: usize = 1024;

/// Loaded model, tokenizer, display ID, and optional offloading provider ready for `EngineHandle`.
pub struct LoadedServerModel {
    pub display_id: String,
    pub model: Qwen4ExpModel,
    pub tokenizer: Arc<GgufTokenizer>,
    #[cfg(feature = "cuda")]
    pub provider: Option<Arc<GpuCacheMoeProvider>>,
    pub device: Device,
    pub dtype: DType,
}

/// Resolve GGUF shard files (and optional `--mmproj` extra GGUF component such as `mtp-component.gguf`).
pub fn resolve_gguf_paths(source: &ModelSource) -> Result<(Vec<PathBuf>, String)> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let display_id: String;

    if let Some(raw_files) = source.quantized_file.as_deref() {
        let expanded = expand_home_in_list(raw_files);
        let entries = split_path_list(&expanded)?;
        if let Some(dir) = &source.model_id {
            ensure_local_dir(dir, "Model directory")?;
            display_id = dir.to_string_lossy().into_owned();
            for entry in entries {
                let p = PathBuf::from(entry);
                let full = if p.is_absolute() { p } else { dir.join(p) };
                ensure_local_file(&full, "GGUF file")?;
                paths.push(full);
            }
        } else {
            display_id = entries[0].to_string();
            for entry in entries {
                let full = PathBuf::from(entry);
                ensure_local_file(&full, "GGUF file")?;
                paths.push(full);
            }
        }
    } else if let Some(dir) = &source.model_id {
        ensure_local_dir(dir, "Model directory")?;
        display_id = dir.to_string_lossy().into_owned();
        let mut discovered: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
            .collect();
        discovered.sort();
        if discovered.is_empty() {
            bail!(
                "no `.gguf` files found in `{}` (naiwa-rs supports GGUF text checkpoints only)",
                dir.display()
            );
        }
        paths.extend(discovered);
    } else {
        bail!("pass `-m <model-dir>` or `-f <model.gguf>`");
    }

    if let Some(extra) = source.mmproj.as_deref() {
        let expanded = expand_home_in_list(extra);
        for entry in split_path_list(&expanded)? {
            let p = PathBuf::from(entry);
            let full = if p.is_absolute() {
                p
            } else if let Some(dir) = &source.model_id {
                dir.join(p)
            } else {
                p
            };
            ensure_local_file(&full, "Extra GGUF component (--mmproj)")?;
            paths.push(full);
        }
    }

    Ok((paths, display_id))
}

/// Load the GGUF model, MoE offloading provider, and tokenizer onto the target device.
pub fn load_model(
    model_opts: &ModelOptions,
    load_mtp: bool,
    chat_template_override: Option<&Path>,
) -> Result<LoadedServerModel> {
    let (paths, display_id) = resolve_gguf_paths(&model_opts.source)?;

    let device = if model_opts.device.cpu {
        Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).context("failed to initialize CUDA device 0")?
        }
        #[cfg(not(feature = "cuda"))]
        {
            Device::Cpu
        }
    };
    let dtype = model_opts.source.dtype.resolve(device.is_cuda());

    #[cfg(feature = "cuda")]
    let provider: Option<Arc<GpuCacheMoeProvider>> = if model_opts.cpu_moe {
        let cache_gb = model_opts
            .expert_cache_gb
            .context("`--cpu-moe` requires `--expert-cache-gb`")?;
        let prefill = model_opts.prefill_stream.then(|| {
            info!(
                "--prefill-stream: prefill chunks >= {} tokens stream routed experts on GPU",
                model_opts.prefill_stream_min_tokens.get()
            );
            PrefillStreams::new(
                model_opts.prefill_stream_min_tokens.get(),
                usize::from(model_opts.prefill_stream_depth),
                usize::from(model_opts.prefill_split),
            )
        });
        let profile = match &model_opts.expert_profile {
            Some(path) => Some(Arc::new(
                naiwa_offload::memory::Profile::read(path, PROFILE_LAYERS, PROFILE_EXPERTS)
                    .map_err(|e| anyhow::anyhow!("--expert-profile {}: {e}", path.display()))?,
            )),
            None => None,
        };
        let p = Arc::new(GpuCacheMoeProvider::new(
            cache_gb,
            prefill,
            model_opts.pcie_frac,
            model_opts.lookahead_swaps,
            !model_opts.no_lookahead,
            profile,
        ));
        info!(
            "--cpu-moe + expert cache: routed experts in host memory; {:.1} GiB GPU cache (pcie_frac={}, lookahead={})",
            cache_gb, model_opts.pcie_frac, !model_opts.no_lookahead
        );
        Some(p)
    } else {
        None
    };

    #[cfg(not(feature = "cuda"))]
    if model_opts.cpu_moe {
        bail!("`--cpu-moe` GPU expert cache offloading requires a CUDA-enabled build of naiwa-rs");
    }

    #[cfg(feature = "cuda")]
    if naiwa_offload::stats::model_partition::enabled() {
        naiwa_models::set_boundary_probe(naiwa_offload::stats::model_partition::dispatch);
    }

    #[cfg(feature = "cuda")]
    let moe_trait: Option<Arc<dyn RoutedExpertsProvider>> =
        provider.as_ref().map(|p| Arc::clone(p) as Arc<dyn RoutedExpertsProvider>);
    #[cfg(not(feature = "cuda"))]
    let moe_trait: Option<Arc<dyn RoutedExpertsProvider>> = None;

    info!(
        "Loading `{display_id}` ({} GGUF file(s), device={device:?}, dtype={dtype:?}, mtp={load_mtp})",
        paths.len()
    );
    let (model, archive) =
        Qwen4ExpModel::load_gguf(&paths, &device, dtype, moe_trait, load_mtp)?;

    let tokenizer = Arc::new(GgufTokenizer::from_gguf(
        &archive,
        model_opts.source.tokenizer.as_deref(),
        chat_template_override,
    )?);

    Ok(LoadedServerModel {
        display_id,
        model,
        tokenizer,
        #[cfg(feature = "cuda")]
        provider,
        device,
        dtype,
    })
}

fn ensure_local_file(path: &Path, what: &str) -> Result<()> {
    if !path.is_file() {
        bail!("{what} `{}` does not exist or is not a file", path.display());
    }
    Ok(())
}

/// Query free CUDA VRAM (bytes) when running on CUDA, else `None`.
pub fn query_free_vram_bytes(device: &Device) -> Option<usize> {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        return naiwa_offload::transport::free_bytes(device).map(|b| b as usize);
    }
    let _ = device;
    None
}
