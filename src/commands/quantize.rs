//! `quantize`: write UQFF artifacts from a local Qwen3.8-Flash-Next checkpoint.

use std::collections::HashSet;

use anyhow::Result;
use mistralrs_core::{expand_isq_value, IsqType, UqffWriteConfig};
use mistralrs_server_core::mistralrs_for_server_builder::{defaults, MistralRsForServerBuilder};
use std::path::PathBuf;
use tracing::{info, warn};

use crate::args::{DeviceOptions, ModelSource, QuantizeOptions};
use crate::load::{select_model, QuantSelection};

pub async fn run_quantize(
    source: ModelSource,
    device: DeviceOptions,
    quantization: QuantizeOptions,
    output: PathBuf,
) -> Result<()> {
    let expanded_isq = expand_isq_values(&quantization.in_situ_quant)?;
    let file_mode = output.extension().is_some_and(|ext| ext == "uqff");

    // Multiple expanded ISQ types require directory output mode
    if expanded_isq.len() > 1 && file_mode {
        anyhow::bail!(
            "Cannot use multiple --isq values with a .uqff output path (ISQ setting produced multiple expanded ISQ values). \
             Use a directory path (e.g., -o output/) to auto-name files per ISQ type."
        );
    }

    let effective_output = if file_mode {
        output.clone()
    } else if expanded_isq.len() == 1 {
        std::fs::create_dir_all(&output)?;
        output.join(format!("{}.uqff", expanded_isq[0]))
    } else {
        std::fs::create_dir_all(&output)?;
        output.clone()
    };
    let requested = expanded_isq
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");

    let base_model = source
        .model_id
        .as_ref()
        .map(|dir| dir.to_string_lossy().into_owned())
        .or_else(|| source.quantized_file.clone());
    let write_uqff = UqffWriteConfig::with_types(effective_output.clone(), expanded_isq)
        .with_report_metadata(base_model, None);
    let local = select_model(
        &source,
        &device,
        QuantSelection {
            isq_organization: quantization.isq_organization,
            imatrix: quantization.imatrix.clone(),
            calibration_file: quantization.calibration_file.clone(),
            from_uqff: None,
            write_uqff: Some(write_uqff),
        },
    )?;

    info!(
        "Starting UQFF generation for ISQ=[{}] -> `{}`",
        requested,
        effective_output.display()
    );

    let mistralrs = MistralRsForServerBuilder::new()
        .with_model(local.selected)
        .with_max_seqs(1)
        .with_no_kv_cache(defaults::NO_KV_CACHE)
        .with_interactive_mode(defaults::INTERACTIVE_MODE)
        .with_prefix_cache_n(0)
        .set_paged_attn(Some(false))
        .with_cpu(device.cpu)
        .with_num_device_layers_optional(device.device_layers.clone())
        .build()
        .await?;
    mistralrs.shutdown().await.map_err(anyhow::Error::msg)?;

    info!("UQFF generation for ISQ=[{}] complete!", requested);
    Ok(())
}

/// Expand numeric ISQ shorthands into concrete variants, deduplicated by type.
fn expand_isq_values(values: &[String]) -> Result<Vec<IsqType>> {
    let mut seen_strings = HashSet::new();
    let mut seen_types = HashSet::new();
    let mut expanded = Vec::new();
    for value in values {
        if !seen_strings.insert(value.to_lowercase()) {
            warn!("Duplicate --isq value '{}'; skipping.", value);
            continue;
        }
        for tp in expand_isq_value(value)? {
            if seen_types.insert(tp) {
                expanded.push(tp);
            }
        }
    }
    Ok(expanded)
}
