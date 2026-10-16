//! `tune`: recommend quantization and device mapping for this machine.

use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use mistralrs_core::{auto_tune, AutoTuneRequest, FitStatus, QualityTier, TokenSource};

use crate::args::{ModelOptions, TuneProfileArg};
use crate::load::{select_model, QuantSelection};

pub async fn run_tune(model: ModelOptions, profile: TuneProfileArg, json: bool) -> Result<()> {
    if model.cpu_moe {
        anyhow::bail!("`tune` estimates GPU-resident experts; `--cpu-moe` is not modeled");
    }
    let local = select_model(
        &model.source,
        &model.device,
        QuantSelection {
            isq_organization: model.quantization.isq_organization,
            imatrix: None,
            calibration_file: None,
            from_uqff: model.quantization.from_uqff.clone(),
            write_uqff: None,
        },
    )?;
    let requested_isq = model
        .quantization
        .in_situ_quant
        .as_deref()
        .map(|s| {
            mistralrs_core::parse_isq_value(s, None)
                .map_err(|err| anyhow::anyhow!("Invalid quantization value: {err}"))
        })
        .transpose()?;

    let result = auto_tune(AutoTuneRequest {
        model: local.selected,
        token_source: TokenSource::None,
        hf_revision: None,
        force_cpu: model.device.cpu,
        profile: profile.into(),
        requested_isq,
    })?;

    if json {
        let out = serde_json::to_string_pretty(&result)?;
        println!("{out}");
        return Ok(());
    }

    // Header
    println!();
    println!("Tuning Analysis");
    println!("===============");
    println!();
    println!("Model: {}", result.model_id);
    println!("Profile: {:?}", result.profile);
    println!("Backend: {}", result.backend);
    if result.total_vram_bytes > 0 {
        println!("Total VRAM: {:.1} GB", result.total_vram_bytes as f64 / 1e9);
    }
    println!();

    // Notes
    if !result.notes.is_empty() {
        for note in &result.notes {
            println!("[INFO] {note}");
        }
        println!();
    }

    // Analysis Matrix Table
    println!("Quantization Options");
    println!("--------------------");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Quant"),
            Cell::new("Est. Size"),
            Cell::new("VRAM %"),
            Cell::new("Context Room"),
            Cell::new("Quality"),
            Cell::new("Status"),
        ]);

    for candidate in &result.candidates {
        let size_str = format!("{:.2} GB", candidate.estimated_size_bytes as f64 / 1e9);
        let vram_pct = format!("{:.0}%", candidate.vram_usage_percent * 100.0);
        let context_str = {
            let base = if candidate.max_context_tokens >= 1000 {
                format!("{}k", candidate.max_context_tokens / 1000)
            } else {
                format!("{}", candidate.max_context_tokens)
            };
            if candidate.context_is_model_max {
                format!("{base} (max)")
            } else {
                base
            }
        };

        let quality_str = match candidate.quality {
            QualityTier::Baseline => "Baseline",
            QualityTier::NearLossless => "Near-lossless",
            QualityTier::Good => "Good",
            QualityTier::Acceptable => "Acceptable",
            QualityTier::Degraded => "Degraded",
        };

        let (status_str, status_color) = match candidate.fit_status {
            FitStatus::Fits if candidate.recommended => ("🚀 Recommended", Color::Green),
            FitStatus::Fits => ("✅ Fits", Color::Green),
            FitStatus::Hybrid if candidate.recommended => {
                ("🚀 Recommended (Hybrid)", Color::Yellow)
            }
            FitStatus::Hybrid => ("⚠️ Hybrid", Color::Yellow),
            FitStatus::TooLarge => ("❌ Too Large", Color::Red),
        };

        table.add_row(vec![
            Cell::new(&candidate.isq_name),
            Cell::new(&size_str),
            Cell::new(&vram_pct),
            Cell::new(&context_str),
            Cell::new(quality_str),
            Cell::new(status_str).fg(status_color),
        ]);
    }

    println!("{table}");
    println!();

    // Warnings
    if !result.warnings.is_empty() {
        println!();
        for warning in &result.warnings {
            println!("[WARN] {warning}");
        }
    }

    // Recommended command
    println!();
    println!("Recommended Command");
    println!("-------------------");
    let command = result.recommended_command.trim_start();
    let command = command.strip_prefix("mistralrs ").unwrap_or(command);
    println!("  naiwa-rs {command}");
    println!();

    if let Some(mode) = &result.paged_attn_mode {
        if mode != "off" {
            println!("[INFO] PagedAttention is available (mode: {mode})");
        }
    }

    Ok(())
}
