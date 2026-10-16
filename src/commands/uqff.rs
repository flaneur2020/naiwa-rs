//! `uqff`: report on or verify UQFF artifacts in a local directory.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::Range,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use mistralrs_core::{
    list_model_files, read_model_file_range, resolve_uqff_shorthand, try_get_model_file,
    TokenSource,
};
use mistralrs_quant::{
    build_uqff_report_from_artifacts, verify_uqff_artifacts, write_uqff_report, UqffArtifactFile,
    UqffArtifactGroup, UqffArtifacts, UqffGeneratedBy, UqffReport, UqffReportOptions,
    UqffVerifyOptions, UQFF_REPORT_JSON,
};

use crate::args::{ensure_local_dir, UqffCommand};

// Local directories ignore the revision; the core file helpers still take one.
const DEFAULT_REVISION: &str = "main";

pub async fn run_uqff(command: UqffCommand, verbose: bool) -> Result<()> {
    match command {
        UqffCommand::Report {
            model_id,
            quant,
            write,
            json,
            base_model,
        } => {
            run_report(
                &model_id,
                quant.as_deref(),
                write,
                json,
                verbose,
                base_model,
            )
            .await
        }
        UqffCommand::Verify {
            model_id,
            quant,
            json,
            strict,
            allow_newer_minor,
        } => run_verify(&model_id, quant.as_deref(), json, strict, allow_newer_minor).await,
    }
}

async fn run_report(
    model_dir: &Path,
    quant: Option<&str>,
    write: bool,
    json: bool,
    verbose: bool,
    base_model: Option<String>,
) -> Result<()> {
    let resolved = resolve_uqff_artifacts(model_dir, quant).await?;
    let report = build_uqff_report_from_artifacts(
        &resolved.artifacts,
        UqffReportOptions {
            generated_by: generated_by("naiwa-rs uqff report"),
            base_model,
            repo_id: None,
        },
    )
    .await
    .map_err(|e| anyhow!("{e}"))?;

    if write {
        let report_path =
            write_uqff_report(&resolved.local_write_path, &report).map_err(|e| anyhow!("{e}"))?;
        eprintln!("Wrote {}", report_path.display());
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report_summary(&report, verbose);
    }

    Ok(())
}

async fn run_verify(
    model_dir: &Path,
    quant: Option<&str>,
    json: bool,
    strict: bool,
    allow_newer_minor: bool,
) -> Result<()> {
    let resolved = resolve_uqff_artifacts(model_dir, quant).await?;
    let result = verify_uqff_artifacts(
        &resolved.artifacts,
        UqffVerifyOptions {
            strict,
            allow_newer_minor,
        },
    )
    .await
    .map_err(|e| anyhow!("{e}"))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        if result.ok {
            println!("OK: UQFF artifacts are structurally valid.");
        } else {
            println!("FAIL: UQFF verification failed.");
        }
        print_report_summary(&result.report, false);
        for warning in &result.warnings {
            println!("warning: {warning}");
        }
        for error in &result.errors {
            println!("error: {error}");
        }
    }

    if result.ok {
        Ok(())
    } else {
        Err(anyhow!("UQFF verification failed"))
    }
}

struct ResolvedUqffArtifacts {
    artifacts: UqffArtifacts,
    local_write_path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct UqffFileGroup {
    quant: String,
    files: Vec<String>,
    report_output: Option<usize>,
}

async fn resolve_uqff_artifacts(
    model_dir: &Path,
    quant: Option<&str>,
) -> Result<ResolvedUqffArtifacts> {
    ensure_local_dir(model_dir, "UQFF directory")?;
    let model_id = model_dir.to_string_lossy().into_owned();
    let model_id = model_id.as_str();
    let revision = DEFAULT_REVISION;
    let token_source = &TokenSource::None;
    let files = list_model_files(model_id, revision, token_source, true).await?;
    let existing_report =
        read_existing_uqff_report(model_id, revision, &files, token_source).await?;
    let selected = select_uqff_groups(&files, quant, existing_report.as_ref())?;
    let selected_report = selected_uqff_report(existing_report, &selected);

    let mut groups = Vec::with_capacity(selected.len());
    for group in selected {
        let artifact_files = group
            .files
            .into_iter()
            .map(|file| {
                let model_id = model_id.to_string();
                let revision = revision.to_string();
                let token_source = token_source.clone();
                let name = file.clone();
                UqffArtifactFile::new(name, move |range: Range<u64>| {
                    let model_id = model_id.clone();
                    let revision = revision.clone();
                    let file = file.clone();
                    let token_source = token_source.clone();
                    async move {
                        read_model_file_range(&model_id, &revision, &file, range, &token_source)
                            .await
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))
                    }
                })
            })
            .collect();
        groups.push(UqffArtifactGroup {
            quant: group.quant,
            files: artifact_files,
        });
    }

    let local_write_path = model_dir.to_path_buf();
    Ok(ResolvedUqffArtifacts {
        artifacts: UqffArtifacts {
            groups,
            existing_report: selected_report,
        },
        local_write_path,
    })
}

fn select_uqff_groups(
    files: &[String],
    quant: Option<&str>,
    report: Option<&UqffReport>,
) -> Result<Vec<UqffFileGroup>> {
    let uqff_files = files
        .iter()
        .filter(|file| file.ends_with(".uqff"))
        .cloned()
        .collect::<Vec<_>>();
    if uqff_files.is_empty() {
        return Err(anyhow!("No `.uqff` files were found."));
    }
    let groups = uqff_groups(&uqff_files, report)?;
    let Some(raw_quant) = quant.map(str::trim).filter(|quant| !quant.is_empty()) else {
        return Ok(groups);
    };
    if raw_quant.eq_ignore_ascii_case("all") {
        return Ok(groups);
    }

    let selected = groups
        .iter()
        .position(|group| group.files.iter().any(|file| file == raw_quant))
        .or_else(|| {
            groups.iter().position(|group| {
                group.report_output.is_some() && group.quant.eq_ignore_ascii_case(raw_quant)
            })
        })
        .or_else(|| resolve_report_quant_shorthand(raw_quant, &groups))
        .or_else(|| {
            groups.iter().position(|group| {
                group.report_output.is_none() && group.quant.eq_ignore_ascii_case(raw_quant)
            })
        })
        .or_else(|| resolve_legacy_quant_shorthand(raw_quant, &groups))
        .ok_or_else(|| {
            anyhow!(
                "No UQFF files matched `--quant {raw_quant}`. Available: {}",
                uqff_files.join(", ")
            )
        })?;
    Ok(vec![groups[selected].clone()])
}

fn uqff_groups(uqff_files: &[String], report: Option<&UqffReport>) -> Result<Vec<UqffFileGroup>> {
    let available = uqff_files
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut claimed = HashMap::new();
    let mut reported_quants = HashSet::new();
    let mut groups = Vec::new();

    if let Some(report) = report {
        if report.outputs.is_empty() {
            return Err(anyhow!("{UQFF_REPORT_JSON}: report has no outputs"));
        }
        groups.reserve(report.outputs.len());
        for (output_index, output) in report.outputs.iter().enumerate() {
            if output.quant.trim().is_empty() {
                return Err(anyhow!(
                    "{UQFF_REPORT_JSON}: output quant must not be empty"
                ));
            }
            if !reported_quants.insert(output.quant.to_ascii_lowercase()) {
                return Err(anyhow!(
                    "{UQFF_REPORT_JSON}: duplicate output quant `{}`",
                    output.quant
                ));
            }
            if output.shards.is_empty() {
                return Err(anyhow!(
                    "{UQFF_REPORT_JSON}: output `{}` has no shards",
                    output.quant
                ));
            }
            for shard in &output.shards {
                if !available.contains(shard.as_str()) {
                    return Err(anyhow!(
                        "{UQFF_REPORT_JSON}: output `{}` references missing shard `{shard}`",
                        output.quant
                    ));
                }
                if let Some(previous_quant) = claimed.insert(shard.clone(), output.quant.clone()) {
                    return Err(anyhow!(
                        "{UQFF_REPORT_JSON}: shard `{shard}` belongs to both `{previous_quant}` and `{}`",
                        output.quant
                    ));
                }
            }
            groups.push(UqffFileGroup {
                quant: output.quant.clone(),
                files: output.shards.clone(),
                report_output: Some(output_index),
            });
        }
    }

    let mut legacy = BTreeMap::<String, Vec<String>>::new();
    for file in uqff_files {
        if !claimed.contains_key(file) {
            legacy
                .entry(uqff_group_key(file))
                .or_default()
                .push(file.clone());
        }
    }
    for (quant, mut files) in legacy {
        files.sort_by_key(|file| uqff_shard_index(file).unwrap_or(0));
        groups.push(UqffFileGroup {
            quant,
            files,
            report_output: None,
        });
    }
    Ok(groups)
}

fn resolve_report_quant_shorthand(raw_quant: &str, groups: &[UqffFileGroup]) -> Option<usize> {
    let aliases = groups
        .iter()
        .filter(|group| group.report_output.is_some())
        .map(|group| format!("{}-0.uqff", group.quant.to_ascii_lowercase()))
        .collect::<Vec<_>>();
    let alias = resolve_uqff_shorthand(raw_quant, &aliases)?;
    groups.iter().position(|group| {
        group.report_output.is_some()
            && format!("{}-0.uqff", group.quant.to_ascii_lowercase()) == alias
    })
}

fn resolve_legacy_quant_shorthand(raw_quant: &str, groups: &[UqffFileGroup]) -> Option<usize> {
    let files = groups
        .iter()
        .filter(|group| group.report_output.is_none())
        .flat_map(|group| group.files.iter().cloned())
        .collect::<Vec<_>>();
    let first = resolve_uqff_shorthand(raw_quant, &files)?;
    groups.iter().position(|group| {
        group.report_output.is_none() && group.files.iter().any(|file| file == &first)
    })
}

fn selected_uqff_report(
    report: Option<UqffReport>,
    groups: &[UqffFileGroup],
) -> Option<UqffReport> {
    let mut report = report?;
    let selected = groups
        .iter()
        .filter_map(|group| group.report_output)
        .collect::<HashSet<_>>();
    if selected.is_empty() {
        return None;
    }
    report.outputs = report
        .outputs
        .into_iter()
        .enumerate()
        .filter_map(|(index, output)| selected.contains(&index).then_some(output))
        .collect();
    Some(report)
}

async fn read_existing_uqff_report(
    model_id: &str,
    revision: &str,
    files: &[String],
    token_source: &TokenSource,
) -> Result<Option<UqffReport>> {
    if !files.iter().any(|file| file == UQFF_REPORT_JSON) {
        return Ok(None);
    }
    let Some(path) = try_get_model_file(model_id, revision, UQFF_REPORT_JSON, token_source).await?
    else {
        return Ok(None);
    };
    let data = tokio::fs::read_to_string(&path).await?;
    serde_json::from_str(&data)
        .map(Some)
        .map_err(|e| anyhow!("{}: {e}", path.display()))
}

fn uqff_group_key(file: &str) -> String {
    let stem = file
        .rsplit_once('/')
        .map_or(file, |(_, name)| name)
        .strip_suffix(".uqff")
        .unwrap_or(file);
    if let Some((prefix, suffix)) = stem.rsplit_once('-') {
        if suffix.chars().all(|ch| ch.is_ascii_digit()) {
            return prefix.to_string();
        }
    }
    stem.to_string()
}

fn uqff_shard_index(file: &str) -> Option<u64> {
    file.rsplit_once('/')
        .map_or(file, |(_, name)| name)
        .strip_suffix(".uqff")
        .and_then(|stem| stem.rsplit_once('-'))
        .and_then(|(_, suffix)| suffix.parse().ok())
}

fn generated_by(tool: &str) -> UqffGeneratedBy {
    UqffGeneratedBy {
        tool: tool.to_string(),
        mistralrs_version: Some(mistralrs_core::MISTRALRS_VERSION.to_string()),
        git_revision: Some(mistralrs_core::MISTRALRS_GIT_REVISION.to_string()),
    }
}

fn print_report_summary(report: &UqffReport, verbose: bool) {
    println!("UQFF report");
    println!("  version: {}", report.uqff_version);
    if let Some(base_model) = &report.base_model {
        println!("  base model: {base_model}");
    }
    if let Some(repo_id) = &report.repo_id {
        println!("  repo id: {repo_id}");
    }
    println!("  generated by: {}", report.generated_by.tool);
    println!();
    println!("Outputs:");
    for output in &report.outputs {
        let counts = output
            .actual_counts
            .iter()
            .map(|(stored, count)| format!("{count} {stored}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  {}: {} layers, {} shard{}, {}, {} fallback{}",
            output.quant,
            output.layers,
            output.shards.len(),
            if output.shards.len() == 1 { "" } else { "s" },
            counts,
            output.fallback_count,
            if output.fallback_count == 1 { "" } else { "s" }
        );
        if verbose {
            for fallback in &output.fallbacks {
                let reason = fallback
                    .reason
                    .as_deref()
                    .map(|reason| format!(": {reason}"))
                    .unwrap_or_default();
                println!(
                    "    {}: {} -> {} {:?}{}",
                    fallback.module, fallback.from, fallback.to, fallback.shape, reason
                );
            }
        }
    }
}
