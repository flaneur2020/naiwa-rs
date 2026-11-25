//! Compiles `naiwa-quant`'s CUDA kernels (`kernels/*/*.cu`) into `libnaiwaquant.a`.

#[cfg(feature = "cuda")]
fn hash_headers(dir: &std::path::Path) -> std::io::Result<u64> {
    fn update(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    fn visit(path: &std::path::Path, hash: &mut u64) -> std::io::Result<()> {
        if path.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(path)?
                .map(|e| e.map(|e| e.path()))
                .collect::<std::io::Result<_>>()?;
            entries.sort();
            for entry in entries {
                visit(&entry, hash)?;
            }
            return Ok(());
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if matches!(ext, "cuh" | "h" | "hpp") {
            println!("cargo:rerun-if-changed={}", path.display());
            update(hash, path.to_string_lossy().as_bytes());
            update(hash, &std::fs::read(path)?);
        }
        Ok(())
    }
    let mut hash = 0xcbf29ce484222325;
    visit(dir, &mut hash)?;
    Ok(hash)
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        use std::path::{Path, PathBuf};

        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=kernels");
        println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
        println!("cargo:rerun-if-env-changed=NAIWA_CUDA_EXTRA_GENCODES");
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
        let build_dir = out_dir.join("quant_kernels");
        std::fs::create_dir_all(&build_dir).expect("create quant kernel build dir");

        let header_hash = hash_headers(Path::new("kernels")).expect("hash CUDA headers");
        let header_hash_arg = format!("-DNAIWA_QUANT_HEADER_HASH=0x{header_hash:016x}");

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("kernels/*/*.cu")
            .watch(["kernels"])
            .out_dir(&build_dir)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--compiler-options")
            .arg("-fPIC")
            .arg(&header_hash_arg);

        if let Ok(extra) = std::env::var("NAIWA_CUDA_EXTRA_GENCODES") {
            for flag in extra.split_whitespace() {
                builder = builder.arg(flag);
            }
        }
        if let Ok(flags) = std::env::var("CUDA_NVCC_FLAGS") {
            builder = builder.arg("--compiler-options").arg(&flags);
        }

        let target = std::env::var("TARGET").unwrap_or_default();
        let archive = if target.contains("msvc") {
            out_dir.join("naiwaquant.lib")
        } else {
            out_dir.join("libnaiwaquant.a")
        };
        builder
            .build_lib(archive)
            .expect("build naiwa-quant CUDA kernels");

        println!("cargo:rustc-link-search={}", out_dir.display());
        println!("cargo:rustc-link-lib=naiwaquant");
        println!("cargo:rustc-link-lib=dylib=cudart");

        if target.contains("msvc") {
            // nothing to link
        } else if target.contains("apple")
            || target.contains("freebsd")
            || target.contains("openbsd")
        {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target.contains("android") {
            println!("cargo:rustc-link-lib=dylib=c++_shared");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}
