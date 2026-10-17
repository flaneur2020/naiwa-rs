//! Compiles this crate's CUDA kernels (`src/doorbell/doorbell.cu`) into `libnaiwadoorbell.a`.
//!
//! The kernels are v2's own sync primitives, so they live here rather than in the mistral.rs fork
//! (whose doorbell had no consumer there). `cudaforge` is the same helper the fork's build uses, so
//! the arch and toolkit detection are identical.
fn main() {
    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;

        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=src/doorbell");
        println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
        let build_dir = out_dir.join("doorbell");
        std::fs::create_dir_all(&build_dir).expect("create the doorbell build dir");

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("src/doorbell/*.cu")
            .watch(["src/doorbell"])
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
            .arg("-fPIC");
        if let Ok(flags) = std::env::var("CUDA_NVCC_FLAGS") {
            builder = builder.arg("--compiler-options").arg(&flags);
        }

        let target = std::env::var("TARGET").unwrap_or_default();
        let archive = if target.contains("msvc") {
            out_dir.join("naiwadoorbell.lib")
        } else {
            out_dir.join("libnaiwadoorbell.a")
        };
        builder.build_lib(archive).expect("build the doorbell kernels");
        println!("cargo:rustc-link-search={}", out_dir.display());
        println!("cargo:rustc-link-lib=naiwadoorbell");

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
