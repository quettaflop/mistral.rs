use anyhow::Result;

#[cfg(feature = "metal")]
include!("src/metal/kernels/source_set.rs");

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn cuda_header_hash(dir: &str) -> Result<u64> {
    use std::path::Path;

    fn update(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }

    fn visit(path: &Path, hash: &mut u64) -> Result<()> {
        if path.is_dir() {
            let mut entries = std::fs::read_dir(path)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            entries.sort();
            for entry in entries {
                visit(&entry, hash)?;
            }
            return Ok(());
        }

        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            return Ok(());
        };
        if ext != "cuh" && ext != "h" {
            return Ok(());
        }

        println!("cargo:rerun-if-changed={}", path.display());
        update(hash, path.to_string_lossy().as_bytes());
        update(hash, &std::fs::read(path)?);
        Ok(())
    }

    let mut hash = 0xcbf29ce484222325;
    visit(Path::new(dir), &mut hash)?;
    Ok(hash)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::path::PathBuf;

    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
    println!("cargo:rerun-if-changed=src/cuda/pagedattention.cuh");
    println!("cargo:rerun-if-changed=src/cuda/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/concat_and_cache_mla_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/gather_mla_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/gather_kv_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer_decode.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer_decode_quant.cu");
    println!("cargo:rerun-if-changed=src/cuda/reshape_and_cache_quant_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/reshape_and_cache_fp8.cu");
    println!("cargo:rerun-if-changed=src/cuda/xqa/launch.cu");
    println!("cargo:rerun-if-changed=src/cuda/xqa/mha.cu");
    println!("cargo:rerun-if-changed=src/cuda/xqa/mha.h");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer_mla_decode.cu");
    println!("cargo:rerun-if-changed=src/cuda/update_kvscales.cu");
    println!("cargo:rerun-if-changed=src/cuda/flash_attn_sinks.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/cp_async.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/exception.h");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/fastdiv.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/fp16.h");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/frag_layout_swizzle.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/layout.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/math.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/mma.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/page.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/permuted_smem.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/pos_enc.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/utils.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/vec_dtypes.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/cascade.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/decode.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/decode_quant.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/default_decode_params.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/mask.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/state.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/variant_helper.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/variants.cuh");

    let header_hash_arg = format!(
        "-DMISTRALRS_CUDA_HEADER_HASH={:016x}",
        cuda_header_hash("src/cuda")?
    );

    let mut builder = cudaforge::KernelBuilder::new()
        .source_glob("src/cuda/*.cu")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose")
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg(&header_hash_arg);

    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    // Enable FP8 if compute capability >= 8.0 (Ampere and newer)
    let using_fp8 = if compute_cap >= 80 {
        builder = builder.arg("-DENABLE_FP8");
        true
    } else {
        false
    };

    // https://github.com/EricLBuehler/mistral.rs/issues/286
    if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
        builder = builder.arg("--compiler-options");
        builder = builder.arg(cuda_nvcc_flags_env);
    }
    println!("cargo:info={builder:?}");

    let target = std::env::var("TARGET").unwrap();
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // https://github.com/EricLBuehler/mistral.rs/issues/588
    let out_file = if target.contains("msvc") {
        // Windows case
        build_dir.join("mistralrspagedattention.lib")
    } else {
        build_dir.join("libmistralrspagedattention.a")
    };
    builder
        .build_lib(out_file)
        .expect("Build paged attention lib failed!");

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=mistralrspagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    build_xqa_fp8(&build_dir, compute_cap)?;

    if using_fp8 {
        println!("cargo:rustc-cfg=has_fp8");
    }
    Ok(())
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn build_xqa_fp8(build_dir: &std::path::Path, compute_cap: usize) -> Result<()> {
    use std::process::Command;

    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let xqa_dir = manifest.join("src/cuda/xqa");
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let arch = format!("arch=compute_{compute_cap},code=sm_{compute_cap}");
    let mut objects = Vec::new();
    for src in ["mha.cu", "launch.cu"] {
        println!("cargo:rerun-if-changed={}", xqa_dir.join(src).display());
        let obj = build_dir.join(format!("xqa_{}.o", src.replace('.', "_")));
        let status = Command::new(&nvcc)
            .args([
                "-c",
                "-std=c++17",
                "-O3",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--compiler-options",
                "-fPIC",
                "-gencode",
                &arch,
                "-DNDEBUG=1",
                "-DBEAM_WIDTH=1",
                "-DUSE_INPUT_KV=0",
                "-DUSE_CUSTOM_BARRIER=1",
                "-DINPUT_FP16=1",
                "-DDTYPE=__half",
                "-DCACHE_ELEM_ENUM=2",
                "-DTOKENS_PER_PAGE=32",
                "-DHEAD_ELEMS=128",
                "-DHEAD_GRP_SIZE=4",
                "-DSLIDING_WINDOW=0",
                "-DLOW_PREC_OUTPUT=0",
                "-DSPEC_DEC=0",
                "-DMLA_WRAPPER=0",
                "-DUSE_SM90_MHA=0",
            ])
            .arg(format!("-I{}", xqa_dir.display()))
            .arg(xqa_dir.join(src))
            .arg("-o")
            .arg(&obj)
            .status()?;
        if !status.success() {
            anyhow::bail!("nvcc failed compiling XQA {src}");
        }
        objects.push(obj);
    }
    let lib = build_dir.join("libxqa_fp8.a");
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&lib);
    for obj in &objects {
        ar.arg(obj);
    }
    let status = ar.status()?;
    if !status.success() {
        anyhow::bail!("ar failed creating libxqa_fp8.a");
    }
    println!("cargo:rustc-link-lib=static=xqa_fp8");
    println!("cargo:rustc-link-lib=stdc++");
    Ok(())
}

#[cfg(feature = "metal")]
fn main() -> Result<(), String> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");

    mistralrs_metal_compile::compile_metallibs(&PAGED_ATTENTION_METAL_SOURCE_SET)
}

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
fn main() -> Result<()> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");
    Ok(())
}
