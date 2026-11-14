// Compiles the C++ shim against the vendored TensorRT-RTX headers. Nothing
// from TensorRT-RTX or the CUDA toolkit is linked; the libraries come from
// koharu-runtime's package at run time.
fn main() {
    for path in ["build.rs", "shim", "include"] {
        println!("cargo:rerun-if-changed={path}");
    }
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("shim/koharu_tensorrt.cpp")
        // The stub cuda_runtime_api.h in shim/ must win over a toolkit's.
        .include("shim")
        .include("include")
        .warnings(false)
        .compile("koharu_tensorrt_shim");
}
