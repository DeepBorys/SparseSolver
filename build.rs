fn main() {
    println!("cargo:rerun-if-changed=src/sparse_nn.cpp");
    println!("cargo:rerun-if-changed=src/sparse_nn_stub.cpp");
    println!("cargo:rerun-if-changed=src/regret_gather.cpp");
    let source = if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") {
        "src/sparse_nn.cpp"
    } else {
        "src/sparse_nn_stub.cpp"
    };
    cc::Build::new()
        .cpp(true)
        .file(source)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .compile("sparse_nn");
    cc::Build::new().cpp(true).file("src/regret_gather.cpp")
        .flag_if_supported("-std=c++17").flag_if_supported("-O3")
        .compile("regret_gather");
}
