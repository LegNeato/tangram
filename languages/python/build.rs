fn main() {
	let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
	if target_os.as_str() == "macos" {
		println!("cargo:rustc-cdylib-link-arg=-undefined");
		println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
	}
	if target_os.as_str() == "windows" {
		println!(
			"cargo:rustc-link-search=native={}",
			std::env::var("CARGO_MANIFEST_DIR").unwrap()
		);
	}
}
