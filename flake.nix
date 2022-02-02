{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
    fenix = {
      url = "github:nix-community/fenix";
    };
    windows_sdk = {
      url = "github:tangramdotdev/windows_sdk";
    };
  };
  outputs =
    inputs: inputs.flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [
          (self: super: {
            abuild = super.abuild.overrideAttrs (old: {
              patches = [
                (pkgs.fetchpatch {
                  url = "https://gitlab.alpinelinux.org/alpine/abuild/-/merge_requests/130.patch";
                  sha256 = "sha256-9+MpH9HTNDzfRd7vwTD2yU7guIYScAuGMpsqSdvZ9p4=";
                })
              ];
              patchPhase = null;
              postPatch = old.patchPhase;
              propagatedBuildInputs = with self; [
                apk-tools
                fakeroot
                libressl
                pax-utils
              ];
            });
            rpm = super.rpm.overrideAttrs (_: {
              patches = [
                (pkgs.fetchpatch {
                  url = "https://github.com/rpm-software-management/rpm/pull/1775.patch";
                  sha256 = "sha256-WYlxPGcPB5lGQmkyJ/IpGoqVfAKtMxKzlr5flTqn638=";
                })
              ];
            });
            rust-cbindgen = super.rust-cbindgen.overrideAttrs (_: {
              doCheck = false;
            });
            zig = super.zig.overrideAttrs (_: {
              src = self.fetchFromGitHub {
                owner = "ziglang";
                repo = "zig";
                rev = "a76910b691179390b122b9c42a5575c00d40583a";
                hash = "sha256-f8CT7BBc1KOLJJs/4OoIqK6/m1lkfLcN3w/bn5enqBQ=";
              };
              patches = [
                (self.fetchpatch {
                  url = "https://github.com/ziglang/zig/pull/9771.patch";
                  sha256 = "sha256-AaMNNBET/x0f3a9oxpgBZXnUdKH4bydKMLJfXLBmvZo=";
                })
              ];
            });
          })
        ];
      };
      rust =
        let
          toolchain = {
            channel = "nightly";
            date = "2022-01-01";
            sha256 = "sha256-XUPcV51ws0fwqPrqglxfQ2j1KdxnYkDXCoMaZCbgtuI=";
          };
        in
        with inputs.fenix.packages.${system}; combine (with toolchainOf toolchain; [
          cargo
          clippy-preview
          rust-src
          rust-std
          rustc
          rustfmt-preview
          (targets.aarch64-unknown-linux-gnu.toolchainOf toolchain).rust-std
          (targets.aarch64-unknown-linux-musl.toolchainOf toolchain).rust-std
          (targets.aarch64-apple-darwin.toolchainOf toolchain).rust-std
          (targets.wasm32-unknown-unknown.toolchainOf toolchain).rust-std
          (targets.x86_64-unknown-linux-gnu.toolchainOf toolchain).rust-std
          (targets.x86_64-unknown-linux-musl.toolchainOf toolchain).rust-std
          (targets.x86_64-apple-darwin.toolchainOf toolchain).rust-std
          (targets.x86_64-pc-windows-gnu.toolchainOf toolchain).rust-std
          (targets.x86_64-pc-windows-msvc.toolchainOf toolchain).rust-std
        ]);
      windows_sdk_packages = pkgs.runCommand "packages"
        {
          nativeBuildInputs = with pkgs; [
            (inputs.windows_sdk.defaultPackage.${system})
          ];
          outputHashMode = "recursive";
          outputHash = "sha256-jrOGFuXPNjw8NqZQTz7Xl99LGfFBkwRcy+xuY7D6TYk=";
        }
        ''
          windows_sdk \
            choose-packages \
            --manifest ${./scripts/windows_sdk.vsman} \
            --package Microsoft.VisualStudio.VC.Llvm.Clang \
            --package Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
            --package Microsoft.VisualStudio.Component.Windows10SDK.19041 \
            --output $out
        '';
      windows_sdk_cache = pkgs.runCommand "packages"
        {
          nativeBuildInputs = with pkgs; [
            (inputs.windows_sdk.defaultPackage.${system})
          ];
          outputHashMode = "recursive";
          outputHash = "sha256-NcIMWa6qG8IZWRPIadvrWK167A9wj8TPmXkAplM1X4c=";
        }
        ''
          windows_sdk \
            download-packages \
            --packages ${windows_sdk_packages} \
            --cache $out
        '';
      windows_sdk = pkgs.runCommand "windows_sdk"
        {
          nativeBuildInputs = with pkgs; [
            (inputs.windows_sdk.defaultPackage.${system})
          ];
          outputHashMode = "recursive";
          outputHash = "sha256-80J04A+sd2ivL5/0OlKGvBF9C6wrx+YMiBqg+zRSGn8=";
        }
        ''
          windows_sdk \
            extract-packages \
            --packages ${windows_sdk_packages} \
            --cache ${windows_sdk_cache} \
            --output $out
        '';
      macos_sdk = builtins.fetchTarball {
        url = "https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX11.3.sdk.tar.xz";
        sha256 = "sha256-BoFWhRSHaD0j3dzDOFtGJ6DiRrdzMJhkjxztxCluFKo=";
      };
    in
    rec {
      defaultApp = inputs.flake-utils.lib.mkApp {
        drv = defaultPackage;
      };
      defaultPackage = (pkgs.makeRustPlatform {
        rustc = rust;
        cargo = rust;
      }).buildRustPackage ({
        name = "tangram";
        src = ./.;
        doCheck = false;
        cargoLock = { lockFile = ./Cargo.lock; };
        cargoBuildFlags = "--package tangram_cli";
      });
      apps.www = inputs.flake-utils.lib.mkApp {
        drv = packages.www;
      };
      packages.www = (pkgs.makeRustPlatform {
        rustc = rust;
        cargo = rust;
      }).buildRustPackage ({
        name = "tangram_www";
        src = ./.;
        doCheck = false;
        cargoLock = { lockFile = ./Cargo.lock; };
        cargoBuildFlags = "--package tangram_www";
      });
      devShell = pkgs.mkShell {
        packages = with pkgs; [
          (inputs.windows_sdk.defaultPackage.${system})
          abuild
          cachix
          cargo-insta
          cargo-outdated
          clang_12
          createrepo_c
          doxygen
          dpkg
          elixir
          gnupg
          go
          lld_12
          llvm_12
          mold
          nodejs
          (php.withExtensions ({ all, ... }: with all; [
            curl
            dom
            ffi
            fileinfo
            filter
            iconv
            mbstring
            simplexml
            tokenizer
          ]))
          php.packages.composer
          (python3.withPackages (p: with p; [
            pandas
            pyarrow
          ]))
          rpm
          ruby
          rust
          rust-cbindgen
          sqlite
          time
          wasm-bindgen-cli
          zig
        ];

        CARGO_UNSTABLE_MULTITARGET = "true";
        CFLAGS = "-fno-sanitize=undefined";

        # aarch64-linux-gnu
        CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER = pkgs.writeShellScriptBin "linker" ''
          for arg do
            shift
            [ "$arg" = "-lgcc_s" ] && continue
            set -- "$@" "$arg"
          done
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target aarch64-linux-gnu.2.28 -lunwind $@
        '' + /bin/linker;
        CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS = "-C target-feature=-outline-atomics";
        CC_aarch64_unknown_linux_gnu = pkgs.writeShellScriptBin "cc" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target aarch64-linux-gnu.2.28 $@
        '' + /bin/cc;

        # aarch64-linux-musl
        CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER = pkgs.writeShellScriptBin "linker" ''
          for arg do
            shift
            [ "$arg" = "-lgcc_s" ] && continue
            set -- "$@" "$arg"
          done
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target aarch64-linux-musl -lunwind -dynamic $@
        '' + /bin/linker;
        CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUSTFLAGS = "-C target-feature=-crt-static";
        CC_aarch64_unknown_linux_musl = pkgs.writeShellScriptBin "cc" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target aarch64-linux-musl $@
        '' + /bin/cc;

        # aarch64-macos
        CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER = pkgs.writeShellScriptBin "linker" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target aarch64-macos.11 -L${inputs.nixpkgs.legacyPackages.aarch64-darwin.libiconv}/lib -F${macos_sdk}/System/Library/Frameworks -Wl,-undefined=dynamic_lookup $@
        '' + /bin/linker;
        CC_aarch64_apple_darwin = pkgs.writeShellScriptBin "cc" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target aarch64-macos.11 $@
        '' + /bin/cc;

        # wasm32
        CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER = "lld";

        # x86_64-linux-gnu
        CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = pkgs.writeShellScriptBin "linker" ''
          for arg do
            shift
            [ "$arg" = "-lgcc_s" ] && continue
            set -- "$@" "$arg"
          done
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target x86_64-linux-gnu.2.28 -lunwind --ld-path=$(which mold) $@
        '' + /bin/linker;
        CC_x86_64_unknown_linux_gnu = pkgs.writeShellScriptBin "cc" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target x86_64-linux-gnu.2.28 $@
        '' + /bin/cc;

        # x86_64-linux-musl
        CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER = pkgs.writeShellScriptBin "linker" ''
          for arg do
            shift
            [ "$arg" = "-lgcc_s" ] && continue
            set -- "$@" "$arg"
          done
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target x86_64-linux-musl -lunwind -dynamic $@
        '' + /bin/linker;
        CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS = "-C target-feature=-crt-static";
        CC_x86_64_unknown_linux_musl = pkgs.writeShellScriptBin "cc" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target x86_64-linux-musl $@
        '' + /bin/cc;

        # x86_64-macos
        CARGO_TARGET_X86_64_APPLE_DARWIN_LINKER = pkgs.writeShellScriptBin "linker" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target x86_64-macos.11 -L${inputs.nixpkgs.legacyPackages.x86_64-darwin.libiconv}/lib -F${macos_sdk}/System/Library/Frameworks -Wl,-undefined=dynamic_lookup $@
        '' + /bin/linker;
        CC_x86_64_apple_darwin = pkgs.writeShellScriptBin "cc" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target x86_64-macos.11 $@
        '' + /bin/cc;

        # x86_64-windows-gnu
        CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = pkgs.writeShellScriptBin "linker" ''
          for arg do
            shift
            [ "$arg" = "-lgcc" ] && continue
            [ "$arg" = "-lgcc_eh" ] && continue
            [ "$arg" = "-l:libpthread.a" ] && continue
            set -- "$@" "$arg"
          done
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target x86_64-windows-gnu -lstdc++ $@
        '' + /bin/linker;
        CC_x86_64_pc_windows_gnu = pkgs.writeShellScriptBin "cc" ''
          ZIG_GLOBAL_CACHE_DIR=$(mktemp -d) zig cc -target x86_64-windows-gnu $@
        '' + /bin/cc;

        # x86_64-windows-msvc
        AR_x86_64_pc_windows_msvc = "llvm-lib";
        CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = pkgs.writeShellScriptBin "linker" ''
          lld-link \
            /libpath:"${windows_sdk}/VC/Tools/Llvm/lib" \
            /libpath:"${windows_sdk}/VC/Tools/MSVC/14.29.30133/lib/x64" \
            /libpath:"${windows_sdk}/Program Files/Windows Kits/10/Lib/10.0.19041.0/ucrt/x64" \
            /libpath:"${windows_sdk}/Program Files/Windows Kits/10/Lib/10.0.19041.0/um/x64" \
            $@
        '' + /bin/linker;
        CC_x86_64_pc_windows_msvc = pkgs.writeShellScriptBin "cc" ''
          clang-cl \
            /I "${windows_sdk}/VC/Tools/Llvm/lib/clang/12.0.0/include" \
            /I "${windows_sdk}/VC/Tools/MSVC/14.29.30133/include" \
            /I "${windows_sdk}/Program Files/Windows Kits/10/Include/10.0.19041.0/ucrt" \
            /I "${windows_sdk}/Program Files/Windows Kits/10/Include/10.0.19041.0/um" \
            /I "${windows_sdk}/Program Files/Windows Kits/10/Include/10.0.19041.0/shared" \
            $@
        '' + /bin/cc;
      };
    }
    );
}
