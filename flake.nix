{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [
          (final: prev: {
            # Reassigning python3 to python39 so that arrow-cpp
            # will be built using it.
            python3 = prev.python39.override {
              packageOverrides = pyfinal: pyprev: {
                # See: https://github.com/NixOS/nixpkgs/pull/172397,
                # https://github.com/pyca/pyopenssl/issues/87
                pyopenssl =
                  pyprev.pyopenssl.overridePythonAttrs
                  (old: {meta.broken = false;});

                # Twisted currently fails tests because of pyopenssl
                # (see linked issues above)
                twisted = pyprev.buildPythonPackage {
                  pname = "twisted";
                  version = "22.4.0";
                  format = "wheel";
                  src = final.fetchurl {
                    url = "https://files.pythonhosted.org/packages/db/99/38622ff95bb740bcc991f548eb46295bba62fcb6e907db1987c4d92edd09/Twisted-22.4.0-py3-none-any.whl";
                    sha256 = "sha256-+fepH5STJHep/DsWnVf1T5bG50oj142c5UA5p/SJKKI=";
                  };
                  propagatedBuildInputs = with pyfinal; [
                    automat
                    constantly
                    hyperlink
                    incremental
                    setuptools
                    typing-extensions
                    zope_interface
                  ];
                };
              };
            };
            python39Packages = prev.python39Packages.override {
              overrides = pyfinal: pyprev: {gitpython = pyprev.GitPython;};
            };
            thrift = prev.thrift.overrideAttrs (old: {
              # Concurrency test fails on Darwin
              # TInterruptTest, TNonblockingSSLServerTest
              # SecurityTest, and SecurityFromBufferTest
              # fail on Linux.
              doCheck = false;
            });
          })
        ];
      };
      inherit (pkgs) poetry2nix lib stdenv fetchurl;
      inherit (pkgs.cudaPackages) cudatoolkit;
      inherit (pkgs.linuxPackages) nvidia_x11;
      python = pkgs.python39;
      pythonEnv = poetry2nix.mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides =
          poetry2nix.overrides.withDefaults
          (pyfinal: pyprev: rec {
            astunparse = pyprev.astunparse.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ [pyfinal.wheel];
            });
            gitpython = pyprev.GitPython;
            orjson = python.pkgs.orjson.override {
              inherit (python) pythonOlder;
              inherit
                (pyprev)
                pytestCheckHook
                buildPythonPackage
                numpy
                psutil
                python-dateutil
                pytz
                xxhash
                ;
            };
          });
      };
    in {
      devShell = pkgs.mkShell {
        buildInputs = [pythonEnv];
        shellHook = ''
          export pythonfaulthandler=1
          export pythonbreakpoint=ipdb.set_trace
          set -o allexport
          source .env
          set +o allexport
        '';
      };
    };
  in
    with utils.lib; eachSystem defaultSystems out;
}
