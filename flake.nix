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
            dollar-lambda = pyprev.dollar-lambda.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ [pyfinal.poetry];
            });
            vega-charts = pyprev.vega-charts.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ [pyfinal.poetry];
            });
            run-logger = pyprev.run-logger.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ [pyfinal.poetry];
            });
            sweep-logger = pyprev.sweep-logger.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ [pyfinal.poetry];
            });
            pandas-stubs = pyprev.pandas-stubs.overridePythonAttrs (old:{
              # Prevent collisions with actual pandas installation.
              postInstall = ''
                rm -rf $out/${python.sitePackages}/pandas
              '';
            });
            # Poetry2Nix tries to apply an override to lowercased
            # gitpython (which does not exist). We first assign
            # it to the GitPython that does exist to prevent a
            # missing attribute error.
            gitpython = pyprev.GitPython;
            # Then we replace the GitPython key with overridden version
            # so that we get the fixes that poetry2nix applies.
            GitPython = pyfinal.gitpython;
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
