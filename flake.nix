{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/a983cc62cc2345443620597697015c1c9c4e5b06";
    utils.url = "github:numtide/flake-utils/93a2b84fc4b70d9e089d029deacc3583435c2ed6";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
      inherit (pkgs) poetry2nix lib stdenv;
      mujoco = fetchTarball {
        url = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz";
        sha256 = "sha256:1lvppcdfca460sqnb0ryrach6lv1g9dwcjfim0dl4vmxg2ryaq7p";
      };
      overrides = pyfinal: pyprev: rec {
        bsuite = pyprev.bsuite.overridePythonAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pyprev.scipy];
        });
        dollar-lambda = pyprev.dollar-lambda.overridePythonAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pyprev.poetry];
        });
        vega-charts = pyprev.vega-charts.overridePythonAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pyprev.poetry];
        });
      };
      poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
        python = pkgs.python39;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };
    in {
      devShell = pkgs.mkShell {
        LD_LIBRARY_PATH = lib.optional stdenv.isLinux "$LD_LIBRARY_PATH:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.mesa.osmesa}/lib:${pkgs.libGL}/lib:${pkgs.gcc-unwrapped.lib}/lib";
        buildInputs = with pkgs; [
          alejandra
          poetry
          poetryEnv
        ];
        PYTHONBREAKPOINT = "ipdb.set_trace";
        shellHook = ''
          set -o allexport
          source .env
          set +o allexport
        '';
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
