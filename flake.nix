{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, systems, ... } @ inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      packages = forEachSystem (system: {
        devenv-up = self.devShells.${system}.default.config.procfileScript;
      });

      devShells = forEachSystem
        (system:
          let
            pkgs = import nixpkgs {
              system = system;
              config.allowUnfree = true;
            };


            mpkgs = import inputs.nixpkgs_master {
              system = system;
              config.allowUnfree = true;
            };
          in
          {
            default = devenv.lib.mkShell {
              inherit inputs pkgs;
              modules = [
                {
                  stdenv = pkgs.clangStdenv;
                  env.NIX_LD = nixpkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
                  env.NIX_LD_LIBRARY_PATH = nixpkgs.lib.makeLibraryPath (with pkgs; [
                  # Add needed packages here
                  cudaPackages.cudatoolkit
                  linuxPackages.nvidia_x11
                  pkgs.libz # for numpy
                  pkgs.stdenv.cc.cc
                  pkgs.libGL
                  ]);
                  # https://devenv.sh/reference/options/
                  packages = with pkgs; [
                    poetry
                    python310
                    # python310Packages.cupy
                  ];
                  enterShell = ''
                    export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
                    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                    poetry install -vv --with dev
                  '';
                }
              ];
            };
          });
    };
}
