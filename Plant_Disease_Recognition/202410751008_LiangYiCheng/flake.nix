{
  description = "WSdlly02's flake for AI Challenge Hub";

  inputs = {
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      flake-parts,
      self,
      nixpkgs,
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, ... }:
      {
        flake.overlays = {
          pytorch-overlay = final: prev: {
            python312 = prev.python312.override {
              packageOverrides = pyfinal: pyprev: {
                torch = pyprev.torch.override {
                  rocmSupport = false;
                  vulkanSupport = true;
                  # pytorch 有一个vulkan后端，打开它
                  # Rocm 后端编译有问题，暂时禁用，使用二进制版本
                };
              };
            };
          };
        };
        perSystem =
          {
            system,
            ...
          }:
          let
            pkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true; # CUDA不是开源软件
                enableParallelBuilding = true;
                cudaSupport = false; # NVIDIA GPU加速，可以打开
                rocmSupport = true; # AMD GPU加速选项，与cudaSupport不兼容
              };
              overlays = [ self.overlays.pytorch-overlay ];
            };
            inherit (pkgs)
              callPackage
              ;
          in
          {
            devShells = {
              default = callPackage ./devShells-default.nix { inherit inputs; };
            };

            formatter = pkgs.nixfmt-rfc-style;

            legacyPackages = {
              python312Env = callPackage ./python312Env.nix { inherit inputs; };
              python312FHSEnv = callPackage ./python312FHSEnv.nix { inherit inputs; }; # depends on python312Env
            };
          };
        systems = [
          "x86_64-linux"
          #其他cpu架构未经测试
        ];
      }
    );
}
