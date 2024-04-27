{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryPackages;

        python = pkgs.python310;
        projectDir = self;
      in
      {
        packages = {
          tigramite = mkPoetryPackages {
            inherit python projectDir;
          };
          default = self.packages.${system}.tigramite;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.tigramite ];
          packages = [
            python
            pkgs.poetry
          ];
        };
      });
}
