{
  extraPackages ? [ ],
  inputs,
  python312,
  system,
}:
python312.withPackages (
  python312Packages: # just formal arguement
  with python312Packages;
  with inputs.self.legacyPackages."${system}";
  [
    numpy
    psutil
    virtualenv
    ultralytics
  ]
  ++ extraPackages
)
