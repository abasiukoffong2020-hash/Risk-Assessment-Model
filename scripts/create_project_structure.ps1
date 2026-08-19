# Recreate the standard project folder structure for this workspace.
# Run this from the workspace root:
#   .\scripts\create_project_structure.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $root

$folders = @(
    "data\\raw",
    "data\\processed",
    "data\\external",
    "notebooks",
    "src\\data",
    "src\\features",
    "src\\models",
    "src\\utils",
    "app",
    "app\\configs",
    "scripts",
    "tests",
    ".github\\workflows",
    "docker",
    "great_expectations",
    "mlruns",
    "artifacts"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}
Write-Host "Project structure created or verified in $root" -ForegroundColor Green
