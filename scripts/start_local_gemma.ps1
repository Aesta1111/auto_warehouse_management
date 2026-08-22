[CmdletBinding()]
param(
    [switch]$Prepare,
    [int]$Port = 8080,
    [int]$ContextSize = 4096,
    [int]$GpuLayers = 24
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = Join-Path $repositoryRoot ".local_runtime"
$runtimeDirectory = Join-Path $runtimeRoot "llama-b10569"
$modelDirectory = Join-Path $runtimeRoot "models\gemma-4-E4B-it-qat-q4_0-gguf"
$modelPath = Join-Path $modelDirectory "gemma-4-E4B_q4_0-it.gguf"
$serverPath = Join-Path $runtimeDirectory "llama-server.exe"
$runtimeArchive = Join-Path $runtimeRoot "llama-b10569-bin-win-cuda-12.4-x64.zip"
$cudaArchive = Join-Path $runtimeRoot "cudart-llama-bin-win-cuda-12.4-x64.zip"

$modelRepository = "google/gemma-4-E4B-it-qat-q4_0-gguf"
$modelRevision = "4b4a2c1d584be7264f87aac328a1bc739ce81b6c"
$modelFilename = "gemma-4-E4B_q4_0-it.gguf"
$modelId = "${modelRepository}:Q4_0"
$modelSha256 = "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee"
$runtimeArchiveSha256 = "f03eb16658f04ae259b759016746f1747d1e3c50c9689e407091aefc34f43635"
$cudaArchiveSha256 = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
$runtimeRelease = "b10569"
$runtimeDownloadUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$runtimeRelease/llama-b10569-bin-win-cuda-12.4-x64.zip"
$cudaDownloadUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$runtimeRelease/cudart-llama-bin-win-cuda-12.4-x64.zip"

function Assert-FileSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "$Label SHA-256 mismatch. Expected $Expected, got $actual."
    }
}

if ($Prepare) {
    New-Item -ItemType Directory -Force -Path $runtimeRoot, $modelDirectory | Out-Null

    if (-not (Test-Path -LiteralPath $runtimeArchive)) {
        Write-Host "Downloading pinned llama.cpp $runtimeRelease CUDA 12.4 runtime..."
        curl.exe --location --fail --retry 3 --output $runtimeArchive $runtimeDownloadUrl
        if ($LASTEXITCODE -ne 0) {
            throw "llama.cpp runtime download failed with exit code $LASTEXITCODE."
        }
    }
    Assert-FileSha256 -Path $runtimeArchive -Expected $runtimeArchiveSha256 -Label "llama.cpp archive"
    if (-not (Test-Path -LiteralPath $cudaArchive)) {
        Write-Host "Downloading pinned llama.cpp CUDA 12.4 DLL bundle..."
        curl.exe --location --fail --retry 3 --output $cudaArchive $cudaDownloadUrl
        if ($LASTEXITCODE -ne 0) {
            throw "llama.cpp CUDA DLL download failed with exit code $LASTEXITCODE."
        }
    }
    Assert-FileSha256 -Path $cudaArchive -Expected $cudaArchiveSha256 -Label "llama.cpp CUDA DLL archive"
    if (-not (Test-Path -LiteralPath $serverPath)) {
        Expand-Archive -LiteralPath $runtimeArchive -DestinationPath $runtimeDirectory -Force
    }
    Expand-Archive -LiteralPath $cudaArchive -DestinationPath $runtimeDirectory -Force

    if (-not (Test-Path -LiteralPath $modelPath)) {
        $pythonPath = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $pythonPath)) {
            throw "The repository virtual environment is missing: $pythonPath"
        }
        $env:PYTHONIOENCODING = "utf-8"
        Write-Host "Downloading pinned Gemma E4B Q4_0 GGUF model..."
        & $pythonPath -m huggingface_hub.cli.hf download $modelRepository $modelFilename `
            --revision $modelRevision --local-dir $modelDirectory --max-workers 4 --format json
        if ($LASTEXITCODE -ne 0) {
            throw "Gemma model download failed with exit code $LASTEXITCODE."
        }
    }
}

if (-not (Test-Path -LiteralPath $serverPath)) {
    throw "llama-server.exe is missing. Run this script with -Prepare first."
}
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "The pinned Gemma model is missing. Run this script with -Prepare first."
}
$requiredCudaDlls = @(
    "ggml-cuda.dll",
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll"
)
$missingCudaDlls = @(
    $requiredCudaDlls | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $runtimeDirectory $_))
    }
)
if ($missingCudaDlls.Count -gt 0) {
    throw "The CUDA 12.4 llama.cpp DLL bundle is incomplete ($($missingCudaDlls -join ', ')). Run this script with -Prepare first."
}

Assert-FileSha256 -Path $modelPath -Expected $modelSha256 -Label "Gemma model"

$serverArguments = @(
    "--model", $modelPath,
    "--alias", $modelId,
    "--host", "127.0.0.1",
    "--port", $Port.ToString(),
    "--ctx-size", $ContextSize.ToString(),
    "--gpu-layers", $GpuLayers.ToString(),
    "--parallel", "1",
    "--flash-attn", "auto",
    "--reasoning", "off",
    "--reasoning-budget", "0",
    "--temp", "0",
    "--top-p", "1"
)

Write-Host "Starting local Gemma actioner server: $modelId"
Write-Host "llama.cpp $runtimeRelease | context=$ContextSize | gpu_layers=$GpuLayers | endpoint=http://127.0.0.1:$Port"
Write-Host "Press Ctrl+C to stop the server."
& $serverPath @serverArguments
exit $LASTEXITCODE
