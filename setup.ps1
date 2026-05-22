# setup.ps1 - GPU Environment Initialization and Dependency Installer
# Installs PyTorch (CUDA), faster-whisper, pyannote.audio,
# llama-cpp-python (CUDA wheel), and other dependencies.
$ErrorActionPreference = "Stop"
$VenvDir = ".venv"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Call Analyzer - GPU Environment Setup     " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow

$pyPath = ""

# 1. Check standard installation directories first
$searchPaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "C:\Program Files\Python312\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python310\python.exe"
)
foreach ($path in $searchPaths) {
    if (Test-Path $path) {
        $pyPath = $path
        break
    }
}

# 2. Fallback to PATH commands but exclude Microsoft Store stubs
if ($pyPath -eq "") {
    $cmd = Get-Command "python" -ErrorAction SilentlyContinue
    if ($cmd -and ($cmd.Source -notlike "*Microsoft\WindowsApps*")) {
        $pyPath = $cmd.Source
    }
}
if ($pyPath -eq "") {
    $cmd = Get-Command "python3" -ErrorAction SilentlyContinue
    if ($cmd -and ($cmd.Source -notlike "*Microsoft\WindowsApps*")) {
        $pyPath = $cmd.Source
    }
}

if ($pyPath -eq "") {
    Write-Host "ERROR: Python not found. Please install Python 3.10+ and add it to PATH." -ForegroundColor Red
    exit 1
}

$pythonVersion = & $pyPath --version 2>&1
Write-Host "  OK: $pythonVersion (Path: $pyPath)" -ForegroundColor Green

# Determine if nightly PyTorch is needed (Python 3.14+ not supported by stable)
$pyMinor = [int](& $pyPath -c "import sys; print(sys.version_info.minor)")
$USE_NIGHTLY = ($pyMinor -ge 14)

# 2. Check CUDA
Write-Host "[2/6] Checking CUDA..." -ForegroundColor Yellow
$nvidiaSmi = nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK: NVIDIA GPU found." -ForegroundColor Green
    # Determine CUDA version from nvidia-smi
    $cudaVersion = (nvidia-smi | Select-String "CUDA Version" | ForEach-Object { ($_ -split "CUDA Version: ")[1].Split(" ")[0] })
    Write-Host "  CUDA Version: $cudaVersion" -ForegroundColor Green
    # Select PyTorch wheel index based on CUDA version
    if ([double]$cudaVersion -ge 12.0) {
        if ($USE_NIGHTLY) {
            $TORCH_INDEX = "https://download.pytorch.org/whl/nightly/cu128"
        } else {
            $TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
        }
        $LLAMA_WHL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
    } elseif ([double]$cudaVersion -ge 11.8) {
        $TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
        $LLAMA_WHL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu118"
    } else {
        Write-Host "  WARNING: CUDA $cudaVersion < 11.8. Recommended to update GPU driver." -ForegroundColor Yellow
        $TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
        $LLAMA_WHL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu118"
    }
} else {
    Write-Host "  WARNING: nvidia-smi not found. Installing CPU versions." -ForegroundColor Yellow
    $TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
    $LLAMA_WHL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
}

# 3. Create virtual environment
Write-Host "[3/6] Setting up virtual environment (.venv)..." -ForegroundColor Yellow
if (Test-Path $VenvDir) {
    Write-Host "  Found existing .venv - skipping creation." -ForegroundColor DarkGray
} else {
    Write-Host "  Creating virtual environment..." -ForegroundColor DarkGray
    & $pyPath -m venv $VenvDir
    Write-Host "  OK: .venv created." -ForegroundColor Green
}

# 4. Upgrade pip
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
& "$VenvDir\Scripts\pip.exe" install --upgrade pip --quiet
Write-Host "  OK" -ForegroundColor Green

# 5. Install PyTorch with CUDA (REQUIRED before other packages)
Write-Host "[5/6] Installing PyTorch + torchaudio (CUDA)..." -ForegroundColor Yellow

$torchCudaOk = $false
if (Test-Path "$VenvDir\Scripts\python.exe") {
    $checkTorch = & "$VenvDir\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($LASTEXITCODE -eq 0 -and $checkTorch -eq "True") {
        $torchCudaOk = $true
    }
}

if ($torchCudaOk) {
    Write-Host "  OK: PyTorch with CUDA is already installed." -ForegroundColor Green
} else {
    Write-Host "  Index URL: $TORCH_INDEX" -ForegroundColor DarkGray
    Write-Host "  (this may take several minutes - ~2.5 GB)" -ForegroundColor DarkGray
    if ($USE_NIGHTLY) {
        Write-Host "  (using nightly build for Python 3.14+)" -ForegroundColor DarkGray
        & "$VenvDir\Scripts\pip.exe" install --pre torch torchaudio `
            --index-url $TORCH_INDEX --no-deps
        # install dependencies separately
        & "$VenvDir\Scripts\pip.exe" install filelock sympy networkx jinja2 fsspec
    } else {
        & "$VenvDir\Scripts\pip.exe" install torch torchaudio `
            --extra-index-url $TORCH_INDEX
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install PyTorch." -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK: PyTorch installed." -ForegroundColor Green
}

# 5b. Install llama-cpp-python with CUDA support
Write-Host "[5b] Installing llama-cpp-python (CUDA wheel)..." -ForegroundColor Yellow

$llamaCudaOk = $false
if (Test-Path "$VenvDir\Scripts\python.exe") {
    $checkLlama = & "$VenvDir\Scripts\python.exe" -c "import os, sys, glob; [os.environ.update({'PATH': p + os.pathsep + os.environ['PATH']}) for p in glob.glob(os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', '*', 'bin'))]; import llama_cpp; print('OK')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $checkLlama -eq "OK") {
        $llamaCudaOk = $true
    }
}

if ($llamaCudaOk) {
    Write-Host "  OK: llama-cpp-python is already installed." -ForegroundColor Green
} else {
    Write-Host "  Index URL: $LLAMA_WHL_INDEX" -ForegroundColor DarkGray
    & "$VenvDir\Scripts\pip.exe" install llama-cpp-python `
        --extra-index-url $LLAMA_WHL_INDEX

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install llama-cpp-python." -ForegroundColor Red
        Write-Host "Please ensure you have internet access and CUDA is installed." -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK: llama-cpp-python (CUDA) installed." -ForegroundColor Green
}

# 6. Install other dependencies from requirements.txt
Write-Host "[6/6] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
Write-Host "  (faster-whisper, pyannote.audio, gspread, ...)" -ForegroundColor DarkGray
& "$VenvDir\Scripts\pip.exe" install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies. Check requirements.txt." -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Dependencies installed." -ForegroundColor Green

# Credentials check
Write-Host "[check] client_secret.json..." -ForegroundColor Yellow
if (Test-Path "client_secret.json") {
    Write-Host "  OK: client_secret.json found." -ForegroundColor Green
} else {
    Write-Host "  WARNING: client_secret.json not found!" -ForegroundColor Red
    Write-Host "  Obtain OAuth 2.0 Client ID and place it in the project root." -ForegroundColor Red
    Write-Host "  More info: README.md -> Step 1" -ForegroundColor DarkGray
}

# Reminder about HF_TOKEN
Write-Host ""
Write-Host "  IMPORTANT: For speaker diarization (pyannote.audio), HF_TOKEN is required." -ForegroundColor Yellow
Write-Host "  Get token at https://huggingface.co/settings/tokens" -ForegroundColor DarkGray
Write-Host "  Accept model terms: https://hf.co/pyannote/speaker-diarization-3.1" -ForegroundColor DarkGray
Write-Host "  Set environment variable:" -ForegroundColor DarkGray
Write-Host "    `$env:HF_TOKEN = 'hf_...your_token...'" -ForegroundColor White

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Done! To run the script, execute:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    `$env:HF_TOKEN = 'hf_...'        # if not set yet" -ForegroundColor White
Write-Host "    .venv\Scripts\activate" -ForegroundColor White
Write-Host "    python main.py" -ForegroundColor White
Write-Host ""
Write-Host "  Or in a single command (without activation):" -ForegroundColor Cyan
Write-Host ""
Write-Host "    .venv\Scripts\python.exe main.py" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
