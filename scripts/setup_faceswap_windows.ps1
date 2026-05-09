param(
    [string]$ProjectRoot = ".",
    [string]$VenvPath = ".venv311",
    [switch]$SkipBuildTools
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "No se encontro el comando requerido: $Name"
    }
}

try {
    Write-Step "Validando prerequisitos"
    Assert-Command -Name "winget"

    $projectFull = Resolve-Path $ProjectRoot
    Set-Location $projectFull

    if (-not (Test-Path $VenvPath)) {
        throw "No se encontro el entorno virtual en: $VenvPath"
    }

    $pythonExe = Join-Path $VenvPath "Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "No se encontro python del entorno virtual en: $pythonExe"
    }

    if (-not $SkipBuildTools) {
        Write-Step "Instalando Microsoft C++ Build Tools (puede tardar varios minutos)"
        winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    }
    else {
        Write-Step "Saltando instalacion de Build Tools por parametro -SkipBuildTools"
    }

    Write-Step "Actualizando pip/setuptools/wheel"
    & $pythonExe -m pip install --upgrade pip setuptools wheel

    Write-Step "Reinstalando insightface 0.7.3 y dependencias"
    & $pythonExe -m pip uninstall -y insightface
    & $pythonExe -m pip install --no-cache-dir insightface==0.7.3 onnxruntime opencv-python numpy

    Write-Step "Verificando imports"
    & $pythonExe -c "import cv2, insightface, onnxruntime, numpy; print('OK:', insightface.__version__)"

    Write-Host "`nListo. Ahora ejecuta:" -ForegroundColor Green
    Write-Host "$pythonExe .\face_swap.py" -ForegroundColor Yellow
}
catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
