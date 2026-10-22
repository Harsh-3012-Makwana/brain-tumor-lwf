# Windows PowerShell Setup Script
Write-Host "=== Setting up Brain Tumor LwF Environment ===" -ForegroundColor Cyan

$PythonExe = "C:\Users\Harsh\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python.exe"
}

Write-Host "Creating virtual environment with: $PythonExe" -ForegroundColor Yellow
& $PythonExe -m venv .venv

Write-Host "Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& .venv\Scripts\pip.exe install -r requirements.txt

Write-Host "Verifying PyTorch CUDA installation..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

Write-Host "Setup complete!" -ForegroundColor Green
