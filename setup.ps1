# setup.ps1 — Ініціалізація віртуального середовища та встановлення залежностей
$ErrorActionPreference = "Stop"
$VenvDir = ".venv"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Аналізатор дзвінків — Ініціалізація  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Перевірка Python
Write-Host "[1/4] Перевірка Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ПОМИЛКА: Python не знайдено. Встановіть Python 3.10+ та додайте до PATH." -ForegroundColor Red
    exit 1
}
Write-Host "  OK: $pythonVersion" -ForegroundColor Green

# 2. Створення venv (якщо ще не існує)
Write-Host "[2/4] Налаштування віртуального середовища (.venv)..." -ForegroundColor Yellow
if (Test-Path $VenvDir) {
    Write-Host "  Знайдено існуюче .venv — пропускаємо створення." -ForegroundColor DarkGray
} else {
    python -m venv $VenvDir
    Write-Host "  OK: .venv створено." -ForegroundColor Green
}

# 3. Активація venv та встановлення залежностей
Write-Host "[3/5] Оновлення pip..." -ForegroundColor Yellow
& "$VenvDir\Scripts\pip.exe" install --upgrade pip --quiet

Write-Host "[4/5] Встановлення llama-cpp-python (pre-built CPU wheel, без C++ компілятора)..." -ForegroundColor Yellow
Write-Host "  (це може зайняти кілька хвилин)" -ForegroundColor DarkGray
& "$VenvDir\Scripts\pip.exe" install llama-cpp-python `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

if ($LASTEXITCODE -ne 0) {
    Write-Host "ПОМИЛКА: Не вдалося встановити llama-cpp-python." -ForegroundColor Red
    Write-Host "Переконайтеся, що ви підключені до інтернету." -ForegroundColor Red
    exit 1
}
Write-Host "  OK: llama-cpp-python встановлено." -ForegroundColor Green

Write-Host "[5/5] Встановлення інших залежностей з requirements.txt..." -ForegroundColor Yellow
& "$VenvDir\Scripts\pip.exe" install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ПОМИЛКА: Не вдалося встановити залежності. Перевірте requirements.txt." -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Залежності встановлено." -ForegroundColor Green

# 6 (колишній 4). Перевірка наявності credentials.json
Write-Host "[перевірка] credentials.json..." -ForegroundColor Yellow
if (Test-Path "credentials.json") {
    Write-Host "  OK: credentials.json знайдено." -ForegroundColor Green
} else {
    Write-Host "  УВАГА: credentials.json не знайдено!" -ForegroundColor Red
    Write-Host "  Отримайте файл Service Account та покладіть його у корінь проєкту." -ForegroundColor Red
    Write-Host "  Детальніше: README.md -> Крок 1" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Готово! Для запуску скрипту виконайте:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    .venv\Scripts\activate" -ForegroundColor White
Write-Host "    python main.py" -ForegroundColor White
Write-Host ""
Write-Host "  Або одним рядком (без активації):" -ForegroundColor Cyan
Write-Host ""
Write-Host "    .venv\Scripts\python.exe main.py" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
