@echo off
chcp 65001 >nul
REM ============================================================
REM LG Aimers 평가 환경 Docker 실행 스크립트 (Windows)
REM ============================================================
REM
REM 사용법:
REM   run_eval.bat build              :: Docker 이미지 빌드
REM   run_eval.bat shell              :: 컨테이너 진입 (인터랙티브)
REM   run_eval.bat eval [옵션...]     :: evaluate_local.py 바로 실행
REM   run_eval.bat check              :: 환경 확인
REM
REM 예시:
REM   run_eval.bat build
REM   run_eval.bat eval --base-model /workspace/models/base_model ^
REM                      --target-model /workspace/models/quantized/my_model ^
REM                      --tasks gsm8k,mmlu
REM   run_eval.bat eval --base-model /workspace/models/base_model ^
REM                      --target-model /workspace/models/quantized/my_model ^
REM                      --skip-speed
REM ============================================================

setlocal enabledelayedexpansion

REM 스크립트가 있는 디렉토리로 이동
cd /d "%~dp0"

set IMAGE_NAME=lg-aimers-eval

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="build" goto build
if "%1"=="shell" goto shell
if "%1"=="eval" goto eval
if "%1"=="check" goto check
goto help

:build
echo [BUILD] Docker 이미지 빌드 중...
docker compose build
if errorlevel 1 (
    echo [ERROR] 빌드 실패!
    echo   - Docker Desktop이 실행 중인지 확인하세요
    exit /b 1
)
echo [OK] 빌드 완료!
goto end

:shell
echo [SHELL] 컨테이너 진입 (인터랙티브 모드)...
docker compose run --rm lg-aimers-eval bash
goto end

:eval
shift
echo [EVAL] evaluate_local.py 실행 중...
REM 나머지 인자 모두 수집 (쉼표 포함 값 보존)
set "EVAL_ARGS="
:eval_loop
if "%~1"=="" goto eval_run
set "EVAL_ARGS=%EVAL_ARGS% %~1"
shift
goto eval_loop
:eval_run
docker compose run --rm lg-aimers-eval python /workspace/evaluate_local.py %EVAL_ARGS%
goto end

:check
echo [CHECK] 환경 확인 중...
docker compose run --rm lg-aimers-eval bash -c "echo '=== OS ===' && cat /etc/os-release | head -3 && echo '' && echo '=== Python ===' && python --version && echo '' && echo '=== CUDA ===' && (nvidia-smi 2>/dev/null || echo 'GPU 미감지 (Docker GPU 패스스루 확인 필요)') && echo '' && echo '=== PyTorch ===' && python -c \"import torch; print(f'torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')\" && echo '' && echo '=== Key Packages ===' && python -c \"import transformers, accelerate, safetensors; print(f'transformers: {transformers.__version__}'); print(f'accelerate: {accelerate.__version__}'); print(f'safetensors: {safetensors.__version__}')\" && echo '' && echo '=== 모델 마운트 확인 ===' && echo 'base_model:' && ls -la /workspace/models/base_model/*.safetensors 2>/dev/null || echo '  base_model 미발견' && echo 'quantized:' && ls /workspace/models/quantized/ 2>/dev/null || echo '  quantized 모델 없음' && echo '' && echo '[OK] 환경 확인 완료!'"
goto end

:help
echo ============================================================
echo   LG Aimers 평가 환경 Docker 실행 스크립트
echo ============================================================
echo.
echo 사용법: %~nx0 {build^|shell^|eval^|check}
echo.
echo   build   - Docker 이미지 빌드
echo   shell   - 컨테이너 진입 (인터랙티브)
echo   eval    - evaluate_local.py 실행
echo   check   - 환경 확인
echo.
echo 예시:
echo   %~nx0 build
echo   %~nx0 check
echo   %~nx0 eval --base-model /workspace/models/base_model ^
echo              --target-model /workspace/models/quantized/my_model ^
echo              --skip-speed
echo   %~nx0 shell
goto end

:end
endlocal
