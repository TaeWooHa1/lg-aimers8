
# LG Aimers 디렉터리 구조 재정리 스크립트
Set-Location "c:\Users\htw02\project_github\lg_aimers"

# ── 새 디렉터리 생성 ──
@("docs","guides","quantize\gptq","quantize\autoround","quantize\fp8",
  "quantize\advanced","quantize\final","teammates",
  "evaluate\docker","configs","results","scripts",
  "weekly\week1","weekly\week2","weekly\week3") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

# ── docs/ ──
Move-Item -Force "대회개요\대회개요.md"       "docs\대회개요.md"
Move-Item -Force "대회개요\대회규칙.md"       "docs\대회규칙.md"
Move-Item -Force "대회개요\코드제출가이드.md" "docs\코드제출가이드.md"
Move-Item -Force "00_evaluate.md"            "docs\00_evaluate.md"
if (Test-Path "03_week3\평가결과_분석.md") { Move-Item -Force "03_week3\평가결과_분석.md" "docs\평가결과_분석.md" }

# ── guides/ ──
Move-Item -Force "99_code\파라미터가이드.md"                  "guides\파라미터가이드.md"
Move-Item -Force "99_code\autoround_파라미터_가이드.md"       "guides\autoround_파라미터_가이드.md"
Move-Item -Force "99_code\21_fp8_개념.md"                    "guides\fp8_개념.md"
Move-Item -Force "99_code\22_1등가이드.md"                   "guides\1등가이드.md"
Move-Item -Force "99_code\22_kd_sparse_quant_walkthrough.md" "guides\kd_sparse_quant_walkthrough.md"
Move-Item -Force "99_code\30_smqt_spst.md"                   "guides\smqt_spst.md"
if (Test-Path "100_study\변수설명.md")   { Move-Item -Force "100_study\변수설명.md"   "guides\변수설명.md" }
if (Test-Path "100_study\12_autoround.md") { Move-Item -Force "100_study\12_autoround.md" "guides\autoround.md" }

# ── quantize/gptq/ ──
Move-Item -Force "99_code\00_sample_local_0.57.py"  "quantize\gptq\w4a16_baseline.py"
Move-Item -Force "99_code\05_sample_colab.py"       "quantize\gptq\w4a16_colab.py"
Move-Item -Force "99_code\05_sample_kaggle.py"      "quantize\gptq\w4a16_kaggle.py"
Move-Item -Force "99_code\19_w4a16_all_params.py"   "quantize\gptq\w4a16_all_params.py"
Move-Item -Force "99_code\20_w4a16_colab.py"        "quantize\gptq\w4a16_colab_optimized.py"
Move-Item -Force "99_code\w8a8_colab.py"            "quantize\gptq\w8a8_colab.py"
Move-Item -Force "99_code\30_w8a8_kaggle.py"        "quantize\gptq\w8a8_kaggle.py"
if (Test-Path "99_code\17_w8a16.py") { Move-Item -Force "99_code\17_w8a16.py" "quantize\gptq\w8a16.py" }

# ── quantize/autoround/ ──
Move-Item -Force "99_code\12_autoround.py"    "quantize\autoround\autoround_v1.py"
Move-Item -Force "99_code\15_autoround_v2.py" "quantize\autoround\autoround_v2.py"

# ── quantize/fp8/ ──
Move-Item -Force "99_code\21_fp8.py" "quantize\fp8\fp8.py"

# ── quantize/advanced/ ──
Move-Item -Force "99_code\23_kd_train.py"    "quantize\advanced\kd_train.py"
Move-Item -Force "99_code\24_sparse_quant.py" "quantize\advanced\sparse_quant.py"

# ── quantize/final/ ──
Move-Item -Force "99_code\40_best.py"        "quantize\final\best.py"
Move-Item -Force "99_code\25_final_submit.py" "quantize\final\final_submit.py"

# ── teammates/ ──
Move-Item -Force "99_code\98_seokbun_0.61.py" "teammates\seokbun_0.61.py"
Move-Item -Force "99_code\99_taehun_0.6.py"   "teammates\taehun_0.6.py"

# ── evaluate/ ──
Move-Item -Force "now_workspace\evaluate_local.py"        "evaluate\evaluate_local.py"
Move-Item -Force "99_code\80_complexity_check.py"         "evaluate\complexity_check.py"
Move-Item -Force "now_workspace\models\copy_configs.py"   "evaluate\copy_configs.py"
Move-Item -Force "now_workspace\docker\Dockerfile"              "evaluate\docker\Dockerfile"
Move-Item -Force "now_workspace\docker\docker-compose.yml"      "evaluate\docker\docker-compose.yml"
Move-Item -Force "now_workspace\docker\requirements_server.txt" "evaluate\docker\requirements_server.txt"
Move-Item -Force "now_workspace\docker\run_eval.bat"            "evaluate\docker\run_eval.bat"
Move-Item -Force "now_workspace\docker\run_eval.sh"             "evaluate\docker\run_eval.sh"

# ── weekly/ ──
Move-Item -Force "01_week\analysis.md"       "weekly\week1\analysis.md"
Move-Item -Force "01_week\01_architecture.md" "weekly\week1\architecture.md"
Move-Item -Force "01_week\colab_baseline.py" "weekly\week1\colab_baseline.py"
Move-Item -Force "01_week\colab_awq.py"      "weekly\week1\colab_awq.py"
Move-Item -Force "01_week\colab_gptq_v2.py"  "weekly\week1\colab_gptq_v2.py"
Move-Item -Force "01_week\colab_quantize.py" "weekly\week1\colab_quantize.py"
Move-Item -Force "01_week\success.py"        "weekly\week1\success.py"
if (Test-Path "02_week2\awq개념.md")   { Move-Item -Force "02_week2\awq개념.md"   "weekly\week2\awq개념.md" }
if (Test-Path "02_week2\진행사항.md")  { Move-Item -Force "02_week2\진행사항.md"  "weekly\week2\진행사항.md" }
if (Test-Path "03_week3\study.md")     { Move-Item -Force "03_week3\study.md"     "weekly\week3\study.md" }
if (Test-Path "03_week3\진행사항.md")  { Move-Item -Force "03_week3\진행사항.md"  "weekly\week3\진행사항.md" }

# ── configs/ ──
Get-ChildItem "now_workspace\models\config\*.json" | ForEach-Object { Move-Item -Force $_.FullName "configs\" }

# ── results/ ──
Move-Item -Force "now_workspace\baseline_result.json"                "results\baseline_result.json"
Move-Item -Force "now_workspace\eval_comparison_20260219_234700.json" "results\eval_comparison.json"
if (Test-Path "now_workspace\models\모델결과.md") { Move-Item -Force "now_workspace\models\모델결과.md" "results\모델결과.md" }

# ── scripts/ ──
Move-Item -Force "run_docker.bat" "scripts\run_docker.bat"

# ── 불필요 파일 삭제 ──
@(
    "99_code\00_sample_colab3.py", "99_code\04_sample_kaggle_0.45.py",
    "99_code\06_0214_kaggle.py",   "99_code\07_sample_kaggle_cali1024.py",
    "99_code\08_samle1024_kaggle.py","99_code\10_autoaround.py",
    "99_code\11_0.62_code.py",     "99_code\13_autoround.py",
    "99_code\14_autoround.py",     "99_code\16_w4a16.py",
    "99_code\18_w4a16.ppy",        "99_code\31_level9.py",
    "99_code\98_seokbun2.py",      "99_code\30_내용정리.md",
    "99_code\41_내용정리.md",      "04_sample_kaggle.py"
) | ForEach-Object { if (Test-Path $_) { Remove-Item -Force $_; Write-Host "Deleted: $_" } }

Write-Host ''
Write-Host '✅ 파일 이동 완료! 이제 아래 명령어 실행:'
Write-Host '  git add -A'
Write-Host '  git commit -m refactor_directory'
Write-Host '  git push new-origin main --force'
