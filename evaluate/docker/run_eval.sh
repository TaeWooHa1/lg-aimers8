#!/bin/bash
# ============================================================
# LG Aimers 평가 환경 Docker 실행 스크립트
# ============================================================
#
# 사용법:
#   ./run_eval.sh build              # Docker 이미지 빌드
#   ./run_eval.sh shell              # 컨테이너 진입 (인터랙티브)
#   ./run_eval.sh eval [옵션...]     # evaluate_local.py 바로 실행
#   ./run_eval.sh check              # 환경 확인
#
# 예시:
#   ./run_eval.sh build
#   ./run_eval.sh eval --base-model /workspace/models/base_model \
#                       --target-model /workspace/models/quantized_model \
#                       --tasks gsm8k,mmlu
#   ./run_eval.sh eval --base-model /workspace/models/base_model \
#                       --target-model /workspace/models/quantized_model \
#                       --skip-speed
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="lg-aimers-eval"

case "${1:-help}" in
  build)
    echo "🔨 Docker 이미지 빌드 중..."
    docker compose build
    echo "✅ 빌드 완료!"
    ;;

  shell)
    echo "🐚 컨테이너 진입 (인터랙티브 모드)..."
    docker compose run --rm lg-aimers-eval bash
    ;;

  eval)
    shift
    echo "📊 evaluate_local.py 실행 중..."
    docker compose run --rm lg-aimers-eval \
      python /workspace/evaluate_local.py "$@"
    ;;

  check)
    echo "🔍 환경 확인 중..."
    docker compose run --rm lg-aimers-eval bash -c '
      echo "=== OS ==="
      cat /etc/os-release | head -3
      echo ""
      echo "=== Python ==="
      python --version
      echo ""
      echo "=== CUDA ==="
      nvidia-smi 2>/dev/null || echo "GPU 미감지 (Docker GPU 패스스루 확인 필요)"
      echo ""
      echo "=== PyTorch ==="
      python -c "import torch; print(f\"torch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")"
      echo ""
      echo "=== Key Packages ==="
      python -c "
import transformers, accelerate, safetensors, vllm
print(f\"transformers: {transformers.__version__}\")
print(f\"accelerate: {accelerate.__version__}\")
print(f\"safetensors: {safetensors.__version__}\")
print(f\"vllm: {vllm.__version__}\")
"
      echo ""
      echo "✅ 환경 확인 완료!"
    '
    ;;

  *)
    echo "사용법: $0 {build|shell|eval|check}"
    echo ""
    echo "  build   - Docker 이미지 빌드"
    echo "  shell   - 컨테이너 진입 (인터랙티브)"
    echo "  eval    - evaluate_local.py 실행"
    echo "  check   - 환경 확인"
    ;;
esac
