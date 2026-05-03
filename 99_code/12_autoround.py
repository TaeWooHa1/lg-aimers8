"""
AutoRound 양자화 - llmcompressor 기반 (평가 서버 호환)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
llmcompressor의 AutoRoundModifier를 사용하여 compressed-tensors 포맷으로 저장.
→ 평가 서버(vllm==0.14.1, compressed-tensors==0.13.0)와 호환.

사용법:
  Kaggle에서 GPU 런타임(T4 이상)으로 실행
  pip install -q llmcompressor auto-round mlflow dagshub
"""

# =========================================================
# 0. 패키지 설치 (Kaggle/Colab에서 먼저 실행!)
# =========================================================
# !pip install -q llmcompressor auto-round==0.9.6 mlflow dagshub
# !pip install -q transformers==4.57.3

import os
import time
import torch
import shutil
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# transformers 4.57.3 호환 패치 (auto-round가 제거된 함수를 참조하는 문제 해결)
import transformers.utils
if not hasattr(transformers.utils, 'is_tf_available'):
    transformers.utils.is_tf_available = lambda: False

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

# MLflow (선택사항 - 없으면 스킵)
try:
    import mlflow
    import dagshub
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[INFO] mlflow/dagshub 미설치. 트래킹 없이 진행합니다.")

# =========================================================
# 1. 경로 설정
# =========================================================
# HuggingFace에서 직접 다운로드 (약 2.5GB)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 출력 폴더
OUT_DIR = "/kaggle/working/model"

# =========================================================
# 2. AutoRound 양자화 설정
# =========================================================
SCHEME = "W4A16"                # 4비트 가중치, 16비트 활성화
TARGETS = "Linear"              # 양자화 대상 레이어
IGNORE = ["lm_head"]           # 양자화 제외 레이어

# AutoRound 최적화 하이퍼파라미터
AUTOROUND_ITERS = 500          # SignSGD 반복 수 (기본 200)
AUTOROUND_BATCH_SIZE = 32       # 캘리브레이션 배치 크기
# lr은 PyPI 버전에서 미지원 (None 기본값 = 1/iters 자동 설정, 동일 결과)

# 캘리브레이션 설정
NUM_CALIBRATION_SAMPLES = 1024  # 캘리브레이션 샘플 수
MAX_SEQUENCE_LENGTH = 512      # 최대 시퀀스 길이 (Kaggle 메모리 고려)

# =========================================================
# 2-1. MLflow 설정 (선택)
# =========================================================
DAGSHUB_REPO = "sthun0211/LGaimers"
MLFLOW_EXPERIMENT = "AutoRound_Quantization_v2"

if MLFLOW_AVAILABLE:
    dagshub.init(repo_owner="sthun0211", repo_name="LGaimers", mlflow=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"[INFO] MLflow 트래킹 서버: https://dagshub.com/{DAGSHUB_REPO}.mlflow")
    print(f"[INFO] Experiment: {MLFLOW_EXPERIMENT}")

# =========================================================
# 3. GPU 메모리 정리
# =========================================================
GPU_NAME = "N/A"
GPU_VRAM = 0.0
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {GPU_NAME}")
    print(f"[INFO] VRAM: {GPU_VRAM:.1f} GB")
else:
    print("[WARNING] GPU를 찾을 수 없습니다. CPU로 실행됩니다 (매우 느림).")

# =========================================================
# 4. 모델 로드 (로컬 base_model)
# =========================================================
print("\n" + "=" * 60)
print(f"[INFO] 모델 다운로드 중... ({MODEL_ID})")
print("       (처음 실행 시 약 2.5GB 다운로드)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"[INFO] 모델 로드 완료!")
print(f"       파라미터: {model.num_parameters() / 1e9:.2f}B")

# =========================================================
# 5. 캘리브레이션 데이터 설정
# =========================================================
# oneshot이 내부에서 wikitext-2를 자동 다운로드 & 토크나이즈 처리
CALIB_DATASET = "wikitext"
CALIB_DATASET_CONFIG = "wikitext-2-raw-v1"

print("\n" + "=" * 60)
print(f"[INFO] 캘리브레이션 데이터: {CALIB_DATASET} ({CALIB_DATASET_CONFIG})")
print(f"       샘플 수: {NUM_CALIBRATION_SAMPLES}")
print(f"       시퀀스 길이: {MAX_SEQUENCE_LENGTH}")
print("=" * 60)

# =========================================================
# 6. AutoRound 양자화
# =========================================================

def run_quantization():
    """양자화 실행 (MLflow 트래킹 포함)"""
    global model

    print("\n" + "=" * 60)
    print("[INFO] AutoRound 양자화 시작 (llmcompressor 기반)")
    print(f"       Scheme: {SCHEME}")
    print(f"       Iterations: {AUTOROUND_ITERS}")
    print(f"       Batch Size: {AUTOROUND_BATCH_SIZE}")
    print(f"       Calibration Samples: {NUM_CALIBRATION_SAMPLES}")
    print(f"       Max Seq Length: {MAX_SEQUENCE_LENGTH}")
    print("=" * 60)

    quant_start = time.time()

    # AutoRoundModifier 설정 (llmcompressor 방식)
    recipe = AutoRoundModifier(
        targets=TARGETS,
        scheme=SCHEME,
        ignore=IGNORE,
        iters=AUTOROUND_ITERS,
        batch_size=AUTOROUND_BATCH_SIZE,
    )

    # llmcompressor oneshot으로 양자화 실행 (데이터셋 이름 직접 전달)
    oneshot(
        model=model,
        dataset=CALIB_DATASET,
        dataset_config_name=CALIB_DATASET_CONFIG,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        shuffle_calibration_samples=False,
    )

    quant_time = time.time() - quant_start
    print(f"[INFO] AutoRound 양자화 완료! (소요 시간: {quant_time:.1f}초)")

    return quant_time


def save_model(quant_time):
    """양자화 모델 저장 + ZIP 생성"""

    # ── 모델 저장 (compressed-tensors 포맷) ──
    print(f"\n[INFO] 모델 저장 중... → {OUT_DIR}")
    print("       포맷: compressed-tensors (vLLM 0.14.1 호환)")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    # 저장 확인
    print("[INFO] 저장된 파일:")
    total_size = 0
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / (1024 * 1024)
            total_size += size
            print(f"       - {f} ({size:.1f} MB)")
    print(f"       총 크기: {total_size:.1f} MB")

    # ── ZIP 생성 ──
    zip_name = "optimized_submit"
    print(f"\n[INFO] {zip_name}.zip 생성 중...")

    shutil.make_archive(
        base_name=f"/kaggle/working/{zip_name}",
        format="zip",
        root_dir="/kaggle/working",
        base_dir="model",
    )

    zip_path = f"/kaggle/working/{zip_name}.zip"
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[INFO] 생성 완료: {zip_path} ({zip_size:.1f} MB)")

    # 제출 파일 용량 검증 (10GB 제한)
    if zip_size > 10 * 1024:
        print(f"[WARNING] ZIP 파일이 10GB를 초과합니다! ({zip_size:.1f} MB)")
    else:
        print(f"[INFO] 제출 용량 OK (제한: 10GB, 현재: {zip_size:.1f} MB)")

    return total_size, zip_size, zip_path


# ── 실행 ──
if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name=f"AutoRound_{SCHEME}_iters{AUTOROUND_ITERS}_llmc"):

        # 파라미터 기록
        mlflow.log_params({
            "model_id": MODEL_ID,
            "quant_method": "AutoRound (llmcompressor)",
            "scheme": SCHEME,
            "iters": AUTOROUND_ITERS,
            "batch_size": AUTOROUND_BATCH_SIZE,
            "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "save_format": "compressed-tensors",
            "gpu": GPU_NAME,
            "gpu_vram_gb": GPU_VRAM,
        })

        quant_time = run_quantization()
        total_size, zip_size, zip_path = save_model(quant_time)

        # 메트릭 기록
        mlflow.log_metric("quantization_time_sec", quant_time)
        mlflow.log_metric("model_size_mb", total_size)
        mlflow.log_metric("zip_size_mb", zip_size)

        # config.json 아티팩트 기록
        config_path = os.path.join(OUT_DIR, "config.json")
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="model_config")

        print(f"\n[INFO] MLflow Run 기록 완료! → https://dagshub.com/{DAGSHUB_REPO}.mlflow")
else:
    quant_time = run_quantization()
    total_size, zip_size, zip_path = save_model(quant_time)


# =========================================================
# 완료!
# =========================================================
print("\n" + "=" * 60)
print("✅ AutoRound 양자화 완료! (llmcompressor 기반)")
print("=" * 60)
print(f"""
📊 설정 요약:
   • Model: {MODEL_ID}
   • Scheme: {SCHEME}
   • Iterations: {AUTOROUND_ITERS}
   • Batch Size: {AUTOROUND_BATCH_SIZE}
   • Calibration Samples: {NUM_CALIBRATION_SAMPLES}
   • Max Seq Length: {MAX_SEQUENCE_LENGTH}
   • Save Format: compressed-tensors (vLLM 호환)

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: {zip_path}

🔧 이전 실패 코드(10_autoaround.py)와의 차이:
   • standalone auto_round → llmcompressor AutoRoundModifier 사용
   • format='llm_compressor' → save_compressed=True (compressed-tensors)
   • 평가 서버 vllm==0.14.1, compressed-tensors==0.13.0 호환

🚀 다운로드된 ZIP 파일을 대회에 제출하세요!
""")
