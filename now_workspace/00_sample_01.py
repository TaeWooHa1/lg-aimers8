"""
GPTQ 양자화 최적화 버전 - Google Colab 전용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
1. Colab에서 GPU 런타임 선택 (T4 이상 권장)
2. 아래 코드를 셀에 복사하여 실행

주의: Colab 무료 버전은 메모리 제한이 있어 OOM 발생 가능
     → num_calibration_samples나 max_seq_length 줄이기
"""

# =========================================================
# 0. 패키지 설치 (Colab/Kaggle에서 먼저 실행!)
# =========================================================
# !pip install -q llmcompressor transformers datasets accelerate dagshub mlflow

import os
os.environ['DAGSHUB_USER_TOKEN'] = '24a9f701d9da66e29e415a1fcf12832c87b28093'
import torch
import shutil
import time
from pathlib import Path
import mlflow
import dagshub
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# DagsHub + MLflow 연결
# =========================================================
dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
mlflow.set_experiment("htw-actorder")

# =========================================================
# 1. 경로 설정 (Colab용)
# =========================================================
# HuggingFace에서 직접 다운로드 (약 2.5GB)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 출력 폴더 (Colab 환경)
OUT_DIR = "/content/model"

# =========================================================
# 2. 데이터셋 설정
# =========================================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ⭐ Colab 메모리 고려 설정
# - Colab Pro: 512 샘플, 1024 길이 가능
# - Colab Free: 256 샘플, 512 길이 권장
NUM_CALIBRATION_SAMPLES = 256  # 메모리 부족 시 128로 줄이기
MAX_SEQUENCE_LENGTH = 512      # 메모리 부족 시 256으로 줄이기

# =========================================================
# 3. 양자화 설정 (최적화)
# =========================================================
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
ACTORDER = "static"           # 활성화 정렬 ('static', 'group', 'weight', 'dynamic')
DAMPENING_FRAC = 0.01

# =========================================================
# 4. GPU 메모리 정리
# =========================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] GPU를 찾을 수 없습니다. CPU로 실행됩니다 (매우 느림).")

# =========================================================
# MLflow 실험 기록 시작
# =========================================================
with mlflow.start_run(run_name="actorder-static"):

    # 설정값(params) 기록
    mlflow.log_params({
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_seq_length": MAX_SEQUENCE_LENGTH,
        "quantization_scheme": SCHEME,
        "targets": str(TARGETS),
        "ignore": str(IGNORE),
        "actorder": ACTORDER,
        "dampening_frac": DAMPENING_FRAC,
    })

    # =========================================================
    # 5. 모델 로드
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
    # 6. 데이터셋 로드 & 전처리
    # =========================================================
    print("\n" + "=" * 60)
    print(f"[INFO] 캘리브레이션 데이터 로드 중...")
    print(f"       데이터셋: {DATASET_ID}")
    print(f"       샘플 수: {NUM_CALIBRATION_SAMPLES}")
    print("=" * 60)

    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    )

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False
            )
        }

    ds = ds.map(preprocess)
    print(f"[INFO] 데이터 전처리 완료 ({len(ds)}개 샘플)")

    # =========================================================
    # 7. GPTQ 양자화
    # =========================================================
    print("\n" + "=" * 60)
    print("[INFO] GPTQ 양자화 시작 (약 5~10분 소요)")
    print(f"       Scheme: {SCHEME}")
    print(f"       ActOrder: {ACTORDER}")
    print(f"       Max Seq Length: {MAX_SEQUENCE_LENGTH}")
    print("=" * 60)

    recipe = [
        GPTQModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
            actorder=ACTORDER,
            dampening_frac=DAMPENING_FRAC,
        )
    ]

    start_time = time.time()

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    quantization_time = time.time() - start_time
    print(f"[INFO] GPTQ 양자화 완료! (소요 시간: {quantization_time:.1f}초)")

    # 양자화 시간 기록
    mlflow.log_metric("quantization_time_sec", quantization_time)

    # =========================================================
    # 8. 모델 저장
    # =========================================================
    print(f"\n[INFO] 모델 저장 중... → {OUT_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    # 저장 확인
    print("[INFO] 저장된 파일:")
    for f in os.listdir(OUT_DIR):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
        print(f"       - {f} ({size:.1f} MB)")

    # =========================================================
    # 9. ZIP 생성 & 다운로드
    # =========================================================
    zip_name = "optimized_submit"
    print(f"\n[INFO] {zip_name}.zip 생성 중...")

    shutil.make_archive(
        base_name=f"/content/{zip_name}",
        format="zip",
        root_dir="/content",
        base_dir="model",
    )

    zip_path = f"/content/{zip_name}.zip"
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[INFO] 생성 완료: {zip_path} ({zip_size:.1f} MB)")

    # ZIP 파일 크기 기록
    mlflow.log_metric("model_zip_size_MB", zip_size)

    # GPU 정보 기록
    if torch.cuda.is_available():
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        mlflow.log_metric("gpu_vram_GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    print("[INFO] MLflow 기록 완료!")

    # Colab에서 자동 다운로드
    try:
        from google.colab import files
        print("\n[INFO] 파일 다운로드 시작...")
        files.download(zip_path)
    except ImportError:
        print(f"\n[INFO] Colab 환경이 아닙니다. 수동으로 다운로드하세요: {zip_path}")

    # =========================================================
    # 완료!
    # =========================================================
    print("\n" + "=" * 60)
    print("✅ 양자화 완료!")
    print("=" * 60)
    print(f"""
📊 설정 요약:
   • Model: {MODEL_ID}
   • Scheme: {SCHEME}
   • ActOrder: {ACTORDER}
   • Samples: {NUM_CALIBRATION_SAMPLES}
   • Max Length: {MAX_SEQUENCE_LENGTH}
   • 양자화 시간: {quantization_time:.1f}초

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: {zip_path} ({zip_size:.1f} MB)

🚀 다운로드된 ZIP 파일을 대회에 제출하세요!
📊 DagsHub에서 실험 기록 확인: https://dagshub.com/sthun0211/LGaimers.mlflow
""")
