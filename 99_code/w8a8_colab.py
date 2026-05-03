"""
W8A8 GPTQ 양자화 - Google Colab 전용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
1. Colab에서 GPU 런타임 선택 (T4 이상 권장)
2. 아래 코드를 셀에 복사하여 실행
3. 완료 후 자동으로 ZIP 파일 다운로드

주의: Colab 무료 버전은 메모리 제한이 있어 OOM 발생 가능
     → num_calibration_samples나 max_seq_length 줄이기
"""

# =========================================================
# 0. 패키지 설치 (Colab에서 먼저 실행!)
# =========================================================
# !pip install -q llmcompressor transformers==4.57.3 datasets accelerate mlflow dagshub

import os
import json
import time
import torch
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# MLflow (선택)
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '1ee266cf0159abb2c8ad8ae564274c6918599acd'
    import mlflow
    import dagshub
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("W8A8-all-params")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# 1. 경로 설정 (Colab용)
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/content/model"

# =========================================================
# 2. 데이터셋 설정
# =========================================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ⭐ Colab 메모리 고려 설정
NUM_CALIBRATION_SAMPLES = 256   # 메모리 부족 시 128로 줄이기
MAX_SEQUENCE_LENGTH = 512       # 메모리 부족 시 256으로 줄이기

# =========================================================
# 3. 양자화 설정 (⭐ 파라미터가이드.md 기반 최적화)
# =========================================================
SCHEME = "W8A8"                 # ⭐ 8-bit weight + 8-bit activation
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
ACTORDER = "dynamic"            # ⭐ static→dynamic (공짜 정확도 ↑↑↑)
BLOCK_SIZE = 64                 # ⭐ 128→64 (정확도 ↑, 속도 영향 미미)
DAMPENING_FRAC = 0.01
SEQUENTIAL_TARGETS = ["Exaone4DecoderLayer"]  # ⭐ 레이어별 순차 처리

# 🔵 공짜 속도 (정확도 영향 0)
MAX_POSITION_EMBEDDINGS = 32768  # ⭐ 65536→32768 (KV Cache 50% 절감!)

# 🔧 서버 호환
TARGET_CT_VERSION = "0.13.0"

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
print(f"[INFO] GPTQ {SCHEME} 양자화 시작")
print(f"       Scheme: {SCHEME}")
print(f"       ActOrder: {ACTORDER}")
print(f"       Block Size: {BLOCK_SIZE}")
print(f"       Sequential Targets: {SEQUENTIAL_TARGETS}")
print(f"       Max Seq Length: {MAX_SEQUENCE_LENGTH}")
print("=" * 60)

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        actorder=ACTORDER,
        block_size=BLOCK_SIZE,
        dampening_frac=DAMPENING_FRAC,
        sequential_targets=SEQUENTIAL_TARGETS,
    )
]

start_time = time.time()

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=True,
)

quant_time = time.time() - start_time
print(f"[INFO] GPTQ 양자화 완료! ({quant_time:.1f}초)")

# =========================================================
# 8. 모델 저장
# =========================================================
print(f"\n[INFO] 모델 저장 중... → {OUT_DIR}")

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# 저장 확인
total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"[INFO] 모델 크기: {total_size / (1024*1024):.1f} MB")
print("[INFO] 저장된 파일:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / (1024 * 1024)
        print(f"       - {f} ({size:.1f} MB)")

# =========================================================
# 9. config.json 최적화 + 서버 호환 패치
# =========================================================
print(f"\n[INFO] config.json 최적화 + 서버 호환 패치")
config_path = os.path.join(OUT_DIR, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

# 🔵 KV Cache 최적화
original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS
print(f"       max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")

# 🔧 compressed-tensors version 패치
if "quantization_config" in config:
    original_version = config["quantization_config"].get("version", "N/A")
    config["quantization_config"]["version"] = TARGET_CT_VERSION
    print(f"       compressed-tensors version: {original_version} → {TARGET_CT_VERSION} ✅")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("       config.json 패치 완료!")

# =========================================================
# 10. ZIP 생성 & 다운로드
# =========================================================
zip_name = "w8a8_submit"
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

# Colab에서 자동 다운로드
try:
    from google.colab import files
    print("\n[INFO] 파일 다운로드 시작...")
    files.download(zip_path)
except ImportError:
    print(f"\n[INFO] Colab 환경이 아닙니다. 수동으로 다운로드하세요: {zip_path}")

# =========================================================
# 11. MLflow 기록 (선택)
# =========================================================
if USE_MLFLOW:
    with mlflow.start_run(run_name=f"{SCHEME}-bs{BLOCK_SIZE}-cal{NUM_CALIBRATION_SAMPLES}"):
        mlflow.log_params({
            "model_id": MODEL_ID,
            "scheme": SCHEME,
            "actorder": ACTORDER,
            "block_size": BLOCK_SIZE,
            "dampening_frac": DAMPENING_FRAC,
            "calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_seq_length": MAX_SEQUENCE_LENGTH,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "sequential_targets": str(SEQUENTIAL_TARGETS),
            "shuffle_calibration_samples": True,
            "ignore": str(IGNORE),
            "ct_version_patch": TARGET_CT_VERSION,
        })
        mlflow.log_metrics({
            "quantization_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })
        print("[INFO] MLflow 기록 완료!")

# =========================================================
# 완료!
# =========================================================
print("\n" + "=" * 60)
print("✅ W8A8 양자화 완료!")
print("=" * 60)
print(f"""
📊 설정 요약:
   • Model: {MODEL_ID}
   • Scheme: {SCHEME}
   • ActOrder: {ACTORDER}
   • Block Size: {BLOCK_SIZE}
   • Sequential Targets: {SEQUENTIAL_TARGETS}
   • Samples: {NUM_CALIBRATION_SAMPLES}
   • Max Length: {MAX_SEQUENCE_LENGTH}
   • max_position_embeddings: {MAX_POSITION_EMBEDDINGS}
   • compressed-tensors version: {TARGET_CT_VERSION}
   • 양자화 시간: {quant_time:.1f}초

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: {zip_path}

🚀 다운로드된 ZIP 파일을 대회에 제출하세요!
""")
