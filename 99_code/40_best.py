"""
🎯 W8A8 (INT8) + max_position_embeddings 최적화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

팀원 검증 완료: W8A8 = 0.612점 ✅
이 코드: W8A8 + actorder + dampening + max_position_embeddings

04_optimized_gptq.py (0.601점) 기반
  → scheme만 W4A16 → W8A8로 변경
  → max_position_embeddings 적용
  → 나머지 설정 100% 동일!
"""

# =========================================================
# Kaggle 패키지 설치
# =========================================================
import subprocess
import sys

packages = [
    "llmcompressor",
    "dagshub",
    "mlflow",
    "datasets",
    "transformers>=4.40.0",
    "accelerate",
]

for pkg in packages:
    print(f"📦 설치 중: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ 패키지 설치 완료!\n")

import os
import torch
import shutil
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# 설정
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

# 캘리브레이션 (0.601과 동일)
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# GPTQ 설정 (0.601과 동일, scheme만 W8A8!)
SCHEME = "W8A8"              # W8A8 변경
ACTORDER = "dynamic"         # 0.601과 동일
DAMPENING_FRAC = 0.01        # 0.601과 동일

# config.json 최적화
MAX_POSITION_EMBEDDINGS = 16384  # 65536 → 16384 (속도비율 ↑)

# DagsHub
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-w8a8")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# 실행
# =========================================================
print("=" * 60)
print(f"🔬 GPTQ {SCHEME} + max_pos={MAX_POSITION_EMBEDDINGS}")
print(f"   actorder={ACTORDER}, dampening={DAMPENING_FRAC}")
print(f"   samples={NUM_CALIBRATION_SAMPLES}, seq_len={MAX_SEQUENCE_LENGTH}")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

# 1. 모델 로드
print("\n[1/5] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터")

# 2. 데이터 준비
print(f"\n[2/5] 캘리브레이션 데이터 {NUM_CALIBRATION_SAMPLES}개 로드 중 (랜덤 샘플링)...")
ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess)
print(f"  → {len(ds)}개 샘플 준비 완료")

# 3. GPTQ W8A8 양자화
print(f"\n[3/5] GPTQ {SCHEME} 양자화 시작...")
start_time = time.time()

recipe = [
    GPTQModifier(
        scheme=SCHEME,                      # "W8A8"
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        actorder=ACTORDER,                  # "dynamic"
        dampening_frac=DAMPENING_FRAC,      # 0.01
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

quant_time = time.time() - start_time
print(f"  → 양자화 완료! ({quant_time:.0f}초)")

# 4. 저장
print(f"\n[4/5] 모델 저장 중...")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  → 모델 크기: {total_size / (1024*1024):.1f} MB")

# 5. config.json 최적화 (max_position_embeddings)
config_path = os.path.join(OUT_DIR, "config.json")
print(f"\n[5/5] config.json 최적화 중...")

with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")

# ZIP 생성
zip_name = "submit_w8a8_simple"
shutil.make_archive(
    base_name=f"/kaggle/working/{zip_name}",
    format="zip",
    root_dir="/kaggle/working",
    base_dir="model",
)

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024*1024)

# MLflow
if USE_MLFLOW:
    with mlflow.start_run(run_name="w8a8-simple"):
        mlflow.log_params({
            "scheme": SCHEME,
            "actorder": ACTORDER,
            "dampening_frac": DAMPENING_FRAC,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "samples": NUM_CALIBRATION_SAMPLES,
            "seq_len": MAX_SEQUENCE_LENGTH,
        })
        mlflow.log_metrics({
            "quant_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

print("\n" + "=" * 60)
print(f"✅ {SCHEME} + max_pos={MAX_POSITION_EMBEDDINGS} 완료!")
print(f"""
📊 설정 요약:
   스키마: {SCHEME} (INT8)
   actorder: {ACTORDER}
   dampening: {DAMPENING_FRAC}
   캘리브레이션: MANTA {NUM_CALIBRATION_SAMPLES}개
   max_position_embeddings: {MAX_POSITION_EMBEDDINGS}
   모델 크기: {total_size / (1024*1024):.1f} MB

🔬 이전 점수:
   0.574: W4A16 기본
   0.601: W4A16 + actorder + dampening (이 코드의 베이스)
   0.612: W8A8 기본 (팀원)
   지금:  W8A8 + actorder + dampening + max_pos 최적화!

🚀 {zip_name}.zip 을 DACON에 제출하세요!
""")
