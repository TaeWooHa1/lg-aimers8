"""
GPTQ 양자화 최적화 버전 - Kaggle Notebook 전용 (T4 x 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
1. Kaggle Notebook 생성 (Accelerator: GPU T4 x 2 선택)
2. Internet Access: ON 설정
3. 아래 코드를 셀에 복사하여 실행

특징:
- 00_sample_colab3.py를 기반으로 Kaggle 경로 및 환경에 맞게 수정됨
- 메모리(32GB) 활용을 위해 샘플 수와 시퀀스 길이 상향
"""

# =========================================================
# 0. 패키지 설치 (Kaggle에서 먼저 실행!)
# =========================================================
# !pip install -q llmcompressor transformers datasets accelerate

import os
import torch
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# 1. 경로 설정 (Kaggle용)
# =========================================================
# HuggingFace에서 직접 다운로드
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 출력 폴더 (Kaggle 환경: /kaggle/working)
OUT_DIR = "/kaggle/working/model"

# =========================================================
# 2. 데이터셋 설정
# =========================================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ⭐ Kaggle T4 x 2 (32GB VRAM) 최적화 설정
# Colab 버전을 기반으로, 넉넉한 VRAM을 활용해 정확도를 높임
NUM_CALIBRATION_SAMPLES = 1024  # (Colab 512 -> Kaggle 1024)
MAX_SEQUENCE_LENGTH = 2048      # (Colab 1024 -> Kaggle 2048)

# =========================================================
# 3. 양자화 설정 (최적화)
# =========================================================
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
ACTORDER = "dynamic"           # 과적합 방지, 일반화 성능 우수
DAMPENING_FRAC = 0.01

# =========================================================
# 4. GPU 메모리 정리
# =========================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[INFO] GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] GPU를 찾을 수 없습니다. CPU로 실행됩니다 (매우 느림).")

# =========================================================
# 5. 모델 로드
# =========================================================
print("\n" + "=" * 60)
print(f"[INFO] 모델 다운로드 중... ({MODEL_ID})")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # T4에서는 bfloat16 미지원이므로 float16 사용
    device_map="auto",          # Multi-GPU 자동 분산
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
print("[INFO] GPTQ 양자화 시작 (약 10~20분 소요)")
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
        # sequential_targets=["Exaone4DecoderLayer"], # ⭐ T4 필수 안전장치 (OOM 방지)
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] GPTQ 양자화 완료!")

# =========================================================
# 8. 모델 저장 (Kaggle 경로)
# =========================================================
print(f"\n[INFO] 모델 저장 중... → {OUT_DIR}")

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# 저장 확인
print("[INFO] 저장된 파일:")
for f in os.listdir(OUT_DIR):
    size = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
    print(f"       - {f} ({size:.1f} MB)")

# =========================================================
# 9. ZIP 생성 (Kaggle Output)
# =========================================================
zip_name = "kaggle_optimized_submit"
print(f"\n[INFO] {zip_name}.zip 생성 중...")

# Kaggle Working 디렉토리에 생성
shutil.make_archive(
    base_name=f"/kaggle/working/{zip_name}",
    format="zip",
    root_dir="/kaggle/working",
    base_dir="model",
)

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"[INFO] 생성 완료: {zip_path} ({zip_size:.1f} MB)")

print("\n" + "=" * 60)
print("✅ 작업 완료! Kaggle 우측 'Data' 패널의 Output에서 다운로드하세요.")
print("=" * 60)
