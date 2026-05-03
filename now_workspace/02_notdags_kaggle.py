"""
GPTQ 양자화 사이트 제출용 - Kaggle 전용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
1. Kaggle Notebook에서 GPU 가속기 (T4 x 2 등) 선택
2. Settings -> Internet -> On (인터넷 활성화 필수)
3. 아래 코드를 셀에 복사하여 실행
"""

# =========================================================
# 0. 패키지 설치 (Kaggle에서 먼저 실행!)
# =========================================================
# !pip install -q torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
# !pip install -q transformers==4.57.3 compressed-tensors==0.13.0 safetensors==0.7.0 accelerate==1.10.1 datasets==4.4.1 huggingface-hub==0.36.0 tokenizers==0.22.1 sentencepiece==0.2.1 llmcompressor

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
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 출력 폴더 (Kaggle 작업 디렉토리)
OUT_DIR = "/kaggle/working/model"

# =========================================================
# 2. 데이터셋 설정
# =========================================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# Kaggle T4 x 2 환경 고려
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# =========================================================
# 3. 양자화 설정 (제출용 최적화)
# =========================================================
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
ACTORDER = "dynamic"           # 'static'보다 빠르고 준수한 성능
DAMPENING_FRAC = 0.01

# =========================================================
# 4. GPU 메모리 정리
# =========================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[WARNING] GPU를 찾을 수 없습니다.")

# =========================================================
# 5. 모델 로드
# =========================================================
print(f"\n[INFO] 모델 로드 중... ({MODEL_ID})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"[INFO] 모델 로드 완료!")

# =========================================================
# 6. 데이터셋 로드 & 전처리
# =========================================================
print(f"\n[INFO] 데이터셋 로드 중... ({DATASET_ID})")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

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
print(f"\n[INFO] GPTQ 양자화 시작 (Scheme: {SCHEME}, ActOrder: {ACTORDER})")

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        actorder=ACTORDER,
        dampening_frac=DAMPENING_FRAC,
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
# 8. 모델 저장
# =========================================================
print(f"\n[INFO] 모델 저장 중... → {OUT_DIR}")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# =========================================================
# 9. ZIP 생성 (제출용)
# =========================================================
zip_name = "optimized_submit"
print(f"\n[INFO] {zip_name}.zip 생성 중...")

# Kaggle Output (/kaggle/working)에 바로 zip 생성
# root_dir=/kaggle/working, base_dir=model -> /kaggle/working/model 폴더를 압축
shutil.make_archive(
    base_name=f"/kaggle/working/{zip_name}",
    format="zip",
    root_dir="/kaggle/working",
    base_dir="model",
)

zip_file_path = f"/kaggle/working/{zip_name}.zip"
print(f"[INFO] 생성 완료: {zip_file_path}")
print(f"       크기: {os.path.getsize(zip_file_path) / (1024*1024):.2f} MB")

print("\n" + "=" * 60)
print("✅ 완료! 오른쪽 [Data] 패널의 Output에서 ZIP 파일을 다운로드하세요.")
print("=" * 60)
