# ============================================================
# Google Colab용 GPTQ 양자화 스크립트
# 실행 방법: Colab에서 이 코드를 복사하여 실행
# ============================================================

# 1. 라이브러리 설치 (Colab에서 첫 번째 셀로 실행)
"""
!pip install -q llmcompressor datasets transformers accelerate
"""

# 2. 양자화 코드 (두 번째 셀로 실행)

import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ============================================================
# 설정
# ============================================================

# Hugging Face에서 직접 모델 다운로드 (Colab용)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"  # Hugging Face 모델 ID
OUT_DIR = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# GPTQ 양자화 설정
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]

# ============================================================
# GPU 확인
# ============================================================

print(f"[INFO] CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 1. 모델/토크나이저 로드
# ============================================================

print("[INFO] 모델 로드 중... (약 2~3분 소요)")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Colab GPU 자동 사용
)

print("[INFO] 모델/토크나이저 로드 완료")

# ============================================================
# 2. 캘리브레이션 데이터 준비
# ============================================================

print("[INFO] 캘리브레이션 데이터 로드 중...")

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

print("[INFO] 데이터 전처리 완료")

# ============================================================
# 3. GPTQ 양자화 실행
# ============================================================

print(f"[INFO] GPTQ 시작 (약 10~15분 소요)...")

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] GPTQ 완료")

# ============================================================
# 4. 모델 저장
# ============================================================

print(f"[INFO] 모델 저장 중: {OUT_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# ============================================================
# 5. chat_template.jinja 다운로드 및 복사
# ============================================================

print("[INFO] chat_template.jinja 다운로드 중...")

from huggingface_hub import hf_hub_download

chat_template_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="chat_template.jinja"
)
shutil.copy(chat_template_path, os.path.join(OUT_DIR, "chat_template.jinja"))

print("[INFO] chat_template.jinja 복사 완료")

# ============================================================
# 6. 제출 파일 압축
# ============================================================

zip_name = "baseline_submit"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

zip_size = os.path.getsize(f"{zip_name}.zip")
print(f"[INFO] 완료: {zip_name}.zip ({zip_size / (1024**3):.2f} GB)")

# ============================================================
# 7. 다운로드 링크 생성 (Colab 전용)
# ============================================================

from google.colab import files
files.download(f"{zip_name}.zip")

print("[SUCCESS] 제출 파일 생성 완료! 다운로드가 시작됩니다.")
