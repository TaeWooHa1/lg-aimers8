# ============================================================
# Google Colab용 GPTQ 양자화 스크립트
# 기반: [Baseline,_LB 0.5±0.02] GPTQ 기반 EXAONE-4.0-1.2B 모델 양자화.ipynb
# ============================================================

# ============================================================
# 셀 1: 라이브러리 설치
# ============================================================
"""
# Colab에서 이 코드를 먼저 실행하세요!
!pip install -q llmcompressor datasets transformers accelerate
"""

# ============================================================
# 셀 2: Import
# ============================================================

import os
import torch
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ============================================================
# 셀 3: Setting
# ============================================================

# Colab용: Hugging Face에서 모델 다운로드
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# Quantization
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]


# GPTQ 고급 설정
DAMPENING_FRAC = 0.01 
BLOCK_SIZE = 64       

# ============================================================
# 셀 4: Model Loads
# ============================================================

print("[INFO] 모델 로드 중...")

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

print("[INFO] 모델/토크나이저 로드 완료")

# ============================================================
# 셀 5: Dataset Loads & Preprocess
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
            tokenize=False)
    }

ds = ds.map(preprocess)

print("[INFO] 데이터 전처리 완료")

# ============================================================
# 셀 6: GPTQ Quantization
# ============================================================

print(f"[INFO] GPTQ 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})...")

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        dampening_frac=DAMPENING_FRAC,
        block_size=BLOCK_SIZE,  
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
# 셀 7: Model Save
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# chat_template.jinja 다운로드 및 복사
chat_template_path = hf_hub_download(repo_id=MODEL_ID, filename="chat_template.jinja")
shutil.copy(chat_template_path, os.path.join(OUT_DIR, "chat_template.jinja"))

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

# ============================================================
# 셀 8: Submission
# ============================================================

zip_name = "baseline_submit2"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] 생성 완료: {zip_name}.zip")

# ============================================================
# 셀 9: Download (Colab 전용)
# ============================================================

from google.colab import files
files.download(f"{zip_name}.zip")

print("[SUCCESS] 다운로드 시작!")
