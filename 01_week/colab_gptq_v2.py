# ============================================================
# Google Colab용 GPTQ 양자화 스크립트 (고품질 버전)
# 실행 방법: Colab에서 셀 단위로 실행
# ============================================================

# ============================================================
# 셀 1: 라이브러리 설치 (이 셀을 먼저 실행!)
# ============================================================
"""
!pip install -q llmcompressor datasets transformers accelerate
"""

# ============================================================
# 셀 2: 양자화 코드 (아래 코드를 복사하여 실행)
# ============================================================

import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ============================================================
# 설정 (조정 가능한 파라미터들)
# ============================================================

# Hugging Face에서 모델 다운로드 (Colab용)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# 캘리브레이션 설정 (품질에 영향)
NUM_CALIBRATION_SAMPLES = 512    # 기본 256, 높을수록 품질 ↑
MAX_SEQUENCE_LENGTH = 1024       # 기본 512, 높을수록 긴 문맥 고려

# 양자화 설정
SCHEME = "W4A16"                 # 가중치 4bit, 활성화 16bit
TARGETS = ["Linear"]             # Linear 레이어만 양자화
IGNORE = ["embed_tokens", "lm_head"]  # 입출력 레이어 제외

# GPTQ 고급 설정
DAMPENING_FRAC = 0.01            # 댐핑 비율 (0.001~0.1)
BLOCK_SIZE = 64                  # 블록 크기 (32, 64, 128)

# ============================================================
# GPU 확인
# ============================================================

print(f"[INFO] CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# 1. 모델/토크나이저 로드
# ============================================================

print("\n[INFO] 모델 로드 중... (약 2~3분 소요)")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # GPU 자동 할당
)

print("[INFO] 모델/토크나이저 로드 완료")

# ============================================================
# 2. 캘리브레이션 데이터 준비
# ============================================================

print("\n[INFO] 캘리브레이션 데이터 로드 중...")

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

print(f"[INFO] 데이터 전처리 완료 (샘플 수: {len(ds)})")

# ============================================================
# 3. GPTQ 양자화 실행
# ============================================================

print(f"\n[INFO] GPTQ 양자화 시작...")
print(f"  - scheme: {SCHEME}")
print(f"  - samples: {NUM_CALIBRATION_SAMPLES}")
print(f"  - max_len: {MAX_SEQUENCE_LENGTH}")
print(f"  - block_size: {BLOCK_SIZE}")
print(f"  - 예상 소요 시간: 약 15~25분")

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

print("[INFO] GPTQ 양자화 완료!")

# ============================================================
# 4. 모델 저장
# ============================================================

print(f"\n[INFO] 모델 저장 중: {OUT_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print("[INFO] 모델/토크나이저 저장 완료")

# ============================================================
# 5. chat_template.jinja 다운로드
# ============================================================

print("\n[INFO] chat_template.jinja 다운로드 중...")

chat_template_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="chat_template.jinja"
)
shutil.copy(chat_template_path, os.path.join(OUT_DIR, "chat_template.jinja"))

print("  ✓ chat_template.jinja 복사 완료")

# ============================================================
# 6. 파일 검증
# ============================================================

print("\n[INFO] 생성된 파일 목록:")
expected_files = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
]

all_ok = True
for fname in expected_files:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        print(f"  ✓ {fname} ({size:,} bytes)")
    else:
        print(f"  ✗ {fname} - 누락!")
        all_ok = False

# ============================================================
# 7. 제출 파일 압축
# ============================================================

zip_name = "gptq_submit_v2"
print(f"\n[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

zip_size = os.path.getsize(f"{zip_name}.zip")
print(f"[INFO] 완료: {zip_name}.zip ({zip_size / (1024**3):.2f} GB)")

# ============================================================
# 8. 다운로드 (Colab 전용)
# ============================================================

print("\n[INFO] 다운로드를 시작합니다...")

from google.colab import files
files.download(f"{zip_name}.zip")

print("\n" + "="*50)
print("[SUCCESS] GPTQ 제출 파일 생성 완료!")
print("="*50)
print(f"\n설정 요약:")
print(f"  - 양자화: {SCHEME}")
print(f"  - 캘리브레이션 샘플: {NUM_CALIBRATION_SAMPLES}")
print(f"  - 시퀀스 길이: {MAX_SEQUENCE_LENGTH}")
print(f"  - 블록 크기: {BLOCK_SIZE}")
