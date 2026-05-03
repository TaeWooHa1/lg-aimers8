# ============================================================
# Google Colab용 AWQ 양자화 스크립트
# ============================================================

# ============================================================
# 셀 1: 라이브러리 설치 (이 셀을 먼저 실행!)
# ============================================================
"""
!pip install -q autoawq transformers accelerate
"""

# ============================================================
# 셀 2: 양자화 실행 (아래 코드를 복사하여 실행)
# ============================================================

import os
import shutil
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# ============================================================
# 설정
# ============================================================

# Hugging Face에서 모델 다운로드 (Colab용)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "./model"

# AWQ 양자화 설정
quant_config = {
    "zero_point": True,      # 비대칭 양자화 (정확도 ↑)
    "q_group_size": 128,     # 그룹 크기
    "w_bit": 4,              # 4bit 양자화
    "version": "GEMM"        # GPU 연산 최적화
}

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

model = AutoAWQForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    safetensors=True,
)

print("[INFO] 모델/토크나이저 로드 완료")

# ============================================================
# 2. AWQ 양자화 실행
# ============================================================

print(f"\n[INFO] AWQ 양자화 시작...")
print(f"  - w_bit: {quant_config['w_bit']}")
print(f"  - q_group_size: {quant_config['q_group_size']}")
print(f"  - 예상 소요 시간: 약 5~10분")

model.quantize(tokenizer, quant_config=quant_config)

print("[INFO] AWQ 양자화 완료!")

# ============================================================
# 3. 모델 저장
# ============================================================

print(f"\n[INFO] 모델 저장 중: {OUT_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)

# AWQ 모델 저장
model.save_quantized(OUT_DIR, safetensors=True)
tokenizer.save_pretrained(OUT_DIR)

print("[INFO] 모델/토크나이저 저장 완료")

# ============================================================
# 4. chat_template.jinja 다운로드
# ============================================================

print("\n[INFO] chat_template.jinja 다운로드 중...")

chat_template_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="chat_template.jinja"
)
shutil.copy(chat_template_path, os.path.join(OUT_DIR, "chat_template.jinja"))

print("  ✓ chat_template.jinja 복사 완료")

# ============================================================
# 5. recipe.yaml 생성
# ============================================================

recipe_content = f"""default_stage:
  default_modifiers:
    AWQModifier:
      zero_point: {str(quant_config['zero_point']).lower()}
      q_group_size: {quant_config['q_group_size']}
      w_bit: {quant_config['w_bit']}
      version: {quant_config['version']}
"""

recipe_path = os.path.join(OUT_DIR, "recipe.yaml")
with open(recipe_path, "w", encoding="utf-8") as f:
    f.write(recipe_content)
print("  ✓ recipe.yaml 생성 완료")

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
    "recipe.yaml",
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

zip_name = "awq_submit"
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
print("[SUCCESS] AWQ 제출 파일 생성 완료!")
print("="*50)
