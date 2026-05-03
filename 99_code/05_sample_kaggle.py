"""
GPTQ 양자화 최적화 버전 - Kaggle 전용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
1. Kaggle Notebook에서 GPU 가속기 선택 (T4 x2 권장)
2. 아래 코드를 셀에 복사하여 실행
3. 완료 후 Output 탭에서 optimized_submit.zip 다운로드

참고: Kaggle GPU 환경 (T4 x2, RAM 30GB, VRAM 15GB x2)
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
# HuggingFace에서 직접 다운로드 (약 2.5GB)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 출력 폴더 (Kaggle 환경)
OUT_DIR = "/kaggle/working/model"

# =========================================================
# 2. 데이터셋 설정
# =========================================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ⭐ Kaggle T4 x2 메모리 고려 설정
# - T4 x2 (VRAM 15GB x2, RAM 30GB): 512 샘플, 1024 길이 가능
# - T4 x1 / P100: 256 샘플, 512 길이 권장
NUM_CALIBRATION_SAMPLES = 256   # 메모리 부족 시 256으로 줄이기
MAX_SEQUENCE_LENGTH = 512      # 메모리 부족 시 512으로 줄이기

# =========================================================
# 3. 양자화 설정 (최적화)
# =========================================================
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
ACTORDER = "static"           # 활성화 정렬 ('static', 'group', 'weight', 'dynamic')
BLOCK_SIZE = 64               # 블록 크기 (기본 128, 작을수록 정밀하지만 느림)
DAMPENING_FRAC = 0.01

# =========================================================
# 4. GPU 메모리 정리
# =========================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_count = torch.cuda.device_count()
    print(f"[INFO] GPU 수: {gpu_count}")
    for i in range(gpu_count):
        print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"[INFO] VRAM {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
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
        block_size=BLOCK_SIZE,
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

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# 저장 확인
print("[INFO] 저장된 파일:")
for f in os.listdir(OUT_DIR):
    size = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
    print(f"       - {f} ({size:.1f} MB)")

# =========================================================
# 9. ZIP 생성 (Kaggle Output으로 자동 저장)
# =========================================================
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

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: {zip_path}

🚀 Kaggle Notebook 우측 Output 탭에서 ZIP 파일을 다운로드하여 대회에 제출하세요!
""")
