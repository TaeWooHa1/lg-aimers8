# %%
"""
W8A8 GPTQ 양자화 - Kaggle 전용 버전
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
파라미터가이드.md 기반으로 모든 최적화 파라미터 적용

W4A16 vs W8A8 차이:
  - Weight: 4bit → 8bit (정밀도 ↑, 모델 크기 ↑)
  - Activation: 16bit → 8bit (추론 속도 ↑, INT8 Tensor Core 활용)
  - T4 GPU의 INT8 Tensor Core를 활용하여 추론 속도 향상

적용된 최적화:
  🟢 공짜 정확도 (추론 속도 영향 0):
    - actorder: static → dynamic
    - num_calibration_samples: 256 → 1024
    - max_seq_length: 512 → 1024
    - shuffle_calibration_samples: True
    - sequential_targets: ["Exaone4DecoderLayer"]
  🔵 공짜 속도 (정확도 영향 0):
    - max_position_embeddings: 65536 → 32768 (KV Cache 50% 절감)
  🔴 트레이드오프 (정확도↑ 속도↓):
    - block_size: 128 → 64

Kaggle 사용법:
  !pip install -q llmcompressor mlflow dagshub
  !pip install -q transformers==4.57.3
"""

# %% [셀 1] 패키지 설치 (Kaggle에서 먼저 실행)
# !pip install -q llmcompressor mlflow dagshub
# !pip install -q transformers==4.57.3

# %% [셀 2] 라이브러리 임포트 + MLflow 설정
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
    mlflow.set_experiment("W8A8-kaggle")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

print("[INFO] 라이브러리 로드 완료")

# %% [셀 3] 설정 (⭐ 여기서 파라미터 조정)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ── 양자화 설정 ──
SCHEME = "W8A8"                       # ⭐ 8-bit weight + 8-bit activation
TARGETS = ["Linear"]
IGNORE = [
    "embed_tokens", "lm_head"           # 임베딩/출력 레이어 보호
]

# ── 🟢 공짜 정확도 (추론 속도 영향 0) ──
ACTORDER = "dynamic"                  # ⭐ static→dynamic (공짜 정확도 ↑↑↑)
DAMPENING_FRAC = 0.01                 # 양자화 안정성
NUM_CALIBRATION_SAMPLES = 1024        # ⭐ 256→1024 (공짜 정확도 ↑↑↑)
MAX_SEQUENCE_LENGTH = 1024            # ⭐ 512→1024 (데이터 보존율 86%)
SEQUENTIAL_TARGETS = ["Exaone4DecoderLayer"]  # ⭐ 레이어별 순차 처리 (공짜 정확도 ↑)

# ── 🔴 트레이드오프 (정확도↑ 속도 약간↓) ──
BLOCK_SIZE = 64                       # ⭐ 128→64 (정확도 ↑, 속도 영향 미미)

# ── 🔵 공짜 속도 (정확도 영향 0) ──
MAX_POSITION_EMBEDDINGS = 32768       # ⭐ 65536→32768 (KV Cache 50% 절감!)

# ── 🔧 서버 호환 ──
TARGET_CT_VERSION = "0.13.0"          # 평가 서버 compressed-tensors 버전

print(f"""
📊 W8A8 전체 파라미터 설정 확인:
┌──────────────────────────────────────────────┐
│ ⭐ W8A8 (Weight 8bit + Activation 8bit)      │
│   → INT8 Tensor Core 활용 (T4 지원 ✅)       │
├──────────────────────────────────────────────┤
│ 🟢 공짜 정확도 (추론 속도 영향 0)              │
│   actorder={ACTORDER}                        │
│   calibration={NUM_CALIBRATION_SAMPLES}      │
│   seq_len={MAX_SEQUENCE_LENGTH}              │
│   sequential_targets={SEQUENTIAL_TARGETS}    │
│   dampening_frac={DAMPENING_FRAC}            │
├──────────────────────────────────────────────┤
│ 🔴 트레이드오프 (정확도↑ 속도↓)                │
│   block_size={BLOCK_SIZE}                    │
│   ignore={IGNORE}                            │
├──────────────────────────────────────────────┤
│ 🔵 공짜 속도 (정확도 영향 0)                   │
│   max_position_embeddings={MAX_POSITION_EMBEDDINGS} │
├──────────────────────────────────────────────┤
│ 🔧 서버 호환                                  │
│   compressed-tensors version={TARGET_CT_VERSION}    │
└──────────────────────────────────────────────┘
""")

# %% [셀 4] GPU 확인
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {gpu_name} ({gpu_vram:.1f}GB)")
else:
    print("[WARNING] GPU 없음")

# %% [셀 5] 모델 로드
print(f"\n[1/8] 모델 로드: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,       # ⭐ T4 네이티브 (bfloat16 에뮬레이션 오버헤드 제거)
    device_map={"": 0},              # ⭐ GPU 0만 사용 (GPU 1 해제 → CPU RAM 절약)
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터")

# %% [셀 6] 캘리브레이션 데이터 로드 + 전처리
print(f"\n[2/8] 캘리브레이션 데이터 {NUM_CALIBRATION_SAMPLES}개 로드 (랜덤 샘플링)")
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False
        )
    }

ds = ds.map(preprocess)
print(f"  → {len(ds)}개 샘플 준비 완료")

# %% [셀 7] GPTQ 양자화 실행 (⭐ 핵심 — 시간 소요)
print(f"\n[3/8] GPTQ W8A8 양자화 시작 (전체 파라미터 최적화)")
print(f"  ⭐ scheme={SCHEME} (Weight 8bit + Activation 8bit)")
print(f"  🟢 actorder={ACTORDER}, dampening={DAMPENING_FRAC}")
print(f"  🟢 calibration={NUM_CALIBRATION_SAMPLES}, seq_len={MAX_SEQUENCE_LENGTH}")
print(f"  🟢 sequential_targets={SEQUENTIAL_TARGETS}")
print(f"  🔴 block_size={BLOCK_SIZE}")
print(f"  🔴 ignore={IGNORE}")

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        actorder=ACTORDER,
        dampening_frac=DAMPENING_FRAC,
        block_size=BLOCK_SIZE,
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
    shuffle_calibration_samples=True,             # ⭐ 데이터 셔플 (공짜 정확도 ↑)
)

quant_time = time.time() - start_time
print(f"  → 양자화 완료! ({quant_time:.1f}초)")

# %% [셀 8] 모델 저장
print(f"\n[4/8] 모델 저장: {OUT_DIR}")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)  # ⭐ 압축 저장
tokenizer.save_pretrained(OUT_DIR)

# 저장 확인
total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  → 모델 크기: {total_size / (1024*1024):.1f} MB")

print("[INFO] 저장된 파일:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size:.1f} MB)")

# %% [셀 9] config.json 최적화 + 서버 호환 패치
print(f"\n[5/8] config.json 최적화 (🔵 공짜 속도 + 🔧 서버 호환)")
config_path = os.path.join(OUT_DIR, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

# 🔵 KV Cache 최적화
original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS
print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")
print(f"  → KV Cache 메모리 {(1 - MAX_POSITION_EMBEDDINGS/65536)*100:.0f}% 절감!")

# 🔧 compressed-tensors version 패치 (평가 서버 호환)
if "quantization_config" in config:
    original_version = config["quantization_config"].get("version", "N/A")
    config["quantization_config"]["version"] = TARGET_CT_VERSION
    print(f"  → compressed-tensors version: {original_version} → {TARGET_CT_VERSION} ✅")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("  → config.json 패치 완료!")

# %% [셀 10] ZIP 생성
print(f"\n[6/8] ZIP 생성")
zip_name = "w8a8_kaggle"
shutil.make_archive(f"/kaggle/working/{zip_name}", "zip", "/kaggle/working", "model")

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"  → {zip_path} ({zip_size:.1f} MB)")

# %% [셀 11] MLflow 기록 (선택)
print(f"\n[7/8] MLflow 기록")
if USE_MLFLOW:
    with mlflow.start_run(run_name=f"W8A8-bs{BLOCK_SIZE}-cal{NUM_CALIBRATION_SAMPLES}"):
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
        print("  → MLflow 기록 완료!")
else:
    print("  → MLflow 미사용")

# %% [셀 12] 완료 요약
print("\n" + "=" * 60)
print("✅ W8A8 전체 파라미터 최적화 양자화 완료!")
print("=" * 60)
print(f"""
📊 적용된 최적화 (W8A8 Kaggle 버전):
 ┌─ ⭐ W8A8 핵심 ──────────────────────────────┐
 │  scheme: W8A8 (Weight 8bit + Activation 8bit)│
 │  → INT8 Tensor Core 활용 (속도 ↑)            │
 ├─ 🟢 공짜 정확도 (추론 속도 영향 0) ──────────┤
 │  actorder: static → {ACTORDER} ⭐           │
 │  calibration: 256 → {NUM_CALIBRATION_SAMPLES} ⭐             │
 │  seq_len: 512 → {MAX_SEQUENCE_LENGTH} ⭐                │
 │  sequential_targets: {SEQUENTIAL_TARGETS} ⭐ │
 │  shuffle_calibration_samples: True ⭐       │
 ├─ 🔴 트레이드오프 (정확도↑ 속도↓) ────────────┤
 │  block_size: 128 → {BLOCK_SIZE} ⭐              │
 │  ignore: embed_tokens, lm_head ⭐           │
 ├─ 🔵 공짜 속도 (정확도 영향 0) ───────────────┤
 │  max_position_embeddings: 65536 → {MAX_POSITION_EMBEDDINGS} ⭐ │
 ├─ 🔧 서버 호환 ──────────────────────────────┤
 │  compressed-tensors: → {TARGET_CT_VERSION} ⭐         │
 └──────────────────────────────────────────────┘

⏱️ 양자화 시간: {quant_time:.1f}초
📁 {zip_name}.zip ({zip_size:.1f} MB) 을 대회에 제출하세요!
""")
