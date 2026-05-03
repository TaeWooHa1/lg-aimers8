# %%
"""
W4A16 GPTQ 양자화 - 정확도 최대 보존 버전
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전략: W4A16에서 정확도를 최대한 유지하기 위해
      "추론 속도에 영향 없는 파라미터"를 모두 최대로 설정

핵심 설정:
  - scheme: W4A16 (4-bit weight, 16-bit activation)
  - actorder: dynamic (공짜 정확도 ↑, 추론 속도 영향 0)
  - num_calibration_samples: 1024 (공짜 정확도 ↑, 추론 속도 영향 0)
  - max_seq_length: 768 (공짜 정확도 ↑, 추론 속도 영향 0)
  - block_size: 32 (정확도 ↑, 추론 속도 영향 미미)
  - dampening_frac: 0.01 (양자화 안정성)
  - max_position_embeddings: 32768 (KV Cache 50% 절감 → 속도 ↑)
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
    mlflow.set_experiment("W4A16-accuracy-max")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

print("[INFO] 라이브러리 로드 완료")

# %% [셀 3] 설정 (⭐ 여기서 파라미터 조정)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ── 양자화 설정 (정확도 최대 보존) ──
SCHEME = "W4A16"                      # 4-bit weight, 16-bit activation
TARGETS = ["Linear"]
IGNORE = [
    "embed_tokens", "lm_head",           # 임베딩/출력 레이어 보호
    "model.layers.0",                     # ⭐ 첫 번째 디코더 레이어 (입력 표현 형성)
    "model.layers.29",                    # ⭐ 마지막 디코더 레이어 (출력 직전, 정확도 민감)
]

# ── 추론 속도 영향 0인 파라미터 → 최대로! ──
ACTORDER = "dynamic"                  # ⭐ 활성화 순서 기반 양자화 (공짜 정확도 ↑)
DAMPENING_FRAC = 0.01                 # 양자화 안정성
NUM_CALIBRATION_SAMPLES = 1024        # ⭐ 캘리브레이션 샘플 수 (공짜 정확도 ↑)
MAX_SEQUENCE_LENGTH = 768             # ⭐ 캘리브레이션 시퀀스 길이 (공짜 정확도 ↑)

# ── 추론 속도 약간 영향 있는 파라미터 → 적당히 ──
BLOCK_SIZE = 32                       # ⭐ 양자화 그룹 크기 128→32 (정확도 ↑, 속도 영향 미미)

# ── config.json 최적화 (속도 ↑) ──
MAX_POSITION_EMBEDDINGS = 32768       # 65536 → 32768 (KV Cache 50% 절감)

print(f"""
📊 설정 확인:
   scheme={SCHEME}, actorder={ACTORDER}
   block_size={BLOCK_SIZE}, dampening={DAMPENING_FRAC}
   calibration={NUM_CALIBRATION_SAMPLES}, seq_len={MAX_SEQUENCE_LENGTH}
   ignore={IGNORE}
   max_position_embeddings={MAX_POSITION_EMBEDDINGS}
""")

# %% [셀 4] GPU 확인
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {gpu_name} ({gpu_vram:.1f}GB)")
else:
    print("[WARNING] GPU 없음")
# ======================================================================

# %% [셀 5] 모델 로드
print(f"[INFO] 모델 로드: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터")

# %% [셀 6] 캘리브레이션 데이터 로드 + 전처리
print(f"[INFO] 캘리브레이션 데이터 {NUM_CALIBRATION_SAMPLES}개 로드")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False
        )
    }
# ========================================================================

ds = ds.map(preprocess)
print(f"  → {len(ds)}개 샘플 준비 완료")

# %% [셀 7] GPTQ 양자화 실행 (⭐ 핵심 — 시간 소요)
print(f"[INFO] GPTQ 양자화 시작 (정확도 최대 보존 설정)")
print(f"  scheme={SCHEME}, actorder={ACTORDER}")
print(f"  block_size={BLOCK_SIZE}, dampening={DAMPENING_FRAC}")
print(f"  calibration={NUM_CALIBRATION_SAMPLES}, seq_len={MAX_SEQUENCE_LENGTH}")

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        actorder=ACTORDER,
        dampening_frac=DAMPENING_FRAC,
        block_size=BLOCK_SIZE,
    )
]

start_time = time.time()

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

quant_time = time.time() - start_time
print(f"  → 양자화 완료! ({quant_time:.1f}초)")

# %% [셀 8] 모델 저장
print(f"[INFO] 모델 저장: {OUT_DIR}")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
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

# %% [셀 9] config.json 최적화 (max_position_embeddings 축소)
print(f"[INFO] config.json 최적화")
config_path = os.path.join(OUT_DIR, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")
print(f"  → KV Cache 메모리 {(1 - MAX_POSITION_EMBEDDINGS/65536)*100:.0f}% 절감!")

# %% [셀 10] ZIP 생성
print(f"[INFO] ZIP 생성")
zip_name = "w4a16_accuracy_max"
shutil.make_archive(f"/kaggle/working/{zip_name}", "zip", "/kaggle/working", "model")

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"  → {zip_path} ({zip_size:.1f} MB)")
# ===========================================================================
# %% [셀 11] MLflow 기록 (선택)
if USE_MLFLOW:
    with mlflow.start_run(run_name=f"W4A16-actdyn-bs{BLOCK_SIZE}-cal{NUM_CALIBRATION_SAMPLES}"):
        mlflow.log_params({
            "model_id": MODEL_ID,
            "scheme": SCHEME,
            "actorder": ACTORDER,
            "block_size": BLOCK_SIZE,
            "dampening_frac": DAMPENING_FRAC,
            "calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_seq_length": MAX_SEQUENCE_LENGTH,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "ignore": str(IGNORE),
        })
        mlflow.log_metrics({
            "quantization_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })
        print("[INFO] MLflow 기록 완료!")
else:
    print("[INFO] MLflow 미사용")

# %% [셀 12] 완료 요약
print("\n" + "=" * 60)
print("✅ W4A16 정확도 최대 보존 양자화 완료!")
print("=" * 60)
print(f"""
📊 설정 (vs 기존 0.57점 코드):
   • scheme: W4A16 (동일)
   • actorder: None → dynamic ⭐ (공짜 정확도 ↑)
   • block_size: 128 → {BLOCK_SIZE} ⭐ (정확도 ↑)
   • calibration: 256 → {NUM_CALIBRATION_SAMPLES} ⭐ (공짜 정확도 ↑)
   • seq_len: 512 → {MAX_SEQUENCE_LENGTH} ⭐ (공짜 정확도 ↑)
   • ignore: +layers.0, +layers.29 ⭐ (정확도 민감 레이어 보호)
   • max_position_embeddings: 65536 → {MAX_POSITION_EMBEDDINGS} (속도 ↑)

🎯 전략: 추론 속도 영향 없는 파라미터를 최대로 → 정확도 극대화
📁 w4a16_accuracy_max.zip 을 대회에 제출하세요!
""")
