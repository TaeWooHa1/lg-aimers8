# %%
"""
Sparse + Quantization 통합 파이프라인
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SparseGPTModifier (2:4 구조적 스파스) + GPTQModifier (W4A16) 원샷 적용

입력: KD 학습된 모델 또는 원본 EXAONE-4.0-1.2B (USE_KD_MODEL 플래그로 전환)
출력: /kaggle/working/model/ (제출용 경량화 모델)

리스크 대응:
  ✅ USE_SPARSE 플래그: vLLM 2:4 지원 미확인 시 OFF 가능
  ✅ compressed-tensors 0.13.0 호환 보장
  ✅ 기존 19_w4a16_all_params.py의 모든 최적화 파라미터 유지
  ✅ 원본 모델 fallback (KD 모델 없을 때)

Kaggle 사용법:
  !pip install -q llmcompressor mlflow dagshub
  !pip install -q transformers==4.57.3
"""

# %% [셀 1] 패키지 설치 (Kaggle에서 먼저 실행)
# !pip install -q llmcompressor mlflow dagshub
# !pip install -q transformers==4.57.3

# %% [셀 2] 라이브러리 임포트
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
    mlflow.set_experiment("Sparse-Quant")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

print("[INFO] 라이브러리 로드 완료")

# %% [셀 3] 설정 (⭐ 여기서 파라미터 조정)

# ══════════════════════════════════════════════
# ⭐ 핵심 플래그: KD 모델 사용 여부 + Sparse ON/OFF
# ══════════════════════════════════════════════
USE_KD_MODEL = True                # True: KD 학습된 모델 사용 / False: 원본 모델 사용
USE_SPARSE = True                  # True: 2:4 구조적 스파스 적용 / False: 양자화만

# ── 모델 경로 ──
KD_MODEL_PATH = "/kaggle/working/kd_model"   # KD 학습된 모델 경로
ORIGINAL_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

# ── 모델 선택 ──
if USE_KD_MODEL and os.path.exists(KD_MODEL_PATH):
    MODEL_ID = KD_MODEL_PATH
    MODEL_SOURCE = "KD 학습 모델"
else:
    MODEL_ID = ORIGINAL_MODEL_ID
    MODEL_SOURCE = "원본 모델"
    if USE_KD_MODEL:
        print(f"[WARNING] KD 모델 경로({KD_MODEL_PATH})가 없습니다. 원본 모델로 fallback합니다.")

# ── 캘리브레이션 데이터 ──
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ── 양자화 설정 ──
SCHEME = "W4A16"                      # 4-bit weight, 16-bit activation
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]

# ── 🟢 공짜 정확도 (추론 속도 영향 0) ──
ACTORDER = "dynamic"                  # ⭐ 정확도 ↑↑↑
DAMPENING_FRAC = 0.01                 # 양자화 안정성
NUM_CALIBRATION_SAMPLES = 1024        # ⭐ 정확도 ↑↑↑
MAX_SEQUENCE_LENGTH = 768             # ⭐ 정확도 ↑↑
SEQUENTIAL_TARGETS = ["Exaone4DecoderLayer"]  # ⭐ 레이어별 순차 처리

# ── 🔴 트레이드오프 ──
BLOCK_SIZE = 64                       # ⭐ 정확도 ↑, 속도 약간 ↓

# ── 🔵 공짜 속도 (정확도 영향 0) ──
MAX_POSITION_EMBEDDINGS = 32768       # ⭐ KV Cache 50% 절감

# ── 2:4 스파스 설정 ──
SPARSITY = 0.5                        # 50% 가지치기
MASK_STRUCTURE = "2:4"                # 2:4 구조적 스파스 (L4 하드웨어 가속)

print(f"""
📊 Sparse + Quantization 설정:
┌──────────────────────────────────────────────┐
│ 📦 모델: {MODEL_SOURCE:>30}   │
│    경로: {MODEL_ID[-45:]:>45} │
├──────────────────────────────────────────────┤
│ 🌿 Sparse: {'ON (2:4 구조적)' if USE_SPARSE else 'OFF':>30}   │
│    sparsity: {SPARSITY if USE_SPARSE else 'N/A'}                     │
│    mask_structure: {MASK_STRUCTURE if USE_SPARSE else 'N/A'}                │
├──────────────────────────────────────────────┤
│ 🔢 Quantization: {SCHEME:>28}   │
│ 🟢 actorder: {ACTORDER:>32}   │
│ 🟢 calibration: {NUM_CALIBRATION_SAMPLES:>29}   │
│ 🟢 seq_len: {MAX_SEQUENCE_LENGTH:>33}   │
│ 🟢 sequential_targets: {SEQUENTIAL_TARGETS}        │
│ 🔴 block_size: {BLOCK_SIZE:>30}   │
│ 🔵 max_position_embeddings: {MAX_POSITION_EMBEDDINGS:>18}   │
└──────────────────────────────────────────────┘
""")

# %% [셀 4] GPU 확인
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {gpu_name} ({gpu_vram:.1f}GB)")
else:
    print("[WARNING] GPU 없음!")

# %% [셀 5] 모델 로드
print(f"\n[1/7] 모델 로드: {MODEL_ID} ({MODEL_SOURCE})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터")

# %% [셀 6] 캘리브레이션 데이터 로드 + 전처리
print(f"\n[2/7] 캘리브레이션 데이터 {NUM_CALIBRATION_SAMPLES}개 로드")
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
print(f"  → {len(ds)}개 샘플 준비 완료")

# %% [셀 7] Sparse + Quantization 레시피 구성
print(f"\n[3/7] 레시피 구성")

recipe = []

# ── Step 1: 2:4 구조적 스파스 (선택) ──
if USE_SPARSE:
    from llmcompressor.modifiers.pruning import SparseGPTModifier
    
    sparse_modifier = SparseGPTModifier(
        sparsity=SPARSITY,
        mask_structure=MASK_STRUCTURE,         # ⭐ 2:4 패턴 → L4 Sparse TC 가속
        targets="Linear",
        ignore=["lm_head"],
        sequential_targets=SEQUENTIAL_TARGETS,
    )
    recipe.append(sparse_modifier)
    print(f"  ✅ SparseGPTModifier: sparsity={SPARSITY}, mask={MASK_STRUCTURE}")
else:
    print(f"  ⏭️  Sparse: OFF (양자화만 적용)")

# ── Step 2: W4A16 GPTQ 양자화 ──
quant_modifier = GPTQModifier(
    scheme=SCHEME,
    targets=TARGETS,
    ignore=IGNORE,
    actorder=ACTORDER,
    dampening_frac=DAMPENING_FRAC,
    block_size=BLOCK_SIZE,
    sequential_targets=SEQUENTIAL_TARGETS,
)
recipe.append(quant_modifier)
print(f"  ✅ GPTQModifier: scheme={SCHEME}, actorder={ACTORDER}, block_size={BLOCK_SIZE}")

# %% [셀 8] Oneshot 실행 (⭐ 시간 소요)
print(f"\n[4/7] Oneshot 압축 실행 ({'Sparse + Quant' if USE_SPARSE else 'Quant만'})")
start_time = time.time()

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=True,
)

compress_time = time.time() - start_time
print(f"  → 압축 완료! ({compress_time:.1f}초 = {compress_time/60:.1f}분)")

# %% [셀 9] 모델 저장
print(f"\n[5/7] 모델 저장: {OUT_DIR}")
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

# %% [셀 10] config.json 최적화
print(f"\n[6/7] config.json 최적화")
config_path = os.path.join(OUT_DIR, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

# ── 🔵 공짜 속도: max_position_embeddings 축소 ──
original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS
print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")

# ── transformers_version 맞추기 ──
config["transformers_version"] = "4.57.3"
print(f"  → transformers_version: → 4.57.3")

# ── compressed-tensors version 확인 ──
quant_config = config.get("quantization_config", {})
if "version" in quant_config:
    print(f"  → quantization_config.version: {quant_config['version']}")
    if quant_config["version"] != "0.13.0":
        quant_config["version"] = "0.13.0"
        print(f"  → ⚠️ version을 0.13.0으로 수정!")

# ── sparsity_config 확인 ──
if "sparsity_config" in quant_config:
    sp_config = quant_config["sparsity_config"]
    print(f"  → sparsity_config: structure={sp_config.get('sparsity_structure', 'N/A')}, "
          f"sparsity={sp_config.get('global_sparsity', 'N/A')}")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → config.json 저장 완료!")

# %% [셀 11] ZIP 생성
print(f"\n[7/7] ZIP 생성")

strategy = "kd_sparse_quant" if USE_KD_MODEL and USE_SPARSE else \
           "kd_quant" if USE_KD_MODEL else \
           "sparse_quant" if USE_SPARSE else "quant_only"

zip_name = f"submit_{strategy}"
shutil.make_archive(f"/kaggle/working/{zip_name}", "zip", "/kaggle/working", "model")

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"  → {zip_path} ({zip_size:.1f} MB)")

if zip_size > 10 * 1024:
    print(f"  ⚠️ ZIP 용량이 10GB를 초과합니다! ({zip_size/1024:.1f}GB)")
else:
    print(f"  ✅ ZIP 용량 OK ({zip_size/1024:.2f}GB < 10GB)")

# %% [셀 12] MLflow 기록

if USE_MLFLOW:
    with mlflow.start_run(run_name=f"{strategy}-bs{BLOCK_SIZE}"):
        mlflow.log_params({
            "model_source": MODEL_SOURCE,
            "model_id": MODEL_ID,
            "use_kd_model": USE_KD_MODEL,
            "use_sparse": USE_SPARSE,
            "scheme": SCHEME,
            "actorder": ACTORDER,
            "block_size": BLOCK_SIZE,
            "dampening_frac": DAMPENING_FRAC,
            "calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_seq_length": MAX_SEQUENCE_LENGTH,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "sparsity": SPARSITY if USE_SPARSE else 0,
            "mask_structure": MASK_STRUCTURE if USE_SPARSE else "none",
        })
        mlflow.log_metrics({
            "compression_time_sec": compress_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })
        print("[INFO] MLflow 기록 완료!")

# %% [셀 13] 완료 요약
print(f"""
{'='*60}
✅ {'KD + ' if USE_KD_MODEL else ''}{'Sparse + ' if USE_SPARSE else ''}Quantization 완료!
{'='*60}

📊 전략:
  모델 소스: {MODEL_SOURCE}
  Sparse: {'2:4 구조적 (50%)' if USE_SPARSE else 'OFF'}
  Quant: {SCHEME} (actorder={ACTORDER}, block_size={BLOCK_SIZE})
  max_position_embeddings: {MAX_POSITION_EMBEDDINGS}
  
📦 결과:
  모델 크기: {total_size / (1024*1024):.1f} MB
  ZIP 크기: {zip_size:.1f} MB
  압축 시간: {compress_time:.1f}초
  
📁 제출 파일: {zip_path}

🎯 예상 점수 (이론적):
  {'KD+Sparse+W4A16: 0.5 × 1.05 + 0.5 × 0.65 = 0.85' if USE_KD_MODEL and USE_SPARSE else
   'KD+W4A16: 0.5 × 1.07 + 0.5 × 0.51 = 0.79' if USE_KD_MODEL else
   'Sparse+W4A16: 0.5 × 1.00 + 0.5 × 0.60 = 0.80' if USE_SPARSE else
   'W4A16: 0.5 × 1.03 + 0.5 × 0.51 = 0.77'}

⚠️ vLLM에서 Sparse 가속이 작동하지 않으면:
  USE_SPARSE = False 로 변경하여 양자화만 적용하세요.
""")
