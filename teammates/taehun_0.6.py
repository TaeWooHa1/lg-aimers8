"""
GPTQ 최적화 실험 3종 - 0.574 → 0.60 돌파!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0.574 모델 분석:
  - samples=256, seq_len=512, actorder=없음, dampening=없음
  - → 기본값만 사용, 개선 여지 큼!

이 스크립트의 3가지 실험:
  A) 캘리브레이션 강화 (samples↑, seq_len↑)
  B) A + actorder="dynamic" + dampening_frac=0.01
  C) A + B + ignore=["lm_head"]만 (embed_tokens 양자화 포함)

사용법:
  Kaggle에서 EXPERIMENT 변수를 "A", "B", "C"로 바꿔서 실행.
  각각 제출하여 점수 비교.
"""

import os
import torch
import shutil
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# ⭐ 실험 선택: "A", "B", "C" 중 하나 입력
# =========================================================
EXPERIMENT = "B"  # ← 이것만 바꿔서 실행!

# =========================================================
# 실험 설정
# =========================================================
EXPERIMENTS = {
    # A: 캘리브레이션 강화만 (가장 안전)
    "A": {
        "name": "calib_boost",
        "num_calibration_samples": 512,    # 256 → 512 (T4 안전)
        "max_sequence_length": 1024,       # 512 → 1024 (T4 안전)
        "scheme": "W4A16",
        "targets": ["Linear"],
        "ignore": ["embed_tokens", "lm_head"],  # 기존과 동일
        "gptq_kwargs": {},  # 추가 설정 없음
    },
    # B: 캘리브레이션 + actorder + dampening (추천!)
    "B": {
        "name": "actorder_dynamic",
        "num_calibration_samples": 512,    # T4 안전
        "max_sequence_length": 1024,       # T4 안전
        "scheme": "W4A16",
        "targets": ["Linear"],
        "ignore": ["embed_tokens", "lm_head"],
        "gptq_kwargs": {
            "actorder": "dynamic",      # ⭐ 활성화 순서 기반 양자화
            "dampening_frac": 0.01,     # ⭐ 양자화 안정성
        },
    },
    # C: B + ignore 축소 (embed_tokens도 양자화)
    "C": {
        "name": "aggressive",
        "num_calibration_samples": 512,    # T4 안전
        "max_sequence_length": 1024,       # T4 안전
        "scheme": "W4A16",
        "targets": ["Linear"],
        "ignore": ["lm_head"],  # ⭐ embed_tokens 제거 → 추가 압축
        "gptq_kwargs": {
            "actorder": "dynamic",
            "dampening_frac": 0.01,
        },
    },
}

cfg = EXPERIMENTS[EXPERIMENT]

# =========================================================
# 기본 설정
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

# DagsHub MLflow (선택사항)
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-tuning")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# 실행
# =========================================================
print("=" * 60)
print(f"🔬 실험 {EXPERIMENT}: {cfg['name']}")
print(f"   samples={cfg['num_calibration_samples']}, "
      f"seq_len={cfg['max_sequence_length']}")
print(f"   ignore={cfg['ignore']}")
print(f"   gptq_kwargs={cfg['gptq_kwargs']}")
print("=" * 60)

# GPU 정보
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

# 1. 모델 로드
print("\n[1/5] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터")

# 2. 데이터 준비
print(f"\n[2/5] 캘리브레이션 데이터 {cfg['num_calibration_samples']}개 로드 중...")
ds = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split=f"train[:{cfg['num_calibration_samples']}]",
)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess)
print(f"  → {len(ds)}개 샘플 준비 완료")

# 3. GPTQ 양자화
print(f"\n[3/5] GPTQ 양자화 시작...")
start_time = time.time()

recipe = [
    GPTQModifier(
        scheme=cfg["scheme"],
        targets=cfg["targets"],
        ignore=cfg["ignore"],
        **cfg["gptq_kwargs"],  # actorder, dampening_frac 등
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=cfg["max_sequence_length"],
    num_calibration_samples=cfg["num_calibration_samples"],
)

quant_time = time.time() - start_time
print(f"  → 소요시간: {quant_time:.1f}초")

# 4. 저장
print(f"\n[4/6] 모델 저장 중... → {OUT_DIR}")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# 저장 확인
total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  → 모델 크기: {total_size / (1024*1024):.1f} MB")

# =========================================================
# config.json 최적화
# =========================================================
# vLLM은 max_model_len이 미지정 시 config.json의 max_position_embeddings를 사용.
# EXAONE 기본값 65536 → 축소하면 KV cache 메모리 절감 → batch↑ → 속도↑
import json

MAX_POSITION_EMBEDDINGS = 32768    # 65536 → 32768 (절반, 안전한 선택)
# MAX_POSITION_EMBEDDINGS = 16384  # 더 공격적 (max_gen_toks와 동일)

config_path = os.path.join(OUT_DIR, "config.json")
print(f"\n[5/6] config.json 최적화 중...")

with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")
print(f"  → KV cache 메모리 {(1 - MAX_POSITION_EMBEDDINGS/65536)*100:.0f}% 절감 예상!")

# 6. ZIP 생성
print(f"\n[6/6] ZIP 생성 중...")
zip_name = f"submit_exp{EXPERIMENT}_{cfg['name']}"
shutil.make_archive(
    base_name=f"/kaggle/working/{zip_name}",
    format="zip",
    root_dir="/kaggle/working",
    base_dir="model",
)

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024*1024)
print(f"  → {zip_path} ({zip_size:.1f} MB)")

# MLflow 기록
if USE_MLFLOW:
    with mlflow.start_run(run_name=f"exp{EXPERIMENT}-{cfg['name']}"):
        mlflow.log_params({
            "experiment": EXPERIMENT,
            "samples": cfg["num_calibration_samples"],
            "seq_len": cfg["max_sequence_length"],
            "scheme": cfg["scheme"],
            "ignore": str(cfg["ignore"]),
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            **{f"gptq_{k}": str(v) for k, v in cfg["gptq_kwargs"].items()},
        })
        mlflow.log_metrics({
            "quant_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

# 완료
print("\n" + "=" * 60)
print(f"✅ 실험 {EXPERIMENT} ({cfg['name']}) 완료!")
print(f"""
📊 비교 기준 (0.574 모델):
   samples: 256 → {cfg['num_calibration_samples']}
   seq_len: 512 → {cfg['max_sequence_length']}
   actorder: 없음 → {cfg['gptq_kwargs'].get('actorder', '없음')}
   dampening: 없음 → {cfg['gptq_kwargs'].get('dampening_frac', '없음')}
   ignore: embed+lm_head → {cfg['ignore']}
   ⭐ max_position_embeddings: 65536 → {MAX_POSITION_EMBEDDINGS}

🚀 {zip_name}.zip 을 DACON에 제출하세요!
""")
