# %%
"""
FP8 양자화 - EXAONE-4.0-1.2B (Kaggle T4x2 → L4 평가서버)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

평가서버: L4 GPU (Ada Lovelace, FP8 Tensor Core 지원)
양자화 실행: Kaggle T4x2 (FP8 연산은 불가하지만 양자화 변환은 가능)

FP8 장점 (vs W4A16):
  📈 정확도: FP16 대비 거의 무손실 (PerfNorm ~0.98~1.0)
  ⚡ 속도:   L4의 FP8 Tensor Core 직접 활용 (SpeedNorm ~0.65~0.75)
  🎯 점수:   50:50 평가에서 W4A16보다 높은 점수 기대

양자화 방식:
  FP8_DYNAMIC: 가중치=FP8 정적, 활성화=FP8 동적(per-token) ← 추천
  FP8:         가중치=FP8 정적, 활성화=FP8 정적(per-tensor)

Kaggle 사용법:
  셀 1의 pip 명령어를 먼저 실행 후, 런타임 재시작, 셀 2부터 실행
"""

# %% [셀 1] 패키지 설치 (⚠️ Kaggle에서 먼저 실행 → 런타임 재시작!)
# !pip install -q llmcompressor mlflow dagshub
# !pip install -q transformers==4.57.3
# !pip install -q compressed-tensors==0.13.0

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
from llmcompressor.modifiers.quantization import QuantizationModifier

# compressed-tensors 버전 확인 (평가서버와 일치해야 함)
try:
    import compressed_tensors
    ct_version = getattr(compressed_tensors, "__version__", "unknown")
    print(f"[INFO] compressed-tensors 버전: {ct_version}")
    if ct_version != "0.13.0":
        print(f"[WARNING] 평가서버는 0.13.0입니다! 현재: {ct_version}")
        print(f"         !pip install compressed-tensors==0.13.0 후 재시작 필요")
except ImportError:
    print("[WARNING] compressed-tensors 미설치")

# MLflow (선택)
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '1ee266cf0159abb2c8ad8ae564274c6918599acd'
    import mlflow
    import dagshub
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("FP8-quantization")
    USE_MLFLOW = True
except Exception:
    USE_MLFLOW = False

print("[INFO] 라이브러리 로드 완료")

# %% [셀 3] 설정 (⭐ 여기서 파라미터 조정)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ── FP8 양자화 설정 ──
# "FP8_DYNAMIC": 활성화를 추론 시 동적으로 양자화 (추천! 정확도 ↑)
# "FP8":         활성화를 캘리브레이션으로 정적 양자화 (약간 더 빠를 수 있음)
FP8_SCHEME = "FP8_DYNAMIC"

TARGETS = ["Linear"]
IGNORE = [
    "lm_head",            # 출력 레이어 보호 (정확도 유지)
]

# ── 캘리브레이션 설정 (FP8 정적 모드에서 사용) ──
NUM_CALIBRATION_SAMPLES = 512     # FP8_DYNAMIC은 적어도 됨
MAX_SEQUENCE_LENGTH = 512

# ── 공짜 속도 최적화 ──
MAX_POSITION_EMBEDDINGS = 32768   # 65536 → 32768 (KV Cache 50% 절감!)

print(f"""
📊 FP8 양자화 설정:
┌──────────────────────────────────────────┐
│ 🎯 Scheme: {FP8_SCHEME:<30}│
│ 📌 Targets: {str(TARGETS):<29}│
│ 🛡️ Ignore: {str(IGNORE):<30}│
│ 📊 Calibration: {NUM_CALIBRATION_SAMPLES} samples, {MAX_SEQUENCE_LENGTH} seq_len  │
│ ⚡ max_position_embeddings: {MAX_POSITION_EMBEDDINGS:<13}│
└──────────────────────────────────────────┘

🔑 FP8 vs W4A16:
  모델 크기: ~1.2GB (W4A16: ~0.7GB) — 약간 큼
  정확도:   거의 무손실 (W4A16: 약간 손실) — 훨씬 좋음
  추론속도: FP8 Tensor Core (W4A16: FP16 연산) — 훨씬 빠름
""")

# %% [셀 4] GPU 확인
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"[INFO] GPU {i}: {gpu_name} ({gpu_vram:.1f}GB)")

    # T4 감지 → float16 사용 안내
    if "T4" in torch.cuda.get_device_name(0):
        print("[INFO] T4 감지 → float16으로 모델 로드 (bfloat16 하드웨어 미지원)")
        print("[INFO] FP8 양자화 변환은 T4에서도 가능합니다!")
        LOAD_DTYPE = torch.float16
    else:
        LOAD_DTYPE = torch.bfloat16
else:
    print("[WARNING] GPU 없음")
    LOAD_DTYPE = torch.float32

# %% [셀 5] 모델 로드
print(f"\n[1/7] 모델 로드: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=LOAD_DTYPE,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터 (dtype: {LOAD_DTYPE})")

# %% [셀 6] 캘리브레이션 데이터 로드
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

# %% [셀 7] FP8 양자화 실행 (⭐ 핵심)
print(f"\n[3/7] FP8 양자화 시작")
print(f"  📌 Scheme: {FP8_SCHEME}")
print(f"  📌 Targets: {TARGETS}")
print(f"  📌 Ignore: {IGNORE}")

recipe = [
    QuantizationModifier(
        targets=TARGETS,
        scheme=FP8_SCHEME,
        ignore=IGNORE,
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
print(f"  → FP8 양자화 완료! ({quant_time:.1f}초)")
print(f"  💡 W4A16 GPTQ 대비 훨씬 빠름 (FP8은 GPTQ 반복 최적화 불필요)")

# %% [셀 8] 모델 저장
print(f"\n[4/7] 모델 저장: {OUT_DIR}")
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

# %% [셀 9] config.json 최적화 (⚡ 공짜 속도!)
print(f"\n[5/7] config.json 최적화")
config_path = os.path.join(OUT_DIR, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

# max_position_embeddings 축소 → KV Cache 메모리 절감
original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

# transformers 버전을 평가서버와 맞춤
config["transformers_version"] = "4.57.3"

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")
print(f"  → KV Cache {(1 - MAX_POSITION_EMBEDDINGS/65536)*100:.0f}% 절감!")
print(f"  → transformers_version: 4.57.3")

# quantization_config 확인
if "quantization_config" in config:
    qc = config["quantization_config"]
    print(f"\n  📋 양자화 설정 확인:")
    print(f"     quant_method: {qc.get('quant_method', 'N/A')}")
    print(f"     format: {qc.get('format', 'N/A')}")
    print(f"     version: {qc.get('version', 'N/A')}")

    # config_groups 내부 확인
    if "config_groups" in qc:
        for group_name, group_cfg in qc["config_groups"].items():
            weights = group_cfg.get("weights", {})
            acts = group_cfg.get("input_activations", {})
            print(f"     [{group_name}]")
            print(f"       weights: type={weights.get('type')}, num_bits={weights.get('num_bits')}")
            if acts:
                print(f"       activations: type={acts.get('type')}, num_bits={acts.get('num_bits')}, dynamic={acts.get('dynamic')}")

# %% [셀 10] ZIP 생성
print(f"\n[6/7] ZIP 생성")
zip_name = "fp8_model"
shutil.make_archive(f"/kaggle/working/{zip_name}", "zip", "/kaggle/working", "model")

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"  → {zip_path} ({zip_size:.1f} MB)")

# 용량 제한 확인 (10GB)
if zip_size > 10 * 1024:
    print(f"  ⚠️ 제출 용량 초과! ({zip_size:.0f}MB > 10240MB)")
else:
    print(f"  ✅ 용량 OK ({zip_size:.0f}MB / 10240MB)")

# %% [셀 11] MLflow 기록
print(f"\n[7/7] MLflow 기록")
if USE_MLFLOW:
    with mlflow.start_run(run_name=f"FP8-{FP8_SCHEME}"):
        mlflow.log_params({
            "model_id": MODEL_ID,
            "scheme": FP8_SCHEME,
            "targets": str(TARGETS),
            "ignore": str(IGNORE),
            "calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_seq_length": MAX_SEQUENCE_LENGTH,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "load_dtype": str(LOAD_DTYPE),
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
print("✅ FP8 양자화 완료!")
print("=" * 60)
print(f"""
📊 FP8 양자화 결과 요약:
┌──────────────────────────────────────────┐
│ Scheme: {FP8_SCHEME:<32}│
│ 모델 크기: {total_size/(1024*1024):.1f} MB{' '*(26-len(f'{total_size/(1024*1024):.1f}'))}│
│ ZIP 크기: {zip_size:.1f} MB{' '*(27-len(f'{zip_size:.1f}'))}│
│ 양자화 시간: {quant_time:.1f}초{' '*(25-len(f'{quant_time:.1f}'))}│
│ max_position_embeddings: {MAX_POSITION_EMBEDDINGS:<15}│
├──────────────────────────────────────────┤
│ 📈 예상 평가 점수 (vs W4A16):            │
│   PerfNorm: ~0.98~1.0 (FP16 거의 무손실) │
│   SpeedNorm: ~0.65~0.75 (FP8 TC 활용)   │
│   Score: ~0.83~0.87 (W4A16: ~0.77)      │
└──────────────────────────────────────────┘

🚀 {zip_name}.zip 을 대회에 제출하세요!
""")
