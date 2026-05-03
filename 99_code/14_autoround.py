"""
AutoRound 양자화 V3 - llmcompressor AutoRoundModifier (평가 서버 호환)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 변경사항:
  - standalone auto-round (X) → llmcompressor의 AutoRoundModifier (O)
  - 저장 포맷: compressed-tensors (quant_method: "compressed-tensors")
  - 평가 서버: vllm==0.14.1, compressed-tensors==0.13.0 호환

Kaggle 사용법:
  !pip install -q compressed-tensors==0.13.0
  !pip install -q "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git" auto-round mlflow dagshub
  !pip install -q transformers==4.57.3
"""

# =========================================================
# 0. 패키지 설치 (Kaggle/Colab에서 먼저 실행!)
# =========================================================
# !pip install -q compressed-tensors==0.13.0
# !pip install -q "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git" auto-round mlflow dagshub
# !pip install -q transformers==4.57.3

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic CuBLAS 경고 해소
import time
import torch
import shutil
from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

# MLflow (선택 - 없으면 스킵)
try:
    import mlflow
    import dagshub
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[INFO] mlflow/dagshub 미설치. 트래킹 없이 진행합니다.")

# =========================================================
# 1. 설정
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

# AutoRound 양자화 설정
SCHEME = "W4A16"                    # 4-bit weight, 16-bit activation
TARGETS = ["Linear"]
IGNORE = ["lm_head"]

# AutoRound 하이퍼파라미터
AUTOROUND_ITERS = 200               # SignSGD 반복 수 (default: 200)
AUTOROUND_BATCH_SIZE = 4            # 배치 크기 (Kaggle T4 기준, OOM 방지)
AUTOROUND_LR = None                 # 학습률 (None → 1/iters 자동)

# 캘리브레이션
NUM_CALIBRATION_SAMPLES = 128       # 캘리브레이션 샘플 수
MAX_SEQUENCE_LENGTH = 1024          # 최대 시퀀스 길이 (T4 OOM 방지: 2048→1024)

# MLflow
DAGSHUB_REPO = "sthun0211/LGaimers"
MLFLOW_EXPERIMENT = "AutoRound_Quantization_v3"

# =========================================================
# 2. 초기화
# =========================================================
if MLFLOW_AVAILABLE:
    dagshub.init(repo_owner="sthun0211", repo_name="LGaimers", mlflow=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] GPU 없음")

# =========================================================
# 3. 모델 로드
# =========================================================
print(f"\n[INFO] 모델 로드: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# ★ device_map="auto" 사용 시 accelerate가 forward를 functools.partial로 래핑하여
#    compressed-tensors offload에서 __func__ 접근 에러 발생
#    → 1.2B 모델은 T4(16GB)에 충분하므로 직접 .to("cuda") 사용
# ★ T4는 bfloat16 미지원 → 내부적으로 float32 fallback하여 메모리 2배 사용
#   float16으로 변경하면 T4에서 네이티브로 동작하여 메모리 절약
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")
print(f"[INFO] 파라미터: {model.num_parameters() / 1e9:.2f}B")

# =========================================================
# 4. 캘리브레이션 데이터셋 (AutoRound 내장 데이터셋 사용)
# =========================================================
print(f"\n[INFO] 캘리브레이션 데이터 준비 ({NUM_CALIBRATION_SAMPLES} samples, seqlen={MAX_SEQUENCE_LENGTH})")

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# =========================================================
# 5. AutoRound 양자화 (llmcompressor 경유 → compressed-tensors 포맷)
# =========================================================
def run():
    print("\n" + "=" * 60)
    print("[INFO] AutoRound 양자화 시작 (llmcompressor AutoRoundModifier)")
    print(f"       Scheme: {SCHEME}")
    print(f"       Targets: {TARGETS}, Ignore: {IGNORE}")
    print(f"       Iters: {AUTOROUND_ITERS}, Batch Size: {AUTOROUND_BATCH_SIZE}")
    print(f"       LR: {AUTOROUND_LR}")
    print(f"       Calibration: {NUM_CALIBRATION_SAMPLES} samples, seqlen={MAX_SEQUENCE_LENGTH}")
    print("=" * 60)

    # AutoRoundModifier 설정
    recipe = AutoRoundModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        iters=AUTOROUND_ITERS,
        batch_size=AUTOROUND_BATCH_SIZE,
    )

    start_time = time.time()

    # oneshot으로 양자화 실행
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        shuffle_calibration_samples=False,
    )

    elapsed = time.time() - start_time
    print(f"[INFO] AutoRound 양자화 완료! ({elapsed:.1f}초)")
    return elapsed


def save(elapsed):
    # ── 모델 저장 (compressed-tensors 포맷) ──
    print(f"\n[INFO] 저장: {OUT_DIR}")
    print(f"       포맷: compressed-tensors (vLLM 0.14.1 호환)")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ★ save_compressed=True → compressed-tensors 포맷으로 저장
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    # 저장 확인
    print("[INFO] 저장된 파일:")
    total_size = 0
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / (1024 * 1024)
            total_size += size
            print(f"       - {f} ({size:.1f} MB)")
    print(f"       총 크기: {total_size:.1f} MB")

    # ── ZIP 생성 ──
    zip_name = "autoround_submit"
    shutil.make_archive(f"/kaggle/working/{zip_name}", "zip", "/kaggle/working", "model")

    zip_path = f"/kaggle/working/{zip_name}.zip"
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)

    if zip_size > 10 * 1024:
        print(f"[WARNING] ZIP이 10GB 초과! ({zip_size:.1f} MB)")
    else:
        print(f"[INFO] ZIP 생성 완료: {zip_path} ({zip_size:.1f} MB)")

    return total_size, zip_size, zip_path


# ── 실행 ──
if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name=f"AutoRound_{SCHEME}_iters{AUTOROUND_ITERS}"):
        mlflow.log_params({
            "model_id": MODEL_ID,
            "quant_method": "AutoRound (llmcompressor)",
            "scheme": SCHEME,
            "iters": AUTOROUND_ITERS,
            "batch_size": AUTOROUND_BATCH_SIZE,
            "lr": str(AUTOROUND_LR),
            "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        })

        elapsed = run()
        total_size, zip_size, zip_path = save(elapsed)

        mlflow.log_metric("quantization_time_sec", elapsed)
        mlflow.log_metric("model_size_mb", total_size)
        mlflow.log_metric("zip_size_mb", zip_size)

        config_path = os.path.join(OUT_DIR, "config.json")
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="model_config")
else:
    elapsed = run()
    total_size, zip_size, zip_path = save(elapsed)


# =========================================================
# 완료
# =========================================================
print("\n" + "=" * 60)
print("✅ AutoRound 양자화 완료! (compressed-tensors 포맷)")
print("=" * 60)
print(f"""
📊 설정:
   • Model: {MODEL_ID}
   • Scheme: {SCHEME}
   • Iters: {AUTOROUND_ITERS}, Batch Size: {AUTOROUND_BATCH_SIZE}
   • Calibration: {NUM_CALIBRATION_SAMPLES} samples, seqlen={MAX_SEQUENCE_LENGTH}

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: /kaggle/working/autoround_submit.zip

🔑 이전 코드와의 핵심 차이:
   ✅ config.json → quant_method: "compressed-tensors" (서버 호환)
   ❌ 이전: quant_method: "auto-round" (서버 미호환)

   • 10/13_autoround.py: standalone auto-round → quant_method: "auto-round" 저장 → 서버 ValidationError
   • 14_autoround.py:    llmcompressor AutoRoundModifier → quant_method: "compressed-tensors" 저장 → 서버 호환!

🚀 ZIP 파일을 대회에 제출하세요!
""")
