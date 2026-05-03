"""
AutoRound 양자화 V4 - Kaggle T4 OOM 해결 + 평가 서버 호환
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
문제 이력:
  - 10_autoaround.py: format="llm_compressor" → 서버 ValidationError
  - 14_autoround.py: llm-compressor + bfloat16/bs8/seq2048 → Kaggle T4 OOM

해결:
  1. llm-compressor AutoRoundModifier (compressed-tensors 포맷, 서버 호환)
  2. float16 (T4 네이티브, bfloat16은 float32 fallback되어 메모리 2배)
  3. batch_size=4, seqlen=1024 (VRAM 절약)
  4. CUBLAS_WORKSPACE_CONFIG 설정 (deterministic 경고 해소)
  5. torch.compile 비활성화 (torch._dynamo 관련 메모리/속도 이슈 방지)

Kaggle 사용법:
  !pip install -q "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git" auto-round mlflow dagshub
  !pip install -q transformers==4.57.3
  # ※ compressed-tensors는 llmcompressor가 자동 설치 (다운그레이드 하면 import 에러)
  # ※ 저장 후 config.json version을 0.13.0으로 자동 패치하여 서버 호환
"""

# =========================================================
# 0. 환경 설정 (import 전에 설정!)
# =========================================================
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # deterministic CuBLAS
os.environ["TOKENIZERS_PARALLELISM"] = "false"       # tokenizer 경고 방지
os.environ["TORCH_COMPILE_DISABLE"] = "1"            # torch.compile 비활성화

import json
import time
import torch
import shutil

# torch.compile 관련 메모리 이슈 방지
torch._dynamo.config.suppress_errors = True

from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

# MLflow (선택)
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

# ★ T4 OOM 방지: batch_size 줄이기, seqlen 줄이기
AUTOROUND_ITERS = 200               # SignSGD 반복 수
AUTOROUND_BATCH_SIZE = 4            # 배치 크기 (8→4, OOM 방지)

# ★ T4 OOM 방지: 시퀀스 길이 줄이기 (attention 메모리 ∝ seqlen²)
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512          # 2048→1024

# MLflow
DAGSHUB_REPO = "sthun0211/LGaimers"
MLFLOW_EXPERIMENT = "AutoRound_Quantization_v4"

# =========================================================
# 2. 초기화
# =========================================================
if MLFLOW_AVAILABLE:
    dagshub.init(repo_owner="sthun0211", repo_name="LGaimers", mlflow=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {gpu_name}")
    print(f"[INFO] VRAM: {gpu_vram:.1f} GB")
else:
    print("[WARNING] GPU 없음")

# =========================================================
# 3. 모델 로드
# =========================================================
print(f"\n[INFO] 모델 로드: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# ★ 핵심 변경: bfloat16 → float16
#   Tesla T4는 bfloat16을 하드웨어로 지원하지 않음 (Turing 아키텍처)
#   → PyTorch가 내부적으로 float32로 변환하여 메모리 2배 사용
#   float16은 T4에서 네이티브로 동작하여 메모리 효율적
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,   # ★ bfloat16 → float16
    trust_remote_code=True,
).to("cuda")
print(f"[INFO] 파라미터: {model.num_parameters() / 1e9:.2f}B")

# 메모리 상태 확인
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU 메모리: {used:.1f}GB / {total:.1f}GB 사용 중")

# =========================================================
# 4. 캘리브레이션 데이터셋
# =========================================================
print(f"\n[INFO] 캘리브레이션 데이터 준비 ({NUM_CALIBRATION_SAMPLES} samples, seqlen={MAX_SEQUENCE_LENGTH})")

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# =========================================================
# 5. AutoRound 양자화 (llmcompressor → compressed-tensors 포맷)
# =========================================================
def run():
    print("\n" + "=" * 60)
    print("[INFO] AutoRound 양자화 시작 (llmcompressor AutoRoundModifier)")
    print(f"       Scheme: {SCHEME}")
    print(f"       Targets: {TARGETS}, Ignore: {IGNORE}")
    print(f"       Iters: {AUTOROUND_ITERS}, Batch Size: {AUTOROUND_BATCH_SIZE}")
    print(f"       Calibration: {NUM_CALIBRATION_SAMPLES} samples, seqlen={MAX_SEQUENCE_LENGTH}")
    print(f"       dtype: float16 (T4 네이티브)")
    print("=" * 60)

    recipe = AutoRoundModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        iters=AUTOROUND_ITERS,
        batch_size=AUTOROUND_BATCH_SIZE,
    )

    start_time = time.time()

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
    print(f"\n[INFO] 저장: {OUT_DIR}")
    print(f"       포맷: compressed-tensors (vLLM 0.14.1 호환)")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ★ save_compressed=True → compressed-tensors 네이티브 포맷
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    # ★ config.json의 compressed-tensors 버전을 서버(0.13.0)와 일치시킴
    #   llmcompressor dev가 0.13.1a를 기록하지만, 실제 포맷은 0.13.0과 동일
    config_path = os.path.join(OUT_DIR, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    if "quantization_config" in config:
        old_ver = config["quantization_config"].get("version", "unknown")
        config["quantization_config"]["version"] = "0.13.0"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[INFO] config.json version 패치: {old_ver} → 0.13.0")

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

    # ZIP 생성
    zip_name = "autoround_submit"
    shutil.make_archive(f"/kaggle/working/{zip_name}", "zip", "/kaggle/working", "model")

    zip_path = f"/kaggle/working/{zip_name}.zip"
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)

    if zip_size > 10 * 1024:
        print(f"[WARNING] ZIP이 10GB 초과! ({zip_size:.1f} MB)")
    else:
        print(f"[INFO] ZIP 생성 완료: {zip_path} ({zip_size:.1f} MB)")

    return total_size, zip_size, zip_path


# =========================================================
# 실행
# =========================================================
if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name=f"AutoRound_{SCHEME}_iters{AUTOROUND_ITERS}_v4"):
        mlflow.log_params({
            "model_id": MODEL_ID,
            "quant_method": "AutoRound (llmcompressor)",
            "scheme": SCHEME,
            "iters": AUTOROUND_ITERS,
            "batch_size": AUTOROUND_BATCH_SIZE,
            "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "torch_dtype": "float16",
            "t4_oom_fix": "float16+bs4+seq1024+no_compile",
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
   • dtype: float16 (T4 OOM 방지)

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: /kaggle/working/autoround_submit.zip

🔑 이전 코드 대비 변경:
   ✅ bfloat16 → float16 (T4 메모리 절약)
   ✅ batch_size: 8 → 4 (OOM 방지)
   ✅ seqlen: 2048 → 1024 (attention 메모리 절약)
   ✅ torch.compile 비활성화 (메모리 최적화)
   ✅ compressed-tensors 포맷 (서버 호환)

🚀 ZIP 파일을 대회에 제출하세요!
""")
