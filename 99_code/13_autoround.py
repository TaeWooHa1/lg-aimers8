"""
AutoRound 양자화 V2 - standalone auto-round (평가 서버 호환)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vLLM 0.14.1이 auto-round 포맷을 네이티브로 지원 (INCConfig).
→ config.json의 quant_method: "auto-round" → vLLM이 GPTQ Marlin 커널로 추론.
→ llmcompressor 의존성 없이 단독 auto-round 사용.

사용법:
  Kaggle에서 GPU 런타임(T4 이상)으로 실행
  !pip install -q auto-round==0.9.6 mlflow dagshub
  !pip install -q transformers==4.57.3
"""

# =========================================================
# 0. 패키지 설치 (Kaggle/Colab에서 먼저 실행!)
# =========================================================
# !pip install -q auto-round==0.9.6 mlflow dagshub
# !pip install -q transformers==4.57.3

import os
import time
import torch
import shutil
from pathlib import Path

# transformers 4.57.3 호환 패치 (auto-round가 제거된 함수를 참조하는 문제 해결)
import transformers.utils
if not hasattr(transformers.utils, 'is_tf_available'):
    transformers.utils.is_tf_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

# MLflow (선택사항 - 없으면 스킵)
try:
    import mlflow
    import dagshub
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[INFO] mlflow/dagshub 미설치. 트래킹 없이 진행합니다.")

# =========================================================
# 1. 경로 설정
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

# =========================================================
# 2. AutoRound 양자화 설정
# =========================================================
# 양자화 비트 설정
BITS = 4                        # 가중치 비트 수 (W4)
GROUP_SIZE = 128                # 그룹 양자화 크기
SYMMETRIC = True                # 대칭 양자화

# AutoRound 최적화 하이퍼파라미터
AUTOROUND_ITERS = 500           # SignSGD 반복 수 (기본 200, 높을수록 정확)
AUTOROUND_LR = None             # 학습률 (None이면 1/iters 자동 설정)
AUTOROUND_MINMAX_LR = None      # 클리핑 범위 학습률 (None이면 lr과 동일)
AUTOROUND_BATCH_SIZE = 8        # 배치 크기 (Kaggle T4 기준)

# 캘리브레이션 설정
NUM_CALIBRATION_SAMPLES = 1024  # 캘리브레이션 샘플 수
MAX_SEQUENCE_LENGTH = 512       # 최대 시퀀스 길이 (Kaggle 메모리 고려)

# ★ 핵심: 저장 포맷 (vLLM 0.14.1 호환)
# "auto_round" → config.json에 quant_method: "auto-round" 기록
# → vLLM INCConfig가 자동 감지하여 GPTQ Marlin 커널로 추론
FORMAT_TYPE = "auto_round"

# =========================================================
# 2-1. MLflow 설정 (선택)
# =========================================================
DAGSHUB_REPO = "sthun0211/LGaimers"
MLFLOW_EXPERIMENT = "AutoRound_Quantization_v2"

if MLFLOW_AVAILABLE:
    dagshub.init(repo_owner="sthun0211", repo_name="LGaimers", mlflow=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"[INFO] MLflow 트래킹 서버: https://dagshub.com/{DAGSHUB_REPO}.mlflow")
    print(f"[INFO] Experiment: {MLFLOW_EXPERIMENT}")

# =========================================================
# 3. GPU 메모리 정리
# =========================================================
GPU_NAME = "N/A"
GPU_VRAM = 0.0
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {GPU_NAME}")
    print(f"[INFO] VRAM: {GPU_VRAM:.1f} GB")
else:
    print("[WARNING] GPU를 찾을 수 없습니다. CPU로 실행됩니다 (매우 느림).")

# =========================================================
# 4. 모델 로드
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
# 5. AutoRound 양자화 & 저장
# =========================================================

def run_quantization():
    """AutoRound 양자화 실행"""
    global model

    print("\n" + "=" * 60)
    print("[INFO] AutoRound 양자화 시작 (standalone auto-round)")
    print(f"       Bits: {BITS}, Group Size: {GROUP_SIZE}, Symmetric: {SYMMETRIC}")
    print(f"       Iterations: {AUTOROUND_ITERS}")
    print(f"       Batch Size: {AUTOROUND_BATCH_SIZE}")
    print(f"       LR: {AUTOROUND_LR}")
    print(f"       Calibration Samples: {NUM_CALIBRATION_SAMPLES}")
    print(f"       Max Seq Length: {MAX_SEQUENCE_LENGTH}")
    print(f"       Save Format: {FORMAT_TYPE} (vLLM INCConfig 호환)")
    print("=" * 60)

    quant_start = time.time()

    # AutoRound 초기화 (캘리브레이션 데이터 자동 로드)
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=SYMMETRIC,
        iters=AUTOROUND_ITERS,
        lr=AUTOROUND_LR,
        minmax_lr=AUTOROUND_MINMAX_LR,
        batch_size=AUTOROUND_BATCH_SIZE,
        nsamples=NUM_CALIBRATION_SAMPLES,
        seqlen=MAX_SEQUENCE_LENGTH,
    )

    # 양자화 실행
    autoround.quantize()

    quant_time = time.time() - quant_start
    print(f"[INFO] AutoRound 양자화 완료! (소요 시간: {quant_time:.1f}초)")

    return autoround, quant_time


def save_model(autoround, quant_time):
    """양자화 모델 저장 + ZIP 생성"""

    # ── 모델 저장 (auto_round 포맷 → vLLM INCConfig 호환) ──
    print(f"\n[INFO] 모델 저장 중... → {OUT_DIR}")
    print(f"       포맷: {FORMAT_TYPE} (quant_method: 'auto-round')")
    print(f"       → vLLM 0.14.1 INCConfig → GPTQ Marlin 커널 추론")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ★ format="auto_round"으로 저장
    # → config.json에 quant_method: "auto-round" 기록
    # → 가중치는 GPTQ 호환 패킹
    autoround.save_quantized(
        output_dir=OUT_DIR,
        format=FORMAT_TYPE,
        inplace=False,
    )

    # 토크나이저도 함께 저장
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

    # 제출 파일 용량 검증 (10GB 제한)
    if zip_size > 10 * 1024:
        print(f"[WARNING] ZIP 파일이 10GB를 초과합니다! ({zip_size:.1f} MB)")
    else:
        print(f"[INFO] 제출 용량 OK (제한: 10GB, 현재: {zip_size:.1f} MB)")

    return total_size, zip_size, zip_path


# ── 실행 ──
if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name=f"AutoRound_W{BITS}A16_gs{GROUP_SIZE}_iters{AUTOROUND_ITERS}"):

        # 파라미터 기록
        mlflow.log_params({
            "model_id": MODEL_ID,
            "quant_method": "AutoRound (standalone)",
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "symmetric": SYMMETRIC,
            "iters": AUTOROUND_ITERS,
            "lr": str(AUTOROUND_LR),
            "batch_size": AUTOROUND_BATCH_SIZE,
            "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "format_type": FORMAT_TYPE,
            "gpu": GPU_NAME,
            "gpu_vram_gb": GPU_VRAM,
        })

        autoround, quant_time = run_quantization()
        total_size, zip_size, zip_path = save_model(autoround, quant_time)

        # 메트릭 기록
        mlflow.log_metric("quantization_time_sec", quant_time)
        mlflow.log_metric("model_size_mb", total_size)
        mlflow.log_metric("zip_size_mb", zip_size)

        # config.json 아티팩트 기록
        config_path = os.path.join(OUT_DIR, "config.json")
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="model_config")

        # quantize_config.json 아티팩트 기록
        qconfig_path = os.path.join(OUT_DIR, "quantize_config.json")
        if os.path.exists(qconfig_path):
            mlflow.log_artifact(qconfig_path, artifact_path="model_config")

        print(f"\n[INFO] MLflow Run 기록 완료! → https://dagshub.com/{DAGSHUB_REPO}.mlflow")
else:
    autoround, quant_time = run_quantization()
    total_size, zip_size, zip_path = save_model(autoround, quant_time)


# =========================================================
# 완료!
# =========================================================
print("\n" + "=" * 60)
print("✅ AutoRound 양자화 완료!")
print("=" * 60)
print(f"""
📊 설정 요약:
   • Model: {MODEL_ID}
   • Bits: {BITS}, Group Size: {GROUP_SIZE}, Symmetric: {SYMMETRIC}
   • Iterations: {AUTOROUND_ITERS}
   • Batch Size: {AUTOROUND_BATCH_SIZE}
   • Calibration Samples: {NUM_CALIBRATION_SAMPLES}
   • Max Seq Length: {MAX_SEQUENCE_LENGTH}
   • Save Format: {FORMAT_TYPE} (vLLM INCConfig 호환)

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: {zip_path}

🔧 이전 코드(10_autoaround.py)와의 차이:
   • format='llm_compressor' → format='auto_round' (vLLM 네이티브 지원)
   • llmcompressor 의존성 완전 제거
   • vLLM 0.14.1 INCConfig → GPTQ Marlin 커널로 추론

🔧 12_autoround.py와의 차이:
   • llmcompressor + AutoRoundModifier → standalone auto_round.AutoRound
   • transformers/auto-round 버전 충돌 문제 해결

🚀 다운로드된 ZIP 파일을 대회에 제출하세요!
""")
