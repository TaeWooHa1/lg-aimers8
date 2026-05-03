# !pip install -q auto-round mlflow dagshub
# !pip install -q transformers==4.57.3

import os
import time
import torch
import shutil
import mlflow
import dagshub
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

# =========================================================
# 1. 경로 설정
# =========================================================
# HuggingFace에서 직접 다운로드 (약 2.5GB)
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 출력 폴더 (Kaggle 환경)
OUT_DIR = "/kaggle/working/model"

# =========================================================
# 2. AutoRound 양자화 설정
# =========================================================
# 양자화 비트 설정
BITS = 4                    # 가중치 비트 수 (W4)
GROUP_SIZE = 128            # 그룹 양자화 크기
SYMMETRIC = True            # 대칭 양자화

# AutoRound 최적화 하이퍼파라미터
AUTOROUND_ITERS = 200       # SignSGD 반복 수 (기본 200, 높을수록 정확)
AUTOROUND_LR = None         # 학습률 (None이면 1/iters 자동 설정)
AUTOROUND_MINMAX_LR = None  # 클리핑 범위 학습률 (None이면 lr과 동일)
AUTOROUND_NROUNDS = 1       # 블록 반복 수 (기본 1)

# 캘리브레이션 설정
NUM_CALIBRATION_SAMPLES = 1024  # 캘리브레이션 샘플 수
MAX_SEQUENCE_LENGTH = 512       # 최대 시퀀스 길이

# 저장 포맷 (vLLM 호환)
FORMAT_TYPE = "llm_compressor"  # vLLM과 호환성이 좋고 compressed-tensors 라이브러리로 로드 가능

# =========================================================
# 2-1. MLflow 설정 (DagsHub)
# =========================================================
DAGSHUB_REPO = "sthun0211/LGaimers"
MLFLOW_EXPERIMENT = "AutoRound_Quantization"

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
# 3-1. FORMAT_TYPE 사전 검증 (양자화 전에 미리 확인!)
# =========================================================
print("\n" + "=" * 60)
print(f"[CHECK] FORMAT_TYPE 유효성 검사 중... → '{FORMAT_TYPE}'")

try:
    from auto_round.formats import SUPPORTED_FORMATS
    if FORMAT_TYPE not in SUPPORTED_FORMATS:
        raise KeyError(
            f"[ERROR] 지원하지 않는 FORMAT_TYPE: '{FORMAT_TYPE}'\n"
            f"        지원 포맷 목록: {SUPPORTED_FORMATS}\n"
            f"        ⚠️  양자화를 시작하지 않습니다. FORMAT_TYPE을 수정하세요!"
        )
    print(f"[CHECK] ✅ FORMAT_TYPE '{FORMAT_TYPE}' 유효! 양자화를 시작합니다.")
except ImportError:
    print("[CHECK] ⚠️  SUPPORTED_FORMATS 임포트 실패 → 검증 건너뜀 (auto_round 버전 확인 필요)")

print("=" * 60)


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
# 5. AutoRound 양자화 (MLflow 기록 시작)
# =========================================================
with mlflow.start_run(run_name=f"AutoRound_W{BITS}A16_gs{GROUP_SIZE}_iters{AUTOROUND_ITERS}"):

    # ── 파라미터 기록 ──
    mlflow.log_params({
        "model_id": MODEL_ID,
        "quant_method": "AutoRound",
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "symmetric": SYMMETRIC,
        "iters": AUTOROUND_ITERS,
        "lr": str(AUTOROUND_LR),
        "minmax_lr": str(AUTOROUND_MINMAX_LR),
        "nrounds": AUTOROUND_NROUNDS,
        "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "format_type": FORMAT_TYPE,
        "gpu": GPU_NAME,
        "gpu_vram_gb": GPU_VRAM,
    })

    print("\n" + "=" * 60)
    print("[INFO] AutoRound 양자화 시작")
    print(f"       Bits: {BITS}")
    print(f"       Group Size: {GROUP_SIZE}")
    print(f"       Iterations: {AUTOROUND_ITERS}")
    print(f"       Calibration Samples: {NUM_CALIBRATION_SAMPLES}")
    print(f"       Max Seq Length: {MAX_SEQUENCE_LENGTH}")
    print(f"       Format: {FORMAT_TYPE}")
    print("=" * 60)

    quant_start = time.time()

    # ── AutoRound 직접 사용 ──
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=SYMMETRIC,
        iters=AUTOROUND_ITERS,
        lr=AUTOROUND_LR,
        minmax_lr=AUTOROUND_MINMAX_LR,
        nsamples=NUM_CALIBRATION_SAMPLES,
        seqlen=MAX_SEQUENCE_LENGTH,
        nblocks=AUTOROUND_NROUNDS,
    )

    # 양자화 실행
    autoround.quantize()

    quant_time = time.time() - quant_start
    print(f"[INFO] AutoRound 양자화 완료! (소요 시간: {quant_time:.1f}초)")

    # ── 양자화 시간 기록 ──
    mlflow.log_metric("quantization_time_sec", quant_time)

    # =========================================================
    # 6. 모델 저장 (compressed-tensors 포맷)
    # =========================================================
    print(f"\n[INFO] 모델 저장 중... → {OUT_DIR}")
    print(f"       포맷: {FORMAT_TYPE} (vLLM 호환)")

    os.makedirs(OUT_DIR, exist_ok=True)

    # compressed_tensors 포맷으로 저장 → vLLM 0.14.1 직접 로드 가능
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

    # ── 모델 크기 기록 ──
    mlflow.log_metric("model_size_mb", total_size)

    # config.json을 아티팩트로 기록
    config_path = os.path.join(OUT_DIR, "config.json")
    if os.path.exists(config_path):
        mlflow.log_artifact(config_path, artifact_path="model_config")

    # =========================================================
    # 7. ZIP 생성
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

    # ── ZIP 크기 기록 ──
    mlflow.log_metric("zip_size_mb", zip_size)

    # 제출 파일 용량 검증 (10GB 제한)
    if zip_size > 10 * 1024:
        print(f"[WARNING] ZIP 파일이 10GB를 초과합니다! ({zip_size:.1f} MB)")
    else:
        print(f"[INFO] 제출 용량 OK (제한: 10GB, 현재: {zip_size:.1f} MB)")

    # Kaggle에서는 /kaggle/working/ 내 파일이 Output으로 자동 저장됨
    print(f"\n[INFO] Kaggle Output에서 다운로드 가능: {zip_path}")
    print(f"[INFO] 또는 모델 폴더 직접 다운로드: {OUT_DIR}/")

    print(f"\n[INFO] MLflow Run 기록 완료! → https://dagshub.com/{DAGSHUB_REPO}.mlflow")

# =========================================================
# 완료!
# =========================================================
print("\n" + "=" * 60)
print("✅ AutoRound 양자화 완료!")
print("=" * 60)
print(f"""
📊 설정 요약:
   • Model: {MODEL_ID}
   • Bits: {BITS}, Group Size: {GROUP_SIZE}
   • Iterations: {AUTOROUND_ITERS}
   • Calibration Samples: {NUM_CALIBRATION_SAMPLES}
   • Max Seq Length: {MAX_SEQUENCE_LENGTH}
   • Format: {FORMAT_TYPE}

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: {zip_path}

🔧 이전 코드와의 차이:
   • llmcompressor 대신 auto-round 라이브러리 직접 사용
   • group_size, lr, minmax_lr 등 모든 파라미터 직접 제어 가능
   • compressed_tensors 포맷으로 저장 → vLLM 호환

🚀 다운로드된 ZIP 파일을 대회에 제출하세요!
""")
