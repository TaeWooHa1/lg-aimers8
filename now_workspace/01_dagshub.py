import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import shutil
from pathlib import Path

# MLflow 관련 라이브러리 추가
import mlflow
import dagshub

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# --- DagsHub 연결 및 실험 세팅 ---
dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
mlflow.set_experiment("htw_Quantization_Experiment")

# 기존 설정값들
MODEL_ID = "../base_model"     
OUT_DIR  = "./model"          
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# 양자화 설정 (00_sample_local_0.57.py 기준)
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head"]
ACTORDER = "group"
DAMPENING_FRAC = 0.01

# --- MLflow 기록 시작 ---
with mlflow.start_run(run_name=f"GPTQ_{SCHEME}_actorder-{ACTORDER}"):
    
    # 1. 설정값 기록 (어떤 세팅으로 양자화했는지 남김)
    mlflow.log_params({
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_seq_length": MAX_SEQUENCE_LENGTH,
        "quantization_scheme": SCHEME,
        "targets": str(TARGETS),
        "ignore": str(IGNORE),
        "actorder": ACTORDER,
        "dampening_frac": DAMPENING_FRAC,
    })

    # 2. 모델 & 토크나이저 로드
    print("[INFO] 모델 로드 중...")

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

    print(f"[INFO] 모델 로드 완료! 파라미터: {model.num_parameters() / 1e9:.2f}B")

    # 3. 데이터셋 로드 & 전처리
    print(f"[INFO] 캘리브레이션 데이터 로드 중... ({NUM_CALIBRATION_SAMPLES}개)")
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

    # 4. GPTQ 양자화
    print(f"[INFO] GPTQ 양자화 시작 (Scheme: {SCHEME}, ActOrder: {ACTORDER})")
    recipe = [
        GPTQModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
            actorder=ACTORDER,
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

    # 5. Model Save 및 Submission 파일 생성
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)
    
    zip_name = "baseline_submit"
    shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=OUT_DIR)
    print(f"[INFO] 생성 완료: {zip_name}.zip")

    # 6. 결과 지표 기록
    file_size_mb = os.path.getsize(f"{zip_name}.zip") / (1024 * 1024)
    mlflow.log_metric("model_zip_size_MB", file_size_mb)
    print(f"[INFO] MLflow 기록 완료 (파일 크기: {file_size_mb:.2f} MB)")

    # (선택 사항) 완성된 zip 파일을 DagsHub 서버에 바로 업로드하고 싶다면 아래 주석 해제
    # mlflow.log_artifact(f"{zip_name}.zip")