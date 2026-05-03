# Import

# Windows OpenMP 충돌 해결
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import shutil
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

import mlflow
import dagshub

# ============= DagsHub/MLflow 연동 =============
dagshub.init(
    repo_owner="sthun0211",
    repo_name="LGaimers",
    mlflow=True
)

mlflow.set_experiment("GPTQ_Quantization")

# Setting

MODEL_ID = "./base_model"     
OUT_DIR  = "./model"          

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 512

# Quantization
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head"]
DAMPENING_FRAC = 0.01
BLOCK_SIZE = 32

# ============= MLflow Run 시작 =============
with mlflow.start_run(run_name=f"GPTQ_{SCHEME}_samples{NUM_CALIBRATION_SAMPLES}"):
    
    # 하이퍼파라미터 기록
    mlflow.log_param("model_id", MODEL_ID)
    mlflow.log_param("scheme", SCHEME)
    mlflow.log_param("num_calibration_samples", NUM_CALIBRATION_SAMPLES)
    mlflow.log_param("max_sequence_length", MAX_SEQUENCE_LENGTH)
    mlflow.log_param("dampening_frac", DAMPENING_FRAC)
    mlflow.log_param("block_size", BLOCK_SIZE)
    mlflow.log_param("targets", str(TARGETS))
    mlflow.log_param("ignore", str(IGNORE))
    
    start_time = time.time()
    
    # Model Loads
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
    
    print("[INFO] 모델/토크나이저 로드 완료")
    
    # Dataset Loads & Preprocess
    print("[INFO] 캘리브레이션 데이터 로드 중...")
    
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    )
    
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False)
        }
    
    ds = ds.map(preprocess)
    
    print("[INFO] 데이터 전처리 완료")
    
    # GPTQ Quantization
    print(f"[INFO] GPTQ 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})...")
    
    quantization_start = time.time()
    
    recipe = [
        GPTQModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
            dampening_frac=DAMPENING_FRAC, 
            block_size=BLOCK_SIZE,  
        )
    ]
    
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    
    quantization_time = time.time() - quantization_start
    mlflow.log_metric("quantization_time_seconds", quantization_time)
    
    print("[INFO] GPTQ 완료")
    
    # Model Save
    os.makedirs(OUT_DIR, exist_ok=True)
    
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)
    
    print(f"[INFO] 모델 저장 완료: {OUT_DIR}")
    
    # 필수 파일 복사 (chat_template.jinja)
    src_template = os.path.join(MODEL_ID, "chat_template.jinja")
    dst_template = os.path.join(OUT_DIR, "chat_template.jinja")
    if os.path.exists(src_template):
        shutil.copy(src_template, dst_template)
        print("[INFO] chat_template.jinja 복사 완료")
    
    # 모델 파일 크기 계산 및 기록
    model_size_mb = sum(f.stat().st_size for f in Path(OUT_DIR).glob("*")) / (1024 * 1024)
    mlflow.log_metric("model_size_mb", model_size_mb)
    
    total_time = time.time() - start_time
    mlflow.log_metric("total_time_seconds", total_time)
    
    # Submission
    zip_name = "htw_submit"
    print(f"[INFO] {zip_name}.zip 생성 중...")
    
    shutil.make_archive(
        base_name=zip_name,
        format="zip",
        root_dir=".",
        base_dir=OUT_DIR,
    )
    
    # 아티팩트 기록 (zip 파일)
    mlflow.log_artifact(f"{zip_name}.zip")
    
    # config.json도 기록
    config_path = os.path.join(OUT_DIR, "config.json")
    if os.path.exists(config_path):
        mlflow.log_artifact(config_path)
    
    print(f"[INFO] 생성 완료: {zip_name}.zip")
    print(f"[INFO] MLflow에 실험 기록 완료!")
    print(f"[INFO] 총 소요 시간: {total_time:.2f}초")
    print(f"[INFO] 양자화 시간: {quantization_time:.2f}초")
    print(f"[INFO] 모델 크기: {model_size_mb:.2f} MB")
