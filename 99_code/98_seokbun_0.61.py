import os
import json
os.environ['DAGSHUB_USER_TOKEN'] = '1ee266cf0159abb2c8ad8ae564274c6918599acd'
import torch
import shutil
import time
import mlflow
import dagshub

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# DagsHub + MLflow
# =========================================================
dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
mlflow.set_experiment("htw-int8-w8a8")

# =========================================================
# 설정
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

SCHEME = "W8A8"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
DAMPENING_FRAC = 0.01

# ⭐ config.json 최적화 설정
MAX_POSITION_EMBEDDINGS = 32768   # 65536 → 32768 (KV Cache 50% 절감)

# =========================================================
# GPU 확인
# =========================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[WARNING] GPU 없음")

# =========================================================
# MLflow 실행
# =========================================================
with mlflow.start_run(run_name="int8-w8a8-maxpos32k"):

    mlflow.log_params({
        "model_id": MODEL_ID,
        "calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_seq_length": MAX_SEQUENCE_LENGTH,
        "scheme": SCHEME,
        "dampening_frac": DAMPENING_FRAC,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
    })

# =========================================================
# 모델 로드
# =========================================================
    print(f"\n[INFO] 모델 로드: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

# =========================================================
# 데이터셋
# =========================================================
    print(f"\n[INFO] 데이터 로드 ({NUM_CALIBRATION_SAMPLES} 샘플)")
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

# =========================================================
# INT8 W8A8 양자화
# =========================================================
    print(f"\n[INFO] INT8 W8A8 양자화 시작")

    recipe = [
        GPTQModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
            dampening_frac=DAMPENING_FRAC,
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

    elapsed = time.time() - start_time
    print(f"[INFO] 양자화 완료! ({elapsed:.1f}초)")
    mlflow.log_metric("quantization_time_sec", elapsed)

# =========================================================
# 저장
# =========================================================
    print(f"\n[INFO] 저장: {OUT_DIR}")
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

# =========================================================
# ⭐ config.json max_position_embeddings 수정
# =========================================================
    print(f"\n[INFO] config.json 최적화 중…")
    config_path = os.path.join(OUT_DIR, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    original_max_pos = config.get("max_position_embeddings", "N/A")
    config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[INFO] max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")
    print(f"[INFO] KV Cache 메모리 {(1 - MAX_POSITION_EMBEDDINGS/65536)*100:.0f}% 절감 예상!")
    mlflow.log_param("original_max_position_embeddings", original_max_pos)

# =========================================================
# ZIP 생성
# =========================================================
    zip_name = "int8_w8a8_maxpos32k"
    shutil.make_archive(f"/kaggle/working/{zip_name}", "zip", "/kaggle/working", "model")

    zip_size = os.path.getsize(f"/kaggle/working/{zip_name}.zip") / (1024 * 1024)
    mlflow.log_metric("model_zip_size_MB", zip_size)

    print(f"""
✅ 완료!
📊 INT8 W8A8 + max_position_embeddings 최적화
   • scheme: {SCHEME}
   • calibration: {NUM_CALIBRATION_SAMPLES}
   • max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}
   • KV Cache 메모리 절감: ~50%

📁 {zip_name}.zip ({zip_size:.1f} MB)
""")
