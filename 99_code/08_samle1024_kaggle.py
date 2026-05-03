"""
GPTQ 양자화 - Kaggle 전용 (캘리브레이션 강화 + 번역 데이터 + KV 캐시 양자화)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기반: 07_sample_kaggle_cali1024.py

변경 사항:
  - 영어→한국어 번역 캘리브레이션 데이터 추가 (TED Talks, CC BY-NC-ND 4.0)
  - MANTA-1M + 번역 데이터 혼합 캘리브레이션
  - max_position_embeddings 축소 (65536 → 32768) 로 KV 캐시 메모리 절감
  - KV 캐시 FP8 양자화 추가 (추론 속도 향상)
  - DagsHub + MLflow 실험 기록

사용법:
1. Kaggle Notebook에서 GPU 가속기 선택 (T4 x2 권장)
2. 아래 코드를 셀에 복사하여 실행
3. 완료 후 Output 탭에서 optimized_submit.zip 다운로드
"""
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# kv cache 8비트 양자화 + 영어데이터셋 캘리브레이션 추가
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# =========================================================
# 0. 패키지 설치 (Kaggle에서 먼저 실행!)
# =========================================================
# !pip install -q llmcompressor transformers datasets accelerate compressed-tensors dagshub mlflow

import os
os.environ['DAGSHUB_USER_TOKEN'] = '1ee266cf0159abb2c8ad8ae564274c6918599acd'
import torch
import shutil
import time
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization import QuantizationArgs
import mlflow
import dagshub

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# DagsHub + MLflow 연결
# =========================================================
dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
mlflow.set_experiment("htw-kvcache-fp8")

# =========================================================
# 1. 경로 설정 (Kaggle용)
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

# =========================================================
# 2. 데이터셋 설정
# =========================================================
# 기존 MANTA-1M 데이터셋
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ⭐ 영어→한국어 번역 데이터셋 (CC BY-NC-ND 4.0)
TRANSLATION_DATASET_ID = "msarmi9/korean-english-multitarget-ted-talks-task"
TRANSLATION_SPLIT = "train"

# ⭐ 캘리브레이션 설정
NUM_MANTA_SAMPLES = 768         # MANTA-1M에서 가져올 샘플 수
NUM_TRANSLATION_SAMPLES = 256   # 번역 데이터에서 가져올 샘플 수
NUM_CALIBRATION_SAMPLES = NUM_MANTA_SAMPLES + NUM_TRANSLATION_SAMPLES  # 총 1024
MAX_SEQUENCE_LENGTH = 512      # 장거리 의존성 반영

# ⭐ max_position_embeddings 축소 (KV 캐시 메모리 절감 → 속도 향상)
MAX_POSITION_EMBEDDINGS = 32768  # 65536 → 32768

# =========================================================
# 3. 양자화 설정
# =========================================================
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
ACTORDER = "static"
DAMPENING_FRAC = 0.01

# ⭐ KV 캐시 FP8 양자화 설정
KV_CACHE_SCHEME = QuantizationArgs(num_bits=8, type="float")

# =========================================================
# 4. GPU 메모리 정리
# =========================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_count = torch.cuda.device_count()
    print(f"[INFO] GPU 수: {gpu_count}")
    for i in range(gpu_count):
        print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"[INFO] VRAM {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] GPU를 찾을 수 없습니다. CPU로 실행됩니다 (매우 느림).")

# =========================================================
# MLflow 실험 기록 시작
# =========================================================
with mlflow.start_run(run_name="W4A16-KVcacheFP8-translation"):

    # 설정값(params) 기록
    mlflow.log_params({
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "translation_dataset_id": TRANSLATION_DATASET_ID,
        "manta_samples": NUM_MANTA_SAMPLES,
        "translation_samples": NUM_TRANSLATION_SAMPLES,
        "calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_seq_length": MAX_SEQUENCE_LENGTH,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "quantization_scheme": SCHEME,
        "targets": str(TARGETS),
        "ignore": str(IGNORE),
        "actorder": ACTORDER,
        "dampening_frac": DAMPENING_FRAC,
        "kv_cache_scheme": "FP8",
    })

    # =========================================================
    # 5. 모델 로드
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
    # 6-A. MANTA-1M 데이터셋 로드 & 전처리
    # =========================================================
    print("\n" + "=" * 60)
    print(f"[INFO] MANTA-1M 캘리브레이션 데이터 로드 중...")
    print(f"       데이터셋: {DATASET_ID}")
    print(f"       샘플 수: {NUM_MANTA_SAMPLES}")
    print("=" * 60)

    ds_manta = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_MANTA_SAMPLES}]",
    )

    def preprocess_manta(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False
            )
        }

    ds_manta = ds_manta.map(preprocess_manta)
    print(f"[INFO] MANTA-1M 전처리 완료 ({len(ds_manta)}개 샘플)")

    # =========================================================
    # 6-B. 영어→한국어 번역 데이터셋 로드 & 전처리
    # =========================================================
    print("\n" + "=" * 60)
    print(f"[INFO] 번역 캘리브레이션 데이터 로드 중...")
    print(f"       데이터셋: {TRANSLATION_DATASET_ID}")
    print(f"       샘플 수: {NUM_TRANSLATION_SAMPLES}")
    print("=" * 60)

    ds_translation = load_dataset(
        TRANSLATION_DATASET_ID,
        split=f"{TRANSLATION_SPLIT}[:{NUM_TRANSLATION_SAMPLES}]",
    )

    def preprocess_translation(example):
        """영어→한국어 번역 프롬프트를 chat template 형식으로 변환"""
        en_text = example.get("en", example.get("english", ""))
        ko_text = example.get("ko", example.get("korean", ""))

        conversations = [
            {"role": "system", "content": "You are a professional English to Korean translator. Translate the following text accurately and naturally."},
            {"role": "user", "content": f"Translate the following English text to Korean:\n\n{en_text}"},
            {"role": "assistant", "content": ko_text},
        ]

        return {
            "text": tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=False,
                tokenize=False
            )
        }

    ds_translation = ds_translation.map(preprocess_translation)
    print(f"[INFO] 번역 데이터 전처리 완료 ({len(ds_translation)}개 샘플)")

    # =========================================================
    # 6-C. 데이터셋 병합
    # =========================================================
    ds_manta_text = ds_manta.select_columns(["text"])
    ds_translation_text = ds_translation.select_columns(["text"])
    ds = concatenate_datasets([ds_manta_text, ds_translation_text])

    print(f"\n[INFO] 캘리브레이션 데이터 병합 완료!")
    print(f"       MANTA-1M: {len(ds_manta_text)}개 + 번역: {len(ds_translation_text)}개 = 총 {len(ds)}개")

    # =========================================================
    # 7. GPTQ 양자화 + KV 캐시 FP8 양자화
    # =========================================================
    print("\n" + "=" * 60)
    print("[INFO] GPTQ 양자화 시작 (+ KV 캐시 FP8 양자화)")
    print(f"       Scheme: {SCHEME}")
    print(f"       ActOrder: {ACTORDER}")
    print(f"       Dampening: {DAMPENING_FRAC}")
    print(f"       KV Cache: FP8 양자화 ⭐")
    print(f"       Calibration: {NUM_CALIBRATION_SAMPLES} samples (MANTA {NUM_MANTA_SAMPLES} + 번역 {NUM_TRANSLATION_SAMPLES})")
    print(f"       Max Seq Length: {MAX_SEQUENCE_LENGTH}")
    print("=" * 60)

    recipe = [
        GPTQModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
            actorder=ACTORDER,
            dampening_frac=DAMPENING_FRAC,
            kv_cache_scheme=KV_CACHE_SCHEME,  # ⭐ KV 캐시 FP8 양자화
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

    quantization_time = time.time() - start_time
    print(f"[INFO] GPTQ 양자화 완료! (소요 시간: {quantization_time:.1f}초)")

    # 양자화 시간 기록
    mlflow.log_metric("quantization_time_sec", quantization_time)

    # =========================================================
    # 8. 모델 저장
    # =========================================================
    # ⭐ max_position_embeddings 축소 (KV 캐시 메모리 절감 → 속도 향상)
    model.config.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    print(f"[INFO] max_position_embeddings 설정: {MAX_POSITION_EMBEDDINGS}")

    print(f"\n[INFO] 모델 저장 중... → {OUT_DIR}")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    # 저장 확인
    print("[INFO] 저장된 파일:")
    for f in os.listdir(OUT_DIR):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
        print(f"       - {f} ({size:.1f} MB)")

    # =========================================================
    # 9. ZIP 생성 (Kaggle Output으로 자동 저장)
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

    # ZIP 파일 크기 기록
    mlflow.log_metric("model_zip_size_MB", zip_size)

    # GPU 정보 기록
    if torch.cuda.is_available():
        mlflow.log_param("gpu_count", torch.cuda.device_count())
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        mlflow.log_metric("gpu_vram_GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    print("[INFO] MLflow 기록 완료!")

    # =========================================================
    # 완료!
    # =========================================================
    print("\n" + "=" * 60)
    print("✅ 양자화 완료!")
    print("=" * 60)
    print(f"""
📊 설정 요약:
   • Model: {MODEL_ID}
   • Scheme: {SCHEME} + KV Cache FP8 ⭐
   • ActOrder: {ACTORDER}
   • Dampening: {DAMPENING_FRAC}
   • Samples: {NUM_CALIBRATION_SAMPLES} (MANTA {NUM_MANTA_SAMPLES} + 번역 {NUM_TRANSLATION_SAMPLES})
   • Max Length: {MAX_SEQUENCE_LENGTH}
   • Max Position Embeddings: {MAX_POSITION_EMBEDDINGS}
   • 양자화 시간: {quantization_time:.1f}초

📁 출력:
   • 모델: {OUT_DIR}/
   • ZIP: {zip_path} ({zip_size:.1f} MB)

🚀 Kaggle Notebook 우측 Output 탭에서 ZIP 파일을 다운로드하여 대회에 제출하세요!
📊 DagsHub에서 실험 기록 확인: https://dagshub.com/sthun0211/LGaimers.mlflow
""")
