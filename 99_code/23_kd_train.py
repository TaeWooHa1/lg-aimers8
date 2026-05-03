# %%
"""
Knowledge Distillation: EXAONE-4.0-32B (Teacher) → EXAONE-4.0-1.2B (Student)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kaggle T4x2 환경 (VRAM 15GB × 2 = 30GB)

전략:
  - Teacher (32B): 4bit 로드 + device_map="auto" (2 GPU 분산, ~16-18GB)
  - Student (1.2B): FP16 로드 (학습 대상, ~2.5GB)  
  - GKD (Generalized Knowledge Distillation) via trl
  - gradient_checkpointing + batch=1 + grad_accum=16 으로 메모리 최적화

출력: /kaggle/working/kd_model/ (KD 학습된 Student 모델)

Kaggle 사용법:
  !pip install -q trl datasets bitsandbytes accelerate
  !pip install -q transformers==4.57.3
"""

# %% [셀 1] 패키지 설치 (Kaggle에서 먼저 실행)
# !pip install -q trl datasets bitsandbytes accelerate
# !pip install -q transformers==4.57.3

# %% [셀 2] 라이브러리 임포트
import os
import time
import torch
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

print("[INFO] 라이브러리 로드 완료")
print(f"[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f}GB)")

# %% [셀 3] 설정
TEACHER_ID = "LGAI-EXAONE/EXAONE-4.0-32B"
STUDENT_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/kd_model"

# ── 데이터 설정 ──
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_TRAIN_SAMPLES = 5000          # ⚠️ T4x2 메모리 + 12시간 세션 고려
NUM_EVAL_SAMPLES = 200

# ── 학습 설정 (T4x2 메모리 최적화) ──
MAX_SEQ_LENGTH = 256              # ⚠️ 32B Teacher 때문에 짧게 제한
PER_DEVICE_BATCH_SIZE = 1         # ⚠️ 메모리 최소화
GRADIENT_ACCUMULATION = 16        # 실효 배치 = 1 × 16 = 16
NUM_EPOCHS = 2                    # 12시간 세션 고려
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# ── GKD 설정 ──
GKD_LMBDA = 0.5                   # on-policy 비율 (0.5 = 반반)
GKD_BETA = 0.5                    # JSD 보간 (0.5 = forward/reverse KL 중간)

# ── Checkpoint 설정 (세션 복구용) ──
SAVE_STEPS = 200                  # 200 step마다 checkpoint 저장
RESUME_FROM_CHECKPOINT = True     # 이전 checkpoint에서 이어서 학습

print(f"""
📊 KD 학습 설정:
┌──────────────────────────────────────────┐
│ Teacher: {TEACHER_ID.split('/')[-1]:>25} │
│ Student: {STUDENT_ID.split('/')[-1]:>25} │
├──────────────────────────────────────────┤
│ 데이터: {NUM_TRAIN_SAMPLES:>6} train / {NUM_EVAL_SAMPLES:>4} eval      │
│ max_seq_length: {MAX_SEQ_LENGTH:>4}                     │
│ batch: {PER_DEVICE_BATCH_SIZE} × accum {GRADIENT_ACCUMULATION} = {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION:>3} effective │
│ epochs: {NUM_EPOCHS}                                │
│ lr: {LEARNING_RATE}                            │
├──────────────────────────────────────────┤
│ GKD lmbda: {GKD_LMBDA} (on-policy 비율)          │
│ GKD beta:  {GKD_BETA} (JSD 보간)                │
│ Checkpoint: every {SAVE_STEPS} steps            │
└──────────────────────────────────────────┘
""")

# %% [셀 4] Teacher 모델 로드 (4bit, 2 GPU 분산)
print(f"[1/6] Teacher 모델 로드: {TEACHER_ID}")
print("  → 4bit 양자화 + 2 GPU 분산 (약 16-18GB)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,     # T4는 bfloat16 미지원
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,           # 이중 양자화로 추가 메모리 절약
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID,
    quantization_config=bnb_config,
    device_map="auto",                        # 2 GPU에 자동 분산
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
teacher_model.eval()

# Teacher 메모리 사용량 확인
teacher_mem = sum(p.numel() * p.element_size() for p in teacher_model.parameters()) / 1e9
print(f"  → Teacher 로드 완료! (파라미터 메모리: ~{teacher_mem:.1f}GB)")

for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1e9
    print(f"  → GPU {i} 할당: {allocated:.2f}GB")

# %% [셀 5] Student 모델 로드 (FP16, 학습 대상)
print(f"\n[2/6] Student 모델 로드: {STUDENT_ID}")

student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# gradient checkpointing 활성화 (메모리 절약)
student_model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

student_params = student_model.num_parameters() / 1e9
print(f"  → Student 로드 완료! ({student_params:.2f}B 파라미터)")
print(f"  → gradient_checkpointing: ON")

for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved = torch.cuda.memory_reserved(i) / 1e9
    print(f"  → GPU {i} 할당: {allocated:.2f}GB / 예약: {reserved:.2f}GB")

# %% [셀 6] 데이터 로드 + 전처리
print(f"\n[3/6] 학습 데이터 {NUM_TRAIN_SAMPLES}개 로드")

# ── MANTA-1M 데이터 로드 ──
train_ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_TRAIN_SAMPLES}]"
)
eval_ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[{NUM_TRAIN_SAMPLES}:{NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES}]"
)

def convert_to_messages(example):
    """MANTA-1M의 conversations 형식을 messages 형식으로 변환"""
    conversations = example["conversations"]
    messages = []
    for turn in conversations:
        role = turn.get("role", turn.get("from", "user"))
        content = turn.get("content", turn.get("value", ""))
        # role 정규화
        if role in ["human", "user"]:
            role = "user"
        elif role in ["gpt", "assistant", "model"]:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "user"
        messages.append({"role": role, "content": content})
    return {"messages": messages}

train_ds = train_ds.map(convert_to_messages, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(convert_to_messages, remove_columns=eval_ds.column_names)

print(f"  → Train: {len(train_ds)}개 / Eval: {len(eval_ds)}개")
print(f"  → 샘플 확인: {train_ds[0]['messages'][:2]}")

# %% [셀 7] GKD 학습 실행
print(f"\n[4/6] GKD Knowledge Distillation 학습 시작")
print(f"  ⚠️ 예상 소요 시간: {NUM_TRAIN_SAMPLES * NUM_EPOCHS / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION) * 2 / 3600:.1f}~{NUM_TRAIN_SAMPLES * NUM_EPOCHS / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION) * 5 / 3600:.1f}시간")

from trl.experimental.gkd import GKDConfig, GKDTrainer

# Checkpoint 경로 확인
checkpoint_dir = os.path.join(OUT_DIR, "checkpoints")
resume_checkpoint = None
if RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        resume_checkpoint = os.path.join(checkpoint_dir, latest)
        print(f"  → 이전 checkpoint에서 이어서 학습: {resume_checkpoint}")

training_args = GKDConfig(
    output_dir=checkpoint_dir,
    
    # ── 학습 하이퍼파라미터 ──
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    
    # ── 메모리 최적화 ──
    fp16=True,                                # T4는 bfloat16 미지원
    gradient_checkpointing=True,              # Activation 메모리 절약
    max_seq_length=MAX_SEQ_LENGTH,
    
    # ── GKD 설정 ──
    lmbda=GKD_LMBDA,
    beta=GKD_BETA,
    
    # ── 저장/로깅 ──
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,                       # 최대 3개 checkpoint 유지
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    logging_steps=10,
    logging_first_step=True,
    
    # ── 기타 ──
    report_to="none",                         # Kaggle에서는 wandb 등 불필요
    remove_unused_columns=False,
    seed=42,
    dataloader_num_workers=2,
)

start_time = time.time()

trainer = GKDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

# 학습 실행
trainer.train(resume_from_checkpoint=resume_checkpoint)

train_time = time.time() - start_time
print(f"\n  → KD 학습 완료! ({train_time / 3600:.1f}시간)")

# %% [셀 8] 학습된 모델 저장
print(f"\n[5/6] KD 모델 저장: {OUT_DIR}")

# 최종 모델 저장 (checkpoint가 아닌 최종 모델)
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# 저장 확인
total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*.safetensors"))
print(f"  → 모델 크기: {total_size / (1024**3):.2f} GB")
print(f"  → 저장 완료!")

# 파일 목록
print("\n[INFO] 저장된 파일:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size:.1f} MB)")

# %% [셀 9] 완료 요약
print(f"""
{'='*60}
✅ Knowledge Distillation 학습 완료!
{'='*60}

📊 학습 정보:
  Teacher: {TEACHER_ID} (4bit)
  Student: {STUDENT_ID} (FP16)
  데이터: {NUM_TRAIN_SAMPLES}개 × {NUM_EPOCHS} epochs
  소요: {train_time / 3600:.1f}시간

📁 출력: {OUT_DIR}

🔜 다음 단계:
  1. 24_sparse_quant.py 에서 USE_KD_MODEL = True 로 설정
  2. KD 모델 경로를 /kaggle/working/kd_model 로 지정
  3. Sparse + Quantization 적용 후 제출
  
  또는 이 모델을 먼저 바로 양자화 없이 제출하여
  PerfNorm > 1.0 인지 확인하는 것을 추천합니다.
""")
