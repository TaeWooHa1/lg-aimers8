# 라벨별 토큰 구간 분석 (MANTA-1M)
# 각 complexity_label(2~9)에 대해 토큰 길이 구간별 개수 출력

from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict

# ── 설정 ──
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
BINS = [0, 256, 512, 768, 1024, 2048, 4096, float("inf")]
BIN_LABELS = ["~256", "257~512", "513~768", "769~1024", "1025~2048", "2049~4096", "4096~"]

# ── 데이터 로드 ──
print("📥 데이터 로드 중...")
ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
TOTAL = len(ds)

# ── 토크나이저 로드 ──
print("🔤 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# ── 라벨별 토큰 길이 분석 ──
print(f"📊 라벨별 토큰 구간 분석 중... ({TOTAL:,}개 전체)\n")

# {label: {bin_label: count}}
label_bin_counts = defaultdict(lambda: defaultdict(int))
label_totals = defaultdict(int)

for i, row in enumerate(ds):
    if (i + 1) % 50000 == 0:
        print(f"  진행: {i+1:,}/{TOTAL:,}")

    label = row["complexity_label"]
    text = tokenizer.apply_chat_template(
        row["conversations"],
        add_generation_prompt=True,
        tokenize=False,
    )
    token_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])

    # 구간 판별
    for j in range(len(BINS) - 1):
        if BINS[j] < token_len <= BINS[j + 1]:
            label_bin_counts[label][BIN_LABELS[j]] += 1
            break

    label_totals[label] += 1

# ── 결과 출력 ──
sorted_labels = sorted(label_totals.keys())

# 헤더
header = f"{'Label':>6} |"
for bl in BIN_LABELS:
    header += f" {bl:>10} |"
header += f" {'합계':>8}"
print(header)
print("-" * len(header))

# 각 라벨 행
grand_bin_totals = defaultdict(int)
for label in sorted_labels:
    row_str = f"{label:>6} |"
    for bl in BIN_LABELS:
        count = label_bin_counts[label][bl]
        row_str += f" {count:>10} |"
        grand_bin_totals[bl] += count
    row_str += f" {label_totals[label]:>8}"
    print(row_str)

# 합계 행
print("-" * len(header))
total_str = f"{'합계':>6} |"
for bl in BIN_LABELS:
    total_str += f" {grand_bin_totals[bl]:>10} |"
total_str += f" {sum(label_totals.values()):>8}"
print(total_str)

# ── 비율 테이블 ──
print(f"\n📊 라벨별 토큰 구간 비율 (행 기준 %)")
header2 = f"{'Label':>6} |"
for bl in BIN_LABELS:
    header2 += f" {bl:>10} |"
print(header2)
print("-" * len(header2))

for label in sorted_labels:
    row_str = f"{label:>6} |"
    total = label_totals[label]
    for bl in BIN_LABELS:
        count = label_bin_counts[label][bl]
        pct = (count / total * 100) if total > 0 else 0
        row_str += f" {pct:>9.1f}% |"
    print(row_str)

# ── 1024 이하 비율 요약 ──
print(f"\n📌 라벨별 1024 토큰 이하 비율 (truncation 없는 데이터):")
for label in sorted_labels:
    under_1024 = sum(
        label_bin_counts[label][bl]
        for bl in ["~256", "257~512", "513~768", "769~1024"]
    )
    total = label_totals[label]
    pct = (under_1024 / total * 100) if total > 0 else 0
    print(f"  Label {label}: {under_1024:>5}/{total:<5} ({pct:.1f}%)")