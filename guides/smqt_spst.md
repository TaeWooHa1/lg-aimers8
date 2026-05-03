# SmoothQuant & 2:4 Sparsity 정리

---

## 1. SmoothQuant

### 한줄 요약
> **Activation의 outlier를 Weight 쪽으로 수학적으로 이동**시켜, W8A8 양자화의 정확도를 높이는 전처리 기법

### 문제: W8A8의 Activation Outlier

INT8은 [-128, 127] 범위만 표현 가능. Activation에 outlier가 있으면:

```
Activation: [0.1, 0.2, 0.3, 100.0]  ← 100.0이 outlier

INT8 변환 (scale = 100/127 ≈ 0.787):
  0.1 → round(0.1/0.787) = 0   ← 0.1이 0으로 뭉개짐!
  0.2 → round(0.2/0.787) = 0   ← 0.2도 0으로 뭉개짐!
  0.3 → round(0.3/0.787) = 0
  100 → round(100/0.787) = 127

→ outlier 하나 때문에 나머지 값이 전부 정보 손실
```

### 해결: 수학적 등가 변환

**핵심 아이디어**: `Y = X · W`에서 X와 W 사이에 스무딩 팩터 `s`를 삽입

```
원래:  Y = X · W
변환:  Y = (X · diag(s)⁻¹) · (diag(s) · W)
       = X_smooth · W_smooth

결과가 수학적으로 완전히 같음 (Y는 변하지 않음)
하지만 X_smooth는 outlier가 줄고,
      W_smooth는 범위가 커지지만 양자화하기 더 쉬움
```

### smoothing_strength (α) 파라미터

```
s = max(|X|)^α / max(|W|)^(1-α)

α = 0.0: 스무딩 안 함 (원래 그대로)
α = 0.5: X와 W 사이 양자화 난이도 균등 분배 ← 최적
α = 1.0: outlier를 전부 W로 이동 (W가 양자화하기 어려워짐)
```

### 적용 전후 비교

```
[ 적용 전 ]
Activation: [0.1, 0.2, 100.0]  → INT8로 뭉개짐 ❌
Weight:     [0.5, 0.3, 0.1]    → INT8로 잘 변환됨 ✅

[ SmoothQuant 적용 후 ]
Activation: [0.1, 0.2, 1.0]    → INT8로 잘 변환됨 ✅
Weight:     [50,  30,  10]     → 범위 커졌지만 여전히 양자화 가능 ✅
```

### 코드 적용 방법

```python
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),  # ① 먼저 스무딩
    GPTQModifier(scheme="W8A8", ...),              # ② 그 다음 양자화
]
```

### 특성 정리

| 항목 | 내용 |
|------|------|
| **역할** | Activation outlier 완화 → W8A8 정확도 ↑ |
| **영향** | PerfNorm ↑ (정확도 향상) |
| **추론 속도** | 영향 없음 (양자화 전처리일 뿐) |
| **추가 비용** | 양자화 시간 약간 증가 |
| **최적 α** | 0.5 (논문에서 증명) |
| **W4A16에서** | 불필요 (Activation을 양자화하지 않으므로) |
| **W8A8에서** | **강력 추천** ⭐ (Activation INT8 양자화의 핵심) |

---

## 2. 2:4 Sparsity

### 한줄 요약
> Weight 행렬에서 **4개 값 중 2개를 0으로 만들어** 연산량을 50% 줄이는 구조적 가지치기 기법

### 동작 방식

```
원본 Weight:
  [0.3, 0.1, 0.5, 0.2,  0.4, 0.7, 0.1, 0.3]

2:4 Sparsity 적용 (4개 중 작은 2개를 0으로):
  [0.3, 0.0, 0.5, 0.0,  0.0, 0.7, 0.0, 0.3]
       ↑         ↑       ↑         ↑
      제거      제거     제거      제거
```

### 왜 "2:4"인가?

```
4개 연속 값 중 반드시 2개만 남기는 규칙적 패턴

[■ □ ■ □] [□ ■ □ ■] [■ □ □ ■] ...
 ■=유지  □=0

→ 하드웨어(GPU)가 이 규칙적 패턴을 인식하여 연산 최적화
→ 불규칙 sparsity와 달리 실제 속도 향상이 보장됨
```

### GPU 하드웨어 지원

NVIDIA Ampere 이상(A100, L4 등)의 **Sparse Tensor Core**가 2:4 패턴을 인식:

```
일반 행렬곱:     A × B = C     (모든 원소 계산)
2:4 Sparse 행렬곱: A_sparse × B = C  (0인 부분 건너뜀)

→ 이론적 2배 속도 향상
→ 실제로는 ~1.3~1.5배 (오버헤드 존재)
```

| GPU | 2:4 Sparse 지원 |
|-----|-----------------|
| T4 (Kaggle) | ❌ 미지원 (Turing 세대) |
| A100 | ✅ 지원 (Ampere 세대) |
| **L4 (평가서버)** | **✅ 지원 (Ada Lovelace 세대)** |

### Sparsity 적용 과정

```
원본 모델 (FP16)
    │
    ├─ 1단계: SparseGPT / Wanda 등으로 2:4 패턴 적용
    │         → 중요도 낮은 weight 2개를 0으로 설정
    │         → 남은 weight에 오차 보상 적용
    │
    ├─ 2단계: (선택) Fine-tuning으로 정확도 복구
    │
    └─ 3단계: GPTQ 양자화 (INT8)
    │
    ▼
최종: 2:4 Sparse + INT8 모델
  → 속도: Sparse Tensor Core + INT8 Tensor Core 동시 활용
  → 크기: 0인 값은 저장하지 않아 모델 크기도 절감
```

### 코드 적용 방법

```python
from llmcompressor.modifiers.pruning import SparseGPTModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),       # ① 스무딩
    SparseGPTModifier(sparsity=0.5, targets=["Linear"], # ② 2:4 Sparsity
                      mask_structure="2:4"),
    GPTQModifier(scheme="W8A8", ...),                  # ③ 양자화
]
```

### 특성 정리

| 항목 | 내용 |
|------|------|
| **역할** | Weight의 50%를 0으로 만들어 연산량 감소 |
| **영향** | SpeedNorm ↑ (추론 속도 향상) |
| **정확도** | 약간 하락 (PerfNorm ↓) |
| **속도 향상** | 이론 2배, 실제 ~1.3~1.5배 |
| **하드웨어** | Ampere 이상 (A100, L4) 필요 |
| **평가서버** | L4 → **지원됨** ✅ |

---

## 3. SmoothQuant vs Sparsity 비교

| | SmoothQuant | 2:4 Sparsity |
|---|---|---|
| **목적** | 정확도 보호 | 속도 향상 |
| **대상** | Activation 분포 | Weight 값 |
| **방법** | Outlier를 재분배 | 작은 값을 0으로 |
| **Score 영향** | PerfNorm ↑ | SpeedNorm ↑ |
| **단독 사용** | 가능 | 가능 |
| **조합** | ✅ 함께 사용 가능 | ✅ 함께 사용 가능 |
| **추가 비용** | 거의 없음 | 정확도 약간 하락 |

---

## 4. 전체 파이프라인 적용 순서

```
원본 모델 (EXAONE-4.0-1.2B, FP16)
    │
    ▼
① SmoothQuant (α=0.5)
    → Activation outlier를 Weight로 이동
    → 양자화 준비 (수학적 등가 변환)
    │
    ▼
② 2:4 Sparsity (SparseGPT)
    → 4개 중 2개 weight를 0으로
    → 나머지 weight에 오차 보상
    │
    ▼
③ GPTQ W8A8 양자화
    → 남은 non-zero weight를 INT8로 양자화
    → Hessian 기반 오류 보상
    │
    ▼
④ config.json 패치
    → max_position_embeddings=32768
    → compressed-tensors version=0.13.0
    │
    ▼
최종 모델: SmoothQuant + 2:4 Sparse + W8A8
    → PerfNorm: SmoothQuant로 정확도 보호 ✅
    → SpeedNorm: Sparsity + INT8로 속도 향상 ✅
```

### 예상 Score 시나리오

```
Score = 0.5 × PerfNorm + 0.5 × SpeedNorm

| 전략                              | PerfNorm | SpeedNorm | Score  |
|-----------------------------------|----------|-----------|--------|
| W8A8만                            | ~1.02    | ~0.55     | ~0.79  |
| W8A8 + SmoothQuant                | ~1.05    | ~0.55     | ~0.80  |
| W8A8 + Sparsity                   | ~0.98    | ~0.70     | ~0.84  |
| W8A8 + SmoothQuant + Sparsity ⭐  | ~1.02    | ~0.70     | ~0.86  |

⚠️ 위 점수는 이론적 추정치. 반드시 실제 제출로 검증 필요.
```
