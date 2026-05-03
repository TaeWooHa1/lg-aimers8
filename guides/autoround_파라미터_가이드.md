# AutoRound 양자화 파라미터 가이드

> 현재 코드 (`10_autoaround.py`) 기준으로, 추가/수정할 수 있는 모든 파라미터를 정리합니다.

---

## 1. 현재 사용 중인 파라미터 (기본 코드)

| 파라미터 | 현재 값 | 기본값 | 역할 |
|---|---|---|---|
| `scheme` | `"W4A16"` | - | 가중치 비트 / Activation 비트 |
| `targets` | `["Linear"]` | `["Linear"]` | 양자화 대상 레이어 타입 |
| `ignore` | `["lm_head"]` | `[]` | 양자화 제외 레이어 |
| `group_size` | `128` | `128` | 그룹 양자화 단위 크기 |
| `num_rounds` | `200` | `200` | 블록 내 라운딩 최적화 반복 수 |
| `iters` | `200` | `200` | SignSGD 최적화 스텝 수 |
| `lr` | `0.0025` | `None` (=1/iters) | 라운딩 파라미터 학습률 |
| `minmax_lr` | `0.0025` | `None` (=lr) | 클리핑 범위 학습률 |

---

## 2. 추가 가능한 AutoRoundModifier 파라미터

### 🔴 정확도에 직접 영향

| 파라미터 | 타입 | 기본값 | 설명 | 권장 실험 범위 |
|---|---|---|---|---|
| `iters` | int | `200` | SignSGD 반복 수. **가장 중요한 파라미터**. 높으면 정확도 ↑, 양자화 시간 ↑ | 200 ~ 1000 |
| `minmax_lr` | float | `None` | 클리핑 범위 학습률. `0`으로 설정 시 클리핑 최적화 비활성화 (GPTQ처럼 동작) | 0.001 ~ 0.005 |
| `lr` | float | `None` | 라운딩 학습률. `None`이면 자동으로 `1/iters`로 설정 | 0.001 ~ 0.01 |
| `num_rounds` | int | `200` | 전체 라운딩 반복 수. 보통 `iters`보다 덜 민감 | 100 ~ 300 |

### 🟡 모델 구조/크기에 영향

| 파라미터 | 타입 | 기본값 | 설명 | 권장 실험 범위 |
|---|---|---|---|---|
| `group_size` | int | `128` | 그룹 양자화 크기. 작을수록 정밀↑ 모델크기↑ | 32, 64, **128**, 256 |
| `scheme` | str | - | 양자화 비트 수. 낮을수록 작은 모델, 낮은 정확도 | `"W4A16"`, `"W8A16"`, `"W3A16"` |
| `ignore` | list | `[]` | 양자화 제외 레이어. 많이 제외할수록 정확도↑ 모델크기↑ | 아래 별도 설명 |

### 🟢 메모리/성능 관련

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `symmetric` | bool | `True` | 대칭 양자화 여부. `False`면 비대칭 (약간 더 정밀, 약간 더 느림) |
| `block_size` | int | `128` | 양자화 블록 크기 (GPTQ의 block_size와 유사) |

---

## 3. `oneshot()` 함수 파라미터

`AutoRoundModifier` 외에 `oneshot()` 함수에 전달하는 파라미터도 결과에 영향을 줍니다:

| 파라미터 | 현재 값 | 설명 | 권장 |
|---|---|---|---|
| `num_calibration_samples` | `1024` | 캘리브레이션 샘플 수. **많을수록 정확도 ↑** | 256 ~ 1024 |
| `max_seq_length` | `512` | 최대 시퀀스 길이. 길수록 정확도 ↑, 메모리 ↑ | 256 ~ 2048 |

> ⚠️ 이 두 파라미터는 **양자화 시에만** 사용되며, 추론 속도에는 영향 없음

---

## 4. `ignore` 레이어 선별 옵션

EXAONE-4.0-1.2B 모델은 30개 레이어 (`model.layers.0` ~ `model.layers.29`)를 가집니다.

```python
# 기본 (현재)
IGNORE = ["lm_head"]

# 보수적 (초기/마지막 레이어 보호)
IGNORE = ["lm_head", "model.layers.0", "model.layers.1", "model.layers.29"]

# 공격적 (더 많은 레이어 보호 → 모델 크기 증가)
IGNORE = ["lm_head", "model.layers.0", "model.layers.1", "model.layers.2",
          "model.layers.27", "model.layers.28", "model.layers.29"]
```

| 전략 | PerfNorm | 모델 크기 | SpeedNorm |
|---|---|---|---|
| `lm_head`만 | 보통 | 최소 | 최대 |
| + 앞뒤 2~3개 | 높음 | 약간 증가 | 거의 동일 |
| + 앞뒤 5개+ | 매우 높음 | 상당히 증가 | 약간 감소 |

---

## 5. 실험 전략 제안

### 전략 A: 정확도 극대화 (PerfNorm 최대)
```python
AUTOROUND_ITERS = 500           # 반복 수 증가
AUTOROUND_MINMAX_LR = 0.003     # 클리핑 범위 학습률 약간 증가
NUM_CALIBRATION_SAMPLES = 1024  # 캘리브레이션 샘플 최대
MAX_SEQUENCE_LENGTH = 1024      # 시퀀스 길이 증가
GROUP_SIZE = 64                 # 그룹 크기 감소 (더 정밀)
IGNORE = ["lm_head", "model.layers.0", "model.layers.29"]
```

### 전략 B: 균형 (현재 코드와 유사)
```python
AUTOROUND_ITERS = 200
AUTOROUND_MINMAX_LR = 0.0025
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 512
GROUP_SIZE = 128
IGNORE = ["lm_head"]
```

### 전략 C: 속도 최적화 (SpeedNorm 최대)
```python
AUTOROUND_ITERS = 100           # 빠른 양자화
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512
GROUP_SIZE = 128                # 기본 유지
IGNORE = ["lm_head"]            # 최소한만 제외
```

---

## 6. 파라미터 영향도 순위

정확도(PerfNorm)에 미치는 영향이 큰 순서:

```
1. scheme (W4A16 vs W8A16)     ★★★★★  ← 비트 수 자체가 가장 큰 영향
2. iters                       ★★★★☆  ← SignSGD 반복 수
3. minmax_lr                   ★★★★☆  ← 클리핑 범위 최적화
4. num_calibration_samples     ★★★☆☆  ← 캘리브레이션 데이터 양
5. group_size                  ★★★☆☆  ← 그룹 크기
6. ignore (레이어 제외)         ★★☆☆☆  ← 모델 크기 증가 동반
7. max_seq_length              ★★☆☆☆  ← 긴 문맥 정확도
8. lr                          ★☆☆☆☆  ← 보통 기본값 유지
9. num_rounds                  ★☆☆☆☆  ← iters보다 덜 민감
```
