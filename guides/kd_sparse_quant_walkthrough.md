# KD + Sparse + Quantization 파이프라인 워크스루

## 구현 완료 내역

리스크를 해결하면서 3단계 파이프라인을 구현했습니다.

### 생성된 파일

| 파일 | 용도 | 단계 |
|------|------|------|
| `23_kd_train.py` | KD 학습 (32B → 1.2B) | Phase 1 |
| `24_sparse_quant.py` | Sparse + Quantization | Phase 2 |
| `25_final_submit.py` | 검증 + ZIP 패키징 | Phase 3 |

---

## 리스크 해결 현황

| 리스크 | 해결 방법 |
|--------|----------|
| **vLLM 2:4 Sparse** | ✅ 지원 확인. `USE_SPARSE` 플래그로 on/off 전환 가능 |
| **Teacher 32B 메모리** | ✅ 4bit + double_quant + 2 GPU 분산 (~16-18GB) |
| **compressed-tensors 0.13.0** | ✅ `25_final_submit.py`에서 version 강제 검증/수정 |
| **Kaggle 12시간 세션** | ✅ 200 step마다 checkpoint 저장 + resume 기능 |
| **KD 모델 없을 때** | ✅ `USE_KD_MODEL=False`로 원본 모델 fallback |

---

## 실행 순서 (Kaggle T4x2)

### 1단계: KD 학습 (`23_kd_train.py`)

```bash
# Kaggle 노트북에서
!pip install -q trl datasets bitsandbytes accelerate
!pip install -q transformers==4.57.3
# → 23_kd_train.py 셀 실행
# → 출력: /kaggle/working/kd_model/
```

> ⚠️ 예상 소요: ~3-8시간. 세션이 끊기면 checkpoint에서 자동 재개됩니다.

### 2단계: Sparse + Quant (`24_sparse_quant.py`)

```bash
# 같은 세션 또는 새 세션에서
!pip install -q llmcompressor
!pip install -q transformers==4.57.3
# → USE_KD_MODEL = True, USE_SPARSE = True 설정
# → 24_sparse_quant.py 셀 실행
# → 출력: /kaggle/working/model/ + submit_kd_sparse_quant.zip
```

### 3단계: 검증 + 제출 (`25_final_submit.py`)

```bash
# → 25_final_submit.py 실행
# → config.json 검증 + submit.zip 생성
# → Dacon에 submit.zip 업로드
```

> 💡 **안전한 순서**: 먼저 `USE_KD_MODEL=False` + `USE_SPARSE=True`로 Sparse+Quant만 테스트 → 점수 확인 → KD 모델 적용

---

## 메모리 구성 (T4x2, 30GB)

```
Teacher (32B, 4bit+double_quant): ~16-18GB (2 GPU 분산)
Student (1.2B, FP16):             ~2.5GB
Optimizer (AdamW):                ~5GB
Gradient + Activations:           ~3-4GB (checkpointing ON)
───────────────────────────────
합계: ~27-30GB / 30GB 가용
```

---

## 평가 서버 호환성 체크리스트

- [x] `vLLM==0.14.1` — compressed-tensors 포맷 호환
- [x] `compressed-tensors==0.13.0` — version 자동 검증
- [x] `transformers==4.57.3` — config.json에 명시
- [x] `max_position_embeddings=32768` — KV Cache 50% 절감
- [x] ZIP 구조: `submit.zip → model/`
- [x] ZIP 용량: ≤ 10GB
- [x] 추론 시간: ≤ 20분

---

## 예상 점수 시나리오

| 시나리오 | PerfNorm | SpeedNorm | **Score** |
|---------|----------|-----------|----------|
| W4A16 최적화만 | 1.03 | 0.51 | **0.77** |
| Sparse + W4A16 | 1.00 | 0.60 | **0.80** |
| KD + W4A16 | 1.07 | 0.51 | **0.79** |
| **KD + Sparse + W4A16** | 1.05 | 0.65 | **0.85** ⭐ |
| **KD + 2:4 Sparse + W4A16** | 1.05 | 0.70 | **0.88** ⭐⭐ |

> ⚠️ 위 점수는 이론적 추정치입니다. 반드시 실제 제출로 검증하세요.
