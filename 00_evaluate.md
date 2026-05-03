# evaluate_local.py 사용 가이드

## 1. 환경 설정

```bash
# 가상환경 생성
conda create -n lg_eval python=3.11 -y
conda activate lg_eval

# PyTorch (CUDA 12.8)
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# 필수 패키지
pip install transformers==4.57.3 accelerate safetensors

# 벤치마크 평가용 (핵심)
pip install lm-eval
```

> `torch==2.9.0`이 설치 안 되면 `pip install torch --index-url https://download.pytorch.org/whl/cu121`로 최신 버전 설치


## 2. 모델 경로 — 코드 수정 불필요

모델 경로는 **실행 시 `--base-model`, `--target-model` 인자**로 지정합니다.
코드 안에 경로가 하드코딩되어 있지 않으므로, **코드 수정 없이** 누구나 사용 가능합니다.

```bash
# 예시: 당신의 경로
python evaluate_local.py ^
  --base-model C:\Users\htw02\project_github\lg_aimers\base_model ^
  --target-model C:\Users\htw02\project_github\lg_aimers\model_DB\optimized_submit\model

# 예시: 다른 사람의 경로
python evaluate_local.py ^
  --base-model /home/user/models/EXAONE-4.0-1.2B ^
  --target-model /home/user/models/my_quantized_model
```


## 3. 배포 시 사용자가 알아야 할 것

| 항목 | 어디서 수정? | 방법 |
|---|---|---|
| 기본 모델 경로 | 명령줄 `--base-model` | 자기 PC의 base model 경로 입력 |
| 양자화 모델 경로 | 명령줄 `--target-model` | 자기 PC의 모델 경로 입력 |
| 벤치마크 종류 | 명령줄 `--tasks` | 기본값 `gsm8k,mmlu` (수정 선택사항) |
| 속도 측정 프롬프트 | **156~169번 줄** `SPEED_PROMPTS` | 필요시 프롬프트 추가/수정 |
| VRAM 자동 dtype | **109번 줄** `if vram_gb < 12:` | VRAM 12GB 미만이면 자동 FP16 적용 |


## 4. 실제 실행 예시

```bash
# Step 1: 기본 모델 baseline 측정 (최초 1회, 약 30~60분 소요)
python evaluate_local.py --base-model ./base_model --mode baseline --skip-speed

# Step 2: 양자화 모델 평가 (baseline 재사용으로 시간 절약)
python evaluate_local.py --baseline-json ./baseline_result.json ^
  --target-model ./model_DB/optimized_submit/model --skip-speed

# Step 3: 여러 모델 한번에 비교
python evaluate_local.py --baseline-json ./baseline_result.json ^
  --target-model ./modelA ./modelB ./modelC --skip-speed
```

> `--skip-speed`를 붙이면 정확도만 비교, 속도 측정 건너뜀 (시간 절약)


## 5. 명령줄 인자 전체 목록

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--base-model` | 기본 모델(EXAONE-4.0-1.2B) 경로 | 없음 (필수) |
| `--target-model` | 양자화 모델 경로 (여러 개 가능) | 없음 (필수) |
| `--mode` | `baseline` 또는 `compare` | `compare` |
| `--tasks` | 벤치마크 태스크 (쉼표 구분) | `gsm8k,mmlu` |
| `--skip-speed` | 속도 측정 생략 | 미사용 |
| `--max-tokens` | 속도 측정 시 최대 생성 토큰 수 | `128` |
| `--baseline-json` | 이전 baseline 결과 JSON 경로 | 없음 |
| `--output` | 결과 저장 경로 | 자동 생성 |


## 6. GPU 요구 사양

EXAONE-4.0-1.2B 기준, 한 번에 모델 1개만 로드하므로:

| GPU VRAM | 실행 가능 여부 |
|---|---|
| 8GB (RTX 4060 등) | ✅ 충분 |
| 6GB | ⚠️ 빠듯하지만 가능 |
| 4GB 이하 | ❌ 부족 |

모델을 10개 비교해도 순서대로 1개씩 로드/해제하므로 **필요 VRAM은 약 3~4GB로 동일**합니다.
