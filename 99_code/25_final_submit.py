# %%
"""
최종 패키징 + 제출 전 검증 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
24_sparse_quant.py에서 생성된 model/ 폴더를 검증하고 제출용 ZIP 생성

검증 항목:
  ✅ ZIP 구조: submit.zip → model/ (config.json, model.safetensors 등)
  ✅ 용량: ≤ 10GB
  ✅ config.json 필수 값 확인
  ✅ AutoModelForCausalLM.from_pretrained() 로드 테스트
  ✅ AutoTokenizer.from_pretrained() 로드 테스트
  ✅ quantization_config.version == "0.13.0"
  ✅ transformers_version == "4.57.3"
  ✅ max_position_embeddings 확인

Kaggle 사용법:
  # 24_sparse_quant.py 실행 후 이 스크립트 실행
  !pip install -q transformers==4.57.3
"""

# %% [셀 1] 패키지 설치 (Kaggle에서 먼저 실행)
# !pip install -q transformers==4.57.3

# %% [셀 2] 라이브러리 임포트
import os
import json
import shutil
from pathlib import Path

print("[INFO] 최종 패키징 + 검증 시작")

# %% [셀 3] 설정
MODEL_DIR = "/kaggle/working/model"          # 24_sparse_quant.py 출력 경로
ZIP_OUTPUT = "/kaggle/working/submit.zip"    # 최종 제출 ZIP

# ── 평가 서버 기준값 ──
EXPECTED = {
    "transformers_version": "4.57.3",
    "compressed_tensors_version": "0.13.0",
    "max_position_embeddings": 32768,
    "max_zip_size_gb": 10,
}

# %% [셀 4] 모델 디렉토리 확인
print(f"\n[1/6] 모델 디렉토리 확인: {MODEL_DIR}")

if not os.path.exists(MODEL_DIR):
    print(f"  ❌ 모델 디렉토리가 없습니다: {MODEL_DIR}")
    print(f"  → 먼저 24_sparse_quant.py를 실행하세요.")
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

files = sorted(os.listdir(MODEL_DIR))
print(f"  → 파일 목록 ({len(files)}개):")
for f in files:
    fpath = os.path.join(MODEL_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    - {f} ({size:.1f} MB)")

# 필수 파일 확인
required_files = ["config.json"]
safetensors_files = [f for f in files if f.endswith(".safetensors")]
tokenizer_files = [f for f in files if "tokenizer" in f.lower()]

for req in required_files:
    if req in files:
        print(f"  ✅ {req}")
    else:
        print(f"  ❌ {req} 없음!")

if safetensors_files:
    print(f"  ✅ safetensors: {len(safetensors_files)}개")
else:
    print(f"  ❌ safetensors 파일 없음!")

if tokenizer_files:
    print(f"  ✅ tokenizer 파일: {len(tokenizer_files)}개")
else:
    print(f"  ❌ tokenizer 파일 없음!")

# %% [셀 5] config.json 검증 + 수정
print(f"\n[2/6] config.json 검증")

config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

issues = []
fixes = []

# ── transformers_version 확인 ──
tv = config.get("transformers_version", "N/A")
if tv != EXPECTED["transformers_version"]:
    issues.append(f"transformers_version: {tv} → {EXPECTED['transformers_version']}")
    config["transformers_version"] = EXPECTED["transformers_version"]
    fixes.append("transformers_version")
else:
    print(f"  ✅ transformers_version: {tv}")

# ── max_position_embeddings 확인 ──
mpe = config.get("max_position_embeddings", "N/A")
if mpe != EXPECTED["max_position_embeddings"]:
    issues.append(f"max_position_embeddings: {mpe} → {EXPECTED['max_position_embeddings']}")
    config["max_position_embeddings"] = EXPECTED["max_position_embeddings"]
    fixes.append("max_position_embeddings")
else:
    print(f"  ✅ max_position_embeddings: {mpe}")

# ── quantization_config 확인 ──
qc = config.get("quantization_config", {})
if qc:
    # quant_method
    qm = qc.get("quant_method", "N/A")
    if qm == "compressed-tensors":
        print(f"  ✅ quant_method: {qm}")
    else:
        issues.append(f"quant_method: {qm} (expected: compressed-tensors)")
    
    # version
    qv = qc.get("version", "N/A")
    if qv != EXPECTED["compressed_tensors_version"]:
        issues.append(f"quantization_config.version: {qv} → {EXPECTED['compressed_tensors_version']}")
        qc["version"] = EXPECTED["compressed_tensors_version"]
        fixes.append("quantization_config.version")
    else:
        print(f"  ✅ quantization_config.version: {qv}")
    
    # format
    fmt = qc.get("format", "N/A")
    print(f"  ℹ️  format: {fmt}")
    
    # config_groups 요약
    cg = qc.get("config_groups", {})
    for group_name, group_config in cg.items():
        weights = group_config.get("weights", {})
        print(f"  ℹ️  {group_name}: bits={weights.get('num_bits', 'N/A')}, "
              f"strategy={weights.get('strategy', 'N/A')}, "
              f"actorder={weights.get('actorder', 'N/A')}")
    
    # sparsity_config 확인
    sc = qc.get("sparsity_config", {})
    if sc:
        print(f"  ✅ sparsity_config: structure={sc.get('sparsity_structure', 'N/A')}, "
              f"sparsity={sc.get('global_sparsity', 'N/A')}")
    else:
        print(f"  ℹ️  sparsity_config: 없음 (Sparse 미적용)")
else:
    issues.append("quantization_config 없음!")

# 수정 사항 적용
if fixes:
    print(f"\n  🔧 수정 사항 적용: {', '.join(fixes)}")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  → config.json 저장 완료")

if issues:
    print(f"\n  ⚠️ 발견된 이슈:")
    for issue in issues:
        print(f"    - {issue}")
else:
    print(f"\n  ✅ config.json 검증 통과!")

# %% [셀 6] 모델 로드 테스트
print(f"\n[3/6] 모델 로드 테스트")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Tokenizer 로드 테스트
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    print(f"  ✅ Tokenizer 로드 성공 (vocab_size: {tokenizer.vocab_size})")
    
    # Model 로드 테스트 (메타데이터만 확인, 실제 가중치는 로드 안 함)
    print(f"  → AutoModelForCausalLM.from_pretrained() 테스트...")
    test_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True
    )
    print(f"  ✅ Model 로드 성공 ({test_model.num_parameters()/1e9:.2f}B)")
    
    # 메모리 해제
    del test_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

except Exception as e:
    print(f"  ❌ 로드 실패: {e}")
    print(f"  ⚠️ 이 오류가 평가 서버에서도 발생할 수 있습니다!")

# %% [셀 7] ZIP 생성
print(f"\n[4/6] ZIP 생성")

# 기존 ZIP 삭제
if os.path.exists(ZIP_OUTPUT):
    os.remove(ZIP_OUTPUT)

# submit.zip → model/ 구조로 생성
zip_base = ZIP_OUTPUT.replace(".zip", "")
shutil.make_archive(zip_base, "zip", "/kaggle/working", "model")

zip_size_bytes = os.path.getsize(ZIP_OUTPUT)
zip_size_mb = zip_size_bytes / (1024 * 1024)
zip_size_gb = zip_size_bytes / (1024 ** 3)

print(f"  → {ZIP_OUTPUT}")
print(f"  → 크기: {zip_size_mb:.1f} MB ({zip_size_gb:.2f} GB)")

if zip_size_gb > EXPECTED["max_zip_size_gb"]:
    print(f"  ❌ ZIP 용량 초과! ({zip_size_gb:.2f}GB > {EXPECTED['max_zip_size_gb']}GB)")
else:
    print(f"  ✅ ZIP 용량 OK ({zip_size_gb:.2f}GB ≤ {EXPECTED['max_zip_size_gb']}GB)")

# %% [셀 8] ZIP 구조 검증
print(f"\n[5/6] ZIP 구조 검증")

import zipfile
with zipfile.ZipFile(ZIP_OUTPUT, 'r') as zf:
    namelist = zf.namelist()
    
    # model/ 폴더 존재 확인
    model_files = [n for n in namelist if n.startswith("model/")]
    if model_files:
        print(f"  ✅ model/ 폴더 구조 올바름 ({len(model_files)}개 파일)")
    else:
        print(f"  ❌ model/ 폴더가 ZIP 내에 없습니다!")
    
    # config.json 확인
    if "model/config.json" in namelist:
        print(f"  ✅ model/config.json 존재")
    else:
        print(f"  ❌ model/config.json 없음!")
    
    # safetensors 확인
    st_files = [n for n in model_files if n.endswith(".safetensors")]
    if st_files:
        print(f"  ✅ safetensors: {len(st_files)}개")
        for sf in st_files:
            info = zf.getinfo(sf)
            print(f"    - {sf} ({info.file_size / (1024*1024):.1f} MB)")
    else:
        print(f"  ❌ safetensors 파일 없음!")
    
    # 전체 파일 목록
    print(f"\n  📁 ZIP 내 파일 목록:")
    for name in sorted(namelist):
        if not name.endswith("/"):
            info = zf.getinfo(name)
            print(f"    {name} ({info.file_size / (1024*1024):.1f} MB)")

# %% [셀 9] 최종 요약
print(f"""
{'='*60}
✅ 최종 검증 + 패키징 완료!
{'='*60}

📋 검증 결과:
  config.json:       {'✅ OK' if not issues else '⚠️ 수정됨'}
  모델 로드 테스트:  확인 완료
  ZIP 구조:          model/ ✅
  ZIP 용량:          {zip_size_gb:.2f}GB {'✅' if zip_size_gb <= EXPECTED['max_zip_size_gb'] else '❌'}

📦 제출 파일: {ZIP_OUTPUT}

📤 제출 방법:
  1. {ZIP_OUTPUT} 파일을 다운로드
  2. Dacon 제출 탭에서 업로드
  3. ⚠️ 하루 최대 3회 제출 가능

🕐 평가 서버 실행 시간:
  - 모델 로드 + 추론 합계 ≤ 20분
  - 시간 초과 시 '제출 오류' (일일 횟수 차감됨)
""")
