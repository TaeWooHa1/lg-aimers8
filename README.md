# LG Aimers 8기 — EXAONE-4.0-1.2B 모델 경량화

> LG Aimers 8기 해커톤: EXAONE-4.0-1.2B 모델을 GPTQ/AutoRound/FP8 양자화를 통해 경량화하여  
> 정확도(PerfNorm)와 추론 속도(SpeedNorm)의 균형을 최적화하는 프로젝트

---

## 📁 디렉터리 구조

```
lg_aimers/
├── README.md                          # 프로젝트 소개
├── .gitignore                         # 모델/대용량 파일 제외
│
├── docs/                              # 대회 개요 + 분석 문서
│   ├── 대회개요.md
│   ├── 대회규칙.md
│   ├── 코드제출가이드.md
│   ├── 00_evaluate.md
│   └── 평가결과_분석.md
│
├── guides/                            # 양자화 가이드 + 학습 자료
│   ├── 파라미터가이드.md
│   ├── autoround_파라미터_가이드.md
│   ├── fp8_개념.md
│   ├── 1등가이드.md
│   ├── kd_sparse_quant_walkthrough.md
│   ├── smqt_spst.md
│   ├── 변수설명.md
│   └── autoround.md
│
├── quantize/                          # 양자화 코드 (기법별 분류)
│   ├── gptq/                          # GPTQ 양자화
│   │   ├── w4a16_baseline.py
│   │   ├── w4a16_colab.py
│   │   ├── w4a16_kaggle.py
│   │   ├── w4a16_all_params.py
│   │   ├── w4a16_colab_optimized.py
│   │   ├── w8a8_colab.py
│   │   ├── w8a8_kaggle.py
│   │   └── w8a16.py
│   ├── autoround/                     # AutoRound 양자화
│   │   ├── autoround_v1.py
│   │   └── autoround_v2.py
│   ├── fp8/                           # FP8 양자화
│   │   └── fp8.py
│   ├── advanced/                      # 고급 기법 (KD, Sparse)
│   │   ├── kd_train.py
│   │   └── sparse_quant.py
│   └── final/                         # 최종 제출 코드
│       ├── best.py
│       └── final_submit.py
│
├── evaluate/                          # 평가 관련
│   ├── evaluate_local.py
│   ├── complexity_check.py
│   ├── copy_configs.py
│   └── docker/                        # Docker 평가 환경
│       ├── Dockerfile
│       ├── docker-compose.yml
│       ├── requirements_server.txt
│       ├── run_eval.bat
│       └── run_eval.sh
│
├── weekly/                            # 주차별 진행 기록
│   ├── week1/
│   │   ├── analysis.md
│   │   ├── architecture.md
│   │   ├── colab_baseline.py
│   │   ├── colab_awq.py
│   │   ├── colab_gptq_v2.py
│   │   ├── colab_quantize.py
│   │   └── success.py
│   ├── week2/
│   │   ├── awq개념.md
│   │   └── 진행사항.md
│   └── week3/
│       ├── study.md
│       └── 진행사항.md
│
├── configs/                           # 양자화 모델 config.json 모음
│   ├── actorder_dynamic_config.json
│   ├── actorder_static1_config.json
│   ├── actorder_static2_config.json
│   ├── actorder_weight_config.json
│   ├── blocksize_64_config.json
│   ├── calibraion_512_config.json
│   ├── kvcache8bit_config.json
│   └── model_config.json
│
├── results/                           # 평가 결과
│   ├── baseline_result.json
│   ├── eval_comparison.json
│   └── 모델결과.md
│
└── scripts/                           # 유틸리티 스크립트
    └── run_docker.bat
```

---

## 🔧 사용 기법

| 기법 | 설명 | 위치 |
|------|------|------|
| **W4A16 GPTQ** | 4-bit 가중치 + 16-bit 활성화 | `quantize/gptq/` |
| **W8A8 GPTQ** | 8-bit 가중치 + 8-bit 활성화 (INT8 Tensor Core) | `quantize/gptq/` |
| **AutoRound** | 반올림 기반 양자화 | `quantize/autoround/` |
| **FP8** | 8-bit 부동소수점 양자화 | `quantize/fp8/` |
| **KD + Sparse** | Knowledge Distillation + 희소화 | `quantize/advanced/` |

---

## ⚙️ 주요 최적화 파라미터

```python
actorder = "dynamic"          # 공짜 정확도 ↑↑↑
num_calibration_samples = 256
max_seq_length = 512
sequential_targets = ["Exaone4DecoderLayer"]
block_size = 64               # 정확도 ↑
max_position_embeddings = 32768  # KV Cache 50% 절감
group_size = 64               # config_groups로 직접 설정
```

---

## 🏆 평가 환경

- **GPU**: NVIDIA L4
- **Framework**: vLLM 0.14.1
- **Library**: compressed-tensors 0.13.0
- **Score**: `0.5 × PerfNorm + 0.5 × SpeedNorm`
