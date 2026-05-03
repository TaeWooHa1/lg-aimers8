"""
LG Aimers - 로컬 모델 평가 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적: 양자화된 모델들을 기본 모델(EXAONE-4.0-1.2B) 대비 비교 평가하여
      가장 좋은 모델을 선별한 후 대회 서버에 제출

평가 산식:
  Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)
  - PerfNorm  = 모델 벤치마크 정확도 / 기본 모델 벤치마크 정확도
  - SpeedNorm = 1 - (모델 토큰당 시간) / (기본 모델 토큰당 시간)

평가 태스크:
  - gsm8k          : 수학 추론 (lm-eval, loglikelihood)
  - mmlu           : 지식 QA (lm-eval, loglikelihood)
  - translation    : 한→영/영→한 번역 (sacrebleu BLEU + langdetect)
  - summarization  : 요약 (rouge_score ROUGE-L)

사전 설치:
  pip install lm-eval torch transformers accelerate safetensors sacrebleu rouge_score langdetect

사용법:
  # 기본 모델 baseline 측정 (최초 1회)
  python evaluate_local.py --base-model ./base_model --mode baseline

  # 양자화 모델 평가 (기본 모델 대비 비교)
  python evaluate_local.py --base-model ./base_model --target-model ./model_DB/optimized_submit/model

  # 여러 모델 한번에 비교
  python evaluate_local.py --base-model ./base_model --target-model ./modelA ./modelB ./modelC

  # 벤치마크 태스크 지정 (기본: gsm8k,mmlu)
  python evaluate_local.py --base-model ./base_model --target-model ./model --tasks gsm8k,mmlu

  # 번역/요약 포함 전체 평가
  python evaluate_local.py --base-model ./base_model --target-model ./model --tasks gsm8k,mmlu,translation,summarization

  # 샘플 수 제한 (빠른 테스트)
  python evaluate_local.py --base-model ./base_model --target-model ./model --tasks translation --limit 5

  # 속도 측정 생략 (정확도만 비교)
  python evaluate_local.py --base-model ./base_model --target-model ./model --skip-speed

  # 이전 baseline 결과 재사용 (시간 절약)
  python evaluate_local.py --target-model ./model --baseline-json ./baseline_result.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict

# vLLM CUDA fork 충돌 방지: 반드시 torch import 전에 설정
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch


# =========================================================
# 데이터 클래스
# =========================================================

@dataclass
class ModelResult:
    """단일 모델 평가 결과"""
    model_path: str
    # 벤치마크 정확도 (PerfNorm 산출용)
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    avg_accuracy: float = 0.0
    # 속도 (SpeedNorm 산출용)
    time_per_token_ms: float = 0.0
    tokens_per_sec: float = 0.0
    total_tokens: int = 0
    total_time_sec: float = 0.0
    # 모델 정보
    num_parameters: int = 0
    model_size_mb: float = 0.0


@dataclass
class ComparisonEntry:
    """모델 간 비교 결과 (한 줄)"""
    model_path: str
    avg_accuracy: float
    perf_norm: float
    time_per_token_ms: float
    speed_norm: float
    score: float
    benchmark_details: Dict[str, float] = field(default_factory=dict)


# =========================================================
# 1. 벤치마크 평가 (PerfNorm용) - lm-evaluation-harness 사용
# =========================================================

def run_benchmarks(model_path: str, tasks: List[str], 
                   batch_size: str = "auto", num_fewshot: int = None,
                   limit: int = None) -> Dict[str, float]:
    """
    lm-evaluation-harness를 사용하여 벤치마크 정확도 측정
    
    Returns:
        Dict[task_name, accuracy]  (0.0 ~ 1.0)
    """
    import lm_eval

    print(f"\n  📊 벤치마크 평가 시작: {', '.join(tasks)}")
    print(f"     모델: {model_path}")
    
    model_args = f"pretrained={model_path},trust_remote_code=True"
    
    # GPU VRAM에 따라 dtype, batch_size 조정
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 16:
            model_args += ",dtype=float16"
        if vram_gb < 16:
            # 16GB 미만: 배치 크기 고정 (auto 사용 시 OOM 발생 가능)
            batch_size = "4"
            print(f"     ⚠️ VRAM {vram_gb:.1f}GB 감지 → batch_size={batch_size}, dtype=float16")
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    
    # 태스크별 정확도 추출
    scores = {}
    for task_name in tasks:
        task_result = results["results"].get(task_name, {})
        
        # lm-eval은 태스크에 따라 다른 metric명을 사용
        # gsm8k: exact_match,flexible-extract / mmlu: acc,none
        acc = None
        for metric_key in ["acc_norm,none", "acc,none", "exact_match,none",
                           "exact_match,flexible-extract", "exact_match,strict-match",
                           "acc_norm", "acc", "exact_match"]:
            if metric_key in task_result:
                acc = task_result[metric_key]
                break
        
        if acc is not None:
            scores[task_name] = acc
            print(f"     ✅ {task_name}: {acc:.4f} ({acc*100:.2f}%)")
        else:
            # 하위 태스크가 있는 경우 (예: mmlu는 여러 subject)
            # 그룹 평균 찾기
            for key, val in task_result.items():
                if "acc" in key and isinstance(val, (int, float)):
                    scores[task_name] = val
                    print(f"     ✅ {task_name}: {val:.4f} ({val*100:.2f}%)")
                    break
            else:
                print(f"     ⚠️ {task_name}: 결과를 찾을 수 없음 (건너뜁니다)")
                print(f"        사용 가능한 키: {list(task_result.keys())}")
    
    return scores


# =========================================================
# 2. 속도 평가 (SpeedNorm용) - HF generate() 상대 비교
# =========================================================

SPEED_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What are the benefits of renewable energy?",
    "Write a short paragraph about artificial intelligence.",
    "Describe the process of photosynthesis.",
    "What is the capital of France and why is it famous?",
    "인공지능의 미래에 대해 설명해주세요.",
    "한국의 전통 음식 중 하나를 소개해주세요.",
    "프로그래밍을 배우는 좋은 방법은 무엇인가요?",
    "Solve: If a train travels 60km/h for 2 hours, how far?",
    "What is the difference between a stack and a queue?",
    "딥러닝과 머신러닝의 차이점을 설명해주세요.",
    "Write a Python function to reverse a string.",
]


def measure_speed(model_path: str = None, max_new_tokens: int = 128,
                  _llm=None) -> Dict:
    """
    vLLM을 사용한 토큰 생성 속도 측정
    (대회 서버도 vLLM 사용 → 로컬과 유사한 결과)
    """
    from vllm import SamplingParams
    
    print(f"\n  ⏱️  속도 측정 시작 ({len(SPEED_PROMPTS)}개 프롬프트, max_tokens={max_new_tokens})")
    
    own_model = False
    if _llm is None:
        _llm = _load_gen_model(model_path)
        own_model = True
    
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    
    # 워밍업
    try:
        _llm.chat(messages=[[{"role": "user", "content": "Hello"}]],
                  sampling_params=SamplingParams(temperature=0, max_tokens=5))
    except Exception:
        _llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=5))
    
    # 속도 측정 — 배치 생성
    messages_list = [[{"role": "user", "content": p}] for p in SPEED_PROMPTS]
    
    start = time.perf_counter()
    try:
        outputs = _llm.chat(messages=messages_list, sampling_params=sampling_params)
    except Exception:
        outputs = _llm.generate(SPEED_PROMPTS, sampling_params)
    elapsed = time.perf_counter() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    if own_model:
        _cleanup_vllm(_llm)
    
    result = {
        "total_time_sec": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
        "time_per_token_ms": (elapsed / total_tokens * 1000) if total_tokens > 0 else 0,
    }
    
    print(f"     Tokens/sec: {result['tokens_per_sec']:.2f}")
    print(f"     Time/token: {result['time_per_token_ms']:.2f} ms")
    
    return result


# =========================================================
# 2-1. 번역 평가 (sacrebleu + langdetect)
# =========================================================

TRANSLATION_DATA = {
    "ko2en": [
        ("인공지능은 인간의 학습 능력을 모방하는 컴퓨터 시스템입니다.",
         "Artificial intelligence is a computer system that mimics human learning ability."),
        ("한국은 사계절이 뚜렷한 나라입니다.",
         "Korea is a country with four distinct seasons."),
        ("딥러닝은 여러 층의 신경망을 사용하는 기계학습 방법입니다.",
         "Deep learning is a machine learning method that uses multiple layers of neural networks."),
        ("서울은 대한민국의 수도이며 가장 큰 도시입니다.",
         "Seoul is the capital and largest city of South Korea."),
        ("양자컴퓨터는 기존 컴퓨터보다 특정 문제를 훨씬 빠르게 풀 수 있습니다.",
         "Quantum computers can solve certain problems much faster than conventional computers."),
        ("재생 에너지는 환경 보호에 중요한 역할을 합니다.",
         "Renewable energy plays an important role in environmental protection."),
        ("자연어 처리는 컴퓨터가 인간의 언어를 이해하는 기술입니다.",
         "Natural language processing is a technology that enables computers to understand human language."),
        ("한국의 전통 음식인 김치는 세계적으로 유명합니다.",
         "Kimchi, a traditional Korean food, is famous worldwide."),
        ("클라우드 컴퓨팅은 인터넷을 통해 컴퓨팅 자원을 제공하는 서비스입니다.",
         "Cloud computing is a service that provides computing resources through the internet."),
        ("데이터 과학은 대량의 데이터에서 유용한 정보를 추출하는 분야입니다.",
         "Data science is a field that extracts useful information from large amounts of data."),
        ("로봇 공학은 로봇의 설계와 제작을 연구하는 학문입니다.",
         "Robotics is a discipline that studies the design and construction of robots."),
        ("사물인터넷은 일상의 기기들이 인터넷으로 연결되는 기술입니다.",
         "The Internet of Things is a technology where everyday devices are connected through the internet."),
        ("블록체인은 분산 원장 기술로 데이터의 무결성을 보장합니다.",
         "Blockchain is a distributed ledger technology that ensures data integrity."),
        ("5G 통신은 이전 세대보다 훨씬 빠른 데이터 전송 속도를 제공합니다.",
         "5G communications provide much faster data transmission speeds than previous generations."),
        ("사이버 보안은 디지털 시스템을 보호하는 것을 목표로 합니다.",
         "Cybersecurity aims to protect digital systems."),
        ("가상현실은 컴퓨터로 만든 가상 세계를 체험하는 기술입니다.",
         "Virtual reality is a technology for experiencing computer-generated virtual worlds."),
        ("빅데이터 분석은 기업의 의사결정에 도움을 줍니다.",
         "Big data analysis helps in business decision-making."),
        ("자율주행 자동차는 운전자 없이 스스로 운행하는 차량입니다.",
         "Self-driving cars are vehicles that operate on their own without a driver."),
        ("유전자 편집 기술은 질병 치료에 새로운 가능성을 열었습니다.",
         "Gene editing technology has opened new possibilities for disease treatment."),
        ("우주 탐사는 인류의 미래를 위한 중요한 과학 활동입니다.",
         "Space exploration is an important scientific activity for the future of humanity."),
    ],
    "en2ko": [
        ("Machine learning algorithms improve through experience without being explicitly programmed.",
         "머신러닝 알고리즘은 명시적으로 프로그래밍되지 않아도 경험을 통해 개선됩니다."),
        ("The global economy is becoming increasingly interconnected through digital technology.",
         "세계 경제는 디지털 기술을 통해 점점 더 상호 연결되고 있습니다."),
        ("Climate change is one of the most pressing challenges facing humanity today.",
         "기후변화는 오늘날 인류가 직면한 가장 시급한 과제 중 하나입니다."),
        ("Artificial neural networks are inspired by the structure of the human brain.",
         "인공 신경망은 인간 뇌의 구조에서 영감을 받았습니다."),
        ("Sustainable development balances economic growth with environmental protection.",
         "지속 가능한 개발은 경제 성장과 환경 보호의 균형을 맞춥니다."),
        ("Transfer learning allows models to apply knowledge from one task to another.",
         "전이 학습은 모델이 한 작업에서 얻은 지식을 다른 작업에 적용할 수 있게 합니다."),
        ("The speed of light is approximately 300,000 kilometers per second.",
         "빛의 속도는 초당 약 30만 킬로미터입니다."),
        ("Photosynthesis is the process by which plants convert sunlight into energy.",
         "광합성은 식물이 햇빛을 에너지로 변환하는 과정입니다."),
        ("Democracy is a system of government where citizens exercise power by voting.",
         "민주주의는 시민들이 투표를 통해 권력을 행사하는 정치 제도입니다."),
        ("The human genome contains approximately three billion base pairs of DNA.",
         "인간 게놈에는 약 30억 개의 DNA 염기쌍이 포함되어 있습니다."),
        ("Electric vehicles are becoming more popular as battery technology improves.",
         "배터리 기술이 발전하면서 전기 자동차가 점점 더 인기를 얻고 있습니다."),
        ("Water is essential for all known forms of life on Earth.",
         "물은 지구상의 모든 알려진 생명체에 필수적입니다."),
        ("The Renaissance was a period of great cultural and intellectual achievement in Europe.",
         "르네상스는 유럽에서 위대한 문화적, 지적 성취가 이루어진 시기였습니다."),
        ("Antibiotics revolutionized medicine by providing effective treatments for bacterial infections.",
         "항생제는 세균 감염에 대한 효과적인 치료법을 제공하여 의학에 혁명을 일으켰습니다."),
        ("The internet has transformed how people communicate and access information.",
         "인터넷은 사람들이 소통하고 정보에 접근하는 방식을 변화시켰습니다."),
        ("Gravity is the force that attracts objects toward one another.",
         "중력은 물체를 서로 끌어당기는 힘입니다."),
        ("Vaccination has been one of the most successful public health interventions in history.",
         "예방접종은 역사상 가장 성공적인 공중보건 개입 중 하나입니다."),
        ("The theory of relativity fundamentally changed our understanding of space and time.",
         "상대성 이론은 시간과 공간에 대한 우리의 이해를 근본적으로 변화시켰습니다."),
        ("Biodiversity is crucial for maintaining the balance of ecosystems.",
         "생물다양성은 생태계의 균형을 유지하는 데 매우 중요합니다."),
        ("Nuclear fusion could potentially provide a nearly unlimited source of clean energy.",
         "핵융합은 잠재적으로 거의 무한한 청정 에너지원을 제공할 수 있습니다."),
    ],
}


def _load_gen_model(model_path: str):
    """vLLM으로 모델 로드 (번역/요약/속도 평가 통합)"""
    from vllm import LLM
    print(f"  🚀 vLLM 모델 로딩: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )
    return llm


def _cleanup_vllm(llm):
    """vLLM 모델 메모리 해제"""
    import gc
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_text(llm, prompt: str, max_new_tokens: int = 256) -> str:
    """vLLM으로 단일 프롬프트 텍스트 생성"""
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    messages = [{"role": "user", "content": prompt}]
    try:
        outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
    except Exception:
        outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def run_translation(model_path: str = None, limit: int = None,
                    _llm=None) -> Dict[str, float]:
    """
    번역 평가: 한→영, 영→한 sacrebleu BLEU + langdetect (vLLM 사용)
    Returns: {"translation_ko2en": bleu, "translation_en2ko": bleu}  (0~1 정규화)
    """
    import sacrebleu
    from langdetect import detect

    print(f"\n  🌐 번역 평가 시작")

    own_model = False
    if _llm is None:
        _llm = _load_gen_model(model_path)
        own_model = True

    scores = {}
    for direction, pairs in TRANSLATION_DATA.items():
        if limit and limit < len(pairs):
            import random
            random.seed(42)
            pairs = random.sample(pairs, limit)

        src_lang = "한국어" if direction == "ko2en" else "English"
        tgt_lang = "English" if direction == "ko2en" else "한국어"
        tgt_lang_code = "en" if direction == "ko2en" else "ko"

        hypotheses = []
        references = []
        lang_correct = 0

        print(f"     [{direction}] {len(pairs)}개 샘플 번역 중...")
        for src, ref in pairs:
            prompt = f"Translate the following {src_lang} text to {tgt_lang}. Output ONLY the translation, nothing else.\n\n{src}"
            hyp = _generate_text(_llm, prompt, max_new_tokens=256)
            hypotheses.append(hyp)
            references.append(ref)

            try:
                detected = detect(hyp)
                if detected == tgt_lang_code:
                    lang_correct += 1
            except Exception:
                pass

        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        bleu_normalized = bleu.score / 100.0
        lang_acc = lang_correct / len(pairs) if pairs else 0

        scores[f"translation_{direction}"] = bleu_normalized
        print(f"     ✅ {direction}: BLEU={bleu.score:.2f} ({bleu_normalized:.4f}), "
              f"언어정확도={lang_acc:.0%}")

    if own_model:
        _cleanup_vllm(_llm)

    return scores


# =========================================================
# 2-2. 요약 평가 (rouge_score)
# =========================================================

SUMMARIZATION_DATA = [
    {
        "text": "인공지능(AI)은 인간의 지능을 모방하여 학습, 추론, 자기 수정 등을 수행하는 컴퓨터 시스템을 말합니다. "
                "최근에는 딥러닝 기술의 발전으로 이미지 인식, 자연어 처리, 게임 등 다양한 분야에서 "
                "인간을 뛰어넘는 성능을 보여주고 있습니다. 특히 대규모 언어 모델(LLM)의 등장으로 "
                "텍스트 생성, 번역, 요약 등의 작업에서 획기적인 발전이 이루어졌습니다.",
        "summary": "인공지능은 인간의 지능을 모방하는 컴퓨터 시스템으로, 딥러닝과 대규모 언어 모델의 발전으로 다양한 분야에서 뛰어난 성능을 보여주고 있습니다."
    },
    {
        "text": "Climate change refers to long-term shifts in global temperatures and weather patterns. "
                "While some shifts are natural, human activities have been the main driver since the 1800s, "
                "primarily due to burning fossil fuels like coal, oil, and gas.",
        "summary": "Climate change involves long-term temperature and weather shifts, mainly driven by human fossil fuel use since the 1800s."
    },
    {
        "text": "양자 컴퓨팅은 양자역학의 원리를 이용하여 정보를 처리하는 새로운 컴퓨팅 패러다임입니다. "
                "기존 컴퓨터가 비트(0 또는 1)를 사용하는 것과 달리, 양자 컴퓨터는 큐비트를 사용하여 "
                "0과 1의 상태를 동시에 가질 수 있는 중첩 상태를 활용합니다.",
        "summary": "양자 컴퓨팅은 큐비트의 중첩 상태를 활용하여 기존 컴퓨터보다 특정 문제를 훨씬 빠르게 처리하는 새로운 컴퓨팅 기술입니다."
    },
    {
        "text": "The human brain contains approximately 86 billion neurons, each connected to thousands of others "
                "through synapses. These neural connections form complex networks that enable thinking, learning, "
                "memory, and consciousness.",
        "summary": "The human brain has 86 billion interconnected neurons forming complex networks for cognition."
    },
    {
        "text": "한국의 반도체 산업은 세계 시장에서 중요한 위치를 차지하고 있습니다. 삼성전자와 SK하이닉스는 "
                "메모리 반도체 분야에서 세계 1위와 2위를 차지하고 있으며, 최근에는 AI 반도체 시장에서도 "
                "경쟁력을 키우고 있습니다.",
        "summary": "한국의 반도체 산업은 삼성전자와 SK하이닉스가 메모리 반도체에서 세계 선두를 달리고 있으며, AI 반도체로 영역을 확장하고 있습니다."
    },
    {
        "text": "Renewable energy sources such as solar, wind, and hydroelectric power are becoming increasingly "
                "cost-competitive with fossil fuels. Solar panel costs have dropped by over 90% in the past decade.",
        "summary": "Renewable energy, especially solar power, has become cost-competitive with fossil fuels due to dramatic cost reductions."
    },
    {
        "text": "자율주행 기술은 레이더, 라이다, 카메라 등 다양한 센서를 활용하여 주변 환경을 인식하고, "
                "인공지능 알고리즘으로 주행 판단을 내리는 기술입니다. 현재 레벨 2~3 수준의 자율주행이 "
                "상용화되어 있습니다.",
        "summary": "자율주행 기술은 다양한 센서와 AI를 활용하여 주행하는 기술로, 현재 레벨 2~3이 상용화되었습니다."
    },
    {
        "text": "CRISPR-Cas9 is a revolutionary gene editing technology that allows scientists to modify DNA sequences "
                "with unprecedented precision. It works by using a guide RNA to direct the Cas9 protein to a specific "
                "location in the genome.",
        "summary": "CRISPR-Cas9 is a precise gene editing technology using guide RNA and Cas9 protein to modify DNA."
    },
    {
        "text": "대한민국의 교육 시스템은 높은 학업 성취도로 국제적으로 인정받고 있습니다. "
                "OECD PISA 평가에서 지속적으로 상위권을 유지하고 있으며, 대학 진학률도 세계 최고 수준입니다.",
        "summary": "대한민국 교육은 PISA 상위권과 높은 대학 진학률로 인정받고 있습니다."
    },
    {
        "text": "Blockchain technology creates a decentralized and immutable ledger of transactions shared across "
                "a network of computers. Each block contains transaction data, a timestamp, and a cryptographic hash "
                "of the previous block, creating a chain.",
        "summary": "Blockchain is a decentralized, immutable ledger technology using linked cryptographic blocks."
    },
]


def run_summarization(model_path: str = None, limit: int = None,
                      _llm=None) -> Dict[str, float]:
    """
    요약 평가: ROUGE-L F1 점수 (vLLM 사용)
    Returns: {"summarization": rouge_l_f1}  (0.0 ~ 1.0)
    """
    from rouge_score import rouge_scorer

    print(f"\n  📝 요약 평가 시작")

    own_model = False
    if _llm is None:
        _llm = _load_gen_model(model_path)
        own_model = True

    data = SUMMARIZATION_DATA[:]
    if limit and limit < len(data):
        import random
        random.seed(42)
        data = random.sample(data, limit)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total_rouge_l = 0.0

    print(f"     {len(data)}개 샘플 요약 중...")
    for i, item in enumerate(data):
        prompt = f"Summarize the following text in one or two sentences. Output ONLY the summary.\n\n{item['text']}"
        hyp = _generate_text(_llm, prompt, max_new_tokens=200)
        score = scorer.score(item["summary"], hyp)
        rouge_l = score["rougeL"].fmeasure
        total_rouge_l += rouge_l
        print(f"       [{i+1}/{len(data)}] ROUGE-L: {rouge_l:.4f}")

    avg_rouge_l = total_rouge_l / len(data) if data else 0.0
    print(f"     ✅ 요약 평균 ROUGE-L: {avg_rouge_l:.4f}")

    if own_model:
        _cleanup_vllm(_llm)

    return {"summarization": avg_rouge_l}


# 커스텀 태스크 목록 (lm-eval이 아닌 자체 평가)
CUSTOM_TASKS = {"translation", "summarization"}


# =========================================================
# 3. 전체 평가 + 점수 계산
# =========================================================

def evaluate_model(model_path: str, tasks: List[str],
                   skip_speed: bool = False, max_new_tokens: int = 128,
                   limit: int = None) -> ModelResult:
    """단일 모델 전체 평가"""
    
    print(f"\n{'━' * 60}")
    print(f"  📌 평가 모델: {model_path}")
    print(f"{'━' * 60}")
    
    # 모델 크기 확인
    model_dir = Path(model_path)
    model_size_mb = 0
    if model_dir.is_dir():
        for f in model_dir.glob("*.safetensors"):
            model_size_mb += f.stat().st_size / (1024 * 1024)
        for f in model_dir.glob("*.bin"):
            model_size_mb += f.stat().st_size / (1024 * 1024)
        print(f"  모델 가중치 크기: {model_size_mb:.1f} MB")
    
    scores = {}
    
    # lm-eval 벤치마크 (gsm8k, mmlu 등)
    lm_eval_tasks = [t for t in tasks if t not in CUSTOM_TASKS]
    if lm_eval_tasks:
        lm_scores = run_benchmarks(model_path, lm_eval_tasks, limit=limit)
        scores.update(lm_scores)
    
    # vLLM이 필요한 작업: 커스텀 태스크 + 속도 측정
    custom_tasks = [t for t in tasks if t in CUSTOM_TASKS]
    need_vllm = bool(custom_tasks) or not skip_speed
    
    speed = {"time_per_token_ms": 0, "tokens_per_sec": 0, "total_tokens": 0, "total_time_sec": 0}
    
    if need_vllm:
        # lm-eval 후 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # vLLM 모델을 한 번만 로드하여 모든 작업에 재사용
        llm = _load_gen_model(model_path)
        
        if "translation" in custom_tasks:
            trans_scores = run_translation(limit=limit, _llm=llm)
            scores.update(trans_scores)
        
        if "summarization" in custom_tasks:
            summ_scores = run_summarization(limit=limit, _llm=llm)
            scores.update(summ_scores)
        
        # 속도 측정 (같은 vLLM 인스턴스 재사용)
        if not skip_speed:
            speed = measure_speed(_llm=llm, max_new_tokens=max_new_tokens)
        
        # vLLM 정리
        _cleanup_vllm(llm)
    
    avg_acc = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  📊 평균 점수: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    
    return ModelResult(
        model_path=model_path,
        benchmark_scores=scores,
        avg_accuracy=avg_acc,
        time_per_token_ms=speed["time_per_token_ms"],
        tokens_per_sec=speed["tokens_per_sec"],
        total_tokens=speed["total_tokens"],
        total_time_sec=speed["total_time_sec"],
        model_size_mb=model_size_mb,
    )


def calculate_score(base: ModelResult, target: ModelResult, skip_speed: bool = False) -> ComparisonEntry:
    """대회 산식에 따른 점수 계산"""
    
    # PerfNorm = target 정확도 / base 정확도
    if base.avg_accuracy > 0:
        perf_norm = target.avg_accuracy / base.avg_accuracy
    else:
        perf_norm = 1.0
    
    # SpeedNorm = 1 - (target time/token) / (base time/token)
    if not skip_speed and base.time_per_token_ms > 0 and target.time_per_token_ms > 0:
        speed_norm = 1 - (target.time_per_token_ms / base.time_per_token_ms)
    else:
        speed_norm = 0.0  # 속도 미측정 시 0으로 처리
    
    # Score
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
    
    return ComparisonEntry(
        model_path=target.model_path,
        avg_accuracy=target.avg_accuracy,
        perf_norm=perf_norm,
        time_per_token_ms=target.time_per_token_ms,
        speed_norm=speed_norm,
        score=score,
        benchmark_details=target.benchmark_scores,
    )


# =========================================================
# 4. 결과 출력
# =========================================================

def print_comparison(base: ModelResult, entries: List[ComparisonEntry], skip_speed: bool):
    """각 모델을 기본 모델 대비 비율로 개별 출력"""
    
    print("\n\n" + "=" * 80)
    print("  🏆 LG Aimers 로컬 평가 결과 (lm-evaluation-harness 기반)")
    print("=" * 80)
    
    # 기본 모델 (기준)
    base_name = Path(base.model_path).name or "base"
    print(f"\n📋 기준 모델: {base_name}")
    print(f"{'─' * 80}")
    print(f"  경로:       {base.model_path}")
    print(f"  평균 정확도: {base.avg_accuracy:.4f} ({base.avg_accuracy*100:.2f}%)")
    for task, score in base.benchmark_scores.items():
        print(f"    - {task}: {score:.4f}")
    if not skip_speed:
        print(f"  Time/token: {base.time_per_token_ms:.2f} ms")
    print(f"  → 이 모델이 PerfNorm=1.0, SpeedNorm=0.0, Score=0.5 의 기준입니다.")
    
    # ── 각 모델을 개별적으로 기본 모델 대비 비교 ──
    entries_sorted = sorted(entries, key=lambda x: x.score, reverse=True)
    tasks = list(base.benchmark_scores.keys())
    
    for idx, e in enumerate(entries_sorted, 1):
        model_name = Path(e.model_path).name or e.model_path
        
        print(f"\n\n{'━' * 80}")
        print(f"  📌 [{idx}] {model_name} / 기준 모델 비교")
        print(f"{'━' * 80}")
        
        # 태스크별 비율
        print(f"\n  🎯 PerfNorm (벤치마크 정확도 비율)")
        print(f"  {'─' * 60}")
        print(f"  {'태스크':<15} {'기준 모델':<12} {'이 모델':<12} {'비율 (모델/기준)':<18}")
        print(f"  {'─' * 60}")
        
        for task in tasks:
            base_score = base.benchmark_scores.get(task, 0)
            target_score = e.benchmark_details.get(task, 0)
            ratio = target_score / base_score if base_score > 0 else 0
            arrow = "✅" if ratio >= 0.95 else ("⚠️" if ratio >= 0.85 else "❌")
            print(f"  {task:<15} {base_score:.4f}       {target_score:.4f}       {ratio:.4f} ({ratio*100:.1f}%)  {arrow}")
        
        # 평균
        print(f"  {'─' * 60}")
        print(f"  {'평균':<15} {base.avg_accuracy:.4f}       {e.avg_accuracy:.4f}       {e.perf_norm:.4f} ({e.perf_norm*100:.1f}%)")
        print(f"\n  → PerfNorm = {e.avg_accuracy:.4f} / {base.avg_accuracy:.4f} = {e.perf_norm:.4f}")
        
        # 속도 비율
        if not skip_speed:
            print(f"\n  ⏱️  SpeedNorm (토큰당 추론 시간 비율)")
            print(f"  {'─' * 60}")
            print(f"  기준 모델 Time/token:  {base.time_per_token_ms:.2f} ms")
            print(f"  이 모델 Time/token:    {e.time_per_token_ms:.2f} ms")
            time_ratio = e.time_per_token_ms / base.time_per_token_ms if base.time_per_token_ms > 0 else 1
            print(f"  시간 비율:             {time_ratio:.4f} ({time_ratio*100:.1f}%)")
            speed_arrow = "✅ 빨라짐" if e.speed_norm > 0 else ("⚡ 동일" if e.speed_norm == 0 else "❌ 느려짐")
            print(f"\n  → SpeedNorm = 1 - {e.time_per_token_ms:.2f} / {base.time_per_token_ms:.2f} = {e.speed_norm:+.4f}  {speed_arrow}")
        
        # 최종 Score
        print(f"\n  🏆 최종 Score")
        print(f"  {'─' * 60}")
        print(f"  Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)")
        print(f"        = max(0.5 × {e.perf_norm:.4f} + 0.5 × {e.speed_norm:+.4f}, 0)")
        print(f"        = {e.score:.4f}")
        
        if e.score > 0.5:
            print(f"\n  ✅ 수료 기준 (> 0.5) 통과!  (기준 대비 +{e.score - 0.5:.4f})")
        else:
            print(f"\n  ❌ 수료 기준 (> 0.5) 미달  (부족분: {0.5 - e.score:.4f})")
    
    # ── 최종 요약 순위 ──
    print(f"\n\n{'=' * 80}")
    print(f"  📊 최종 순위 요약 (모든 모델 / 기준 모델 비교)")
    print(f"{'=' * 80}")
    print(f"  {'순위':<4} {'모델':<28} {'PerfNorm':<10} {'SpeedNorm':<11} {'Score':<8} {'판정'}")
    print(f"{'─' * 80}")
    
    for i, e in enumerate(entries_sorted, 1):
        name = Path(e.model_path).name or e.model_path
        if len(name) > 26:
            name = name[:23] + "..."
        verdict = "✅ 통과" if e.score > 0.5 else "❌ 미달"
        star = " ⭐ BEST" if i == 1 else ""
        print(f"  {i:<4} {name:<28} {e.perf_norm:.4f}    {e.speed_norm:+.4f}    {e.score:.4f}  {verdict}{star}")
    
    print(f"{'─' * 80}")
    print(f"  ref  {'기준(EXAONE-4.0-1.2B)':<28} 1.0000    +0.0000    0.5000  기준선")
    print(f"{'=' * 80}")
    
    if skip_speed:
        print(f"  ⚠️  SpeedNorm 미측정: 실제 Score는 속도 개선분만큼 더 높을 수 있음")
    print(f"  ⚠️  PerfNorm은 공개 벤치마크 기준이며 대회 비공개 벤치셋과 차이 가능")
    
    return entries_sorted


def save_result(base: ModelResult, entries: List[ComparisonEntry], output_path: str):
    """결과 JSON 저장"""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": asdict(base),
        "models": [asdict(e) for e in entries],
        "ranking": [
            {"rank": i+1, "model": Path(e.model_path).name, "score": e.score}
            for i, e in enumerate(sorted(entries, key=lambda x: x.score, reverse=True))
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 결과 저장: {output_path}")


# =========================================================
# 메인
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="LG Aimers 로컬 모델 평가 (lm-evaluation-harness 기반)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 1. baseline 측정 + 저장
  python evaluate_local.py --base-model ./base_model --mode baseline

  # 2. 양자화 모델 1개 평가
  python evaluate_local.py --base-model ./base_model --target-model ./model_DB/optimized_submit/model

  # 3. 여러 모델 한번에 비교
  python evaluate_local.py --base-model ./base_model --target-model ./modelA ./modelB ./modelC

  # 4. 저장된 baseline 재사용 (시간 절약)
  python evaluate_local.py --baseline-json ./baseline_result.json --target-model ./modelA

  # 5. 정확도만 비교 (속도 생략)
  python evaluate_local.py --base-model ./base_model --target-model ./model --skip-speed
        """
    )
    
    parser.add_argument("--base-model", type=str, default=None,
                        help="기본 모델 경로 (EXAONE-4.0-1.2B)")
    parser.add_argument("--target-model", type=str, nargs="+", default=None,
                        help="평가할 양자화 모델 경로 (여러 개 가능)")
    parser.add_argument("--mode", choices=["baseline", "compare"], default="compare",
                        help="baseline: 기본 모델만 측정 / compare: 비교 평가")
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu",
                        help="평가 태스크 (쉼표 구분, 기본: gsm8k,mmlu) "
                             "사용 가능: gsm8k, mmlu, translation, summarization")
    parser.add_argument("--skip-speed", action="store_true",
                        help="속도 측정 생략 (정확도만 비교)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="속도 측정 시 최대 생성 토큰 수")
    parser.add_argument("--baseline-json", type=str, default=None,
                        help="이전에 저장한 baseline 결과 JSON 경로 (재측정 생략)")
    parser.add_argument("--limit", type=int, default=None,
                        help="태스크당 평가할 샘플 수 (기본: 전체, 예: 300)")
    parser.add_argument("--output", type=str, default=None,
                        help="결과 저장 경로 (기본: 자동 생성)")
    
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    
    print("\n" + "=" * 80)
    print("  LG Aimers 로컬 모델 평가 (lm-evaluation-harness 기반)")
    print("=" * 80)
    print(f"  벤치마크: {', '.join(tasks)}")
    print(f"  속도 측정: {'생략' if args.skip_speed else '실행'}")
    
    # ─── Baseline 처리 ─────────────────────────────
    base_result = None
    
    if args.baseline_json:
        # 저장된 baseline 로드
        print(f"\n  📂 Baseline 로드: {args.baseline_json}")
        with open(args.baseline_json, "r") as f:
            data = json.load(f)
        base_data = data if "benchmark_scores" in data else data.get("baseline", data)
        base_result = ModelResult(**{k: v for k, v in base_data.items() if k in ModelResult.__dataclass_fields__})
        print(f"     평균 정확도: {base_result.avg_accuracy:.4f}")
    
    elif args.base_model:
        # 기본 모델 평가
        base_result = evaluate_model(args.base_model, tasks, args.skip_speed, args.max_tokens, args.limit)
        
        # Baseline 결과 저장
        baseline_path = "baseline_result.json"
        save_result(base_result, [], baseline_path)
        print(f"  💾 Baseline 저장됨 → 다음부터 --baseline-json {baseline_path} 로 재사용 가능")
    
    if args.mode == "baseline":
        if base_result:
            print(f"\n✅ Baseline 측정 완료!")
            print(f"   평균 정확도: {base_result.avg_accuracy:.4f}")
            for t, s in base_result.benchmark_scores.items():
                print(f"   - {t}: {s:.4f}")
        else:
            print("❌ --base-model 을 지정해주세요")
        return
    
    # ─── Target 모델 평가 ─────────────────────────
    if not args.target_model:
        print("❌ --target-model 을 지정해주세요")
        return
    
    if base_result is None:
        print("❌ --base-model 또는 --baseline-json 을 지정해주세요")
        return
    
    entries = []
    for model_path in args.target_model:
        target_result = evaluate_model(model_path, tasks, args.skip_speed, args.max_tokens, args.limit)
        entry = calculate_score(base_result, target_result, args.skip_speed)
        entries.append(entry)
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ─── 결과 출력 ─────────────────────────────────
    sorted_entries = print_comparison(base_result, entries, args.skip_speed)
    
    # ─── 결과 저장 ─────────────────────────────────
    if args.output is None:
        output_path = f"eval_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_path = args.output
    
    save_result(base_result, sorted_entries, output_path)


if __name__ == "__main__":
    main()
