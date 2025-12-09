#!/usr/bin/env python3
"""
모델 다운로드 및 기본 추론 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# 환경변수 로드
load_dotenv(project_root / ".env")

# HuggingFace 토큰
HF_TOKEN = os.getenv("HF_TOKEN")


def check_gpu():
    """GPU 상태 확인"""
    print("=" * 50)
    print("GPU 상태 확인")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"현재 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("CUDA를 사용할 수 없습니다.")
    print()


def test_qwen_model():
    """Qwen2.5-7B-Instruct 모델 테스트"""
    print("=" * 50)
    print("Qwen2.5-7B-Instruct 모델 테스트")
    print("=" * 50)

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print(f"모델 로딩: {model_id}")
    print("(첫 실행시 다운로드에 시간이 걸립니다...)")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    # 모델 로드 (bfloat16으로 메모리 절약)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"모델 로드 완료!")
    print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print()

    # 테스트 프롬프트 (법률 관련 질문)
    test_prompts = [
        "상속포기의 절차와 기한에 대해 간단히 설명해주세요.",
        "임대차보호법에서 보증금 보호 범위는 어떻게 되나요?",
    ]

    for prompt in test_prompts:
        print(f"질문: {prompt}")
        print("-" * 40)

        # 메시지 포맷
        messages = [
            {"role": "system", "content": "당신은 한국 법률 전문가입니다. 정확하고 이해하기 쉽게 답변해주세요."},
            {"role": "user", "content": prompt}
        ]

        # 토큰화
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 디코딩
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"답변: {response}")
        print()

    # 메모리 정리
    del model
    torch.cuda.empty_cache()

    print("테스트 완료!")


def test_embedding_model():
    """BGE-M3 임베딩 모델 테스트"""
    print("=" * 50)
    print("BGE-M3 임베딩 모델 테스트")
    print("=" * 50)

    from sentence_transformers import SentenceTransformer

    model_id = "BAAI/bge-m3"

    print(f"모델 로딩: {model_id}")

    model = SentenceTransformer(model_id, device="cuda")

    print(f"모델 로드 완료!")
    print(f"임베딩 차원: {model.get_sentence_embedding_dimension()}")
    print()

    # 테스트 문장
    sentences = [
        "상속포기란 상속인이 상속을 거부하는 것입니다.",
        "임대차보호법은 임차인을 보호하는 법률입니다.",
        "오늘 날씨가 좋습니다.",
    ]

    # 임베딩 생성
    embeddings = model.encode(sentences, normalize_embeddings=True)

    print("임베딩 생성 완료!")
    print(f"임베딩 shape: {embeddings.shape}")
    print()

    # 유사도 계산
    from numpy import dot

    print("문장 유사도:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = dot(embeddings[i], embeddings[j])
            print(f"  [{i}] vs [{j}]: {similarity:.4f}")
            print(f"    - \"{sentences[i][:30]}...\"")
            print(f"    - \"{sentences[j][:30]}...\"")

    print()
    print("테스트 완료!")


if __name__ == "__main__":
    check_gpu()

    print("\n" + "=" * 50)
    print("어떤 테스트를 실행하시겠습니까?")
    print("1. Qwen2.5-7B-Instruct (LLM)")
    print("2. BGE-M3 (Embedding)")
    print("3. 둘 다")
    print("=" * 50)

    choice = input("선택 (1/2/3): ").strip()

    if choice == "1":
        test_qwen_model()
    elif choice == "2":
        test_embedding_model()
    elif choice == "3":
        test_qwen_model()
        print("\n" * 2)
        test_embedding_model()
    else:
        print("잘못된 선택입니다.")
