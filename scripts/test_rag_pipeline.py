#!/usr/bin/env python3
"""
RAG 파이프라인 전체 테스트
VectorDB + LLM 통합 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv(project_root / ".env")


def test_rag_pipeline():
    """RAG 파이프라인 전체 테스트"""
    import torch
    from src.rag.pipeline import RAGPipeline

    print("=" * 70)
    print("RAG 파이프라인 전체 테스트")
    print("=" * 70)

    # GPU 상태
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"현재 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # RAG 파이프라인 초기화
    print("\n" + "-" * 70)
    print("RAG 파이프라인 초기화 중...")
    print("-" * 70)

    pipeline = RAGPipeline(
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        embedding_model="BAAI/bge-m3",
        vector_db_path="./data/embeddings/chroma",
        collection_name="korean_legal_docs",
        device="cuda",
    )

    print(f"\nVRAM 사용량 (초기화 후): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 테스트 질문들
    test_questions = [
        "귀속재산의 임대차 계약에서 임차인의 권리는 어떻게 보호되나요?",
        "법인이 해산된 경우 청산 절차는 어떻게 되나요?",
        "토지 소유권 이전 등기 관련 판례가 있나요?",
    ]

    print("\n" + "=" * 70)
    print("RAG 질의 테스트")
    print("=" * 70)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"[질문 {i}] {question}")
        print("─" * 70)

        # RAG 질의 실행
        result = pipeline.query(
            question=question,
            top_k=3,
            max_new_tokens=512,
            return_sources=True,
        )

        # 답변 출력
        print(f"\n📝 답변:")
        print(result["answer"])

        # 소스 출력
        print(f"\n📚 참고 자료 ({len(result['sources'])}건):")
        for j, source in enumerate(result["sources"], 1):
            meta = source["metadata"]
            print(f"  [{j}] {meta.get('case_name', 'N/A')} ({meta.get('court', 'N/A')})")
            print(f"      점수: {source['score']:.4f}")

    # RAG vs Non-RAG 비교
    print("\n" + "=" * 70)
    print("RAG vs Non-RAG 비교")
    print("=" * 70)

    comparison_question = "귀속재산 임대차 계약에서 임차권 포기의 효력은 언제 발생하나요?"

    print(f"\n질문: {comparison_question}")

    # RAG 응답
    print("\n[RAG 응답]")
    rag_result = pipeline.query(
        question=comparison_question,
        top_k=3,
        max_new_tokens=300,
    )
    print(rag_result["answer"])

    # Non-RAG 응답 (컨텍스트 없이)
    print("\n[Non-RAG 응답 (컨텍스트 없음)]")
    non_rag_answer = pipeline.generate(
        query=comparison_question,
        context="(참고 자료 없음)",
        max_new_tokens=300,
    )
    print(non_rag_answer)

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)

    # 메모리 정리
    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_rag_pipeline()
