#!/usr/bin/env python3
"""
VectorDB 빌드 스크립트
한국 법률 판례 데이터를 ChromaDB에 저장
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv(project_root / ".env")


def main(max_samples: int = 1000, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    VectorDB 빌드 메인 함수

    Args:
        max_samples: 처리할 최대 문서 수
        chunk_size: 청크 크기 (문자 수)
        chunk_overlap: 청크 간 오버랩
    """
    from src.data.loader import (
        load_korean_legal_precedents,
        preprocess_for_rag,
        chunk_documents,
    )
    from src.embedding.vectordb import VectorStore

    print("=" * 60)
    print("한국 법률 판례 VectorDB 빌드")
    print("=" * 60)

    # 1. 데이터셋 로드
    print(f"\n[1/4] 데이터셋 로드 중...")
    dataset = load_korean_legal_precedents()
    print(f"전체 데이터: {len(dataset):,} 건")

    # 2. RAG용 전처리
    print(f"\n[2/4] RAG용 전처리 중 (최대 {max_samples:,}건)...")
    documents = preprocess_for_rag(dataset, max_samples=max_samples)
    print(f"전처리된 문서: {len(documents):,} 건")

    # 샘플 출력
    if documents:
        print(f"\n샘플 문서 메타데이터:")
        sample = documents[0]
        for key, value in sample["metadata"].items():
            print(f"  - {key}: {value[:50] if isinstance(value, str) and len(value) > 50 else value}")

    # 3. 청킹
    print(f"\n[3/4] 문서 청킹 중 (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"생성된 청크: {len(chunks):,} 개")
    print(f"문서당 평균 청크: {len(chunks) / len(documents):.1f} 개")

    # 4. VectorDB 저장
    print(f"\n[4/4] ChromaDB에 저장 중...")
    vector_store = VectorStore(
        persist_directory="./data/embeddings/chroma",
        collection_name="korean_legal_docs",
        embedding_model="BAAI/bge-m3",
        device="cuda",
    )

    # 기존 문서 수 확인
    existing_count = vector_store.count()
    if existing_count > 0:
        print(f"기존 문서 수: {existing_count:,}")
        user_input = input("기존 데이터에 추가하시겠습니까? (y/n): ").strip().lower()
        if user_input != "y":
            print("새로운 컬렉션으로 재생성합니다...")
            vector_store.delete_collection()
            vector_store = VectorStore(
                persist_directory="./data/embeddings/chroma",
                collection_name="korean_legal_docs",
                embedding_model="BAAI/bge-m3",
                device="cuda",
            )

    # 청크에 고유 ID 부여
    for i, chunk in enumerate(chunks):
        chunk["metadata"]["id"] = f"chunk_{chunk['metadata'].get('id', 'unknown')}_{chunk['metadata'].get('chunk_index', i)}"

    # 저장
    added = vector_store.add_documents(chunks, batch_size=100)

    print(f"\n{'=' * 60}")
    print(f"VectorDB 빌드 완료!")
    print(f"- 저장된 청크: {added:,} 개")
    print(f"- 총 문서 수: {vector_store.count():,} 개")
    print(f"- 저장 경로: ./data/embeddings/chroma")
    print(f"{'=' * 60}")

    # 검색 테스트
    print(f"\n검색 테스트:")
    test_queries = [
        "상속포기 절차는 어떻게 되나요?",
        "임대차 보증금 반환 문제",
        "근로계약 해지 관련 판례",
    ]

    for query in test_queries:
        print(f"\n질문: {query}")
        results = vector_store.search(query, top_k=2)
        for j, r in enumerate(results, 1):
            print(f"  [{j}] 점수: {r['score']:.4f}")
            print(f"      사건: {r['metadata'].get('case_name', 'N/A')[:40]}")
            print(f"      내용: {r['content'][:80]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국 법률 판례 VectorDB 빌드")
    parser.add_argument("--max-samples", type=int, default=1000, help="처리할 최대 문서 수")
    parser.add_argument("--chunk-size", type=int, default=1000, help="청크 크기")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="청크 오버랩")

    args = parser.parse_args()
    main(
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
