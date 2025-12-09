#!/usr/bin/env python3
"""
VectorDB 병렬 빌드 스크립트
CPU 멀티프로세싱 + GPU 임베딩 최적화
"""

import os
import sys
import time
import argparse
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def chunk_single_document(args: tuple) -> list[dict]:
    """단일 문서 청킹 (멀티프로세싱용)"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    doc, chunk_size, chunk_overlap = args

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    chunks = []
    splits = text_splitter.split_text(doc["content"])
    for j, split in enumerate(splits):
        chunk_metadata = doc["metadata"].copy()
        chunk_metadata["chunk_index"] = j
        chunks.append({
            "content": split,
            "metadata": chunk_metadata,
        })

    return chunks


def parallel_chunk_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    num_workers: int = None,
) -> list[dict]:
    """
    멀티프로세싱을 이용한 병렬 청킹

    Args:
        documents: 문서 리스트
        chunk_size: 청크 크기
        chunk_overlap: 오버랩
        num_workers: 워커 수 (기본: CPU 코어 수)

    Returns:
        list[dict]: 청크 리스트
    """
    from tqdm import tqdm

    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # 최대 16개로 제한

    print(f"  병렬 청킹 시작 (워커: {num_workers}개)")

    # 작업 준비
    tasks = [(doc, chunk_size, chunk_overlap) for doc in documents]

    all_chunks = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(chunk_single_document, task): i for i, task in enumerate(tasks)}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="  청킹"):
            chunks = future.result()
            all_chunks.extend(chunks)

    return all_chunks


def main(
    max_samples: int = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 256,
    num_workers: int = None,
    checkpoint_interval: int = 10000,
    reset_db: bool = False,
):
    """
    VectorDB 병렬 빌드 메인 함수

    Args:
        max_samples: 처리할 최대 문서 수 (None이면 전체)
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        batch_size: 임베딩 배치 크기
        num_workers: CPU 워커 수
        checkpoint_interval: 체크포인트 저장 간격
        reset_db: 기존 DB 초기화 여부
    """
    from src.data.loader import load_korean_legal_precedents, preprocess_for_rag
    from src.embedding.vectordb import VectorStore
    from tqdm import tqdm

    start_time = time.time()

    print("=" * 70)
    print("한국 법률 판례 VectorDB 병렬 빌드")
    print("=" * 70)
    print(f"  - CPU 워커: {num_workers or min(cpu_count(), 16)}개")
    print(f"  - GPU 배치 크기: {batch_size}")
    print(f"  - 청크 설정: size={chunk_size}, overlap={chunk_overlap}")
    print("=" * 70)

    # 체크포인트 경로
    checkpoint_dir = project_root / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunks_checkpoint = checkpoint_dir / "chunks.pkl"

    # 1. 청크 로드 또는 생성
    if chunks_checkpoint.exists() and not reset_db:
        print(f"\n[1/3] 체크포인트에서 청크 로드 중...")
        with open(chunks_checkpoint, "rb") as f:
            chunks = pickle.load(f)
        print(f"  로드된 청크: {len(chunks):,}개")
    else:
        # 데이터셋 로드
        print(f"\n[1/3] 데이터셋 로드 및 청킹...")
        dataset = load_korean_legal_precedents()
        total_docs = len(dataset)
        print(f"  전체 데이터: {total_docs:,}건")

        if max_samples:
            print(f"  처리 대상: {max_samples:,}건")

        # RAG용 전처리
        print(f"\n  전처리 중...")
        documents = preprocess_for_rag(dataset, max_samples=max_samples)
        print(f"  전처리된 문서: {len(documents):,}건")

        # 병렬 청킹
        print(f"\n  병렬 청킹 중...")
        chunks = parallel_chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_workers=num_workers,
        )
        print(f"  생성된 청크: {len(chunks):,}개")
        print(f"  문서당 평균 청크: {len(chunks) / len(documents):.1f}개")

        # 청크에 고유 ID 부여
        for i, chunk in enumerate(chunks):
            doc_id = chunk["metadata"].get("id", "unknown")
            chunk_idx = chunk["metadata"].get("chunk_index", i)
            chunk["metadata"]["id"] = f"chunk_{doc_id}_{chunk_idx}"

        # 체크포인트 저장
        print(f"\n  체크포인트 저장: {chunks_checkpoint}")
        with open(chunks_checkpoint, "wb") as f:
            pickle.dump(chunks, f)

    preprocess_time = time.time() - start_time
    print(f"\n  전처리 시간: {preprocess_time:.1f}초")

    # 2. VectorStore 초기화
    print(f"\n[2/3] VectorStore 초기화...")
    vector_store = VectorStore(
        persist_directory="./data/embeddings/chroma",
        collection_name="korean_legal_docs",
        embedding_model="BAAI/bge-m3",
        device="cuda",
    )

    existing_count = vector_store.count()
    if existing_count > 0:
        print(f"  기존 문서 수: {existing_count:,}")
        if reset_db:
            print("  기존 데이터 삭제 중...")
            vector_store.delete_collection()
            vector_store = VectorStore(
                persist_directory="./data/embeddings/chroma",
                collection_name="korean_legal_docs",
                embedding_model="BAAI/bge-m3",
                device="cuda",
            )

    # 3. 임베딩 및 저장
    print(f"\n[3/3] 임베딩 생성 및 저장 (배치 크기: {batch_size})...")
    embed_start = time.time()

    # 이미 저장된 ID 확인
    if existing_count > 0 and not reset_db:
        # 기존 청크 건너뛰기
        existing_ids = set()
        # ChromaDB에서 기존 ID 조회
        try:
            result = vector_store.collection.get(limit=existing_count, include=[])
            existing_ids = set(result["ids"])
        except Exception:
            pass

        chunks_to_add = [c for c in chunks if c["metadata"]["id"] not in existing_ids]
        print(f"  새로 추가할 청크: {len(chunks_to_add):,}개 (기존: {existing_count:,}개)")
    else:
        chunks_to_add = chunks

    if chunks_to_add:
        # 배치 처리로 임베딩 및 저장
        total_chunks = len(chunks_to_add)
        added = 0

        for i in tqdm(range(0, total_chunks, batch_size), desc="  임베딩"):
            batch = chunks_to_add[i:i + batch_size]

            # 텍스트 추출
            texts = [c["content"] for c in batch]

            # 임베딩 생성 (GPU)
            embeddings = vector_store.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,  # 내부 배치 크기도 동일하게
            ).tolist()

            # ID 및 메타데이터 (None 값을 빈 문자열로 변환)
            ids = [c["metadata"]["id"] for c in batch]
            metadatas = []
            for c in batch:
                cleaned_meta = {}
                for k, v in c["metadata"].items():
                    if v is None:
                        cleaned_meta[k] = ""
                    else:
                        cleaned_meta[k] = v
                metadatas.append(cleaned_meta)

            # ChromaDB에 추가
            vector_store.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            added += len(batch)

        print(f"\n  추가된 청크: {added:,}개")

    embed_time = time.time() - embed_start
    total_time = time.time() - start_time

    # 최종 결과
    print(f"\n{'=' * 70}")
    print(f"VectorDB 빌드 완료!")
    print(f"{'=' * 70}")
    print(f"  - 총 문서 수: {vector_store.count():,}개")
    print(f"  - 저장 경로: ./data/embeddings/chroma")
    print(f"  - 전처리 시간: {preprocess_time:.1f}초")
    print(f"  - 임베딩 시간: {embed_time:.1f}초")
    print(f"  - 총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    print(f"{'=' * 70}")

    # 검색 테스트
    print(f"\n검색 테스트:")
    test_queries = [
        "상속포기 절차는 어떻게 되나요?",
        "임대차 보증금 반환 문제",
        "근로계약 해지 관련 판례",
    ]

    for query in test_queries:
        print(f"\n  질문: {query}")
        results = vector_store.search(query, top_k=2)
        for j, r in enumerate(results, 1):
            print(f"    [{j}] 점수: {r['score']:.4f}")
            case_name = r['metadata'].get('case_name', 'N/A')[:35]
            print(f"        사건: {case_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="한국 법률 판례 VectorDB 병렬 빌드",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="처리할 최대 문서 수 (기본: 전체)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="청크 크기 (문자 수)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="청크 오버랩",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="GPU 임베딩 배치 크기",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="CPU 워커 수 (기본: 자동)",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=10000,
        help="체크포인트 저장 간격",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="기존 DB 초기화 후 새로 빌드",
    )

    args = parser.parse_args()

    main(
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        reset_db=args.reset,
    )
