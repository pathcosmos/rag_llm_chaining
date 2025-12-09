"""
데이터 로더 모듈
한국 법률 판례 데이터셋 로드 및 전처리
"""

from typing import Generator, Optional

from datasets import Dataset, load_dataset


def load_korean_legal_precedents(
    split: str = "train",
    streaming: bool = False,
) -> Dataset:
    """
    한국 법률 판례 데이터셋 로드

    Args:
        split: 데이터셋 분할 (train)
        streaming: 스트리밍 모드 여부

    Returns:
        Dataset: HuggingFace Dataset 객체
    """
    ds = load_dataset(
        "joonhok-exo-ai/korean_law_open_data_precedents",
        split=split,
        streaming=streaming,
    )
    return ds


def preprocess_for_rag(
    dataset: Dataset,
    text_columns: list[str] = None,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """
    RAG용 데이터 전처리

    Args:
        dataset: 원본 데이터셋
        text_columns: 텍스트로 사용할 컬럼들
        max_samples: 최대 샘플 수 (None이면 전체)

    Returns:
        list[dict]: 전처리된 문서 리스트
    """
    if text_columns is None:
        text_columns = ["판시사항", "판결요지", "전문"]

    documents = []

    # 샘플 수 제한
    data = dataset.select(range(max_samples)) if max_samples else dataset

    for i, item in enumerate(data):
        # 텍스트 결합
        text_parts = []
        for col in text_columns:
            if col in item and item[col]:
                text_parts.append(f"[{col}]\n{item[col]}")

        if not text_parts:
            continue

        content = "\n\n".join(text_parts)

        # 메타데이터 추출
        metadata = {
            "id": item.get("판례정보일련번호", str(i)),
            "case_name": item.get("사건명", ""),
            "case_number": item.get("사건번호", ""),
            "judgment_date": item.get("선고일자", ""),
            "court": item.get("법원명", ""),
            "case_type": item.get("사건종류명", ""),
            "judgment_type": item.get("판결유형", ""),
        }

        documents.append({
            "content": content,
            "metadata": metadata,
        })

    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    문서를 청크로 분할

    Args:
        documents: 문서 리스트
        chunk_size: 청크 크기 (문자 수)
        chunk_overlap: 청크 간 오버랩

    Returns:
        list[dict]: 청크 리스트
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        for j, split in enumerate(splits):
            chunk_metadata = doc["metadata"].copy()
            chunk_metadata["chunk_index"] = j
            chunks.append({
                "content": split,
                "metadata": chunk_metadata,
            })

    return chunks


if __name__ == "__main__":
    # 테스트
    print("데이터셋 로드...")
    ds = load_korean_legal_precedents()
    print(f"전체 데이터: {len(ds):,} 건")

    print("\nRAG용 전처리 (100건 샘플)...")
    docs = preprocess_for_rag(ds, max_samples=100)
    print(f"전처리된 문서: {len(docs)} 건")

    print("\n청킹...")
    chunks = chunk_documents(docs)
    print(f"청크 수: {len(chunks)} 개")

    print("\n샘플 청크:")
    print(f"내용: {chunks[0]['content'][:200]}...")
    print(f"메타데이터: {chunks[0]['metadata']}")
