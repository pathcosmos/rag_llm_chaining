"""
VectorDB 모듈
ChromaDB를 이용한 임베딩 저장 및 검색
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """ChromaDB 기반 벡터 저장소"""

    def __init__(
        self,
        persist_directory: str = "./data/embeddings/chroma",
        collection_name: str = "korean_legal_docs",
        embedding_model: str = "BAAI/bge-m3",
        device: str = "cuda",
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
    ):
        """
        VectorStore 초기화

        Args:
            persist_directory: ChromaDB 저장 경로 (로컬 모드)
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델 ID
            device: 디바이스 (cuda/cpu)
            chroma_host: ChromaDB 서버 호스트 (원격 모드)
            chroma_port: ChromaDB 서버 포트 (원격 모드)
        """
        self.collection_name = collection_name
        self.device = device
        self.is_remote = chroma_host is not None

        # 임베딩 모델 로드
        print(f"임베딩 모델 로딩: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"임베딩 차원: {self.embedding_dim}")

        # ChromaDB 클라이언트 초기화
        if chroma_host:
            # 원격 ChromaDB 서버 연결
            print(f"원격 ChromaDB 연결: {chroma_host}:{chroma_port}")
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
            )
        else:
            # 로컬 PersistentClient
            self.persist_directory = Path(persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )

        # 컬렉션 가져오기 또는 생성
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"컬렉션 '{collection_name}' 준비 완료 (문서 수: {self.collection.count()})")

    def add_documents(
        self,
        documents: list[dict],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        문서 추가

        Args:
            documents: 문서 리스트 [{"content": str, "metadata": dict}, ...]
            batch_size: 배치 크기
            show_progress: 진행 상황 표시

        Returns:
            int: 추가된 문서 수
        """
        from tqdm import tqdm

        total = len(documents)
        added = 0

        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="임베딩 및 저장")

        for i in iterator:
            batch = documents[i:i + batch_size]

            # 텍스트 추출
            texts = [doc["content"] for doc in batch]

            # 임베딩 생성
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            # ID 생성
            ids = [
                doc["metadata"].get("id", f"doc_{i + j}")
                for j, doc in enumerate(batch)
            ]

            # 메타데이터 추출
            metadatas = [doc["metadata"] for doc in batch]

            # ChromaDB에 추가
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            added += len(batch)

        print(f"총 {added}개 문서 추가 완료")
        return added

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> list[dict]:
        """
        유사 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            filter_dict: 메타데이터 필터

        Returns:
            list[dict]: 검색 결과
        """
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )

        # 결과 정리
        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine distance to similarity
            }
            documents.append(doc)

        return documents

    def count(self) -> int:
        """저장된 문서 수 반환"""
        return self.collection.count()

    def delete_collection(self):
        """컬렉션 삭제"""
        self.client.delete_collection(self.collection_name)
        print(f"컬렉션 '{self.collection_name}' 삭제됨")


if __name__ == "__main__":
    # 테스트
    print("VectorStore 테스트")
    print("=" * 50)

    # 초기화
    store = VectorStore(
        persist_directory="./data/embeddings/chroma_test",
        collection_name="test_collection",
    )

    # 테스트 문서
    test_docs = [
        {
            "content": "상속포기란 상속인이 상속을 포기하는 것으로, 상속개시가 있음을 안 날로부터 3개월 내에 가정법원에 신고해야 합니다.",
            "metadata": {"id": "doc_1", "type": "상속"}
        },
        {
            "content": "임대차보호법은 주택 임차인의 주거 안정을 보장하기 위한 법률로, 대항력과 우선변제권을 규정하고 있습니다.",
            "metadata": {"id": "doc_2", "type": "임대차"}
        },
        {
            "content": "근로기준법에 따르면 연차휴가는 1년간 80% 이상 출근한 근로자에게 15일의 유급휴가가 부여됩니다.",
            "metadata": {"id": "doc_3", "type": "근로"}
        },
    ]

    # 문서 추가
    store.add_documents(test_docs)

    # 검색 테스트
    print("\n검색 테스트:")
    query = "상속을 포기하려면 어떻게 해야 하나요?"
    results = store.search(query, top_k=2)

    for r in results:
        print(f"\n[점수: {r['score']:.4f}]")
        print(f"내용: {r['content'][:100]}...")

    print("\n✅ VectorStore 테스트 완료!")
