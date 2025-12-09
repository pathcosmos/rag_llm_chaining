"""
RAG 파이프라인 모듈
검색-생성 통합 파이프라인
"""

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.embedding.vectordb import VectorStore


class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) 파이프라인"""

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_model: str = "BAAI/bge-m3",
        vector_db_path: str = "./data/embeddings/chroma",
        collection_name: str = "korean_legal_docs",
        device: str = "cuda",
        load_in_4bit: bool = False,
    ):
        """
        RAG 파이프라인 초기화

        Args:
            llm_model: LLM 모델 ID
            embedding_model: 임베딩 모델 ID
            vector_db_path: VectorDB 저장 경로
            collection_name: 컬렉션 이름
            device: 디바이스
            load_in_4bit: 4비트 양자화 사용 여부
        """
        self.device = device

        # VectorStore 초기화
        print("VectorStore 초기화...")
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            device=device,
        )

        # LLM 초기화
        print(f"LLM 로딩: {llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            trust_remote_code=True,
        )

        # 양자화 설정
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        print("RAG 파이프라인 초기화 완료!")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        관련 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수

        Returns:
            list[dict]: 검색된 문서 리스트
        """
        return self.vector_store.search(query, top_k=top_k)

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        컨텍스트 기반 응답 생성

        Args:
            query: 사용자 질문
            context: 검색된 컨텍스트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도

        Returns:
            str: 생성된 응답
        """
        # 프롬프트 구성
        system_prompt = """당신은 한국 법률 전문가입니다.
주어진 참고 자료를 바탕으로 질문에 정확하고 이해하기 쉽게 답변해주세요.
참고 자료에 없는 내용은 답변하지 마세요."""

        user_prompt = f"""참고 자료:
{context}

질문: {query}

답변:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 토큰화
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 디코딩
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 512,
        return_sources: bool = False,
    ) -> dict:
        """
        RAG 질의 실행

        Args:
            question: 사용자 질문
            top_k: 검색할 문서 수
            max_new_tokens: 최대 생성 토큰 수
            return_sources: 소스 문서 반환 여부

        Returns:
            dict: {"answer": str, "sources": list (optional)}
        """
        # 1. 검색
        retrieved_docs = self.retrieve(question, top_k=top_k)

        # 2. 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc["metadata"]
            case_info = f"[{metadata.get('case_name', '판례')} - {metadata.get('court', '')}]"
            context_parts.append(f"{i}. {case_info}\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # 3. 생성
        answer = self.generate(question, context, max_new_tokens=max_new_tokens)

        result = {"answer": answer}
        if return_sources:
            result["sources"] = retrieved_docs

        return result

    def add_documents(self, documents: list[dict], batch_size: int = 100):
        """VectorDB에 문서 추가"""
        return self.vector_store.add_documents(documents, batch_size=batch_size)


if __name__ == "__main__":
    # 간단한 테스트 (VectorDB만)
    print("RAG Pipeline 모듈 로드 테스트")

    # VectorStore만 테스트 (LLM 로딩 없이)
    from src.embedding.vectordb import VectorStore

    store = VectorStore(
        persist_directory="./data/embeddings/chroma_test",
        collection_name="test",
    )

    print(f"VectorStore 문서 수: {store.count()}")
    print("✅ 모듈 로드 성공!")
