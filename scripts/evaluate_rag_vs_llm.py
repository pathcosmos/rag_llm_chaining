"""
LLM vs RAG+LLM 비교 평가 스크립트
RAG 추가의 효과를 정량적/정성적으로 측정
"""

import json
import time
import os
import sys
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embedding.vectordb import VectorStore


def clean_korean_text(text: str) -> str:
    """한글, 숫자, 기본 문장부호만 남기고 필터링

    - 한글 (가-힣)
    - 숫자 (0-9)
    - 기본 문장부호 (., ,, !, ?, :, ;, ', ", (, ), -, /)
    - 공백
    - 한글이 충분하지 않으면 원본 반환
    """
    # 한글, 숫자, 공백, 기본 문장부호만 유지
    cleaned = re.sub(r'[^\uAC00-\uD7A3\u3131-\u3163\u1100-\u11FF0-9\s.,!?:;\'"()\-/]', '', text)
    # 연속 공백 제거
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 한글 글자 수 확인 (최소 10글자 이상이어야 유효)
    korean_chars = re.findall(r'[\uAC00-\uD7A3]', cleaned)
    if len(korean_chars) < 10:
        # 한글이 너무 적으면 원본 반환 (영어 응답인 경우)
        return text

    return cleaned


class LLMOnlyGenerator:
    """LLM만 사용하는 생성기 (RAG 없음)"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
    ):
        print(f"LLM 로딩: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("LLM 로딩 완료!")

    def generate(
        self,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """LLM 단독 응답 생성 (RAG 없음)"""
        system_prompt = """당신은 한국 법률 전문가입니다.
질문에 정확하고 이해하기 쉽게 답변해주세요.
반드시 한국어로만 답변하세요. 영어, 한자, 일본어 등 다른 언어를 사용하지 마세요."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        # 한글 필터링 적용
        return clean_korean_text(response)


class RAGGenerator:
    """RAG+LLM 생성기"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_model: str = "BAAI/bge-m3",
        chroma_host: str = "211.231.121.68",
        chroma_port: int = 8081,
        collection_name: str = "korean_legal_docs",
        device: str = "cuda",
        use_4bit: bool = True,
    ):
        print(f"RAG 시스템 초기화...")

        # VectorStore 초기화 (원격 ChromaDB)
        self.vector_store = VectorStore(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name=collection_name,
            embedding_model=embedding_model,
            device=device,
        )

        # LLM 로딩 (4bit 양자화로 속도 향상)
        print(f"LLM 로딩: {model_name} (4bit={use_4bit})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        print("RAG 시스템 초기화 완료!")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """관련 문서 검색"""
        return self.vector_store.search(query, top_k=top_k)

    def generate(
        self,
        query: str,
        top_k: int = 3,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> dict:
        """RAG 기반 응답 생성"""
        # 1. 검색
        retrieved_docs = self.retrieve(query, top_k=top_k)

        # 2. 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            case_info = f"[{metadata.get('case_name', '판례')} - {metadata.get('court', '')}]"
            context_parts.append(f"{i}. {case_info}\n{doc['content'][:500]}")

        context = "\n\n".join(context_parts)

        # 3. 프롬프트 구성
        system_prompt = """당신은 한국 법률 전문가입니다.
주어진 참고 자료를 바탕으로 질문에 정확하고 이해하기 쉽게 답변해주세요.
참고 자료에 없는 내용은 "참고 자료에서 관련 정보를 찾을 수 없습니다"라고 답변하세요.
반드시 한국어로만 답변하세요. 영어, 한자, 일본어 등 다른 언어를 사용하지 마세요."""

        user_prompt = f"""참고 자료:
{context}

질문: {query}

답변:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # 한글 필터링 적용
        return {
            "answer": clean_korean_text(response),
            "sources": retrieved_docs,
            "context_length": len(context),
        }


def calculate_keyword_score(response: str, keywords: list[str]) -> float:
    """키워드 매칭 점수 계산"""
    if not keywords:
        return 0.0

    response_lower = response.lower()
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    return matched / len(keywords)


def evaluate_response_quality(
    question: str,
    llm_response: str,
    rag_response: str,
    expected_keywords: list[str],
) -> dict:
    """응답 품질 평가"""
    return {
        "llm_length": len(llm_response),
        "rag_length": len(rag_response),
        "llm_keyword_score": calculate_keyword_score(llm_response, expected_keywords),
        "rag_keyword_score": calculate_keyword_score(rag_response, expected_keywords),
    }


def run_evaluation(
    test_data_path: str,
    output_dir: str,
    sample_size: int = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """평가 실행"""
    # 테스트 데이터 로드
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_cases = test_data["test_cases"]
    if sample_size:
        test_cases = test_cases[:sample_size]

    print(f"\n{'='*60}")
    print(f"LLM vs RAG+LLM 비교 평가")
    print(f"{'='*60}")
    print(f"테스트 케이스: {len(test_cases)}개")
    print(f"모델: {model_name}")
    print(f"{'='*60}\n")

    # 생성기 초기화 (모델 공유를 위해 순차 초기화)
    print("1. RAG 생성기 초기화...")
    rag_gen = RAGGenerator(model_name=model_name)

    # LLM 생성기는 RAG 생성기의 모델 재사용
    print("\n2. LLM 생성기 설정 (모델 공유)...")

    results = []

    print(f"\n3. 평가 시작 ({len(test_cases)}개 질문)...\n")

    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        category = test_case["category"]
        subcategory = test_case["subcategory"]
        expected_keywords = test_case.get("expected_keywords", [])

        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(test_cases)}] {category}/{subcategory}", flush=True)
        print(f"  질문: {question}", flush=True)
        print(f"  예상 키워드: {expected_keywords}", flush=True)
        print(f"-"*60, flush=True)

        # LLM 단독 응답
        print(f"  📝 LLM 추론 시작...", flush=True)
        start_time = time.time()
        llm_system_prompt = """당신은 한국 법률 전문가입니다.
질문에 정확하고 이해하기 쉽게 답변해주세요.
반드시 한국어로만 답변하세요. 영어, 한자, 일본어 등 다른 언어를 사용하지 마세요."""

        llm_messages = [
            {"role": "system", "content": llm_system_prompt},
            {"role": "user", "content": question},
        ]
        llm_text = rag_gen.tokenizer.apply_chat_template(
            llm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        llm_inputs = rag_gen.tokenizer(llm_text, return_tensors="pt").to(rag_gen.model.device)

        with torch.no_grad():
            llm_outputs = rag_gen.model.generate(
                **llm_inputs,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=rag_gen.tokenizer.pad_token_id,
            )

        llm_response = rag_gen.tokenizer.decode(
            llm_outputs[0][llm_inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        # 한글 필터링 적용
        llm_response = clean_korean_text(llm_response)
        llm_time = time.time() - start_time
        print(f"  ✅ LLM 완료: {llm_time:.1f}초", flush=True)
        print(f"  📄 LLM 응답 (앞 200자): {llm_response[:200]}...", flush=True)

        # RAG+LLM 응답
        print(f"  🔍 RAG 검색 + LLM 추론 시작...", flush=True)
        start_time = time.time()
        rag_result = rag_gen.generate(question, top_k=3)
        rag_response = rag_result["answer"]
        rag_time = time.time() - start_time
        print(f"  ✅ RAG 완료: {rag_time:.1f}초", flush=True)
        print(f"  📄 RAG 응답 (앞 200자): {rag_response[:200]}...", flush=True)

        # 품질 평가
        quality = evaluate_response_quality(
            question, llm_response, rag_response, expected_keywords
        )

        result = {
            "id": test_case["id"],
            "category": category,
            "subcategory": subcategory,
            "question": question,
            "expected_keywords": expected_keywords,
            "llm_response": llm_response,
            "rag_response": rag_response,
            "llm_time": round(llm_time, 2),
            "rag_time": round(rag_time, 2),
            "llm_keyword_score": quality["llm_keyword_score"],
            "rag_keyword_score": quality["rag_keyword_score"],
            "llm_length": quality["llm_length"],
            "rag_length": quality["rag_length"],
            "context_length": rag_result.get("context_length", 0),
            "sources_count": len(rag_result.get("sources", [])),
        }
        results.append(result)

        # 결과 비교 출력
        print(f"-"*60, flush=True)
        print(f"  📊 결과 비교:", flush=True)
        print(f"     LLM: 시간={llm_time:.1f}s, 키워드점수={quality['llm_keyword_score']:.2f}, 길이={quality['llm_length']}자", flush=True)
        print(f"     RAG: 시간={rag_time:.1f}s, 키워드점수={quality['rag_keyword_score']:.2f}, 길이={quality['rag_length']}자", flush=True)
        winner = "RAG 👑" if quality['rag_keyword_score'] > quality['llm_keyword_score'] else "LLM 👑" if quality['llm_keyword_score'] > quality['rag_keyword_score'] else "동점 🤝"
        print(f"     승자: {winner}", flush=True)

        # 누적 통계
        completed = i + 1
        elapsed_total = sum(r['llm_time'] + r['rag_time'] for r in results)
        avg_per_question = elapsed_total / completed
        remaining = len(test_cases) - completed
        eta_seconds = remaining * avg_per_question
        eta_minutes = eta_seconds / 60
        print(f"  ⏱️  진행: {completed}/{len(test_cases)} 완료 | 예상 남은 시간: {eta_minutes:.1f}분", flush=True)

    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 상세 결과 JSON
    result_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 요약 통계 계산
    summary = calculate_summary(results)

    # 요약 저장
    summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 리포트 출력
    print_report(summary, results)

    return results, summary


def calculate_summary(results: list[dict]) -> dict:
    """요약 통계 계산"""
    total = len(results)

    # 전체 통계
    llm_keyword_scores = [r["llm_keyword_score"] for r in results]
    rag_keyword_scores = [r["rag_keyword_score"] for r in results]
    llm_times = [r["llm_time"] for r in results]
    rag_times = [r["rag_time"] for r in results]

    # 카테고리별 통계
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {
                "count": 0,
                "llm_keyword_scores": [],
                "rag_keyword_scores": [],
            }
        categories[cat]["count"] += 1
        categories[cat]["llm_keyword_scores"].append(r["llm_keyword_score"])
        categories[cat]["rag_keyword_scores"].append(r["rag_keyword_score"])

    # 카테고리별 평균
    for cat, data in categories.items():
        data["llm_avg"] = sum(data["llm_keyword_scores"]) / len(data["llm_keyword_scores"])
        data["rag_avg"] = sum(data["rag_keyword_scores"]) / len(data["rag_keyword_scores"])
        data["improvement"] = data["rag_avg"] - data["llm_avg"]

    # RAG가 더 좋은 경우
    rag_wins = sum(1 for r in results if r["rag_keyword_score"] > r["llm_keyword_score"])
    llm_wins = sum(1 for r in results if r["llm_keyword_score"] > r["rag_keyword_score"])
    ties = total - rag_wins - llm_wins

    return {
        "total_tests": total,
        "overall": {
            "llm_keyword_avg": sum(llm_keyword_scores) / total,
            "rag_keyword_avg": sum(rag_keyword_scores) / total,
            "llm_time_avg": sum(llm_times) / total,
            "rag_time_avg": sum(rag_times) / total,
        },
        "comparison": {
            "rag_wins": rag_wins,
            "llm_wins": llm_wins,
            "ties": ties,
            "rag_win_rate": rag_wins / total * 100,
        },
        "by_category": {
            cat: {
                "count": data["count"],
                "llm_avg": round(data["llm_avg"], 3),
                "rag_avg": round(data["rag_avg"], 3),
                "improvement": round(data["improvement"], 3),
            }
            for cat, data in categories.items()
        },
    }


def print_report(summary: dict, results: list[dict]):
    """평가 리포트 출력"""
    print("\n" + "="*60)
    print("📊 LLM vs RAG+LLM 비교 평가 결과")
    print("="*60)

    print("\n▶ 전체 요약")
    print(f"  총 테스트: {summary['total_tests']}개")
    print(f"  LLM 키워드 점수 평균: {summary['overall']['llm_keyword_avg']:.3f}")
    print(f"  RAG 키워드 점수 평균: {summary['overall']['rag_keyword_avg']:.3f}")
    print(f"  LLM 평균 응답시간: {summary['overall']['llm_time_avg']:.2f}초")
    print(f"  RAG 평균 응답시간: {summary['overall']['rag_time_avg']:.2f}초")

    print("\n▶ 승패 비교 (키워드 매칭 기준)")
    print(f"  RAG 우세: {summary['comparison']['rag_wins']}건 ({summary['comparison']['rag_win_rate']:.1f}%)")
    print(f"  LLM 우세: {summary['comparison']['llm_wins']}건")
    print(f"  동점: {summary['comparison']['ties']}건")

    print("\n▶ 카테고리별 결과")
    for cat, data in summary["by_category"].items():
        improvement_pct = data["improvement"] * 100
        arrow = "↑" if data["improvement"] > 0 else "↓" if data["improvement"] < 0 else "="
        print(f"  {cat}: LLM={data['llm_avg']:.3f} → RAG={data['rag_avg']:.3f} ({arrow}{abs(improvement_pct):.1f}%)")

    # RAG 효과가 큰 예시 출력
    print("\n▶ RAG 효과가 큰 예시 (Top 3)")
    sorted_by_improvement = sorted(
        results,
        key=lambda x: x["rag_keyword_score"] - x["llm_keyword_score"],
        reverse=True
    )[:3]

    for i, r in enumerate(sorted_by_improvement, 1):
        improvement = r["rag_keyword_score"] - r["llm_keyword_score"]
        print(f"\n  {i}. [{r['subcategory']}]")
        print(f"     질문: {r['question'][:40]}...")
        print(f"     LLM: {r['llm_keyword_score']:.2f} → RAG: {r['rag_keyword_score']:.2f} (+{improvement:.2f})")

    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM vs RAG+LLM 비교 평가")
    parser.add_argument(
        "--test-data",
        default="./data/test_dataset.json",
        help="테스트 데이터 경로",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/evaluation_results",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="테스트할 샘플 수 (기본: 전체)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="사용할 LLM 모델",
    )

    args = parser.parse_args()

    run_evaluation(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        model_name=args.model,
    )
