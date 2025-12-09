"""
Ollama (Llama 3.1) - LLM vs RAG+LLM 비교 평가 스크립트
"""

import json
import time
import os
import sys
import re
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embedding.vectordb import VectorStore

# Ollama 연동
import requests


class OllamaClient:
    """Ollama API 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b-instruct-q8_0"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, system: str = None, temperature: float = 0.1) -> str:
        """Ollama API로 텍스트 생성"""
        url = f"{self.base_url}/api/generate"

        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\nUser: {prompt}\n\nAssistant:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 512,
            }
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        return response.json().get("response", "")


def clean_korean_text(text: str) -> str:
    """한글, 숫자, 기본 문장부호만 남기고 필터링"""
    cleaned = re.sub(r'[^\uAC00-\uD7A3\u3131-\u3163\u1100-\u11FF0-9\s.,!?:;\'"()\-/]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    korean_chars = re.findall(r'[\uAC00-\uD7A3]', cleaned)
    if len(korean_chars) < 10:
        return text

    return cleaned


def calculate_keyword_score(response: str, keywords: list[str]) -> float:
    """키워드 매칭 점수 계산"""
    if not keywords:
        return 0.0

    response_lower = response.lower()
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    return matched / len(keywords)


def run_evaluation(
    test_data_path: str,
    output_dir: str,
    sample_size: int = None,
    ollama_model: str = "llama3.1:8b-instruct-q8_0",
    chroma_host: str = "211.231.121.68",
    chroma_port: int = 8081,
):
    """평가 실행"""

    # 테스트 데이터 로드
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_cases = test_data["test_cases"]
    if sample_size:
        test_cases = test_cases[:sample_size]

    print(f"\n{'='*60}")
    print(f"🦙 Ollama LLM vs RAG+LLM 비교 평가")
    print(f"{'='*60}")
    print(f"테스트 케이스: {len(test_cases)}개")
    print(f"모델: {ollama_model}")
    print(f"ChromaDB: {chroma_host}:{chroma_port}")
    print(f"{'='*60}\n")

    # Ollama 클라이언트 초기화
    print("1. Ollama 클라이언트 초기화...")
    ollama = OllamaClient(model=ollama_model)

    # 연결 테스트
    try:
        test_response = ollama.generate("안녕", temperature=0.1)
        print(f"   ✅ Ollama 연결 성공: {test_response[:50]}...")
    except Exception as e:
        print(f"   ❌ Ollama 연결 실패: {e}")
        return

    # VectorStore 초기화
    print("\n2. VectorStore 초기화 (원격 ChromaDB)...")
    try:
        vector_store = VectorStore(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name="korean_legal_docs",
            embedding_model="BAAI/bge-m3",
            device="cuda",
        )
        print(f"   ✅ VectorDB 연결 성공: {vector_store.count()}개 문서")
    except Exception as e:
        print(f"   ❌ VectorDB 연결 실패: {e}")
        print("   로컬 ChromaDB 시도 중...")
        vector_store = VectorStore(
            persist_directory="./data/embeddings/chroma",
            collection_name="korean_legal_docs",
            embedding_model="BAAI/bge-m3",
            device="cuda",
        )
        print(f"   ✅ 로컬 VectorDB 연결 성공: {vector_store.count()}개 문서")

    results = []

    print(f"\n3. 평가 시작 ({len(test_cases)}개 질문)...\n")

    system_prompt_llm = """당신은 한국 법률 전문가입니다.
질문에 정확하고 이해하기 쉽게 답변해주세요.
반드시 한국어로만 답변하세요."""

    system_prompt_rag = """당신은 한국 법률 전문가입니다.
주어진 참고 자료를 바탕으로 질문에 정확하고 이해하기 쉽게 답변해주세요.
참고 자료에 없는 내용은 "참고 자료에서 관련 정보를 찾을 수 없습니다"라고 답변하세요.
반드시 한국어로만 답변하세요."""

    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        category = test_case["category"]
        subcategory = test_case["subcategory"]
        expected_keywords = test_case.get("expected_keywords", [])

        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(test_cases)}] {category}/{subcategory}", flush=True)
        print(f"  질문: {question[:60]}...", flush=True)
        print(f"-"*60, flush=True)

        # 1. LLM 단독 응답
        print(f"  📝 LLM Only 추론 중...", flush=True)
        start_time = time.time()
        try:
            llm_response = ollama.generate(question, system=system_prompt_llm, temperature=0.1)
            llm_response = clean_korean_text(llm_response)
        except Exception as e:
            llm_response = f"오류: {e}"
        llm_time = time.time() - start_time
        print(f"     ✅ LLM: {llm_time:.1f}초 | {llm_response[:80]}...", flush=True)

        # 2. RAG 검색
        print(f"  🔍 RAG 검색 중...", flush=True)
        start_time = time.time()
        try:
            retrieved_docs = vector_store.search(question, top_k=3)

            # 컨텍스트 구성
            context_parts = []
            for j, doc in enumerate(retrieved_docs, 1):
                metadata = doc.get("metadata", {})
                case_info = f"[{metadata.get('case_name', '판례')} - {metadata.get('court', '')}]"
                context_parts.append(f"{j}. {case_info}\n{doc['content'][:500]}")

            context = "\n\n".join(context_parts)

            # RAG 프롬프트
            rag_prompt = f"""참고 자료:
{context}

질문: {question}

답변:"""

            rag_response = ollama.generate(rag_prompt, system=system_prompt_rag, temperature=0.1)
            rag_response = clean_korean_text(rag_response)

        except Exception as e:
            rag_response = f"오류: {e}"
            retrieved_docs = []
            context = ""

        rag_time = time.time() - start_time
        print(f"     ✅ RAG: {rag_time:.1f}초 | {rag_response[:80]}...", flush=True)

        # 점수 계산
        llm_score = calculate_keyword_score(llm_response, expected_keywords)
        rag_score = calculate_keyword_score(rag_response, expected_keywords)

        # 결과 저장
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
            "llm_keyword_score": llm_score,
            "rag_keyword_score": rag_score,
            "llm_length": len(llm_response),
            "rag_length": len(rag_response),
            "context_length": len(context) if context else 0,
            "sources_count": len(retrieved_docs),
        }
        results.append(result)

        # 결과 출력
        winner = "RAG 👑" if rag_score > llm_score else "LLM 👑" if llm_score > rag_score else "동점 🤝"
        print(f"  📊 LLM={llm_score:.2f} vs RAG={rag_score:.2f} → {winner}", flush=True)

        # 진행률
        completed = i + 1
        elapsed_total = sum(r['llm_time'] + r['rag_time'] for r in results)
        avg_per_q = elapsed_total / completed
        remaining = len(test_cases) - completed
        eta_min = (remaining * avg_per_q) / 60
        print(f"  ⏱️  {completed}/{len(test_cases)} 완료 | ETA: {eta_min:.1f}분", flush=True)

    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_file = os.path.join(output_dir, f"ollama_eval_{timestamp}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 요약 계산
    summary = calculate_summary(results)

    summary_file = os.path.join(output_dir, f"ollama_summary_{timestamp}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 리포트 출력
    print_report(summary, results)

    print(f"\n📁 결과 저장: {result_file}")
    print(f"📁 요약 저장: {summary_file}")

    return results, summary


def calculate_summary(results: list[dict]) -> dict:
    """요약 통계 계산"""
    total = len(results)

    llm_scores = [r["llm_keyword_score"] for r in results]
    rag_scores = [r["rag_keyword_score"] for r in results]
    llm_times = [r["llm_time"] for r in results]
    rag_times = [r["rag_time"] for r in results]

    # 카테고리별 통계
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"llm": [], "rag": []}
        categories[cat]["llm"].append(r["llm_keyword_score"])
        categories[cat]["rag"].append(r["rag_keyword_score"])

    # 승패
    rag_wins = sum(1 for r in results if r["rag_keyword_score"] > r["llm_keyword_score"])
    llm_wins = sum(1 for r in results if r["llm_keyword_score"] > r["rag_keyword_score"])
    ties = total - rag_wins - llm_wins

    return {
        "total_tests": total,
        "model": "llama3.1:8b-instruct-q8_0",
        "overall": {
            "llm_keyword_avg": sum(llm_scores) / total,
            "rag_keyword_avg": sum(rag_scores) / total,
            "llm_time_avg": sum(llm_times) / total,
            "rag_time_avg": sum(rag_times) / total,
            "improvement": (sum(rag_scores) - sum(llm_scores)) / total,
        },
        "comparison": {
            "rag_wins": rag_wins,
            "llm_wins": llm_wins,
            "ties": ties,
            "rag_win_rate": rag_wins / total * 100,
        },
        "by_category": {
            cat: {
                "llm_avg": sum(data["llm"]) / len(data["llm"]),
                "rag_avg": sum(data["rag"]) / len(data["rag"]),
                "improvement": (sum(data["rag"]) - sum(data["llm"])) / len(data["llm"]),
            }
            for cat, data in categories.items()
        },
    }


def print_report(summary: dict, results: list[dict]):
    """평가 리포트 출력"""
    print("\n" + "="*60)
    print("🦙 Ollama LLM vs RAG+LLM 비교 평가 결과")
    print("="*60)

    print(f"\n▶ 모델: {summary['model']}")
    print(f"▶ 총 테스트: {summary['total_tests']}개")

    print(f"\n▶ 전체 점수 (키워드 매칭)")
    print(f"  LLM Only:  {summary['overall']['llm_keyword_avg']:.3f}")
    print(f"  RAG + LLM: {summary['overall']['rag_keyword_avg']:.3f}")
    improvement_pct = summary['overall']['improvement'] * 100
    arrow = "↑" if improvement_pct > 0 else "↓"
    print(f"  개선율: {arrow}{abs(improvement_pct):.1f}%")

    print(f"\n▶ 응답 시간")
    print(f"  LLM Only:  {summary['overall']['llm_time_avg']:.2f}초")
    print(f"  RAG + LLM: {summary['overall']['rag_time_avg']:.2f}초")

    print(f"\n▶ 승패 (키워드 매칭 기준)")
    print(f"  RAG 우세: {summary['comparison']['rag_wins']}건 ({summary['comparison']['rag_win_rate']:.1f}%)")
    print(f"  LLM 우세: {summary['comparison']['llm_wins']}건")
    print(f"  동점: {summary['comparison']['ties']}건")

    print(f"\n▶ 카테고리별 결과")
    for cat, data in summary["by_category"].items():
        imp = data["improvement"] * 100
        arrow = "↑" if imp > 0 else "↓" if imp < 0 else "="
        print(f"  {cat}: LLM={data['llm_avg']:.3f} → RAG={data['rag_avg']:.3f} ({arrow}{abs(imp):.1f}%)")

    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ollama LLM vs RAG+LLM 비교 평가")
    parser.add_argument("--test-data", default="./data/test_dataset_case_specific_200.json")
    parser.add_argument("--output-dir", default="./data/evaluation_ollama")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--model", default="llama3.1:8b-instruct-q8_0")
    parser.add_argument("--chroma-host", default="211.231.121.68")
    parser.add_argument("--chroma-port", type=int, default=8081)

    args = parser.parse_args()

    run_evaluation(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        ollama_model=args.model,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
    )
