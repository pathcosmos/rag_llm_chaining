#!/usr/bin/env python3
"""
RAG 변별력 검증용 테스트 데이터셋 생성 스크립트
ChromaDB의 판례 데이터를 기반으로 LLM을 활용해 질문-답변 쌍 생성
HuggingFace Transformers 사용 (캐시된 Qwen2.5-7B-Instruct)
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# 로깅 설정
def setup_logging(log_file: str = None):
    """상세 로깅 설정"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    if log_file is None:
        log_file = log_dir / f"generate_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 로거 설정
    logger = logging.getLogger("dataset_generator")
    logger.setLevel(logging.DEBUG)

    # 파일 핸들러 (상세 로그)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # 콘솔 핸들러 (간략 로그)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file

# 전역 로거
logger = None


# 카테고리 및 서브카테고리 정의
CATEGORIES = {
    "case_specific": {
        "description": "특정 판례의 구체적 판시사항을 묻는 질문",
        "subcategories": [
            "상속", "임대차", "부동산", "근로", "손해배상",
            "채권", "형사", "가사", "행정", "회사법"
        ]
    },
    "legal_principle": {
        "description": "법률 원칙과 해석을 묻는 질문",
        "subcategories": [
            "민법원칙", "형법원칙", "소송법", "헌법", "상법원칙"
        ]
    },
    "procedure_detail": {
        "description": "법적 절차의 세부사항을 묻는 질문",
        "subcategories": [
            "소송절차", "등기절차", "집행절차", "신청절차", "항소절차"
        ]
    }
}


class QwenGenerator:
    """HuggingFace Transformers 기반 Qwen 모델 래퍼"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"모델 로딩 완료 (device: {self.model.device})")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """텍스트 생성"""
        messages = [
            {"role": "system", "content": "당신은 한국 법률 전문가입니다. 요청에 따라 정확하게 JSON 형식으로 응답하세요."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()


def sample_diverse_documents(num_samples: int = 250) -> List[Dict]:
    """ChromaDB에서 다양한 판례 샘플링"""
    import chromadb

    client = chromadb.PersistentClient(path=str(project_root / "data/embeddings/chroma"))
    collection = client.get_collection("korean_legal_docs")

    total = collection.count()
    print(f"전체 문서 수: {total:,}")

    samples = []
    seen_cases = set()
    batch_size = 100
    attempts = 0
    max_attempts = num_samples * 10

    while len(samples) < num_samples and attempts < max_attempts:
        offset = random.randint(0, max(0, total - batch_size))
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )

        for doc, meta in zip(batch["documents"], batch["metadatas"]):
            case_name = meta.get("case_name", "")
            case_number = meta.get("case_number", "")
            unique_key = f"{case_name}_{case_number}"

            # [판시사항] 또는 [판결요지] 포함 문서 우선
            if unique_key not in seen_cases and ("[판시사항]" in doc or "[판결요지]" in doc):
                seen_cases.add(unique_key)
                samples.append({
                    "content": doc,
                    "metadata": meta
                })

                if len(samples) >= num_samples:
                    break

        attempts += 1

    print(f"샘플링된 고유 판례: {len(samples)}")
    return samples


def generate_qa_from_document(generator: QwenGenerator, doc: Dict, category: str, subcategory: str, idx: int) -> Optional[Dict]:
    """판례 문서에서 질문-답변 쌍 생성"""
    global logger

    content = doc["content"][:1500]  # 토큰 제한
    meta = doc["metadata"]
    case_name = meta.get("case_name", "법률 사건")
    case_number = meta.get("case_number", "")
    case_type = meta.get("case_type", "민사")

    if logger:
        logger.debug(f"[{idx}] 처리 시작 - 사건: {case_name} ({case_number})")
        logger.debug(f"[{idx}] 카테고리: {category}/{subcategory}")

    prompt = f"""아래 판례 내용을 기반으로 RAG 시스템 테스트용 질문과 정답을 생성하세요.

[판례 내용]
{content}

[요구사항]
1. 카테고리: {category} ({CATEGORIES[category]['description']})
2. 이 판례의 핵심 법리나 판시사항을 묻는 질문을 한글로 작성
3. 질문은 구체적이고 판례 내용을 알아야만 정확히 답변 가능해야 함
4. 정답은 판례 내용 기반으로 2-3문장으로 작성

[출력 형식] 반드시 아래 JSON 형식으로만 응답:
{{"question": "질문 내용", "ground_truth": "정답 내용", "keywords": ["키워드1", "키워드2", "키워드3"]}}"""

    try:
        import time
        start_time = time.time()
        response = generator.generate(prompt, max_new_tokens=400, temperature=0.7)
        gen_time = time.time() - start_time

        if logger:
            logger.debug(f"[{idx}] LLM 응답 시간: {gen_time:.2f}초")
            logger.debug(f"[{idx}] LLM 응답 (처음 200자): {response[:200]}...")

        # JSON 파싱
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            question = data.get("question", "").strip()
            ground_truth = data.get("ground_truth", "").strip()
            keywords = data.get("keywords", [])

            if question and ground_truth and len(question) > 10 and len(ground_truth) > 20:
                result = {
                    "id": f"{category}_{idx:03d}",
                    "category": category,
                    "subcategory": subcategory,
                    "question": question,
                    "ground_truth": ground_truth,
                    "expected_keywords": keywords[:5],
                    "difficulty": random.choice(["medium", "hard"]),
                    "rag_advantage": "high",
                    "source_case": {
                        "case_name": case_name,
                        "case_number": case_number,
                        "case_type": case_type,
                        "court": meta.get("court", "")
                    }
                }
                if logger:
                    logger.info(f"[{idx}] ✓ 생성 성공 - Q: {question[:50]}...")
                    logger.debug(f"[{idx}] 전체 결과: {json.dumps(result, ensure_ascii=False)[:300]}")
                return result
            else:
                if logger:
                    logger.warning(f"[{idx}] ✗ 검증 실패 - 질문/답변 길이 부족")
        else:
            if logger:
                logger.warning(f"[{idx}] ✗ JSON 파싱 실패 - JSON 구조 없음")

    except json.JSONDecodeError as e:
        if logger:
            logger.error(f"[{idx}] ✗ JSON 파싱 오류: {e}")
            logger.debug(f"[{idx}] 원본 응답: {response[:500]}")
    except Exception as e:
        if logger:
            logger.error(f"[{idx}] ✗ 예외 발생: {type(e).__name__}: {e}")

    return None


def generate_dataset(num_samples: int = 200, output_path: str = None, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """테스트 데이터셋 생성"""

    print("=" * 60)
    print(f"RAG 변별력 테스트 데이터셋 생성 ({num_samples}개)")
    print("=" * 60)

    # 1. 모델 로딩
    print("\n[1/4] LLM 모델 로딩 중...")
    generator = QwenGenerator(model_name=model_name)

    # 2. 판례 샘플링
    print("\n[2/4] 판례 샘플링 중...")
    documents = sample_diverse_documents(num_samples + 100)

    # 3. 카테고리별 분배
    categories = list(CATEGORIES.keys())
    samples_per_category = num_samples // len(categories)

    # 4. 질문-답변 생성
    print("\n[3/4] 질문-답변 생성 중...")
    test_cases = []
    category_counts = {cat: 0 for cat in categories}

    doc_idx = 0
    pbar = tqdm(total=num_samples, desc="생성 중")

    while len(test_cases) < num_samples and doc_idx < len(documents):
        doc = documents[doc_idx]
        doc_idx += 1

        # 카테고리 순환
        for category in categories:
            if category_counts[category] >= samples_per_category + 10:
                continue

            subcategory = random.choice(CATEGORIES[category]["subcategories"])

            qa = generate_qa_from_document(
                generator,
                doc,
                category,
                subcategory,
                len(test_cases) + 1
            )

            if qa:
                test_cases.append(qa)
                category_counts[category] += 1
                pbar.update(1)

                if len(test_cases) >= num_samples:
                    break

            break

    pbar.close()

    # 5. 데이터셋 저장
    print("\n[4/4] 데이터셋 저장 중...")

    dataset = {
        "metadata": {
            "version": "2.0",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "판례 DB 특화 테스트 - RAG 변별력 검증용 (확장판)",
            "purpose": "RAG가 판례 DB를 활용해야만 정확히 답변할 수 있는 질문들",
            "total_questions": len(test_cases),
            "categories": {cat: category_counts[cat] for cat in categories},
            "generation_model": model_name
        },
        "test_cases": test_cases
    }

    if output_path is None:
        output_path = project_root / "data" / f"test_dataset_case_specific_{len(test_cases)}.json"
    else:
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"데이터셋 생성 완료!")
    print(f"- 총 질문 수: {len(test_cases)}")
    print(f"- 카테고리별:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count}")
    print(f"- 저장 경로: {output_path}")
    print(f"{'=' * 60}")

    return dataset


def main():
    global logger

    parser = argparse.ArgumentParser(description="RAG 테스트 데이터셋 생성 (HuggingFace Transformers)")
    parser.add_argument("--num-samples", type=int, default=200, help="생성할 질문 수")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace 모델명")
    parser.add_argument("--log-file", type=str, default=None, help="로그 파일 경로")

    args = parser.parse_args()

    # 로깅 초기화
    logger, log_file = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("RAG 테스트 데이터셋 생성 시작")
    logger.info(f"로그 파일: {log_file}")
    logger.info("=" * 60)

    # GPU 확인
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU 사용 가능: {gpu_name}")
        logger.info(f"VRAM: {vram:.1f} GB")
    else:
        logger.warning("GPU를 사용할 수 없습니다. CPU로 실행됩니다 (매우 느림)")

    logger.info(f"설정: num_samples={args.num_samples}, model={args.model}")

    generate_dataset(num_samples=args.num_samples, output_path=args.output, model_name=args.model)

    logger.info("=" * 60)
    logger.info("작업 완료")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
