# RAG LLM Chaining

Korean Legal RAG System with LLM Chaining

## Overview

상용 배포 가능한 오픈소스 LLM과 한글 법률 데이터셋을 활용한 RAG(Retrieval-Augmented Generation) 시스템

## Features

- **상용 배포 가능**: Apache 2.0 라이선스 모델 사용
- **한글 특화**: 법률 도메인 한글 데이터셋 활용
- **RAG 파이프라인**: LangChain 기반 검색-생성 시스템
- **VectorDB**: ChromaDB를 이용한 임베딩 저장 및 검색

## Tech Stack

- **LLM**: Qwen2.5-7B-Instruct (Apache 2.0)
- **Embedding**: BGE-M3 (MIT)
- **VectorDB**: ChromaDB
- **Framework**: LangChain
- **API**: FastAPI
- **UI**: Gradio

## Installation

```bash
# uv 사용
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 또는 pip 사용
pip install -e ".[dev]"
```

## Usage

```python
from src.rag.pipeline import RAGPipeline

# RAG 파이프라인 초기화
rag = RAGPipeline()

# 질의
response = rag.query("상속포기 절차와 기한은?")
print(response)
```

## Project Structure

```
rag_llm_chaining/
├── config/           # 설정 파일
├── data/             # 데이터 디렉토리
├── src/              # 소스 코드
│   ├── data/         # 데이터 로더
│   ├── embedding/    # 임베딩 모델
│   ├── llm/          # LLM 래퍼
│   ├── rag/          # RAG 파이프라인
│   └── api/          # FastAPI
├── tests/            # 테스트
├── notebooks/        # Jupyter 노트북
└── scripts/          # 스크립트
```

## License

MIT License
