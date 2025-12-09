# RAG + LLM Chaining 프로젝트

## 프로젝트 개요

상용 배포 가능한 오픈소스 LLM과 한글 공개 데이터셋(법률 등)을 활용한 RAG(Retrieval-Augmented Generation) 시스템 구축 프로젝트

## 프로젝트 목표

1. **상용 배포 가능 LLM 확보**: Apache 2.0, MIT 등 상용 라이선스 모델 선정
2. **한글 데이터셋 수집**: 법률, 의료, 금융 등 공개 데이터셋 탐색 및 확보
3. **VectorDB 구축**: 임베딩 저장 및 유사도 검색 시스템 구현
4. **RAG 파이프라인**: LangChain 기반 검색-생성 파이프라인 구축
5. **품질 검증**: 데이터셋 적용 효과 측정 및 검증

---

## 인프라 환경

### 역할 분담

```
┌─────────────────────────────────────────────────────────────────────┐
│                        작업 분배 전략                                │
├─────────────────────────────────────────────────────────────────────┤
│  로컬 PC (GPU 작업)              │  리모트 서버 (Non-GPU 작업)       │
│  ─────────────────               │  ──────────────────────           │
│  • LLM 추론 (vLLM/Ollama)        │  • 데이터 전처리                  │
│  • 임베딩 생성                    │  • VectorDB (ChromaDB)           │
│  • 모델 파인튜닝                  │  • API 서버 (FastAPI)            │
│  • 추론 API 서버                  │  • 데이터 스크래핑               │
│                                   │  • 배치 작업                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 로컬 PC (GPU 서버) - Primary

| 항목 | 값 |
|------|-----|
| GPU | **NVIDIA GeForce RTX 5070 Ti** |
| VRAM | **16,303 MiB** (가용: ~15,800 MiB) |
| CPU | 20코어 |
| RAM | 125.5 GB |
| OS | Ubuntu 24.04.3 LTS |
| Docker | v28.5.2 |
| CUDA Driver | 580.105.08 |
| Docker GPU | CDI 모드 (`--device nvidia.com/gpu=all`) |

**Docker GPU 사용법**:
```bash
# CDI 모드로 GPU 컨테이너 실행
docker run --rm --device nvidia.com/gpu=all <이미지> <명령>

# 예시: LLM 추론 컨테이너
docker run --rm --device nvidia.com/gpu=all \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  vllm/vllm-openai:latest
```

**권장 모델 (16GB VRAM)**:

| 모델 | 양자화 | VRAM 사용량 | 성능 |
|------|--------|-------------|------|
| Qwen2.5-14B | Q4_K_M | ~10GB | 최고 |
| Qwen2.5-7B | Q8_0 | ~8GB | 우수 |
| SOLAR-10.7B | Q4_K_M | ~7GB | 우수 (한국어) |
| Llama-3.1-8B | FP16 | ~16GB | 우수 |

### 리모트 서버 (데이터 처리) - Secondary

| 항목 | 값 |
|------|-----|
| Host | 211.231.121.68 |
| Port | 22225 |
| User | lanco |
| OS | Ubuntu (Linux 6.8.0-88-generic) |
| Docker | v28.5.2 |
| 디스크 | 루트: 418GB, /home: 888GB 가용 |
| 용도 | 데이터 처리, VectorDB, API 서버 |

**접속 방법**:
```bash
# SSH 접속 (비밀번호 인증)
sshpass -p '$REMOTE_PASSWORD' ssh -p 22225 lanco@211.231.121.68

# 또는 로컬에서 직접 명령 실행
sshpass -p '$REMOTE_PASSWORD' ssh -p 22225 lanco@211.231.121.68 "docker ps"
```

**리모트 서버 작업 예시**:
```bash
# 데이터 처리 컨테이너 실행
ssh remote "docker run -d --name vectordb \
  -v /home/lanco/data:/data \
  -p 8080:8080 \
  chromadb/chroma"
```

### HuggingFace 계정

| 항목 | 값 |
|------|-----|
| 사용자 | somebody-to-love (ghong) |
| 토큰 권한 | read (읽기 전용) |
| 토큰 이름 | dy_ai_server |

### 환경변수 (.env)

```bash
# 리모트 서버 접속
REMOTE_HOST=211.231.121.68
REMOTE_PORT=22225
REMOTE_USER=lanco
REMOTE_PASSWORD=<비밀번호>

# HuggingFace
HF_TOKEN=<토큰>

# 로컬 GPU Docker (CDI 모드)
DOCKER_GPU_FLAG="--device nvidia.com/gpu=all"
```

### 서비스 포트 계획

| 서비스 | 로컬 PC | 리모트 서버 | 설명 |
|--------|---------|-------------|------|
| LLM API | 8000 | - | vLLM/Ollama 추론 API |
| Embedding API | 8001 | - | 임베딩 생성 API |
| VectorDB | - | 8080 | ChromaDB |
| RAG API | - | 8081 | FastAPI 메인 서버 |
| Gradio UI | - | 7860 | 데모 UI |

---

## Phase 1: LLM 모델 탐색 및 선정

### 선정된 1차 모델: Qwen2.5-7B-Instruct

**법률 RAG 프로젝트 최적 모델 선정 이유**:

| 평가 항목 | Qwen2.5-7B | SOLAR-10.7B | Bllossom-8B |
|-----------|------------|-------------|-------------|
| 라이선스 | ⭐ Apache 2.0 | ⭐ Apache 2.0 | Llama (MAU 제한) |
| 한글 성능 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 컨텍스트 | ⭐ 128K | 4K | ⭐ 128K |
| VRAM (FP16) | ⭐ ~14GB | ~22GB | ~16GB |
| 멀티턴 대화 | ⭐ 우수 | 제한적 | 우수 |
| JSON 출력 | ⭐ 우수 | 보통 | 보통 |
| 파인튜닝 | ⭐ 용이 | 용이 | 용이 |

**Qwen2.5-7B-Instruct 선정 이유**:
1. **완전한 상용 라이선스** (Apache 2.0) - 판매/배포 제한 없음
2. **128K 컨텍스트** - 긴 법률 문서 처리 가능
3. **구조화된 출력** - JSON, 테이블 출력 우수 (RAG 응답 포맷팅)
4. **16GB VRAM에서 FP16 실행 가능** - 양자화 없이 풀 성능
5. **멀티턴 대화** - 법률 상담 시나리오에 적합
6. **GGUF 공식 제공** - 다양한 양자화 옵션

```yaml
# 1차 테스트 모델
primary_model:
  name: Qwen2.5-7B-Instruct
  huggingface: Qwen/Qwen2.5-7B-Instruct
  license: Apache-2.0
  parameters: 7B
  context_length: 128K
  vram_fp16: ~14GB
  vram_q4: ~5GB
```

### 모델 테스트 순서 (순차 교체)

| 순서 | 모델 | 목적 | VRAM |
|------|------|------|------|
| 1 | **Qwen2.5-7B-Instruct** | 기본 RAG 파이프라인 구축 | ~14GB (FP16) |
| 2 | Qwen2.5-14B-Instruct | 성능 향상 테스트 | ~10GB (Q4) |
| 3 | SOLAR-10.7B-Instruct | 한국어 특화 비교 | ~7GB (Q4) |
| 4 | Bllossom-8B | 한글 최적화 비교 | ~5GB (Q4) |

### 상용 배포 가능 LLM 후보군 (전체)

#### Tier 1: 완전 상용 가능 (Apache 2.0 / MIT)

| 모델 | 라이선스 | 파라미터 | Instruct | 한글 | HuggingFace |
|------|----------|----------|----------|------|-------------|
| **Qwen 2.5** | Apache 2.0 | 0.5B~72B | ✅ | ⭐⭐⭐⭐ | [Qwen/Qwen2.5-*](https://huggingface.co/Qwen) |
| **Qwen 3** | Apache 2.0 | 235B MoE | ✅ | ⭐⭐⭐⭐⭐ | [Qwen/Qwen3-*](https://huggingface.co/Qwen) |
| **SOLAR 10.7B** | Apache 2.0 | 10.7B | ✅ | ⭐⭐⭐⭐⭐ | [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) |
| **DeepSeek R1/V3** | MIT | 671B MoE | ✅ | ⭐⭐⭐⭐ | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| **Mistral** | Apache 2.0 | 7B~22B | ✅ | ⭐⭐⭐ | [mistralai/Mistral-*](https://huggingface.co/mistralai) |

#### Tier 2: 상용 가능 (조건부)

| 모델 | 라이선스 | 파라미터 | Instruct | 한글 | 제한 |
|------|----------|----------|----------|------|------|
| **Llama 3.1/3.2** | Llama Community | 1B~70B | ✅ | ⭐⭐⭐ | MAU 7억 초과시 협의 |
| **Gemma 2** | Gemma License | 9B/27B | ✅ | ⭐⭐⭐ | 일부 사용 제한 |
| **Bllossom** | Llama License | 3B~70B | ✅ | ⭐⭐⭐⭐⭐ | MAU 7억 초과시 협의 |

#### Tier 3: 비상용 (연구/참고용)

| 모델 | 라이선스 | 파라미터 | 한글 | 비고 |
|------|----------|----------|------|------|
| EXAONE 3.5 | 비상용 | 2.4B~32B | ⭐⭐⭐⭐⭐ | LG, 연구용만 |
| EEVE-Korean | CC-BY-NC | 10.8B | ⭐⭐⭐⭐⭐ | Yanolja, 비상용 |

### 임베딩 모델 후보

| 모델 | 라이선스 | 차원 | 한글 | HuggingFace |
|------|----------|------|------|-------------|
| **BGE-M3** | MIT | 1024 | ⭐⭐⭐⭐⭐ | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| **multilingual-e5-large** | MIT | 1024 | ⭐⭐⭐⭐ | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) |
| **KoSimCSE-roberta** | MIT | 768 | ⭐⭐⭐⭐⭐ | [BM-K/KoSimCSE-roberta](https://huggingface.co/BM-K/KoSimCSE-roberta) |
| **ko-sroberta-multitask** | MIT | 768 | ⭐⭐⭐⭐ | [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask) |

### 선정 기준 체크리스트
- [x] 상용 배포 라이선스 (Apache 2.0, MIT 우선)
- [x] 한글 성능 벤치마크 확인
- [x] 로컬 실행 가능 크기 (16GB VRAM)
- [x] Instruct 모델 및 파인튜닝 지원
- [x] 긴 컨텍스트 지원 (법률 문서용)

---

## Phase 2: 한글 공개 데이터셋 탐색

### 법률 분야 (우선순위 높음)

| 데이터셋 | 출처 | 규모 | 라이선스 |
|----------|------|------|----------|
| 법률 QA 데이터셋 | AI Hub | 10만+ | 공공누리 |
| 판례 데이터 | 법제처/대법원 | 50만+ | 공공데이터 |
| 법령 데이터 | 국가법령정보센터 | 전체 법령 | 공공 |
| 법률 상담 데이터 | AI Hub | 5만+ | 공공누리 |

### 기타 분야 후보

| 분야 | 데이터셋 | 출처 |
|------|----------|------|
| 의료 | 의료 QA, 진료기록 | AI Hub |
| 금융 | 금융 상담, 뉴스 | AI Hub, KRX |
| 행정 | 민원 상담, 정책 | AI Hub |
| 특허 | 특허 문서 | KIPRIS |

### 데이터셋 확보 체크리스트
- [ ] AI Hub 회원가입 및 데이터 신청
- [ ] 국가법령정보센터 API 키 발급
- [ ] 대법원 판례 Open API 신청
- [ ] 라이선스 조건 검토 (상용 가능 여부)

---

## Phase 3: 기술 스택 및 아키텍처

### 핵심 기술 스택

```yaml
LLM:
  inference: vLLM | Ollama | llama.cpp
  framework: Transformers, PEFT

Embedding:
  model: sentence-transformers

VectorDB:
  primary: ChromaDB | Milvus | Qdrant
  backup: FAISS | Pinecone

RAG Framework:
  primary: LangChain
  alternative: LlamaIndex

Backend:
  api: FastAPI
  async: asyncio, aiohttp

Frontend (Optional):
  ui: Gradio | Streamlit
```

### 시스템 아키텍처

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   사용자    │────▶│  FastAPI    │────▶│  LangChain  │
│   질의      │     │  서버       │     │  Pipeline   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
            ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
            │  Embedding  │            │  VectorDB   │            │    LLM      │
            │   Model     │            │  (Chroma)   │            │  (Qwen/    │
            └─────────────┘            └─────────────┘            │   SOLAR)   │
                                                                  └─────────────┘
```

---

## Phase 4: 구현 태스크

### Task 1: 환경 설정
- [ ] Python 가상환경 구성 (3.10+)
- [ ] 의존성 패키지 설치 (requirements.txt)
- [ ] GPU/CUDA 환경 확인
- [ ] 디렉토리 구조 생성

### Task 2: 데이터 수집 및 전처리
- [ ] 법률 데이터셋 다운로드
- [ ] 데이터 정제 (노이즈 제거, 포맷 통일)
- [ ] 청킹 전략 수립 (문단/문장/토큰 기준)
- [ ] 메타데이터 추출 및 구조화

### Task 3: VectorDB 구축
- [ ] ChromaDB 설정
- [ ] 임베딩 모델 로드
- [ ] 문서 임베딩 및 저장
- [ ] 인덱스 최적화

### Task 4: RAG 파이프라인 구축
- [ ] LangChain 설정
- [ ] Retriever 구현 (유사도 검색)
- [ ] Prompt Template 설계
- [ ] Chain 구성 (RetrievalQA)

### Task 5: LLM 연동
- [ ] 모델 다운로드 및 로드
- [ ] 추론 최적화 (quantization)
- [ ] LangChain LLM wrapper 구현

### Task 6: 검증 및 평가
- [ ] 테스트 질의 세트 구성
- [ ] RAG vs Non-RAG 비교
- [ ] 응답 품질 평가 지표 (RAGAS, 정확도)
- [ ] 레이턴시 및 처리량 측정

### Task 7: API 및 UI
- [ ] FastAPI 엔드포인트 구현
- [ ] Gradio 데모 UI 구축
- [ ] 에러 핸들링 및 로깅

---

## Phase 5: 품질 검증 방법

### 평가 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| Faithfulness | 검색 문서 기반 응답 충실도 | >0.8 |
| Answer Relevance | 질문-응답 관련성 | >0.8 |
| Context Precision | 검색 문맥 정확도 | >0.7 |
| Context Recall | 필요 정보 검색률 | >0.7 |
| Latency | 응답 시간 | <5s |

### 테스트 시나리오

```python
test_queries = [
    "상속포기 절차와 기한은?",
    "임대차보호법상 보증금 보호 범위는?",
    "근로기준법상 연차휴가 일수 계산 방법은?",
    "형사소송에서 국선변호인 선정 요건은?",
]
```

### 비교 실험
1. **Baseline**: LLM 단독 (RAG 없음)
2. **RAG v1**: 기본 유사도 검색
3. **RAG v2**: 리랭킹 적용
4. **RAG v3**: HyDE (가설 문서 생성)

---

## 디렉토리 구조

```
rag_llm_chaining/
├── CLAUDE.md              # 프로젝트 문서
├── README.md              # 사용자 가이드
├── requirements.txt       # 의존성
├── .env.example           # 환경변수 템플릿
├── config/
│   ├── model_config.yaml  # 모델 설정
│   └── vector_config.yaml # VectorDB 설정
├── data/
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리 데이터
│   └── embeddings/        # 캐시된 임베딩
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py      # 데이터 로더
│   │   └── preprocessor.py # 전처리기
│   ├── embedding/
│   │   ├── model.py       # 임베딩 모델
│   │   └── vectordb.py    # VectorDB 연동
│   ├── llm/
│   │   ├── model.py       # LLM 래퍼
│   │   └── prompts.py     # 프롬프트 템플릿
│   ├── rag/
│   │   ├── chain.py       # RAG 체인
│   │   ├── retriever.py   # 검색기
│   │   └── pipeline.py    # 전체 파이프라인
│   └── api/
│       ├── main.py        # FastAPI 앱
│       └── routes.py      # API 라우트
├── tests/
│   ├── test_embedding.py
│   ├── test_retrieval.py
│   └── test_rag.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_rag_evaluation.ipynb
└── scripts/
    ├── download_data.py   # 데이터 다운로드
    ├── build_vectordb.py  # VectorDB 구축
    └── evaluate.py        # 평가 스크립트
```

---

## 개발 규칙

### 코드 스타일
- Python 3.10+ 타입 힌트 사용
- Black + isort 포매팅
- docstring 필수 (Google style)

### Git 커밋 컨벤션
- `feat:` 새 기능
- `fix:` 버그 수정
- `docs:` 문서 수정
- `refactor:` 리팩토링
- `test:` 테스트 추가

### 보안
- API 키, 토큰 → `.env` 파일 (git 제외)
- 민감 데이터 로깅 금지
- 모델 파일 → `.gitignore`

---

## 참고 리소스

### 문서
- [LangChain 공식 문서](https://python.langchain.com/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [AI Hub](https://aihub.or.kr/)

### 관련 프로젝트
- [ko-llm-leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)
- [RAGAS](https://docs.ragas.io/)

---

## 현재 진행 상태

- [x] 프로젝트 초기화
- [x] CLAUDE.md 작성
- [ ] Phase 1: LLM 모델 선정
- [ ] Phase 2: 데이터셋 확보
- [ ] Phase 3: 환경 구축
- [ ] Phase 4: 구현
- [ ] Phase 5: 검증
