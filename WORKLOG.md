# 작업 로그 (WORKLOG)

프로젝트 진행 과정, 결정 사항, 변경 이력을 기록합니다.

---

## 2025-12-05

### 세션 1: 프로젝트 초기화 및 환경 설정

#### 1. 프로젝트 생성 및 목표 설정

**목표**: 상용 배포 가능한 오픈소스 LLM + 한글 법률 데이터셋을 활용한 RAG 시스템 구축

**주요 요구사항**:
- 상용 판매/배포 제한 없는 라이선스 (Apache 2.0, MIT)
- 한글 언어에 유리한 LLM 모델
- Instruct 모델 지원
- 파인튜닝 가능

---

#### 2. 인프라 환경 검증

**로컬 PC (GPU 작업용)**:
| 항목 | 값 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM | 16,303 MiB |
| CPU | 20코어 |
| RAM | 125.5 GB |
| OS | Ubuntu 24.04.3 LTS |
| Docker | v28.5.2 (CDI 모드) |

**리모트 서버 (데이터 처리용)**:
| 항목 | 값 |
|------|-----|
| Host | 211.231.121.68:22225 |
| GPU | GTX 1060 3GB (미사용) |
| 용도 | 데이터 처리, VectorDB, API 서버 |

**결정**:
- GPU 집약 작업 → 로컬 PC Docker
- Non-GPU 작업 → 리모트 서버 Docker

**검증 완료**:
- [x] SSH 연결 테스트 (비밀번호 인증)
- [x] HuggingFace 토큰 유효성 확인
- [x] Docker GPU 접근 테스트 (CDI 모드)

---

#### 3. LLM 모델 탐색 및 선정

**탐색 조건**:
- 상용 배포 가능 라이선스
- 한글 성능 우수
- Instruct 모델 제공
- 16GB VRAM에서 실행 가능
- 파인튜닝 지원

**후보 모델 비교**:

| 모델 | 라이선스 | 한글 | 컨텍스트 | VRAM (FP16) |
|------|----------|------|----------|-------------|
| Qwen2.5-7B-Instruct | Apache 2.0 | ⭐⭐⭐⭐ | 128K | ~14GB |
| SOLAR-10.7B-Instruct | Apache 2.0 | ⭐⭐⭐⭐⭐ | 4K | ~22GB |
| Bllossom-8B | Llama | ⭐⭐⭐⭐⭐ | 128K | ~16GB |

**1차 선정: Qwen2.5-7B-Instruct**

**선정 이유**:
1. **Apache 2.0 라이선스** - 완전한 상용 배포 가능
2. **128K 컨텍스트** - 긴 법률 문서 처리 가능 (SOLAR는 4K로 부적합)
3. **JSON 출력 우수** - RAG 응답 포맷팅에 유리
4. **멀티턴 대화** - 법률 상담 시나리오 적합 (SOLAR는 싱글턴만)
5. **16GB VRAM에서 FP16 실행** - 양자화 없이 풀 성능

**모델 테스트 순서**:
```
1. Qwen2.5-7B-Instruct  → 기본 RAG 파이프라인 구축
2. Qwen2.5-14B-Instruct → 성능 향상 테스트 (Q4)
3. SOLAR-10.7B-Instruct → 한국어 특화 비교
4. Bllossom-8B          → 한글 최적화 비교
```

**임베딩 모델 선정: BGE-M3**
- MIT 라이선스
- 1024 차원
- 한글 성능 우수

---

#### 4. 프로젝트 환경 설정

**Python 환경**:
```bash
uv venv --python 3.11
# Python 3.11.13 가상환경 생성
```

**디렉토리 구조**:
```
rag_llm_chaining/
├── .venv/                 # Python 가상환경
├── .env                   # 환경변수 (비공개)
├── .env.example           # 환경변수 템플릿
├── .gitignore
├── CLAUDE.md              # 프로젝트 문서
├── WORKLOG.md             # 작업 로그 (이 파일)
├── pyproject.toml         # 의존성 관리
├── config/
│   ├── model_config.yaml  # 모델 설정
│   └── vector_config.yaml # VectorDB 설정
├── data/
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리 데이터
│   └── embeddings/        # 임베딩 캐시
├── src/
│   ├── data/              # 데이터 로더
│   ├── embedding/         # 임베딩 모델
│   ├── llm/               # LLM 래퍼
│   ├── rag/               # RAG 파이프라인
│   └── api/               # FastAPI
├── tests/
├── notebooks/
└── scripts/
```

**핵심 의존성**:
- LLM: transformers, torch, vllm, accelerate
- Embedding: sentence-transformers
- VectorDB: chromadb, faiss-cpu
- RAG: langchain, langchain-community
- API: fastapi, uvicorn, gradio

---

#### 5. 의존성 설치 및 모델 테스트

**의존성 설치**:
```bash
uv pip install -e ".[dev]"
# 설치 완료: torch 2.9.0, vllm 0.12.0, transformers 4.57.3 등
```

**BGE-M3 임베딩 모델 테스트**:
```
✅ 모델 로드 성공
- 임베딩 차원: 1024
- 문장 유사도 테스트: 법률 문장 간 유사도 > 비관련 문장 유사도
```

**Qwen2.5-7B-Instruct LLM 테스트**:
```
✅ 모델 로드 성공
- VRAM 사용량: 12.30 GB (16GB 중)
- 다운로드 시간: ~20분 (첫 실행)
```

**테스트 질문**: "상속포기의 절차와 기한에 대해 간단히 설명해주세요."

**모델 답변**:
> 상속포기를 하는 경우, 일반적으로 다음과 같은 절차를 따릅니다:
> 1. 상속통지서 받기
> 2. 상속포기 의사표시
> 3. 상속포기서 법원 제출
>
> 상속포기는 보통 상속통지서를 받은 후 **6개월 이내**에 이루어져야 합니다.

---

#### 6. 다음 단계

- [ ] 데이터셋 탐색 (AI Hub 법률 데이터)
- [ ] VectorDB (ChromaDB) 설정
- [ ] RAG 파이프라인 기본 구현
- [ ] 테스트 질의 세트 구성

---

### 세션 2: RAG 파이프라인 구현 및 테스트

#### 1. VectorDB 구축

**데이터 처리**:
- HuggingFace 데이터셋: `joonhok-exo-ai/korean_law_open_data_precedents`
- 전체 데이터: 85,660건
- 테스트 샘플: 500건
- 청크 수: 3,878개 (문서당 평균 7.8개)
- 청크 설정: chunk_size=1000, overlap=200

**ChromaDB 설정**:
```
- 저장 경로: ./data/embeddings/chroma
- 컬렉션: korean_legal_docs
- 임베딩 모델: BAAI/bge-m3 (1024차원)
- 거리 메트릭: cosine
```

**빌드 시간**: ~60초 (500건 기준)

---

#### 2. RAG 파이프라인 테스트

**시스템 구성**:
- VectorStore + Qwen2.5-7B-Instruct
- VRAM 사용량: 12.68 GB (초기화 후)
- 검색 top_k: 3

**테스트 결과**:

| 질문 | 검색 점수 | 참조 판례 |
|------|----------|-----------|
| 귀속재산 임대차 임차인 권리 | 0.67 | 행정처분취소 외 |
| 법인 해산 청산 절차 | 0.62 | 근저당권설정등기말소 외 |
| 토지 소유권 이전 등기 | 0.61 | 소유권확인등 외 |

**RAG vs Non-RAG 비교**:
- RAG: 귀속재산처리법 제15조, 29조 언급하며 구체적 답변
- Non-RAG: "참고 자료가 없어 답변 불가" 응답

---

#### 3. 생성된 파일

| 파일 | 설명 |
|------|------|
| `src/data/loader.py` | 데이터셋 로드, 전처리, 청킹 |
| `src/embedding/vectordb.py` | ChromaDB 기반 벡터 저장소 |
| `src/rag/pipeline.py` | RAG 파이프라인 (검색+생성) |
| `scripts/build_vectordb.py` | VectorDB 빌드 스크립트 |
| `scripts/test_rag_pipeline.py` | RAG 파이프라인 테스트 |

---

#### 4. 다음 단계

- [ ] 전체 데이터셋 (85,660건) VectorDB 빌드
- [ ] FastAPI REST API 구현
- [ ] Gradio 데모 UI 구현
- [ ] 다른 LLM 모델 비교 테스트 (SOLAR, Bllossom)
- [ ] 성능 최적화 (vLLM 적용)

---

## 참고 자료

**LLM 모델**:
- [Qwen2.5 공식 블로그](https://qwenlm.github.io/blog/qwen2.5/)
- [SOLAR-10.7B HuggingFace](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)
- [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)

**임베딩 모델**:
- [BGE-M3 HuggingFace](https://huggingface.co/BAAI/bge-m3)

**데이터셋**:
- [AI Hub](https://aihub.or.kr/)
- [국가법령정보센터](https://www.law.go.kr/)
