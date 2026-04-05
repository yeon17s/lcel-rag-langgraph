### LCEL · RAG · LangGraph 실습 프로젝트

데이터셋 기반 RAG 체인과 LangGraph 에이전트를 주제별로 구현하는 실습 프로젝트입니다.

---

### 프로젝트 구조

```
lcel-rag-langgraph/
├── topic1/                  # GitHub 이슈 트래커 분석
│   ├── dataset.csv          # 이슈·댓글 데이터셋
│   ├── q1.md                # 요구사항
│   └── solution.py          # 구현 코드
├── topic2/                  # KBO 리그 규정 QA
│   ├── dataset.csv          # KBO 규정 데이터셋
│   ├── q2.md                # 요구사항
│   └── solution.py          # 구현 코드
├── topic3/                  # 뱅킹 에이전트 규제 검증
│   ├── dataset.csv          # 금융 규제 데이터셋
│   ├── q3.md                # 요구사항
│   └── solution.py          # 구현 코드
├── topic4/                  # Kubernetes 보안 감사
│   ├── dataset.csv          # K8s·Docker 규정 데이터셋
│   ├── q4.md                # 요구사항
│   └── solution.py          # 구현 코드
├── .env                     # API 키 설정
└── requirements.txt
```

---

### 주제별 구현 내용

| 주제 | 요구사항 1 | 요구사항 2 | 요구사항 3 |
|------|-----------|-----------|-----------|
| **topic1** GitHub 이슈 | Tool Choice 에러 요약 체인 (LCEL) | PostgresSaver 도입 체크리스트 RAG | 코드 버그 탐지 에이전트 (LangGraph) |
| **topic2** KBO 규정 | 규정 번호 인용 QA 챗봇 | 부상자 명단 복귀 날짜 계산 체인 | 경기 최종 상태 판정 논리 체인 |
| **topic3** 뱅킹 규제 | 이체 가능 여부 및 MFA 안내 RAG | FDS 차단 근거 분석기 | 해외송금·대출 순차 검증 에이전트 (LangGraph) |
| **topic4** K8s 보안 | K8s 상태값 진단 챗봇 RAG | 502 에러 인프라 흐름 점검 리스트 | Pod YAML 보안 취약점 분석 에이전트 (LangGraph) |

---

### 공통 기술 스택

| 항목 | 내용 |
|------|------|
| LLM | Groq (`llama-3.1-8b-instant`) |
| 임베딩 | HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`) |
| 벡터 스토어 | FAISS (로컬 캐시) |
| 체인 | LangChain LCEL |
| 에이전트 | LangGraph StateGraph |

---

### 실행 방법

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
pip install faiss-cpu langchain-huggingface sentence-transformers

# 각 주제 실행
python topic1/solution.py
python topic2/solution.py
python topic3/solution.py
python topic4/solution.py
```

실행 시 아래 순서로 출력됩니다.

```
======================================================================
▶ 요구사항 1: ...
======================================================================
(결과 출력)

======================================================================
▶ 요구사항 2: ...
======================================================================
(결과 출력)

======================================================================
▶ 요구사항 3: ...
======================================================================
(결과 출력)
```

첫 실행 시 HuggingFace 임베딩 모델(약 400MB)을 자동 다운로드하고, FAISS 인덱스를 각 topic 폴더에 저장합니다. 이후 실행부터는 캐시를 재사용합니다.

---

### 환경 변수

`.env` 파일에 Groq API 키를 설정합니다.

```
GROQ_API_KEY=your_groq_api_key_here
```

Groq API 키는 [console.groq.com](https://console.groq.com) 에서 발급받을 수 있습니다.
