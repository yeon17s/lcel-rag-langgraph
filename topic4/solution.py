"""
topic4/solution.py
요구사항 1 : K8s 상태값(CrashLoopBackOff, Pending 등) 원인·명령어 제공 챗봇 (ID 20, 21)
요구사항 2 : 502 에러 인프라 흐름 추적 단계별 점검 리스트 RAG (ID 10 → 7 → 8)
요구사항 3 : Pod YAML 보안 취약점 분석 에이전트 (ID 2, 14, 42, 43) - LangGraph
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated, TypedDict


# 공통: 데이터 로드

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")

FAISS_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    DataFrame 전체를 LangChain Document 리스트로 변환합니다.

    Args:
        df: K8s/Docker 규정 데이터프레임

    Returns:
        list[Document]: LangChain Document 목록
    """
    docs = []
    for _, row in df.iterrows():
        content = f"[규정 #{row['id']}] ({row['category']} > {row['sub_category']}) {row['title']}\n{row['content']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "id": int(row["id"]),
                "category": row["category"],
                "sub_category": row["sub_category"],
                "title": row["title"],
            },
        ))
    return docs


def get_rules_by_ids(df: pd.DataFrame, ids: list[int]) -> str:
    """
    특정 ID 목록에 해당하는 규정을 문자열로 반환합니다.

    Args:
        df: 규정 데이터프레임
        ids: 조회할 규정 ID 목록

    Returns:
        str: 규정 내용 문자열
    """
    rows = df[df["id"].isin(ids)].sort_values("id")
    lines = []
    for _, row in rows.iterrows():
        lines.append(f"[규정 #{row['id']}] {row['title']}\n{row['content']}")
    return "\n\n".join(lines)


# 공통: LLM / Embeddings 초기화

def get_llm(model: str = "llama-3.1-8b-instant") -> ChatGroq:
    return ChatGroq(model=model, temperature=0)


def get_embeddings():
    """
    로컬 HuggingFace 임베딩을 반환합니다.

    Returns:
        HuggingFaceEmbeddings: 임베딩 모델
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def build_vectorstore(df: pd.DataFrame):
    """
    전체 규정을 벡터 스토어에 인덱싱합니다. 로컬 캐시를 사용합니다.

    Args:
        df: 규정 데이터프레임

    Returns:
        FAISS: 벡터 스토어
    """
    embeddings = get_embeddings()
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = build_documents(df)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore


# 요구사항 1: K8s 상태값 원인·명령어 제공 챗봇 (ID 20, 21)

K8S_TROUBLESHOOT_IDS = [20, 21]


def build_k8s_diagnostic_chain():
    """
    CrashLoopBackOff, Pending 등 K8s 상태값 발생 시
    ID 20, 21에서 원인과 확인 명령어를 추출해 제공하는 RAG 챗봇 체인을 반환합니다.

    Returns:
        Chain: K8s 진단 챗봇 체인
    """
    df = load_dataset()
    vectorstore = build_vectorstore(df)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 Kubernetes 운영 전문가입니다. "
            "아래 검색된 규정을 참고하여 K8s 상태 이슈의 원인과 진단 방법을 안내하세요.\n\n"
            "## 답변 형식\n"
            "### 상태 설명\n"
            "(해당 K8s 상태의 의미)\n\n"
            "### 주요 원인\n"
            "(발생 원인 목록)\n\n"
            "### 확인 명령어\n"
            "```bash\n(진단에 사용할 kubectl 명령어)\n```\n\n"
            "### 해결 방향\n"
            "(점검해야 할 항목)\n\n"
            "## 참고 규정\n{context}"
        )),
        ("human", "{question}"),
    ])

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 요구사항 2: 502 에러 인프라 흐름 추적 단계별 점검 리스트 RAG (ID 10 → 7 → 8)

INFRA_FLOW_IDS = [10, 7, 8]


def build_502_checklist_chain():
    """
    외부 유저 502 에러 발생 시 ID 10(Ingress) → ID 7(Readiness) → ID 8(Service)
    인프라 흐름을 추적하여 단계별 점검 리스트를 생성하는 RAG 체인을 반환합니다.

    Returns:
        Chain: 502 에러 점검 체인
    """
    df = load_dataset()
    # 흐름 순서에 맞게 ID 10 → 7 → 8 순으로 정렬
    rows = df[df["id"].isin(INFRA_FLOW_IDS)]
    ordered = pd.concat([rows[rows["id"] == i] for i in INFRA_FLOW_IDS])
    rules_text = "\n\n".join(
        f"[규정 #{row['id']}] {row['title']}\n{row['content']}"
        for _, row in ordered.iterrows()
    )

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 Kubernetes 인프라 전문가입니다. "
            "외부 유저가 502 에러를 경험할 때, 아래 규정을 기반으로 "
            "Ingress → Readiness → Service 순서로 인프라 흐름을 추적하며 "
            "단계별 점검 리스트를 작성하세요.\n\n"
            "## 점검 리스트 형식\n"
            "### 1단계: Ingress 점검 (규정 #10)\n"
            "- [ ] (점검 항목)\n\n"
            "### 2단계: Readiness Probe 점검 (규정 #7)\n"
            "- [ ] (점검 항목)\n\n"
            "### 3단계: Service 점검 (규정 #8)\n"
            "- [ ] (점검 항목)\n\n"
            "각 단계마다 확인 명령어도 함께 제시하세요.\n\n"
            "## 참고 규정\n{rules}"
        )),
        ("human", "{scenario}"),
    ])

    chain = (
        {"rules": RunnableLambda(lambda _: rules_text), "scenario": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 요구사항 3: Pod YAML 보안 취약점 분석 에이전트 (LangGraph)

SECURITY_RULE_IDS = [2, 14, 42, 43]


class SecurityAuditState(TypedDict):
    messages: Annotated[list, add_messages]
    pod_yaml: str
    violations: list[str]
    final_report: str


def build_security_audit_agent():
    """
    Pod YAML 설정을 분석하여 ID 2(비루트), ID 14(NetPol), ID 42(SecurityContext),
    ID 43(Docker 소켓) 규정 위반 보안 취약점을 탐지하고 수정 권고안을 작성하는
    LangGraph 에이전트를 반환합니다.

    Returns:
        CompiledGraph: 컴파일된 보안 감사 에이전트
    """
    df = load_dataset()
    llm = get_llm()

    def check_rule(state: SecurityAuditState, rule_id: int, check_desc: str) -> tuple[list[str], str]:
        """단일 보안 규정 위반 여부를 LLM으로 판단합니다."""
        rule = get_rules_by_ids(df, [rule_id])
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                f"다음 보안 규정을 기준으로 Pod YAML에서 위반 사항을 탐지하세요.\n\n"
                f"{rule}\n\n"
                "위반이 있으면 'VIOLATION: (구체적 위반 내용과 위치)' 형식으로, "
                "위반이 없으면 'PASS'로만 답변하세요."
            )),
            ("human", "분석할 Pod YAML:\n```yaml\n{pod_yaml}\n```"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"pod_yaml": state["pod_yaml"]})

        violations = list(state["violations"])
        if result.startswith("VIOLATION"):
            violations.append(f"[규정 #{rule_id}] {result.replace('VIOLATION: ', '')}")

        return violations, result

    # 노드 1: 비루트 사용자 실행 검증 (규정 #2)
    def check_non_root(state: SecurityAuditState) -> SecurityAuditState:
        violations, result = check_rule(state, 2, "비루트 사용자 실행")
        return {
            **state,
            "violations": violations,
            "messages": state["messages"] + [AIMessage(content=f"비루트 검증 (규정 #2): {result}")],
        }

    # 노드 2: Network Policy 검증 (규정 #14)
    def check_network_policy(state: SecurityAuditState) -> SecurityAuditState:
        violations, result = check_rule(state, 14, "Network Policy 설정")
        return {
            **state,
            "violations": violations,
            "messages": state["messages"] + [AIMessage(content=f"Network Policy 검증 (규정 #14): {result}")],
        }

    # 노드 3: SecurityContext 검증 (규정 #42)
    def check_security_context(state: SecurityAuditState) -> SecurityAuditState:
        violations, result = check_rule(state, 42, "SecurityContext 권한 설정")
        return {
            **state,
            "violations": violations,
            "messages": state["messages"] + [AIMessage(content=f"SecurityContext 검증 (규정 #42): {result}")],
        }

    # 노드 4: Docker 소켓 마운트 검증 (규정 #43)
    def check_docker_socket(state: SecurityAuditState) -> SecurityAuditState:
        violations, result = check_rule(state, 43, "Docker 소켓 노출")
        return {
            **state,
            "violations": violations,
            "messages": state["messages"] + [AIMessage(content=f"Docker 소켓 검증 (규정 #43): {result}")],
        }

    # 노드 5: 보안 감사 보고서 생성
    def make_report(state: SecurityAuditState) -> SecurityAuditState:
        all_rules = get_rules_by_ids(df, SECURITY_RULE_IDS)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "당신은 Kubernetes 보안 전문가입니다. "
                "아래 보안 규정과 탐지된 위반 사항을 바탕으로 최종 보안 감사 보고서를 작성하세요.\n\n"
                "## 보고서 형식\n"
                "### 보안 감사 결과 요약\n"
                "(총 위반 건수 및 심각도)\n\n"
                "### 위반 사항 상세\n"
                "(각 위반 항목, 근거 규정 번호, 위험도)\n\n"
                "### 수정 권고안\n"
                "(각 위반에 대한 YAML 수정 예시 포함)\n\n"
                "## 적용 보안 규정\n{rules}"
            )),
            ("human", (
                "분석 대상 YAML:\n```yaml\n{pod_yaml}\n```\n\n"
                "탐지된 위반 사항:\n{violations}"
            )),
        ])
        chain = prompt | llm | StrOutputParser()
        violations_text = "\n".join(state["violations"]) if state["violations"] else "위반 사항 없음"
        report = chain.invoke({
            "rules": all_rules,
            "pod_yaml": state["pod_yaml"],
            "violations": violations_text,
        })

        return {
            **state,
            "final_report": report,
            "messages": state["messages"] + [AIMessage(content=report)],
        }

    # 그래프 구성
    graph = StateGraph(SecurityAuditState)
    graph.add_node("check_non_root", check_non_root)
    graph.add_node("check_network_policy", check_network_policy)
    graph.add_node("check_security_context", check_security_context)
    graph.add_node("check_docker_socket", check_docker_socket)
    graph.add_node("make_report", make_report)

    graph.set_entry_point("check_non_root")
    graph.add_edge("check_non_root", "check_network_policy")
    graph.add_edge("check_network_policy", "check_security_context")
    graph.add_edge("check_security_context", "check_docker_socket")
    graph.add_edge("check_docker_socket", "make_report")
    graph.add_edge("make_report", END)

    return graph.compile()


if __name__ == "__main__":
    print("=" * 70)
    print("▶ 요구사항 1: K8s 상태값 진단 챗봇")
    print("=" * 70)
    diagnostic_chain = build_k8s_diagnostic_chain()

    question_crash = "Pod가 CrashLoopBackOff 상태입니다. 원인과 확인 명령어를 알려주세요."
    print(f"Q: {question_crash}")
    print(diagnostic_chain.invoke(question_crash))

    print()
    question_pending = "Pod가 Pending 상태에서 벗어나지 못하고 있습니다. 어떻게 진단해야 하나요?"
    print(f"Q: {question_pending}")
    print(diagnostic_chain.invoke(question_pending))

    print("\n" + "=" * 70)
    print("▶ 요구사항 2: 502 에러 인프라 흐름 추적 점검 리스트")
    print("=" * 70)
    checklist_chain = build_502_checklist_chain()
    scenario = "외부 유저가 서비스에 접속했는데 502 Bad Gateway 에러가 발생합니다. 단계별로 점검해야 할 항목을 알려주세요."
    print(f"Q: {scenario}")
    print(checklist_chain.invoke(scenario))

    print("\n" + "=" * 70)
    print("▶ 요구사항 3: Pod YAML 보안 취약점 분석 에이전트")
    print("=" * 70)
    audit_agent = build_security_audit_agent()

    sample_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: vulnerable-pod
  namespace: default
spec:
  containers:
  - name: app
    image: myapp:latest
    securityContext:
      privileged: true
      runAsUser: 0
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
"""

    initial_state: SecurityAuditState = {
        "messages": [HumanMessage(content="이 Pod YAML의 보안 취약점을 분석해주세요.")],
        "pod_yaml": sample_yaml,
        "violations": [],
        "final_report": "",
    }
    final_state = audit_agent.invoke(initial_state)
    print(final_state["final_report"])
