"""
topic3/solution.py
요구사항 1 : 등급·금액 기반 이체 가능 여부 및 추가 인증 안내 RAG 체인
요구사항 2 : FDS 차단 근거 및 에이전트 다음 노드 설명 뱅킹 히스토리 분석기
요구사항 3 : 해외송금·대출 순차 검증 에이전트 (ID 26, 30, 39) - LangGraph
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
        df: 뱅킹 규제 데이터프레임

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


# 요구사항 1: 이체 가능 여부 및 추가 인증 안내 RAG 체인

def build_transfer_check_chain():
    """
    사용자 등급(VIP/일반)과 요청 금액을 입력받아
    이체 가능 여부와 필요한 추가 인증(MFA 등)을 안내하는 RAG 체인을 반환합니다.

    Returns:
        Chain: 이체 검증 RAG 체인
    """
    df = load_dataset()
    vectorstore = build_vectorstore(df)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 Neo-Finance 뱅킹 에이전트입니다. "
            "아래 규정을 참고하여 사용자의 이체 요청에 대해 답변하세요.\n\n"
            "## 답변 형식\n"
            "1. 이체 가능 여부 (가능/불가)\n"
            "2. 적용 한도 근거 (규정 번호 인용)\n"
            "3. 추가 인증 필요 여부 및 절차 (해당 시)\n"
            "4. 최종 안내 메시지\n\n"
            "관련 규정:\n{context}"
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


# 요구사항 2: FDS 차단 근거 및 에이전트 다음 노드 설명 - 뱅킹 히스토리 분석기

FDS_RULE_IDS = [9, 10]


def build_fds_analyzer_chain():
    """
    FDS(이상거래탐지)에 의해 거래가 차단되었을 때,
    차단 근거(ID 9)와 에이전트의 다음 처리 노드(ID 10)를 설명하는 분석기 체인을 반환합니다.

    Returns:
        Chain: FDS 분석기 체인
    """
    df = load_dataset()
    rules_text = get_rules_by_ids(df, FDS_RULE_IDS)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 Neo-Finance 뱅킹 에이전트 규정 분석기입니다. "
            "아래 규정을 바탕으로 거래 차단 상황을 분석하세요.\n\n"
            "## 분석 형식\n"
            "### 1. 차단 근거 (규정 #9)\n"
            "(FDS가 이 거래를 차단한 조건 명시)\n\n"
            "### 2. 차단 판단 적합성\n"
            "(제시된 상황이 규정 #9의 차단 조건에 해당하는지 여부)\n\n"
            "### 3. 에이전트 다음 처리 노드 (규정 #10)\n"
            "(FDS 차단 후 에이전트가 수행하게 될 다음 단계 설명)\n\n"
            "## 적용 규정\n{rules}"
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


# 요구사항 3: 해외송금·대출 순차 검증 에이전트 (LangGraph)

VALIDATION_RULE_IDS = [26, 30, 39]


class ValidationState(TypedDict):
    messages: Annotated[list, add_messages]
    request_type: str
    user_info: dict
    violations: list[str]
    final_decision: str


def build_validation_agent():
    """
    ID 26(해외한도), ID 30(DSR), ID 39(투자적합성)을 순차적으로 검증하고
    위반 항목 발생 시 법적 근거와 함께 거절 사유를 제시하는 LangGraph 에이전트를 반환합니다.

    Returns:
        CompiledGraph: 컴파일된 검증 에이전트
    """
    df = load_dataset()
    rules_text = get_rules_by_ids(df, VALIDATION_RULE_IDS)
    llm = get_llm()

    # 노드 1: 해외 송금 한도 검증 (규정 #26)
    def validate_overseas(state: ValidationState) -> ValidationState:
        rule = get_rules_by_ids(df, [26])
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "다음 규정을 기준으로 해외 송금 요청이 한도를 위반하는지 판단하세요.\n\n"
                "{rule}\n\n"
                "위반이면 'VIOLATION: (사유)' 형식으로, 통과면 'PASS'로만 답변하세요."
            )),
            ("human", "요청 정보: {user_info}"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"rule": rule, "user_info": str(state["user_info"])})

        violations = list(state["violations"])
        if result.startswith("VIOLATION"):
            violations.append(f"[규정 #26 외국환거래법] {result.replace('VIOLATION: ', '')}")

        return {
            **state,
            "violations": violations,
            "messages": state["messages"] + [
                AIMessage(content=f"해외 송금 한도 검증 완료: {result}")
            ],
        }

    # 노드 2: DSR 검증 (규정 #30)
    def validate_dsr(state: ValidationState) -> ValidationState:
        rule = get_rules_by_ids(df, [30])
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "다음 규정을 기준으로 대출 요청이 DSR 기준을 위반하는지 판단하세요.\n\n"
                "{rule}\n\n"
                "위반이면 'VIOLATION: (사유)' 형식으로, 통과면 'PASS'로만 답변하세요."
            )),
            ("human", "요청 정보: {user_info}"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"rule": rule, "user_info": str(state["user_info"])})

        violations = list(state["violations"])
        if result.startswith("VIOLATION"):
            violations.append(f"[규정 #30 DSR 기준] {result.replace('VIOLATION: ', '')}")

        return {
            **state,
            "violations": violations,
            "messages": state["messages"] + [
                AIMessage(content=f"DSR 검증 완료: {result}")
            ],
        }

    # 노드 3: 투자 적합성 검증 (규정 #39)
    def validate_investment(state: ValidationState) -> ValidationState:
        rule = get_rules_by_ids(df, [39])
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "다음 규정을 기준으로 투자 요청이 적합성 원칙을 위반하는지 판단하세요.\n\n"
                "{rule}\n\n"
                "위반이면 'VIOLATION: (사유)' 형식으로, 통과면 'PASS'로만 답변하세요."
            )),
            ("human", "요청 정보: {user_info}"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"rule": rule, "user_info": str(state["user_info"])})

        violations = list(state["violations"])
        if result.startswith("VIOLATION"):
            violations.append(f"[규정 #39 적합성 원칙] {result.replace('VIOLATION: ', '')}")

        return {
            **state,
            "violations": violations,
            "messages": state["messages"] + [
                AIMessage(content=f"투자 적합성 검증 완료: {result}")
            ],
        }

    # 노드 4: 최종 판정
    def make_decision(state: ValidationState) -> ValidationState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "당신은 Neo-Finance 뱅킹 에이전트입니다. "
                "아래 검증 결과를 바탕으로 요청에 대한 최종 판정을 내리세요.\n\n"
                "## 적용된 전체 규정\n{rules}\n\n"
                "위반 항목이 있으면 각 항목의 법적 근거를 명시하며 거절하고, "
                "모든 항목을 통과하면 승인 안내를 작성하세요."
            )),
            ("human", (
                "요청 정보: {user_info}\n\n"
                "검증 결과 위반 사항:\n{violations}"
            )),
        ])
        chain = prompt | llm | StrOutputParser()
        violations_text = "\n".join(state["violations"]) if state["violations"] else "없음 (전 항목 통과)"
        decision = chain.invoke({
            "rules": rules_text,
            "user_info": str(state["user_info"]),
            "violations": violations_text,
        })

        return {
            **state,
            "final_decision": decision,
            "messages": state["messages"] + [AIMessage(content=decision)],
        }

    # 그래프 구성
    graph = StateGraph(ValidationState)
    graph.add_node("validate_overseas", validate_overseas)
    graph.add_node("validate_dsr", validate_dsr)
    graph.add_node("validate_investment", validate_investment)
    graph.add_node("make_decision", make_decision)

    graph.set_entry_point("validate_overseas")
    graph.add_edge("validate_overseas", "validate_dsr")
    graph.add_edge("validate_dsr", "validate_investment")
    graph.add_edge("validate_investment", "make_decision")
    graph.add_edge("make_decision", END)

    return graph.compile()


if __name__ == "__main__":
    print("=" * 70)
    print("▶ 요구사항 1: 이체 가능 여부 및 추가 인증 안내")
    print("=" * 70)
    transfer_chain = build_transfer_check_chain()

    question_vip = "저는 VIP 등급 사용자입니다. 오늘 1,500만 원을 이체하려고 합니다. 가능한가요?"
    print(f"Q: {question_vip}")
    print(transfer_chain.invoke(question_vip))

    print()
    question_general = "일반 등급 사용자인데 300만 원 이체가 가능한가요?"
    print(f"Q: {question_general}")
    print(transfer_chain.invoke(question_general))

    print("\n" + "=" * 70)
    print("▶ 요구사항 2: FDS 차단 분석기")
    print("=" * 70)
    fds_chain = build_fds_analyzer_chain()
    fds_scenario = (
        "고객 계좌에서 최근 1시간 내 소액 결제(1,000원~5,000원)가 7회 연속으로 발생했고, "
        "동시에 미국 IP에서 접근 시도가 있었습니다. "
        "해당 거래가 FDS에 의해 차단되었습니다. "
        "차단 근거와 이후 에이전트의 처리 절차를 분석해주세요."
    )
    print(f"Q: {fds_scenario}")
    print(fds_chain.invoke(fds_scenario))

    print("\n" + "=" * 70)
    print("▶ 요구사항 3: 해외송금·대출 순차 검증 에이전트")
    print("=" * 70)
    agent = build_validation_agent()
    user_info = {
        "request_type": "해외송금 및 대출",
        "overseas_amount_usd": 60000,
        "annual_income_krw": 50000000,
        "total_loan_repayment_krw": 25000000,
        "investment_profile": "안정형",
        "product_requested": "해외 파생상품 펀드",
    }
    initial_state: ValidationState = {
        "messages": [HumanMessage(content="해외 송금 6만 달러와 파생상품 펀드 가입을 요청합니다.")],
        "request_type": "해외송금+대출+투자",
        "user_info": user_info,
        "violations": [],
        "final_decision": "",
    }
    final_state = agent.invoke(initial_state)
    print(final_state["final_decision"])
