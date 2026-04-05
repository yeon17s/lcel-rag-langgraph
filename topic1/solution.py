"""
topic1/solution.py
요구사항 1 : Tool Choice 에러 이슈/댓글 요약 체인 (LCEL)
요구사항 2 : PostgresSaver + DB 설정 이슈 통합 체크리스트 RAG
요구사항 3 : 코드 버그 탐지 에이전트 (LangGraph)
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

def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    DataFrame 전체를 LangChain Document 리스트로 변환합니다.

    Args:
        df: 이슈/댓글 데이터프레임

    Returns:
        list[Document]: LangChain Document 목록
    """
    docs = []
    for _, row in df.iterrows():
        content = f"[{row['type'].upper()} #{row['id']}] {row['title']}\n{row['content']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "id": int(row["id"]),
                "type": row["type"],
                "title": row["title"],
                "author": row["author"],
                "created_at": row["created_at"],
            },
        ))
    return docs


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


# 요구사항 1: Tool Choice 에러 요약 체인

TOOL_CHOICE_KEYWORDS = ["tool choice", "tool call", "invalid tool", "bind_tools"]


def filter_tool_choice_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Tool Choice' 관련 이슈와 해당 이슈에 달린 댓글을 함께 반환합니다.

    Args:
        df: 전체 데이터프레임

    Returns:
        pd.DataFrame: 필터링된 이슈 + 댓글
    """
    # 키워드를 포함한 이슈 찾기
    mask = df.apply(
        lambda r: any(kw in (r["title"] + " " + r["content"]).lower()
                    for kw in TOOL_CHOICE_KEYWORDS),
        axis=1,
    )
    issue_rows = df[mask]

    # 해당 이슈 제목을 포함하는 댓글 찾기
    issue_titles = issue_rows["title"].tolist()
    comment_mask = df["type"] == "comment"
    related_comments = df[
        comment_mask & df["title"].apply(
            lambda t: any(it in t for it in issue_titles)
        )
    ]

    return pd.concat([issue_rows, related_comments]).sort_values("id").reset_index(drop=True)


def build_tool_choice_chain():
    """
    Tool Choice 에러 이슈·댓글을 분석하여
    원인과 최종 해결책을 한 장의 리포트로 요약하는 LCEL 체인을 반환합니다.

    Returns:
        Chain: LCEL 체인
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 기술 이슈 분석 전문가입니다. "
            "아래 이슈와 댓글 데이터를 바탕으로 다음 형식의 리포트를 작성하세요.\n\n"
            "# Tool Choice 에러 분석 리포트\n"
            "## 1. 에러 개요\n"
            "## 2. 근본 원인\n"
            "## 3. 최종 해결책\n"
            "## 4. 예방 방법\n"
        )),
        ("human", "분석할 이슈/댓글 데이터:\n\n{context}"),
    ])

    def prepare_context(input_dict: dict) -> dict:
        df = load_dataset()
        rows = filter_tool_choice_rows(df)
        lines = []
        for _, r in rows.iterrows():
            lines.append(f"[{r['type'].upper()} #{r['id']}] {r['title']}\n  {r['content']}\n  (작성자: {r['author']}, 날짜: {r['created_at']})")
        return {"context": "\n\n".join(lines)}

    chain = (
        RunnablePassthrough()
        | prepare_context
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 요구사항 2: PostgresSaver + 연관 DB 이슈 → 통합 체크리스트 RAG

DB_RELATED_IDS = {1, 2, 3, 10, 11}  # PostgresSaver + MySQL Isolation Level

FAISS_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_db")
FAISS_ALL_PATH = os.path.join(os.path.dirname(__file__), "faiss_all")


def build_db_vectorstore(df: pd.DataFrame):
    """
    DB 관련 이슈와 댓글만 벡터 스토어에 인덱싱합니다. 로컬 캐시를 사용합니다.

    Args:
        df: 전체 데이터프레임

    Returns:
        FAISS: 벡터 스토어
    """
    embeddings = get_embeddings()
    if os.path.exists(FAISS_DB_PATH):
        return FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    db_rows = df[df["id"].isin(DB_RELATED_IDS)]
    docs = build_documents(db_rows)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_DB_PATH)
    return vectorstore


def build_checklist_rag():
    """
    PostgresSaver(ID 1)와 연관 DB 설정 이슈(ID 10, 11)를 검색하여
    기술 도입 시 주의해야 할 통합 체크리스트를 생성하는 RAG 체인을 반환합니다.

    Returns:
        Chain: RAG 체인
    """
    df = load_dataset()
    vectorstore = build_db_vectorstore(df)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 데이터베이스 아키텍처 전문가입니다. "
            "아래 검색된 이슈 및 댓글을 참고하여 "
            "PostgresSaver / SQL 기반 체크포인터 도입 시 반드시 확인해야 할 "
            "통합 체크리스트를 마크다운 체크박스(- [ ]) 형식으로 작성하세요. "
            "카테고리(연결 관리, 트랜잭션 설정, 운영 주의사항 등)로 분류하세요.\n\n"
            "참고 이슈/댓글:\n{context}"
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


# 요구사항 3: 코드 버그 탐지 에이전트 (LangGraph)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_code: str
    similar_issues: str
    bug_report: str


def build_bug_agent():
    """
    사용자 코드를 받아 유사 이슈를 검색하고
    잠재적 버그와 수정 코드 스니펫을 제안하는 LangGraph 에이전트를 반환합니다.

    Returns:
        CompiledGraph: 컴파일된 LangGraph 에이전트
    """
    df = load_dataset()
    embeddings = get_embeddings()
    if os.path.exists(FAISS_ALL_PATH):
        vectorstore = FAISS.load_local(FAISS_ALL_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        all_docs = build_documents(df)
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local(FAISS_ALL_PATH)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = get_llm()

    # 노드 1: 유사 이슈 검색
    def retrieve_similar_issues(state: AgentState) -> AgentState:
        query = f"코드 문제: {state['user_code'][:500]}"
        docs = retriever.invoke(query)
        issues_text = "\n\n".join(
            f"[{d.metadata['type'].upper()} #{d.metadata['id']}] {d.page_content}"
            for d in docs
        )
        return {
            **state,
            "similar_issues": issues_text,
            "messages": state["messages"] + [
                AIMessage(content=f"유사 이슈 {len(docs)}건을 찾았습니다. 분석을 시작합니다.")
            ],
        }

    # 노드 2: 버그 분석 및 수정 코드 생성
    def analyze_and_fix(state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "당신은 시니어 소프트웨어 엔지니어입니다. "
                "아래 데이터베이스 이슈 사례들을 참고하여 "
                "사용자 코드의 잠재적 버그를 발견하고 수정안을 제시하세요.\n\n"
                "## 참고 이슈 사례\n{similar_issues}\n\n"
                "## 분석 형식\n"
                "### 발견된 잠재적 버그\n"
                "(각 버그를 번호로 나열, 관련 이슈 ID 언급)\n\n"
                "### 수정된 코드\n"
                "```python\n(수정된 전체 코드)\n```\n\n"
                "### 수정 설명\n"
                "(각 수정 사항 설명)"
            )),
            ("human", "분석할 코드:\n```python\n{user_code}\n```"),
        ])

        chain = prompt | llm | StrOutputParser()
        bug_report = chain.invoke({
            "similar_issues": state["similar_issues"],
            "user_code": state["user_code"],
        })

        return {
            **state,
            "bug_report": bug_report,
            "messages": state["messages"] + [AIMessage(content=bug_report)],
        }

    # 그래프 구성
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_similar_issues)
    graph.add_node("analyze", analyze_and_fix)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", END)

    return graph.compile()


if __name__ == "__main__":
    print("=" * 70)
    print("▶ 요구사항 1: Tool Choice 에러 분석 리포트")
    print("=" * 70)
    chain1 = build_tool_choice_chain()
    report1 = chain1.invoke({})
    print(report1)

    print("\n" + "=" * 70)
    print("▶ 요구사항 2: PostgresSaver 도입 통합 체크리스트 (RAG)")
    print("=" * 70)
    rag_chain = build_checklist_rag()
    query = "PostgresSaver를 프로덕션에 도입할 때 반드시 확인해야 할 항목은?"
    checklist = rag_chain.invoke(query)
    print(checklist)

    print("\n" + "=" * 70)
    print("▶ 요구사항 3: 코드 버그 탐지 에이전트")
    print("=" * 70)
    agent = build_bug_agent()

    sample_code = """
import psycopg2
from langgraph.checkpoint.postgres import PostgresSaver

def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="mydb",
        user="postgres",
        password="secret"
    )
    return conn

def save_checkpoint(data):
    conn = get_db_connection()
    saver = PostgresSaver(conn)
    saver.put({"state": data})
    # conn.close() 누락
    return "saved"

def process_loop(items):
    for item in items:
        result = save_checkpoint(item)  # 루프마다 커넥션 생성
        print(result)
"""

    initial_state: AgentState = {
        "messages": [HumanMessage(content="이 코드의 버그를 찾아주세요.")],
        "user_code": sample_code,
        "similar_issues": "",
        "bug_report": "",
    }

    final_state = agent.invoke(initial_state)
    print(final_state["bug_report"])
