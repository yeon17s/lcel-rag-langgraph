"""
topic2/solution.py
요구사항 1 : 규정 번호·내용 인용 RAG 챗봇
요구사항 2 : 부상자 명단 복귀 날짜 계산 체인 (ID 8, 9, 10)
요구사항 3 : ABS 고장 + 강우 중단 경기 최종 상태 판정 체인 (ID 2, 23, 24)
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


# 공통: 데이터 로드

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")

FAISS_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    DataFrame 전체를 LangChain Document 리스트로 변환합니다.

    Args:
        df: 규정 데이터프레임

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


# 요구사항 1: 규정 번호·내용 인용 RAG 챗봇

def build_rule_qa_chain():
    """
    질문에 대해 관련 규정 번호와 내용을 인용하여 답변하는 RAG 챗봇 체인을 반환합니다.

    Returns:
        Chain: RAG QA 체인
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
            "당신은 KBO 리그 규정 전문가입니다. "
            "아래 검색된 규정을 바탕으로 질문에 답변하세요. "
            "반드시 관련 규정 번호(예: 규정 #4)를 명시하고 해당 내용을 직접 인용하여 근거를 제시하세요. "
            "규정에 없는 내용은 답변하지 마세요.\n\n"
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


# 요구사항 2: 부상자 명단 복귀 날짜 계산 체인 (ID 8, 9, 10)

IL_RULE_IDS = [8, 9, 10]


def build_il_return_chain():
    """
    규정 ID 8, 9, 10을 논리적으로 조합하여
    소급 적용을 포함한 부상자 명단 복귀 가능일을 계산하는 체인을 반환합니다.

    Returns:
        Chain: 복귀일 계산 체인
    """
    df = load_dataset()
    rules_text = get_rules_by_ids(df, IL_RULE_IDS)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 KBO 리그 규정 전문가입니다. "
            "아래 규정을 단계적으로 적용하여 선수의 1군 복귀 가능일을 계산하세요.\n\n"
            "## 적용 규정\n{rules}\n\n"
            "## 계산 절차\n"
            "1. 소급 적용 가능 여부 판단 (규정 #9: 최대 3일 전, 단 해당 기간 경기 출전 없어야 함)\n"
            "2. 실제 부상자 명단 시작일 산정\n"
            "3. 최소 의무 기간(규정 #10: 10일) 적용\n"
            "4. 복귀 가능일 도출\n\n"
            "각 단계를 명시하며 날짜를 계산하세요."
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


# 요구사항 3: ABS 고장 + 강우 중단 경기 최종 상태 판정 체인 (ID 2, 23, 24)

GAME_STATUS_RULE_IDS = [2, 23, 24]


def build_game_status_chain():
    """
    규정 ID 2(ABS 실패), ID 23(서스펜디드), ID 24(강우콜드)를 모두 참조하여
    경기의 최종 상태(종료/일시중단)를 판정하는 논리 체인을 반환합니다.

    Returns:
        Chain: 경기 상태 판정 체인
    """
    df = load_dataset()
    rules_text = get_rules_by_ids(df, GAME_STATUS_RULE_IDS)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 KBO 리그 심판 규정 전문가입니다. "
            "아래 규정을 순서대로 검토하여 경기의 최종 상태를 판정하세요.\n\n"
            "## 적용 규정\n{rules}\n\n"
            "## 판정 절차\n"
            "1. ABS 고장 영향 검토 (규정 #2)\n"
            "2. 강우 콜드게임 성립 여부 판단 (규정 #24)\n"
            "3. 서스펜디드 게임 해당 여부 판단 (규정 #23)\n"
            "4. 최종 경기 상태 판정 및 근거 제시\n\n"
            "각 규정을 번호와 함께 명시하며 논리적으로 판정하세요."
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


if __name__ == "__main__":
    print("=" * 70)
    print("▶ 요구사항 1: 규정 번호·내용 인용 RAG 챗봇")
    print("=" * 70)
    qa_chain = build_rule_qa_chain()
    question = "피치클락 위반 시 타자와 투수에게 각각 어떤 페널티가 부여되나요?"
    print(f"Q: {question}")
    print(qa_chain.invoke(question))

    print("\n" + "=" * 70)
    print("▶ 요구사항 2: 부상자 명단 복귀 날짜 계산")
    print("=" * 70)
    il_chain = build_il_return_chain()
    scenario = (
        "오늘은 4월 5일입니다. "
        "A선수는 4월 2일(3일 전)에 마지막으로 경기에 출전했고, "
        "오늘 부상으로 엔트리에서 말소되어 10일 부상자 명단에 등록되었습니다. "
        "소급 적용을 포함하면 A선수가 1군에 복귀할 수 있는 가장 빠른 날짜는 언제인가요?"
    )
    print(f"Q: {scenario}")
    print(il_chain.invoke(scenario))

    print("\n" + "=" * 70)
    print("▶ 요구사항 3: 경기 최종 상태 판정")
    print("=" * 70)
    game_chain = build_game_status_chain()
    game_scenario = (
        "경기 중 5회초 도중 ABS 시스템이 기술적 결함으로 작동을 멈췄습니다. "
        "이후 같은 이닝 중 비가 내리기 시작하여 경기를 지속할 수 없는 상황이 되었습니다. "
        "현재 점수는 원정팀 2점, 홈팀 3점입니다. "
        "이 경기의 최종 상태를 판정하고 그 근거를 설명하세요."
    )
    print(f"Q: {game_scenario}")
    print(game_chain.invoke(game_scenario))
