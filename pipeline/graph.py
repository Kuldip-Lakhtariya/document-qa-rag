from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict, Tuple

from pipeline.question_classifier import is_broad_question
from pipeline.embedder import embed_query
from pipeline.generator import generate_answer
from pipeline.relevance import is_relevant

MAX_RETRIEVAL_ATTEMPTS = 2   # 1 initial + 1 retry — caps the loop, avoids runaway cost


class RAGState(TypedDict):
    question: str
    current_query: str                 
    history: List[Tuple[str, str]]
    is_broad: Optional[bool]
    context_chunks: Optional[List[Dict]]
    answer: Optional[str]
    provider: Optional[str]
    retrieval_attempts: int            


def build_rag_graph(vector_db):

    def classify_question(state: RAGState) -> dict:
        broad = is_broad_question(state["question"])
        return {"is_broad": broad}

    def retrieve_narrow(state: RAGState) -> dict:
        query_embedding = embed_query(state["current_query"])
        chunks = vector_db.search(query_embedding, top_k=10)
        return {
            "context_chunks": chunks,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
        }

    def retrieve_broad(state: RAGState) -> dict:
        chunks = vector_db.get_all_chunks()
        return {"context_chunks": chunks}

    def check_relevance(state: RAGState) -> dict:
        relevant = is_relevant(state["question"], state["context_chunks"])
        return {"is_relevant": relevant}

    def reformulate_query(state: RAGState) -> dict:  # Simple, cheap reformulation — no LLM call
        return {"current_query": state["question"]}

    def generate(state: RAGState) -> dict:
        answer, provider = generate_answer(
            state["context_chunks"],
            state["question"],
            state["history"],
        )
        return {"answer": answer, "provider": provider}

    def route_after_classify(state: RAGState) -> str:
        return "retrieve_broad" if state["is_broad"] else "retrieve_narrow"

    def route_after_relevance(state: RAGState) -> str:
        if state["is_relevant"]:
            return "generate"
        if state["retrieval_attempts"] >= MAX_RETRIEVAL_ATTEMPTS:
            return "generate"
        return "reformulate_query"

    graph = StateGraph(RAGState)
    graph.add_node("classify_question", classify_question)
    graph.add_node("retrieve_narrow", retrieve_narrow)
    graph.add_node("retrieve_broad", retrieve_broad)
    graph.add_node("check_relevance", check_relevance)
    graph.add_node("reformulate_query", reformulate_query)
    graph.add_node("generate", generate)

    graph.set_entry_point("classify_question")
    graph.add_conditional_edges(
        "classify_question",
        route_after_classify,
        {"retrieve_narrow": "retrieve_narrow", "retrieve_broad": "retrieve_broad"},
    )

    graph.add_edge("retrieve_narrow", "check_relevance")
    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {"generate": "generate", "reformulate_query": "reformulate_query"},
    )
    graph.add_edge("reformulate_query", "retrieve_narrow") 

    graph.add_edge("retrieve_broad", "generate")   # broad path skips relevance check entirely
    graph.add_edge("generate", END)

    return graph.compile()
