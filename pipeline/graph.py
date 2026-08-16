from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict, Tuple

from pipeline.question_classifier import is_broad_question
from pipeline.embedder import embed_query
from pipeline.generator import generate_answer


class RAGState(TypedDict):
    question: str
    history: List[Tuple[str, str]]
    is_broad: Optional[bool]
    context_chunks: Optional[List[Dict]]
    answer: Optional[str]


def build_rag_graph(vector_db):
    """
    Builds a graph wired to THIS session's vector_db via closure.
    vector_db stays out of RAGState deliberately — it persists across many
    questions in a session, while RAGState is scoped to a single question's
    run through the graph and is discarded once generate() returns.
    """

    def classify_question(state: RAGState) -> dict:
        broad = is_broad_question(state["question"])
        return {"is_broad": broad}

    def retrieve_narrow(state: RAGState) -> dict:
        query_embedding = embed_query(state["question"])
        chunks = vector_db.search(query_embedding, top_k=10)
        return {"context_chunks": chunks}

    def retrieve_broad(state: RAGState) -> dict:
        chunks = vector_db.get_all_chunks()
        return {"context_chunks": chunks}

    def generate(state: RAGState) -> dict:
        answer = generate_answer(
            state["context_chunks"],
            state["question"],
            state["history"],
        )
        return {"answer": answer}

    def route_after_classify(state: RAGState) -> str:
        return "retrieve_broad" if state["is_broad"] else "retrieve_narrow"

    graph = StateGraph(RAGState)
    graph.add_node("classify_question", classify_question)
    graph.add_node("retrieve_narrow", retrieve_narrow)
    graph.add_node("retrieve_broad", retrieve_broad)
    graph.add_node("generate", generate)

    graph.set_entry_point("classify_question")
    graph.add_conditional_edges(
        "classify_question",
        route_after_classify,
        {"retrieve_narrow": "retrieve_narrow", "retrieve_broad": "retrieve_broad"},
    )
    graph.add_edge("retrieve_narrow", "generate")
    graph.add_edge("retrieve_broad", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
