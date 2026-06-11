"""Answer generation with a LangChain LCEL chain.

When an LLM key is configured, a prompt -> model -> parser chain produces a
grounded answer. With no key, the API falls back to the most relevant retrieved
chunk, so it always works for demos.
"""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def generate_answer(question: str, contexts: list[str]) -> str:
    llm = _get_llm()
    if llm is None:
        return contexts[0] if contexts else "No relevant information found."
    chain = _PROMPT | llm | StrOutputParser()
    try:
        return chain.invoke(
            {"context": "\n\n".join(contexts), "question": question}
        ).strip()
    except Exception:
        return contexts[0] if contexts else "No relevant information found."


def _get_llm():
    """Return a LangChain chat model if a key is set, else None."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    if os.getenv("SARVAM_API_KEY"):
        # Sarvam exposes an OpenAI-compatible endpoint.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="sarvam-m",
            temperature=0.2,
            api_key=os.environ["SARVAM_API_KEY"],
            base_url="https://api.sarvam.ai/v1",
        )
    return None
