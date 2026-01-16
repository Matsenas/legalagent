from langchain_core.prompts import ChatPromptTemplate
from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from operator import itemgetter
import os
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from typing import Optional
from data_prepros import *
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
import logging

logger = logging.getLogger(__name__)
load_dotenv()
index_name= os.environ.get("INDEX_NAME")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("MODEL")
emb_model = os.environ.get("EMB_MODEL")

# Reranking configuration
RERANK_ENABLED = True
FLASHRANK_MODEL = "ms-marco-MultiBERT-L-12"
INITIAL_RETRIEVAL_K = 15
FINAL_K = 3


llm = ChatOpenAI(model=MODEL,api_key=OPENAI_API_KEY)

# FlashRank compressor singleton
_cached_compressor = None

def get_flashrank_compressor():
    """Lazy-load FlashRank compressor (singleton pattern)."""
    global _cached_compressor
    if _cached_compressor is None:
        try:
            _cached_compressor = FlashrankRerank(
                model=FLASHRANK_MODEL,
                top_n=FINAL_K
            )
            logger.info(f"FlashRank compressor initialized with model: {FLASHRANK_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize FlashRank compressor: {e}")
            return None
    return _cached_compressor

def get_retriever(with_reranking: bool = True):
    """
    Get retriever with optional FlashRank reranking.

    Args:
        with_reranking: If True, wraps base retriever with FlashRank reranking

    Returns:
        Retriever (with or without compression)
    """
    docsearch = load_pinecone(index_name)
    base_retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": INITIAL_RETRIEVAL_K if with_reranking else FINAL_K}
    )

    if with_reranking and RERANK_ENABLED:
        compressor = get_flashrank_compressor()
        if compressor:
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )

    return base_retriever

class GraphState(TypedDict):
    query: str
    generation: str
    type: int


def classify_query(query):

    class RouteQuery(BaseModel):
        """Route a user query to the most relevant data source."""

        datasource: Literal["irrelevant_query","greeting","advice_query"] = Field(
            ...,
            description="Given a user question choose to route it to relevant node",
        )
    classify_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_router = classify_llm.with_structured_output(RouteQuery)


    system = """
        You are a query classification agent for a Czech legal advisor system.

        Your task:
        Classify the user's query into exactly ONE of the following categories.

        **CATEGORIES**

        1. "greeting"
        - The user is greeting, thanking, or engaging in casual conversation.
        - Examples:
        - "Hi"
        - "Hello"
        - "Thanks"
        - "How are you?"

        2. "advice_query"
        - The user is asking for legal information, legal consequences, legal procedures,
        rights, obligations, or interpretation of law.
        - The question may be formal or informal.
        - The question may or may not explicitly mention a law.
        - Examples:
        - "Jaká je výpovědní lhůta?"
        - "Co se stane při porušení autorských práv?"
        - "Can my employer fire me without notice?"

        3. "irrelevant_query"
        - The query is NOT a greeting and NOT related to legal advice.
        - Includes technical questions, random facts, coding, math, jokes,
        or any unrelated topics.
        - Examples:
        - "Write a Python function"
        - "Who won the football match?"
        - "Explain quantum physics"

        **RULES**

        - Choose ONLY one category.
        - Do NOT explain your choice.
        - Do NOT answer the user's question.
        - Output must strictly follow the provided structured schema.

        USER QUERY:
        {query}
        """


    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{query}"),
        ]
    )
    question_router = route_prompt | structured_llm_router
    generated_response=question_router.invoke(
        {"query": query}
    )
    
    return generated_response.datasource

def _combine_documents3(docs, document_separator="\n\n"):
    try:
        if docs:
            combined_list = [
                f"Content:{doc.page_content} " for index, doc in enumerate(docs)
            ]
            combined = document_separator.join(combined_list)
        else:
            combined = ""  # No documents found
        return combined
    except Exception as ex:
        raise
def advice_query(text):
    prompt = """
        You are a Czech legal advisor providing general legal guidance.

        IMPORTANT RULES:
        - You are NOT a licensed attorney.
        - You do NOT provide binding or personalized legal advice.
        - Your role is to explain legal rules, procedures, consequences, and common interpretation.
        - All statements MUST be based on the provided legal text.
        - Do NOT invent laws, penalties, or procedures.
        - If information is missing or unclear, explicitly say so.
        - Always be cautious and neutral in tone.

        REQUIREMENTS:
        - Cite relevant Czech laws by act name, year, and section (§) where applicable.
        - Explain typical legal consequences, not guaranteed outcomes.
        - Clearly mention assumptions, exceptions, or conditions.
        - Do NOT suggest illegal actions.
        - Do NOT give step-by-step instructions for committing wrongdoing.

        RESPONSE FORMAT (MANDATORY):
        Shrnutí:
        - Stručná odpověď na dotaz

        Právní úprava:
        - Relevantní zákon(y), rok, paragraf (§)

        Výklad a běžná praxe:
        - Jak se právní úprava obvykle vykládá
        - Jaké mohou nastat důsledky

        Omezení a poznámky:
        - Výjimky, podmínky, nejistoty
        - Kdy je vhodné obrátit se na odborníka

        USER INPUT:
        {text}

    """
    summarizer_prompt = ChatPromptTemplate.from_template(prompt)
    # chat_llm = ChatOpenAI(model="gpt-4o-mini")
    summarizer_chain = {"text": itemgetter("text")} |summarizer_prompt | llm
    response=summarizer_chain.invoke({"text":text}).content
    return response

def irrelevant_query(query):
    prompt = """
    
    You are a Czech legal advisor assistant.

    The user's query is NOT related to legal advice.

    Your task:
    - Politely inform the user that you can only assist with legal questions.
    - Do NOT answer the unrelated query.
    - Do NOT mention internal routing or classification.
    - Keep the response short, clear, and respectful.
    - Encourage the user to ask a legal-related question.

    Tone:
    Professional, polite, and neutral.

    Response language:
    Same language as the user query but say I don't know for what you are looking for.

    User query: {query}
    
    """
    _prompt = ChatPromptTemplate.from_template(prompt)
    # chat_llm = ChatOpenAI(model="gpt-4o-mini")

    _chain = {"query": itemgetter("query")} |_prompt | llm
    response=_chain.invoke({"query":query}).content
    return {"response": response}


def genericRag(state):
    query = state.get("query")

    # Get retriever with FlashRank reranking via LangChain ContextualCompressionRetriever
    retriever = get_retriever(with_reranking=RERANK_ENABLED)
    docs = retriever.invoke(query)

    if not docs:
        return {
            "generation": "K tomuto dotazu nemám k dispozici relevantní právní informace."
        }

    # Limit to FINAL_K documents (compression retriever should handle this, but ensure consistency)
    docs = docs[:FINAL_K]
    logger.info(f"Retrieved {len(docs)} documents (reranking={'enabled' if RERANK_ENABLED else 'disabled'})")

    context = _combine_documents3(docs)

    prompt_str = """
        You are a Czech legal advisor providing general legal guidance.

        Rules:
        - You are NOT a licensed attorney.
        - You do NOT provide binding legal advice.
        - Use ONLY the provided legal text.
        - Cite Czech law (act name, year, §) if mentioned.
        - If information is unclear or missing, explicitly say so.
        - Keep the answer concise and professional.

        User question:
        {query}

        Relevant legal text:
        {context}

        Answer in Czech.
        """

    prompt = ChatPromptTemplate.from_template(prompt_str)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "query": query,
        "context": context
    })

    return {"generation": response}



def greeting(state):
    query = state.get("question")

    prompt_str = """
    You are a Czech legal advisor assistant.

    Task:
    Respond politely to the user's greeting.

    Rules:
    - Do NOT provide legal advice.
    - Do NOT ask follow-up questions.
    - Keep the response short and professional.
    - Invite the user to ask a legal-related question.

    Response language:
    Same language as the user's message.

    User message:
    {query}
    """

    _prompt = ChatPromptTemplate.from_template(prompt_str)
    query_fetcher = itemgetter("query")
    setup = {"query": query_fetcher}

    _chain = setup | _prompt | llm
    response = _chain.invoke({"query": query}).content
    return {"response": response}


def graph_builder():
    workflow = StateGraph(GraphState)

    workflow.add_node("advice_query", advice_query)
    workflow.add_node("genericRag", genericRag)
    workflow.add_node("greeting", greeting)
    workflow.add_node("irrelevant_query", irrelevant_query)

    workflow.add_conditional_edges(
        START,
        classify_query,
        {
            "advice_query": "genericRag",        # legal questions → RAG
            "greeting": "greeting",              # greetings
            "irrelevant_query": "irrelevant_query"  # non-legal
        },
    )

    workflow.add_edge("genericRag", END)
    workflow.add_edge("greeting", END)
    workflow.add_edge("irrelevant_query", END)


    app = workflow.compile()
    return app


def chatbot_interactor_generator(query):
    input={"query": query}
    graph_app = graph_builder()
    response = graph_app.invoke(input)
    return response
