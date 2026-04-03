from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


# ## Subgraph State Definition
# We need to define the state structure for processing individual queries through our subgraph.
# This state will track the journey of a single query from initial
# analysis through retrieval, evaluation, and generation.
# Subgraph state for individual query processing
class ProcessQueryGraphState(TypedDict):
    """State for processing a single query through the subgraph."""
    query: str
    rewritten_query: str
    reason_for_rewrite: str
    needs_rewrite: bool
    num_retries: int
    routing_decision: str
    retrieved_docs: list[Document]
    relevant_docs: list[Document]
    subquery_answer: str


class QueryRouting(BaseModel):
    """Routing decision for a single query."""

    collection_needed: Literal["catalog", "faq", "troubleshooting", "web_search"] = Field(
        description="Collections that should be searched",
    )
    reasoning: str = Field(description="Explanation of the routing decision")


class DocumentGrade(BaseModel):
    """Grade for document relevance."""

    relevant: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to the query",
    )
    reasoning: str = Field(description="Brief explanation of the relevance assessment")


class QueryRewrite(BaseModel):
    """Rewritten query for better retrieval."""

    rewritten_query: str = Field(
        description="Improved version of the original query",
    )
    improvements: str = Field(
        description="Explanation of what was improved",
    )


class QueryAnalysis(BaseModel):
    """Analysis of query characteristics - focused on decomposition only."""

    needs_decomposition: bool = Field(
        description="Whether the query should be broken down into sub-queries",
    )
    sub_queries: list[str] = Field(
        description="List of sub-queries if decomposition is needed",
        default_factory=list,
    )
    execution_plan: Literal["sequential", "parallel"] = Field(
        description="How sub-queries should be executed",
        default="parallel",
    )
    reasoning: str = Field(
        description="Explanation of the decomposition analysis",
    )
    direct_answer: bool = Field(
        description="The question might be a general question that does not need RAG or web search",
        default=False,
    )


# ## Define Main Graph State
# The main graph coordinates the overall adaptive RAG process. It analyzes queries for decomposition,
# processes subqueries (either sequentially or in parallel), and combines results into a final answer.
class MainGraphState(TypedDict):
    """State for our 3-node Adaptive RAG workflow."""

    # Input
    messages: Annotated[list[BaseMessage], add_messages(format="langchain-openai")]
    query: str
    original_query: str

    # Query decomposition results
    needs_decomposition: bool
    sub_queries: list[str]
    execution_plan: str  # "sequential" or "parallel"

    # There are queries that don't need any RAG or web search - we can answer directly
    direct_answer: str

    # Results storage: query -> list of documents
    query_results: dict  # Dict[str, List[Document]]

    # Final results
    all_retrieved_docs: list[Document]

    # Output
    answer: str



