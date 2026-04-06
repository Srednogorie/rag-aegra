import asyncio
import os
import sys
sys.path.append("/home/middlefour/Development/oreilly/agentic_rag_with_langgraph")

from llama_index.core.schema import NodeWithScore

from dotenv import load_dotenv
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Check if Tavily API key is available
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is required for web search functionality")

print("OpenAI API key found!")
print("Tavily API key found!")
# from IPython.display import Image, display
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph

from src.vector_collections import collections
from src.backend.state import (
    DocumentGrade,
    MainGraphState,
    ProcessQueryGraphState,
    QueryAnalysis,
    QueryRewrite,
    QueryRouting,
)
from src.backend.prompts import (
    final_generate_prompt,
    grading_prompt,
    query_analysis_prompt,
    rewrite_prompt,
    routing_prompt,
    subquery_answer_generation_prompt,
)

print("All imports successful!")

analysis_llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=openai_api_key,
)

# Generation LLM - Balanced model for answer generation
generation_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=openai_api_key,
)

# Evaluation LLM - Capable model for document relevance evaluation
evaluation_llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=openai_api_key,
)

print(f"Analysis language model initialized: {analysis_llm.model_name}")
print(f"Generation language model initialized: {generation_llm.model_name}")
print(f"Evaluation language model initialized: {evaluation_llm.model_name}")

# %%
# Create routing chain
routing_chain = routing_prompt | analysis_llm.with_structured_output(QueryRouting)
print("Query router created!")


# ## Document Grading System
# After retrieving documents, we need to evaluate their relevance to the query.
# This grading system helps us filter out irrelevant documents and determine if we need to retry with a rewritten query.
# Create grading chain with evaluation LLM
grading_chain = grading_prompt | evaluation_llm.with_structured_output(DocumentGrade)


async def evaluate_documents(query: str, documents: list[Document]) -> list[Document]:
    """Evaluate document relevance in parallel and return relevant docs with quality score."""
    if not documents:
        print("No documents to evaluate")
        return []

    print(f"Evaluating {len(documents)} documents for query: {query}")

    # Evaluate all documents in parallel
    tasks = [
        grading_chain.ainvoke({"query": query, "document": doc.node.text})
        if isinstance(doc, NodeWithScore) else
        grading_chain.ainvoke({"query": query, "document": doc.page_content})
        for doc in documents
    ]
    grades = await asyncio.gather(*tasks)

    # Process results
    relevant_docs = []

    for doc, grade in zip(documents, grades):
        if grade.relevant == "yes":
            relevant_docs.append(doc)
            print(f"✓ Relevant doc: {grade.reasoning}")
        else:
            print(f"✗ Irrelevant doc: {grade.reasoning}")

    print(f"Document evaluation complete: {len(relevant_docs)}/{len(documents)} relevant")

    return relevant_docs


print("Document evaluation system created!")

# ## Query Rewriting and Expansion
# When initial results are poor, we can rewrite the query to improve retrieval.
# Create rewriting chain with analysis LLM
rewrite_chain = rewrite_prompt | analysis_llm.with_structured_output(QueryRewrite)
print("Enhanced query rewriting system created using gpt-4.1!")

# ## Answer Generation for Subqueries
# Each subquery processed through our subgraph needs to generate a focused answer based on the relevant documents found.
# This answer will later be combined with other subquery results to form the final response.
# Subquery answer generation
subquery_answer_generation_chain = subquery_answer_generation_prompt | generation_llm
print("Subquery answer generation chain created!")

# ## Define Subgraph Nodes and Conditional Logic
#
# Now we'll define the individual nodes that make up our subgraph workflow.
# Each node handles a specific step in processing a single query:
#
# - **Query Rewriter**: Enhances queries with context (first node in the pipeline)
# - **Router**: Determines which collection to search based on query content
# - **Retrieval Nodes**: Search specific collections (catalog, FAQ, troubleshooting, web)
# - **Evaluation**: Grades document relevance and determines if retry is needed
# - **Answer Generation**: Creates focused responses from relevant documents
#
# The conditional logic determines the flow between nodes,
# including retry mechanisms when initial retrieval fails to find relevant documents.

# define the conditional edges that determine the next node to visit
def should_retry_with_rewrite(state: ProcessQueryGraphState) -> str:
    """Determine if query should be retried with rewriting or proceed to generation."""
    needs_rewrite = state.get("needs_rewrite", False)
    num_retries = state.get("num_retries", 0)

    # Proceed to rewrite if flag is set and we haven't exceeded retry limit
    if needs_rewrite and num_retries < 2:
        print(f"Will retry with query rewriting (attempt {num_retries}, needs_rewrite={needs_rewrite})")
        return "query_rewriter_node"
    # Either we don't need rewrite or already retried maximum times
    if needs_rewrite and num_retries >= 2:
        print(f"Max retries reached ({num_retries}), proceeding to generation despite needing rewrite")
    else:
        print("No rewrite needed or found relevant documents, proceeding to generation")
    return "generate_subquery_answer"


# define the nodes
def query_rewriter_node(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Rewrite query with context if needed, otherwise pass through."""
    query = state["query"]
    needs_rewrite = state.get("needs_rewrite", False)
    reason_for_rewrite = state.get("reason_for_rewrite", "")
    num_retries = state.get("num_retries", 0)

    # Check if this is a retry due to no relevant docs
    if num_retries > 0:
        print(f"Retrying query rewrite due to no relevant documents: {query}")
        reason_for_rewrite = "No relevant documents found, improving query for better retrieval"
        needs_rewrite = True

    if not needs_rewrite:
        # Skip rewriting, pass through original query
        print(f"Skipping query rewrite for: {query}")
        return {
            "rewritten_query": query,
            "needs_rewrite": False,
            "num_retries": num_retries,
        }

    # Rewrite query using previous context
    print(f"Rewriting query with context: {query}")
    print(f"Previous context: {reason_for_rewrite}")

    rewrite_result = rewrite_chain.invoke({
        "query": query,
        "previous_context": reason_for_rewrite,
    })

    print(f"Original query: {query}")
    print(f"Rewritten query: {rewrite_result.rewritten_query}")
    print(f"Improvements: {rewrite_result.improvements}")

    return {
        "rewritten_query": rewrite_result.rewritten_query,
        "reason_for_rewrite": f"Enhanced: {rewrite_result.improvements}",
        "needs_rewrite": False,  # Always reset flag after rewriting
        "num_retries": num_retries,  # Don't modify counter here
    }


def route_query(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Route individual query to appropriate collections."""
    # Use rewritten_query instead of query
    query = state["rewritten_query"]

    # Route the query
    routing = routing_chain.invoke({"query": query})

    print(f"Routing query: {query}")
    print(f"Reasoning: {routing.reasoning}")

    return {
        "routing_decision": routing.collection_needed or "catalog",
    }


def retrieve_from_catalog(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Retrieve documents from the product catalog."""
    query = state["rewritten_query"]  # Use rewritten_query
    retrieved_docs = collections["catalog"].retrieve(query)

    print(f"Retrieved {len(retrieved_docs)} documents from catalog")
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("filename", "Unknown")
        content_preview = doc.node.text[:100]
        print(f"  Doc {i} [{source}]: {content_preview}...")

    return {"retrieved_docs": retrieved_docs}


def retrieve_from_faq(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Retrieve documents from the FAQ collection."""
    query = state["rewritten_query"]  # Use rewritten_query
    retrieved_docs = collections["faq"].retrieve(query)

    print(f"Retrieved {len(retrieved_docs)} documents from FAQ")
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("filename", "Unknown")
        content_preview = doc.node.text[:100]
        print(f"  Doc {i} [{source}]: {content_preview}...")

    return {"retrieved_docs": retrieved_docs}


def retrieve_from_troubleshooting(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Retrieve documents from the troubleshooting collection."""
    query = state["rewritten_query"]  # Use rewritten_query
    retrieved_docs = collections["troubleshooting"].retrieve(query)

    print(f"Retrieved {len(retrieved_docs)} documents from troubleshooting")
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("filename", "Unknown")
        content_preview = doc.node.text[:100]
        print(f"  Doc {i} [{source}]: {content_preview}...")

    return {"retrieved_docs": retrieved_docs}


# Initialize Tavily retriever for web search
tavily_retriever = TavilySearchAPIRetriever(
    api_key=tavily_api_key,
    k=5,  # Number of search results to return
    include_generated_answer=False,
    include_raw_content=False,
)


def retrieve_from_web_search(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Retrieve documents from web search using Tavily."""
    query = state["rewritten_query"]  # Use rewritten_query

    # Use Tavily to search the web
    retrieved_docs = tavily_retriever.invoke(query)

    print(f"Retrieved {len(retrieved_docs)} documents from web search")
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        content_preview = doc.page_content[:100]
        print(f"  Doc {i} [{source}]: {content_preview}...")

    return {"retrieved_docs": retrieved_docs}


async def evaluate_retrieved_documents(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Evaluate the quality of retrieved documents for a single query."""
    # Use rewritten_query for evaluation
    query = state["rewritten_query"]
    docs = state["retrieved_docs"]
    num_retries = state.get("num_retries", 0)

    print(f"Evaluating {len(docs)} documents for query: {query}")

    # Evaluate document relevance
    relevant_docs = await evaluate_documents(query, docs)

    print("Query Quality Assessment:")
    print(f"Relevant docs: {len(relevant_docs)}/{len(docs)}")

    # Handle retry logic based on results
    if len(relevant_docs) == 0:
        # No relevant docs found - signal for retry and increment counter
        new_retries = num_retries + 1
        print(
            f"No relevant docs found, setting needs_rewrite=True and incrementing retries: "
            f"{num_retries} → {new_retries}",
        )
        return {
            "relevant_docs": relevant_docs,
            "needs_rewrite": True,
            "num_retries": new_retries,
        }

    # Found relevant docs - no retry needed
    print(f"Found {len(relevant_docs)} relevant docs, no retry needed")
    return {
        "relevant_docs": relevant_docs,
        "needs_rewrite": False,
        "num_retries": num_retries,  # Keep current value
    }


def generate_subquery_answer(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Generate answer for the subquery using relevant documents."""
    # Use rewritten_query for generation
    query = state["rewritten_query"]
    relevant_docs = state["relevant_docs"]

    print(f"Generating answer for subquery: {query}")

    # Prepare context from relevant documents
    if relevant_docs:
        context_parts = []
        for doc in relevant_docs:
            collection = doc.metadata.get("category", "unknown")
            source = doc.metadata.get("filename", "unknown")
            context_parts.append(f"[{collection.upper()}] {source}: {doc.node.text}")
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No relevant documents found for this subquery."

    # Generate response using the subquery generation chain
    response = subquery_answer_generation_chain.invoke({"query": query, "context": context_text})

    print(f"Generated subquery answer: {response.content}")

    return {"subquery_answer": response.content}


def rewrite_query(state: ProcessQueryGraphState) -> ProcessQueryGraphState:
    """Rewrite query for better retrieval."""
    original_query = state["original_query"]
    retry_count = state["retry_count"]

    print(f"Rewriting query (attempt {retry_count + 1})...")

    # Rewrite the query
    rewrite = rewrite_chain.invoke({"query": original_query})

    print(f"Original: {original_query}")
    print(f"Rewritten: {rewrite.rewritten_query}")
    print(f"Improvements: {rewrite.improvements}")

    return {
        "query": rewrite.rewritten_query,
        "rewritten_query": rewrite.rewritten_query,
        "needs_rewrite": False,
        "retry_count": retry_count + 1,
    }


print("Subgraph nodes with fixed retry logic created!")


# ## Building the Subgraph
#
# The subgraph handles the complete life cycle of processing a single query, starting with query rewriting/enhancement,
# then routing → retrieval → evaluation → generation.
# It includes self-correction mechanisms to retry with rewritten queries when initial retrieval fails.

# Define routing function for conditional edges
def route_to_retrieval_source(state: ProcessQueryGraphState) -> str:
    """Determine which retrieve node to call based on router decision."""
    routing_decision = state["routing_decision"]
    return f"retrieve_from_{routing_decision}"


# Build the subgraph
subgraph_builder = StateGraph(ProcessQueryGraphState)
# Add query rewriter node as new entry point
subgraph_builder.add_node("query_rewriter_node", query_rewriter_node)
# Add routing node
subgraph_builder.add_node("route_query", route_query)
# Add separate retrieval nodes
subgraph_builder.add_node("retrieve_from_catalog", retrieve_from_catalog)
subgraph_builder.add_node("retrieve_from_faq", retrieve_from_faq)
subgraph_builder.add_node("retrieve_from_troubleshooting", retrieve_from_troubleshooting)
subgraph_builder.add_node("retrieve_from_web_search", retrieve_from_web_search)
# Add evaluation node
subgraph_builder.add_node("evaluate_retrieved_documents", evaluate_retrieved_documents)
# Add generation node
subgraph_builder.add_node("generate_subquery_answer", generate_subquery_answer)
# Set entry point for subgraph to query rewriter
subgraph_builder.set_entry_point("query_rewriter_node")
# Add edge from query rewriter to router
subgraph_builder.add_edge("query_rewriter_node", "route_query")
# Add conditional edges from router to appropriate retrieve node
subgraph_builder.add_conditional_edges(
    "route_query",
    route_to_retrieval_source,
    {
        "retrieve_from_catalog": "retrieve_from_catalog",
        "retrieve_from_faq": "retrieve_from_faq",
        "retrieve_from_troubleshooting": "retrieve_from_troubleshooting",
        "retrieve_from_web_search": "retrieve_from_web_search",
    },
)
# All retrieve nodes go to evaluation
subgraph_builder.add_edge("retrieve_from_catalog", "evaluate_retrieved_documents")
subgraph_builder.add_edge("retrieve_from_faq", "evaluate_retrieved_documents")
subgraph_builder.add_edge("retrieve_from_troubleshooting", "evaluate_retrieved_documents")
subgraph_builder.add_edge("retrieve_from_web_search", "evaluate_retrieved_documents")
# Add conditional edges from evaluation - either retry or proceed to generation
subgraph_builder.add_conditional_edges(
    "evaluate_retrieved_documents",
    should_retry_with_rewrite,
    {
        "query_rewriter_node": "query_rewriter_node",
        "generate_subquery_answer": "generate_subquery_answer",
    },
)
# Generation is the end
subgraph_builder.add_edge("generate_subquery_answer", END)
# Compile the subgraph
process_query_subgraph = subgraph_builder.compile()
print("Query processing subgraph with self-correction mechanism created and compiled!")

# visualize the subgraph
# display(Image(process_query_subgraph.get_graph().draw_mermaid_png()))


# ## Query Analysis and Decomposition
#
# The first step in our adaptive RAG system is analyzing the query to determine the optimal processing strategy.
# This analysis focuses specifically on **query decomposition** -
# deciding whether a complex query should be broken down into simpler sub-queries.
#
# ### Key Analysis Components:
#
# 1. **Decomposition Detection**: Identifies queries that contain multiple distinct questions or span unrelated domains
# 2. **Execution Planning**: Determines whether sub-queries should run in parallel
# (independent) or sequentially (dependent)
# 3. **Dependency Analysis**: Recognizes when later queries need results from earlier ones
#
# ### Examples of Decomposition:
# - **Parallel**: "What laptops do you have AND what are your return policies?" → Two independent questions
# - **Sequential**: "Find the best gaming mouse, THEN check if it has issues" → Second query depends on first result
# - **No Decomposition**: "What are the specs of UltraBook Pro 14?" → Single cohesive question
#
# The analysis enables our system to handle complex, multi-part queries more effectively than simple routing approaches.
# Create decomposition analysis chain with analysis LLM
query_analysis_chain = query_analysis_prompt | analysis_llm.with_structured_output(QueryAnalysis)

print("Query analysis system created using gpt-4.1!")


# define a helper function to process a single query using the subgraph
async def _process_single_query(query: str, needs_rewrite: bool = False, previous_context: str = "") -> dict:
    """Process a single query through the subgraph."""
    # Create logger for this specific query
    # logger = log_manager.get_logger(query)
    print(f"Starting processing for query: {query}")

    # Initialize subgraph state with logger and all required fields
    subgraph_state = {
        "query": query,
        "rewritten_query": "",  # Will be set by query_rewriter_node
        "reason_for_rewrite": previous_context,
        "needs_rewrite": needs_rewrite,
        "num_retries": 0,  # Initialize retry counter
        "routing_decision": "",
        "retrieved_docs": [],
        "relevant_docs": [],
        "subquery_answer": "",
        # "logger": logger,
    }

    # Run the subgraph
    result = await process_query_subgraph.ainvoke(subgraph_state)

    print(f"Completed processing for query: {query}")

    return {
        "query": query,
        "documents": result.get("relevant_docs", []),
        "subquery_answer": result.get("subquery_answer", ""),
    }

# ## Main Graph Nodes
# The main graph consists of three primary nodes that orchestrate the entire adaptive RAG workflow: query analysis,
# query processing, and final answer generation.
def get_last_human_message(messages):
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

# 3-node main graph
def query_analysis(state: MainGraphState) -> MainGraphState:
    """Decompose query and determine execution plan."""
    query = get_last_human_message(state["messages"])

    # Analyze the query for decomposition only
    analysis = query_analysis_chain.invoke({"query": query})

    print("Query Decomposition Analysis:")
    print(f"  Needs Decomposition: {analysis.needs_decomposition}")
    if analysis.needs_decomposition:
        print(f"  Sub-queries: {analysis.sub_queries}")
        print(f"  Execution Plan: {analysis.execution_plan}")
    print(f"  Reasoning: {analysis.reasoning}")

    return {
        "original_query": query,
        "needs_decomposition": analysis.needs_decomposition,
        "sub_queries": analysis.sub_queries if analysis.needs_decomposition else [query],
        "execution_plan": analysis.execution_plan,
        "query_results": {},
        "direct_answer": analysis.direct_answer
    }


async def process_queries(state: MainGraphState) -> MainGraphState:
    """Process queries using asyncio for sequential/parallel execution with organized logging."""
    sub_queries = state["sub_queries"]
    execution_plan = state["execution_plan"]
    needs_decomposition = state["needs_decomposition"]
    direct_answer = state["direct_answer"]

    query_results = {}
    all_docs = []
    if not direct_answer:
        if not needs_decomposition:
            # Single query - no rewriting needed
            print(f"\nProcessing single query: {sub_queries[0]}")
            result = await _process_single_query(sub_queries[0], needs_rewrite=False)
            query_results[result["query"]] = result["documents"]
            all_docs.extend(result["documents"])

        elif execution_plan == "parallel":
            # Parallel execution - no rewriting needed for any query
            print(f"\nExecuting {len(sub_queries)} queries in parallel")

            # Create tasks for all queries (no context needed for parallel)
            tasks = [_process_single_query(query, needs_rewrite=False) for query in sub_queries]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Process results
            for result in results:
                query_results[result["query"]] = result["documents"]
                all_docs.extend(result["documents"])


        else:  # sequential
            # Sequential execution with context accumulation
            print(f"\nExecuting {len(sub_queries)} queries sequentially")

            accumulated_context = ""
            for i, query in enumerate(sub_queries, 1):
                print(f"\n--- Sequential Step {i}/{len(sub_queries)} ---")

                # First query doesn't need rewriting, subsequent ones do
                needs_rewrite = i > 1

                result = await _process_single_query(
                    query,
                    needs_rewrite=needs_rewrite,
                    previous_context=accumulated_context,
                )

                query_results[result["query"]] = result["documents"]
                all_docs.extend(result["documents"])

                # Accumulate context for next query
                accumulated_context += f"Step {i} result: {result['subquery_answer']}\n"

    print(f"\n{'=' * 80}")
    print("Query Processing Summary:")
    print(f"  Total queries processed: {len(sub_queries)}")
    print(f"  Total documents retrieved: {len(all_docs)}")
    print(f"  Execution strategy: {execution_plan}")
    print(f"{'=' * 80}")

    return {
        "query_results": query_results,
        "all_retrieved_docs": all_docs,
    }


def generate(state: MainGraphState) -> MainGraphState:
    """Generate final answer using combined results from all queries."""
    original_query = state["original_query"]
    needs_decomposition = state["needs_decomposition"]
    query_results = state["query_results"]
    all_docs = state["all_retrieved_docs"]
    direct_answer = state["direct_answer"]

    print(f"Generating final answer with {len(all_docs)} total documents")

    # Prepare context from all documents
    if all_docs:
        context_parts = []
        for doc in all_docs:
            collection = doc.metadata.get("category", "unknown")
            source = doc.metadata.get("filename", "unknown")
            context_parts.append(f"[{collection.upper()}] {source}: {doc.node.text}")
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No relevant documents found."

    # Prepare decomposition info
    if needs_decomposition and query_results:
        decomposition_info = "Sub-queries processed:\n"
        for query, docs in query_results.items():
            decomposition_info += f"  • {query} ({len(docs)} documents)\n"
        was_decomposed = "Yes"
    else:
        decomposition_info = ""
        was_decomposed = "No"

    # Generate response
    messages = final_generate_prompt.invoke({
        "context": context_text,
        "original_question": original_query,
        "was_decomposed": was_decomposed,
        "decomposition_info": decomposition_info,
        "num_docs": len(all_docs),
        "direct_answer": direct_answer,
        "messages": state["messages"],
    })

    response = generation_llm.invoke(messages)

    print("Generated final answer")
    messages_response = state["messages"] + [AIMessage(content=response.content)]
    return {
        "answer": response.content,
        "messages": messages_response,
    }


print("Main graph nodes created!")


# ## Build the Main Graph
#
# Now we'll create the main graph that uses subgraphs and the Send API for improved query processing.

# Create the 3-node main graph

# Build the main graph
graph_builder = StateGraph(MainGraphState)

# Add the 3 main nodes
graph_builder.add_node("query_analysis", query_analysis)
graph_builder.add_node("process_queries", process_queries)
graph_builder.add_node("generate", generate)

# Set entry point
graph_builder.set_entry_point("query_analysis")

# Linear flow: analysis -> process -> generate
graph_builder.add_edge("query_analysis", "process_queries")
graph_builder.add_edge("process_queries", "generate")
graph_builder.add_edge("generate", END)

# Compile the graph
main_graph = graph_builder.compile()
