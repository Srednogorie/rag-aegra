from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Query routing prompt for individual queries
routing_prompt = ChatPromptTemplate.from_template(
    """
        You are a query router for TechMart's RAG system.
        Determine which collections should be searched for this specific query.

        Available collections:
        - **catalog**: Product information, specifications, recommendations
        - **faq**: Customer service, shipping, returns, policies
        - **troubleshooting**: Technical support, problem diagnosis, solutions for TechMart products
        - **web_search**: For troubleshooting questions about issues not covered in our internal
        troubleshooting database, or for getting the latest information about software, drivers,
        or technical problems that require current information from the internet

        Search strategies:
        - **single_collection**: Search one most relevant collection
        - **multi_collection**: Search 2-3 relevant collections
        - **comprehensive**: Search all collections for complex/vague queries

        Collection selection rules:
        - Use **web_search** for troubleshooting issues that are likely not covered in our limited internal database
        - Use **web_search** for current software, driver, or OS compatibility issues
        - Use internal collections (catalog, faq, troubleshooting) for TechMart-specific information

        Examples:
        1. "What gaming laptops do you have?" -> Collections: ["catalog"]
        2. "What are your return policies?" -> Collections: ["faq"]
        3. "My laptop won't turn on" -> Collections: ["troubleshooting"]
        4. "Windows 11 blue screen error 0x0000007E" -> Collections: ["web_search"]
        5. "Latest NVIDIA driver causing display issues" -> Collections: ["web_search"]
        6. "I need a laptop" -> Collections: ["catalog"]

        Query: {query}

        Determine the routing:
    """,
)

# Document grading prompt
grading_prompt = ChatPromptTemplate.from_template(
    """
        You are evaluating the relevance of a retrieved document to a user query.

        Query: {query}

        Document:
        {document}

        Instructions:
        - Mark as "yes" if the document contains information that helps answer the query, even if:
        - The SKU number differs
        - The model variant or version name (e.g., “Wireless” vs. base model)
        is different but from the same product family
        - Mark as "no" only if the product in the document is clearly a different, unrelated product line.
        - Focus on the product family and core features described, not exact SKUs or small variant descriptors.
        - Give brief reasoning for your assessment.
        Evaluate the document relevance.
    """,
)

# Query rewriting prompt - enhanced to handle context
rewrite_prompt = ChatPromptTemplate.from_template(
    """
        You are helping improve a query for better document retrieval.
        The original query may need enhancement either for general improvement or by incorporating previous context.

        Original query: {query}
        Previous context: {previous_context}

        Context: This is for a TechMart electronics store with products, customer service info,
        and troubleshooting guides.

        Rewriting strategies:
        - If previous_context is provided, incorporate specific findings (product names, models, details) into the query
        - Add more specific terms and context from previous results
        - Expand abbreviations and clarify ambiguous terms
        - Add relevant synonyms or alternative phrasings
        - Make implicit requirements explicit

        Examples with context:
        - Original: "Check if has issues" + Context: "Found GlideMaster MX Wireless Mouse as best mouse"
        → "Check if GlideMaster MX Wireless Mouse has known issues, problems, or user complaints"
        - Original: "What about warranty" + Context: "User interested in ZenithBook 13 Evo laptop"
        → "What is the warranty coverage and terms for ZenithBook 13 Evo laptop"

        Examples without context:
        - "fast computer" → "high performance laptop with fast processor and SSD storage"
        - "setup help" → "step by step guide for setting up and configuring new computer"
        - "won't work" → "troubleshooting device not functioning properly or not turning on"

        Provide an improved query:
    """,
)

subquery_answer_generation_prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful assistant for TechMart, an electronics retailer.
        Use the following relevant documents to answer the specific query.

        Query: {query}

        Context from relevant documents:
        {context}

        Instructions:
        - Provide a focused answer to this specific query
        - Be concise but informative
        - Include specific product details when available
        - If no relevant documents were found, acknowledge the limitation

        Answer:
    """,
)

# Query analysis prompt - focused only on decomposition, not collection routing
query_analysis_prompt = ChatPromptTemplate.from_template(
    """
        You are a query decomposition analyzer for TechMart's adaptive RAG system.
        TechMart is a tech retailer specializing in gaming laptops.
        Your job is to determine if a query needs decomposition. However, some queries are not related
        to the tech retailer's products and should not be decomposed even if complex. Such queries don't
        even require RAG or web search because they are not product-related, general questions.
        Such queries can be answered directly without decomposition or retrieval.

        Query decomposition rules:
        - **Decompose if**: Query contains multiple distinct questions, has complex AND/OR logic,
        or spans multiple unrelated domains, or asks about multiple products
        - **Don't decompose if**: Query is cohesive even if complex, or sub-parts heavily depend on each other

        Execution plans:
        - **sequential**: When later sub-queries depend on earlier results
        (e.g., "Find gaming laptops, then tell me about warranty for the best one")
        - **parallel**: When sub-queries are independent
        (e.g., "What gaming laptops do you have and what are your shipping options?")

        Simple queries:
        - **direct_answer**: When the query is a simple question that does not need decomposition, RAG or web search.

        Examples:
        1. "What gaming laptops do you have and what are your return policies?"
        - Decompose: Yes, Sub-queries: ["What gaming laptops do you have?", "What are your return policies?"],
        Execution: parallel

        2. "I need a laptop under $1000, then tell me how long shipping takes for my choice"
        - Decompose: Yes, Sub-queries: ["Show me laptops under $1000", "How long does shipping take?"],
        Execution: sequential

        3. "What are the specs of the UltraBook Pro 14?"
        - Decompose: No

        4. "Windows 11 blue screen error 0x0000007E - how do I fix this?"
        - Decompose: No

        5. "My laptop won't turn on"
        - Decompose: No

        6. "I need a laptop and want to know about Chrome crashes on macOS"
        - Decompose: Yes, Sub-queries: ["I need a laptop", "How to fix Chrome crashes on macOS"], Execution: parallel

        7. "Hello agent"
        - direct_answer: Yes

        8. "How are you"
        - direct_answer: Yes

        9. "I had a bad day and didn't sleep well"
        - direct_answer: Yes

        10. "Tell me about the sun and the moon"
        - direct_answer: Yes

        Query: {query}

        Provide your query analysis:
    """,
)

final_generate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
            You are a helpful assistant for TechMart, an electronics retailer.
            Use the following context to answer the user's question accurately and helpfully.

            Direct Answer: {direct_answer}

            Original Question: {original_question}
            Query was decomposed: {was_decomposed}
            {decomposition_info}

            Total Relevant Documents: {num_docs}

            Context:
            {context}

            Instructions:
            - Provide a comprehensive answer based on the retrieved information
            - If the query was decomposed, address each part of the original question
            - If information spans multiple domains, organize your response clearly
            - Be specific and helpful, mentioning product names and details when available
            - If no relevant documents were found, provide a helpful response indicating limitations
            - If direct answer is set to True that means the question is most probably not related to the
            company's products or services but it's any other general question. In such cases, very likely
            not relevant documents will be found.
        """,
    ),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Now, generate the final Answer based on the history and context above:"),
])
