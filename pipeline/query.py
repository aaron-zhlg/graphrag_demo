from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.bedrock import Bedrock
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.graph_stores.neptune import NeptuneDatabaseGraphStore

from llama_index.core.prompts.base import (
    PromptTemplate,
    PromptType,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

llm = Bedrock(model="anthropic.claude-3-sonnet-20240229-v1:0")
Settings.llm = llm


graph_store = NeptuneDatabaseGraphStore(
    host="test.cluster-c9yyqywmes63.us-east-1.neptune.amazonaws.com",
    port=8182,
    node_label="Post"
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

ENTITY_EXTRACT_TMPL_STR = """
A question is provided below. 
Given the question, extract up to {max_keywords} information that identify a given user in the question. Avoid stopwords.
Focus on extracting complete information from question, it can be more than one single word.
---------------------
{question}
---------------------
Provide information in the following comma-separated format: 'KEYWORDS: <information>'
"""

ENTITY_EXTRACT_PROMPT = PromptTemplate(
    ENTITY_EXTRACT_TMPL_STR,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)

AMAZON_NEPTUNE_NL2CYPHER_PROMPT_TMPL_STR = """
Create a **Amazon Neptune flavor Cypher query** based on provided relationship paths and a question.
The query should be able to try best answer the question with the given graph schema.
The query should follow the following guidance:
- Fully qualify property references with the node's label.
```
// Incorrect
MATCH (p:person)-[:follow]->(:person) RETURN p.name
// Correct
MATCH (p:person)-[:follow]->(i:person) RETURN i.name
```
- Strictly follow the relationship on schema:
Given the relationship ['(:`Art`)-[:`BY_ARTIST`]->(:`Artist`)']:
```
// Incorrect
MATCH (a:Artist)-[:BY_ARTIST]->(t:Art)
RETURN DISTINCT t
// Correct
MATCH (a:Art)-[:BY_ARTIST]->(t:Artist)
RETURN DISTINCT t
```
- Follow single direction (from left to right) query model:
```
// Incorrect
MATCH (a:Artist)<-[:BY_ARTIST]-(t:Art)
RETURN DISTINCT t
// Correct
MATCH (a:Art)-[:BY_ARTIST]->(t:Artist)
RETURN DISTINCT t
```
Given any relationship property, you should just use them following the relationship paths provided, respecting the direction of the relationship path.
With these information, construct a Amazon Neptune Cypher query to provide the necessary information for answering the question, only return the plain text query, no explanation, apologies, or other text.
NOTE:
0. Try to get as much graph data as possible to answer the question
1. Put a limit of 30 results in the query.
---
Question: {query_str}
---
Schema: {schema}
---
Amazon Neptune flavor Query:
"""

NL2CYPHER_PROMPT = PromptTemplate(
    AMAZON_NEPTUNE_NL2CYPHER_PROMPT_TMPL_STR,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    entity_extract_template=ENTITY_EXTRACT_PROMPT,
    with_nl2graphquery=True,
    graph_query_synthesis_prompt=NL2CYPHER_PROMPT,
    graph_traversal_depth=3
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
    response_mode=ResponseMode.REFINE,
)
