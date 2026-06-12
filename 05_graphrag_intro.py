"""
Lesson 6.5: GraphRAG Introduction
Local / Free RAG Stack Version

Traditional RAG:
Good for semantic similarity search.

Problem:
Traditional RAG struggles with multi-hop reasoning.

Example:
"Who works in the same department as the CEO's assistant?"

To answer:
CEO -> assistant -> department -> other employees

Vector search alone is weak at relationship traversal.

GraphRAG:
Builds a knowledge graph:
Entities = nodes
Relationships = edges

Your Stack:
LLM      : qwen3:8b via Ollama
Graph    : NetworkX for demo
Framework: LangChain
API Key  : Not required
"""

import json
import re
from dotenv import load_dotenv

import networkx as nx

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------------------------------------
# Before running:
#
# ollama serve
# ollama pull qwen3:8b
#
# Install:
# pip install langchain-ollama langchain-core networkx python-dotenv
# ---------------------------------------------------------


# Local Ollama LLM
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)


# ============================================================
# THE PROBLEM: MULTI-HOP REASONING
# ============================================================

def demonstrate_multihop_problem():
    """
    Shows why normal vector RAG can fail.

    User query:
    "Who works in the same department as CEO's assistant?"

    This cannot be answered from one obvious chunk.
    It needs multiple relationship hops.
    """

    print("=" * 60)
    print("THE PROBLEM: MULTI-HOP REASONING")
    print("=" * 60)

    documents = [
        "John Smith is the CEO of TechCorp. He joined the company in 2015.",
        "Sarah Johnson is John Smith's executive assistant. She manages his calendar.",
        "Sarah Johnson works in the Executive Department on the 5th floor.",
        "The Executive Department also includes Mike Brown and Lisa Chen.",
        "Mike Brown is the Chief Financial Officer. Lisa Chen is the Chief Legal Officer.",
    ]

    query = "Who works in the same department as the CEO's assistant?"

    print(f'\nQuery: "{query}"')

    print("\nDocuments:")
    for index, doc in enumerate(documents, start=1):
        print(f"{index}. {doc}")

    print("\nWhy vector RAG may struggle:")
    print("""
To answer, we need:

1. Find CEO:
   John Smith is CEO.

2. Find CEO's assistant:
   Sarah Johnson assists John Smith.

3. Find assistant's department:
   Sarah Johnson works in Executive Department.

4. Find others in same department:
   Mike Brown and Lisa Chen.

This is multi-hop reasoning.
Vector similarity alone does not naturally traverse these relationships.
""")


# ============================================================
# BUILD KNOWLEDGE GRAPH MANUALLY
# ============================================================

def build_knowledge_graph():
    """
    Creates a simple knowledge graph manually using NetworkX.

    Nodes:
    - people
    - organization
    - department

    Edges:
    - CEO_OF
    - ASSISTANT_TO
    - WORKS_IN
    """

    print("\n" + "=" * 60)
    print("THE SOLUTION: KNOWLEDGE GRAPH")
    print("=" * 60)

    graph = nx.DiGraph()

    entities = [
        ("John Smith", {"type": "Person", "role": "CEO"}),
        ("Sarah Johnson", {"type": "Person", "role": "Executive Assistant"}),
        ("Mike Brown", {"type": "Person", "role": "CFO"}),
        ("Lisa Chen", {"type": "Person", "role": "CLO"}),
        ("TechCorp", {"type": "Organization"}),
        ("Executive Department", {"type": "Department", "floor": "5th"}),
    ]

    for name, attributes in entities:
        graph.add_node(name, **attributes)

    relationships = [
        ("John Smith", "TechCorp", {"relation": "CEO_OF"}),
        ("Sarah Johnson", "John Smith", {"relation": "ASSISTANT_TO"}),
        ("Sarah Johnson", "Executive Department", {"relation": "WORKS_IN"}),
        ("Mike Brown", "Executive Department", {"relation": "WORKS_IN"}),
        ("Lisa Chen", "Executive Department", {"relation": "WORKS_IN"}),
        ("John Smith", "Executive Department", {"relation": "WORKS_IN"}),
    ]

    for source, target, attributes in relationships:
        graph.add_edge(source, target, **attributes)

    print("\nEntities / Nodes:")
    for node, attrs in graph.nodes(data=True):
        print(f"- {node} | {attrs}")

    print("\nRelationships / Edges:")
    for source, target, attrs in graph.edges(data=True):
        print(f"- {source} --[{attrs['relation']}]--> {target}")

    return graph


# ============================================================
# GRAPH TRAVERSAL
# ============================================================

def traverse_graph_for_answer(graph: nx.DiGraph):
    """
    Answers multi-hop question using graph traversal.

    This is the key idea of GraphRAG.

    Instead of only searching similar text,
    we follow relationships.
    """

    print("\n" + "=" * 60)
    print("GRAPH TRAVERSAL FOR MULTI-HOP ANSWER")
    print("=" * 60)

    query = "Who works in the same department as the CEO's assistant?"

    print(f'\nQuery: "{query}"')
    print("\nTraversal:")

    # Step 1: Find CEO
    ceo = None

    for node, attrs in graph.nodes(data=True):
        if attrs.get("role") == "CEO":
            ceo = node
            break

    print(f"1. CEO found: {ceo}")

    # Step 2: Find CEO's assistant
    assistant = None

    for source, target, attrs in graph.edges(data=True):
        if target == ceo and attrs.get("relation") == "ASSISTANT_TO":
            assistant = source
            break

    print(f"2. CEO's assistant found: {assistant}")

    # Step 3: Find assistant's department
    department = None

    for source, target, attrs in graph.edges(data=True):
        if source == assistant and attrs.get("relation") == "WORKS_IN":
            department = target
            break

    print(f"3. Assistant department found: {department}")

    # Step 4: Find coworkers in same department
    coworkers = []

    for source, target, attrs in graph.edges(data=True):
        if target == department and attrs.get("relation") == "WORKS_IN":
            if source != assistant:
                coworkers.append(source)

    print(f"4. Others in same department: {coworkers}")

    print("\nAnswer:")
    print(
        f"The CEO is {ceo}. "
        f"The CEO's assistant is {assistant}. "
        f"{assistant} works in {department}. "
        f"Others in the same department are: {', '.join(coworkers)}."
    )


# ============================================================
# LLM ENTITY + RELATION EXTRACTION
# ============================================================

def extract_entities_with_llm():
    """
    Uses local qwen3:8b to extract entities and relationships.

    In real GraphRAG:
    documents -> LLM extraction -> graph database

    Here:
    sample text -> qwen3:8b -> JSON-like entity relation output
    """

    print("\n" + "=" * 60)
    print("LLM-BASED ENTITY EXTRACTION")
    print("=" * 60)

    sample_text = """
Acme Corporation announced that Jennifer Lee has been appointed as the new
Chief Technology Officer. She will report directly to CEO Marcus Chen.
Jennifer previously led the AI research team at DataTech Inc. in Boston.
The company's headquarters will remain in San Francisco.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You extract knowledge graph data from text.

Return ONLY valid JSON.

Format:
{
  "entities": [
    {"name": "Entity Name", "type": "Person/Organization/Place/Role", "description": "short description"}
  ],
  "relationships": [
    {"source": "Entity A", "relation": "RELATION_TYPE", "target": "Entity B"}
  ]
}

Use uppercase snake_case relation names.
""",
            ),
            (
                "human",
                """
Extract entities and relationships from this text:

{text}
""",
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "text": sample_text,
        }
    )

    raw_output = response.content.strip()

    print("\nInput Text:")
    print(sample_text.strip())

    print("\nRaw LLM Output:")
    print(raw_output)

    # Local models sometimes add extra text.
    # Try to parse JSON safely.
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)

        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if parsed:
        print("\nParsed Entities:")
        for entity in parsed.get("entities", []):
            print(f"- {entity}")

        print("\nParsed Relationships:")
        for relation in parsed.get("relationships", []):
            print(f"- {relation}")

    else:
        print("\nCould not parse JSON cleanly, but raw output is shown above.")

    return raw_output


# ============================================================
# GRAPHRAG ARCHITECTURE
# ============================================================

def show_graphrag_architecture():
    """
    Explains GraphRAG architecture.
    """

    print("\n" + "=" * 60)
    print("GRAPHRAG ARCHITECTURE")
    print("=" * 60)

    architecture = """
INDEXING PHASE:
------------------------------------------------------------
Documents
  ↓
Split documents
  ↓
LLM extracts entities and relationships
  ↓
Build knowledge graph
  ↓
Optional: detect communities/clusters
  ↓
Summarize graph communities


QUERY PHASE:
------------------------------------------------------------
User query
  ↓
Identify entities in query
  ↓
Traverse graph relationships
  ↓
Retrieve connected facts/documents
  ↓
LLM generates final answer


TWO TYPES OF GRAPHRAG SEARCH:
------------------------------------------------------------

1. Local Search:
   Used for specific relationship questions.

   Example:
   "Who reports to the CEO?"

2. Global Search:
   Used for broad summary questions.

   Example:
   "What are the main themes across all documents?"
"""

    print(architecture)


# ============================================================
# IMPLEMENTATION OPTIONS FOR YOUR STACK
# ============================================================

def show_implementation_options():
    """
    Practical implementation options for your stack.
    """

    print("\n" + "=" * 60)
    print("IMPLEMENTATION OPTIONS")
    print("=" * 60)

    options = """
OPTION 1: NetworkX Demo
------------------------------------------------------------
Best for learning.

Use:
- Python dictionaries
- NetworkX graph
- qwen3:8b for extraction

Good for tutorials and interviews.


OPTION 2: Chroma + Graph Hybrid
------------------------------------------------------------
Best for your current RAG stack.

Flow:
1. Use Chroma for semantic retrieval.
2. Use qwen3:8b to extract entities from retrieved docs.
3. Build small temporary graph.
4. Traverse graph for multi-hop answer.


OPTION 3: Neo4j Later
------------------------------------------------------------
Best for production graph storage.

Flow:
Documents -> Entity extraction -> Neo4j graph DB -> Cypher queries


OPTION 4: Microsoft GraphRAG Later
------------------------------------------------------------
Powerful but heavier.

Use when:
- many documents
- relationship-heavy domain
- global summaries needed


WHEN TO USE GRAPHRAG:
------------------------------------------------------------
Use it when:
✓ documents contain many relationships
✓ queries need multiple hops
✓ org charts / legal docs / research papers / finance relations
✓ user asks "who is connected to whom?"

Do not use it for:
✗ simple FAQ
✗ small docs
✗ basic semantic search
"""

    print(options)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.5: GRAPHRAG INTRODUCTION")
    print("=" * 60)

    demonstrate_multihop_problem()

    graph = build_knowledge_graph()

    traverse_graph_for_answer(graph)

    extract_entities_with_llm()

    show_graphrag_architecture()

    show_implementation_options()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)

    print(
        """
1. Traditional RAG is weak for multi-hop relationship questions.
2. GraphRAG represents entities as nodes and relationships as edges.
3. Graph traversal can answer questions that vector search may miss.
4. qwen3:8b can extract entities and relationships locally.
5. NetworkX is good for learning.
6. Later, use Neo4j or Microsoft GraphRAG for production.
"""
    )