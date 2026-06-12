"""
Lesson 6.6: Multimodal RAG
Local / Free Stack Version

Problem:
Text extraction from PDFs can destroy visual information.

Examples:
- tables become messy text
- charts lose meaning
- diagrams disappear
- layout is lost

Your Stack:
Text LLM     : qwen3:8b via Ollama
Vision LLM   : Not available locally in this setup
PDF/Text RAG : PyPDF + Chroma + Ollama embeddings
Image RAG    : Concept/demo only
Later Option : ColPali / LLaVA / Qwen2.5-VL
API Key      : Not required
"""

from dotenv import load_dotenv

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
# pip install langchain-ollama langchain-core python-dotenv
# ---------------------------------------------------------


llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)


# ============================================================
# THE PROBLEM: TEXT EXTRACTION LOSES VISUAL INFO
# ============================================================

def demonstrate_extraction_problem():
    """
    Shows why normal PDF text extraction may fail.

    Traditional RAG:
    PDF -> extract text -> chunk -> embed -> retrieve

    Problem:
    If PDF has tables/charts/diagrams, text extraction can lose structure.
    """

    print("=" * 60)
    print("THE PROBLEM: TEXT EXTRACTION DESTROYS INFORMATION")
    print("=" * 60)

    original_table = """
┌─────────────────────────────────────────────────────────┐
│              Q1 2025 Sales by Region                    │
├─────────────┬─────────────┬─────────────┬───────────────┤
│ Region      │ Q1 Target   │ Q1 Actual   │ Variance      │
├─────────────┼─────────────┼─────────────┼───────────────┤
│ North       │ $2.5M       │ $2.8M       │ +12% ✓        │
│ South       │ $1.8M       │ $1.5M       │ -17% ✗        │
│ East        │ $3.2M       │ $3.4M       │ +6% ✓         │
│ West        │ $2.1M       │ $2.0M       │ -5%           │
├─────────────┼─────────────┼─────────────┼───────────────┤
│ TOTAL       │ $9.6M       │ $9.7M       │ +1%           │
└─────────────┴─────────────┴─────────────┴───────────────┘
"""

    extracted_text = """
Q1 2025 Sales by Region Region Q1 Target Q1 Actual Variance
North $2.5M $2.8M +12% South $1.8M $1.5M -17%
East $3.2M $3.4M +6% West $2.1M $2.0M -5%
TOTAL $9.6M $9.7M +1%
"""

    print("\nOriginal table in PDF:")
    print(original_table)

    print("\nAfter bad text extraction:")
    print(extracted_text)

    print("\nWhat got lost?")
    print("""
1. Row/column structure
2. Visual indicators like checkmarks/cross marks
3. Alignment between region and numbers
4. Chart/diagram meaning
5. Layout context
""")


# ============================================================
# MULTIMODAL RAG EXPLANATION
# ============================================================

def explain_multimodal_rag():
    """
    Explains visual/document-image RAG.

    In pure local/free stack, we usually start with text RAG.
    For visual-heavy PDFs, we later add vision models.
    """

    print("\n" + "=" * 60)
    print("MULTIMODAL RAG APPROACH")
    print("=" * 60)

    print("""
TEXT RAG:
PDF
  ↓
Extract text using PyPDF
  ↓
Chunk text
  ↓
Embed using nomic-embed-text
  ↓
Store in Chroma
  ↓
Answer using qwen3:8b


MULTIMODAL RAG:
PDF
  ↓
Convert each page to image
  ↓
Embed page images using ColPali or vision embedding model
  ↓
Retrieve relevant page images
  ↓
Send page image to vision-capable LLM
  ↓
Answer using visual understanding


For your current stack:
- Use text RAG for normal PDFs.
- Use multimodal RAG only when PDFs contain tables, charts, forms, or diagrams.
""")


# ============================================================
# LOCAL SIMULATION OF VISION ANALYSIS
# ============================================================

def demo_document_visual_analysis_simulation():
    """
    Simulates vision LLM analysis using text description.

    Why simulation?
    qwen3:8b is text-only.
    It cannot directly see images.

    Later you can use:
    - Qwen2.5-VL
    - LLaVA
    - ColPali
    - GPT-4o / Claude if using paid APIs
    """

    print("\n" + "=" * 60)
    print("SIMULATED DOCUMENT VISUAL ANALYSIS")
    print("=" * 60)

    document_description = """
The page contains a sales table titled "Q1 2025 Sales by Region".

Rows:
- North: target $2.5M, actual $2.8M, variance +12%, green check
- South: target $1.8M, actual $1.5M, variance -17%, red cross
- East: target $3.2M, actual $3.4M, variance +6%, green check
- West: target $2.1M, actual $2.0M, variance -5%
- Total: target $9.6M, actual $9.7M, variance +1%

The red cross visually marks South as underperforming.
"""

    query = "Which region is underperforming and by how much?"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are analyzing a document page description.

Answer using only the provided document description.
Focus on numbers, visual indicators, and comparisons.
""",
            ),
            (
                "human",
                """
Document Description:
{document}

Question:
{query}

Answer:
""",
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "document": document_description,
            "query": query,
        }
    )

    print(f"\nQuery: {query}")
    print("\nAnswer:")
    print(response.content)


# ============================================================
# PRACTICAL LOCAL IMPLEMENTATION OPTIONS
# ============================================================

def show_local_implementation_options():
    """
    Shows practical options for your stack.
    """

    print("\n" + "=" * 60)
    print("LOCAL IMPLEMENTATION OPTIONS")
    print("=" * 60)

    print("""
OPTION 1: Text RAG only
------------------------------------------------------------
Use this for normal PDFs.

PDF -> PyPDFLoader -> chunks -> nomic-embed-text -> Chroma -> qwen3:8b

Best for:
- plain text PDFs
- articles
- docs
- simple reports


OPTION 2: OCR + Text RAG
------------------------------------------------------------
Use this for scanned PDFs.

PDF/Image -> Tesseract OCR -> text -> chunks -> Chroma -> qwen3:8b

Install later:
pip install pytesseract pillow pdf2image


OPTION 3: Table Extraction + Text RAG
------------------------------------------------------------
Use this for table-heavy PDFs.

PDF -> camelot/tabula/pdfplumber -> structured tables -> markdown text -> RAG

Good because markdown tables preserve row-column meaning better.


OPTION 4: Local Vision Model Later
------------------------------------------------------------
Use this when you want full visual understanding.

Possible models:
- qwen2.5-vl
- llava
- minicpm-v
- moondream

Flow:
PDF -> page images -> retrieve relevant image -> vision model answers


OPTION 5: ColPali Later
------------------------------------------------------------
Best for serious multimodal document retrieval.

But:
- needs GPU
- more complex setup
- not required for basic RAG learning
""")


# ============================================================
# SAMPLE PSEUDO CODE FOR FUTURE MULTIMODAL RAG
# ============================================================

def show_future_multimodal_pipeline_code():
    """
    Prints future implementation skeleton.

    Not executed now.
    This is for understanding architecture.
    """

    print("\n" + "=" * 60)
    print("FUTURE MULTIMODAL RAG PIPELINE SKELETON")
    print("=" * 60)

    code = r'''
# Future multimodal RAG flow

from pdf2image import convert_from_path
from PIL import Image

def pdf_to_images(pdf_path: str):
    """
    Converts PDF pages into images.
    """
    pages = convert_from_path(pdf_path, dpi=150)
    return pages


def extract_text_with_ocr(image: Image.Image):
    """
    OCR fallback using Tesseract.
    """
    import pytesseract
    return pytesseract.image_to_string(image)


def multimodal_rag_index(pdf_path: str):
    """
    Indexing idea:

    1. Convert PDF pages to images.
    2. OCR text if needed.
    3. Store:
       - page number
       - OCR text
       - image path
       - metadata
    4. Embed OCR text using nomic-embed-text.
    5. Store vectors in Chroma.
    """
    images = pdf_to_images(pdf_path)

    indexed_pages = []

    for page_number, image in enumerate(images, start=1):
        text = extract_text_with_ocr(image)

        indexed_pages.append({
            "page_number": page_number,
            "text": text,
            "image": image,
        })

    return indexed_pages


def answer_with_local_vision_model(query: str, image_path: str):
    """
    Later you can replace this with:
    - Qwen2.5-VL
    - LLaVA
    - MiniCPM-V
    """
    pass
'''

    print(code)


# ============================================================
# USE CASES
# ============================================================

def show_use_cases():
    """
    Explains when multimodal RAG is useful.
    """

    print("\n" + "=" * 60)
    print("WHEN TO USE MULTIMODAL RAG")
    print("=" * 60)

    print("""
USE MULTIMODAL RAG FOR:
------------------------------------------------------------
✓ financial reports with tables/charts
✓ technical docs with architecture diagrams
✓ scientific papers with figures
✓ scanned PDFs
✓ invoices/forms
✓ legal docs with complex formatting
✓ dashboards exported as PDFs


NORMAL TEXT RAG IS ENOUGH FOR:
------------------------------------------------------------
✓ plain articles
✓ books
✓ simple PDFs
✓ markdown docs
✓ code documentation
✓ FAQ pages


MY RECOMMENDATION FOR YOU:
------------------------------------------------------------
Start with:
1. PyPDFLoader
2. RecursiveCharacterTextSplitter
3. nomic-embed-text
4. Chroma
5. qwen3:8b

Then add:
1. pdfplumber for tables
2. pytesseract for scanned PDFs
3. local vision model later
""")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.6: MULTIMODAL RAG")
    print("=" * 60)

    demonstrate_extraction_problem()

    explain_multimodal_rag()

    demo_document_visual_analysis_simulation()

    show_local_implementation_options()

    show_future_multimodal_pipeline_code()

    show_use_cases()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)

    print(
        """
1. Text extraction can lose visual information.
2. Tables, charts, and diagrams are weak points for normal RAG.
3. Your current qwen3:8b stack is text-only.
4. Start with text RAG first.
5. Add OCR/table extraction for practical improvement.
6. Add true vision RAG later only when needed.
"""
    )