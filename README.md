# PDF Embeddings System

A production-grade RAG (Retrieval-Augmented Generation) system for querying PDF documents with automatic page tracking and clickable source citations.

## Overview

This system allows you to:
1. **Upload PDFs** - Process multiple PDF documents
2. **Ask Questions** - Query in natural language
3. **Get Answers** - Receive AI-generated responses
4. **See Sources** - Click links to exact pages in source PDFs

## Key Features

- 📄 **Multi-PDF Support** - Process unlimited PDF documents
- 🔍 **Semantic Search** - Find relevant information across all documents
- 📌 **Page Citations** - Every answer includes source page numbers
- 🔗 **Clickable Links** - Direct links to exact pages in PDFs
- 🧠 **Context-Aware** - Hierarchical chunking with parent summaries
- 🎯 **Production-Ready** - Comprehensive logging, error handling, type safety

## Quick Start

### 1. Installation

```bash
# Clone or create project directory
mkdir pdf-embeddings-system
cd pdf-embeddings-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your Azure OpenAI credentials:

```bash
# Copy example file
cp .env.example .env

# Edit with your credentials
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

### 3. Add PDFs

```bash
# Copy your PDF files to the data directory
cp /path/to/your/pdfs/*.pdf data/pdfs/
```

### 4. Process PDFs

```bash
# Run the processing pipeline
python scripts/process_pdfs.py
```

**What this does:**
- Loads PDFs page-by-page
- Creates hierarchical chunks (4096 → 1024 → 512 tokens)
- Generates context summaries for parent sections
- Creates embeddings using Azure OpenAI
- Stores everything in ChromaDB with page metadata

**Time estimate:** ~5 minutes per 100-page PDF

### 5. Query Your PDFs

```bash
# Start interactive query interface
python scripts/query.py
```

**Example session:**
```
Available collections:
  0. Search ALL collections
  1. user_manual

Select a collection (0-1): 1

✓ Connected to collection: user_manual

Query: How do I reset the device?

Searching...

To reset the device, press and hold the reset button for 3 seconds 
until the LED flashes red. Release the button and wait for the device 
to reboot.

================================================================================
📄 SOURCES: user_manual.pdf
================================================================================

  🔗 Page 87
     file:///C:/path/to/data/pdfs/user_manual.pdf#page=87

  🔗 Pages 112-114
     file:///C:/path/to/data/pdfs/user_manual.pdf#page=112

================================================================================
💡 Tip: Ctrl+Click the blue links to open PDF at exact page
```

**Ctrl+Click the blue link** → PDF opens at that exact page! ✨

## Project Structure

```
pdf-embeddings-system/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration management
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py            # PDF loading (per-page documents)
│   ├── metadata_manager.py      # Page tracking & link generation
│   ├── chunker.py               # Hierarchical chunking
│   ├── embeddings.py            # Azure OpenAI integration
│   ├── storage_manager.py       # ChromaDB + docstore management
│   ├── retriever.py             # Smart retrieval
│   └── source_formatter.py      # Source citation formatting
│
├── scripts/
│   ├── process_pdfs.py          # Main processing pipeline
│   └── query.py                 # Interactive query interface
│
├── data/
│   ├── pdfs/                    # Your PDF files (input)
│   ├── chroma_db/               # Vector database (auto-created)
│   └── docstore/                # Node storage (auto-created)
│
├── tests/
│   └── test_basic.py            # Basic tests
│
├── .env                         # Your configuration (create from .env.example)
├── .env.example                 # Configuration template
├── .gitignore
├── requirements.txt
└── README.md
```

## How It Works

### Architecture Overview

```
1. PDF Loading (Per-Page Documents)
   └─> Each page becomes a Document with metadata
       {page: 5, filename: 'manual.pdf', file_path: '/path/...'}

2. Hierarchical Chunking
   └─> Documents split into parent/child chunks
       Metadata automatically inherited by all chunks

3. Parent Summaries
   └─> LLM generates summaries of parent sections

4. Context Enrichment
   └─> Leaf chunks get context: [CONTEXT: summary → summary] + content

5. Embedding & Storage
   └─> ChromaDB: vector embeddings for search
       Docstore: all nodes for auto-merging

6. Query & Retrieval
   └─> Query → Semantic search → Auto-merge → LLM → Answer

7. Source Citation
   └─> Extract pages → Merge ranges → Generate clickable links
```

### Key Innovation: Automatic Metadata Inheritance

**Instead of manual tracking:**
```python
# ❌ Fragile approach
text = "[PAGE_5] content here..."
page = extract_from_text(text)  # Can break easily
```

**We use automatic inheritance:**
```python
# ✅ Robust approach
doc = Document(
    text="content here",
    metadata={'page': 5, 'filename': 'manual.pdf', 'file_path': '/path/...'}
)

chunks = chunker.chunk(doc)
# All chunks automatically inherit: page=5, filename='manual.pdf'
```

LlamaIndex automatically propagates metadata from documents to all chunks - no manual tracking needed!

## Configuration

All settings are in `.env`:

```env
# Azure OpenAI (Required)
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Embeddings
EMBEDDING_DIMENSIONS=3072

# Chunking (sizes in tokens, largest to smallest)
CHUNK_SIZES=4096,1024,512

# Retrieval
SIMILARITY_TOP_K=6
ENABLE_AUTO_MERGING=true

# Paths
PDF_DIRECTORY=data/pdfs
CHROMA_DB_PATH=data/chroma_db
DOCSTORE_PATH=data/docstore

# Logging
LOG_LEVEL=INFO
```

## Usage Examples

### Interactive Mode

```bash
python scripts/query.py
```

### Command-Line Mode

```bash
# Query all collections
python scripts/query.py "How do I troubleshoot errors?"

# Query specific collection
python scripts/query.py manual_collection "How do I reset the device?"
```

### Programmatic Usage

```python
from src import PDFLoader, DocumentChunker, StorageManager, SmartRetriever
from src.embeddings import EmbeddingsManager
from pathlib import Path

# Initialize components
loader = PDFLoader()
embeddings_mgr = EmbeddingsManager()
chunker = DocumentChunker(embeddings_mgr.get_llm())
storage = StorageManager()

# Load and process PDF
docs, collection_name = loader.load_pdf(Path("manual.pdf"))
all_nodes, enriched = chunker.process_documents(docs)
storage.save_collection(collection_name, all_nodes, enriched, embeddings_mgr.get_embed_model())

# Query
retriever = SmartRetriever(collection_name)
response = retriever.query("How do I reset?")

print(response.answer)
print(f"Pages: {response.source_nodes}")
```

## Advanced Features

### Auto-Merging Retrieval

When enabled (`ENABLE_AUTO_MERGING=true`), the system automatically combines related chunks during retrieval for better context.

**How it works:**
- Retrieves similar leaf chunks
- Checks if they share parent chunks
- Merges related chunks together
- Provides more complete context to LLM

### Multiple Output Formats

```python
from src.source_formatter import SourceFormatter

formatter = SourceFormatter()

# Terminal (colored, clickable)
print(formatter.format_for_terminal(nodes))

# Plain text (for logs)
print(formatter.format_for_plain_text(nodes))

# JSON (for APIs)
data = formatter.format_for_json(nodes)

# HTML (for web)
html = formatter.format_for_html(nodes)
```

### Structured Responses

```python
from src.retriever import SmartRetriever

retriever = SmartRetriever("manual")
response = retriever.query("How do I reset?")

# Response is a QueryResponse object with:
response.answer              # str: The answer text
response.source_nodes        # List: Retrieved chunks
response.collection_name     # str: Which collection
response.retrieval_successful # bool: Success status
response.error_message       # str: Error if failed
```

## Testing

```bash
# Run basic tests
python tests/test_basic.py

# Test individual components
python src/pdf_loader.py
python src/metadata_manager.py
python src/chunker.py
python src/storage_manager.py
python src/retriever.py
python src/source_formatter.py
```

## Troubleshooting

### "No PDFs found"
**Solution:** 
- Ensure PDFs are in `data/pdfs/` directory
- Check files have `.pdf` extension
- Verify files aren't empty

### "Configuration Error"
**Solution:**
- Verify `.env` file exists
- Check Azure OpenAI credentials are correct
- Run `python config/settings.py` to test configuration

### "Collection not found"
**Solution:**
- Run `python scripts/process_pdfs.py` first
- Collections are named after PDF filenames (e.g., `manual.pdf` → `manual`)

### "No page information available"
**Solution:**
- PDFs need to be reprocessed with the new system
- Delete old collections and re-run `python scripts/process_pdfs.py`

### "Rate limit error"
**Solution:**
- Azure OpenAI has rate limits on your tier
- Wait a few seconds and retry
- Consider upgrading your Azure tier
- Process PDFs one at a time if needed

### Links don't open
**Solution:**
- Some terminals don't support clickable links
- Try: Ctrl+Click (Windows/Linux) or Cmd+Click (Mac)
- Copy-paste the URL into your browser manually
- Use a modern terminal (Windows Terminal, iTerm2, etc.)

## Cost Estimation

**Per 100-page PDF:**
- Text embeddings: ~$0.50
- Summary generation: ~$1.50
- **Total:** ~$2.00 per document (one-time processing cost)

**Querying:**
- ~$0.01 per query

**Note:** Costs vary based on Azure OpenAI tier and usage.

## Performance

**Processing Speed:**
- ~5 minutes per 100-page PDF
- Parallel processing available for multiple PDFs

**Query Speed:**
- ~1-2 seconds per query
- Depends on similarity_top_k setting

**Storage:**
- ~10MB per 100-page PDF (embeddings + metadata)

## Production Deployment

### Scalability Options

1. **Vector Store:** Upgrade from ChromaDB to Pinecone/Weaviate/Qdrant for production scale
2. **Docstore:** Move from JSON files to MongoDB/PostgreSQL
3. **Caching:** Add Redis for query result caching
4. **Processing:** Use Celery for background PDF processing
5. **Monitoring:** Integrate with DataDog, New Relic, or Prometheus

### Security Considerations

- Store `.env` securely (never commit to git)
- Use Azure Key Vault for production credentials
- Implement rate limiting for API endpoints
- Add user authentication for multi-tenant deployments

## Requirements

- Python 3.8+
- Azure OpenAI account with API access
- 2GB+ free disk space
- Internet connection for API calls

## Dependencies

See `requirements.txt` for full list. Main dependencies:

- `llama-index-core` - RAG framework
- `llama-index-llms-azure-openai` - Azure OpenAI integration
- `llama-index-embeddings-azure-openai` - Embedding models
- `llama-index-vector-stores-chroma` - ChromaDB integration
- `chromadb` - Vector database
- `pymupdf` - PDF processing
- `python-dotenv` - Environment management
- `tqdm` - Progress bars
- `colorama` - Terminal colors

## Contributing

Contributions welcome! This is a production-grade template designed to be extended.

**Areas for enhancement:**
- Additional vector store backends
- Web interface (FastAPI + React)
- Advanced retrieval strategies
- Query analytics and monitoring
- Multi-language support

## License

MIT License - Free to use for personal and commercial projects.

## Acknowledgments

Built with:
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) - LLM and embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing

## Support

- **Documentation:** See this README and `PRODUCTION_SETUP.md`
- **Issues:** Check troubleshooting section above
- **Configuration:** Run `python config/settings.py` to validate setup
- **Testing:** Run component tests to isolate issues

## Changelog

### Version 1.0.0 (Production)
- ✅ Per-page document loading with automatic metadata
- ✅ Hierarchical chunking with parent summaries
- ✅ Automatic page tracking throughout pipeline
- ✅ Clickable page links to source PDFs
- ✅ Smart retrieval with auto-merging
- ✅ Multi-format source citations
- ✅ Production-grade error handling and logging
- ✅ Comprehensive documentation

---

**Ready to query your PDFs with AI! 🚀**

For detailed setup instructions, see `PRODUCTION_SETUP.md`