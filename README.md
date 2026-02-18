# PDF Embeddings System

A production-grade system for processing multiple PDF documents and creating searchable embeddings using Azure OpenAI and ChromaDB.

## Features

- 📚 Process multiple PDF documents
- 🧠 Hierarchical document chunking with context summaries
- 🔍 Semantic search using vector embeddings
- 💾 Persistent storage with ChromaDB
- ⚙️ Easy configuration via environment variables

# Archive powershell 👇:
```
git archive --format=zip HEAD -o Chunking_with_LlamaIndex.zip
```

## Project Structure

```
pdf-embeddings-system/
│
├── config/
│   └── settings.py          # Configuration management
│
├── data/
│   ├── pdfs/                # Place your PDF files here
│   └── chroma_db/           # ChromaDB storage (auto-created)
│
├── src/
│   ├── __init__.py
│   ├── embeddings.py        # Embedding generation logic
│   ├── pdf_processor.py     # PDF loading and processing
│   ├── chunker.py           # Document chunking with summaries
│   └── query_engine.py      # Query and retrieval logic
│
├── scripts/
│   ├── process_pdfs.py      # Main script to process PDFs
│   └── query.py             # Script to query the system
│
├── tests/
│   └── test_basic.py        # Basic tests
│
├── .env.example             # Example environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with API access

### Setup Steps

1. **Clone or create the project directory**
```bash
mkdir pdf-embeddings-system
cd pdf-embeddings-system
```

2. **Create a virtual environment**
```bash
py -3.11 -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your Azure OpenAI credentials
```

5. **Add your PDF files**
```bash
# Place your PDF files in the data/pdfs/ directory
cp /path/to/your/files/*.pdf data/pdfs/
```

## Configuration

Edit the `.env` file with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_API_VERSION=2024-12-01-preview
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
```

## Usage

### Processing PDFs

Run the main processing script to create embeddings from your PDFs:

```bash
python scripts/process_pdfs.py
```

This will:
- Scan all PDFs in `data/pdfs/`
- Create a separate ChromaDB collection for each PDF
- Generate hierarchical chunks with context summaries
- Store embeddings for efficient retrieval

### Querying the System

Query a specific PDF collection:

```bash
```

Or use it programmatically:

```python
from src.query_engine import QueryEngine

# Initialize with collection name (PDF filename without extension)
engine = QueryEngine(collection_name="GMDSS_System-IOM_Manual")

# Ask questions
response = engine.query("How do I adjust the squelch level?")
print(response)
```

## How It Works

### 1. PDF Processing
- PDFs are loaded using PyMuPDF
- Text is extracted and combined into documents

### 2. Hierarchical Chunking
- Documents are split into hierarchical chunks (4096 → 1024 → 512 tokens)
- Parent chunks provide context for smaller child chunks

### 3. Summary Generation
- GPT-4o generates concise summaries for parent nodes
- Summaries provide context breadcrumbs for leaf nodes

### 4. Embedding Creation
- Each chunk is converted to a vector embedding using text-embedding-3-large
- Embeddings are stored in ChromaDB for fast retrieval

### 5. Querying
- User queries are converted to embeddings
- Similar chunks are retrieved using vector similarity
- Auto-merging combines related chunks for better context
- GPT-4o generates natural language responses

## Advanced Configuration

### Chunk Sizes
Modify chunk sizes in `src/chunker.py`:

```python
chunk_sizes = [4096, 1024, 512]  # Parent → Child hierarchy
```

### Retrieval Settings
Adjust retrieval parameters in `src/query_engine.py`:

```python
similarity_top_k = 6  # Number of chunks to retrieve
```

## Troubleshooting

### Rate Limit Errors
If you encounter rate limit errors:
- Reduce batch size in processing
- Add delays between API calls
- Upgrade your Azure OpenAI tier

### Out of Memory
For large PDFs:
- Reduce chunk sizes
- Process PDFs one at a time
- Increase system RAM

### Collection Not Found
Ensure PDFs are processed before querying:
```bash
python scripts/process_pdfs.py
```

## Cost Estimation

Approximate costs per PDF (200 pages):
- Text embedding: ~$0.50 - $1.00
- Summary generation: ~$2.00 - $4.00
- Total: ~$2.50 - $5.00 per document

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - feel free to use this for your projects!

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review Azure OpenAI documentation
3. Open an issue on GitHub

---

**Happy Embedding! 🚀**