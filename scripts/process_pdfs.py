"""
Production-grade PDF processing pipeline.
"""
import sys
import logging
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from src.pdf_loader import PDFLoader
from src.embeddings import EmbeddingsManager
from src.chunker import DocumentChunker
from src.storage_manager import StorageManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_pdf(
    page_documents, 
    collection_name, 
    pdf_path, 
    chunker, 
    storage_manager,
    embed_model
):
    """
    Process a single PDF through the complete pipeline.
    
    Args:
        page_documents: List of per-page documents
        collection_name: Name for the collection
        pdf_path: Path to PDF file
        chunker: DocumentChunker instance
        storage_manager: StorageManager instance
        embed_model: Embedding model
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"{'='*80}")
    
    try:
        # Process documents (metadata automatically flows!)
        all_nodes, enriched_nodes = chunker.process_documents(page_documents)
        
        # Save to storage
        success = storage_manager.save_collection(
            collection_name,
            all_nodes,
            enriched_nodes,
            embed_model
        )
        
        if success:
            logger.info(f"✓ Successfully processed {pdf_path.name}")
            return True
        else:
            logger.error(f"✗ Failed to save {pdf_path.name}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error processing {pdf_path.name}: {e}", exc_info=True)
        return False


def main():
    """Main processing function."""
    print("\n" + "="*80)
    print("PDF EMBEDDINGS PROCESSING - PRODUCTION")
    print("="*80)
    
    # Validate configuration
    try:
        settings.validate_config()
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease set up your .env file with Azure OpenAI credentials.")
        return 1
    
    # Initialize components
    logger.info("\nInitializing components...")
    print("\nInitializing components...")
    
    try:
        pdf_loader = PDFLoader()
        embeddings_manager = EmbeddingsManager()
        chunker = DocumentChunker(embeddings_manager.get_llm())
        storage_manager = StorageManager()
        
        print("✓ All components initialized")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        print(f"\n❌ Initialization Error: {e}")
        return 1
    
    # Load PDFs
    logger.info("\nLoading PDFs...")
    print("\nLoading PDFs...")
    
    try:
        pdf_data = pdf_loader.load_all_pdfs(settings.PDF_DIRECTORY)
    except ValueError as e:
        logger.error(f"PDF loading failed: {e}")
        print(f"\n❌ Error: {e}")
        print(f"\nPlease add PDF files to: {settings.PDF_DIRECTORY}")
        return 1
    
    if not pdf_data:
        logger.error("No PDFs were successfully loaded")
        print("\n❌ No PDFs were successfully loaded")
        return 1
    
    # Process each PDF
    logger.info(f"\nProcessing {len(pdf_data)} PDF(s)")
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(pdf_data)} PDF(S)")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    
    for i, (page_documents, collection_name, pdf_path) in enumerate(pdf_data, 1):
        print(f"[{i}/{len(pdf_data)}] Processing {pdf_path.name}...")
        
        try:
            success = process_single_pdf(
                page_documents,
                collection_name,
                pdf_path,
                chunker,
                storage_manager,
                embeddings_manager.get_embed_model()
            )
            
            if success:
                successful += 1
                print(f"✓ {pdf_path.name} completed\n")
            else:
                failed += 1
                print(f"✗ {pdf_path.name} failed\n")
                
        except Exception as e:
            logger.error(f"Unexpected error processing {pdf_path.name}: {e}", exc_info=True)
            print(f"✗ {pdf_path.name} failed: {e}\n")
            failed += 1
            continue
    
    # Summary
    print(f"{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Successful: {successful}")
    if failed > 0:
        print(f"✗ Failed: {failed}")
    
    # Show available collections
    print(f"\nAvailable collections:")
    collections = storage_manager.list_collections()
    
    for coll in collections:
        info = storage_manager.get_collection_info(coll)
        docstore_status = "✓" if info.get('has_docstore') else "✗"
        print(f"  {docstore_status} {coll} ({info.get('count', 0)} chunks)")
    
    print(f"\n💡 Query these collections using: python scripts/query.py")
    
    logger.info("Processing pipeline completed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)