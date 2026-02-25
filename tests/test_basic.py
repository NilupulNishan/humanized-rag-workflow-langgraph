"""
Basic tests for the PDF Embeddings System.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings


def test_configuration():
    """Test that configuration is loadable."""
    assert settings.PROJECT_ROOT.exists()
    assert settings.PDF_DIRECTORY.exists()
    assert settings.CHROMA_DB_PATH.exists()


def test_chunk_sizes():
    """Test that chunk sizes are properly configured."""
    assert len(settings.CHUNK_SIZES) > 0
    assert all(isinstance(size, int) for size in settings.CHUNK_SIZES)
    assert settings.CHUNK_SIZES == sorted(settings.CHUNK_SIZES, reverse=True)


def test_pdf_processor_import():
    """Test that PDFProcessor can be imported."""
    from src.pdf_loader import PDFProcessor
    processor = PDFProcessor()
    assert processor is not None


def test_collection_name_generation():
    """Test collection name generation from filenames."""
    from src.pdf_loader import PDFProcessor
    
    processor = PDFProcessor()
    
    # Test various filenames
    test_cases = [
        (Path("Test Document.pdf"), "test_document"),
        (Path("GMDSS_Manual-v2.pdf"), "gmdss_manual_v2"),
        (Path("file with spaces.pdf"), "file_with_spaces"),
        (Path("Special!@#$%Chars.pdf"), "special_____chars"),
    ]
    
    for pdf_path, expected in test_cases:
        result = processor.get_collection_name(pdf_path)
        assert result == expected, f"Expected {expected}, got {result}"


if __name__ == "__main__":
    # Run tests
    print("Running basic tests...")
    
    try:
        test_configuration()
        print("✓ Configuration test passed")
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
    
    try:
        test_chunk_sizes()
        print("✓ Chunk sizes test passed")
    except Exception as e:
        print(f"✗ Chunk sizes test failed: {e}")
    
    try:
        test_pdf_processor_import()
        print("✓ PDF processor import test passed")
    except Exception as e:
        print(f"✗ PDF processor import test failed: {e}")
    
    try:
        test_collection_name_generation()
        print("✓ Collection name generation test passed")
    except Exception as e:
        print(f"✗ Collection name generation test failed: {e}")
    
    print("\nAll basic tests completed!")