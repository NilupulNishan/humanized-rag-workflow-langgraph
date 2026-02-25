"""
Unified storage management for ChromaDB and docstore.
"""
import json
import logging
from typing import List, Optional
from pathlib import Path
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages all storage operations: ChromaDB + Docstore.
    
    Features:
    - Create/delete collections
    - Persist docstore to disk
    - Load collections with docstore
    - Handle auto-merging setup
    """
    
    def __init__(self, chroma_path: Path = None, docstore_path: Path = None):
        """
        Initialize storage manager.
        
        Args:
            chroma_path: Path to ChromaDB directory
            docstore_path: Path to docstore directory
        """
        self.chroma_path = chroma_path or settings.CHROMA_DB_PATH
        self.docstore_path = docstore_path or settings.DOCSTORE_PATH
        
        # Ensure paths exist
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.docstore_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        logger.info(f"StorageManager initialized")
        logger.debug(f"  ChromaDB: {self.chroma_path}")
        logger.debug(f"  Docstore: {self.docstore_path}")
    
    def get_docstore_path(self, collection_name: str) -> Path:
        """Get path for a collection's docstore file."""
        return self.docstore_path / f"{collection_name}_docstore.json"
    
    def save_docstore(self, docstore: SimpleDocumentStore, collection_name: str) -> bool:
        """
        Save docstore to disk as JSON.
        
        Args:
            docstore: Document store to save
            collection_name: Name of the collection
            
        Returns:
            True if successful, False otherwise
        """
        docstore_file = self.get_docstore_path(collection_name)
        
        try:
            # Convert docstore to JSON
            docs_dict = {}
            for doc_id, doc in docstore.docs.items():
                docs_dict[doc_id] = doc.to_dict()
            
            # Save to file
            with open(docstore_file, 'w', encoding='utf-8') as f:
                json.dump(docs_dict, f, indent=2)
            
            logger.info(f"  ✓ Saved docstore: {docstore_file.name}")
            logger.debug(f"    Documents: {len(docs_dict)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save docstore for {collection_name}: {e}")
            return False
    
    def load_docstore(self, collection_name: str) -> Optional[SimpleDocumentStore]:
        """
        Load docstore from disk.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Loaded docstore or None if not found
        """
        docstore_file = self.get_docstore_path(collection_name)
        
        if not docstore_file.exists():
            logger.warning(f"Docstore not found: {docstore_file.name}")
            return None
        
        try:
            docstore = SimpleDocumentStore()
            
            with open(docstore_file, 'r', encoding='utf-8') as f:
                docs_dict = json.load(f)
            
            # Reconstruct nodes from dict
            for doc_id, doc_dict in docs_dict.items():
                node = TextNode.from_dict(doc_dict)
                docstore.add_documents([node])
            
            logger.info(f"  ✓ Loaded docstore: {docstore_file.name}")
            logger.debug(f"    Documents: {len(docs_dict)}")
            
            return docstore
            
        except Exception as e:
            logger.error(f"Failed to load docstore for {collection_name}: {e}")
            return None
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.chroma_client.get_collection(collection_name)
            return True
        except:
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and its docstore.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if successful
        """
        try:
            # Delete ChromaDB collection
            if self.collection_exists(collection_name):
                self.chroma_client.delete_collection(collection_name)
                logger.info(f"  ✓ Deleted ChromaDB collection: {collection_name}")
            
            # Delete docstore file
            docstore_file = self.get_docstore_path(collection_name)
            if docstore_file.exists():
                docstore_file.unlink()
                logger.info(f"  ✓ Deleted docstore: {docstore_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    def save_collection(
        self, 
        collection_name: str, 
        all_nodes: List[BaseNode], 
        enriched_nodes: List[BaseNode],
        embed_model
    ) -> bool:
        """
        Save a complete collection: vectors + docstore.
        
        Args:
            collection_name: Name of the collection
            all_nodes: All nodes (for docstore/auto-merging)
            enriched_nodes: Enriched leaf nodes (for indexing)
            embed_model: Embedding model
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Saving collection: {collection_name}")
            
            # Delete existing collection if it exists
            if self.collection_exists(collection_name):
                self.delete_collection(collection_name)
            
            # Create ChromaDB collection
            chroma_collection = self.chroma_client.create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create docstore with all nodes
            docstore = SimpleDocumentStore()
            docstore.add_documents(all_nodes)
            
            # Save docstore to disk
            self.save_docstore(docstore, collection_name)
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=docstore
            )
            
            # Create index
            logger.info(f"  Creating embeddings for {len(enriched_nodes)} nodes...")
            index = VectorStoreIndex(
                enriched_nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=False
            )
            
            logger.info(f"  ✓ Collection saved: {collection_name}")
            logger.debug(f"    Total nodes: {len(all_nodes)}")
            logger.debug(f"    Indexed nodes: {len(enriched_nodes)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save collection {collection_name}: {e}")
            return False
    
    def load_index(
        self, 
        collection_name: str, 
        embed_model,
        enable_auto_merging: bool = True
    ):
        """
        Load an index from storage.
        
        Args:
            collection_name: Name of collection to load
            embed_model: Embedding model
            enable_auto_merging: Whether to load docstore for auto-merging
            
        Returns:
            Tuple of (index, storage_context, has_docstore)
            
        Raises:
            ValueError: If collection not found
        """
        try:
            # Load ChromaDB collection
            chroma_collection = self.chroma_client.get_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Try to load docstore if auto-merging enabled
            docstore = None
            if enable_auto_merging:
                docstore = self.load_docstore(collection_name)
            
            # Create storage context
            if docstore:
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    docstore=docstore
                )
            else:
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
            
            # Load index
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
                embed_model=embed_model
            )
            
            logger.info(f"  ✓ Loaded collection: {collection_name}")
            logger.debug(f"    Has docstore: {docstore is not None}")
            
            return index, storage_context, docstore is not None
            
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {e}")
            raise ValueError(f"Collection '{collection_name}' not found") from e
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        collections = self.chroma_client.list_collections()
        return [col.name for col in collections]
    
    def get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Dictionary with collection info
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            docstore_file = self.get_docstore_path(collection_name)
            
            info = {
                'name': collection_name,
                'count': collection.count(),
                'has_docstore': docstore_file.exists(),
                'docstore_path': str(docstore_file) if docstore_file.exists() else None,
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get info for {collection_name}: {e}")
            return {'name': collection_name, 'error': str(e)}


if __name__ == "__main__":
    # Test storage manager
    storage = StorageManager()
    
    print("Testing StorageManager:\n")
    
    # List collections
    collections = storage.list_collections()
    print(f"1. Available collections: {collections}")
    
    # Get info for each collection
    for coll in collections:
        info = storage.get_collection_info(coll)
        print(f"\n2. Collection info for '{coll}':")
        print(f"   {info}")