"""
Production-grade hierarchical document chunker with automatic metadata inheritance.
"""
import logging
from typing import List, Dict
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import TextNode, NodeRelationship
from config import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Hierarchical document chunking with context summaries.
    
    Key Feature: Metadata from Document automatically flows to all chunks!
    No manual page tracking needed.
    """
    
    def __init__(self, llm, chunk_sizes: List[int] = None):
        """
        Initialize document chunker.
        
        Args:
            llm: Language model for generating summaries
            chunk_sizes: List of chunk sizes for hierarchy (largest to smallest)
        """
        self.llm = llm
        self.chunk_sizes = chunk_sizes or settings.CHUNK_SIZES
        
        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes
        )
        
        logger.info(f"DocumentChunker initialized with chunk sizes: {self.chunk_sizes}")
    
    def create_nodes(self, documents: List[Document]) -> List:
        """
        Parse documents into hierarchical nodes.
        Metadata automatically propagates from documents to all chunks!
        
        Args:
            documents: List of documents to parse
            
        Returns:
            List of all nodes (parent and leaf)
        """
        logger.info(f"Creating hierarchical nodes from {len(documents)} documents...")
        
        nodes = self.parser.get_nodes_from_documents(documents)
        
        leaf_count = len(get_leaf_nodes(nodes))
        parent_count = len(nodes) - leaf_count
        
        logger.info(f"  Created {len(nodes)} total nodes")
        logger.info(f"    Parent nodes: {parent_count}")
        logger.info(f"    Leaf nodes: {leaf_count}")
        
        return nodes
    
    def generate_parent_summaries(self, nodes: List) -> Dict[str, str]:
        """
        Generate concise summaries for parent nodes.
        
        Args:
            nodes: List of all nodes
            
        Returns:
            Dictionary mapping node_id to summary
        """
        summaries = {}
        parent_nodes = [n for n in nodes if NodeRelationship.CHILD in n.relationships]
        
        if not parent_nodes:
            logger.info("  No parent nodes to summarize")
            return summaries
        
        logger.info(f"Generating summaries for {len(parent_nodes)} parent nodes...")
        
        for i, node in enumerate(parent_nodes):
            if i > 0 and i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(parent_nodes)}")
            
            prompt = f"""Provide a concise summary (2-3 sentences, max 100 tokens) of this text section:

{node.get_content()[:3000]}

Summary:"""
            
            try:
                response = self.llm.complete(prompt)
                summaries[node.node_id] = response.text.strip()
            except Exception as e:
                logger.warning(f"Failed to generate summary for node {node.node_id}: {e}")
                summaries[node.node_id] = node.get_content()[:150] + "..."
        
        logger.info(f"  ✓ Generated {len(summaries)} summaries")
        return summaries
    
    def enrich_leaf_nodes(self, nodes: List, parent_summaries: Dict[str, str]) -> List[TextNode]:
        """
        Add parent context to leaf nodes.
        Metadata is already inherited - we just add context breadcrumbs.
        
        Args:
            nodes: List of all nodes
            parent_summaries: Dictionary of parent summaries
            
        Returns:
            List of enriched leaf nodes
        """
        leaf_nodes = get_leaf_nodes(nodes)
        enriched_nodes = []
        
        logger.info(f"Enriching {len(leaf_nodes)} leaf nodes with parent context...")
        
        for leaf in leaf_nodes:
            # Build hierarchy chain
            hierarchy_chain = []
            current = leaf
            
            # Traverse up the parent chain
            while NodeRelationship.PARENT in current.relationships:
                parent_id = current.relationships[NodeRelationship.PARENT].node_id
                parent_node = next((n for n in nodes if n.node_id == parent_id), None)
                
                if parent_node and parent_node.node_id in parent_summaries:
                    hierarchy_chain.insert(0, parent_summaries[parent_node.node_id])
                
                current = parent_node
                if current is None:
                    break
            
            # Create enriched content with context breadcrumbs
            if hierarchy_chain:
                context_str = " → ".join(hierarchy_chain)
                enriched_content = f"[CONTEXT: {context_str}]\n\n{leaf.get_content()}"
            else:
                enriched_content = leaf.get_content()
            
            # Create new enriched node
            # Metadata is automatically inherited from parent document!
            enriched_node = TextNode(
                text=enriched_content,
                metadata={
                    **leaf.metadata,  # Already has page, filename, file_path!
                    'hierarchy_depth': len(hierarchy_chain),
                    'has_context': len(hierarchy_chain) > 0,
                    'original_node_id': leaf.node_id,
                },
                relationships=leaf.relationships
            )
            enriched_node.node_id = leaf.node_id
            
            enriched_nodes.append(enriched_node)
        
        logger.info(f"  ✓ Enriched {len(enriched_nodes)} nodes")
        
        # Validate metadata
        valid_count = sum(1 for node in enriched_nodes if 'page' in node.metadata)
        logger.info(f"  ✓ {valid_count}/{len(enriched_nodes)} nodes have page metadata")
        
        return enriched_nodes
    
    def process_documents(self, documents: List[Document]) -> tuple:
        """
        Complete processing pipeline for documents.
        
        Args:
            documents: Documents to process (each with page metadata)
            
        Returns:
            Tuple of (all_nodes, enriched_leaf_nodes)
        """
        logger.info(f"Processing {len(documents)} documents...")
        
        # Create hierarchical nodes (metadata flows automatically!)
        nodes = self.create_nodes(documents)
        
        # Generate parent summaries
        parent_summaries = self.generate_parent_summaries(nodes)
        
        # Enrich leaf nodes with context
        enriched_leaf_nodes = self.enrich_leaf_nodes(nodes, parent_summaries)
        
        logger.info("✓ Document processing complete")
        
        return nodes, enriched_leaf_nodes


if __name__ == "__main__":
    # Test chunker
    from src.embeddings import EmbeddingsManager
    
    logger.info("Testing DocumentChunker...")
    
    # Initialize
    embeddings_manager = EmbeddingsManager()
    chunker = DocumentChunker(embeddings_manager.get_llm())
    
    # Create test documents with page metadata
    test_docs = [
        Document(
            text="This is page 1 content. " * 100,
            metadata={'page': 1, 'filename': 'test.pdf', 'file_path': '/test.pdf'}
        ),
        Document(
            text="This is page 2 content. " * 100,
            metadata={'page': 2, 'filename': 'test.pdf', 'file_path': '/test.pdf'}
        ),
    ]
    
    # Process
    all_nodes, enriched = chunker.process_documents(test_docs)
    
    print(f"\nTest Results:")
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Enriched leaf nodes: {len(enriched)}")
    print(f"\nFirst enriched node metadata:")
    print(f"  {enriched[0].metadata}")
    print(f"\nMetadata has page? {'page' in enriched[0].metadata}")