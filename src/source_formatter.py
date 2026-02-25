"""
Production-grade source formatting for multiple output formats.
"""
import logging
from typing import List
from colorama import Fore, Style
from src.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)


class SourceFormatter:
    """
    Formats source citations with page links for multiple outputs.
    
    Supports:
    - Terminal (colored, clickable links)
    - Plain text (for logging)
    - JSON (for API responses)
    """
    
    def __init__(self):
        """Initialize source formatter."""
        self.metadata_manager = MetadataManager()
        logger.debug("SourceFormatter initialized")
    
    def format_for_terminal(self, nodes: List, show_tips: bool = True) -> str:
        """
        Format sources for terminal display with colors and clickable links.
        
        Args:
            nodes: List of retrieved nodes
            show_tips: Whether to show usage tips
            
        Returns:
            Formatted colored string for terminal
        """
        # Extract metadata
        pages = self.metadata_manager.extract_pages_from_nodes(nodes)
        
        if not pages:
            return f"\n{Fore.YELLOW}📄 No page information available{Style.RESET_ALL}"
        
        file_path = self.metadata_manager.extract_file_path_from_nodes(nodes)
        filename = self.metadata_manager.extract_filename_from_nodes(nodes)
        ranges = self.metadata_manager.merge_consecutive_pages(pages)
        
        # Build formatted output
        output = f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n"
        output += f"{Fore.GREEN}📄 SOURCES: {filename}{Style.RESET_ALL}\n"
        output += f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n\n"
        
        for start, end in ranges:
            page_text = self.metadata_manager.format_page_range(start, end)
            
            if file_path:
                link = self.metadata_manager.generate_file_url(file_path, start)
                output += f"{Fore.YELLOW}  🔗 {page_text}{Style.RESET_ALL}\n"
                output += f"     {Fore.BLUE}{link}{Style.RESET_ALL}\n\n"
            else:
                output += f"{Fore.YELLOW}  📄 {page_text}{Style.RESET_ALL}\n\n"
        
        output += f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n"
        
        if show_tips:
            output += f"{Fore.YELLOW}💡 Tip: Ctrl+Click the blue links to open PDF at exact page{Style.RESET_ALL}\n"
        
        return output
    
    def format_for_plain_text(self, nodes: List) -> str:
        """
        Format sources as plain text (no colors).
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            Plain text formatted string
        """
        pages = self.metadata_manager.extract_pages_from_nodes(nodes)
        
        if not pages:
            return "\n📄 No page information available"
        
        file_path = self.metadata_manager.extract_file_path_from_nodes(nodes)
        filename = self.metadata_manager.extract_filename_from_nodes(nodes)
        ranges = self.metadata_manager.merge_consecutive_pages(pages)
        
        output = "\n" + "="*80 + "\n"
        output += f"📄 SOURCES: {filename}\n"
        output += "="*80 + "\n\n"
        
        for start, end in ranges:
            page_text = self.metadata_manager.format_page_range(start, end)
            
            if file_path:
                link = self.metadata_manager.generate_file_url(file_path, start)
                output += f"  🔗 {page_text}\n     {link}\n\n"
            else:
                output += f"  📄 {page_text}\n\n"
        
        output += "="*80 + "\n"
        
        return output
    
    def format_for_json(self, nodes: List) -> dict:
        """
        Format sources as JSON structure (for APIs).
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            Dictionary with source information
        """
        pages = self.metadata_manager.extract_pages_from_nodes(nodes)
        file_path = self.metadata_manager.extract_file_path_from_nodes(nodes)
        filename = self.metadata_manager.extract_filename_from_nodes(nodes)
        ranges = self.metadata_manager.merge_consecutive_pages(pages)
        
        sources = []
        for start, end in ranges:
            source = {
                'start_page': start,
                'end_page': end,
                'page_text': self.metadata_manager.format_page_range(start, end),
            }
            
            if file_path:
                source['link'] = self.metadata_manager.generate_file_url(file_path, start)
                source['file_path'] = file_path
            
            sources.append(source)
        
        return {
            'filename': filename,
            'total_pages_referenced': len(pages),
            'page_ranges': sources,
            'has_links': file_path is not None
        }
    
    def format_for_html(self, nodes: List) -> str:
        """
        Format sources as HTML (for web interfaces).
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            HTML string
        """
        pages = self.metadata_manager.extract_pages_from_nodes(nodes)
        
        if not pages:
            return "<p>📄 No page information available</p>"
        
        file_path = self.metadata_manager.extract_file_path_from_nodes(nodes)
        filename = self.metadata_manager.extract_filename_from_nodes(nodes)
        ranges = self.metadata_manager.merge_consecutive_pages(pages)
        
        html = '<div class="sources">\n'
        html += f'  <h3>📄 Sources: {filename}</h3>\n'
        html += '  <ul>\n'
        
        for start, end in ranges:
            page_text = self.metadata_manager.format_page_range(start, end)
            
            if file_path:
                link = self.metadata_manager.generate_file_url(file_path, start)
                html += f'    <li><a href="{link}" target="_blank">{page_text}</a></li>\n'
            else:
                html += f'    <li>{page_text}</li>\n'
        
        html += '  </ul>\n'
        html += '</div>\n'
        
        return html
    
    def get_summary(self, nodes: List) -> dict:
        """
        Get summary statistics about sources.
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            Dictionary with summary info
        """
        return self.metadata_manager.get_metadata_summary(nodes)


if __name__ == "__main__":
    # Test formatter
    from llama_index.core.schema import TextNode
    
    test_nodes = [
        TextNode(
            text="Test 1", 
            metadata={
                'page': 5, 
                'filename': 'test.pdf', 
                'file_path': '/path/to/test.pdf'
            }
        ),
        TextNode(
            text="Test 2", 
            metadata={
                'page': 6, 
                'filename': 'test.pdf', 
                'file_path': '/path/to/test.pdf'
            }
        ),
        TextNode(
            text="Test 3", 
            metadata={
                'page': 7, 
                'filename': 'test.pdf', 
                'file_path': '/path/to/test.pdf'
            }
        ),
        TextNode(
            text="Test 4", 
            metadata={
                'page': 10, 
                'filename': 'test.pdf', 
                'file_path': '/path/to/test.pdf'
            }
        ),
    ]
    
    formatter = SourceFormatter()
    
    print("Testing SourceFormatter:\n")
    
    print("1. Terminal format:")
    print(formatter.format_for_terminal(test_nodes))
    
    print("\n2. Plain text format:")
    print(formatter.format_for_plain_text(test_nodes))
    
    print("\n3. JSON format:")
    import json
    print(json.dumps(formatter.format_for_json(test_nodes), indent=2))
    
    print("\n4. HTML format:")
    print(formatter.format_for_html(test_nodes))
    
    print("\n5. Summary:")
    print(formatter.get_summary(test_nodes))