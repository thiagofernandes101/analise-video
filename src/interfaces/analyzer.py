"""
Analyzer protocol interface.

Defines the contract for analysis components (emotion, activity, etc.).
"""
from typing import Protocol, Any


class AnalyzerProtocol(Protocol):
    """
    Protocol for analyzer components.
    
    Any analyzer implementation should provide these methods.
    """
    
    def analyze(self, data: Any) -> Any:
        """
        Perform analysis on input data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Analysis results
        """
        ...
    
    def get_result(self, identifier: Any) -> Any:
        """
        Get cached analysis result for a specific identifier.
        
        Args:
            identifier: Unique identifier for the result
            
        Returns:
            Cached analysis result
        """
        ...
