"""
Bounding box model representing a rectangular region in an image.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BoundingBox:
    """
    Represents a rectangular bounding box in an image.
    
    Attributes:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of the bounding box
        height: Height of the bounding box
    """
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x1(self) -> int:
        """Left edge coordinate."""
        return self.x
    
    @property
    def y1(self) -> int:
        """Top edge coordinate."""
        return self.y
    
    @property
    def x2(self) -> int:
        """Right edge coordinate."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom edge coordinate."""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Returns the center point (x, y) of the bounding box."""
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        return (center_x, center_y)
    
    @property
    def area(self) -> int:
        """Returns the area of the bounding box."""
        return self.width * self.height
    
    def contains_point(self, point_x: float, point_y: float) -> bool:
        """
        Check if a point is inside this bounding box.
        
        Args:
            point_x: X coordinate of the point
            point_y: Y coordinate of the point
            
        Returns:
            True if the point is inside the box, False otherwise
        """
        return (self.x1 <= point_x <= self.x2 and 
                self.y1 <= point_y <= self.y2)
    
    def to_xyxy_tuple(self) -> Tuple[int, int, int, int]:
        """Returns coordinates as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_xywh_tuple(self) -> Tuple[int, int, int, int]:
        """Returns coordinates as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> 'BoundingBox':
        """
        Create a BoundingBox from (x1, y1, x2, y2) coordinates.
        
        Args:
            x1: Left edge coordinate
            y1: Top edge coordinate
            x2: Right edge coordinate
            y2: Bottom edge coordinate
            
        Returns:
            BoundingBox instance
        """
        width = x2 - x1
        height = y2 - y1
        return cls(x=x1, y=y1, width=width, height=height)
    
    @classmethod
    def from_xywh(cls, x: int, y: int, width: int, height: int) -> 'BoundingBox':
        """
        Create a BoundingBox from (x, y, width, height) coordinates.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of the box
            height: Height of the box
            
        Returns:
            BoundingBox instance
        """
        return cls(x=x, y=y, width=width, height=height)
