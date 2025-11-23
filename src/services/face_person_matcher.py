"""
Face-to-person matching service.

Responsible for associating detected faces with tracked persons.
Single Responsibility: Spatial matching of faces to persons.
"""
from typing import Optional

from models.face_detection import FaceDetection
from models.person_tracking import PersonTracking


class FacePersonMatcher:
    """
    Matches detected faces to tracked persons based on spatial overlap.
    
    Uses bounding box containment to determine face ownership.
    """
    
    def match_face_to_person(
        self, 
        face: FaceDetection, 
        person: PersonTracking
    ) -> bool:
        """
        Check if a face belongs to a person.
        
        Args:
            face: Detected face
            person: Tracked person
            
        Returns:
            True if face is likely part of this person
        """
        face_center_x, face_center_y = face.bounding_box.center
        
        return person.bounding_box.contains_point(face_center_x, face_center_y)
    
    def find_person_for_face(
        self,
        face: FaceDetection,
        persons: list[PersonTracking]
    ) -> Optional[int]:
        """
        Find which person a face belongs to.
        
        Args:
            face: Detected face
            persons: List of tracked persons
            
        Returns:
            Person ID if match found, None otherwise
        """
        for person in persons:
            if self.match_face_to_person(face, person):
                return person.track_id
        
        return None
