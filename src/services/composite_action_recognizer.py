"""
Composite Action Logic.

Combines ST-GCN Action + Emotion + Heuristics to detect complex scenarios.
"""
from typing import Optional, Dict

class CompositeActionRecognizer:
    """
    Recognizes complex actions by combining multiple signal sources.
    """
    
    @staticmethod
    def refine_action(stgcn_action: str, emotion: str, frame_info: Dict) -> str:
        """
        Refine the action label based on context.
        
        Args:
            stgcn_action: Label from ST-GCN (e.g., "sitting", "waving hand")
            emotion: Detected emotion (e.g., "happy", "sad")
            frame_info: Dictionary containing other context (e.g., hand_near_face)
            
        Returns:
            Refined action label (e.g., "Crying on Sofa")
        """
        stgcn_action = stgcn_action.lower()
        emotion = emotion.lower()
        
        # Scenario 1: "Waving hand while sitting with a smiley face"
        if "waving" in stgcn_action and "happy" in emotion:
            return "Happy Greeting (Waving)"
            
        # Scenario 2: "Sad woman slowly covering her face to cry while lying down"
        # ST-GCN might see "lying down" or "sleeping" or "headache"
        # Heuristic: Hand near face
        if ("lying" in stgcn_action or "sleeping" in stgcn_action) and "sad" in emotion:
            if frame_info.get('hand_near_face', False):
                 return "Crying / Distress (Lying Down)"
            return "Depressed / Sad (Lying Down)"
            
        # Scenario 3: "People working in office, typing"
        # ST-GCN has "typing" and "using computer"
        if stgcn_action in ["typing", "using computer"]:
            return "Office Work (Typing)"
            
        # Scenario 4: "Doctor unwrapping gases..." - Very hard
        # ST-GCN might see "bandaging"? 
        if "bandaging" in stgcn_action:
             return "Medical / First Aid (Bandaging)"
             
        # Scenario 5: "Shaking hands" - Direct mapping
        if "shaking hands" in stgcn_action:
            return "Handshake Interaction"

        return stgcn_action.title()
