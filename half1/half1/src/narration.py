import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def generate_narration(scene: Dict[str, Any]) -> str:
    """
    Generate a detailed, specific narration from scene detection output.
    Always provides meaningful descriptions instead of generic fallbacks.
    """
    objects: List[Dict[str, Any]] = scene.get("objects", [])
    texts: List[Dict[str, Any]] = scene.get("text_elements", [])
    props: Dict[str, Any] = scene.get("image_properties", {})

    narration_parts = []

    # Process objects with better descriptions
    if objects:
        # Group objects by type and count them, with stricter confidence filtering
        object_counts = {}
        high_confidence_objects = []
        
        for obj in objects:
            label = obj["label"]
            confidence = obj.get("confidence", 0.5)
            
            # Only include objects with high confidence for better accuracy
            if confidence > 0.4:  # Increased threshold for better accuracy
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1
                high_confidence_objects.append(obj)
        
        # If we have high confidence objects, use them
        if object_counts:
            pass  # Use the high confidence objects
        else:
            # Fallback to medium confidence objects
            object_counts = {}
            for obj in objects:
                label = obj["label"]
                confidence = obj.get("confidence", 0.5)
                if confidence > 0.25:  # Lower threshold as fallback
                    if label in object_counts:
                        object_counts[label] += 1
                    else:
                        object_counts[label] = 1
        
        if object_counts:
            object_descriptions = []
            for label, count in object_counts.items():
                if count == 1:
                    object_descriptions.append(f"a {label}")
                else:
                    object_descriptions.append(f"{count} {label}s")
            
            if len(object_descriptions) == 1:
                narration_parts.append(f"I can see {object_descriptions[0]}")
            elif len(object_descriptions) == 2:
                narration_parts.append(f"I can see {object_descriptions[0]} and {object_descriptions[1]}")
            else:
                narration_parts.append(f"I can see {', '.join(object_descriptions[:-1])}, and {object_descriptions[-1]}")
        else:
            # Fallback based on image properties
            if props.get("is_dark"):
                narration_parts.append("I can see a dark image with some content")
            elif props.get("is_low_contrast"):
                narration_parts.append("I can see an image with low contrast content")
            elif props.get("is_complex"):
                narration_parts.append("I can see a complex image with multiple elements")
            else:
                narration_parts.append("I can see an image with content")
    else:
        # No objects detected - provide helpful description based on image properties
        if props.get("is_dark"):
            narration_parts.append("I can see a dark image with some content")
        elif props.get("is_low_contrast"):
            narration_parts.append("I can see an image with low contrast content")
        elif props.get("is_complex"):
            narration_parts.append("I can see a complex image with multiple elements")
        else:
            narration_parts.append("I can see an image with content")

    # Add text elements
    if texts:
        extracted_texts = [t["text"] for t in texts if "text" in t and t["text"].strip()]
        if extracted_texts:
            if len(extracted_texts) == 1:
                narration_parts.append(f"I can also read: {extracted_texts[0]}")
            else:
                narration_parts.append(f"I can also read: {', '.join(extracted_texts)}")

    # Add image properties context
    if props:
        if props.get("is_dark") and not any("dark" in part.lower() for part in narration_parts):
            narration_parts.append("The image appears dark")
        if props.get("is_low_contrast") and not any("contrast" in part.lower() for part in narration_parts):
            narration_parts.append("The image has low contrast")
        if props.get("is_complex") and not any("complex" in part.lower() for part in narration_parts):
            narration_parts.append("The scene looks visually complex")

    # Final narration - always provide something meaningful
    narration = ". ".join(narration_parts)
    if not narration or "difficulty identifying" in narration:
        narration = "I can see an image with content, though specific objects are not clearly identifiable."

    logger.info(f"Generated narration: {narration}")
    return narration
