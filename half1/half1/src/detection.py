import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
yolo_model = None
text_detection_model = None

# Common object categories for accessibility
ACCESSIBILITY_OBJECTS = {
    'person': 'person',
    'bicycle': 'bicycle', 
    'car': 'car',
    'motorcycle': 'motorcycle',
    'airplane': 'airplane',
    'bus': 'bus',
    'train': 'train',
    'truck': 'truck',
    'boat': 'boat',
    'traffic light': 'traffic light',
    'fire hydrant': 'fire hydrant',
    'stop sign': 'stop sign',
    'parking meter': 'parking meter',
    'bench': 'bench',
    'bird': 'bird',
    'cat': 'cat',
    'dog': 'dog',
    'horse': 'horse',
    'sheep': 'sheep',
    'cow': 'cow',
    'elephant': 'elephant',
    'bear': 'bear',
    'zebra': 'zebra',
    'giraffe': 'giraffe',
    'backpack': 'backpack',
    'umbrella': 'umbrella',
    'handbag': 'handbag',
    'tie': 'tie',
    'suitcase': 'suitcase',
    'frisbee': 'frisbee',
    'skis': 'skis',
    'snowboard': 'snowboard',
    'sports ball': 'sports ball',
    'kite': 'kite',
    'baseball bat': 'baseball bat',
    'baseball glove': 'baseball glove',
    'skateboard': 'skateboard',
    'surfboard': 'surfboard',
    'tennis racket': 'tennis racket',
    'bottle': 'bottle',
    'wine glass': 'wine glass',
    'cup': 'cup',
    'fork': 'fork',
    'knife': 'knife',
    'spoon': 'spoon',
    'bowl': 'bowl',
    'banana': 'banana',
    'apple': 'apple',
    'sandwich': 'sandwich',
    'orange': 'orange',
    'broccoli': 'broccoli',
    'carrot': 'carrot',
    'hot dog': 'hot dog',
    'pizza': 'pizza',
    'donut': 'donut',
    'cake': 'cake',
    'chair': 'chair',
    'couch': 'couch',
    'potted plant': 'potted plant',
    'bed': 'bed',
    'dining table': 'dining table',
    'toilet': 'toilet',
    'tv': 'television',
    'laptop': 'laptop',
    'mouse': 'computer mouse',
    'remote': 'remote control',
    'keyboard': 'keyboard',
    'cell phone': 'cell phone',
    'microwave': 'microwave',
    'oven': 'oven',
    'toaster': 'toaster',
    'sink': 'sink',
    'refrigerator': 'refrigerator',
    'book': 'book',
    'clock': 'clock',
    'vase': 'vase',
    'scissors': 'scissors',
    'teddy bear': 'teddy bear',
    'hair drier': 'hair drier',
    'toothbrush': 'toothbrush'
}


def _remove_duplicate_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate or very similar detections with improved logic"""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
    
    filtered = []
    for detection in detections:
        is_duplicate = False
        bbox1 = detection['bbox']
        label1 = detection['label']
        conf1 = detection.get('confidence', 0)
        
        for existing in filtered:
            bbox2 = existing['bbox']
            label2 = existing['label']
            conf2 = existing.get('confidence', 0)
            
            # Calculate IoU (Intersection over Union)
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                # More sophisticated duplicate detection
                is_duplicate = False
                
                # Same label with high IoU
                if label1 == label2 and iou > 0.3:
                    is_duplicate = True
                # Very high IoU regardless of label (likely same object misclassified)
                elif iou > 0.7:
                    is_duplicate = True
                # Similar labels with moderate IoU (e.g., "cup" and "bottle")
                elif iou > 0.2 and _are_similar_objects(label1, label2):
                    is_duplicate = True
                
                if is_duplicate:
                    # Keep the one with higher confidence
                    if conf1 > conf2:
                        # Replace existing with current detection
                        filtered.remove(existing)
                        filtered.append(detection)
                    break
        
        if not is_duplicate:
            filtered.append(detection)
    
    return filtered


def _are_similar_objects(label1: str, label2: str) -> bool:
    """Check if two object labels are similar enough to be considered duplicates"""
    similar_groups = [
        ['cup', 'bottle', 'wine glass'],
        ['apple', 'orange', 'banana'],
        ['car', 'truck', 'bus'],
        ['dog', 'cat'],
        ['person', 'person'],
        ['book', 'magazine'],
        ['laptop', 'computer'],
        ['cell phone', 'phone'],
        ['sports ball', 'tennis ball', 'soccer ball'],
        ['chair', 'stool'],
        ['table', 'dining table'],
        ['vase', 'bowl'],
    ]
    
    for group in similar_groups:
        if label1 in group and label2 in group:
            return True
    
    return False


def _ensure_yolo():
    """Initialize YOLO model for object detection"""
	global yolo_model
	if yolo_model is not None:
		return
    
	try:
        from ultralytics import YOLO
		model_name = os.environ.get('YOLO_MODEL', 'yolov8n.pt')
        logger.info(f"Loading YOLO model: {model_name}")
		yolo_model = YOLO(model_name)
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
		yolo_model = None


def _run_yolo_detection(image: Image.Image) -> List[Dict[str, Any]]:
    """Run YOLO object detection on image"""
	_ensure_yolo()
	if yolo_model is None:
        logger.warning("YOLO model not available, using fallback detection")
        return _fallback_object_detection(image)
    
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run inference with higher confidence threshold for better accuracy
        results = yolo_model.predict(img_array, verbose=False, conf=0.3, iou=0.5)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Calculate image dimensions for size filtering
                img_h, img_w = img_array.shape[:2]
                total_area = img_w * img_h
                
                # Object-specific minimum area requirements
                min_areas = {
                    'person': total_area * 0.05,      # 5% for people
                    'car': total_area * 0.08,         # 8% for cars
                    'bicycle': total_area * 0.03,     # 3% for bicycles
                    'dog': total_area * 0.02,         # 2% for dogs
                    'cat': total_area * 0.02,         # 2% for cats
                    'apple': total_area * 0.005,      # 0.5% for small objects
                    'banana': total_area * 0.005,
                    'orange': total_area * 0.005,
                    'cup': total_area * 0.01,         # 1% for cups
                    'bottle': total_area * 0.01,
                    'vase': total_area * 0.02,        # 2% for vases
                    'book': total_area * 0.01,
                    'laptop': total_area * 0.05,
                    'cell phone': total_area * 0.005,
                    'sports ball': total_area * 0.01,
                    'frisbee': total_area * 0.01,
                }
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = box
                    class_name = result.names.get(cls_id, f"class_{cls_id}")
                    
                    # Calculate bounding box area and aspect ratio
                    bbox_area = (x2 - x1) * (y2 - y1)
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    aspect_ratio = bbox_width / max(1, bbox_height)
                    
                    # Get minimum area for this object type
                    min_area = min_areas.get(class_name, total_area * 0.01)  # Default 1%
                    
                    # Calculate confidence score with additional factors
                    confidence_score = float(conf)
                    
                    # Penalize very small or very large detections
                    if bbox_area < min_area:
                        confidence_score *= 0.5
                    elif bbox_area > total_area * 0.8:  # Too large
                        confidence_score *= 0.3
                    
                    # Penalize extreme aspect ratios (likely false positives)
                    if aspect_ratio > 10 or aspect_ratio < 0.1:
                        confidence_score *= 0.4
                    
                    # Higher confidence threshold and size filtering for better accuracy
                    if confidence_score > 0.3 and bbox_area > min_area:
                        # Use accessibility-friendly names if available, otherwise use original
                        friendly_name = ACCESSIBILITY_OBJECTS.get(class_name, class_name)
                        detections.append({
                            'label': friendly_name,
                            'confidence': confidence_score,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': int(cls_id),
                            'area': float(bbox_area),
                            'aspect_ratio': float(aspect_ratio)
                        })
        
        # If no objects detected with high confidence, try lower threshold
        if not detections:
            # Try with lower confidence but still filter by size and quality
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                class_name = result.names.get(cls_id, f"class_{cls_id}")
                
                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                aspect_ratio = bbox_width / max(1, bbox_height)
                
                # Get minimum area for this object type
                min_area = min_areas.get(class_name, total_area * 0.01)
                
                # Calculate confidence score
                confidence_score = float(conf)
                
                # Apply penalties
                if bbox_area < min_area:
                    confidence_score *= 0.5
                elif bbox_area > total_area * 0.8:
                    confidence_score *= 0.3
                
                if aspect_ratio > 10 or aspect_ratio < 0.1:
                    confidence_score *= 0.4
                
                # Lower threshold but still require reasonable size and quality
                if confidence_score > 0.15 and bbox_area > min_area:
                    friendly_name = ACCESSIBILITY_OBJECTS.get(class_name, class_name)
                    detections.append({
                        'label': friendly_name,
                        'confidence': confidence_score,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'class_id': int(cls_id),
                        'area': float(bbox_area),
                        'aspect_ratio': float(aspect_ratio)
                    })
        
        # If still no objects detected, use fallback
        if not detections:
            detections = _fallback_object_detection(image)
        
        # Remove duplicate or very similar detections
        detections = _remove_duplicate_detections(detections)
        
        logger.info(f"YOLO detected {len(detections)} objects")
        return detections
        
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}, using fallback")
        return _fallback_object_detection(image)


def _fallback_object_detection(image: Image.Image) -> List[Dict[str, Any]]:
    """Fallback object detection using basic image analysis"""
    try:
        # Try to import cv2, but don't fail if it's not available
        try:
            import cv2
            cv2_available = True
        except ImportError:
            cv2_available = False
            logger.warning("OpenCV not available, using basic image analysis")
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        if cv2_available:
            # Use OpenCV for better analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Get image properties for better descriptions
            brightness = np.mean(gray)
            contrast = np.std(gray)
            is_dark = brightness < 85
            is_low_contrast = contrast < 30
            
            # Detect edges and contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            h, w = gray.shape
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (w * h) * 0.01:  # Only significant objects
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = cw / max(1, ch)
                    
                    # Classify based on shape
                    if aspect_ratio > 2:
                        label = "rectangular object"
                    elif aspect_ratio < 0.5:
                        label = "tall object"
                    elif area > (w * h) * 0.1:
                        label = "large object"
                    else:
                        label = "object"
                    
                    detections.append({
                        'label': label,
                        'confidence': 0.6,  # Medium confidence for fallback
                        'bbox': [float(x), float(y), float(x + cw), float(y + ch)]
                    })
            
            # If still no objects, add a generic detection based on image properties
            if not detections:
                if is_dark:
                    detections.append({
                        'label': 'dark image content',
                        'confidence': 0.6,
                        'bbox': [0, 0, float(w), float(h)]
                    })
                elif is_low_contrast:
                    detections.append({
                        'label': 'low contrast image content',
                        'confidence': 0.6,
                        'bbox': [0, 0, float(w), float(h)]
                    })
                else:
                    detections.append({
                        'label': 'image content',
                        'confidence': 0.5,
                        'bbox': [0, 0, float(w), float(h)]
                    })
        else:
            # Basic analysis without OpenCV
            gray = np.mean(img_array, axis=2)  # Convert to grayscale
            brightness = np.mean(gray)
            contrast = np.std(gray)
            is_dark = brightness < 85
            is_low_contrast = contrast < 30
            
		h, w = gray.shape
            detections = []
            
            # Simple analysis based on brightness patterns
            if is_dark:
                detections.append({
                    'label': 'dark image content',
                    'confidence': 0.6,
                    'bbox': [0, 0, float(w), float(h)]
                })
            elif is_low_contrast:
                detections.append({
                    'label': 'low contrast image content',
                    'confidence': 0.6,
                    'bbox': [0, 0, float(w), float(h)]
                })
            else:
                detections.append({
                    'label': 'image content',
                    'confidence': 0.5,
                    'bbox': [0, 0, float(w), float(h)]
                })
        
        logger.info(f"Fallback detection found {len(detections)} objects")
        return detections
        
    except Exception as e:
        logger.error(f"Fallback detection failed: {e}")
        # Ultimate fallback
        return [{
            'label': 'image',
            'confidence': 0.3,
            'bbox': [0, 0, float(image.width), float(image.height)]
        }]


def _detect_text_in_image(image: Image.Image) -> List[Dict[str, Any]]:
    """Detect text in image using OCR"""
    try:
        # Try to import required libraries
        try:
            import cv2
            import pytesseract
            ocr_available = True
        except ImportError as e:
            ocr_available = False
            logger.warning(f"OCR libraries not available: {e}")
        
        if not ocr_available:
            return []
        
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Use Tesseract OCR
        text_data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        text_elements = []
        for i, conf in enumerate(text_data['conf']):
            if int(conf) > 30:  # Confidence threshold
                text = text_data['text'][i].strip()
                if text:
                    x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
                    text_elements.append({
                        'label': f'text: "{text}"',
                        'confidence': float(conf) / 100.0,
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'text': text
                    })
        
        logger.info(f"Detected {len(text_elements)} text elements")
        return text_elements
        
    except Exception as e:
        logger.error(f"Text detection failed: {e}")
		return []


def _analyze_image_properties(image: Image.Image) -> Dict[str, Any]:
    """Analyze basic image properties for accessibility"""
    try:
        # Try to import cv2, but don't fail if it's not available
        try:
            import cv2
            cv2_available = True
        except ImportError:
            cv2_available = False
            logger.warning("OpenCV not available for image analysis, using basic analysis")
        
        # Convert to numpy array
        img_array = np.array(image)
        
        if cv2_available:
            # Use OpenCV for better analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = np.std(gray)
            
            # Detect dominant colors
            pixels = img_array.reshape(-1, 3)
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))
            
            # Detect edges (for structure analysis)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'unique_colors': int(unique_colors),
                'edge_density': float(edge_density),
            'is_dark': bool(brightness < 85),
            'is_low_contrast': bool(contrast < 30),
            'is_complex': bool(edge_density > 0.1)
            }
        else:
            # Basic analysis without OpenCV
            gray = np.mean(img_array, axis=2)  # Convert to grayscale
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = np.std(gray)
            
            # Detect dominant colors
            pixels = img_array.reshape(-1, 3)
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))
            
            # Simple edge detection using numpy
            # Calculate gradient magnitude as a proxy for edge density
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 255.0
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'unique_colors': int(unique_colors),
                'edge_density': float(edge_density),
            'is_dark': bool(brightness < 85),
            'is_low_contrast': bool(contrast < 30),
            'is_complex': bool(edge_density > 0.1)
            }
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return {
            'brightness': 128.0,
            'contrast': 50.0,
            'unique_colors': 1000,
            'edge_density': 0.1,
            'is_dark': False,
            'is_low_contrast': False,
            'is_complex': False
        }


def detect_scene(image: Image.Image) -> Dict[str, Any]:
    """
    Comprehensive scene detection for accessibility
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing detected objects, text, and image properties
    """
    logger.info(f"Starting scene detection for image {image.size}")
    
    # Detect objects using YOLO
    objects = _run_yolo_detection(image)
    
    # Detect text in image
    text_elements = _detect_text_in_image(image)
    
    # Analyze image properties
    image_properties = _analyze_image_properties(image)
    
    # Combine all detections
    result = {
        'objects': objects,
        'text_elements': text_elements,
        'image_properties': image_properties,
        'image_size': {'width': image.width, 'height': image.height},
        'total_detections': len(objects) + len(text_elements)
    }
    
    logger.info(f"Scene detection complete: {result['total_detections']} total detections")
    return result
