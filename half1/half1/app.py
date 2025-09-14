import base64
import io
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any
from werkzeug.exceptions import RequestEntityTooLarge

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

from src.detection import detect_scene
from src.narration import generate_narration
from src.tts import synthesize_speech, get_available_voices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure file upload limits
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size


# -------------------------------
# Error Handlers
# -------------------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    logger.warning(f"File too large: {e}")
    return jsonify({
        'error': 'File too large. Please upload an image smaller than 10MB.',
        'error_code': 'FILE_TOO_LARGE'
    }), 413


@app.errorhandler(413)
def handle_413(e):
    return jsonify({
        'error': 'File too large. Please upload an image smaller than 10MB.',
        'error_code': 'FILE_TOO_LARGE'
    }), 413


@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({
        'error': 'Bad request. Please check your input.',
        'error_code': 'BAD_REQUEST'
    }), 400


@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error. Please try again later.',
        'error_code': 'INTERNAL_ERROR'
    }), 500


# -------------------------------
# Helpers
# -------------------------------
def load_image_from_request() -> Image.Image:
    """Load and validate image from request"""
    logger.info(f"Processing request - Method: {request.method}, Content-Type: {request.content_type}")
    
    try:
        # Handle multipart file upload
        if 'image' in request.files:
            file = request.files['image']
            if not file or file.filename == '':
                raise ValueError('No file selected')
            
            logger.info(f"File received: {file.filename}, size: {file.content_length}")
            
            # Validate file type
            if file.content_type and not file.content_type.startswith('image/'):
                filename = file.filename.lower()
                if not any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
                    raise ValueError('Invalid file type. Please upload an image file.')
            elif not file.content_type:
                filename = file.filename.lower()
                if not any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
                    raise ValueError('Invalid file type. Please upload an image file.')
            
            try:
                image = Image.open(file.stream)
                image = image.convert('RGB')  # Ensure RGB format
                logger.info(f"Image loaded successfully: {image.size}")
                return image
            except Exception as e:
                raise ValueError(f'Invalid image file: {str(e)}')
        
        # Handle base64 image
        if request.is_json and request.json and request.json.get('image_base64'):
            img_b64 = request.json['image_base64']
            logger.info(f"Base64 image received, length: {len(img_b64)}")
            
            try:
                if ',' in img_b64:  # Handle data URL format
                    img_b64 = img_b64.split(',')[-1]
                
                img_bytes = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(img_bytes))
                image = image.convert('RGB')
                logger.info(f"Base64 image loaded successfully: {image.size}")
                return image
            except Exception as e:
                raise ValueError(f'Invalid base64 image: {str(e)}')
        
        raise ValueError('No image provided. Please upload an image file or provide base64 data.')
        
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise


def validate_preferences(prefs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize user preferences"""
    valid_styles = ['short', 'detailed']
    valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'hi', 'ja', 'ko', 'zh']
    valid_speeds = ['slow', 'normal', 'fast']
    
    validated = {
        'style': prefs.get('style', 'detailed'),
        'language': prefs.get('language', 'en'),
        'voiceSpeed': prefs.get('voiceSpeed', 'normal')
    }
    
    if validated['style'] not in valid_styles:
        validated['style'] = 'detailed'
    if validated['language'] not in valid_languages:
        validated['language'] = 'en'
    if validated['voiceSpeed'] not in valid_speeds:
        validated['voiceSpeed'] = 'normal'
    
    return validated


# -------------------------------
# Routes
# -------------------------------
@app.route('/', methods=['GET'])
def index():
    """Serve the main application page"""
    try:
        with open('static/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return f"Error loading page: {str(e)}", 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "time": datetime.utcnow().isoformat() + 'Z',
        "version": "2.0.0",
        "service": "Accessible Image Narrator"
    })


@app.route('/api/voices', methods=['GET'])
def voices():
    """Get available TTS voices and languages"""
    try:
        voices_info = get_available_voices()
        return jsonify(voices_info)
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        return jsonify({
            'engines': [],
            'languages': ['en'],
            'default_engine': 'auto'
        })


@app.route('/api/narrate', methods=['POST'])
def narrate():
    """Main narration endpoint"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    logger.info(f"[{request_id}] Starting narration request")
    
    try:
        image = load_image_from_request()
        
        prefs = {}
        if request.is_json and request.json:
            prefs = request.json.get('preferences', {})
        
        validated_prefs = validate_preferences(prefs)
        logger.info(f"[{request_id}] Preferences: {validated_prefs}")
        
        logger.info(f"[{request_id}] Starting object detection")
        detections = detect_scene(image)
        logger.info(f"[{request_id}] Detection complete: {detections.get('total_detections', 0)} items found")
        
        logger.info(f"[{request_id}] Generating narration")
        narration_text = generate_narration(detections)
        
        if not narration_text or not narration_text.strip():
            narration_text = 'I can see an image, but I\'m having difficulty providing a detailed description at this time.'
        
        logger.info(f"[{request_id}] Narration generated: {len(narration_text)} characters")
        
        logger.info(f"[{request_id}] Generating audio")
        audio_b64 = synthesize_speech(
            narration_text, 
            language=validated_prefs.get('language', 'en')
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[{request_id}] Request completed in {elapsed_ms}ms")
        
        response_data = {
            'text': narration_text,
            'audio_base64': audio_b64,
            'detections': {
                'objects': detections.get('objects', []),
                'text_elements': detections.get('text_elements', []),
                'total_count': detections.get('total_detections', 0)
            },
            'image_info': {
                'size': detections.get('image_size', {}),
                'properties': detections.get('image_properties', {})
            },
            'preferences': validated_prefs,
            'elapsed_ms': elapsed_ms,
            'request_id': request_id
        }
        
        return jsonify(response_data)
        
    except ValueError as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"[{request_id}] Validation error: {str(e)}")
        return jsonify({
            'error': str(e),
            'error_code': 'VALIDATION_ERROR',
            'elapsed_ms': elapsed_ms,
            'request_id': request_id
        }), 400
        
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}',
            'error_code': 'PROCESSING_ERROR',
            'elapsed_ms': elapsed_ms,
            'request_id': request_id
        }), 500


@app.route('/api/preferences', methods=['GET', 'POST'])
def preferences():
    """Handle user preferences"""
    if request.method == 'POST':
        try:
            prefs = request.json or {}
            validated_prefs = validate_preferences(prefs)
            return jsonify({
                'saved': True, 
                'preferences': validated_prefs
            })
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
            return jsonify({
                'error': 'Failed to save preferences',
                'error_code': 'PREFERENCES_ERROR'
            }), 400
    
    return jsonify({
        'preferences': {
            'style': 'detailed',
            'language': 'en',
            'voiceSpeed': 'normal'
        }
    })


@app.route('/api/status', methods=['GET'])
def status():
    """Get system status and capabilities"""
    try:
        from src.detection import yolo_model
        from src.narration import _openai_vision_available
        
        status_info = {
            'status': 'operational',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'capabilities': {
                'object_detection': yolo_model is not None,
                'text_detection': True,
                'openai_vision': _openai_vision_available(),
                'tts': True
            },
            'limits': {
                'max_file_size_mb': 10,
                'supported_formats': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
                'max_processing_time_seconds': 30
            }
        }
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'status': 'degraded',
            'error': 'Unable to determine system status'
        }), 500


# -------------------------------
# Run App
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Accessible Image Narrator on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug, use_reloader=debug)
