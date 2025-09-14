import base64
import os
import logging
import io
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language mapping for TTS engines
LANGUAGE_MAPPING = {
    'en': 'en',
    'es': 'es', 
    'fr': 'fr',
    'de': 'de',
    'hi': 'hi',
    'it': 'it',
    'pt': 'pt',
    'ru': 'ru',
    'ja': 'ja',
    'ko': 'ko',
    'zh': 'zh-cn'
}

# TTS engine preferences for accessibility
TTS_ENGINES = ['pyttsx3', 'gtts', 'espeak']


def _synthesize_with_pyttsx3(text: str, language: str = 'en') -> Optional[str]:
    """Use pyttsx3 for offline TTS synthesis"""
    try:
        import pyttsx3
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Configure voice properties for accessibility
        voices = engine.getProperty('voices')
        
        # Try to find a suitable voice for the language
        if voices:
            for voice in voices:
                if language in voice.id.lower() or language in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        # Set speech rate and volume for accessibility
        engine.setProperty('rate', 150)  # Slower rate for better comprehension
        engine.setProperty('volume', 0.9)  # High volume
        
        # Save to buffer
        buffer = io.BytesIO()
        
        # Create a temporary file approach since pyttsx3 doesn't directly support BytesIO
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save to temporary file
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            # Read the file and convert to base64
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            b64 = base64.b64encode(audio_data).decode('utf-8')
            return f"data:audio/wav;base64,{b64}"
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"pyttsx3 TTS failed: {e}")
        return None


def _synthesize_with_gtts(text: str, language: str = 'en') -> Optional[str]:
    """Use Google Text-to-Speech for online TTS synthesis"""
    try:
        from gtts import gTTS
        
        # Map language code
        lang_code = LANGUAGE_MAPPING.get(language, 'en')
        
        # Create TTS object
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Save to buffer
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        
        # Convert to base64
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:audio/mpeg;base64,{b64}"
        
    except Exception as e:
        logger.error(f"gTTS synthesis failed: {e}")
        return None


def _synthesize_with_espeak(text: str, language: str = 'en') -> Optional[str]:
    """Use espeak for system TTS synthesis"""
    try:
        import subprocess
        import tempfile
        
        # Map language codes for espeak
        espeak_lang_map = {
            'en': 'en',
            'es': 'es',
            'fr': 'fr', 
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'hi': 'hi'
        }
        
        lang_code = espeak_lang_map.get(language, 'en')
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Run espeak command
            cmd = [
                'espeak', 
                '-s', '150',  # Slower speed for accessibility
                '-v', lang_code,
                '-w', temp_path,
                text
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                # Read the generated audio file
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                
                b64 = base64.b64encode(audio_data).decode('utf-8')
                return f"data:audio/wav;base64,{b64}"
            else:
                logger.error(f"espeak failed: {result.stderr}")
                return None
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"espeak TTS failed: {e}")
        return None


def synthesize_speech(text: str, language: str = 'en', engine: str = 'auto') -> Optional[str]:
    """
    Synthesize speech from text with multiple engine fallbacks
    
    Args:
        text: Text to synthesize
        language: Language code (en, es, fr, de, etc.)
        engine: TTS engine to use ('auto', 'pyttsx3', 'gtts', 'espeak')
        
    Returns:
        Base64 encoded audio data URL or None if synthesis fails
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for TTS")
        return None
    
    # Clean and prepare text for TTS
    clean_text = text.strip()
    if len(clean_text) > 1000:  # Limit text length for performance
        clean_text = clean_text[:1000] + "..."
    
    logger.info(f"Synthesizing speech: {len(clean_text)} characters in {language}")
    
    # Test mode - return None to use browser TTS
    if os.environ.get('TEST_MODE', '0') == '1':
        logger.info("Test mode: skipping server-side TTS")
        return None
    
    # Try engines in order of preference
    engines_to_try = []
    
    if engine == 'auto':
        engines_to_try = TTS_ENGINES
    else:
        engines_to_try = [engine] + [e for e in TTS_ENGINES if e != engine]
    
    for tts_engine in engines_to_try:
        try:
            if tts_engine == 'pyttsx3':
                result = _synthesize_with_pyttsx3(clean_text, language)
            elif tts_engine == 'gtts':
                result = _synthesize_with_gtts(clean_text, language)
            elif tts_engine == 'espeak':
                result = _synthesize_with_espeak(clean_text, language)
            else:
                continue
            
            if result:
                logger.info(f"TTS synthesis successful using {tts_engine}")
                return result
                
        except Exception as e:
            logger.warning(f"TTS engine {tts_engine} failed: {e}")
            continue
    
    logger.error("All TTS engines failed")
    return None


def get_available_voices() -> Dict[str, Any]:
    """Get information about available TTS voices"""
    voices_info = {
        'engines': [],
        'languages': list(LANGUAGE_MAPPING.keys()),
        'default_engine': 'auto'
    }
    
    # Check available engines
    try:
        import pyttsx3
        voices_info['engines'].append('pyttsx3')
    except ImportError:
        pass
    
    try:
        import gtts
        voices_info['engines'].append('gtts')
    except ImportError:
        pass
    
    try:
        import subprocess
        result = subprocess.run(['espeak', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            voices_info['engines'].append('espeak')
    except Exception:
        pass
    
    return voices_info
