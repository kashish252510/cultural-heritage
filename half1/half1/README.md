# Accessible Image Narrator

An AI-powered web application designed specifically for visually impaired users to get detailed audio descriptions of images. Upload any image and receive comprehensive descriptions of objects, text, and scenes through both text and speech synthesis.

## Features

### 🎯 **Accessibility-First Design**
- **Screen Reader Compatible**: Full ARIA support and semantic HTML
- **Keyboard Navigation**: Complete keyboard accessibility
- **High Contrast Support**: Respects user's contrast preferences
- **Voice Speed Control**: Adjustable speech rate for better comprehension

### 🔍 **Advanced Object Detection**
- **YOLO Integration**: State-of-the-art object detection
- **Text Recognition**: OCR for reading text in images
- **Scene Analysis**: Understanding of image properties and complexity
- **Confidence Filtering**: Only reports high-confidence detections

### 🎤 **Multiple TTS Options**
- **Server-Side TTS**: High-quality audio generation with multiple engines
- **Browser TTS**: Fallback using Web Speech API
- **Multi-Language Support**: 10+ languages supported
- **Voice Customization**: Speed and language preferences

### 🎨 **Modern Interface**
- **Drag & Drop Upload**: Easy image selection
- **Real-time Feedback**: Processing status and error messages
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Clean UI**: Focused on usability and accessibility

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment (Optional)
Create a `.env` file for advanced features:
```env
# For OpenAI Vision API (optional)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_VISION_MODEL=gpt-4o-mini

# For custom YOLO model (optional)
YOLO_MODEL=yolov8n.pt

# For test mode (disables external APIs)
TEST_MODE=0
```

### 3. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## How to Use

1. **Upload Image**: Click the upload area or drag and drop an image file
2. **Adjust Settings**: Choose description style, language, and voice speed
3. **Process**: Click "Describe Image" to analyze your image
4. **Listen/Read**: Get both audio and text descriptions

## Supported Features

### Image Formats
- JPEG/JPG
- PNG
- GIF
- WebP
- Maximum file size: 10MB

### Languages
- English (default)
- Spanish, French, German
- Italian, Portuguese
- Hindi, Japanese, Korean, Chinese

### Description Styles
- **Short**: Brief, concise descriptions
- **Detailed**: Comprehensive scene analysis

## Technical Architecture

### Backend Components
- **Flask API**: RESTful endpoints with error handling
- **Object Detection**: YOLO v8 for real-time object recognition
- **Text Recognition**: Tesseract OCR for text extraction
- **Narration Engine**: Multiple AI models for description generation
- **TTS System**: Multiple speech synthesis engines

### Frontend Features
- **Progressive Enhancement**: Works without JavaScript
- **Accessibility**: WCAG 2.1 AA compliant
- **Responsive**: Mobile-first design
- **Performance**: Optimized for fast loading

## API Endpoints

- `GET /` - Main application interface
- `POST /api/narrate` - Process image and generate description
- `GET /api/health` - Health check
- `GET /api/status` - System capabilities
- `GET /api/voices` - Available TTS voices
- `GET/POST /api/preferences` - User preferences

## Accessibility Features

### Screen Reader Support
- Semantic HTML structure
- ARIA labels and descriptions
- Live regions for status updates
- Focus management

### Keyboard Navigation
- Tab order optimization
- Keyboard shortcuts (Ctrl+U for upload, Ctrl+Enter to process)
- Focus indicators
- Skip links

### Visual Accessibility
- High contrast mode support
- Reduced motion preferences
- Scalable text and interface
- Clear visual feedback

## Development

### Project Structure
```
videonarrative/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── src/
│   ├── detection.py      # Object detection logic
│   ├── narration.py      # Description generation
│   └── tts.py           # Text-to-speech system
├── static/
│   ├── index.html       # Main interface
│   ├── styles.css       # Styling
│   └── app.js          # Frontend logic
└── README.md           # This file
```

### Contributing
This project is designed to help visually impaired users. When contributing:
- Test with screen readers
- Ensure keyboard accessibility
- Maintain high contrast ratios
- Provide clear error messages

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please check the error messages in the application or review the server logs for detailed information.
