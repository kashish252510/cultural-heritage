// Accessible Image Narrator - JavaScript
'use strict';

// DOM elements
const elements = {
    fileInput: document.getElementById('fileInput'),
    uploadArea: document.querySelector('.upload-area'),
    browseButton: document.getElementById('browseButton'),
    cameraButton: document.getElementById('cameraButton'),
    narrateBtn: document.getElementById('narrateUpload'),
    styleSelect: document.getElementById('style'),
    languageSelect: document.getElementById('language'),
    voiceSpeedSelect: document.getElementById('voice-speed'),
    statusIndicator: document.getElementById('status'),
    textOutput: document.getElementById('text'),
    audioElement: document.getElementById('audio'),
    audioControls: document.getElementById('audioControls'),
    replayBtn: document.getElementById('replayAudio')
};

// State management
let currentFile = null;
let isProcessing = false;
let currentAudioSrc = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    loadPreferences();
    setupEventListeners();
    updateStatus('Ready to upload an image', 'success');
    announceToScreenReader('Accessible Image Narrator loaded. Please upload an image to get started.');
}

function setupEventListeners() {
    // File input handling
    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', handleFileSelect);
    }

    // Upload area interactions
    if (elements.uploadArea) {
        elements.uploadArea.addEventListener('click', (e) => {
            e.preventDefault();
            elements.fileInput?.click();
        });
        elements.uploadArea.addEventListener('keydown', handleUploadAreaKeydown);
        elements.uploadArea.addEventListener('dragover', handleDragOver);
        elements.uploadArea.addEventListener('dragleave', handleDragLeave);
        elements.uploadArea.addEventListener('drop', handleDrop);
    }

    // Browse button
    if (elements.browseButton) {
        elements.browseButton.addEventListener('click', () => {
            elements.fileInput?.click();
        });
    }

    // Camera button
    if (elements.cameraButton) {
        elements.cameraButton.addEventListener('click', openCamera);
    }

    // Process button
    if (elements.narrateBtn) {
        elements.narrateBtn.addEventListener('click', processImage);
    }

    // Preference changes
    if (elements.styleSelect) {
        elements.styleSelect.addEventListener('change', savePreferences);
    }
    if (elements.languageSelect) {
        elements.languageSelect.addEventListener('change', savePreferences);
    }
    if (elements.voiceSpeedSelect) {
        elements.voiceSpeedSelect.addEventListener('change', savePreferences);
    }

    // Audio controls
    if (elements.replayBtn) {
        elements.replayBtn.addEventListener('click', replayAudio);
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleUploadAreaKeydown(event) {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        elements.fileInput?.click();
    }
}

function handleDragOver(event) {
    event.preventDefault();
    elements.uploadArea?.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    elements.uploadArea?.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    elements.uploadArea?.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        updateStatus('Please select a valid image file (JPG, PNG, GIF, WebP)', 'error');
        announceToScreenReader('Invalid file type. Please select an image file.');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        updateStatus('File too large. Please select an image smaller than 10MB', 'error');
        announceToScreenReader('File too large. Please select a smaller image.');
        return;
    }

    currentFile = file;
    elements.narrateBtn.disabled = false;
    updateStatus(`Image selected: ${file.name} (${formatFileSize(file.size)})`, 'success');
    announceToScreenReader(`Image selected: ${file.name}. You can now click the Describe Image button to process it.`);
}

function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        updateStatus('Camera not supported in this browser', 'error');
        announceToScreenReader('Camera not supported. Please use the browse files option instead.');
        return;
    }

    // Create a temporary file input for camera capture
    const cameraInput = document.createElement('input');
    cameraInput.type = 'file';
    cameraInput.accept = 'image/*';
    cameraInput.capture = 'environment'; // Use back camera on mobile
    
    cameraInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });
    
    cameraInput.click();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function processImage() {
    if (!currentFile || isProcessing) return;

    isProcessing = true;
    elements.narrateBtn.disabled = true;
    updateStatus('Processing image...', 'processing');
    announceToScreenReader('Processing your image. Please wait.');

    try {
        const dataUrl = await fileToDataURL(currentFile);
        const result = await narrateImage(dataUrl);
        
        if (result.success) {
            displayResult(result);
            announceToScreenReader('Image processing complete. Description is now available.');
        } else {
            throw new Error(result.error || 'Processing failed');
        }
    } catch (error) {
        console.error('Processing error:', error);
        updateStatus(`Error: ${error.message}`, 'error');
        announceToScreenReader(`Error processing image: ${error.message}`);
    } finally {
        isProcessing = false;
        elements.narrateBtn.disabled = false;
    }
}

async function narrateImage(imageBase64) {
    const preferences = getPreferences();
    
    try {
        const response = await fetch('/api/narrate', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                image_base64: imageBase64, 
                preferences: preferences 
            })
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Request failed');
        }

        return {
            success: true,
            text: data.text || 'No description available',
            audio: data.audio_base64,
            detections: data.detections,
            processingTime: data.elapsed_ms
        };
    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}

function displayResult(result) {
    // Update status
    const timeInfo = result.processingTime ? ` (processed in ${result.processingTime}ms)` : '';
    updateStatus(`Description generated successfully${timeInfo}`, 'success');

    // Display text description
    elements.textOutput.textContent = result.text;

    // Handle audio
    if (result.audio) {
        currentAudioSrc = result.audio;
        elements.audioElement.src = result.audio;
        elements.audioControls.style.display = 'flex';
        
        // Auto-play audio for accessibility
        elements.audioElement.play().catch(error => {
            console.log('Auto-play prevented:', error);
            announceToScreenReader('Audio description is ready. Click the play button to listen.');
        });
    } else {
        // Fallback to browser TTS
        speakText(result.text);
    }
}

function speakText(text) {
    if (!window.speechSynthesis) {
        announceToScreenReader('Text-to-speech not available in this browser.');
        return;
    }

    // Stop any current speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    const preferences = getPreferences();
    
    utterance.lang = preferences.language || 'en';
    utterance.rate = getSpeechRate(preferences.voiceSpeed);
    utterance.volume = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
        announceToScreenReader('Speaking description...');
    };

    utterance.onend = () => {
        announceToScreenReader('Description finished.');
    };

    utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        announceToScreenReader('Error with text-to-speech.');
    };

    window.speechSynthesis.speak(utterance);
}

function getSpeechRate(speed) {
    switch (speed) {
        case 'slow': return 0.7;
        case 'fast': return 1.3;
        default: return 1.0;
    }
}

function replayAudio() {
    if (currentAudioSrc) {
        elements.audioElement.currentTime = 0;
        elements.audioElement.play().catch(error => {
            console.error('Audio replay error:', error);
            announceToScreenReader('Error replaying audio.');
        });
    } else {
        // Replay with TTS
        const text = elements.textOutput.textContent;
        if (text) {
            speakText(text);
        }
    }
}

function updateStatus(message, type) {
    if (elements.statusIndicator) {
        elements.statusIndicator.textContent = message;
        elements.statusIndicator.className = `status-indicator ${type}`;
    }
}

function announceToScreenReader(message) {
    // Create a temporary element for screen reader announcements
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    // Remove after announcement
    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 1000);
}

function handleKeyboardShortcuts(event) {
    // Ctrl/Cmd + U to focus upload area
    if ((event.ctrlKey || event.metaKey) && event.key === 'u') {
        event.preventDefault();
        elements.uploadArea?.focus();
    }
    
    // Ctrl/Cmd + Enter to process image
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        if (!elements.narrateBtn.disabled) {
            processImage();
        }
    }
    
    // Space to replay audio when focused on replay button
    if (event.key === ' ' && event.target === elements.replayBtn) {
        event.preventDefault();
        replayAudio();
    }
}

// Preferences management
function loadPreferences() {
    const saved = JSON.parse(localStorage.getItem('accessible_narrator_prefs') || '{}');
    
    if (elements.styleSelect && saved.style) {
        elements.styleSelect.value = saved.style;
    }
    if (elements.languageSelect && saved.language) {
        elements.languageSelect.value = saved.language;
    }
    if (elements.voiceSpeedSelect && saved.voiceSpeed) {
        elements.voiceSpeedSelect.value = saved.voiceSpeed;
    }
}

function savePreferences() {
    const preferences = {
        style: elements.styleSelect?.value || 'detailed',
        language: elements.languageSelect?.value || 'en',
        voiceSpeed: elements.voiceSpeedSelect?.value || 'normal'
    };
    
    localStorage.setItem('accessible_narrator_prefs', JSON.stringify(preferences));
}

function getPreferences() {
    return {
        style: elements.styleSelect?.value || 'detailed',
        language: elements.languageSelect?.value || 'en',
        voiceSpeed: elements.voiceSpeedSelect?.value || 'normal'
    };
}

// Utility functions
function fileToDataURL(file) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result);
		reader.onerror = reject;
		reader.readAsDataURL(file);
	});
}

// Add screen reader only class for announcements
const style = document.createElement('style');
style.textContent = `
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }
`;
document.head.appendChild(style);
