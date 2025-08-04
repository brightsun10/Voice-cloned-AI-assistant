# Voice Cloned Terminal AI Assistant

A free, open-source terminal-based AI assistant that clones your voice and responds in your own voice, running completely locally to ensure privacy. No external API keys are required for core functionality.

## Features
- **Voice Cloning**: Record a few samples of your voice to create a personalized voice model.
- **Speech Recognition**: Use free Google Web Speech API for speech-to-text input.
- **Text-to-Speech**: Supports multiple TTS methods:
  - Cloned voice (based on your voice samples)
  - Local TTS (pyttsx3)
  - Google TTS (gTTS, free version)
  - Auto-selection of the best available method
- **AI Conversation**: Powered by local AI or free external APIs:
  - Local responses for basic queries
  - Optional integration with local Ollama or free Hugging Face-like models via OpenRouter
- **Privacy-Focused**: All voice cloning and processing happens locally, ensuring your voice data stays private.
- **Cross-Platform**: Works on Windows, macOS, and Linux with special Windows audio handling.

## Requirements
- Python 3.6+
- Microphone for voice input and cloning
- Speakers/headphones for audio output
- Internet connection (optional, for Google TTS and OpenRouter API)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voice-clone-ai-assistant.git
   cd voice-clone-ai-assistant
   ```

2. Install dependencies:
   ```bash
   pip install speechrecognition pyttsx3 pyaudio requests colorama rich gtts pygame librosa soundfile numpy scipy scikit-learn joblib
   ```

3. For Windows users (PyAudio installation):
   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```

4. Optional dependencies for advanced features:
   ```bash
   pip install torch torchaudio
   ```

5. Check and install dependencies automatically:
   ```bash
   python voice_clone_assistant.py --check-deps
   ```

## Usage
1. Run the application:
   ```bash
   python voice_clone_assistant.py
   ```

2. Available commands:
   - `setup voice`: Start voice cloning setup (record 5 short voice samples)
   - `voice status`: Check voice cloning status
   - `listen` or `l`: Enable voice input
   - `speak` or `s`: Enable voice output
   - `voice cloned`: Use your cloned voice
   - `voice local`: Use local TTS
   - `voice gtts`: Use Google TTS
   - `voice auto`: Auto-select best voice method
   - `text` or `t`: Switch to text-only mode
   - `clear` or `c`: Clear conversation history
   - `quit` or `q`: Exit the application
   - `help` or `h`: Show help menu

3. To set up voice cloning immediately:
   ```bash
   python voice_clone_assistant.py --setup-voice
   ```

4. To specify a default voice method:
   ```bash
   python voice_clone_assistant.py --voice-method cloned
   ```

## Voice Cloning Setup
1. Run `setup voice` or use the `--setup-voice` flag.
2. Follow prompts to record 5 short voice samples (4 seconds each).
3. The system will train a voice model based on your samples.
4. Once trained, use `voice cloned` to respond in your own voice.

## Notes
- Voice cloning is processed locally and requires no external services.
- The quality of the cloned voice improves with more samples (minimum 5 required).
- If you encounter audio issues on Windows, try running as administrator or updating audio drivers.
- For advanced AI responses, set up a local Ollama server or use the provided OpenRouter API key for free Hugging Face-like models.

## Troubleshooting
- **Microphone issues**: Ensure your microphone is properly connected and configured.
- **Audio playback issues**: Check speaker/headphone connections and try `voice local` or `voice gtts`.
- **Dependency errors**: Run with `--check-deps` to automatically install missing packages.
- **Voice cloning quality**: Record in a quiet environment and speak clearly during setup.

## License
MIT License - feel free to use, modify, and distribute this code.

## Contributing
Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## Disclaimer
This project uses free services and local processing. The cloned voice quality may vary based on hardware and sample quality. For professional-grade voice cloning, consider commercial solutions.
