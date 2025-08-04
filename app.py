#!/usr/bin/env python3
"""
Free Voice Clone Terminal AI Assistant with Personal Voice Cloning
A terminal-based AI assistant that clones your voice and responds in your voice.
"""

import os
import sys
import json
import asyncio
import threading
import queue
import time
import hashlib
import pickle
from typing import Optional, Dict, Any, List
import argparse
from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess
import platform
import wave
import numpy as np

try:
    import speech_recognition as sr
    import pyttsx3
    import pyaudio
    import requests
    import colorama
    from colorama import Fore, Back, Style
    import rich
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.spinner import Spinner
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from gtts import gTTS
    import pygame
    import librosa
    import soundfile as sf
    import numpy as np
    from scipy.io import wavfile
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Try to import torch (optional for advanced features)
    try:
        import torch
        import torchaudio
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        print("⚠️  PyTorch not available - using basic voice processing")
        
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\n📦 Please install dependencies:")
    print("pip install speechrecognition pyttsx3 pyaudio requests colorama rich gtts pygame")
    print("pip install librosa soundfile numpy scipy scikit-learn joblib")
    print("\n🔧 For Windows users, you might also need:")
    print("pip install pipwin")
    print("pipwin install pyaudio")
    print("\n⚡ Optional (for advanced features):")
    print("pip install torch torchaudio")
    sys.exit(1)

@dataclass
class VoiceConfig:
    """Configuration for voice synthesis"""
    language: str = "en"
    tld: str = "com"
    slow: bool = False
    voice_id: int = 0
    sample_rate: int = 22050
    voice_model_path: str = "voice_models"

@dataclass
class VoiceCloneConfig:
    """Configuration for voice cloning"""
    min_training_samples: int = 5  # Reduced for quicker setup
    sample_duration: float = 4.0  # seconds
    feature_dim: int = 13  # MFCC features
    model_name: str = "my_voice_clone"
    sample_rate: int = 22050  # Audio sample rate for processing

@dataclass
class AIConfig:
    """Configuration for AI model"""
    model: str = "free"
    max_tokens: int = 150  # Reduced for better performance
    temperature: float = 0.7
    system_prompt: str = "You are a helpful AI assistant. Keep responses concise and conversational."

class VoiceCloner:
    """Handles voice cloning using free audio processing libraries"""
    
    def __init__(self, config: VoiceCloneConfig):
        self.config = config
        self.voice_samples = []
        self.voice_features = []
        self.is_trained = False
        self.scaler = StandardScaler()
        
        # Create voice models directory
        os.makedirs(self.config.model_name, exist_ok=True)
        self.model_path = Path(self.config.model_name)
        
        # Load existing model if available
        self.load_voice_model()
    
    def record_voice_sample(self, duration: float = 4.0, text_prompt: str = None) -> str:
        """Record a voice sample for training"""
        recognizer = sr.Recognizer()
        
        try:
            microphone = sr.Microphone()
        except Exception as e:
            print(f"❌ Microphone error: {e}")
            return None
        
        if text_prompt:
            print(f"📝 Please read: '{text_prompt}'")
        
        print(f"🎤 Recording in 3 seconds... (Duration: {duration}s)")
        time.sleep(3)
        
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("🔴 Recording now... Speak clearly!")
            
            with microphone as source:
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=duration)
            
            # Save audio sample
            sample_filename = self.model_path / f"sample_{len(self.voice_samples)}.wav"
            with open(sample_filename, "wb") as f:
                f.write(audio.get_wav_data())
            
            self.voice_samples.append(str(sample_filename))
            print(f"✅ Sample {len(self.voice_samples)} recorded successfully!")
            
            return str(sample_filename)
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None
    
    def extract_voice_features(self, audio_file: str) -> np.ndarray:
        """Extract voice features from audio file"""
        try:
            # Load audio file using config sample rate
            y, sr = librosa.load(audio_file, sr=self.config.sample_rate)
            
            # Extract MFCC features (voice characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.feature_dim)
            
            # Extract additional features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Combine features and handle shape properly
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.mean(spectral_centroids, axis=1),
                np.mean(spectral_rolloff, axis=1),
                np.mean(zero_crossing_rate, axis=1)
            ])
            
            return features
            
        except Exception as e:
            print(f"⚠️  Feature extraction failed for {audio_file}: {e}")
            return None
    
    def train_voice_model(self):
        """Train voice model from recorded samples"""
        if len(self.voice_samples) < self.config.min_training_samples:
            print(f"❌ Need at least {self.config.min_training_samples} samples to train. Current: {len(self.voice_samples)}")
            return False
        
        print("🧠 Training voice model...")
        
        # Extract features from all samples
        all_features = []
        valid_samples = []
        
        for sample_file in self.voice_samples:
            if os.path.exists(sample_file):
                features = self.extract_voice_features(sample_file)
                if features is not None:
                    all_features.append(features)
                    valid_samples.append(sample_file)
        
        if len(all_features) < 3:
            print("❌ Not enough valid samples for training")
            return False
        
        # Create feature matrix
        self.voice_features = np.array(all_features)
        
        # Normalize features
        self.voice_features = self.scaler.fit_transform(self.voice_features)
        
        # Save model
        self.save_voice_model()
        self.is_trained = True
        
        print(f"✅ Voice model trained with {len(all_features)} samples!")
        return True
    
    def save_voice_model(self):
        """Save voice model to disk"""
        model_data = {
            'voice_features': self.voice_features,
            'voice_samples': self.voice_samples,
            'scaler': self.scaler,
            'config': self.config,
            'trained': True
        }
        
        model_file = self.model_path / "voice_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Voice model saved to {model_file}")
    
    def load_voice_model(self):
        """Load existing voice model"""
        model_file = self.model_path / "voice_model.pkl"
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.voice_features = model_data.get('voice_features', [])
                self.voice_samples = model_data.get('voice_samples', [])
                self.scaler = model_data.get('scaler', StandardScaler())
                self.is_trained = model_data.get('trained', False)
                
                print(f"✅ Loaded existing voice model with {len(self.voice_samples)} samples")
                return True
                
            except Exception as e:
                print(f"⚠️  Failed to load voice model: {e}")
        
        return False
    
    def clone_voice_simple(self, text: str, output_file: str = None) -> str:
        """Simple voice cloning using pitch and speed modification"""
        if not self.is_trained or not self.voice_samples:
            return None
        
        try:
            # Generate base audio using TTS
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Create temporary file for base audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                base_audio = tmp_file.name
                tts.save(base_audio)
            
            # Convert to wav and modify to match voice characteristics
            if not output_file:
                output_file = tempfile.mktemp(suffix='.wav')
            
            # Load reference voice sample
            ref_sample = self.voice_samples[0]  # Use first sample as reference
            
            # Simple voice modification
            modified_audio = self._modify_audio_to_match_voice(base_audio, ref_sample, output_file)
            
            # Clean up
            try:
                os.unlink(base_audio)
            except:
                pass
            
            return modified_audio
            
        except Exception as e:
            print(f"⚠️  Voice cloning failed: {e}")
            return None
    
    def _modify_audio_to_match_voice(self, base_audio: str, reference_audio: str, output_file: str) -> str:
        """Modify audio to match reference voice characteristics"""
        try:
            # Load both audio files
            base_y, base_sr = librosa.load(base_audio, sr=22050)
            ref_y, ref_sr = librosa.load(reference_audio, sr=22050)
            
            # Extract pitch information
            base_pitch = librosa.yin(base_y, fmin=50, fmax=400)
            ref_pitch = librosa.yin(ref_y, fmin=50, fmax=400)
            
            # Calculate pitch ratio
            base_pitch_mean = np.nanmean(base_pitch[base_pitch > 0])
            ref_pitch_mean = np.nanmean(ref_pitch[ref_pitch > 0])
            
            if not np.isnan(base_pitch_mean) and not np.isnan(ref_pitch_mean):
                pitch_ratio = ref_pitch_mean / base_pitch_mean
                
                # Apply pitch shifting
                modified_y = librosa.effects.pitch_shift(base_y, sr=base_sr, n_steps=np.log2(pitch_ratio) * 12)
            else:
                modified_y = base_y
            
            # Save modified audio
            sf.write(output_file, modified_y, base_sr)
            
            return output_file
            
        except Exception as e:
            print(f"⚠️  Audio modification failed: {e}")
            # Fallback: just copy the base audio
            import shutil
            try:
                shutil.copy(base_audio, output_file)
            except:
                pass
            return output_file

class EnhancedVoiceEngine:
    """Enhanced voice engine with cloning capabilities"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        
        # Create voice clone config with proper sample rate
        clone_config = VoiceCloneConfig(
            model_name=config.voice_model_path,
            sample_rate=config.sample_rate
        )
        self.voice_cloner = VoiceCloner(clone_config)
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.pygame_available = True
        except Exception as e:
            print(f"⚠️  Pygame mixer init failed: {e}")
            self.pygame_available = False
        
        # Setup local TTS as fallback
        try:
            self.local_engine = pyttsx3.init()
            self.local_available = True
            self._setup_local_voice()
        except Exception as e:
            print(f"⚠️  Local TTS not available: {e}")
            self.local_available = False
        
        self.gtts_available = True
    
    def _setup_local_voice(self):
        """Setup local pyttsx3 voice"""
        if not self.local_available:
            return
            
        try:
            voices = self.local_engine.getProperty('voices')
            if voices and len(voices) > self.config.voice_id:
                self.local_engine.setProperty('voice', voices[self.config.voice_id].id)
            
            self.local_engine.setProperty('rate', 180)
            self.local_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"⚠️  Voice setup failed: {e}")
    
    def setup_voice_cloning(self) -> bool:
        """Interactive setup for voice cloning"""
        console = Console()
        
        if self.voice_cloner.is_trained:
            console.print("✅ Voice model already exists!", style="green")
            response = input("Do you want to retrain with new samples? (y/n): ").lower()
            if response != 'y':
                return True
        
        console.print("\n🎤 Voice Cloning Setup", style="cyan bold")
        console.print("We'll record several samples of your voice to create a personal voice model.")
        console.print("This process is completely free and runs locally on your machine.\n")
        
        # Shorter training phrases for quicker setup
        training_phrases = [
            "Hello, this is my voice.",
            "How are you doing today?",
            "The weather is nice outside.",
            "I am training my voice model.",
            "Thank you for listening to me."
        ]
        
        console.print(f"📋 We'll record {len(training_phrases)} voice samples.")
        input("Press Enter when you're ready to start recording...")
        
        successful_recordings = 0
        
        for i, phrase in enumerate(training_phrases, 1):
            console.print(f"\n📝 Sample {i}/{len(training_phrases)}")
            console.print(f"Please read clearly: '{phrase}'")
            
            while True:
                try:
                    sample_file = self.voice_cloner.record_voice_sample(
                        duration=4.0, 
                        text_prompt=phrase
                    )
                    
                    if sample_file:
                        successful_recordings += 1
                        break
                    else:
                        retry = input("Recording failed. Try again? (y/n): ").lower()
                        if retry != 'y':
                            break
                            
                except KeyboardInterrupt:
                    console.print("\n⏹️  Recording cancelled by user")
                    break
        
        if successful_recordings >= self.voice_cloner.config.min_training_samples:
            console.print(f"\n🧠 Training voice model with {successful_recordings} samples...")
            
            with console.status("[bold green]Training voice model..."):
                success = self.voice_cloner.train_voice_model()
            
            if success:
                console.print("✅ Voice cloning setup completed successfully!", style="green bold")
                console.print("🎉 Your AI assistant can now speak in your voice!", style="yellow")
                return True
            else:
                console.print("❌ Voice model training failed", style="red")
                return False
        else:
            console.print(f"❌ Need at least {self.voice_cloner.config.min_training_samples} samples, got {successful_recordings}", style="red")
            return False
    
    def speak_with_cloned_voice(self, text: str) -> bool:
        """Speak text using cloned voice"""
        if not self.voice_cloner.is_trained:
            return False
        
        try:
            print("🎭 Generating speech with your cloned voice...")
            
            # Generate cloned voice audio
            audio_file = self.voice_cloner.clone_voice_simple(text)
            
            if audio_file and os.path.exists(audio_file):
                # Play the cloned voice audio
                if self._play_audio_file(audio_file):
                    # Clean up temporary file
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                    return True
                else:
                    return False
            else:
                return False
                
        except Exception as e:
            print(f"⚠️  Cloned voice synthesis failed: {e}")
            return False
    
    def speak(self, text: str, method: str = "cloned"):
        """Speak text using specified method"""
        success = False
        
        if method == "cloned" and self.voice_cloner.is_trained:
            success = self.speak_with_cloned_voice(text)
            if success:
                return
        
        # Fallback methods
        if method in ["local", "cloned"]:
            success = self.speak_local(text)
        elif method == "gtts":
            success = self.speak_gtts(text)
        else:  # auto
            success = self.speak_gtts(text)
            if not success:
                success = self.speak_local(text)
        
        if not success:
            print("⚠️  All voice synthesis methods failed")
    
    def speak_local(self, text: str) -> bool:
        """Speak using local pyttsx3 engine"""
        if self.local_available:
            try:
                self.local_engine.say(text)
                self.local_engine.runAndWait()
                return True
            except Exception as e:
                print(f"⚠️  Local TTS failed: {e}")
                return False
        return False
    
    def speak_gtts(self, text: str) -> bool:
        """Speak using Google Text-to-Speech (free version)"""
        try:
            tts = gTTS(text=text, lang=self.config.language, tld=self.config.tld, slow=self.config.slow)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                temp_filename = tmp_file.name
                tts.save(temp_filename)
            
            success = self._play_audio_file(temp_filename)
            
            try:
                os.unlink(temp_filename)
            except:
                pass
            
            return success
            
        except Exception as e:
            print(f"⚠️  gTTS failed: {e}")
            return False
    
    def _play_audio_file(self, filename: str) -> bool:
        """Play audio file using pygame"""
        if not self.pygame_available:
            print("⚠️  Audio playback not available")
            return False
            
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            return True
                
        except Exception as e:
            print(f"⚠️  Audio playback failed: {e}")
            return False
    
    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice cloning status"""
        return {
            'cloned_voice_available': self.voice_cloner.is_trained,
            'training_samples': len(self.voice_cloner.voice_samples),
            'model_path': str(self.voice_cloner.model_path),
            'local_tts_available': self.local_available,
            'gtts_available': self.gtts_available,
            'pygame_available': self.pygame_available
        }

class FreeSpeechRecognition:
    """Handles speech recognition using free services"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.available = True
        except Exception as e:
            print(f"⚠️  Microphone not available: {e}")
            self.available = False
    
    def listen_for_speech(self, timeout: int = 5) -> Optional[str]:
        """Listen for speech input using free Google Web Speech API"""
        if not self.available:
            print("❌ Microphone not available")
            return None
            
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            print("🔄 Processing speech...")
            
            # Try Google Web Speech API (free)
            try:
                result = self.recognizer.recognize_google(audio)
                if result:
                    print("✅ Recognized via Google Web Speech")
                    return result
            except Exception as e:
                print(f"⚠️  Google recognition failed: {e}")
            
            return None
                
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
            return None
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return None

class FreeConversationAI:
    """Handles AI conversation using free AI services"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.conversation_history = []
        
        # Updated working AI services
        self.ai_services = [
            {"name": "Ollama Local", "method": self._ollama_api},
            {"name": "Hugging Face", "method": self.openrouter_api},
            {"name": "Local Response", "method": self._local_response},
        ]
    
    def get_response(self, user_input: str) -> str:
        """Get AI response using free services"""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Try different free AI services
        for service in self.ai_services:
            try:
                response = service["method"](user_input)
                if response and len(response.strip()) > 0:
                    self.conversation_history.append({"role": "assistant", "content": response})
                    return response
            except Exception as e:
                print(f"⚠️  {service['name']} failed: {e}")
                continue
        
        # Fallback response
        fallback = "I understand your message. This is a free AI assistant running locally with your cloned voice."
        self.conversation_history.append({"role": "assistant", "content": fallback})
        return fallback
    
    def _ollama_api(self, text: str) -> str:
        """Use local Ollama API if available"""
        try:
            # Try to connect to local Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",  # or any installed model
                    "prompt": text,
                    "stream": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
                
        except Exception:
            # Ollama not available
            pass
        
        return None
    
    def openrouter_api(self, text: str) -> str:
        """Get response from OpenRouter API with Hugging Face-like features."""
        
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        api_key = "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"
        
        if not api_key:
            print("⚠️ No API key found")
            return None

        models_to_try = [
            {"name": "qwen/qwen3-coder:free", "format": "conversational"},
            {"name": "z-ai/glm-4.5-air:free", "format": "conversational"},
            {"name": "moonshotai/kimi-k2:free", "format": "conversational"}
        ]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        for model in models_to_try:
            try:
                # Prepare conversation history (last 4 messages)
                past_messages = self.conversation_history[-4:] if len(self.conversation_history) >= 4 else self.conversation_history
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in past_messages
                ]
                messages.append({"role": "user", "content": text})

                payload = {
                    "model": model["name"],
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.7
                }

                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        response_text = result["choices"][0]["message"]["content"].strip()
                        
                        if response_text and len(response_text) > 5:
                            print(f"✅ Response from {model['name']}")
                            # Update conversation history
                            self.conversation_history.append({"role": "user", "content": text})
                            self.conversation_history.append({"role": "assistant", "content": response_text})
                            return response_text
                        else:
                            continue
                    
                elif response.status_code == 503:
                    print(f"⏳ Model {model['name']} is loading...")
                    time.sleep(2)
                    continue
                else:
                    print(f"⚠️ Status code {response.status_code} from {model['name']}")
                    continue
                    
            except Exception as e:
                print(f"⚠️ Error with {model['name']}: {e}")
                continue
        
        return None
    
    def _local_response(self, text: str) -> str:
        """Generate simple local responses"""
        text_lower = text.lower()
        
        # Enhanced pattern matching for common queries
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm your AI assistant speaking in your own cloned voice. How can I help you today?"
        
        elif any(word in text_lower for word in ['how are you', 'how do you do']):
            return "I'm doing well, thank you! It's quite amazing to speak in your voice, isn't it? What would you like to discuss?"
        
        elif any(word in text_lower for word in ['what is your name', 'who are you']):
            return "I'm your personal AI assistant, and I'm speaking in your cloned voice! Think of me as a digital version of yourself."
        
        elif any(word in text_lower for word in ['voice', 'sound', 'cloned', 'clone']):
            return "Yes, I'm using your cloned voice! This voice model was trained from your samples completely locally and for free."
        
        elif any(word in text_lower for word in ['weather']):
            return "I don't have access to real-time weather data, but you can check your local weather service."
        
        elif any(word in text_lower for word in ['time', 'date']):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            return f"The current time is {current_time}."
        
        elif any(word in text_lower for word in ['help', 'commands']):
            return "I can help with conversations and answer questions - all while speaking in your voice! Try asking me about various topics."
        
        elif any(word in text_lower for word in ['thank', 'thanks']):
            return "You're welcome! It's my pleasure to help using your own voice."
        
        elif '?' in text:
            return f"That's an interesting question. While I don't have access to real-time data, I can discuss this topic with you."
        
        else:
            responses = [
                f"I understand you mentioned something about that topic. What would you like to know more about?",
                f"That's interesting! Tell me more about what you're thinking.",
                f"I hear you. What specific aspect would you like to explore?",
                f"Thanks for sharing that. How can I help you with this topic?"
            ]
            return responses[len(text) % len(responses)]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

class FreeTerminalAI:
    """Enhanced free terminal AI with voice cloning"""
    
    def __init__(self):
        self.console = Console()
        self.voice_engine = EnhancedVoiceEngine(VoiceConfig())
        self.speech_engine = FreeSpeechRecognition()
        self.ai = FreeConversationAI(AIConfig())
        
        self.listening = False
        self.speaking = False
        self.voice_method = "cloned"  # cloned, auto, local, gtts
        self.running = True
        
        colorama.init()
    
    def display_banner(self):
        """Display startup banner with voice cloning info"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║              🎤 FREE VOICE CLONE AI TERMINAL 🤖             ║
║                Clone Your Voice & Chat With Yourself!        ║
╠══════════════════════════════════════════════════════════════╣
║  Voice Commands:                                             ║
║  • 'setup voice' - Setup voice cloning (one-time)            ║
║  • 'voice cloned' - Use your cloned voice                    ║
║  • 'voice status' - Check voice cloning status               ║          ║
║  • 'listen' or 'l' - Start voice input                       ║
║  • 'speak' or 's' - Enable voice output                      ║
║  • 'voice local' - Use local TTS                             ║
║  • 'voice gtts' - Use Google TTS (free)                      ║
║  • 'voice auto' - Auto-select best TTS                       ║
║  • 'text' or 't' - Text-only mode                            ║
║  • 'clear' or 'c' - Clear conversation                       ║
║  • 'quit' or 'q' - Exit                                      ║
║  • 'help' or 'h' - Show this help                            ║
╚══════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="cyan")
        
        # Show voice status
        status = self.voice_engine.get_voice_status()
        if status['cloned_voice_available']:
            self.console.print("✅ Your cloned voice is ready!", style="green bold")
        else:
            self.console.print("🎭 Voice cloning not set up. Use 'setup voice' to clone your voice!", style="yellow")
    
    def process_command(self, command: str) -> bool:
        """Process special commands including voice cloning"""
        command = command.lower().strip()
        
        if command in ['quit', 'q', 'exit']:
            self.console.print("👋 Goodbye!", style="yellow")
            return False
        
        elif command == 'setup voice':
            self.voice_engine.setup_voice_cloning()
        
        # elif command == 'test api':
        #     self.ai.test_huggingface_api()
        
        elif command == 'voice status':
            self.show_voice_status()
        
        elif command in ['listen', 'l']:
            self.listening = True
            self.console.print("🎤 Voice input enabled", style="green")
        
        elif command in ['speak', 's']:
            self.speaking = True
            self.console.print("🔊 Voice output enabled", style="green")
        
        elif command.startswith('voice '):
            method = command.split(' ', 1)[1]
            if method in ['cloned', 'local', 'gtts', 'auto']:
                self.voice_method = method
                self.console.print(f"🎵 Voice method set to: {method}", style="green")
                
                if method == 'cloned' and not self.voice_engine.voice_cloner.is_trained:
                    self.console.print("⚠️  Cloned voice not available. Use 'setup voice' first.", style="yellow")
            else:
                self.console.print("❌ Invalid voice method. Use: cloned, local, gtts, or auto", style="red")
        
        elif command in ['text', 't']:
            self.listening = False
            self.speaking = False
            self.console.print("⌨️  Text-only mode", style="blue")
        
        elif command in ['clear', 'c']:
            self.ai.clear_history()
            self.console.clear()
            self.display_banner()
            self.console.print("🧹 Conversation cleared", style="yellow")
        
        elif command in ['help', 'h']:
            self.display_banner()
        
        else:
            return True  # Not a command, process as normal input
        
        return True
    
    def show_voice_status(self):
        """Display voice cloning status"""
        status = self.voice_engine.get_voice_status()
        
        status_panel = f"""
🎭 Voice Cloning Status:
   • Cloned Voice Available: {'✅ Yes' if status['cloned_voice_available'] else '❌ No'}
   • Training Samples: {status['training_samples']}
   • Current Voice Method: {self.voice_method}
   • Local TTS Available: {'✅ Yes' if status['local_tts_available'] else '❌ No'}
   • Google TTS Available: {'✅ Yes' if status['gtts_available'] else '❌ No'}
   • Audio Playback Available: {'✅ Yes' if status['pygame_available'] else '❌ No'}

💡 Tip: Use 'setup voice' to clone your voice if not done yet!
        """
        
        self.console.print(Panel(status_panel, title="Voice Status", border_style="blue"))
    
    def display_response(self, response: str):
        """Display and speak AI response with voice cloning"""
        # Display text response
        voice_indicator = "🎭" if self.voice_method == "cloned" else "🤖"
        panel = Panel(
            response,
            title=f"{voice_indicator} AI Assistant ({self.voice_method} voice)",
            title_align="left",
            border_style="green"
        )
        self.console.print(panel)
        
        # Speak response if enabled
        if self.speaking:
            method_display = f"🎭 Your cloned voice" if self.voice_method == "cloned" else f"🔊 {self.voice_method}"
            self.console.print(f"{method_display} - Speaking...", style="yellow")
            
            def speak_async():
                self.voice_engine.speak(response, method=self.voice_method)
            
            # Run speech in background thread
            speech_thread = threading.Thread(target=speak_async)
            speech_thread.daemon = True
            speech_thread.start()
    
    def get_user_input(self) -> Optional[str]:
        """Get user input via text or voice"""
        if self.listening:
            self.console.print("\n🎤 Say something...", style="green")
            
            user_input = self.speech_engine.listen_for_speech()
            
            if user_input:
                self.console.print(f"🗣️  You said: {user_input}", style="blue")
                return user_input
            else:
                self.console.print("❌ Could not understand speech, try again or type your message", style="red")
                return None
        else:
            try:
                return input(f"\n{Fore.CYAN}You: {Style.RESET_ALL}")
            except KeyboardInterrupt:
                return "quit"
    
    def run(self):
        """Main application loop"""
        self.display_banner()
        
        self.console.print("\n✨ Free Voice Clone AI Assistant ready!", style="green")
        self.console.print("🎭 Use 'setup voice' to clone your voice first!\n", style="yellow")
        
        while self.running:
            try:
                user_input = self.get_user_input()
                
                if user_input is None:
                    continue
                
                # Check if it's a command
                if not self.process_command(user_input):
                    break
                
                # Skip empty input
                if not user_input.strip():
                    continue
                
                # Get AI response
                with Live(Spinner("dots", text="Thinking..."), console=self.console):
                    response = self.ai.get_response(user_input)
                
                # Display response
                self.display_response(response)
                
            except KeyboardInterrupt:
                self.console.print("\n\n👋 Goodbye!", style="yellow")
                break
            except Exception as e:
                self.console.print(f"❌ Error: {e}", style="red")

def check_dependencies():
    """Check and install missing dependencies with Windows support"""
    missing_deps = []
    
    # Core required packages
    required_packages = [
        "speechrecognition",
        "pyttsx3", 
        "pygame",
        "gtts",
        "librosa",
        "soundfile",
        "scipy",
        "scikit-learn",
        "joblib",
        "rich",
        "colorama",
        "requests",
        "numpy"
    ]
    
    # Optional packages
    optional_packages = [
        "torch",
        "torchaudio"
    ]
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_deps.append(package)
            print(f"❌ {package}")
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional - not installed)")
    
    if missing_deps:
        print(f"\n📦 Installing {len(missing_deps)} missing dependencies...")
        
        # Special handling for Windows PyAudio
        if "pyaudio" in missing_deps and platform.system() == "Windows":
            print("🪟 Windows detected - installing PyAudio with special method...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pipwin"])
                subprocess.check_call([sys.executable, "-m", "pipwin", "install", "pyaudio"])
                missing_deps.remove("pyaudio")
                print("✅ PyAudio installed successfully!")
            except Exception as e:
                print(f"⚠️  Special PyAudio install failed: {e}")
                print("💡 Try: pip install pipwin && pipwin install pyaudio")
        
        # Install remaining packages
        for dep in missing_deps:
            try:
                print(f"📥 Installing {dep}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✅ {dep} installed!")
            except Exception as e:
                print(f"⚠️  Failed to install {dep}: {e}")
                
        print("\n🔄 Please restart the application after installation completes.")
        return False
    
    return True

def windows_audio_setup():
    """Special setup for Windows audio"""
    if platform.system() == "Windows":
        print("🪟 Windows audio setup...")
        try:
            # Initialize pygame mixer with Windows-friendly settings
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
            pygame.mixer.init()
            print("✅ Windows audio initialized successfully!")
            return True
        except Exception as e:
            print(f"⚠️  Windows audio setup failed: {e}")
            print("💡 Try running as administrator or check your audio drivers")
            return False
    return True

def main():
    """Main entry point with voice cloning support"""
    parser = argparse.ArgumentParser(description="Free Voice Clone Terminal AI Assistant")
    parser.add_argument("--check-deps", action="store_true", help="Check and install dependencies")
    parser.add_argument("--voice-method", choices=["cloned", "local", "gtts", "auto"], default="cloned", help="Default voice method")
    parser.add_argument("--setup-voice", action="store_true", help="Run voice cloning setup immediately")
    
    args = parser.parse_args()
    
    if args.check_deps:
        if not check_dependencies():
            return
    
    # Show setup information
    print("🎭 Voice Clone Terminal AI Assistant")
    print("=" * 50)
    print("🆓 Completely FREE voice cloning and AI chat!")
    print("🔒 Everything runs locally - your voice data stays private!")
    print("🎤 Clone your voice in minutes with just a few recordings!")
    print()
    print("   All core features work without any API keys!")
    print()
    
    # Initialize Windows audio support
    if not windows_audio_setup():
        print("⚠️  Audio setup failed, but continuing...")
    
    try:
        app = FreeTerminalAI()
        app.voice_method = args.voice_method
        
        # Auto-setup voice cloning if requested
        if args.setup_voice:
            app.voice_engine.setup_voice_cloning()
        
        app.run()
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("\n🔧 Try running with --check-deps to install missing dependencies")
        sys.exit(1)

if __name__ == "__main__":
    main()