import torch
import numpy as np
import librosa
import tempfile
import os
from typing import Optional, Dict, Any
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)

class PlantAudioProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.processor = None
        self._load_models()

    # Loads the HuggingFace Whisper model for transcribing speech about plant symptoms.
    def _load_models(self):
        try:
            logger.info("Loading Whisper model for audio processing...")
            model_name = "openai/whisper-small"

            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

            if torch.cuda.is_available() and self.device == 'cuda':
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info(f"Whisper model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.model = None
            self.processor = None

    # Turns the user’s speech (audio) into written text
    # 1- Loads and processes the audio
    # 2- Runs Whisper model to transcribe speech
    # 3- Cleans up the transcript for botanical use
    def transcribe_audio(self, audio_file) -> str:
        if self.model is None or self.processor is None:
            return "Error: Whisper model not loaded"

        try:
            audio_data = self._load_audio(audio_file)
            if audio_data is None:
                return "Error: Could not load audio file"

            inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt")

            if torch.cuda.is_available() and self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                predicted_ids = self.model.generate(**inputs)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Changed to botanical context preprocessing
            botanical_transcript = self.preprocess_for_botanical_context(transcription)
            return botanical_transcript

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error: Transcription failed - {str(e)}"

    # Turns audio into a feature vector (numerical summary) for further AI analysis
    # 1- Loads the audio file
    # 2- Extracts MFCCs (standard features in audio analysis)
    # 3- Combines stats (mean, std, max, min) into a single vector
    def extract_features(self, audio_file) -> Optional[np.ndarray]:
        try:
            audio_data = self._load_audio(audio_file)
            if audio_data is None:
                return None

            mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)

            features = np.concatenate([
                mfccs.mean(axis=1),
                mfccs.std(axis=1),
                mfccs.max(axis=1),
                mfccs.min(axis=1)
            ])

            target_size = 384
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            else:
                features = features[:target_size]

            return features

        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None

    # Reads audio from different file formats (path or in-memory bytes)
    # 1- Handles both file paths and uploaded files
    # 2- Always converts to 16kHz mono format for consistency
    def _load_audio(self, audio_file) -> Optional[np.ndarray]:
        try:
            if isinstance(audio_file, str):
                audio_data, sr = librosa.load(audio_file, sr=16000, mono=True)
            else:
                if hasattr(audio_file, 'read'):
                    audio_bytes = audio_file.read()
                else:
                    audio_bytes = audio_file

                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name

                try:
                    audio_data, sr = librosa.load(tmp_path, sr=16000, mono=True)
                finally:
                    os.unlink(tmp_path)

            return audio_data

        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None

    # Checks if the audio is clear and useful
    # 1- Measures duration and loudness (RMS energy)
    # 2- Flags issues (too short, too loud, too quiet, too long)
    # 3- Gives a quality score and a “good/fair/poor” label
    def analyze_audio_quality(self, audio_file) -> Dict[str, Any]:
        try:
            audio_data = self._load_audio(audio_file)
            if audio_data is None:
                return {"quality": "error", "issues": ["Could not load audio"]}

            duration = len(audio_data) / 16000
            rms_energy = np.sqrt(np.mean(audio_data**2))

            issues = []
            quality_score = 100

            if duration < 1:
                issues.append("Audio too short")
                quality_score -= 30
            elif duration > 300:
                issues.append("Audio very long")
                quality_score -= 10

            if rms_energy < 0.01:
                issues.append("Audio too quiet")
                quality_score -= 25
            elif rms_energy > 0.5:
                issues.append("Audio may be too loud")
                quality_score -= 15

            quality = "good" if quality_score >= 70 else "fair" if quality_score >= 50 else "poor"

            return {
                "quality": quality,
                "score": max(0, quality_score),
                "duration": duration,
                "energy": rms_energy,
                "issues": issues
            }

        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return {"quality": "error", "issues": ["Quality analysis failed"]}

    # Replaces casual words in the transcript with formal botanical terms
    # Example: “spots” becomes “lesions”; “yellow leaves” becomes “chlorotic leaves”
    def preprocess_for_botanical_context(self, transcript: str) -> str:
        try:
            botanical_corrections = {
                'spots': 'lesions',
                'spotty': 'lesions',
                'powder': 'powdery mildew',
                'powdery': 'powdery mildew',
                'mold': 'mold or mildew',
                'yellow': 'chlorotic',
                'yellowing': 'chlorosis',
                'brown': 'necrotic',
                'browning': 'necrosis',
                'dead': 'necrotic tissue',
                'curly': 'curling',
                'wilting': 'wilted',
                'rust': 'rust-like pustules'
            }

            processed = transcript.lower()

            for informal, formal in botanical_corrections.items():
                processed = processed.replace(informal, formal)

            return processed.capitalize()

        except Exception as e:
            logger.error(f"Error preprocessing transcript: {e}")
            return transcript
