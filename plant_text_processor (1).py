import numpy as np
from typing import Dict, Any
import logging
import re
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class PlantTextProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        # Changed to load botanical keywords
        self.botanical_keywords = self._load_botanical_keywords()
        self._load_models()

    # Loads the sentence transformer (MiniLM) to turn text into embeddings (vectors)
    def _load_models(self):
        try:
            logger.info("Loading sentence transformer for text processing...")
            model_name = 'all-MiniLM-L6-v2'

            self.model = SentenceTransformer(model_name)

            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to(self.device)

            logger.info(f"Sentence transformer loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            self.model = None

    # Sets up lists of important botanical keywords for symptoms, descriptors, time, and severity
    # Usage: Helps in matching relevant words in the user's text
    def _load_botanical_keywords(self) -> Dict[str, list]:
        return {
            'plant_symptoms': [
                'lesion', 'spot', 'blight', 'mildew', 'rust', 'wilt', 'rot',
                'discoloration', 'scaling', 'curling', 'stunting', 'galls'
            ],
            'symptom_descriptors': [
                'powdery', 'necrotic', 'chlorotic', 'yellowing', 'browning', 'spreading',
                'oozing', 'stunted', 'water-soaked', 'velvety', 'sunken'
            ],
            'temporal': [
                'sudden', 'gradual', 'recent', 'chronic', 'acute',
                'days', 'weeks', 'months', 'overnight'
            ],
            'severity': [
                'mild', 'moderate', 'severe', 'widespread', 'slight',
                'worsening', 'improving', 'stable', 'spreading rapidly'
            ]
        }

    # Converts the text into a numerical vector (“embedding”) that captures its meaning
    # 1- Cleans and standardizes the text
    # 2- Uses the sentence transformer to create the embedding
    def extract_features(self, text: str) -> np.ndarray:
        try:
            if self.model is None:
                return np.random.randn(384)

            processed_text = self._preprocess_text(text)
            embeddings = self.model.encode(processed_text, convert_to_numpy=True)

            return embeddings

        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return np.random.randn(384)

    # Cleans up the text for analysis
    # 1- Lowercases, removes punctuation, and standardizes terms (e.g., “spots” -> “lesions”)
    def _preprocess_text(self, text: str) -> str:
        try:
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text).strip()
            # Changed to use botanical standardization
            text = self._standardize_botanical_terms(text)
            return text

        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text

    # Replaces casual terms with formal botanical language (e.g., “spots” -> “lesions”)
    def _standardize_botanical_terms(self, text: str) -> str:
        replacements = {
            'spots': 'lesions',
            'spotty': 'lesions',
            'powder': 'powdery mildew',
            'yellow leaves': 'chlorotic leaves',
            'browning': 'necrosis',
            'wilting': 'wilted'
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    # Looks for botanical keywords in the text and determines how severe the symptoms sound
    # 1- Scans for each category of keyword
    # 2- Assigns a severity level (“low”, “medium”, or “high”) based on the number of urgent words found
    def analyze_symptoms(self, text: str) -> Dict[str, Any]:
        try:
            text_lower = text.lower()
            analysis = {}

            for category, keywords in self.botanical_keywords.items():
                found_keywords = [kw for kw in keywords if kw in text_lower]
                analysis[category] = found_keywords

            # Updated urgent keywords for plant context
            urgent_keywords = ['sudden', 'severe', 'widespread', 'spreading rapidly', 'rot', 'oozing']
            severity_score = sum(1 for kw in urgent_keywords if kw in text_lower)

            if severity_score >= 2:
                severity = 'high'
            elif severity_score == 1:
                severity = 'medium'
            else:
                severity = 'low'

            analysis['severity'] = severity
            analysis['severity_score'] = severity_score

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing symptoms: {e}")
            return {'severity': 'unknown'}
