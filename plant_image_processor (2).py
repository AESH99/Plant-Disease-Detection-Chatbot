import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

class PlantImageProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.processor = None
        self._load_models()

    def _load_models(self):
        try:
            logger.info("Loading BLIP model for image analysis...")
            model_name = "Salesforce/blip-image-captioning-base"

            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)

            if torch.cuda.is_available() and self.device == 'cuda':
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info(f"BLIP model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}")
            self.model = None
            self.processor = None

    # Turns the image into a “feature vector” (a list of numbers) that summarizes its content for the AI.
    def process_image(self, image: Image.Image) -> np.ndarray:
        try:
            if self.model is None:
                return np.random.randn(768)

            processed_image = self._preprocess_image(image) # Preprocess the image (resize, convert to correct color)
            inputs = self.processor(processed_image, return_tensors="pt")

            if torch.cuda.is_available() and self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.vision_model(**{k: v for k, v in inputs.items() if k in ['pixel_values']})
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0] # Use BLIP’s vision model to extract features

            return features

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return np.random.randn(768) # If the model fails, returns random features (so the pipeline doesn’t break)

    # Describes the image in botanical terms
    def generate_caption(self, image: Image.Image) -> str:
        try:
            if self.model is None:
                return "Image analysis unavailable - model not loaded"

            # Changed prompt from medical to botanical
            prompt = "A detailed botanical description of this plant leaf showing symptoms of"
            processed_image = self._preprocess_image(image)

            inputs = self.processor(processed_image, prompt, return_tensors="pt")

            if torch.cuda.is_available() and self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
                caption = self.processor.decode(outputs[0], skip_special_tokens=True) # Uses BLIP to generate a detailed caption

            # Changed to botanical enhancement and fallback
            enhanced_caption = self._enhance_botanical_terminology(caption)
            return enhanced_caption

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return f"Basic image analysis: {self._basic_botanical_analysis(image)}" # If the model fails, uses a basic color/shape analysis instead

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image

    # Replaced medical terms with botanical/plant pathology terms
    def _enhance_botanical_terminology(self, caption: str) -> str:
        botanical_terms = {
            'spot': 'lesion',
            'spots': 'lesions',
            'mark': 'blotch',
            'marks': 'blotches',
            'bump': 'pustule',
            'bumps': 'pustules',
            'powder': 'powdery mildew',
            'white': 'powdery or chlorotic',
            'yellow': 'chlorotic',
            'yellowing': 'chlorosis',
            'brown': 'necrotic',
            'dark': 'necrotic',
            'dead': 'necrotic',
            'hole': 'shot hole'
        }

        enhanced = caption.lower()
        for common, botanical in botanical_terms.items():
            enhanced = enhanced.replace(common, botanical)

        return enhanced.capitalize()

    # This function provides a basic, quick description of the image by looking at its colors, without using any AI model
    # Logic updated for plant symptoms (yellow, brown)
    def _basic_botanical_analysis(self, image: Image.Image) -> str:
        try:
            img_array = np.array(image)
            mean_color = img_array.mean(axis=(0, 1))

            # Check for yellowing (high red and green, low blue)
            if mean_color[0] > 140 and mean_color[1] > 140 and mean_color[2] < 100:
                color_desc = "yellowish chlorotic leaf"
            # Check for browning/dark spots (low overall brightness)
            elif mean_color.mean() < 100:
                color_desc = "dark or necrotic lesions"
            # Check for white/powdery (high overall brightness)
            elif mean_color.mean() > 200:
                color_desc = "possible powdery mildew"
            else:
                color_desc = "plant leaf symptom"

            return f"Visible {color_desc} requiring botanical analysis"

        except Exception:
            return "Plant leaf visible in image"

    # Checks if the image is suitable for analysis (e.g., not blurry)
    # This function is generic and required no changes
    def analyze_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        try:
            img_array = np.array(image)
            sharpness = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
            brightness = img_array.mean()

            quality_score = 100
            issues = []

            if sharpness < 100:
                issues.append("Image may be blurry")
                quality_score -= 20

            if brightness < 50:
                issues.append("Image is too dark")
                quality_score -= 15
            elif brightness > 200:
                issues.append("Image is too bright")
                quality_score -= 15

            if min(image.size) < 200:
                issues.append("Image resolution is low")
                quality_score -= 25

            quality = "good" if quality_score >= 70 else "fair" if quality_score >= 50 else "poor"

            return {
                "quality": quality,
                "score": max(0, quality_score),
                "sharpness": sharpness,
                "brightness": brightness,
                "resolution": image.size,
                "issues": issues
            }

        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            return {"quality": "unknown", "issues": ["Quality analysis failed"]}
