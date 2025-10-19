import numpy as np
from typing import Dict, List, Any, Optional
import logging
# Updated imports for the PlantGuard project
from plant_image_processor import PlantImageProcessor
from plant_audio_processor import PlantAudioProcessor
from plant_text_processor import PlantTextProcessor
from fusion_layer import AttentionFusionLayer
from plant_guidlines import PlantScienceGuidelines

logger = logging.getLogger(__name__)

# Renamed the class to reflect its new purpose
class MultimodalPlantProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        # Instantiate the correct plant-focused processor classes
        self.image_processor = PlantImageProcessor(device)
        self.audio_processor = PlantAudioProcessor(device)
        self.text_processor = PlantTextProcessor(device)
        self.fusion_layer = AttentionFusionLayer(device=device)
        self.guidelines = PlantScienceGuidelines()

        logger.info("Multimodal plant processor initialized")

    # Runs the full pipeline (image, audio, text analysis)
    # Fuses all available features
    # Generates assessment, severity, recommendations, and confidence
    # Returns a summary dictionary for the case
    def process_case(self, image=None, audio=None, text_data=None, context_info=None) -> Dict[str, Any]:
        try:
            results = {
                'image_analysis': None,
                'audio_analysis': None,
                'text_analysis': None,
                'fusion_results': None,
                'botanical_assessment': None, # Renamed from clinical_assessment
                'recommendations': [],
                'severity': 'unknown', # Renamed from urgency
                'confidence': 0.0
            }

            if image is not None:
                results['image_analysis'] = self._process_image(image)

            if audio is not None:
                results['audio_analysis'] = self._process_audio(audio)

            if text_data is not None:
                results['text_analysis'] = self._process_text(text_data, context_info)

            if any([results['image_analysis'], results['audio_analysis'], results['text_analysis']]):
                results['fusion_results'] = self._perform_fusion(results)
                results['botanical_assessment'] = self._generate_botanical_assessment(results)
                results['recommendations'] = self._generate_recommendations(results)
                results['severity'] = self._assess_severity(results)
                results['confidence'] = self._calculate_confidence(results)

            return results

        except Exception as e:
            logger.error(f"Error processing case: {e}")
            return self._generate_error_response(str(e))

    # Runs all image tasks: feature extraction, captioning, quality check
    def _process_image(self, image) -> Dict[str, Any]:
        try:
            features = self.image_processor.process_image(image)
            caption = self.image_processor.generate_caption(image)
            quality = self.image_processor.analyze_image_quality(image)

            return {
                'features': features,
                'caption': caption,
                'quality': quality,
                'available': True
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {'available': False, 'error': str(e)}

    # Runs all audio tasks: transcription, feature extraction, quality check
    def _process_audio(self, audio) -> Dict[str, Any]:
        try:
            transcript = self.audio_processor.transcribe_audio(audio)
            features = self.audio_processor.extract_features(audio)
            quality = self.audio_processor.analyze_audio_quality(audio)

            return {
                'transcript': transcript,
                'features': features,
                'quality': quality,
                'available': True
            }

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {'available': False, 'error': str(e)}

    # Combines user text with contextual info, extracts features, analyzes symptoms
    def _process_text(self, text_data, context_info=None) -> Dict[str, Any]:
        try:
            combined_text = text_data
            if context_info:
                # Updated context to be relevant to plants
                context_text = f"Plant Species: {context_info.get('species', 'unknown')}, Location: {context_info.get('location', 'unknown')}"
                combined_text = f"{context_text}. {text_data}"

            features = self.text_processor.extract_features(combined_text)
            symptom_analysis = self.text_processor.analyze_symptoms(combined_text)

            return {
                'features': features,
                'symptom_analysis': symptom_analysis,
                'combined_text': combined_text,
                'available': True
            }

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {'available': False, 'error': str(e)}

    # Collects all features (image, text, audio), runs the fusion layer, and returns the fused vector
    # Handles missing modalities by substituting with random noise
    def _perform_fusion(self, results) -> Dict[str, Any]:
        try:
            image_features = None
            text_features = None
            audio_features = None

            if results['image_analysis'] and results['image_analysis'].get('available'):
                image_features = results['image_analysis']['features']

            if results['text_analysis'] and results['text_analysis'].get('available'):
                text_features = results['text_analysis']['features']

            if results['audio_analysis'] and results['audio_analysis'].get('available'):
                audio_features = results['audio_analysis']['features']

            if sum([f is not None for f in [image_features, text_features, audio_features]]) >= 1:
                if image_features is None:
                    image_features = np.random.randn(384) * 0.1
                if text_features is None:
                    text_features = np.random.randn(384) * 0.1

                fused_features = self.fusion_layer.fuse_modalities(
                    image_features, text_features, audio_features
                )

                return {
                    'fused_features': fused_features,
                    'fusion_successful': True
                }

            return {'fusion_successful': False, 'reason': 'Insufficient modalities'}

        except Exception as e:
            logger.error(f"Error in fusion: {e}")
            return {'fusion_successful': False, 'error': str(e)}

    # Compiles a botanical summary:
    # Lists main observations from all sources
    # Provides potential diseases (what the problem might be)
    def _generate_botanical_assessment(self, results) -> Dict[str, Any]:
        assessment = {
            'key_observations': [],
            'potential_diseases': [],
            'symptom_summary': ''
        }

        if results['image_analysis'] and results['image_analysis'].get('available'):
            assessment['key_observations'].append(
                f"Visual: {results['image_analysis']['caption']}"
            )

        if results['audio_analysis'] and results['audio_analysis'].get('available'):
            assessment['key_observations'].append(
                f"User report: {results['audio_analysis']['transcript']}"
            )

        if results['text_analysis'] and results['text_analysis'].get('available'):
            symptoms = results['text_analysis']['symptom_analysis']
            if symptoms.get('plant_symptoms'):
                assessment['key_observations'].extend(
                    [f"Reported: {cond}" for cond in symptoms['plant_symptoms']]
                )

        # Updated with plausible plant diseases
        assessment['potential_diseases'] = [
            {'condition': 'Powdery Mildew', 'likelihood': 'moderate'},
            {'condition': 'Fungal Leaf Spot (Blight)', 'likelihood': 'moderate'},
            {'condition': 'Nutrient Deficiency', 'likelihood': 'low'}
        ]

        assessment['symptom_summary'] = '. '.join(assessment['key_observations']) if assessment['key_observations'] else "Limited information available"

        return assessment

    # Suggests what to do next based on severity (isolate plant, apply treatment, or monitor)
    def _generate_recommendations(self, results) -> List[str]:
        recommendations = []
        severity = self._assess_severity(results)

        if severity == 'high':
            recommendations.extend([
                "Isolate the affected plant immediately to prevent spread",
                "Consider removing and destroying severely affected parts",
                "Apply an appropriate fungicide or pesticide based on diagnosis"
            ])
        elif severity == 'medium':
            recommendations.extend([
                "Apply a targeted treatment (e.g., neem oil, fungicide)",
                "Monitor plant closely for changes over the next 2-3 days",
                "Improve air circulation and check watering practices"
            ])
        else:
            recommendations.extend([
                "Monitor the plant's condition",
                "Ensure proper watering, light, and nutrient levels",
                "Prune away any mildly affected leaves to improve health"
            ])

        return recommendations

    # Calculates the severity of the case using symptom and image findings
    # Keywords and concerning terms (e.g., “rot”) raise the severity level
    def _assess_severity(self, results) -> str:
        severity_score = 0

        if results['text_analysis'] and results['text_analysis'].get('available'):
            symptoms = results['text_analysis']['symptom_analysis']
            text_severity = symptoms.get('severity', 'low')

            if text_severity == 'high':
                severity_score += 3
            elif text_severity == 'medium':
                severity_score += 2
            else:
                severity_score += 1

        if results['image_analysis'] and results['image_analysis'].get('available'):
            caption = results['image_analysis']['caption'].lower()
            # Updated concerning terms for plant pathology
            concerning_terms = ['rot', 'widespread', 'oozing', 'spreading rapidly', 'severe necrosis']
            concern_count = sum(1 for term in concerning_terms if term in caption)
            severity_score += concern_count

        if severity_score >= 4:
            return 'high'
        elif severity_score >= 2:
            return 'medium'
        else:
            return 'low'

    # Computes how confident the system is, based on the quality of inputs
    # Uses average of available quality scores
    def _calculate_confidence(self, results) -> float:
        confidence_factors = []

        if results['image_analysis'] and results['image_analysis'].get('available'):
            quality = results['image_analysis']['quality']
            confidence_factors.append(quality.get('score', 50) / 100)

        if results['text_analysis'] and results['text_analysis'].get('available'):
            confidence_factors.append(0.7) # Text is generally reliable

        if results['audio_analysis'] and results['audio_analysis'].get('available'):
            quality = results['audio_analysis']['quality']
            confidence_factors.append(quality.get('score', 50) / 100)

        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.3

    # Returns a safe default response in case of errors, advising the user to seek professional help
    def _generate_error_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            'error': True,
            'message': error_msg,
            # Updated recommendation for plant context
            'recommendations': ["Please try again or consult with a local agricultural extension or plant pathologist"],
            'severity': 'medium',
            'confidence': 0.0
        }
