import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class AttentionFusionLayer:
    def __init__(self, feature_dim: int = 384, device='cuda'):
        self.feature_dim = feature_dim
        self.device = device
        self.attention_network = self._build_attention_network()
        self.fusion_network = self._build_fusion_network()

    # Creates a mini neural network (attention) that decides how much to trust each input (image, text, audio).
    # Learns weights for each modality by looking at all their features together.
    # Inside:
    #   - Uses two linear layers with ReLU and softmax for final attention weights.
    #   - Returns three numbers: the weights for image, text, audio (which sum to 1).

    def _build_attention_network(self) -> nn.Module:
        class MultimodalAttention(nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.Linear(feature_dim * 3, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, 3),
                    nn.Softmax(dim=1)
                )

            def forward(self, image_features, text_features, audio_features=None):
                if audio_features is None:
                    audio_features = torch.zeros_like(text_features)

                combined = torch.cat([image_features, text_features, audio_features], dim=1)
                weights = self.attention(combined)
                return weights

        model = MultimodalAttention(self.feature_dim)
        if torch.cuda.is_available() and self.device == 'cuda':
            model = model.to(self.device)

        return model

    # Builds another small neural network to further process the weighted, combined features.
    # This refines the information for better predictions about plant health.
    # Inside:
    #   - Two linear layers with ReLU and Dropout for regularization.

    def _build_fusion_network(self) -> nn.Module:
        class FusionNetwork(nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.fusion = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(feature_dim, feature_dim)
                )

            def forward(self, fused_features):
                return self.fusion(fused_features)

        model = FusionNetwork(self.feature_dim)
        if torch.cuda.is_available() and self.device == 'cuda':
            model = model.to(self.device)

        return model

    # The main method: Takes features from each input, runs the attention network to get weights,
    # multiplies each feature by its weight, sums them, and passes the result through the fusion network.
    # Returns the final, normalized fused vector for plant disease analysis.
    # Handles:
    #   - If any input is missing, it fills with zeros.

    def fuse_modalities(self, image_features: np.ndarray, text_features: np.ndarray,
                          audio_features: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            img_tensor = self._prepare_features(image_features)
            text_tensor = self._prepare_features(text_features)

            if audio_features is not None:
                audio_tensor = self._prepare_features(audio_features)
            else:
                audio_tensor = torch.zeros_like(text_tensor)

            with torch.no_grad():
                attention_weights = self.attention_network(img_tensor, text_tensor, audio_tensor)

                weighted_features = (
                    attention_weights[:, 0:1] * img_tensor +
                    attention_weights[:, 1:2] * text_tensor +
                    attention_weights[:, 2:3] * audio_tensor
                )

                fused = self.fusion_network(weighted_features)
                result = fused.cpu().numpy()[0]
                result = self._normalize_features(result)

                return result

        except Exception as e:
            logger.error(f"Error in attention fusion: {e}")
            return self._fallback_fusion(image_features, text_features, audio_features)

    # Makes sure each input feature is the correct shape and size.
    # Pads or trims features as needed and converts them to PyTorch tensors.

    def _prepare_features(self, features: np.ndarray) -> torch.Tensor:
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        if features.shape[1] < self.feature_dim:
            pad_width = ((0, 0), (0, self.feature_dim - features.shape[1]))
            features = np.pad(features, pad_width, mode='constant')
        elif features.shape[1] > self.feature_dim:
            features = features[:, :self.feature_dim]

        tensor = torch.FloatTensor(features)

        if torch.cuda.is_available() and self.device == 'cuda':
            tensor = tensor.to(self.device)

        return tensor

    # If the attention network fails, this does a simple weighted average of the available features
    # (using fixed weights, e.g., 0.4 for image and text).
    # This makes the system more robust in case of errors.

    def _fallback_fusion(self, image_features: np.ndarray, text_features: np.ndarray,
                           audio_features: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            features_list = [image_features, text_features]
            weights = [0.4, 0.4]

            if audio_features is not None:
                features_list.append(audio_features)
                weights = [0.3, 0.4, 0.3]

            target_size = self.feature_dim
            normalized_features = []

            for feat in features_list:
                if len(feat) < target_size:
                    feat = np.pad(feat, (0, target_size - len(feat)))
                else:
                    feat = feat[:target_size]
                normalized_features.append(feat)

            fused = np.zeros(target_size)
            for feat, weight in zip(normalized_features, weights):
                fused += weight * feat

            return self._normalize_features(fused)

        except Exception as e:
            logger.error(f"Error in fallback fusion: {e}")
            return np.random.randn(self.feature_dim)

    # Ensures the final output vector has a length (norm) of 1.
    # This is a standard step for feature vectors, making comparison and further processing easier.

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        try:
            norm = np.linalg.norm(features)
            if norm > 0:
                return features / norm
            return features
        except Exception:
            return features
