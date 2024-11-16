import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from deepface import DeepFace

from ..config import Settings


class EmbeddingCodesV2:
    SUCCESS = 0
    FUNCTION_ERROR = 1
    NO_FACE_DETECTED = 2
    SPOOF_DETECTED = 3


class EmbeddingMessagesV2:
    SUCCESS = "Embedding generation completed successfully"
    FUNCTION_ERROR = "Unexpected error in embedding generation"
    NO_FACE_DETECTED = "No face detected"
    SPOOF_DETECTED = "Spoof detected in the given image"


class EmbeddingErrorV2(Exception):
    """Base exception for embedding generation errors"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class EmbeddingValidationErrorV2(EmbeddingErrorV2):
    """Raised when validation fails"""

    pass


# embedding dimensions
expected_dims = {
    "VGG-Face": 2622,
    "OpenFace": 128,
    "Facenet": 128,
    "Facenet512": 512,
    "DeepFace": 4096,
    "DeepID": 512,
    "Dlib": 128,
    "ArcFace": 512,
    "SFace": 512,
    "GhostFaceNet": 128,
}


@dataclass
class EmbeddingResponseV2:
    code: int
    embeddings: Optional[List[List[float]]] = None
    face_detected: bool = False
    is_real: bool = False
    model_info: Dict[str, Any] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "embeddings": self.embeddings,
            "face_detected": self.face_detected,
            "is_real": self.is_real,
            "model_info": self.model_info,
            "message": self.message,
        }


def generate_embeddings_v2(image_path: str) -> Dict[str, Any]:
    """
    Generate face embeddings for a given image path using DeepFace.

    Args:
        image_path (str): Path to the image file to process

    Returns:
        dict: Response containing:
            - success (bool): Whether the embedding generation was successful
            - embeddings (list|None): List of face embeddings if successful
            - error (str|None): Error message if any
            - face_detected (bool): Whether a face was detected
            - model_info (dict): Information about the model used

    Raises:
        ValueError: If image_path is invalid or file doesn't exist
    """
    try:
        # Validate input
        if not image_path or not isinstance(image_path, str):
            raise EmbeddingValidationErrorV2(
                EmbeddingCodesV2.FUNCTION_ERROR, "Invalid image path provided"
            )

        if not os.path.exists(image_path):
            raise EmbeddingValidationErrorV2(
                EmbeddingCodesV2.FUNCTION_ERROR, f"Image file not found: {image_path}"
            )

        # Configure embedding generation parameters
        embedding_config = {
            "img_path": image_path,
            "model_name": Settings.FACE_RECOGNITION_MODEL,
            "detector_backend": Settings.DETECTOR_BACKEND,
            "enforce_detection": True,  # Changed to True for better reliability
            "align": True,
            "normalization": "ArcFace",
            "anti_spoofing": True,
        }

        # Attempt to generate embeddings
        embeddings = DeepFace.represent(**embedding_config)

        logging.debug("Embeddings of v2: %s", embeddings)

        if not embeddings:
            raise EmbeddingValidationErrorV2(
                EmbeddingCodesV2.NO_FACE_DETECTED, EmbeddingMessagesV2.NO_FACE_DETECTED
            )

        # Handle different embedding return formats
        if isinstance(embeddings, dict):
            embeddings = [embeddings]
        elif not isinstance(embeddings, list):
            embeddings = [embeddings]

        # Process and validate embeddings
        processed_embeddings = []
        for embedding in embeddings:
            if isinstance(embedding, dict) and "embedding" in embedding:
                embedding_vector = embedding["embedding"]
            else:
                embedding_vector = embedding

            if (
                not isinstance(embedding_vector, (list, np.ndarray))
                or len(embedding_vector) == 0
            ):
                raise EmbeddingValidationErrorV2(
                    EmbeddingCodesV2.FUNCTION_ERROR,
                    "Invalid embedding format received from model",
                )

            if isinstance(embedding_vector, np.ndarray):
                embedding_vector = embedding_vector.tolist()

            processed_embeddings.append(embedding_vector)

        # Validate embedding dimensions
        expected_dim = expected_dims.get(Settings.FACE_RECOGNITION_MODEL)
        if expected_dim and len(processed_embeddings[0]) != expected_dim:
            raise EmbeddingValidationErrorV2(
                EmbeddingCodesV2.FUNCTION_ERROR,
                f"Unexpected embedding dimension: got {len(processed_embeddings[0])}, "
                f"expected {expected_dim} for model {Settings.FACE_RECOGNITION_MODEL}",
            )

        response = EmbeddingResponseV2(
            code=EmbeddingCodesV2.SUCCESS,
            embeddings=processed_embeddings,
            face_detected=True,
            is_real=True,
            model_info={
                "face_recognition_model": Settings.FACE_RECOGNITION_MODEL,
                "face_detection_model": Settings.DETECTOR_BACKEND,
                "embedding_dimension": len(processed_embeddings[0]),
                "embeddings_count": len(processed_embeddings),
            },
            message=EmbeddingMessagesV2.SUCCESS,
        )

        logging.info(
            "Embedding generation successful: generated %d embeddings with dimension %d",
            len(processed_embeddings),
            len(processed_embeddings[0]),
        )

        return response.to_dict()

    except EmbeddingErrorV2 as e:
        logging.warning(
            "Embedding generation failed: code=%d, message=%s", e.code, e.message
        )
        return EmbeddingResponseV2(code=e.code, message=e.message).to_dict()

    except Exception as e:
        error_msg = str(e)
        logging.error(
            f"Unexpected error in embedding generation: {error_msg}", exc_info=True
        )

        # Check if the error is related to face detection
        if "Face could not be detected" in error_msg:
            return EmbeddingResponseV2(
                code=EmbeddingCodesV2.NO_FACE_DETECTED,
                embeddings=None,
                face_detected=False,
                is_real=False,
                model_info={
                    "face_recognition_model": Settings.FACE_RECOGNITION_MODEL,
                    "face_detection_model": Settings.DETECTOR_BACKEND,
                    "embedding_dimension": None,
                    "embeddings_count": 0,
                },
                message=EmbeddingMessagesV2.NO_FACE_DETECTED,
            ).to_dict()

        # Handle spoofing detection
        elif "Spoof detected in the given image" in error_msg:
            return EmbeddingResponseV2(
                code=EmbeddingCodesV2.SPOOF_DETECTED,
                embeddings=None,
                face_detected=True,
                is_real=False,
                model_info={
                    "face_recognition_model": Settings.FACE_RECOGNITION_MODEL,
                    "face_detection_model": Settings.DETECTOR_BACKEND,
                    "embedding_dimension": None,
                    "embeddings_count": 0,
                },
                message=EmbeddingMessagesV2.SPOOF_DETECTED,
            ).to_dict()

        # Handle other unexpected errors
        return EmbeddingResponseV2(
            code=EmbeddingCodesV2.FUNCTION_ERROR,
            message=f"Unexpected error in embedding generation: {error_msg}",
        ).to_dict()
