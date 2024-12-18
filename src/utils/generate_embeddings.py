import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from deepface import DeepFace

from ..config import Settings


class EmbeddingCodes:
    SUCCESS = 0
    FUNCTION_ERROR = 1
    NO_FACE_DETECTED = 2


class EmbeddingMessages:
    SUCCESS = "Embedding generation completed successfully"
    FUNCTION_ERROR = "Unexpected error in embedding generation"
    NO_FACE_DETECTED = "No face detected"


class EmbeddingError(Exception):
    """Base exception for embedding generation errors"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class EmbeddingValidationError(EmbeddingError):
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
class EmbeddingResponse:
    code: int
    embeddings: Optional[List[List[float]]] = None
    face_detected: bool = False
    model_info: Dict[str, Any] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "embeddings": self.embeddings,
            "face_detected": self.face_detected,
            "model_info": self.model_info,
            "message": self.message,
        }


def generate_embeddings(image_path: str) -> Dict[str, Any]:
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
            raise EmbeddingValidationError(
                EmbeddingCodes.FUNCTION_ERROR, "Invalid image path provided"
            )

        if not os.path.exists(image_path):
            raise EmbeddingValidationError(
                EmbeddingCodes.FUNCTION_ERROR, f"Image file not found: {image_path}"
            )

        # Configure embedding generation parameters
        embedding_config = {
            "img_path": image_path,
            "model_name": Settings.FACE_RECOGNITION_MODEL,
            "detector_backend": Settings.DETECTOR_BACKEND,
            "enforce_detection": True,  # Changed to True for better reliability
            "align": True,
            "normalization": "base",
        }

        # Attempt to generate embeddings
        embeddings = DeepFace.represent(**embedding_config)

        if not embeddings:
            raise EmbeddingValidationError(
                EmbeddingCodes.NO_FACE_DETECTED, EmbeddingMessages.NO_FACE_DETECTED
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
                raise EmbeddingValidationError(
                    EmbeddingCodes.FUNCTION_ERROR,
                    "Invalid embedding format received from model",
                )

            if isinstance(embedding_vector, np.ndarray):
                embedding_vector = embedding_vector.tolist()

            processed_embeddings.append(embedding_vector)

        # Validate embedding dimensions
        expected_dim = expected_dims.get(Settings.FACE_RECOGNITION_MODEL)
        if expected_dim and len(processed_embeddings[0]) != expected_dim:
            raise EmbeddingValidationError(
                EmbeddingCodes.FUNCTION_ERROR,
                f"Unexpected embedding dimension: got {len(processed_embeddings[0])}, "
                f"expected {expected_dim} for model {Settings.FACE_RECOGNITION_MODEL}",
            )

        response = EmbeddingResponse(
            code=EmbeddingCodes.SUCCESS,
            embeddings=processed_embeddings,
            face_detected=True,
            model_info={
                "face_recognition_model": Settings.FACE_RECOGNITION_MODEL,
                "face_detection_model": Settings.DETECTOR_BACKEND,
                "embedding_dimension": len(processed_embeddings[0]),
                "embeddings_count": len(processed_embeddings),
            },
            message=EmbeddingMessages.SUCCESS,
        )

        logging.info(
            "Embedding generation successful: generated %d embeddings with dimension %d",
            len(processed_embeddings),
            len(processed_embeddings[0]),
        )

        return response.to_dict()

    except EmbeddingError as e:
        logging.warning(
            "Embedding generation failed: code=%d, message=%s", e.code, e.message
        )
        return EmbeddingResponse(code=e.code, message=e.message).to_dict()

    except Exception as e:
        error_msg = f"Unexpected error in embedding generation: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return EmbeddingResponse(
            code=EmbeddingCodes.FUNCTION_ERROR, message=error_msg
        ).to_dict()
