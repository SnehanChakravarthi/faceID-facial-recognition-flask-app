import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

from deepface import DeepFace

from ..config import Settings


class AntiSpoofingCodes:
    SUCCESS = 0
    FUNCTION_ERROR = 1
    NO_FACE_DETECTED = 2
    MULTIPLE_FACES = 3
    SPOOFING_DETECTED = 4


class AntiSpoofingMessages:
    SUCCESS = "Anti-spoofing check completed successfully"
    FUNCTION_ERROR = "Unexpected error in anti-spoofing check"
    NO_FACE_DETECTED = "No face detected"
    MULTIPLE_FACES = "Multiple faces detected - only one face allowed"
    SPOOFING_DETECTED = "Spoofing detected - only live people allowed"


class AntiSpoofingError(Exception):
    """Base exception for anti-spoofing errors"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class AntiSpoofingValidationError(AntiSpoofingError):
    """Raised when validation fails"""

    pass


class AntiSpoofingDetectionError(AntiSpoofingError):
    """Raised when detection fails"""

    pass


@dataclass
class AntiSpoofingResponse:
    code: int
    is_real: Optional[bool] = None
    antispoof_score: Optional[float] = None
    confidence: Optional[float] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[int, bool, float, str, None]]:
        return {
            "code": self.code,
            "is_real": self.is_real,
            "antispoof_score": self.antispoof_score,
            "confidence": self.confidence,
            "message": self.message,
        }


def check_anti_spoofing(
    image_path: str,
) -> Dict[str, Union[int, bool, float, str, None]]:
    """
    Check if an image is real or fake using DeepFace's anti-spoofing detection.

    Args:
        image_path (str): Path to the image file to check

    Returns:
        dict: Response containing:
            - code (int): Response code
            - is_real (bool|None): Whether the face is determined to be real
            - antispoof_score (float|None): Anti-spoofing confidence score
            - confidence (float|None): Overall detection confidence
            - message (str|None): Error message if any

    Raises:
        ValueError: If image_path is invalid or file doesn't exist
    """
    try:
        # Validate input
        if not image_path or not isinstance(image_path, str):
            raise AntiSpoofingValidationError(
                AntiSpoofingCodes.FUNCTION_ERROR,
                "Invalid image path provided",
            )

        if not os.path.exists(image_path):
            raise AntiSpoofingValidationError(
                AntiSpoofingCodes.FUNCTION_ERROR,
                f"Image file not found: {image_path}",
            )

        # Configure detection parameters
        detection_config = {
            "img_path": image_path,
            "detector_backend": Settings.ANTI_SPOOFING_DETECTOR_BACKEND,
            "enforce_detection": True,
            "align": True,
            "anti_spoofing": True,
        }

        # Attempt face detection with anti-spoofing
        faces = DeepFace.extract_faces(**detection_config)

        if not faces:
            raise AntiSpoofingDetectionError(
                AntiSpoofingCodes.NO_FACE_DETECTED,
                AntiSpoofingMessages.NO_FACE_DETECTED,
            )

        if len(faces) > 1:
            raise AntiSpoofingDetectionError(
                AntiSpoofingCodes.MULTIPLE_FACES,
                AntiSpoofingMessages.MULTIPLE_FACES,
            )

        face_result = faces[0]

        # Validate required fields
        required_fields = ["is_real", "antispoof_score", "confidence"]
        missing_fields = [
            field for field in required_fields if field not in face_result
        ]
        if missing_fields:
            raise AntiSpoofingValidationError(
                AntiSpoofingCodes.FUNCTION_ERROR,
                f"Missing required fields in detection result: {missing_fields}",
            )

        # Extract and validate scores
        antispoof_score = float(face_result.get("antispoof_score", 0.0))
        confidence = float(face_result.get("confidence", 0.0))
        is_real = bool(face_result.get("is_real", False))

        if not (0 <= antispoof_score <= 1) or not (0 <= confidence <= 1):
            raise AntiSpoofingValidationError(
                AntiSpoofingCodes.FUNCTION_ERROR,
                "Invalid score values returned from detection",
            )

        # Create response object with common parameters
        response = AntiSpoofingResponse(
            code=(
                AntiSpoofingCodes.SPOOFING_DETECTED
                if not is_real
                else AntiSpoofingCodes.SUCCESS
            ),
            is_real=is_real,
            antispoof_score=round(antispoof_score, 3),
            confidence=confidence,
            message=(
                AntiSpoofingMessages.SPOOFING_DETECTED
                if not is_real
                else AntiSpoofingMessages.SUCCESS
            ),
        )

        if is_real:
            logging.info(
                "Anti-spoofing check completed successfully: is_real=%s, score=%f, confidence=%f",
                is_real,
                antispoof_score,
                confidence,
            )

        return response.to_dict()

    except AntiSpoofingError as e:
        logging.warning(
            "Anti-spoofing check failed: code=%d, message=%s", e.code, e.message
        )
        return AntiSpoofingResponse(code=e.code, message=e.message).to_dict()

    except Exception as e:
        error_msg = str(e)
        if "Face could not be detected in" in error_msg:
            logging.warning("No face detected in image")
            return AntiSpoofingResponse(
                code=AntiSpoofingCodes.NO_FACE_DETECTED,
                message=AntiSpoofingMessages.NO_FACE_DETECTED,
            ).to_dict()

        error_msg = f"Unexpected error in anti-spoofing check: {error_msg}"
        logging.error(error_msg, exc_info=True)
        return AntiSpoofingResponse(
            code=AntiSpoofingCodes.FUNCTION_ERROR, message=error_msg
        ).to_dict()
