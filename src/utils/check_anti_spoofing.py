import logging
import os
from deepface import DeepFace
from ..config import Settings


def check_anti_spoofing(image_path):
    """
    Check if an image is real or fake using DeepFace's anti-spoofing detection.

    Args:
        image_path (str): Path to the image file to check

    Returns:
        dict: Response containing:
            - success (bool): Whether the check completed successfully
            - is_real (bool|None): Whether the face is determined to be real
            - antispoof_score (float|None): Anti-spoofing confidence score
            - confidence (float|None): Overall detection confidence
            - error (str|None): Error message if any

    Raises:
        ValueError: If image_path is invalid or file doesn't exist
    """
    logging.debug("Starting anti-spoofing check for image: %s", image_path)

    response = {
        "success": False,
        "is_real": None,
        "antispoof_score": None,
        "confidence": None,
        "error": None,
        "face_detected": False,
    }

    # Validate input
    if not image_path or not isinstance(image_path, str):
        raise ValueError("Invalid image path provided")

    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")

    try:
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
            response.update(
                {"error": "No face detected in the image", "face_detected": False}
            )
            logging.warning("Anti-spoofing check failed: no face detected")
            return response

        if len(faces) > 1:
            response.update(
                {
                    "error": "Multiple faces detected - only one face allowed",
                    "face_detected": True,
                }
            )
            logging.warning("Anti-spoofing check failed: multiple faces detected")
            return response

        face_result = faces[0]

        # Validate required fields exist in the result
        required_fields = ["is_real", "antispoof_score", "confidence"]
        missing_fields = [
            field for field in required_fields if field not in face_result
        ]

        if missing_fields:
            raise ValueError(
                f"Missing required fields in detection result: {missing_fields}"
            )

        # Extract and validate scores
        antispoof_score = float(face_result.get("antispoof_score", 0.0))
        confidence = float(face_result.get("confidence", 0.0))
        is_real = bool(face_result.get("is_real", False))

        if not (0 <= antispoof_score <= 1) or not (0 <= confidence <= 1):
            raise ValueError("Invalid score values returned from detection")

        response.update(
            {
                "success": True,
                "is_real": is_real,
                "antispoof_score": antispoof_score,
                "confidence": confidence,
                "face_detected": True,
            }
        )

        logging.info(
            "Anti-spoofing check completed successfully: is_real=%s, score=%f, confidence=%f",
            is_real,
            antispoof_score,
            confidence,
        )
        return response

    except ValueError as ve:
        error_msg = f"Validation error in anti-spoofing check: {str(ve)}"
        logging.error(error_msg)
        response["error"] = error_msg
        return response

    except Exception as e:
        error_msg = f"Unexpected error in anti-spoofing check: {str(e)}"
        logging.exception(error_msg)  # This logs the full stack trace
        response["error"] = error_msg
        return response
