import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from flask import jsonify, Request

from ..config import Settings
from ..utils import process_face_image_v2


class AuthenticateCodesV2:
    SUCCESS = 0
    FUNCTION_ERROR = 1
    NO_FACE_DETECTED = 2
    MULTIPLE_FACES_DETECTED = 3
    SPOOFING_DETECTED = 4
    NO_MATCH = 5
    BELOW_THRESHOLD = 6
    UNEXPECTED_ERROR = 7


class AuthenticateMessagesV2:
    SUCCESS = "Authentication completed successfully"
    FUNCTION_ERROR = "Unexpected error in authentication"
    NO_FACE_DETECTED = "No face detected"
    MULTIPLE_FACES_DETECTED = "Multiple faces detected"
    SPOOFING_DETECTED = "Spoofing detected"
    NO_MATCH = "No matching face found"
    BELOW_THRESHOLD = "Face match below threshold"
    UNEXPECTED_ERROR = "Validation error"


class AuthenticateErrorV2(Exception):
    """Base exception for authentication errors"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class AuthenticateValidationErrorV2(AuthenticateErrorV2):
    """Raised when validation fails"""

    pass


@dataclass
class AuthenticateResponseV2:
    code: int
    message: Optional[str] = None
    match: Optional[dict] = None
    similarity_score: Optional[float] = None
    anti_spoofing: Optional[dict] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "match": self.match,
            "similarity_score": self.similarity_score,
            "anti_spoofing": self.anti_spoofing,
            "details": self.details,
        }


def handle_authenticate_request_v2(
    request: Request, pinecone_service
) -> Tuple[Any, int]:
    """
    Endpoint to authenticate a face against enrolled faces.

    Expected form data:
        - image (required): Image file containing the face to authenticate

    Returns:
        tuple: (JSON response, HTTP status code)
            Response contains:
                - code (int): Authentication code
                - message (str): Success/error message
                - match (dict, optional): Matching face details if found
                - similarity_score (float, optional): Similarity score if found
                - anti_spoofing (dict): Anti-spoofing details
                - details (dict): Processing details and timing information
    """
    start_time = time.time()
    try:
        # Request validation
        image_file = request.files.get("image")

        logging.info("Received authentication request")

        if not image_file or not image_file.filename:
            raise ValueError("Missing required image file")

        # Process face image
        face_processing_start = time.time()
        face_processing_task = process_face_image_v2(image_file)

        # Map process_face_image_v2 codes to authenticate codes
        code_mapping = {
            0: AuthenticateCodesV2.SUCCESS,  # Success
            1: AuthenticateCodesV2.FUNCTION_ERROR,  # Function Error
            2: AuthenticateCodesV2.NO_FACE_DETECTED,  # No Face Detected
            3: AuthenticateCodesV2.SPOOFING_DETECTED,  # Spoof Detected
        }

        # Check for face processing errors
        if face_processing_task["code"] != 0:  # Any non-success code
            mapped_code = code_mapping.get(
                face_processing_task["code"], AuthenticateCodesV2.UNEXPECTED_ERROR
            )

            response = AuthenticateResponseV2(
                code=mapped_code,
                message=face_processing_task["message"],
                match=None,
                similarity_score=None,
                anti_spoofing=face_processing_task.get("anti_spoofing"),
                details={
                    "processing_times": {
                        "total": round(time.time() - start_time, 3),
                        "face_processing": round(
                            time.time() - face_processing_start, 3
                        ),
                        "vector_search": None,
                    },
                    "face_detected": face_processing_task.get("face_detected", False),
                },
            )

            logging.warning(
                "Face processing failed during authentication: %s (code=%d)",
                face_processing_task["message"],
                face_processing_task["code"],
            )
            return jsonify(response.to_dict()), 200

        # If we reach here, face processing was successful
        face_processing_time = time.time() - face_processing_start

        # Search for similar faces
        search_result = pinecone_service.search_similar_faces(
            query_vector=face_processing_task["embeddings"][0],
            threshold=Settings.MATCH_THRESHOLD,
        )

        # Common processing times for all scenarios
        processing_times = {
            "total": round(time.time() - start_time, 3),
            "face_processing": round(face_processing_time, 3),
            "vector_search": round(search_result["details"]["processing_time"], 3),
        }

        # Handle different search result codes
        if search_result["code"] == AuthenticateCodesV2.SUCCESS:
            response = AuthenticateResponseV2(
                code=AuthenticateCodesV2.SUCCESS,
                message=AuthenticateMessagesV2.SUCCESS,
                match=search_result["match"],
                similarity_score=search_result["details"]["similarity_score"],
                anti_spoofing=face_processing_task.get("anti_spoofing"),
                details={"processing_times": processing_times},
            )

        elif search_result["code"] == AuthenticateCodesV2.BELOW_THRESHOLD:
            response = AuthenticateResponseV2(
                code=AuthenticateCodesV2.BELOW_THRESHOLD,
                message=AuthenticateMessagesV2.BELOW_THRESHOLD,
                match=None,
                similarity_score=search_result["details"]["similarity_score"],
                anti_spoofing=face_processing_task.get("anti_spoofing"),
                details={"processing_times": processing_times},
            )

        elif search_result["code"] == AuthenticateCodesV2.NO_MATCH:
            response = AuthenticateResponseV2(
                code=AuthenticateCodesV2.NO_MATCH,
                message=AuthenticateMessagesV2.NO_MATCH,
                match=None,
                similarity_score=None,
                anti_spoofing=face_processing_task.get("anti_spoofing"),
                details={"processing_times": processing_times},
            )

        elif search_result["code"] == 7 or search_result["code"] == 8:
            raise ValueError(f"Search validation error: {search_result['message']}")

        else:  # Includes SearchCodes.UNEXPECTED_ERROR and any unknown codes
            raise RuntimeError(f"Search failed: {search_result['message']}")

        logging.info(
            "Authentication completed - code=%d, message='%s', processing_time=%.3fs",
            response.code,
            response.message,
            processing_times["total"],
        )
        return jsonify(response.to_dict()), 200

    except ValueError as ve:
        error_msg = str(ve)
        logging.warning("Validation error during authentication: %s", error_msg)
        response = AuthenticateResponseV2(
            code=AuthenticateCodesV2.FUNCTION_ERROR,
            message=error_msg,
            details={
                "error_type": "validation",
                "processing_times": {"total": round(time.time() - start_time, 3)},
            },
        )
        return jsonify(response.to_dict()), 400

    except RuntimeError as re:
        error_msg = str(re)
        logging.error("Runtime error during authentication: %s", error_msg)
        response = AuthenticateResponseV2(
            code=AuthenticateCodesV2.UNEXPECTED_ERROR,
            message=error_msg,
            details={
                "error_type": "runtime",
                "processing_times": {"total": round(time.time() - start_time, 3)},
            },
        )
        return jsonify(response.to_dict()), 500

    except Exception as e:
        error_msg = f"Unexpected error during authentication: {str(e)}"
        logging.exception(error_msg)
        response = AuthenticateResponseV2(
            code=AuthenticateCodesV2.UNEXPECTED_ERROR,
            message=error_msg,
            details={
                "error_type": "unexpected",
                "processing_times": {"total": round(time.time() - start_time, 3)},
            },
        )
        return jsonify(response.to_dict()), 500
