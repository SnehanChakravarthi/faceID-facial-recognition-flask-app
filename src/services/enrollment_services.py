import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from flask import Request, jsonify

from ..utils import process_face_image


class EnrollmentCodes:
    SUCCESS = 0
    FUNCTION_ERROR = 1
    NO_FACE_DETECTED = 2
    MULTIPLE_FACES_DETECTED = 3
    SPOOFING_DETECTED = 4
    STORAGE_ERROR = 5
    UNEXPECTED_ERROR = 6


class EnrollmentMessages:
    SUCCESS = "Enrollment completed successfully"
    FUNCTION_ERROR = "Unexpected error in enrollment"
    NO_FACE_DETECTED = "No face detected"
    MULTIPLE_FACES_DETECTED = "Multiple faces detected"
    SPOOFING_DETECTED = "Spoofing detected"
    STORAGE_ERROR = "Failed to store face data"
    UNEXPECTED_ERROR = "Unexpected error"


class EnrollmentError(Exception):
    """Base exception for enrollment errors"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class EnrollmentValidationError(EnrollmentError):
    """Raised when validation fails"""

    pass


@dataclass
class EnrollmentResponse:
    code: int
    message: Optional[str] = None
    anti_spoofing: Optional[dict] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "anti_spoofing": self.anti_spoofing,
            "details": self.details,
        }


def handle_enrollment_request(request: Request, pinecone_service) -> Tuple[Any, int]:
    """
    Endpoint to enroll a new face in the system.

    Expected form data:
        - firstName (required): Person's first name
        - lastName (required): Person's last name
        - id: Unique identifier
        - age: Person's age
        - gender: Person's gender
        - email: Person's email
        - phone: Person's phone number
        - image (required): Image file containing the face

    Returns:
        tuple: (JSON response, HTTP status code)
            Response contains:
                - success (bool): Whether enrollment was successful
                - message (str): Success/error message
                - details (dict): Processing details and timing information
                - error (str, optional): Error message if unsuccessful
    """
    start_time = time.time()

    try:
        # Request validation
        form_data = request.form
        image_file = request.files.get("image")

        logging.info("Received enrollment request")
        logging.debug(
            "Form data: %s",
            {k: v for k, v in form_data.items() if k not in ["email", "phone"]},
        )

        # Validate required fields
        required_fields = {
            "firstName": form_data.get("firstName"),
            "lastName": form_data.get("lastName"),
            "image": image_file and image_file.filename,
        }

        missing_fields = [
            field for field, value in required_fields.items() if not value
        ]
        if missing_fields:
            raise EnrollmentValidationError(
                EnrollmentCodes.UNEXPECTED_ERROR,
                EnrollmentMessages.UNEXPECTED_ERROR,
            )

        # Process face image
        face_processing_start = time.time()
        face_processing_task = process_face_image(image_file)

        # Check for face processing errors
        if face_processing_task["code"] != EnrollmentCodes.SUCCESS:
            response = EnrollmentResponse(
                code=face_processing_task["code"],
                message=face_processing_task["message"],
                anti_spoofing=face_processing_task.get("anti_spoofing"),
                details={
                    "processing_times": {
                        "total": round(time.time() - start_time, 3),
                        "face_processing": round(
                            time.time() - face_processing_start, 3
                        ),
                        "database_storage": None,
                    },
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

        # Store embeddings
        storage_start = time.time()
        storage_result = pinecone_service.store_embeddings(
            id=form_data.get("id") or str(uuid.uuid4()),
            firstName=form_data.get("firstName"),
            lastName=form_data.get("lastName"),
            age=form_data.get("age"),
            gender=form_data.get("gender"),
            email=form_data.get("email"),
            phone=form_data.get("phone"),
            embeddings=[face_processing_task["embeddings"][0]],
        )

        storage_time = time.time() - storage_start

        if storage_result["code"] != EnrollmentCodes.SUCCESS:
            response = EnrollmentResponse(
                code=EnrollmentCodes.STORAGE_ERROR,
                message=storage_result["message"],
                anti_spoofing=face_processing_task.get("anti_spoofing"),
                details={
                    "processing_times": {
                        "total": round(time.time() - start_time, 3),
                        "face_processing": round(face_processing_time, 3),
                        "database_storage": round(storage_time, 3),
                    },
                },
            )
            return jsonify(response.to_dict()), 200

        # Prepare success response
        total_time = time.time() - start_time
        response = EnrollmentResponse(
            code=EnrollmentCodes.SUCCESS,
            message=EnrollmentMessages.SUCCESS,
            anti_spoofing=face_processing_task.get("anti_spoofing"),
            details={
                "processing_times": {
                    "total": round(total_time, 3),
                    "face_processing": round(face_processing_time, 3),
                    "database_storage": round(storage_time, 3),
                },
            },
        )

        logging.info(
            "Enrollment successful - processing_time=%.3fs, face_processing=%.3fs, storage=%.3fs",
            total_time,
            face_processing_time,
            storage_time,
        )
        return jsonify(response), 200

    except ValueError as ve:
        error_msg = str(ve)
        logging.warning("Validation error during enrollment: %s", error_msg)
        response = EnrollmentResponse(
            code=EnrollmentCodes.FUNCTION_ERROR,
            message=error_msg,
        ).to_dict()
        return jsonify(response), 400

    except RuntimeError as re:
        error_msg = str(re)
        logging.error("Runtime error during enrollment: %s", error_msg)
        response = EnrollmentResponse(
            code=EnrollmentCodes.FUNCTION_ERROR,
            message=error_msg,
        ).to_dict()
        return jsonify(response), 500

    except Exception as e:
        error_msg = f"Unexpected error during enrollment: {str(e)}"
        logging.exception(error_msg)
        response = EnrollmentResponse(
            code=EnrollmentCodes.UNEXPECTED_ERROR,
            message=error_msg,
        ).to_dict()
        return jsonify(response), 500
