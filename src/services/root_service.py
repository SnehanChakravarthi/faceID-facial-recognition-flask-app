import logging
import os
import time

from flask import jsonify

from ..config import Models, Settings


def handle_root_request():
    """Base endpoint providing service information, status, and API documentation."""
    try:
        service_info = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "service": {
                "name": "face-id-api",
                "version": os.getenv("SERVICE_VERSION", "1.0.0"),
                "environment": os.getenv("PYTHON_ENV", "production"),
                "description": "Face recognition and authentication service using DeepFace",
            },
            "endpoints": {
                "health": {
                    "path": "/health",
                    "method": "GET",
                    "description": "Detailed health check endpoint for monitoring",
                },
                "enroll": {
                    "path": "/api/v1/enroll",
                    "method": "POST",
                    "description": "Enroll a new face in the system",
                    "content_type": "multipart/form-data",
                    "parameters": {
                        "image": "Image file containing a face",
                        "firstName": "First name of the person",
                        "lastName": "Last name of the person",
                        "age": "Age of the person (optional)",
                        "gender": "Gender of the person (optional)",
                        "email": "Email address (optional)",
                        "phone": "Phone number (optional)",
                    },
                },
                "authenticate": {
                    "path": "/api/v1/authenticate",
                    "method": "POST",
                    "description": "Authenticate a face against enrolled faces",
                    "content_type": "multipart/form-data",
                    "parameters": {
                        "image": "Image file containing a face to authenticate"
                    },
                },
            },
            "configuration": {
                "face_recognition": {
                    "available_models": Models.FACE_RECOGNITION_MODELS,
                    "current_model": Settings.FACE_RECOGNITION_MODEL,
                    "available_detectors": Models.FACE_DETECTION_MODELS,
                    "current_detector": Settings.DETECTOR_BACKEND,
                },
                "anti_spoofing": {
                    "available_backends": Models.ANTI_SPOOFING_MODEL,
                    "current_backend": Models.ANTI_SPOOFING_MODEL[0],
                },
            },
        }
        return jsonify(service_info), 200

    except Exception as e:
        logging.error(f"Error in root endpoint: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "timestamp": int(time.time()),
                    "message": "Internal server error",
                }
            ),
            500,
        )
