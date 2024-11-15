import logging
import os
from deepface import DeepFace
from flask import Flask, jsonify, request
from pinecone import Pinecone
from pinecone import ServerlessSpec
import uuid
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class Models:
    FACE_RECOGNITION_MODELS = [
        "VGG-Face",  # 0
        "OpenFace",  # 1
        "Facenet",  # 2
        "Facenet512",  # 3
        "DeepFace",  # 4
        "DeepID",  # 5
        "Dlib",  # 6
        "ArcFace",  # 7
        "SFace",  # 8
        "GhostFaceNet",  # 9
    ]
    FACE_DETECTION_MODELS = [
        "opencv",
        "mtcnn",  # 1
        "ssd",  # 2
        "dlib",  # 3
        "retinaface",  # 4
        "mediapipe",  # 5
        "yolov8",  # 6
        "yunet",  # 7
        "fastmtcnn",  # 8
        "centerface",  # 9
    ]
    ANTI_SPOOFING_MODEL = ["Fasnet"]


class Config:
    TEMP_IMAGE_PATH = "/tmp"
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


class Settings:
    MATCH_THRESHOLD = 0.5
    FACE_RECOGNITION_MODEL = Models.FACE_RECOGNITION_MODELS[7]  # "ArcFace"
    DETECTOR_BACKEND = Models.FACE_DETECTION_MODELS[0]  # "opencv"
    ANTI_SPOOFING_DETECTOR_BACKEND = Models.FACE_DETECTION_MODELS[0]  # "opencv"


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="\033[1;32m%(asctime)s\033[0m - \033[1;34m%(levelname)s\033[0m - %(message)s",
)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
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
            "dependencies": {
                "deepface": {
                    "status": "healthy",
                    "version": DeepFace.__version__,
                    "models": {
                        "face_recognition_model": Settings.FACE_RECOGNITION_MODEL,
                        "face_detection_model": Settings.DETECTOR_BACKEND,
                        "anti_spoofing_model": Models.ANTI_SPOOFING_MODEL[0],
                    },
                },
                "pinecone": {"status": "healthy"},
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

        # Quick dependency checks
        try:
            pc = Pinecone()
            pc.list_indexes()
        except Exception as e:
            service_info["dependencies"]["pinecone"]["status"] = "degraded"
            service_info["status"] = "degraded"
            logging.warning(f"Pinecone health check failed: {str(e)}")

        try:
            DeepFace.build_model(Models.FACE_RECOGNITION_MODEL)
        except Exception as e:
            service_info["dependencies"]["deepface"]["status"] = "degraded"
            service_info["status"] = "degraded"
            logging.warning(f"DeepFace model build failed: {str(e)}")

        status_code = 200 if service_info["status"] == "healthy" else 503
        return jsonify(service_info), status_code

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


@app.route("/health", methods=["GET"])
def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "components": {
            "deepface": {
                "status": "healthy",
                "version": DeepFace.__version__,
                "models": {
                    "face_recognition": Models.FACE_RECOGNITION_MODEL,
                    "detector": Models.DETECTOR_BACKEND,
                    "anti_spoofing": Models.ANTI_SPOOFING_MODEL[0],
                },
            },
            "pinecone": {"status": "unknown"},
        },
        "environment": {
            "python_env": os.environ.get("PYTHON_ENV", "development"),
            "debug_mode": app.debug,
        },
    }

    # Check Pinecone connection
    try:
        pc = Pinecone()
        index_list = pc.list_indexes()
        health_status["components"]["pinecone"]["status"] = "healthy"
    except Exception as e:
        health_status["components"]["pinecone"]["status"] = "unhealthy"
        health_status["components"]["pinecone"]["error"] = str(e)
        health_status["status"] = "degraded"

    # Check if models are accessible
    try:
        # Attempt to load the face recognition model
        DeepFace.build_model(Models.FACE_RECOGNITION_MODEL)
    except Exception as e:
        health_status["components"]["deepface"]["status"] = "unhealthy"
        health_status["components"]["deepface"]["error"] = str(e)
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code


@app.route("/api/v1/enroll", methods=["POST"])
def enroll():
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

    response = {
        "success": False,
        "message": None,
        "details": {
            "processing_times": {
                "total": None,
                "face_processing": None,
                "database_storage": None,
            },
            "face_detection": None,
            "anti_spoofing": None,
            "embeddings_stored": None,
        },
        "error": None,
    }

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
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Missing required fields: {', '.join(missing_fields)}",
                    }
                ),
                400,
            )

        # Initialize tasks concurrently
        face_processing_task = None
        pinecone_service = None

        try:
            # Start both tasks concurrently using threads
            face_processing_start = time.time()
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                face_processing_future = executor.submit(process_face_image, image_file)
                pinecone_future = executor.submit(PineconeService)

                # Wait for both tasks to complete
                face_processing_task = face_processing_future.result()
                pinecone_service = pinecone_future.result()

        except ValueError as ve:
            raise ValueError(f"Initialization error: {str(ve)}")
        except Exception as e:
            raise RuntimeError(f"Setup failed: {str(e)}")

        # Validate face processing results
        if not face_processing_task["success"]:
            raise ValueError(face_processing_task["error"] or "Face processing failed")

        # try:
        #     # Start face processing
        #     face_processing_start = time.time()
        #     face_processing_task = process_face_image(image_file)

        #     # Initialize Pinecone service while face is being processed
        #     pinecone_service = PineconeService()

        # except ValueError as ve:
        #     raise ValueError(f"Initialization error: {str(ve)}")
        # except Exception as e:
        #     raise RuntimeError(f"Setup failed: {str(e)}")

        # # Validate face processing results
        # if not face_processing_task["success"]:
        #     raise ValueError(face_processing_task["error"] or "Face processing failed")

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

        if not storage_result["success"]:
            raise RuntimeError(f"Database storage failed: {storage_result['error']}")

        # Prepare success response
        total_time = time.time() - start_time

        response.update(
            {
                "success": True,
                "message": "Enrollment successful",
                "details": {
                    "processing_times": {
                        "total": round(total_time, 3),
                        "face_processing": round(face_processing_time, 3),
                        "database_storage": round(storage_time, 3),
                    },
                    "face_detection": face_processing_task["details"]["face_detected"],
                    "anti_spoofing": face_processing_task["details"]["anti_spoofing"],
                    "embeddings_stored": storage_result["vectors_stored"],
                    "model_info": face_processing_task["details"]["model_info"],
                },
            }
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
        response.update(
            {
                "error": error_msg,
                "details": {**response["details"], "error_type": "validation"},
            }
        )
        return jsonify(response), 400

    except RuntimeError as re:
        error_msg = str(re)
        logging.error("Runtime error during enrollment: %s", error_msg)
        response.update(
            {
                "error": error_msg,
                "details": {**response["details"], "error_type": "runtime"},
            }
        )
        return jsonify(response), 500

    except Exception as e:
        error_msg = f"Unexpected error during enrollment: {str(e)}"
        logging.exception(error_msg)
        response.update(
            {
                "error": error_msg,
                "details": {**response["details"], "error_type": "unexpected"},
            }
        )
        return jsonify(response), 500


@app.route("/api/v1/authenticate", methods=["POST"])
def authenticate():
    """
    Endpoint to authenticate a face against enrolled faces.

    Expected form data:
        - image (required): Image file containing the face to authenticate

    Returns:
        tuple: (JSON response, HTTP status code)
            Response contains:
                - success (bool): Whether authentication was successful
                - message (str): Success/error message
                - details (dict): Processing details and timing information
                - match (dict, optional): Matching face details if found
                - error (str, optional): Error message if unsuccessful
    """
    start_time = time.time()

    response = {
        "success": False,
        "message": None,
        "details": {
            "processing_times": {
                "total": None,
                "face_processing": None,
                "search": None,
            },
            "face_detection": None,
            "anti_spoofing": None,
            "match_found": False,
            "similarity_score": None,
        },
        "match": None,
        "error": None,
    }

    try:
        # Request validation
        image_file = request.files.get("image")

        logging.info("Received authentication request")

        if not image_file or not image_file.filename:
            return (
                jsonify({"success": False, "error": "Missing required image file"}),
                400,
            )

        # Initialize tasks concurrently
        face_processing_task = None
        pinecone_service = None

        try:
            # Start both tasks concurrently using threads
            face_processing_start = time.time()
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                face_processing_future = executor.submit(process_face_image, image_file)
                pinecone_future = executor.submit(PineconeService)

                # Wait for both tasks to complete
                face_processing_task = face_processing_future.result()
                pinecone_service = pinecone_future.result()

        except ValueError as ve:
            raise ValueError(f"Initialization error: {str(ve)}")
        except Exception as e:
            raise RuntimeError(f"Setup failed: {str(e)}")

        # Validate face processing results
        if not face_processing_task["success"]:
            raise ValueError(face_processing_task["error"] or "Face processing failed")

        face_processing_time = time.time() - face_processing_start

        # Search for similar faces
        search_start = time.time()
        search_result = pinecone_service.search_similar_faces(
            query_vector=face_processing_task["embeddings"][0],
            threshold=Settings.MATCH_THRESHOLD,
        )
        search_time = time.time() - search_start

        if not search_result["success"]:
            raise RuntimeError(f"Face search failed: {search_result.get('error')}")

        # Prepare success response
        total_time = time.time() - start_time

        # Update response based on search results
        match_found = bool(search_result.get("match"))
        similarity_score = search_result.get("details", {}).get("similarity_score")

        response.update(
            {
                "success": True,
                "message": (
                    "Authentication successful"
                    if match_found
                    else "No matching face found"
                ),
                "details": {
                    "processing_times": {
                        "total": round(total_time, 3),
                        "face_processing": round(face_processing_time, 3),
                        "search": round(search_time, 3),
                    },
                    "face_detection": face_processing_task["details"]["face_detected"],
                    "anti_spoofing": face_processing_task["details"]["anti_spoofing"],
                    "match_found": match_found,
                    "similarity_score": similarity_score,
                    "threshold_used": Settings.MATCH_THRESHOLD,
                    "model_info": face_processing_task["details"]["model_info"],
                },
                "match": search_result.get("match"),
            }
        )

        status_code = 200 if match_found else 404

        logging.info(
            "Authentication completed - match_found=%s, similarity=%.3f, processing_time=%.3fs",
            match_found,
            similarity_score or 0,
            total_time,
        )
        return jsonify(response), status_code

    except ValueError as ve:
        error_msg = str(ve)
        logging.warning("Validation error during authentication: %s", error_msg)
        response.update(
            {
                "error": error_msg,
                "details": {**response["details"], "error_type": "validation"},
            }
        )
        return jsonify(response), 400

    except RuntimeError as re:
        error_msg = str(re)
        logging.error("Runtime error during authentication: %s", error_msg)
        response.update(
            {
                "error": error_msg,
                "details": {**response["details"], "error_type": "runtime"},
            }
        )
        return jsonify(response), 500

    except Exception as e:
        error_msg = f"Unexpected error during authentication: {str(e)}"
        logging.exception(error_msg)
        response.update(
            {
                "error": error_msg,
                "details": {**response["details"], "error_type": "unexpected"},
            }
        )
        return jsonify(response), 500


def process_face_image(image_file):
    """
    Process an uploaded face image through multiple steps:
    1. Save image temporarily
    2. Perform anti-spoofing check
    3. Generate face embeddings

    Args:
        image_file: FileStorage object from Flask's request.files

    Returns:
        dict: A response containing:
            - success (bool): Overall success status
            - embeddings (list|None): Generated face embeddings if successful
            - error (str|None): Error message if any
            - details (dict): Processing details including:
                - anti_spoofing (dict): Anti-spoofing check results
                - temp_path (str): Temporary file path
                - face_detected (bool): Whether a face was detected
                - model_info (dict): Model information
                - processing_time (float): Total processing time in seconds

    Raises:
        ValueError: For validation errors
        IOError: For file handling errors
    """
    start_time = time.time()

    response = {
        "success": False,
        "embeddings": None,
        "error": None,
        "details": {
            "anti_spoofing": None,
            "temp_path": None,
            "face_detected": False,
            "model_info": None,
            "processing_time": None,
        },
    }

    temp_image_path = None

    try:
        # Step 1: Save image temporarily
        logging.info("Starting image processing pipeline")
        try:
            temp_image_path = save_image_temporarily(image_file)
            response["details"]["temp_path"] = temp_image_path
            logging.debug("Image saved temporarily at: %s", temp_image_path)
        except (ValueError, IOError) as e:
            raise ValueError(f"Failed to save image: {str(e)}")

        # Step 2: Perform anti-spoofing check
        logging.debug("Initiating anti-spoofing check")
        anti_spoofing_result = check_anti_spoofing(temp_image_path)
        response["details"]["anti_spoofing"] = anti_spoofing_result
        response["details"]["face_detected"] = anti_spoofing_result.get(
            "face_detected", False
        )

        if not anti_spoofing_result["success"]:
            raise ValueError(
                f"Anti-spoofing check failed: {anti_spoofing_result['error']}"
            )

        if not anti_spoofing_result["is_real"]:
            raise ValueError(
                f"Potential spoofing detected - confidence: {anti_spoofing_result['confidence']}, "
                f"score: {anti_spoofing_result['antispoof_score']}"
            )

        # Step 3: Generate embeddings
        logging.debug("Generating face embeddings")
        embedding_result = generate_embeddings(temp_image_path)

        if not embedding_result["success"]:
            raise ValueError(
                f"Failed to generate embeddings: {embedding_result['error']}"
            )

        # Update response with embedding results
        response.update(
            {
                "success": True,
                "embeddings": embedding_result["embeddings"],
                "details": {
                    **response["details"],
                    "model_info": embedding_result["model_info"],
                    "face_detected": embedding_result["face_detected"],
                },
            }
        )

        processing_time = time.time() - start_time
        response["details"]["processing_time"] = round(processing_time, 3)

        logging.info(
            "Image processing completed successfully in %.3f seconds", processing_time
        )
        return response

    except ValueError as ve:
        error_msg = f"Validation error in process_face_image: {str(ve)}"
        logging.warning(error_msg)
        response["error"] = str(ve)
        return response

    except Exception as e:
        error_msg = f"Unexpected error in process_face_image: {str(e)}"
        logging.exception(error_msg)
        response["error"] = f"Failed to process image: {str(e)}"
        return response

    finally:
        # Clean up temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logging.debug("Cleaned up temporary file: %s", temp_image_path)
            except Exception as e:
                logging.error(
                    "Failed to clean up temporary file %s: %s", temp_image_path, str(e)
                )


def save_image_temporarily(image_file):
    """
    Safely saves an uploaded image file to a temporary location with security measures.

    Args:
        image_file: FileStorage object from Flask's request.files

    Returns:
        str: Path to the saved temporary file

    Raises:
        ValueError: If file validation fails
        IOError: If file saving operations fail
    """
    try:
        # Validate file type
        if not image_file.content_type.startswith("image/"):
            raise ValueError(f"Invalid file type: {image_file.content_type}")

        # Generate secure filename with UUID to prevent conflicts
        file_extension = os.path.splitext(image_file.filename)[1].lower()
        if file_extension not in (".png", ".jpg", ".jpeg", ".webp"):
            raise ValueError(f"Unsupported file extension: {file_extension}")

        secure_filename = f"{uuid.uuid4()}{file_extension}"

        # Ensure temp directory exists
        os.makedirs(Config.TEMP_IMAGE_PATH, exist_ok=True)

        # Create full path with secure filename
        image_path = os.path.join(Config.TEMP_IMAGE_PATH, secure_filename)

        # Check file size before saving (e.g., 10MB limit)
        image_file.seek(0, os.SEEK_END)
        size = image_file.tell()
        if size > 10 * 1024 * 1024:  # 10MB
            raise ValueError("File size exceeds maximum limit of 10MB")
        image_file.seek(0)  # Reset file pointer

        # Save file
        image_file.save(image_path)

        # Verify file was saved and is readable
        if not os.path.exists(image_path):
            raise IOError("Failed to save image file")

        return image_path

    except (ValueError, IOError) as e:
        logging.error(f"Failed to save image temporarily: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while saving image: {str(e)}")
        raise IOError(f"Failed to save image: {str(e)}")


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


def generate_embeddings(image_path):
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
    logging.debug("Starting face embedding generation for image: %s", image_path)

    response = {
        "success": False,
        "embeddings": None,
        "error": None,
        "face_detected": False,
        "model_info": {
            "face_recognition_model": Settings.FACE_RECOGNITION_MODEL,
            "face_detection_model": Settings.DETECTOR_BACKEND,
        },
    }

    # Validate input
    if not image_path or not isinstance(image_path, str):
        raise ValueError("Invalid image path provided")

    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")

    try:
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
            response.update(
                {
                    "error": "No embeddings generated - face may not be detected",
                    "face_detected": False,
                }
            )
            logging.warning("Embedding generation failed: no embeddings generated")
            return response

        # Validate embeddings structure and values
        if isinstance(
            embeddings, dict
        ):  # Handle case where single embedding is returned as dict
            embeddings = [embeddings]
        elif not isinstance(embeddings, list):
            embeddings = [embeddings]  # Ensure consistent list format

        processed_embeddings = []
        for embedding in embeddings:
            # Handle case where embedding is a dict with 'embedding' key
            if isinstance(embedding, dict) and "embedding" in embedding:
                embedding_vector = embedding["embedding"]
            else:
                embedding_vector = embedding

            # Validate and convert the embedding vector
            if (
                not isinstance(embedding_vector, (list, np.ndarray))
                or len(embedding_vector) == 0
            ):
                raise ValueError("Invalid embedding format received from DeepFace")

            # Convert numpy arrays to lists for JSON serialization
            if isinstance(embedding_vector, np.ndarray):
                embedding_vector = embedding_vector.tolist()

            processed_embeddings.append(embedding_vector)

        embeddings = (
            processed_embeddings  # Replace original embeddings with processed ones
        )

        # Validate embedding dimensions
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

        expected_dim = expected_dims.get(Settings.FACE_RECOGNITION_MODEL)
        if expected_dim and len(embeddings[0]) != expected_dim:
            raise ValueError(
                f"Unexpected embedding dimension: got {len(embeddings[0])}, "
                f"expected {expected_dim} for model {Settings.FACE_RECOGNITION_MODEL}"
            )

        response.update(
            {
                "success": True,
                "embeddings": embeddings,
                "face_detected": True,
                "model_info": {
                    **response["model_info"],
                    "embedding_dimension": len(embeddings[0]),
                    "embeddings_count": len(embeddings),
                },
            }
        )

        logging.info(
            "Embedding generation successful: generated %d embeddings with dimension %d",
            len(embeddings),
            len(embeddings[0]),
        )
        return response

    except ValueError as ve:
        error_msg = f"Validation error in embedding generation: {str(ve)}"
        logging.error(error_msg)
        response["error"] = error_msg
        return response

    except Exception as e:
        error_msg = f"Unexpected error in embedding generation: {str(e)}"
        logging.exception(error_msg)  # This logs the full stack trace
        response["error"] = error_msg
        return response


class PineconeService:
    """
    Service class for managing face embeddings storage in Pinecone.

    Attributes:
        pc: Pinecone client instance
        index_name (str): Name of the Pinecone index
        dimension (int): Dimension of face embeddings
        batch_size (int): Size of batches for vector upsert operations
    """

    def __init__(self, api_key=None):
        """
        Initialize PineconeService with configuration.

        Args:
            api_key (str, optional): Pinecone API key. Defaults to Config.PINECONE_API_KEY

        Raises:
            ValueError: If API key is not provided or invalid
        """
        self.dimension = 512  # ArcFace embedding dimension
        self.batch_size = 100
        self.index_name = "deepface-embeddings"

        if not api_key and not Config.PINECONE_API_KEY:
            raise ValueError("Pinecone API key is required")

        try:
            self.pc = Pinecone(api_key=api_key or Config.PINECONE_API_KEY)
            self.ensure_index_exists()
        except Exception as e:
            logging.error("Failed to initialize Pinecone client: %s", str(e))
            raise

    def ensure_index_exists(self):
        """
        Create Pinecone index if it doesn't exist.

        Raises:
            RuntimeError: If index creation fails
        """
        try:
            existing_indexes = self.pc.list_indexes().names()

            if self.index_name not in existing_indexes:
                logging.info("Creating new Pinecone index: %s", self.index_name)

                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logging.info("Successfully created Pinecone index")
            else:
                logging.debug("Pinecone index already exists")

        except Exception as e:
            error_msg = f"Failed to ensure index exists: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def store_embeddings(
        self, id, firstName, lastName, age, gender, email, phone, embeddings
    ):
        """
        Store face embeddings in Pinecone with associated metadata.

        Args:
            id (str): Unique identifier for the person
            firstName (str): Person's first name
            lastName (str): Person's last name
            age (str): Person's age
            gender (str): Person's gender
            email (str): Person's email
            phone (str): Person's phone number
            embeddings (list): List of face embedding vectors

        Returns:
            dict: Response containing:
                - success (bool): Whether the operation was successful
                - message (str): Success message if successful
                - vectors_stored (int): Number of vectors stored
                - error (str, optional): Error message if unsuccessful

        Raises:
            ValueError: If input validation fails
            RuntimeError: If Pinecone operations fail
        """
        start_time = time.time()

        # Input validation
        if not all([id, firstName, lastName]):
            raise ValueError("ID, first name, and last name are required")

        if not embeddings or not isinstance(embeddings, list):
            raise ValueError("Valid embeddings list is required")

        try:
            index = self.pc.Index(self.index_name)
            vectors = []
            timestamp = int(time.time())

            # Prepare vectors for upsert
            for i, embedding in enumerate(embeddings):
                # Validate embedding dimension
                if len(embedding) != self.dimension:
                    raise ValueError(
                        f"Invalid embedding dimension: expected {self.dimension}, got {len(embedding)}"
                    )

                # Generate unique vector ID
                vector_id = f"{firstName.lower().replace(' ', '-')}-{lastName.lower().replace(' ', '-')}-{uuid.uuid4()}"

                # Prepare vector with metadata
                vector = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "id": id,
                        "firstName": firstName,
                        "lastName": lastName,
                        "age": age,
                        "gender": gender,
                        "email": email,
                        "phone": phone,
                        "timestamp": timestamp,
                        "embedding_number": i + 1,
                        "total_embeddings": len(embeddings),
                    },
                }
                vectors.append(vector)

            # Upsert vectors in batches
            total_vectors = len(vectors)
            vectors_stored = 0

            for i in range(0, total_vectors, self.batch_size):
                batch = vectors[i : i + self.batch_size]
                index.upsert(vectors=batch)
                vectors_stored += len(batch)
                logging.debug(
                    "Stored batch of %d vectors (%d/%d)",
                    len(batch),
                    vectors_stored,
                    total_vectors,
                )

            processing_time = time.time() - start_time

            success_message = (
                f"Successfully stored {vectors_stored} embeddings for {firstName} {lastName} "
                f"in {processing_time:.2f} seconds"
            )
            logging.info(success_message)

            return {
                "success": True,
                "message": success_message,
                "vectors_stored": vectors_stored,
                "processing_time": round(processing_time, 3),
            }

        except ValueError as ve:
            error_msg = f"Validation error in store_embeddings: {str(ve)}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg, "vectors_stored": 0}

        except Exception as e:
            error_msg = f"Failed to store embeddings: {str(e)}"
            logging.exception(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "vectors_stored": vectors_stored if "vectors_stored" in locals() else 0,
            }

    def search_similar_faces(self, query_vector, threshold=0.6):
        """
        Search for the most similar face in the Pinecone index.

        Args:
            query_vector (list|np.ndarray): Face embedding vector to search for
            threshold (float, optional): Similarity threshold (0-1). Defaults to 0.6.

        Returns:
            dict: Response containing:
                - success (bool): Whether the search was successful
                - match (dict|None): Best matching result if successful and above threshold
                - error (str|None): Error message if unsuccessful
                - details (dict): Additional search details including:
                    - similarity_score: Similarity score of the match
                    - threshold_used: Similarity threshold applied
                    - processing_time: Search execution time

        Raises:
            ValueError: If input validation fails
            RuntimeError: If Pinecone operations fail
        """
        start_time = time.time()

        response = {
            "success": False,
            "match": None,
            "error": None,
            "details": {
                "similarity_score": None,
                "threshold_used": threshold,
                "processing_time": None,
            },
        }

        try:
            # Input validation
            if not isinstance(query_vector, (list, np.ndarray)):
                raise ValueError("Query vector must be a list or numpy array")

            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
                )

            if not 0 <= threshold <= 1:
                raise ValueError("Threshold must be between 0 and 1")

            # Initialize index and perform search
            index = self.pc.Index(self.index_name)
            search_result = index.query(
                vector=query_vector,
                top_k=1,  # Only fetch the best match
                include_metadata=True,
                include_values=False,
            )

            if not search_result or not search_result["matches"]:
                logging.warning("No matches found in search results")
                response["error"] = "No matches found"
                return response

            # Process the single match
            best_match = search_result["matches"][0]
            similarity_score = round(best_match.get("score", 0), 4)

            # Check if match meets threshold
            if similarity_score < threshold:
                logging.info(
                    "Best match (score=%.4f) below threshold (%.2f)",
                    similarity_score,
                    threshold,
                )
                response["error"] = f"No matches found above threshold ({threshold})"
                response["details"]["similarity_score"] = similarity_score
                return response

            # Format the match result
            response.update(
                {
                    "success": True,
                    "match": {
                        "id": best_match.get("id"),
                        "score": similarity_score,
                        "metadata": best_match.get("metadata", {}),
                    },
                    "details": {
                        "similarity_score": similarity_score,
                        "threshold_used": threshold,
                        "processing_time": round(time.time() - start_time, 3),
                    },
                }
            )

            logging.info(
                "Face search completed - match_found=True, similarity=%.4f, time=%.3fs",
                similarity_score,
                response["details"]["processing_time"],
            )
            return response

        except ValueError as ve:
            error_msg = f"Validation error in search_similar_faces: {str(ve)}"
            logging.error(error_msg)
            response["error"] = error_msg
            return response

        except Exception as e:
            error_msg = f"Unexpected error in search_similar_faces: {str(e)}"
            logging.exception(error_msg)
            response["error"] = error_msg
            return response


if __name__ == "__main__":
    app.run(debug=True)
