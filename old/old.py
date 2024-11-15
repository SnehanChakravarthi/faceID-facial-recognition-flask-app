import os
from deepface import DeepFace
from flask import Flask, jsonify, request
from pinecone import Pinecone
from pinecone import ServerlessSpec
import uuid
import time
from PIL import Image
from werkzeug.datastructures import FileStorage
from typing import Optional
import re
import logging
from datetime import datetime

# from dotenv import load_dotenv
# load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    TEMP_IMAGE_PATH = "/path/to/temp/images"
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    # CONTENT_TYPE = "multipart/form-data"


class Models:
    FACE_RECOGNITION_MODELS = [
        "VGG-Face",
        "OpenFace",
        "Facenet",
        "Facenet512",
        "DeepFace",
        "DeepID",
        "Dlib",
        "ArcFace",
        "SFace",
        "GhostFaceNet",
    ]
    DETECTOR_BACKENDS = [
        "opencv",
        "mtcnn",
        "ssd",
        "dlib",
        "retinaface",
        "mediapipe",
        "yolov8",
        "yunet",
        "fastmtcnn",
        "centerface",
    ]
    ANTI_SPOOFING_BACKENDS = ["Fasnet"]

    # Use indices to select the models
    FACE_RECOGNITION_MODEL = FACE_RECOGNITION_MODELS[7]  # "ArcFace"
    DETECTOR_BACKEND = DETECTOR_BACKENDS[0]  # "opencv"
    ANTI_SPOOFING_DETECTOR_BACKEND = DETECTOR_BACKENDS[0]  # "opencv"


app = Flask(__name__)


@app.route("/api/v1/enroll", methods=["POST"])
def enroll():
    try:
        # Log the headers for debugging
        form_data = request.form
        logging.debug("Received form data: %s", form_data)

        image_file = request.files.get("image")
        logging.debug("Received image file: %s", image_file)

        # Basic validation
        if not form_data.get("firstName") or not form_data.get("lastName"):
            raise ValueError("First name and last name are required")

        # Image file is required to authenticate
        if not image_file or not image_file.filename:
            raise ValueError("Image file is required")

        logging.debug(
            "Received image file with filename: %s, content type: %s",
            image_file.filename,
            image_file.content_type,
        )

        # Save the image temporarily in /tmp
        image_path = save_image_temporarily(image_file)
        if not image_path:
            raise RuntimeError("Failed to save image temporarily")

        logging.debug("Starting authentication process.")

        # Check for anti-spoofing and if the image is real
        spoofing_result = check_anti_spoofing(image_path)
        if not spoofing_result.get("is_real", False):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Spoofing or no real face detected",
                        "is_real": spoofing_result.get("is_real", False),
                        "antispoof_score": spoofing_result.get("antispoof_score", 0.0),
                        "confidence": spoofing_result.get("confidence", 0.0),
                    }
                ),
                400,
            )

        # Generate face embeddings for the image
        embeddings_response = generate_embeddings(image_path)
        if not embeddings_response.get("success", False):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": embeddings_response.get(
                            "error", "Failed to generate embeddings"
                        ),
                    }
                ),
                400,
            )

        embeddings = embeddings_response.get("embeddings")
        if not embeddings:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No embeddings generated.",
                    }
                ),
                400,
            )

        pinecone_service = PineconeService()

        # Store embeddings in the database
        logging.debug("Storing embeddings in the database.")
        result = pinecone_service.store_embeddings(
            id=form_data.get("id"),
            firstName=form_data.get("firstName"),
            lastName=form_data.get("lastName"),
            age=form_data.get("age"),
            gender=form_data.get("gender"),
            email=form_data.get("email"),
            phone=form_data.get("phone"),
            embeddings=[embeddings[0]["embedding"]],
        )

        if not result["success"]:
            logging.error(f"Failed to store embeddings: {result['error']}")
            return jsonify({"error": result["error"]}), 500

        logging.info("Enrollment successful.")
        return jsonify({"message": "Enrollment successful"}), 200

    except ValueError as ve:
        logging.error(f"ValueError during enrollment: {str(ve)}")
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        logging.exception("Unexpected error during enrollment")
        return jsonify({"error": "An unexpected error occurred during enrollment"}), 500

    finally:
        # Clean up the temporary image file
        if os.path.exists(image_path):
            logging.debug("Removing temporary image file.")
            os.remove(image_path)


@app.route("/api/v1/authenticate", methods=["POST"])
def authenticate():
    logging.debug("Received authentication request from client.")

    # Image file is required to authenticate
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Image file is required",
                }
            ),
            400,
        )

    logging.debug(
        "Received image file with filename: %s, content type: %s",
        image_file.filename,
        image_file.content_type,
    )

    # Save the image temporarily in /tmp
    image_path = save_image_temporarily(image_file)
    if not image_path:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Failed to save image temporarily",
                }
            ),
            500,
        )

    try:
        logging.debug("Starting authentication process.")

        # Check for anti-spoofing and if the image is real
        spoofing_result = check_anti_spoofing(image_path)
        if not spoofing_result.get("is_real", False):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Spoofing or no real face detected",
                        "is_real": spoofing_result.get("is_real", False),
                        "antispoof_score": spoofing_result.get("antispoof_score", 0.0),
                        "confidence": spoofing_result.get("confidence", 0.0),
                    }
                ),
                400,
            )

        # Generate face embeddings for the image
        embeddings_response = generate_embeddings(image_path)
        if not embeddings_response.get("success", False):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": embeddings_response.get(
                            "error", "Failed to generate embeddings"
                        ),
                    }
                ),
                400,
            )

        embeddings = embeddings_response.get("embeddings")
        if not embeddings:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No embeddings generated.",
                    }
                ),
                400,
            )

        # Initialize PineconeService
        pinecone_service = PineconeService()

        # Perform vector search using the first embedding
        query_vector = embeddings[0]["embedding"]
        search_results = pinecone_service.search_similar_faces(query_vector, top_k=2)

        # If no matches are found
        if not search_results.get("match_found", False):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": search_results.get(
                            "message", "No matching faces found"
                        ),
                        "is_real": spoofing_result.get("is_real", False),
                        "antispoof_score": spoofing_result.get("antispoof_score", 0.0),
                        "confidence": spoofing_result.get("confidence", 0.0),
                    }
                ),
                401,
            )

        # If matches are found
        response = {
            "success": True,
            "matches": search_results.get("matches"),
            "is_real": spoofing_result.get("is_real", False),
            "antispoof_score": spoofing_result.get("antispoof_score", 0.0),
            "confidence": spoofing_result.get("confidence", 0.0),
        }
        logging.debug(f"Authentication response: {response}")
        return jsonify(response), 200

    except Exception as e:
        logging.exception("Unexpected error during authentication")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "message": "An unexpected error occurred during authentication",
                }
            ),
            500,
        )

    finally:
        # Clean up the temporary image file
        if os.path.exists(image_path):
            logging.debug("Removing temporary image file.")
            os.remove(image_path)


def save_image_temporarily(image_file: FileStorage) -> str:
    """
    Safely saves an uploaded image file to a temporary location with validation.

    Args:
        image_file (FileStorage): The uploaded file object from Flask/Werkzeug

    Returns:
        str: Path to the saved temporary file

    Raises:
        ValueError: If file is invalid or missing
        SecurityError: If file fails security checks
        IOError: If file cannot be saved
    """
    logger = logging.getLogger(__name__)

    try:
        # Validate input
        if not image_file:
            raise ValueError("No image file provided")

        # Validate file type
        if not _is_allowed_file(image_file.filename):
            raise SecurityError(f"Unsupported file type: {image_file.filename}")

        # Create sanitized filename
        secure_filename = _create_secure_filename(image_file.filename)

        # Ensure temp directory exists
        os.makedirs(Config.TEMP_IMAGE_PATH, exist_ok=True)

        # Generate unique path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{secure_filename}"
        image_path = os.path.join(Config.TEMP_IMAGE_PATH, unique_filename)

        # Validate file size before saving
        if _exceeds_size_limit(image_file):
            raise SecurityError("File size exceeds maximum limit")

        # Save file with additional checks
        try:
            image_file.save(image_path)

            # Verify the saved file
            if not os.path.exists(image_path):
                raise IOError("File was not saved successfully")

            # Validate saved file is actually an image
            if not _is_valid_image(image_path):
                os.remove(image_path)
                raise SecurityError("Invalid image file content")

            logger.debug(
                "Image saved successfully",
                extra={
                    "original_filename": image_file.filename,
                    "saved_path": image_path,
                    "file_size": os.path.getsize(image_path),
                },
            )

            return image_path

        except Exception as e:
            # Clean up if save failed
            if os.path.exists(image_path):
                os.remove(image_path)
            raise IOError(f"Failed to save image: {str(e)}")

    except (ValueError, SecurityError, IOError) as e:
        logger.error(str(e), exc_info=True)
        raise
    except Exception as e:
        logger.error("Unexpected error saving image", exc_info=True)
        raise IOError(f"Unexpected error saving image: {str(e)}")


def _is_allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_IMAGE_EXTENSIONS
    )


def _create_secure_filename(filename: str) -> str:
    """Create a secure version of the filename."""
    # Remove any path components
    filename = os.path.basename(filename)
    # Remove special characters and spaces
    filename = re.sub(r"[^a-zA-Z0-9._-]", "", filename)
    return filename


def _exceeds_size_limit(file_obj: FileStorage) -> bool:
    """Check if file size exceeds the configured limit."""
    file_obj.seek(0, os.SEEK_END)
    size = file_obj.tell()
    file_obj.seek(0)  # Reset file pointer
    return size > Config.MAX_IMAGE_SIZE


class SecurityError(Exception):
    """Custom exception for security-related issues."""

    pass


def check_anti_spoofing(image_path: str) -> dict:
    """
    Determine the authenticity of an image using DeepFace's anti-spoofing feature.

    Args:
        image_path (str): The path to the image file to check.

    Returns:
        dict: A structured response containing:
            - is_real (bool): Indicates if the image is real
            - antispoof_score (float|None): Score indicating likelihood of being real
            - confidence (float|None): Confidence level of the detection
            - error (str|None): Error message if unsuccessful

    Raises:
        ValueError: If image_path is invalid
    """
    logger = logging.getLogger(__name__)
    logger.debug("Starting anti-spoofing check")

    if not image_path:
        raise ValueError("Image path cannot be empty")

    response = {
        "is_real": False,
        "antispoof_score": None,
        "confidence": None,
        "error": None,
    }

    try:
        # Validate file exists and is accessible
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Validate file is an image
        if not _is_valid_image(image_path):
            raise ValueError(f"Invalid image file format: {image_path}")

        # Configure anti-spoofing parameters
        detection_params = {
            "img_path": image_path,
            "detector_backend": Models.ANTI_SPOOFING_DETECTOR_BACKEND,
            "enforce_detection": True,
            "align": True,
            "anti_spoofing": True,
        }

        faces = DeepFace.extract_faces(**detection_params)

        if not faces:
            response["error"] = "No face detected in the image"
            logger.warning("Anti-spoofing check failed: no face detected")
            return response

        if len(faces) > 1:
            logger.warning(f"Multiple faces detected in image: {len(faces)}")

        # Process the first detected face
        face_result = faces[0]

        # Extract and validate scores
        antispoof_score = _validate_score(face_result.get("antispoof_score"))
        confidence = _validate_score(face_result.get("confidence"))
        is_real = face_result.get("is_real", False)

        response.update(
            {
                "is_real": is_real,
                "antispoof_score": antispoof_score,
                "confidence": confidence,
            }
        )

        logger.debug(
            "Anti-spoofing check completed",
            extra={
                "is_real": is_real,
                "antispoof_score": antispoof_score,
                "confidence": confidence,
                "faces_detected": len(faces),
            },
        )
        return response

    except FileNotFoundError as e:
        response["error"] = str(e)
        logger.error("File not found error", exc_info=True)
    except ValueError as e:
        response["error"] = str(e)
        logger.error("Validation error", exc_info=True)
    except Exception as e:
        response["error"] = f"Unexpected error in anti-spoofing check: {str(e)}"
        logger.error("Anti-spoofing check failed", exc_info=True)

    return response


def _validate_score(score: float) -> float:
    """
    Validate and normalize the anti-spoofing score.

    Args:
        score (float): Raw score from the model

    Returns:
        float: Normalized score between 0 and 1
    """
    try:
        score = float(score)
        return max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        return 0.0


def _is_valid_image(file_path: str) -> bool:
    """
    Validate if the file is a supported image format.

    Args:
        file_path (str): Path to the image file

    Returns:
        bool: True if valid image file, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def generate_embeddings(image_path: str) -> dict:
    """
    Generate face embeddings for a given image path.

    Args:
        image_path (str): Path to the image file

    Returns:
        dict: Response containing:
            - success (bool): Whether the operation was successful
            - embeddings (list|None): Face embeddings if successful
            - error (str|None): Error message if unsuccessful

    Raises:
        ValueError: If image_path is invalid
    """
    logger = logging.getLogger(__name__)
    logger.debug("Starting face embeddings generation.")

    if not image_path:
        raise ValueError("Image path cannot be empty")

    response = {"success": False, "embeddings": None, "error": None}

    try:
        # Validate file exists and is accessible
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Validate file is an image
        if not _is_valid_image(image_path):
            raise ValueError(f"Invalid image file format: {image_path}")

        embeddings = DeepFace.represent(
            img_path=image_path,
            model_name=Models.FACE_RECOGNITION_MODEL,
            detector_backend=Models.DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )

        if not embeddings:
            response["error"] = "No faces detected in the image"
            logger.warning(response["error"])
            return response

        response.update({"success": True, "embeddings": embeddings})
        logger.debug(
            "Embeddings generated successfully",
            extra={"embeddings_count": len(embeddings)},
        )
        return response

    except FileNotFoundError as e:
        response["error"] = str(e)
        logger.error("File not found error", exc_info=True)
    except ValueError as e:
        response["error"] = str(e)
        logger.error("Validation error", exc_info=True)
    except Exception as e:
        response["error"] = f"Unexpected error generating embeddings: {str(e)}"
        logger.error("Embedding generation failed", exc_info=True)

    return response


def _is_valid_image(file_path: str) -> bool:
    """
    Validate if the file is a supported image format.

    Args:
        file_path (str): Path to the image file

    Returns:
        bool: True if valid image file, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


class PineconeService:
    def __init__(self):
        self.pc = Pinecone(
            api_key=Config.PINECONE_API_KEY
        )  # Use the API key from Config
        self.index_name = "deepface-embeddings"
        self.ensure_index_exists()

    def ensure_index_exists(self):
        """Create index if it doesn't exist"""
        try:
            # Create a serverless index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=512,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
        except Exception as e:
            logging.error(f"Error creating index: {str(e)}")

    def store_embeddings(
        self, id, firstName, lastName, age, gender, email, phone, embeddings
    ):
        """
        Store face embeddings in Pinecone
        Args:
            id (str): Unique identifier for the person
            firstName (str): Person's first name
            lastName (str): Person's last name
            age (str): Person's age
            gender (str): Person's gender
            email (str): Person's email
            phone (str): Person's phone number
            embeddings (list): List of face embedding vectors
        """
        try:
            index = self.pc.Index(self.index_name)

            # Prepare vectors for upsert
            vectors = []
            timestamp = int(time.time())

            for i, embedding in enumerate(embeddings):
                vector_id = f"{firstName.lower().replace(' ', '-')}-{lastName.lower().replace(' ', '-')}-{uuid.uuid4()}"
                vectors.append(
                    {
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
                        },
                    }
                )

            # Upsert vectors in batches of 100
            batch_size = 100
            batches = [
                vectors[i : i + batch_size] for i in range(0, len(vectors), batch_size)
            ]
            for batch in batches:
                index.upsert(vectors=batch)

            logging.info(f"Stored {len(vectors)} embeddings for {firstName} {lastName}")

            return {"success": True, "error": None}
        except Exception as e:
            logging.error(f"Error storing embeddings: {str(e)}")
            return {"success": False, "error": str(e)}


def search_similar_faces(self, query_vector, top_k=2):
    """
    Search for similar faces.
    Args:
        query_vector (list): Face embedding vector to search for.
        top_k (int): Number of results to return.
    Returns:
        dict: A structured response with search results and any messages or errors encountered.
    """
    logging.debug(f"Starting search for similar faces with index: {self.index_name}")
    response = {"match_found": False, "matches": None, "message": None, "error": None}

    try:
        index = self.pc.Index(self.index_name)
        if index is None:
            raise ValueError("Index could not be initialized.")

        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        if not results or "matches" not in results or not results["matches"]:
            response["message"] = "No matching faces found."
            logging.info(response["message"])
            return response

        response.update(
            {
                "match_found": True,
                "matches": results["matches"],
                "message": "Matches found.",
            }
        )
        logging.debug(f"Search results: {response['matches']}")
        return response

    except ValueError as ve:
        response["error"] = str(ve)
        logging.error(f"ValueError: {response['error']}")
        return response

    except Exception as e:
        response["error"] = f"Unexpected error during search: {str(e)}"
        logging.error(response["error"])
        return response


if __name__ == "__main__":
    app.run(debug=True)
