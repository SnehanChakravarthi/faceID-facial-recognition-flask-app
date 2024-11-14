import logging
import os
from deepface import DeepFace
from flask import Flask, jsonify, request
from pinecone import Pinecone
from pinecone import ServerlessSpec
import uuid
import time

# from dotenv import load_dotenv
# load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    TEMP_IMAGE_PATH = "/tmp"
    # MAX_CONTENT_LENGTH = 16 * 1024 * 1024
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
    DETECTOR_BACKEND = DETECTOR_BACKENDS[1]  # "mtcnn"
    ANTI_SPOOFING_DETECTOR_BACKEND = DETECTOR_BACKENDS[0]  # "opencv"


app = Flask(__name__)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": int(time.time())}), 200


@app.route("/api/v1/enroll", methods=["POST"])
def enroll():
    # Log the headers for debugging
    form_data = request.form
    logging.debug("Received form data: %s", form_data)

    image_file = request.files.get("image")
    logging.debug("Received image file: %s", image_file)

    # Basic validation
    if not form_data.get("firstName") or not form_data.get("lastName"):
        return jsonify({"error": "First name and last name are required"}), 400

    if not image_file or not image_file.filename:
        return jsonify({"error": "Image file is required"}), 400

    # Save the image temporarily
    image_path = save_image_temporarily(image_file)

    try:
        # Call the separate function to handle enrollment logic
        response, status_code = handle_enrollment(form_data, image_path)
        return response, status_code
    except Exception as e:
        logging.exception("Unexpected error during enrollment")
        return (
            jsonify({"error": "An unexpected error occurred during enrollment"}),
            500,
        )


@app.route("/api/v1/authenticate", methods=["POST"])
def authenticate():
    # Log the headers for debugging

    image_file = request.files.get("image")
    logging.debug("Received image file: %s", image_file)

    if not image_file or not image_file.filename:
        return jsonify({"error": "Image file is required"}), 400

    # Save the image temporarily
    image_path = save_image_temporarily(image_file)

    try:
        response = handle_authentication(image_path)

        if response.is_json:
            data = response.get_json()

        # If authentication was successful
        if data.get("success"):
            return (
                jsonify(
                    {
                        "success": True,
                        "data": data,  # Return the complete data from Pinecone
                        "message": "Authentication successful",
                    }
                ),
                200,
            )

        # If authentication failed
        return (
            jsonify(
                {
                    "success": False,
                    "error": data.get("error", "No matching faces found"),
                    "message": "Authentication failed",
                }
            ),
            401,
        )

    except Exception as e:
        logging.exception("Unexpected error during authentication")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "message": "An unexpected error occurred",
                }
            ),
            500,
        )


def save_image_temporarily(image_file):
    try:
        image_path = os.path.join(Config.TEMP_IMAGE_PATH, image_file.filename)
        image_file.save(image_path)
        return image_path
    except Exception as e:
        logging.error(f"Failed to save image temporarily: {str(e)}")
        raise


def handle_enrollment(form_data, image_path):
    try:
        logging.debug("Starting enrollment process.")

        # Generate face embeddings with anti-spoofing
        embeddings = generate_embeddings(image_path)

        # Initialize Pinecone service
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
        logging.exception("Error during enrollment.")
        return jsonify({"error": "Enrollment failed"}), 500

    finally:
        # Clean up the temporary image file
        if os.path.exists(image_path):
            logging.debug("Removing temporary image file.")
            os.remove(image_path)


def handle_authentication(image_path):
    try:
        logging.debug("Starting authentication process.")

        spoofing_result = check_anti_spoofing(image_path)

        # Generate face embeddings with anti-spoofing
        embeddings = generate_embeddings(image_path)

        # Initialize PineconeService
        pinecone_service = PineconeService()

        # Assuming embeddings is a list and you want to search using the first embedding
        if embeddings:
            query_vector = embeddings[0]["embedding"]

            # Perform vector search

            search_results = pinecone_service.search_similar_faces(
                query_vector, top_k=2
            )

            search_results_dict = search_results.to_dict()

            if "matches" in search_results_dict and search_results_dict["matches"]:
                response = {
                    "success": True,
                    **search_results_dict,
                    "is_real": spoofing_result.get("is_real", False),
                    "antispoof_score": spoofing_result.get("antispoof_score", 0.0),
                    "confidence": spoofing_result.get("confidence", 0.0),
                }
                logging.debug(f"Authentication response: {response}")
                return jsonify(response)

            # If no matches or invalid results
            return jsonify(
                {
                    "success": False,
                    "error": "No matching faces found or invalid search results",
                    "is_real": spoofing_result.get("is_real", False),
                    "antispoof_score": spoofing_result.get("antispoof_score", 0.0),
                    "confidence": spoofing_result.get("confidence", 0.0),
                }
            )

        else:
            logging.error("No embeddings generated.")
            return jsonify({"success": False, "error": "No embeddings generated."})

    except Exception as e:
        logging.error(f"Error during authentication: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


def check_anti_spoofing(image_path):
    """
    Check if an image is real or fake using DeepFace's anti-spoofing detection.
    Returns a structured response with the result and any errors encountered.
    """
    logging.debug("Starting anti-spoofing check.")
    response = {
        "success": False,
        "is_real": None,
        "antispoof_score": None,
        "confidence": None,
        "error": None,
    }

    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=Models.ANTI_SPOOFING_DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
            anti_spoofing=True,
        )

        if not faces:
            response["error"] = "No face detected in the image"
            logging.warning(response["error"])
            return response

        face_result = faces[0]
        response.update(
            {
                "success": True,
                "is_real": face_result.get("is_real", False),
                "antispoof_score": face_result.get("antispoof_score", 0.0),
                "confidence": face_result.get("confidence", 0.0),
            }
        )
        logging.debug(f"Anti-spoofing result: {response}")
        return response

    except Exception as e:
        response["error"] = f"Error in anti-spoofing check: {str(e)}"
        logging.error(response["error"])
        return response


def generate_embeddings(image_path):
    """
    Generate face embeddings for a given image path.
    """
    logging.debug("Generating face embeddings.")
    embeddings = DeepFace.represent(
        img_path=image_path,
        model_name=Models.FACE_RECOGNITION_MODEL,
        detector_backend=Models.DETECTOR_BACKEND,
        enforce_detection=False,
        align=True,
    )
    logging.debug(f"Embeddings generated: {embeddings}")
    return embeddings


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
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                index.upsert(vectors=batch)

            return {
                "success": True,
                "message": f"Stored {len(vectors)} embeddings for {firstName} {lastName}",
                "vectors_stored": len(vectors),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_similar_faces(self, query_vector, top_k=2):
        """
        Search for similar faces
        Args:
            query_vector (list): Face embedding vector to search for
            top_k (int): Number of results to return
        """
        try:
            logging.debug(f"Initializing index with name: {self.index_name}")
            index = self.pc.Index(self.index_name)
            if index is None:
                raise ValueError("Index could not be initialized.")

            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )

            if not results or "matches" not in results:
                logging.warning("No matches found or invalid results structure.")
                return {
                    "success": False,
                    "error": "No matches found or invalid results structure.",
                }

            return {"success": True, "matches": results["matches"]}

        except ValueError as ve:
            logging.error(f"ValueError: {str(ve)}")
            return {"success": False, "error": str(ve)}

        except Exception as e:
            logging.error(f"Unexpected error during search: {str(e)}")
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    app.run(debug=True)
