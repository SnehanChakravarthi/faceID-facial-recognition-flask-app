import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from flask import jsonify
from ..utils import process_face_image
from ..utils import PineconeService


def handle_enrollment_request(request):
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
