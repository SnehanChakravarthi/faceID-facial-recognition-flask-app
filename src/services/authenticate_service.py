import logging
import time
from concurrent.futures import ThreadPoolExecutor
from flask import jsonify
from ..config import Settings
from ..utils import process_face_image
from ..utils import PineconeService


def handle_authenticate_request(request):
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
