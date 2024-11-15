import logging
import os
import time
from .save_image_temporarily import save_image_temporarily
from .check_anti_spoofing import check_anti_spoofing
from .generate_embeddings import generate_embeddings


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
