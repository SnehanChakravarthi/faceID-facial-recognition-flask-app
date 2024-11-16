import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Union

from werkzeug.datastructures import FileStorage

from .generate_embeddings_v2 import generate_embeddings_v2
from .save_image_temporarily import save_image_temporarily


class ProcessFaceImageCodesV2:
    SUCCESS = 0
    FUNCTION_ERROR = 1
    NO_FACE_DETECTED = 2
    SPOOFING_DETECTED = 3
    MULTIPLE_FACES_DETECTED = 4


class ProcessFaceImageMessagesV2:
    SUCCESS = "Image processing completed successfully"
    FUNCTION_ERROR = "Unexpected error in image processing"
    NO_FACE_DETECTED = "No face detected in image"
    SPOOFING_DETECTED = "Spoofing detected - only live people allowed"
    MULTIPLE_FACES_DETECTED = "Multiple faces detected - only one face allowed"


class ProcessFaceImageErrorV2(Exception):
    """Base exception for image processing errors"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ProcessFaceImageValidationErrorV2(ProcessFaceImageErrorV2):
    """Raised when validation fails"""

    pass


@dataclass
class ProcessFaceImageResponseV2:
    code: int
    message: Optional[str] = None
    anti_spoofing: Optional[dict] = None
    face_detected: Optional[bool] = None
    embeddings: Optional[list] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Union[int, str, dict, list, float, None]]:
        return {
            "code": self.code,
            "message": self.message,
            "anti_spoofing": self.anti_spoofing,
            "face_detected": self.face_detected,
            "embeddings": self.embeddings,
            "processing_time": self.processing_time,
        }


def process_face_image_v2(
    image_file: FileStorage,
) -> Dict[str, Union[int, str, dict, list, float, None]]:
    """
    Execute a series of operations on an uploaded face image:
    1. Temporarily store the image
    2. Conduct an anti-spoofing verification
    3. Create facial embeddings

    Parameters:
        image_file: A FileStorage instance from Flask's request.files

    Returns:
        dict: A structured response containing:
            - code (int): Status code indicating the result of processing
            - message (str): Description of the processing result
            - anti_spoofing (dict|None): Results from anti-spoofing verification including:
                - is_real (bool): Whether the image is of a real person
                - antispoof_score (float): Anti-spoofing confidence score
                - confidence (float): Overall confidence in the assessment
            - embeddings (list|None): The generated face embeddings if successful
            - processing_time (float): Total processing time in seconds

    Raises:
        ProcessFaceImageError: Base exception for image processing errors
        ProcessFaceImageValidationError: Raised when validation fails
    """
    start_time = time.time()
    temp_image_path = None

    try:
        # Step 1: Save image temporarily
        logging.info("Starting image processing pipeline")
        save_result = save_image_temporarily(image_file)

        if save_result["code"] != 0:  # Any non-success code
            raise ProcessFaceImageErrorV2(
                ProcessFaceImageCodesV2.FUNCTION_ERROR,
                save_result["message"],
            )

        temp_image_path = save_result["path"]
        logging.debug("Image saved temporarily at: %s", temp_image_path)

        # # Step 2: Perform anti-spoofing check
        # logging.debug("Initiating anti-spoofing check")
        # anti_spoofing_result = check_anti_spoofing(temp_image_path)

        # spoof_code = anti_spoofing_result.get("code")
        # if spoof_code != 0:  # Any non-success code
        #     # For spoofing detection (code 4), include the anti-spoofing details
        #     anti_spoofing_details = {
        #         "is_real": anti_spoofing_result.get("is_real"),
        #         "antispoof_score": anti_spoofing_result.get("antispoof_score"),
        #         "confidence": anti_spoofing_result.get("confidence"),
        #     }

        #     return ProcessFaceImageResponseV2(
        #         code=spoof_code,
        #         message=anti_spoofing_result.get("message"),
        #         anti_spoofing=anti_spoofing_details,
        #         processing_time=round(time.time() - start_time, 3),
        #     ).to_dict()

        # # Continue with the rest of the processing if anti-spoofing check passed
        # logging.debug("Anti-spoofing check passed successfully")

        # Step 3: Generate embeddings
        logging.debug("Generating face embeddings")
        embedding_result = generate_embeddings_v2(temp_image_path)

        # Get the embedding code
        embedding_code = embedding_result.get("code")

        # Handle specific cases based on embedding_code
        if embedding_code == 2:  # NO_FACE_DETECTED
            return ProcessFaceImageResponseV2(
                code=ProcessFaceImageCodesV2.NO_FACE_DETECTED,
                message=ProcessFaceImageMessagesV2.NO_FACE_DETECTED,
                anti_spoofing=None,
                face_detected=False,
                embeddings=None,
                processing_time=round(time.time() - start_time, 3),
            ).to_dict()

        elif embedding_code == 3:  # SPOOF_DETECTED
            return ProcessFaceImageResponseV2(
                code=ProcessFaceImageCodesV2.SPOOFING_DETECTED,
                message=ProcessFaceImageMessagesV2.SPOOFING_DETECTED,
                anti_spoofing={
                    "is_real": False,
                    "antispoof_score": None,
                    "confidence": None,
                },
                face_detected=True,
                embeddings=None,
                processing_time=round(time.time() - start_time, 3),
            ).to_dict()

        elif embedding_code != 0:  # Other errors
            raise ProcessFaceImageValidationErrorV2(
                embedding_code,
                embedding_result.get(
                    "message", ProcessFaceImageMessagesV2.FUNCTION_ERROR
                ),
            )

        # Success case
        return ProcessFaceImageResponseV2(
            code=ProcessFaceImageCodesV2.SUCCESS,
            message=ProcessFaceImageMessagesV2.SUCCESS,
            anti_spoofing={
                "is_real": embedding_result.get("is_real"),
                "antispoof_score": None,
                "confidence": None,
            },
            face_detected=embedding_result.get("face_detected"),
            embeddings=embedding_result.get("embeddings"),
            processing_time=round(time.time() - start_time, 3),
        ).to_dict()

    except ProcessFaceImageErrorV2 as e:
        # Simply return the error response without the unnecessary attribute check
        return ProcessFaceImageResponseV2(
            code=e.code,
            message=e.message,
            anti_spoofing=None,
            face_detected=None,
            embeddings=None,
            processing_time=round(time.time() - start_time, 3),
        ).to_dict()

    except Exception as e:
        logging.exception("Unexpected error in process_face_image")
        return ProcessFaceImageResponseV2(
            code=ProcessFaceImageCodesV2.FUNCTION_ERROR,
            message=f"{ProcessFaceImageMessagesV2.FUNCTION_ERROR}: {str(e)}",
            anti_spoofing=None,
            face_detected=None,
            embeddings=None,
            processing_time=round(time.time() - start_time, 3),
        ).to_dict()

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