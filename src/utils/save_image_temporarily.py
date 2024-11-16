import logging
import os
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Union

from werkzeug.datastructures import FileStorage

from ..config import Config


# Constants
class ImageProcessingCodes:
    SUCCESS = 0
    FUNCTION_ERROR = 1
    INVALID_FILE_TYPE = 2
    INVALID_EXTENSION = 3
    FILE_SIZE_ERROR = 4
    SAVE_ERROR = 5


class ImageProcessingMessages:
    SUCCESS = "Image saved successfully"
    INVALID_FILE_TYPE = "Invalid file type - only images allowed"
    INVALID_EXTENSION = (
        "Unsupported file extension - only .png, .jpg, .jpeg, .webp allowed"
    )
    FILE_SIZE_ERROR = "File size exceeds maximum limit of 10MB"
    SAVE_ERROR = "Failed to save image file"


class FileConstraints:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


# Custom Exceptions
class ImageProcessingError(Exception):
    """Base exception for image processing errors"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ImageValidationError(ImageProcessingError):
    """Raised when image validation fails"""

    pass


class ImageSaveError(ImageProcessingError):
    """Raised when image saving fails"""

    pass


# Response Class
@dataclass
class ImageResponse:
    code: int
    path: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[int, str, None]]:
        return {"code": self.code, "path": self.path, "message": self.message}


# Validators
def validate_config() -> None:
    """Validate required configuration settings"""
    if not hasattr(Config, "TEMP_IMAGE_PATH"):
        raise ValueError("TEMP_IMAGE_PATH not configured")
    if not Config.TEMP_IMAGE_PATH:
        raise ValueError("TEMP_IMAGE_PATH cannot be empty")


def validate_file_size(size: int) -> None:
    """Validate file size against maximum limit"""
    if size > FileConstraints.MAX_FILE_SIZE:
        raise ImageValidationError(
            ImageProcessingCodes.FILE_SIZE_ERROR,
            ImageProcessingMessages.FILE_SIZE_ERROR,
        )


def save_image_temporarily(image_file: FileStorage) -> Dict[str, Union[int, str, None]]:
    """
     Save an uploaded image file to a temporary location with security measures.

    This function performs several validation steps before saving:
    - Validates file type (must be an image)
    - Validates file extension (.png, .jpg, .jpeg, .webp)
    - Validates file size (max 10MB)
    - Generates a secure UUID-based filename
    - Ensures the temporary directory exists

    Args:
        image_file (FileStorage): The uploaded file object from Flask's request.files

    Returns:
        Dict[str, Union[int, str, None]]: A dictionary containing:
            - code (int): Status code indicating the result
                0: Success
                1: Function error
                2: Invalid file type
                3: Invalid extension
                4: File size error
                5: Save error
            - path (str, optional): Path to the saved temporary file if successful
            - message (str, optional): Success or error message describing the result

    Raises:
        ImageValidationError: When file validation fails (type, extension, size)
        ImageSaveError: When file saving operations fail
        ValueError: When configuration is invalid or missing
    """
    try:
        # Validate configuration
        validate_config()

        # Validate file type
        if not image_file.content_type.startswith("image/"):
            raise ImageValidationError(
                ImageProcessingCodes.INVALID_FILE_TYPE,
                f"{ImageProcessingMessages.INVALID_FILE_TYPE}: {image_file.content_type}",
            )

        # Validate file extension
        file_extension = os.path.splitext(image_file.filename)[1].lower()
        if file_extension not in FileConstraints.ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                ImageProcessingCodes.INVALID_EXTENSION,
                f"{ImageProcessingMessages.INVALID_EXTENSION}: {file_extension}",
            )

        # Check file size
        image_file.seek(0, os.SEEK_END)
        size = image_file.tell()
        validate_file_size(size)
        image_file.seek(0)

        # Generate secure filename and save
        secure_filename = f"{uuid.uuid4()}{file_extension}"
        os.makedirs(Config.TEMP_IMAGE_PATH, exist_ok=True)
        image_path = os.path.join(Config.TEMP_IMAGE_PATH, secure_filename)

        try:
            image_file.save(image_path)
        except Exception as e:
            raise ImageSaveError(
                ImageProcessingCodes.SAVE_ERROR,
                f"{ImageProcessingMessages.SAVE_ERROR}: {str(e)}",
            )

        # Verify file was saved
        if not os.path.exists(image_path):
            raise ImageSaveError(
                ImageProcessingCodes.SAVE_ERROR, ImageProcessingMessages.SAVE_ERROR
            )

        # Create successful response
        response = ImageResponse(
            code=ImageProcessingCodes.SUCCESS,
            path=image_path,
            message=ImageProcessingMessages.SUCCESS,
        )

        logging.info(
            "Image saved successfully: path=%s, size=%d bytes", image_path, size
        )

        return response.to_dict()

    except ImageProcessingError as e:
        logging.warning(
            "Image processing failed: code=%d, message=%s", e.code, e.message
        )
        return ImageResponse(code=e.code, message=e.message).to_dict()

    except Exception as e:
        error_msg = f"Unexpected error while saving image: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return ImageResponse(
            code=ImageProcessingCodes.FUNCTION_ERROR, message=error_msg
        ).to_dict()
