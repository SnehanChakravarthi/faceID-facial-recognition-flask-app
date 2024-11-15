import logging
import os
import uuid
from ..config import Config


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
