import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from ..config import Config, Settings


class SearchCodes:
    SUCCESS = 0
    VALIDATION_ERROR = 1
    NO_MATCHES = 5
    BELOW_THRESHOLD = 6
    UNEXPECTED_ERROR = 7


class StoreEmbeddingsCodes:
    SUCCESS = 0
    VALIDATION_ERROR = 1
    UNEXPECTED_ERROR = 3


class SearchMessages:
    SUCCESS = "Match found successfully"
    NO_MATCHES = "No matches found in database"
    BELOW_THRESHOLD = "No matches found above threshold"
    VALIDATION_ERROR = "Invalid input parameters"
    UNEXPECTED_ERROR = "Unexpected error during search"


class StoreEmbeddingsMessages:
    SUCCESS = "Embeddings stored successfully"
    VALIDATION_ERROR = "Invalid input parameters"
    UNEXPECTED_ERROR = "Unexpected error during embedding storage"


@dataclass
class SearchResponse:
    code: int
    match: Optional[dict] = None
    message: Optional[str] = None
    details: Optional[dict] = None

    def to_dict(self) -> Dict[str, Union[int, dict, str, None]]:
        return {
            "code": self.code,
            "match": self.match,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class StoreEmbeddingsResponse:
    code: int
    message: Optional[str] = None
    vectors_stored: Optional[int] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Union[int, str, int, float, None]]:
        return {
            "code": self.code,
            "message": self.message,
            "vectors_stored": self.vectors_stored,
            "processing_time": self.processing_time,
        }


class PineconeService:
    """
    Service class for managing face embeddings storage in Pinecone.

    Attributes:
        pc: Pinecone client instance
        index_name (str): Name of the Pinecone index
        dimension (int): Dimension of face embeddings
        batch_size (int): Size of batches for vector upsert operations
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, api_key=None, pc_instance=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key=None, pc_instance=None):
        if self._initialized:
            return

        self.dimension = 512  # ArcFace embedding dimension
        self.batch_size = 100
        self.index_name = "deepface-embeddings"

        if pc_instance:
            self.pc = pc_instance
        else:
            if not api_key and not Config.PINECONE_API_KEY:
                raise ValueError("Pinecone API key is required")

            try:
                self.pc = Pinecone(api_key=api_key or Config.PINECONE_API_KEY)
                self.ensure_index_exists()
            except Exception as e:
                logging.error("Failed to initialize Pinecone client: %s", str(e))
                raise

        self._initialized = True

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
        vectors_stored = 0

        try:
            # Input validation
            if not all([id, firstName, lastName]):
                return StoreEmbeddingsResponse(
                    code=StoreEmbeddingsCodes.VALIDATION_ERROR,
                    message=f"{StoreEmbeddingsMessages.VALIDATION_ERROR}: ID, first name, and last name are required",
                ).to_dict()

            if not embeddings or not isinstance(embeddings, list):
                return StoreEmbeddingsResponse(
                    code=StoreEmbeddingsCodes.VALIDATION_ERROR,
                    message=f"{StoreEmbeddingsMessages.VALIDATION_ERROR}: Valid embeddings list is required",
                ).to_dict()

            index = self.pc.Index(self.index_name)
            vectors = []
            timestamp = int(time.time())

            # Prepare vectors for upsert
            for i, embedding in enumerate(embeddings):
                # Validate embedding dimension
                if len(embedding) != self.dimension:
                    return StoreEmbeddingsResponse(
                        code=StoreEmbeddingsCodes.VALIDATION_ERROR,
                        message=f"{StoreEmbeddingsMessages.VALIDATION_ERROR}: Invalid embedding dimension: expected {self.dimension}, got {len(embedding)}",
                    ).to_dict()

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

            return StoreEmbeddingsResponse(
                code=StoreEmbeddingsCodes.SUCCESS,
                message=success_message,
                vectors_stored=vectors_stored,
                processing_time=round(processing_time, 3),
            ).to_dict()

        except Exception as e:
            error_msg = f"{StoreEmbeddingsMessages.UNEXPECTED_ERROR}: {str(e)}"
            logging.exception(error_msg)
            return StoreEmbeddingsResponse(
                code=StoreEmbeddingsCodes.UNEXPECTED_ERROR,
                message=error_msg,
                vectors_stored=vectors_stored,
            ).to_dict()

    def search_similar_faces(self, query_vector, threshold=Settings.MATCH_THRESHOLD):
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
        details = {
            "similarity_score": None,
            "threshold_used": threshold,
            "processing_time": None,
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

            # Update processing time
            details["processing_time"] = round(time.time() - start_time, 3)

            if not search_result or not search_result["matches"]:
                logging.warning("No matches found in search results")
                return SearchResponse(
                    code=SearchCodes.NO_MATCHES,
                    message=SearchMessages.NO_MATCHES,
                    details=details,
                ).to_dict()

            # Process the single match
            best_match = search_result["matches"][0]
            similarity_score = round(best_match.get("score", 0), 4)
            details["similarity_score"] = similarity_score

            # Check if match meets threshold
            if similarity_score < threshold:
                logging.info(
                    "Best match (score=%.4f) below threshold (%.2f)",
                    similarity_score,
                    threshold,
                )
                return SearchResponse(
                    code=SearchCodes.BELOW_THRESHOLD,
                    message=f"{SearchMessages.BELOW_THRESHOLD} ({threshold})",
                    details=details,
                ).to_dict()

            # Format successful match
            match_data = {
                "id": best_match.get("id"),
                "score": similarity_score,
                "metadata": best_match.get("metadata", {}),
            }

            logging.info(
                f"Face search completed - match_found=True, similarity={similarity_score}, time={details['processing_time']}s"
            )

            return SearchResponse(
                code=SearchCodes.SUCCESS,
                match=match_data,
                message=SearchMessages.SUCCESS,
                details=details,
            ).to_dict()

        except ValueError as ve:
            error_msg = f"{SearchMessages.VALIDATION_ERROR}: {str(ve)}"
            logging.error(error_msg)
            return SearchResponse(
                code=SearchCodes.VALIDATION_ERROR,
                message=error_msg,
                details=details,
            ).to_dict()

        except Exception as e:
            error_msg = f"{SearchMessages.UNEXPECTED_ERROR}: {str(e)}"
            logging.exception(error_msg)
            return SearchResponse(
                code=SearchCodes.UNEXPECTED_ERROR, message=error_msg, details=details
            ).to_dict()
