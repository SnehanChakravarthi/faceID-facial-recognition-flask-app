import os


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
    FACE_DETECTION_MODELS = [
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
    ANTI_SPOOFING_MODEL = ["Fasnet"]


class Config:
    TEMP_IMAGE_PATH = "/tmp"
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


class Settings:
    MATCH_THRESHOLD = 0.8
    FACE_RECOGNITION_MODEL = Models.FACE_RECOGNITION_MODELS[7]  # "ArcFace"
    DETECTOR_BACKEND = Models.FACE_DETECTION_MODELS[0]  # "retinaface"
    ANTI_SPOOFING_DETECTOR_BACKEND = Models.FACE_DETECTION_MODELS[0]  # "retinaface"
