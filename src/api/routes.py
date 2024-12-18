import threading

from flask import request

from ..services import (
    handle_authenticate_request,
    handle_enrollment_request,
    handle_health_check_request,
    handle_root_request,
    handle_authenticate_request_v2,
    handle_enrollment_request_v2,
)
from ..utils.pinecone_functions import PineconeService


def register_routes(app):
    pinecone_service = PineconeService()

    @app.route("/", methods=["GET"])
    def root():
        return handle_root_request()

    @app.route("/health", methods=["GET"])
    def health_check():
        return handle_health_check_request(app)

    @app.route("/api/v1/enroll", methods=["POST"])
    def enroll():
        return handle_enrollment_request(request, pinecone_service)

    @app.route("/api/v2/enroll", methods=["POST"])
    def enroll_v2():
        return handle_enrollment_request_v2(request, pinecone_service)

    @app.route("/api/v1/authenticate", methods=["POST"])
    def authenticate():
        return handle_authenticate_request(request, pinecone_service)

    @app.route("/api/v2/authenticate", methods=["POST"])
    def authenticate_v2():
        return handle_authenticate_request_v2(request, pinecone_service)
