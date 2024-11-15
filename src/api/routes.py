from ..services import (
    handle_root_request,
    handle_health_check_request,
    handle_enrollment_request,
    handle_authenticate_request,
)
from flask import request


def register_routes(app):
    @app.route("/", methods=["GET"])
    def root():
        return handle_root_request()

    @app.route("/health", methods=["GET"])
    def health_check():
        return handle_health_check_request(app)

    @app.route("/api/v1/enroll", methods=["POST"])
    def enroll():
        return handle_enrollment_request(request)

    @app.route("/api/v1/authenticate", methods=["POST"])
    def authenticate():
        return handle_authenticate_request(request)
