import os
import time
from flask import jsonify


def handle_health_check_request(app):
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "environment": {
            "python_env": os.environ.get("PYTHON_ENV", "development"),
            "debug_mode": app.debug,
        },
    }

    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code
