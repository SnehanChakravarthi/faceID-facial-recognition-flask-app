import logging
from flask import Flask
from api import register_routes

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="\033[1;32m%(asctime)s\033[0m - \033[1;34m%(levelname)s\033[0m - %(message)s",
)


def create_face_id_api(config_object=None):
    """Create and configure the Flask application.

    Args:
        config_object: Configuration object or path to config file (optional)

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Load configuration if provided
    if config_object is not None:
        app.config.from_object(config_object)

    # Register routes
    register_routes(app)

    return app


def main():
    """Run the application in development mode."""
    app = create_face_id_api()
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
