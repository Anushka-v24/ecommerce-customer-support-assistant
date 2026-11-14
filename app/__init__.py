"""Flask application factory."""

from flask import Flask, request

from app.api import api_blueprint
from app.config import Config


def create_app(config_object: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)
    origins = app.config.get("CORS_ALLOWED_ORIGINS", "")
    if isinstance(origins, str):
        configured_origins = [item.strip() for item in origins.split(",") if item.strip()]
    else:
        configured_origins = list(origins or [])
    if "*" in configured_origins:
        raise ValueError("CORS_ALLOWED_ORIGINS must list exact origins; wildcard origins are not supported.")
    app.config["CORS_ALLOWED_ORIGINS"] = configured_origins
    if app.config.get("MAX_CHAT_MESSAGE_LENGTH", 0) < 1 or app.config.get("MAX_SESSION_ID_LENGTH", 0) < 1:
        raise ValueError("Chat input length limits must be positive integers.")
    if app.config.get("MAX_CONTENT_LENGTH", 0) < 1:
        raise ValueError("MAX_CONTENT_LENGTH must be a positive integer.")
    app.register_blueprint(api_blueprint, url_prefix="/api")

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        if origin and origin in app.config["CORS_ALLOWED_ORIGINS"]:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Max-Age"] = "600"
        return response

    return app
