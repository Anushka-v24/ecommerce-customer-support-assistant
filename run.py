"""Run the local Flask development server with ``python run.py``."""

from app import create_app
from app.config import Config
from app.graph.workflow import close_graph_resources


application = create_app()


if __name__ == "__main__":
    try:
        application.run(
            host=Config.APP_HOST,
            port=Config.APP_PORT,
            debug=Config.APP_DEBUG,
        )
    finally:
        close_graph_resources()
