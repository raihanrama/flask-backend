from flask import Flask
from flask_cors import CORS
from .routes import api_blueprint

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for frontend access
    
    # Register API routes
    app.register_blueprint(api_blueprint, url_prefix="/api")
    
    return app