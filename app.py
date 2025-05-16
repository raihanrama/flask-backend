from flask import Flask
from flask_cors import CORS
from app import create_app
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Create Flask app
app = create_app()

# Optional debug print (untuk development, bukan production)
print("Loaded key:", os.getenv("OPENAI_API_KEY"))

# Run only if local
if __name__ == '__main__':
    app.run(debug=True, port=5000)
