import joblib
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Call Model
model_path = os.path.join(os.path.dirname(base_dir), "models", "knn_model.pkl")
scaler_path = os.path.join(os.path.dirname(base_dir), "models", "scaler.pkl")

# Load Model & Scaler
try:
    knn = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print(f"Looking for models in: {model_path}")


# Fungsi untuk menghasilkan resep menggunakan LLM lokal (Ollama)
def llm_generate_recipe(food_name):
    try:
        prompt = f"Berikan resep lengkap untuk makanan {food_name}. Sertakan bahan dan langkah-langkah memasak. Gunakan bahasa indonesia dan jangan terlalu panjang, dan berikan jawaban yang interaktif seperti menggunakan icon yang menarik"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3", 
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Resep tidak ditemukan.")
        else:
            print(f"Error from Ollama API: {response.status_code} - {response.text}")
            return "Maaf, tidak dapat menghasilkan resep saat ini."

    except Exception as e:
        print(f"Error generating recipe with Ollama: {e}")
        return "Maaf, terjadi kesalahan saat menghasilkan resep."

# Fungsi untuk menghasilkan deskripsi makanan menggunakan LLM lokal (Ollama)
def llm_generate_description(food_name):
    try:
        prompt = (
            f"Berikan deskripsi singkat dan menarik tentang makanan atau bahan baku atau daging bernama {food_name}. "
            f"Ceritakan dalam 2 paragraf, gunakan bahasa Indonesia yang ringan serta informatif. "
            f"Tambahkan emoji agar lebih menarik untuk dibaca."
        )

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Deskripsi tidak tersedia.")
        else:
            print(f"Error from Ollama API: {response.status_code} - {response.text}")
            return "Maaf, tidak dapat menghasilkan deskripsi saat ini."

    except Exception as e:
        print(f"Error generating description with Ollama: {e}")
        return "Maaf, terjadi kesalahan saat menghasilkan deskripsi."
