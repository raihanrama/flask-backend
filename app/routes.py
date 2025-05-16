from flask import Blueprint, request, jsonify, Response, stream_with_context
from .models import knn, llm_generate_description, scaler, llm_generate_recipe
from .utils import calculate_bmi, get_nutrition_needs
import pandas as pd
import numpy as np
import os
import logging
import json
import requests

# Set up
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API blueprint
api_blueprint = Blueprint("api", __name__)

# Load dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(base_dir, "data", "nutrition.csv")

# Try to load the dataset
try:
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Successfully loaded dataset from {DATA_PATH}")
except FileNotFoundError:
    logger.warning(f"Dataset not found at {DATA_PATH}. Creating empty dataset for testing.")
    df = pd.DataFrame(columns=["name", "calories", "proteins", "fat", "carbohydrate", "meal_time", "image"])

#End Point Nutirisi
@api_blueprint.route("/recommend/nutrition", methods=["POST"])
def recommend_by_nutrition():
    try:
        data = request.json
        logger.info(f"Received nutrition recommendation request: {data}")

        required_fields = ['calories', 'proteins', 'fat', 'carbohydrate', 'meal_time']
        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing required field: {field}"}), 400

        meal_time = data['meal_time']
        if meal_time not in ["Pagi", "Siang", "Malam"]:
            return jsonify({"status": "error", "message": "Invalid meal_time. Choose from 'Pagi', 'Siang', or 'Malam'"}), 400

        user_input = np.array([[data['calories'], data['proteins'], data['fat'], data['carbohydrate']]])
        user_input_scaled = scaler.transform(user_input)
        _, indices = knn.kneighbors(user_input_scaled, n_neighbors=10)

        recommendations = df.iloc[indices[0]]
        recommendations = recommendations[recommendations["meal_time"] == meal_time]

        if recommendations.empty:
            recommendations = df[df["meal_time"] == meal_time].sample(n=5, replace=True)

        recommendations = recommendations[["name", "calories", "proteins", "fat", "carbohydrate", "image"]].to_dict(orient="records")

        return jsonify({
            "status": "success",
            "recommendations": recommendations,
            "request_data": {
                "meal_time": meal_time
            }
        })

    except Exception as e:
        logger.error(f"Error in nutrition recommendation: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@api_blueprint.route("/recommend/bmi", methods=["POST"])
def recommend_by_bmi():
    try:
        data = request.json
        logger.info(f"Received BMI recommendation request: {data}")

        required_fields = ['weight', 'height', 'meal_time']
        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing required field: {field}"}), 400

        weight, height, meal_time = data['weight'], data['height'], data['meal_time']

        if weight <= 0 or height <= 0:
            return jsonify({"status": "error", "message": "Weight and height must be positive values"}), 400

        if meal_time not in ["Pagi", "Siang", "Malam"]:
            return jsonify({"status": "error", "message": "Invalid meal_time. Choose from 'Pagi', 'Siang', or 'Malam'"}), 400

        bmi = calculate_bmi(weight, height)
        nutrition_needs = get_nutrition_needs(bmi)

        user_input = np.array([[nutrition_needs["calories"], nutrition_needs["proteins"], nutrition_needs["fat"], nutrition_needs["carbohydrate"]]])
        user_input_scaled = scaler.transform(user_input)
        _, indices = knn.kneighbors(user_input_scaled, n_neighbors=10)

        recommendations = df.iloc[indices[0]]
        recommendations = recommendations[recommendations["meal_time"] == meal_time]

        if recommendations.empty:
            recommendations = df[df["meal_time"] == meal_time].sample(n=5, replace=True)

        recommendations = recommendations[["name", "calories", "proteins", "fat", "carbohydrate", "image"]].to_dict(orient="records")

        return jsonify({
            "status": "success",
            "bmi": round(bmi, 2),
            "bmi_category": get_bmi_category(bmi),
            "nutrition_needs": nutrition_needs,
            "recommendations": recommendations,
            "request_data": {
                "meal_time": meal_time
            }
        })
    except Exception as e:
        logger.error(f"Error in BMI recommendation: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obesity"

@api_blueprint.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "data_loaded": len(df) > 0
    })

@api_blueprint.route("/foods", methods=["GET"])
def get_all_foods():
    try:
        if df.empty:
            return jsonify({"status": "error", "message": "No food data available"}), 404

        foods = df.to_dict(orient="records")
        return jsonify({
            "status": "success",
            "foods": foods
        })
    except Exception as e:
        logger.error(f"Error fetching food data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@api_blueprint.route("/get_recipe", methods=["POST"])
def get_recipe():
    try:
        data = request.json
        if "food_name" not in data:
            return jsonify({"status": "error", "message": "Missing required field: food_name"}), 400

        food_name = data["food_name"]
        recipe = llm_generate_recipe(food_name)

        return jsonify({
            "status": "success",
            "food_name": food_name,
            "recipe": recipe
        })
    except Exception as e:
        logger.error(f"Error in get_recipe endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@api_blueprint.route("/get_description", methods=["POST"])
def get_description():
    try:
        data = request.json
        if "food_name" not in data:
            return jsonify({"status": "error", "message": "Missing required field: food_name"}), 400

        food_name = data["food_name"]
        description = llm_generate_description(food_name)

        return jsonify({
            "status": "success",
            "food_name": food_name,
            "description": description
        })
    except Exception as e:
        logger.error(f"Error in get_description endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_blueprint.route("/chat_stream", methods=["GET"])
def chat_stream():
    try:
        question = request.args.get("question", "")
        history = json.loads(request.args.get("history", "[]"))

        def generate():
            prompt = build_chat_prompt(history, question)
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": True},
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    json_data = json.loads(line.decode('utf-8'))
                    token = json_data.get("response", "")
                    yield f"data: {token}\n\n"
                    
            yield f"data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Error in chat_stream: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

def build_chat_prompt(history, question):
    prompt = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Asisten"
        prompt += f"{role}: {msg['content']}\n\n"
    
    # Improved prompt with specific styling guidance
    prompt += f"""User: {question}

Asisten: 

Saat merespons, gunakan format Markdown agar bisa dipahami oleh react-markdown saya yang sesuai dengan panduan berikut:

"""
    return prompt