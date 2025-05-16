def calculate_bmi(weight, height):
    """Menghitung BMI berdasarkan berat (kg) dan tinggi (meter)."""
    height_m = height / 100  # Konversi cm ke meter
    return weight / (height_m ** 2)

def get_nutrition_needs(bmi):
    """Menentukan kebutuhan nutrisi berdasarkan kategori BMI."""
    if bmi < 18.5:
        return {"calories": 500, "proteins": 20, "fat": 15, "carbohydrate": 60}
    elif 18.5 <= bmi < 24.9:
        return {"calories": 300, "proteins": 15, "fat": 10, "carbohydrate": 50}
    elif 25 <= bmi < 29.9:
        return {"calories": 250, "proteins": 20, "fat": 8, "carbohydrate": 40}
    else:
        return {"calories": 200, "proteins": 18, "fat": 5, "carbohydrate": 30}
 