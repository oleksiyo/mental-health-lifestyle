import pickle
import numpy as np
from typing import Dict


MODEL_PATH = "model.bin"


def load_artifacts():
    """
    Load trained model and DictVectorizer from disk.
    """
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    dv = artifact["dict_vectorizer"]

    return model, dv


def predict(features: Dict) -> Dict:
    """
    Make prediction for a single observation.

    Returns:
        - probability of mental health issue
        - predicted class (0 / 1)
    """
    model, dv = load_artifacts()

    X = dv.transform([features])
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    return {
        "probability": float(prob),
        "prediction": pred,
    }


if __name__ == "__main__":

    example = {
        "Age": 34,
        "Gender": "Male",
        "Country": "United States",
        "Education": "Bachelor",
        "Marital_Status": "Single",
        "Income_Level": "Medium",
        "Employment_Status": "Employed",
        "Work_Hours_Per_Week": 45,
        "Remote_Work": "Yes",
        "Job_Satisfaction": 3,
        "Work_Stress_Level": 4,
        "Work_Life_Balance": 2,
        "Ever_Bullied_At_Work": 0,
        "Company_Mental_Health_Support": "No",
        "Exercise_Per_Week": "1-2",
        "Sleep_Hours_Night": 6.0,
        "Caffeine_Drinks_Day": 3,
        "Alcohol_Frequency": "Weekly",
        "Smoking": "No",
        "Screen_Time_Hours_Day": 7.5,
        "Social_Media_Hours_Day": 3.0,
        "Hobby_Time_Hours_Week": 4.0,
        "Diet_Quality": "Average",
        "Financial_Stress": 4,
        "Social_Support": 3,
        "Close_Friends_Count": 2,
        "Feel_Understood": 2,
        "Loneliness": 4,
        "Discuss_Mental_Health": "No",
        "Family_History_Mental_Illness": "Yes",
        "Previously_Diagnosed": "No",
        "Ever_Sought_Treatment": "No",
        "On_Therapy_Now": "No",
        "On_Medication": "No",
        "Trauma_History": "Yes",
    }

    result = predict(example)

    print("Prediction result:")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Class: {result['prediction']}")
