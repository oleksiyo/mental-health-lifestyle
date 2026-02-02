"""
Test script for Mental Health API
Demonstrates how to interact with the deployed service
"""
import requests
import json


# Configuration
API_URL = "http://localhost:9696"  # Change this for cloud deployment


def test_health_check():
    """Test the health endpoint"""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_prediction(example_data):
    """Test the prediction endpoint"""
    print("=" * 60)
    print("Testing Prediction Endpoint")
    print("=" * 60)
    
    print("Input Data:")
    print(json.dumps(example_data, indent=2))
    print()
    
    response = requests.post(
        f"{API_URL}/predict",
        headers={"Content-Type": "application/json"},
        json=example_data
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    if response.status_code == 200:
        result = response.json()
        prob = result.get("probability", 0)
        pred = result.get("prediction", 0)
        
        print("Interpretation:")
        print(f"  → Probability of mental health issue: {prob:.1%}")
        print(f"  → Prediction: {'Yes (at risk)' if pred == 1 else 'No (low risk)'}")
    
    return response


def main():
    """Run all tests"""
    
    # Test 1: Health check
    test_health_check()
    
    # Test 2: High risk profile
    print("\n" + "=" * 60)
    print("TEST 1: High Risk Profile")
    print("=" * 60)
    
    high_risk_data = {
        "Age": 34,
        "Gender": "Male",
        "Country": "United States",
        "Education": "Bachelor",
        "Marital_Status": "Single",
        "Income_Level": "Medium",
        "Employment_Status": "Employed",
        "Work_Hours_Per_Week": 50,
        "Remote_Work": "Yes",
        "Job_Satisfaction": 2,
        "Work_Stress_Level": 8,
        "Work_Life_Balance": 2,
        "Ever_Bullied_At_Work": 1,
        "Company_Mental_Health_Support": "No",
        "Exercise_Per_Week": "None",
        "Sleep_Hours_Night": 5.0,
        "Caffeine_Drinks_Day": 5,
        "Alcohol_Frequency": "Daily",
        "Smoking": "Yes",
        "Screen_Time_Hours_Day": 12.0,
        "Social_Media_Hours_Day": 6.0,
        "Hobby_Time_Hours_Week": 1.0,
        "Diet_Quality": "Poor",
        "Financial_Stress": 8,
        "Social_Support": 2,
        "Close_Friends_Count": 1,
        "Feel_Understood": 1,
        "Loneliness": 8,
        "Discuss_Mental_Health": "No",
        "Family_History_Mental_Illness": "Yes",
        "Previously_Diagnosed": "Yes",
        "Ever_Sought_Treatment": "No",
        "On_Therapy_Now": "No",
        "On_Medication": "No",
        "Trauma_History": "Yes",
    }
    
    test_prediction(high_risk_data)
    
    # Test 3: Low risk profile
    print("\n" + "=" * 60)
    print("TEST 2: Low Risk Profile")
    print("=" * 60)
    
    low_risk_data = {
        "Age": 28,
        "Gender": "Female",
        "Country": "Canada",
        "Education": "Master",
        "Marital_Status": "Married",
        "Income_Level": "High",
        "Employment_Status": "Employed",
        "Work_Hours_Per_Week": 40,
        "Remote_Work": "Hybrid",
        "Job_Satisfaction": 8,
        "Work_Stress_Level": 3,
        "Work_Life_Balance": 8,
        "Ever_Bullied_At_Work": 0,
        "Company_Mental_Health_Support": "Yes",
        "Exercise_Per_Week": "4-5 times",
        "Sleep_Hours_Night": 8.0,
        "Caffeine_Drinks_Day": 1,
        "Alcohol_Frequency": "Rarely",
        "Smoking": "No",
        "Screen_Time_Hours_Day": 4.0,
        "Social_Media_Hours_Day": 1.0,
        "Hobby_Time_Hours_Week": 10.0,
        "Diet_Quality": "Good",
        "Financial_Stress": 2,
        "Social_Support": 9,
        "Close_Friends_Count": 5,
        "Feel_Understood": 9,
        "Loneliness": 2,
        "Discuss_Mental_Health": "Yes",
        "Family_History_Mental_Illness": "No",
        "Previously_Diagnosed": "No",
        "Ever_Sought_Treatment": "No",
        "On_Therapy_Now": "No",
        "On_Medication": "No",
        "Trauma_History": "No",
    }
    
    test_prediction(low_risk_data)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to API at {API_URL}")
        print("Make sure the service is running:")
        print("  python serve.py")
        print("or")
        print("  docker run -p 9696:9696 mental-health-api")
    except Exception as e:
        print(f"ERROR: {e}")
