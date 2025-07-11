# This script generates realistic, synthetic data for a single example user, "Mia Grayne".
# This provides a working demo for new users to see.

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# --- Configuration ---
EXAMPLE_USER_ID = "Mia Grayne"
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)
N_DAYS = (END_DATE - START_DATE).days

# Define file paths
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'{EXAMPLE_USER_ID}_data.csv')

# Define possible values for the categorical fields
FOOD_LEVELS = ['none', 'little', 'lot']
MEAL_TIMES = ['on_time', 'delayed']
EXERCISE_LEVELS = ['none', 'little', 'moderate', 'heavy']
WEATHER_TYPES = ['sunny', 'rainy', 'gloomy', 'hot']
SOUND_LEVELS = ['light', 'moderate', 'heavy']
MEDICATION_STATUS = ['yes', 'no']
CONFLICT_LEVELS = ['none', 'light', 'moderate', 'high']

def generate_example_data():
    """Generates a synthetic dataset for the example user."""
    data = []
    print(f"Generating example data for user: {EXAMPLE_USER_ID}...")
    
    migraine_season_months = [6, 7, 8] # Example season
    
    for day_offset in range(N_DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        
        log = {
            'user_id': EXAMPLE_USER_ID,
            'date': current_date.strftime('%Y-%m-%d'),
            'chocolate': random.choice(FOOD_LEVELS),
            'salty_food': random.choice(FOOD_LEVELS),
            'spicy_food': random.choice(FOOD_LEVELS),
            'meal_times': random.choice(MEAL_TIMES),
            'exercise_level': random.choice(EXERCISE_LEVELS),
            'weather': random.choice(WEATHER_TYPES),
            'sound_exposure': random.choice(SOUND_LEVELS),
            'screen_time_hours': random.randint(1, 12),
            'water_cups': random.randint(2, 15),
            'other_medication': random.choice(MEDICATION_STATUS),
            'conflict': random.choice(CONFLICT_LEVELS),
        }
        
        migraine_chance = 0.05
        if log['meal_times'] == 'delayed': migraine_chance += 0.15
        if log['water_cups'] < 5: migraine_chance += 0.15
        if log['screen_time_hours'] > 8: migraine_chance += 0.15
        if log['conflict'] in ['moderate', 'high']: migraine_chance += 0.20
        if current_date.month in migraine_season_months: migraine_chance += 0.25
            
        log['migraine_attack'] = 1 if random.random() < migraine_chance else 0
        data.append(log)

    df = pd.DataFrame(data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Synthetic data for {EXAMPLE_USER_ID} saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_example_data()
