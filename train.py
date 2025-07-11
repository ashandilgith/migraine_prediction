# This script trains a personalized TensorFlow model for a specific user.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def train_model(user_id):
    """Loads a user's data, preprocesses it with one-hot encoding, and trains a model."""
    
    print(f"Starting training for user: {user_id}...")
    
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', f'{user_id}_data.csv')
    model_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(model_dir, f'{user_id}_model.keras')
    scaler_path = os.path.join(model_dir, f'{user_id}_scaler.joblib')
    columns_path = os.path.join(model_dir, f'{user_id}_columns.joblib')

    if not os.path.exists(data_path):
        return False, f"Data file for {user_id} not found. Please log some data first."
        
    df = pd.read_csv(data_path)
    if len(df) < 10:
        return False, "Not enough data to train. Please log at least 10 days."

    df['date'] = pd.to_datetime(df['date'])
    
    # --- 1. Feature Engineering ---
    df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)
    
    y = df['migraine_attack']
    X = df.drop(columns=['user_id', 'date', 'migraine_attack'])

    # --- 2. Preprocessing ---
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    scaler = StandardScaler()
    X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(X_encoded.columns.tolist(), columns_path)

    # --- 3. Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    class_weight = {0: 1, 1: (len(df) - y.sum()) / y.sum() if y.sum() > 0 else 1}
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, class_weight=class_weight, verbose=0)
    
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)
    
    print(f"Training complete for {user_id}. Model saved to {model_path}")
    return True, "Training complete! Your model is now updated."

if __name__ == "__main__":
    # This block now only trains the model for the example user when run directly.
    print("Running training script for example user 'Mia Grayne'...")
    train_model(user_id="Mia Grayne")
