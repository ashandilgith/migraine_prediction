# This is the main Gradio application with the new detailed input fields.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from train import train_model

# --- Configuration & Helper Functions ---
BASE_DIR = os.path.dirname(__file__)
USERS_FILE_PATH = os.path.join(BASE_DIR, 'authorized_users.txt')

def get_authorized_users():
    if not os.path.exists(USERS_FILE_PATH): return []
    with open(USERS_FILE_PATH, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

def load_user_assets(user_id):
    model_path = os.path.join(BASE_DIR, 'models', f'{user_id}_model.keras')
    scaler_path = os.path.join(BASE_DIR, 'models', f'{user_id}_scaler.joblib')
    columns_path = os.path.join(BASE_DIR, 'models', f'{user_id}_columns.joblib')
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        columns = joblib.load(columns_path)
        return model, scaler, columns
    except IOError:
        return None, None, None

def log_data(user_id, date, *args):
    if not user_id: return "Please log in first."
    data_path = os.path.join(BASE_DIR, 'data', f'{user_id}_data.csv')
    
    migraine_outcome_str = args[-1]
    migraine_attack = 0 if migraine_outcome_str == "None" else 1
    
    new_log = {
        "user_id": user_id, "date": date,
        "chocolate": args[0], "salty_food": args[1], "spicy_food": args[2],
        "meal_times": args[3], "exercise_level": args[4],
        "weather": args[5], "sound_exposure": args[6], "screen_time_hours": args[7],
        "water_cups": args[8], "other_medication": args[9], "conflict": args[10],
        "migraine_attack": migraine_attack
    }
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if date in df['date'].values: return f"An entry for {date} already exists."
        df = pd.concat([df, pd.DataFrame([new_log])], ignore_index=True)
    else:
        df = pd.DataFrame([new_log])
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df.to_csv(data_path, index=False)
    return f"Successfully logged data for {date}. You now have {len(df)} entries."

def forecast_risk(user_id, date, *args):
    if not user_id: return "Please log in first.", ""
    model, scaler, columns = load_user_assets(user_id)
    if model is None: return "Model not trained yet.", "Please go to the 'Train Model' tab."

    data_path = os.path.join(BASE_DIR, 'data', f'{user_id}_data.csv')
    warning_message = ""
    if os.path.exists(data_path) and len(pd.read_csv(data_path)) < 30:
        warning_message = f"Warning: Model has only {len(pd.read_csv(data_path))} data points. Accuracy improves after 30."

    input_data = {
        "chocolate": args[0], "salty_food": args[1], "spicy_food": args[2],
        "meal_times": args[3], "exercise_level": args[4],
        "weather": args[5], "sound_exposure": args[6], "screen_time_hours": args[7],
        "water_cups": args[8], "other_medication": args[9], "conflict": args[10]
    }
    input_df = pd.DataFrame([input_data])
    
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    input_df['day_sin'] = np.sin(2 * np.pi * date_obj.timetuple().tm_yday / 365.25)
    input_df['day_cos'] = np.cos(2 * np.pi * date_obj.timetuple().tm_yday / 365.25)
    
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_aligned = input_encoded.reindex(columns=columns, fill_value=0)
    
    numerical_features = [col for col in ['screen_time_hours', 'water_cups', 'day_sin', 'day_cos'] if col in input_aligned.columns]
    input_aligned[numerical_features] = scaler.transform(input_aligned[numerical_features])
    
    risk_prob = model.predict(input_aligned)[0][0]
    risk_score = risk_prob * 100
    
    advice = "Risk is LOW."
    if risk_score > 70: advice = "Risk is VERY HIGH. Prioritize rest and be prepared."
    elif risk_score > 40: advice = "Risk is MODERATE. Be mindful of your triggers."
        
    return f"{risk_score:.0f}%", f"{advice}\n\n{warning_message}".strip()

def handle_training(user_id):
    if not user_id: yield "Please log in first."
    else:
        yield "Training started... This might take a minute."
        success, message = train_model(user_id)
        yield message

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    user_state = gr.State(None)

    with gr.Row(visible=True) as login_view:
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("# Welcome to Migraine Gabay")
            username_input = gr.Textbox(label="Enter Your Username", placeholder="e.g., user_jane")
            login_btn = gr.Button("Login")
            login_status = gr.Markdown("")

    with gr.Column(visible=False) as app_view:
        gr.Markdown("# 🧠 Migraine Gabay: Personalized Forecaster")
        welcome_message = gr.Markdown("")
        
        with gr.Tabs():
            with gr.TabItem("Log Daily Data"):
                gr.Markdown("### 📝 Add a New Daily Entry")
                with gr.Row():
                    log_date_input = gr.Textbox(label="Date", value=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))
                    log_migraine_outcome = gr.Radio(["None", "Weak Migraine", "Strong Migraine"], label="Migraine Outcome?", value="None")
                
                with gr.Accordion("Food & Drink", open=False):
                    log_chocolate = gr.Radio(['none', 'little', 'lot'], label="Chocolate", value='none')
                    log_salty = gr.Radio(['none', 'little', 'lot'], label="Salty Food", value='none')
                    log_spicy = gr.Radio(['none', 'little', 'lot'], label="Spicy Food", value='none')
                    log_meal_times = gr.Radio(['on_time', 'delayed'], label="Meal Times", value='on_time')
                    log_water = gr.Slider(0, 20, value=8, step=1, label="Water (200ml cups)")

                with gr.Accordion("Activities & Environment", open=False):
                    log_exercise = gr.Radio(['none', 'little', 'moderate', 'heavy'], label="Exercise Level", value='little')
                    log_weather = gr.Radio(['sunny', 'rainy', 'gloomy', 'hot'], label="Weather", value='sunny')
                    log_sound = gr.Radio(['light', 'moderate', 'heavy'], label="Sound Exposure", value='light')
                    log_screen = gr.Slider(0, 16, value=6, step=1, label="Screen Time (hours)")

                with gr.Accordion("Health & Stress", open=False):
                    log_meds = gr.Radio(['yes', 'no'], label="Took Other Medication?", value='no')
                    log_conflict = gr.Radio(['none', 'light', 'moderate', 'high'], label="Conflict/Confrontation", value='none')
                
                log_btn = gr.Button("Log This Day's Data", variant="primary")
                log_status = gr.Textbox(label="Log Status", interactive=False)
                
                log_inputs = [log_chocolate, log_salty, log_spicy, log_meal_times, log_exercise, log_weather, log_sound, log_screen, log_water, log_meds, log_conflict, log_migraine_outcome]

            with gr.TabItem("Forecast"):
                gr.Markdown("### 📈 Get Your Personalized Forecast")
                fc_date_input = gr.Textbox(label="Date to Forecast", value=datetime.now().strftime('%Y-%m-%d'))
                
                with gr.Accordion("Food & Drink", open=True):
                    fc_chocolate = gr.Radio(['none', 'little', 'lot'], label="Chocolate", value='none')
                    fc_salty = gr.Radio(['none', 'little', 'lot'], label="Salty Food", value='none')
                    fc_spicy = gr.Radio(['none', 'little', 'lot'], label="Spicy Food", value='none')
                    fc_meal_times = gr.Radio(['on_time', 'delayed'], label="Meal Times", value='on_time')
                    fc_water = gr.Slider(0, 20, value=8, step=1, label="Water (200ml cups)")

                with gr.Accordion("Activities & Environment", open=True):
                    fc_exercise = gr.Radio(['none', 'little', 'moderate', 'heavy'], label="Exercise Level", value='little')
                    fc_weather = gr.Radio(['sunny', 'rainy', 'gloomy', 'hot'], label="Weather", value='sunny')
                    fc_sound = gr.Radio(['light', 'moderate', 'heavy'], label="Sound Exposure", value='light')
                    fc_screen = gr.Slider(0, 16, value=6, step=1, label="Screen Time (hours)")

                with gr.Accordion("Health & Stress", open=True):
                    fc_meds = gr.Radio(['yes', 'no'], label="Took Other Medication?", value='no')
                    fc_conflict = gr.Radio(['none', 'light', 'moderate', 'high'], label="Conflict/Confrontation", value='none')

                fc_btn = gr.Button("Forecast My Risk", variant="primary")
                fc_risk = gr.Textbox(label="Migraine Risk Score", interactive=False)
                fc_advice = gr.Textbox(label="Actionable Advice", lines=4, interactive=False)
                
                fc_inputs = [fc_chocolate, fc_salty, fc_spicy, fc_meal_times, fc_exercise, fc_weather, fc_sound, fc_screen, fc_water, fc_meds, fc_conflict]

            with gr.TabItem("Train Model"):
                train_btn = gr.Button("Train My Personal Model")
                train_status = gr.Textbox(label="Training Status", interactive=False)

    def login(username):
        if username in get_authorized_users():
            return {user_state: username, login_view: gr.update(visible=False), app_view: gr.update(visible=True), welcome_message: gr.update(value=f"### Welcome, {username}!"), login_status: ""}
        else:
            return {login_status: gr.update(value="<p style='color:red;'>Username not found.</p>")}

    login_btn.click(login, inputs=[username_input], outputs=[user_state, login_view, app_view, welcome_message, login_status])
    log_btn.click(log_data, inputs=[user_state, log_date_input] + log_inputs, outputs=[log_status])
    fc_btn.click(forecast_risk, inputs=[user_state, fc_date_input] + fc_inputs, outputs=[fc_risk, fc_advice])
    train_btn.click(handle_training, inputs=[user_state], outputs=[train_status])

if __name__ == "__main__":
    demo.launch(share=True)
