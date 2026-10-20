import gradio as gr
import joblib
import numpy as np
import os


def load_assets():
    model_path = 'crop_model.pkl'
    encoder_path = 'label_encoder.joblib' 
    
    model = None
    label_encoder = None
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    
    if os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
        
    return model, label_encoder

model, label_encoder = load_assets()

def predict_crop(N, P, K, temp, hum, ph, rain):
    if model is None or label_encoder is None:
        return "❌ Error: Required files (model or encoder) are missing or not uploaded correctly."
    
    try:
        
        input_data = np.array([[float(N), float(P), float(K), float(temp), float(hum), float(ph), float(rain)]])
        
        
        prediction = model.predict(input_data)
        
        crop_name = label_encoder.inverse_transform(prediction)
        
        return f"🌱 Recommended Crop: {crop_name[0].upper()}"
            
    except Exception as e:
        return f"⚠️ Prediction Error: {str(e)}"

# الواجهة
app = gr.Interface(
    fn=predict_crop,
    inputs=[gr.Number(label=name) for name in [" Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "pH", "Rainfall"]],
    outputs=gr.Textbox(label="Result"),
    title="Crop Recommendation System",
    description="Smart Crop Recommendation System using Machine Learning"
)

if __name__ == "__main__":
    app.launch()