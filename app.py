from flask import Flask, request, render_template
import joblib
import numpy as np
import threading
import webbrowser

app = Flask(__name__, template_folder="templates")

model = joblib.load("model.pkl")
all_symptoms = joblib.load("symptom_list.pkl")
le = joblib.load("label_encoder.pkl")
disease_info = joblib.load("disease_info.pkl")

@app.route('/')
def home():
    return render_template("index.html", all_symptoms=all_symptoms)

@app.route('/predict')
def predict():
    name = request.args.get("name")
    selected_symptoms = request.args.getlist("symptoms")
    input_features = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
    proba = model.predict_proba([input_features])[0]
    top_indices = np.argsort(proba)[-3:][::-1]

    top_diseases = []
    for idx in top_indices:
        disease_name = le.inverse_transform([idx])[0]
        info = disease_info.get(disease_name, {})
        top_diseases.append({
            "name": disease_name,
            "description": info.get("description", "No description available."),
            "precautions": info.get("precautions", [])
        })

    return render_template("result.html",
                           name=name,
                           selected_symptoms=selected_symptoms,
                           top_diseases=top_diseases)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001/")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False, port=5001)
