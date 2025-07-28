from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your models and data
all_symptoms = joblib.load("symptom_list.pkl")
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
disease_info = joblib.load("disease_info.pkl")

@app.route('/')
def home():
    return render_template("index.html", symptoms=all_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get("name")
    age = request.form.get("age")
    sex = request.form.get("sex")
    selected_symptoms = request.form.getlist("symptoms")

    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
    prediction = model.predict([input_vector])[0]
    disease = le.inverse_transform([prediction])[0]

    description = disease_info[disease]["description"]
    precautions = disease_info[disease]["precautions"]

    return render_template("result.html", name=name, disease=disease,
                           description=description, precautions=precautions)
