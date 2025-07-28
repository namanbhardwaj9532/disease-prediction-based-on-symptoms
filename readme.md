#  Disease Prediction Based on Symptoms

This is a web-based machine learning project that predicts the possible disease based on user-input symptoms. The project uses a trained ML model on a labeled dataset of symptoms and diseases. The interface is built with HTML and styled using Bootstrap, while the backend is powered by Python and Flask.

---

##  Project Structure

```
disease_prediction_advanced_ui/
│
├── static/
│   └── (Bootstrap, custom CSS, etc.)
│
├── templates/
│   └── index.html (Main UI)
│
├── app.py                # Flask backend
├── model.pkl             # Trained ML model
├── requirements.txt      # Python dependencies
└── README.md             # Project overview (this file)
```

---

##  How It Works

1. The user selects symptoms from a dropdown UI.
2. These symptoms are sent to the Flask backend.
3. The backend uses a trained model (`model.pkl`) to predict the most probable disease.
4. The predicted disease is shown on the frontend.

---

##  Dependencies

- Flask
- scikit-learn
- pandas
- joblib
- (Others as listed in `requirements.txt`)

---

##  Dataset Used

A labeled dataset of symptoms and diseases. Each row represents a patient case with 1 or more symptoms mapped to a disease. Data is preprocessed and used to train a classification model.

---

##  ML Model

The model is a classification algorithm (e.g., Decision Tree / Random Forest) trained on symptom data and stored in `model.pkl` using `joblib`.

---

##  UI Features

- Symptom input using dropdowns or checkboxes.
- Bootstrap-based clean design.
- Real-time prediction shown after clicking submit.

---

##  Future Improvements

- Add multiple disease predictions with confidence scores.
- Use deep learning for improved accuracy.
- Connect to a real-time medical database or API.
- Add user registration & medical history tracking.

---

##  Author

**Naman Bhardwaj**  
B.Tech in Data Science & AI  
Graphic Era Hill University

---

##  License

This project is for educational purposes. You can modify and use it under an open license.