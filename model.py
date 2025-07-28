import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
symptom_severity = pd.read_csv(os.path.join(BASE_DIR, "Symptom-severity.csv"))
precautions = pd.read_excel(os.path.join(BASE_DIR, "symptom_precaution (1).xlsx"))
descriptions = pd.read_excel(os.path.join(BASE_DIR, "symptom_Description.xlsx"))

# Combine description and precautions
disease_info = {}
for _, row in descriptions.iterrows():
    disease = row['Disease']
    description = row['Description']

    # Match precaution row for this disease
    prec_row = precautions[precautions['Disease'] == disease]

    #  FIX: Extract actual string values, not Series
    disease_precautions = [
        prec_row.iloc[0][f'Precaution_{i}'] for i in range(1, 5)
        if not prec_row.empty and pd.notna(prec_row.iloc[0][f'Precaution_{i}'])
    ]

    disease_info[disease] = {
        'description': description,
        'precautions': disease_precautions
    }

# Generate synthetic training data
synthetic_data = []
for disease in disease_info.keys():
    num_symptoms = np.random.randint(3, 8)
    symptoms = symptom_severity.sample(num_symptoms)['Symptom'].tolist()
    row = {'Disease': disease}
    for i, symptom in enumerate(symptoms, 1):
        row[f"Symptom_{i}"] = symptom
    synthetic_data.append(row)

# Create DataFrame
df = pd.DataFrame(synthetic_data)
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
all_symptoms = sorted({s for val in df[symptom_cols].values.flatten() if pd.notna(s := val)})

# Encode symptoms into binary format
def encode_symptoms(row):
    return [1 if s in row.values else 0 for s in all_symptoms]

X = df[symptom_cols].apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms
y = df['Disease']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save all files
joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(all_symptoms, os.path.join(BASE_DIR, "symptom_list.pkl"))
joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))
joblib.dump(disease_info, os.path.join(BASE_DIR, "disease_info.pkl"))

print("Model training complete. Files saved.")
