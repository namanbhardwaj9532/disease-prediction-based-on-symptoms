import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load datasets
symptom_severity = pd.read_csv("Symptom-severity.csv")
precautions = pd.read_excel("symptom_precaution (1).xlsx")
descriptions = pd.read_excel("symptom_Description.xlsx")

# Create disease_info dictionary
disease_info = {}
for _, row in descriptions.iterrows():
    disease = row['Disease']
    description = row['Description']
    
    prec_row = precautions[precautions['Disease'] == disease]
    if not prec_row.empty:
        prec_row = prec_row.iloc[0]
        disease_precautions = [
            prec_row[f'Precaution_{i}']
            for i in range(1, 5)
            if pd.notna(prec_row[f'Precaution_{i}'])
        ]
    else:
        disease_precautions = ["No precautions available"]
    
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
    for i in range(1, len(symptoms)+1):
        row[f'Symptom_{i}'] = symptoms[i-1]
    synthetic_data.append(row)

df = pd.DataFrame(synthetic_data)

symptom_columns = [col for col in df.columns if col.startswith("Symptom")]
label_column = "Disease"

# Get unique symptoms
all_symptoms = pd.unique(df[symptom_columns].values.ravel())
all_symptoms = sorted([s for s in all_symptoms if pd.notna(s)])

# Encode symptoms into binary vector
def encode_symptoms(row):
    row_symptoms = set(row[symptom_columns])
    return [1 if symptom in row_symptoms else 0 for symptom in all_symptoms]

X = df.apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms
y = df[label_column]

# Label encode disease names
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and related files
joblib.dump(model, "model.pkl")
joblib.dump(all_symptoms, "symptom_list.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(disease_info, "disease_info.pkl")

print("✅ Model and data saved successfully.")
