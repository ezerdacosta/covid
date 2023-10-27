import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.combination import average
from pyod.utils.utility import standardizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


data = pd.read_csv('/Users/baptiald/Downloads/Covid Data.csv')


data.fillna(value=np.nan, inplace=True)


age_bins = [0, 18, 30, 50, 100]
age_labels = ['0-17', '18-29', '30-49', '50+']
data['AGE'] = pd.cut(data['AGE'], bins=age_bins, labels=age_labels)
label_encoder = LabelEncoder()
data['AGE'] = label_encoder.fit_transform(data['AGE'])


data['DIED'] = data['DATE_DIED'].apply(lambda x: False if x == '9999-99-99' else True)

label_encoder = LabelEncoder()
data['AGE'] = label_encoder.fit_transform(data['AGE'])


columns_to_use = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'AGE', 'PREGNANT',
                  'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
                  'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU', 'DIED']
data = data[columns_to_use]


data = pd.DataFrame(standardizer(data), columns=columns_to_use)

# Train multiple anomaly detection models
models = {
    "Isolation Forest": IForest(contamination=0.05),  # Adjust contamination as needed
    "Local Outlier Factor": LOF(),
    "One-Class SVM": OCSVM()
}

results = {}

for model_name, model in models.items():
    model.fit(data)
    data[f'Anomaly_{model_name}'] = model.predict(data)

# Ensemble the results of individual models
data['Anomaly_Ensemble'] = average(
    data[['Anomaly_Isolation Forest', 'Anomaly_Local Outlier Factor', 'Anomaly_One-Class SVM']], axis=1)

# Evaluate the results
y_true = data['DIED'].astype(int)

for model_name in models.keys():
    y_pred = data[f'Anomaly_{model_name}']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    results[model_name] = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

ensemble_precision = precision_score(y_true, data['Anomaly_Ensemble'])
ensemble_recall = recall_score(y_true, data['Anomaly_Ensemble'])
ensemble_f1 = f1_score(y_true, data['Anomaly_Ensemble'])
ensemble_roc_auc = roc_auc_score(y_true, data['Anomaly_Ensemble'])

results['Ensemble'] = {
    'Precision': ensemble_precision,
    'Recall': ensemble_recall,
    'F1-Score': ensemble_f1,
    'ROC-AUC': ensemble_roc_auc
}

print("Individual Model Results:")
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print()

print("Ensemble Model Results:")
print(f"Precision: {ensemble_precision:.4f}")
print(f"Recall: {ensemble_recall:.4f}")
print(f"F1-Score: {ensemble_f1:.4f}")
print(f"ROC-AUC: {ensemble_roc_auc:.4f}")
