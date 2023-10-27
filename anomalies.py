import pandas as pd
import numpy as np
from pyod.models.iforest import IForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('/Users/baptiald/Downloads/Covid Data.csv')

# Handle missing values
data.fillna(value=np.nan, inplace=True)

# Convert 'AGE' into categorical age groups (you can define your own age groups)
age_bins = [0, 18, 30, 50, 100]
age_labels = ['0-17', '18-29', '30-49', '50+']
data['AGE'] = pd.cut(data['AGE'], bins=age_bins, labels=age_labels)
label_encoder = LabelEncoder()
data['AGE'] = label_encoder.fit_transform(data['AGE'])

# Create the 'DIED' column based on 'DATE_DIED'
data['DIED'] = data['DATE_DIED'].apply(lambda x: False if x == '9999-99-99' else True)

columns_to_use = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'AGE', 'PREGNANT',
                  'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
                  'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU', 'DIED']
data = data[columns_to_use]

# Train the Isolation Forest model
clf = IForest(contamination=0.05)  # Adjust the contamination parameter as needed
clf.fit(data)

data['Anomaly'] = clf.predict(data)


y_true = data['DIED'].astype(int)
y_pred = data['Anomaly'].apply(lambda x: 1 if x == 1 else 0)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'ROC-AUC: {roc_auc}')