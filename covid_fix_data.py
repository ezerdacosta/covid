import pandas as pd
import numpy as np

input_file = ''
output_file = ''

df = pd.read_csv(input_file)


def date_to_died(date):
    if date == '9999-99-99':
        return 0
    else:
        return 1

df['DIED'] = df['DATE_DIED'].apply(date_to_died)

def age_to_age_discrete(age):
    if age == 'Unknown':
        return 'Unknown'
    age = int(age)
    if age < 0:
        return 'Negative'
    age_interval = (age // 10) * 10
    return f'{age_interval}-{age_interval + 9}'


df['AGE_DISCRETE'] = df['AGE'].apply(age_to_age_discrete)



columns_to_check = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'DATE_DIED', 'INTUBED', 'PNEUMONIA', 'AGE',
                    'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE',
                    'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU']

def replace_with_nan(value):
    if value not in [1, 2]:
        return np.nan
    return value

for column in columns_to_check:
    df[column] = df[column].apply(replace_with_nan)


df = df.drop(columns=['AGE', 'PREGNANT', 'INTUBED', 'ICU', 'DATE_DIED','CLASIFFICATION_FINAL'])
df.to_csv(output_file, index=False)



print("Conversion completed. The result is saved to", output_file)
