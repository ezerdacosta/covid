import pandas as pd


input_file = ''
output_file = ''

df = pd.read_csv(input_file)
df = df.drop(columns=['DIED'])
df.to_csv(output_file, index=False)

print("Conversion completed. The result is saved to", output_file)