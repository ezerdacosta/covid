import pandas as pd


input_file = '/Users/baptiald/Downloads/Covid_Data_TROUBLESHOOTING[REDUCED].csv'
output_file = '/Users/baptiald/Downloads/Covid_Data_TROUBLESHOOTING[REDUCEDA_ATRIBUTE].csv'

df = pd.read_csv(input_file)
df = df.drop(columns=['DIED'])
df.to_csv(output_file, index=False)

print("Conversion completed. The result is saved to", output_file)