import pandas as pd

# Carica il file CSV in un DataFrame
df = pd.read_csv('HPC/performance/tests/performance.csv')

# Filtra le righe che non contengono "PPHT" nella colonna HT_version
df_filtrato = df[df['HT_version'] != 'PPHT']

# Salva il DataFrame filtrato in un nuovo file CSV
df_filtrato.to_csv('HPC/performance/tests/file_filtrato.csv', index=False, float_format='%.6f')
