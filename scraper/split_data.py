import pandas as pd

# Učitaj podatke
df = pd.read_parquet('properties_raw.parquet')

# Podijeli na stanove i kuće
stanovi = df[df['vrsta_nekretnine'] == 'stan']
kuce = df[df['vrsta_nekretnine'] == 'kuca']

# Spremi u zasebne datoteke
stanovi.to_parquet('apartments.parquet', index=False)
kuce.to_parquet('houses.parquet', index=False)

print(f"Stanovi: {len(stanovi)} zapisa -> apartments.parquet")
print(f"Kuće: {len(kuce)} zapisa -> houses.parquet")
