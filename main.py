import pandas as pd

RUTA_CSV = "historico_agosto2026_BTC.csv"

# Cargar el CSV
df = pd.read_csv(RUTA_CSV)
print(f"Registros cargados: {len(df)}")

# convierto timestamp_original al formato pedido
df["timestamp_original"] = df["timestamp"]
# guardo el texto original para exportarlo asi despues
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# filtro agosto 2026
inicio = pd.Timestamp("2026-08-01 00:00:00", tz="UTC")
fin    = pd.Timestamp("2026-08-31 23:45:00", tz="UTC")
df = df[(df["timestamp"] >= inicio) & (df["timestamp"] <= fin)]
print(f"Registros tras filtrar agosto: {len(df)}")

# ordenar cronológicamente
df = df.sort_values("timestamp").reset_index(drop=True)

# compruebo la integridad temporal
assert not df["timestamp"].duplicated().any(), "Hay timestamps duplicados"

diffs = df["timestamp"].diff().dropna()
huecos = diffs[diffs != pd.Timedelta(minutes=15)]
if len(huecos) > 0:
    for idx, delta in huecos.items():
        print(f"Salto entre {df['timestamp'].iloc[idx-1]} y {df['timestamp'].iloc[idx]}: {delta}")
    raise ValueError("El dataset presenta huecos temporales")

print("Integridad temporal OK: todas las ventanas están separadas exactamente 15 min")
print(f"Rango final: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")