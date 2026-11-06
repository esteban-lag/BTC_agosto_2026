import json
import logging
from pathlib import Path

import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_ORIGEN = "aruba_readings_20260924.json"

CARPETA_SALIDA = Path("datos_salida")
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

ESTACIONES = {
    "963fdb57-b257-472d-b56e-892b265c2524": "San Nicolaas",
    "82a31d3a-9895-42e5-8ea0-11e8cb15b964": "Savaneta",
    "acb2fa5c-518a-4a38-b68d-bf8e3b041ccf": "Arikok National Park",
    "de2ee07b-de63-49db-a669-64177a688bbf": "Queen Beatrix Airport",
    "4ca4c103-894d-4f59-a911-f3abb8077e3c": "Alto Vista"
}

VARIABLES = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "wind_direction_deg",
    "pressure_hpa",
    "precipitation_mm",
    "visibility_km",
    "uv_index"
]

RANGOS = {
    "temperature_c": (0, 55),
    "humidity_pct": (0, 100)
}

ZONA_HORARIA = "America/Aruba"

INICIO_AGOSTO = pd.Timestamp(
    "2026-08-01 00:00:00",
    tz=ZONA_HORARIA
)

FIN_AGOSTO = pd.Timestamp(
    "2026-09-01 00:00:00",
    tz=ZONA_HORARIA
)

FRECUENCIA_ORIGINAL = "5min"

LATITUD = 12.5211
LONGITUD = -69.9683

URL_API = "https://archive-api.open-meteo.com/v1/archive"

VARIABLES_API = {
    "shortwave_radiation": "solar_shortwave_wm2",
    "direct_radiation": "solar_direct_wm2",
    "diffuse_radiation": "solar_diffuse_wm2",
    "wind_speed_100m": "wind_speed_100m_kmh",
    "wind_direction_100m": "wind_direction_100m_deg"
}

TOLERANCIA_UNION = pd.Timedelta("60min")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# CONEXIÓN CON OPEN-METEO
# ============================================================

def crear_sesion():

    reintentos = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adaptador = HTTPAdapter(max_retries=reintentos)

    sesion = requests.Session()
    sesion.mount("https://", adaptador)

    sesion.headers.update({
        "User-Agent": "Meteoguess/1.0"
    })

    return sesion


# ============================================================
# LECTURA Y SELECCIÓN
# ============================================================

def cargar_datos():

    logging.info("Leyendo archivo original...")

    ruta = Path(ARCHIVO_ORIGEN)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra {ARCHIVO_ORIGEN}"
        )

    df = pd.read_json(ruta)

    filas_originales = len(df)

    columnas_requeridas = [
        "station_id",
        "timestamp"
    ] + VARIABLES

    faltantes = [
        c for c in columnas_requeridas
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias: {faltantes}"
        )

    # Selección de las cinco estaciones.

    df["station_id"] = (
        df["station_id"]
        .astype("string")
        .str.strip()
    )

    df = df[
        df["station_id"].isin(ESTACIONES.keys())
    ].copy()

    if df.empty:
        raise ValueError(
            "No se han encontrado las estaciones seleccionadas."
        )

    ids_presentes = set(df["station_id"].unique())

    ids_ausentes = set(ESTACIONES.keys()) - ids_presentes

    if ids_ausentes:
        raise ValueError(
            "Faltan estaciones: "
            + str([
                ESTACIONES[x] for x in ids_ausentes
            ])
        )

    # Transformación de fechas y variables.

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Existen timestamps inválidos."
        )

    for variable in VARIABLES:
        df[variable] = pd.to_numeric(
            df[variable],
            errors="coerce"
        )

    # Ordenación y eliminación del identificador individual.

    df = df.sort_values(
        ["station_id", "timestamp"],
        kind="stable"
    )

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df, filas_originales


# ============================================================
# LIMPIEZA Y ETL
# ============================================================

def limpiar_datos(df):

    logging.info("Iniciando limpieza de datos...")

    calidad = {}

    calidad["filas_despues_seleccion"] = len(df)

    calidad["estaciones"] = int(
        df["station_id"].nunique()
    )

    calidad["registros_por_estacion_antes"] = (
        df.groupby("station_id").size().to_dict()
    )

    # Auditoría de duplicados.

    clave = ["timestamp", "station_id"]

    mascara_duplicados = df.duplicated(
        subset=clave,
        keep=False
    )

    filas_marcadas = int(
        mascara_duplicados.sum()
    )

    filas_eliminadas = int(
        df.duplicated(
            subset=clave,
            keep="first"
        ).sum()
    )

    grupos_duplicados = int(
        df.loc[mascara_duplicados]
        .groupby(clave)
        .ngroups
    )

    calidad["filas_marcadas_como_duplicadas"] = (
        filas_marcadas
    )

    calidad["grupos_duplicados"] = (
        grupos_duplicados
    )

    calidad["filas_eliminadas_por_duplicidad"] = (
        filas_eliminadas
    )

    # Guardar los registros repetidos para su revisión.

    if mascara_duplicados.any():

        duplicados = df.loc[
            mascara_duplicados
        ].copy()

        duplicados["station_name"] = (
            duplicados["station_id"].map(ESTACIONES)
        )

        duplicados.to_csv(
            CARPETA_SALIDA / "duplicados_detectados.csv",
            index=False
        )

    # Aplicar exactamente el criterio de limpieza establecido.

    df = df.drop_duplicates(
        subset=["timestamp", "station_id"],
        keep="first"
    ).copy()

    calidad["filas_despues_duplicados"] = len(df)

    # Identificación de valores fuera de rango.

    invalidos = pd.Series(
        False,
        index=df.index
    )

    for variable, (minimo, maximo) in RANGOS.items():

        invalidos |= (
            ~df[variable].between(minimo, maximo)
        )

        calidad[
            f"{variable}_fuera_rango"
        ] = int(
            (~df[variable].between(minimo, maximo)).sum()
        )

    calidad["filas_fuera_rango"] = int(
        invalidos.sum()
    )

    # Conservar los registros descartados por el filtro.

    if invalidos.any():

        df.loc[invalidos].to_csv(
            CARPETA_SALIDA / "registros_fuera_rango.csv",
            index=False
        )

    df = df.loc[~invalidos].copy()

    df["station_name"] = (
        df["station_id"].map(ESTACIONES)
    )

    df["timestamp_local"] = (
        df["timestamp"].dt.tz_convert(ZONA_HORARIA)
    )

    df = (
        df.sort_values(
            ["station_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    calidad["filas_limpias"] = len(df)

    calidad["registros_por_estacion_despues"] = (
        df.groupby("station_name").size().to_dict()
    )

    calidad["valores_nulos_por_variable"] = (
        df[VARIABLES].isna().sum().to_dict()
    )

    logging.info(
        "Registros después de la limpieza: %s",
        len(df)
    )

    return df, calidad


# ============================================================
# CONTROL DE FRECUENCIA Y COBERTURA
# ============================================================

def comprobar_temporalidad(df):

    logging.info("Comprobando continuidad temporal...")

    detalle = {}

    for station_id, grupo in df.groupby("station_id"):

        nombre = ESTACIONES[station_id]

        fechas = (
            grupo["timestamp"]
            .sort_values()
        )

        diferencias = fechas.diff().dropna()

        intervalos_irregulares = diferencias[
            diferencias != pd.Timedelta(
                FRECUENCIA_ORIGINAL
            )
        ]

        detalle[nombre] = {
            "filas": int(len(grupo)),
            "primera_fecha": str(fechas.min()),
            "ultima_fecha": str(fechas.max()),
            "intervalo_mediano_minutos": (
                float(
                    diferencias.dt.total_seconds().median() / 60
                )
                if len(diferencias)
                else None
            ),
            "intervalos_irregulares": int(
                len(intervalos_irregulares)
            )
        }

    return detalle


# ============================================================
# EXTRACCIÓN DE OPEN-METEO
# ============================================================

def extraer_open_meteo(df):

    logging.info("Consultando Open-Meteo...")

    fecha_inicio = (
        df["timestamp"].min().strftime("%Y-%m-%d")
    )

    fecha_fin = (
        df["timestamp"].max().strftime("%Y-%m-%d")
    )

    parametros = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "start_date": fecha_inicio,
        "end_date": fecha_fin,
        "hourly": ",".join(VARIABLES_API.keys()),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
        "timeformat": "iso8601"
    }

    respuesta = crear_sesion().get(
        URL_API,
        params=parametros,
        timeout=(20, 180)
    )

    respuesta.raise_for_status()

    datos = respuesta.json()

    if datos.get("error"):
        raise RuntimeError(
            datos.get("reason", "Error en Open-Meteo.")
        )

    if "hourly" not in datos:
        raise ValueError(
            "Open-Meteo no ha devuelto datos horarios."
        )

    hourly = datos["hourly"]

    df_api = pd.DataFrame({
        "timestamp": pd.to_datetime(
            hourly["time"],
            errors="coerce",
            utc=True
        )
    })

    for variable_api, nombre_final in VARIABLES_API.items():

        if variable_api not in hourly:
            raise ValueError(
                f"No se ha recibido {variable_api}."
            )

        df_api[nombre_final] = pd.to_numeric(
            hourly[variable_api],
            errors="coerce"
        )

    if df_api["timestamp"].isna().any():
        raise ValueError(
            "Open-Meteo ha devuelto fechas inválidas."
        )

    df_api = (
        df_api.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    diferencias = df_api["timestamp"].diff().dropna()

    if (
        len(diferencias)
        and (diferencias != pd.Timedelta("1h")).any()
    ):
        logging.warning(
            "Hay intervalos horarios irregulares en Open-Meteo."
        )

    # Conservar la respuesta original.

    (CARPETA_SALIDA / "open_meteo_original.json").write_text(
        json.dumps(
            datos,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    df_api.to_csv(
        CARPETA_SALIDA / "open_meteo_horario.csv",
        index=False
    )

    metadatos = {
        "fuente": "Open-Meteo Historical Weather API",
        "endpoint": URL_API,
        "latitud": LATITUD,
        "longitud": LONGITUD,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
        "zona_horaria": datos.get("timezone"),
        "unidades": datos.get("hourly_units", {}),
        "registros_horarios": len(df_api),
        "variables": list(VARIABLES_API.values())
    }

    logging.info(
        "Registros meteorológicos descargados: %s",
        len(df_api)
    )

    return df_api, metadatos


# ============================================================
# UNIÓN DE FUENTES
# ============================================================

def fusionar_datos(df, df_api):

    logging.info("Integrando las fuentes...")

    datos = df.sort_values(
        "timestamp"
    ).copy()

    meteo = (
        df_api.rename(
            columns={
                "timestamp": "timestamp_open_meteo"
            }
        )
        .sort_values("timestamp_open_meteo")
    )

    # Se conserva el dato horario anterior o coincidente.

    final = pd.merge_asof(
        datos,
        meteo,
        left_on="timestamp",
        right_on="timestamp_open_meteo",
        direction="backward",
        tolerance=TOLERANCIA_UNION
    )

    if len(final) != len(df):
        raise ValueError(
            "La unión ha cambiado el número de filas."
        )

    final["open_meteo_age_minutes"] = (
        final["timestamp"] -
        final["timestamp_open_meteo"]
    ).dt.total_seconds() / 60

    final["open_meteo_latitude"] = LATITUD
    final["open_meteo_longitude"] = LONGITUD

    final["open_meteo_source"] = (
        "Open-Meteo Historical Weather API"
    )

    variables_meteo = list(VARIABLES_API.values())

    cobertura = (
        final[variables_meteo].notna().all(axis=1)
    )

    calidad = {
        "filas_entrada": len(df),
        "filas_salida": len(final),
        "filas_con_todas_las_variables_meteo": int(
            cobertura.sum()
        ),
        "filas_sin_todas_las_variables_meteo": int(
            (~cobertura).sum()
        ),
        "cobertura_completa_pct": round(
            float(cobertura.mean() * 100),
            2
        ),
        "direccion_union": "backward",
        "tolerancia": str(TOLERANCIA_UNION),
        "interpolacion": False
    }

    calidad["cobertura_por_estacion_pct"] = (
        final.groupby("station_name")
        .apply(
            lambda grupo: round(
                float(
                    grupo[variables_meteo]
                    .notna()
                    .all(axis=1)
                    .mean() * 100
                ),
                2
            ),
            include_groups=False
        )
        .to_dict()
    )

    return final, calidad


# ============================================================
# EXPORTACIÓN DEL DATASET DE AGOSTO
# ============================================================

def extraer_agosto(df):

    local = df["timestamp"].dt.tz_convert(
        ZONA_HORARIA
    )

    mask = (
        (local >= INICIO_AGOSTO)
        &
        (local < FIN_AGOSTO)
    )

    return df.loc[mask].copy().reset_index(drop=True)


def comprobar_agosto(df):

    calendario = pd.date_range(
        start=INICIO_AGOSTO,
        end=FIN_AGOSTO,
        freq=FRECUENCIA_ORIGINAL,
        inclusive="left"
    ).tz_convert("UTC")

    detalle = {}

    total_esperado = len(calendario) * len(ESTACIONES)
    total_encontrado = len(df)

    for station_id, nombre in ESTACIONES.items():

        grupo = df[
            df["station_id"] == station_id
        ]

        fechas = pd.DatetimeIndex(grupo["timestamp"])

        ausentes = calendario.difference(fechas)

        detalle[nombre] = {
            "registros": int(len(grupo)),
            "esperados": int(len(calendario)),
            "ausentes": int(len(ausentes))
        }

    return {
        "registros_esperados": total_esperado,
        "registros_encontrados": total_encontrado,
        "detalle_estaciones": detalle
    }


# ============================================================
# INFORME DE CALIDAD
# ============================================================

def generar_informe(
    filas_originales,
    df_limpio,
    df_final,
    calidad_etl,
    temporalidad,
    calidad_union,
    calidad_agosto
):

    lineas = [
        "INFORME DE CALIDAD - METEOGUESS",
        "=" * 55,
        "",
        "1. FUENTE ORIGINAL",
        f"Archivo: {ARCHIVO_ORIGEN}",
        f"Filas originales: {filas_originales}",
        "",
        "2. LIMPIEZA Y TRANSFORMACIÓN",
        json.dumps(
            calidad_etl,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "",
        "3. COBERTURA TEMPORAL",
        json.dumps(
            temporalidad,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "",
        "4. INTEGRACIÓN OPEN-METEO",
        json.dumps(
            calidad_union,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "",
        "5. COMPROBACIÓN DE AGOSTO 2026",
        json.dumps(
            calidad_agosto,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "",
        "6. VALORES NULOS DEL DATASET FINAL"
    ]

    nulos = (
        df_final.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    for columna, porcentaje in nulos.items():
        lineas.append(
            f"{columna}: {porcentaje:.2f}%"
        )

    lineas.extend([
        "",
        "7. CRITERIOS APLICADOS",
        "- Selección de cinco estaciones mediante station_id.",
        "- Conversión de timestamps a UTC.",
        "- Ordenación por estación y fecha.",
        "- Eliminación de id.",
        "- Duplicados por timestamp y station_id: keep='first'.",
        "- Temperatura válida entre 0 y 55 grados.",
        "- Humedad válida entre 0 y 100 por ciento.",
        "- Extracción histórica mediante Open-Meteo.",
        "- Unión temporal hacia atrás, sin interpolación.",
        "",
        "8. LIMITACIONES",
        "- Open-Meteo utiliza una única coordenada de referencia.",
        "- Las variables meteorológicas añadidas son comunes a las cinco estaciones.",
        "- Las variables horarias no representan nuevas mediciones de cinco minutos.",
        "- El dataset meteorológico no contiene por sí solo datos de generación o consumo eléctrico."
    ])

    ruta = CARPETA_SALIDA / "informe_calidad.txt"

    ruta.write_text(
        "\n".join(lineas),
        encoding="utf-8"
    )


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():

    logging.info("Iniciando ETL y enriquecimiento de Meteoguess")

    # Extracción y selección.

    original, filas_originales = cargar_datos()

    # Limpieza.

    limpio, calidad_etl = limpiar_datos(original)

    # Cobertura temporal del histórico.

    temporalidad = comprobar_temporalidad(limpio)

    # Guardar dataset limpio.

    limpio.to_csv(
        CARPETA_SALIDA / "aruba_dataset_limpio.csv",
        index=False,
        encoding="utf-8"
    )

    # Enriquecimiento meteorológico.

    df_api, metadatos_api = extraer_open_meteo(
        limpio
    )

    final, calidad_union = fusionar_datos(
        limpio,
        df_api
    )

    # Guardar dataset enriquecido completo.

    final.to_csv(
        CARPETA_SALIDA / "aruba_mix_renovable.csv",
        index=False,
        encoding="utf-8"
    )

    # Obtener también el subconjunto de agosto.

    agosto = extraer_agosto(final)

    calidad_agosto = comprobar_agosto(agosto)

    agosto.to_csv(
        CARPETA_SALIDA / "aruba_mix_renovable_agosto2026.csv",
        index=False,
        encoding="utf-8"
    )

    # Informe y metadatos.

    generar_informe(
        filas_originales,
        limpio,
        final,
        calidad_etl,
        temporalidad,
        calidad_union,
        calidad_agosto
    )

    metadatos = {
        "proyecto": "Meteoguess",
        "archivo_origen": ARCHIVO_ORIGEN,
        "estaciones": ESTACIONES,
        "variables_originales": VARIABLES,
        "variables_open_meteo": list(VARIABLES_API.values()),
        "calidad_etl": calidad_etl,
        "temporalidad": temporalidad,
        "metadatos_api": metadatos_api,
        "calidad_union": calidad_union,
        "calidad_agosto": calidad_agosto
    }

    (CARPETA_SALIDA / "metadatos.json").write_text(
        json.dumps(
            metadatos,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    logging.info("Proceso terminado correctamente")

    print("\n" + "=" * 55)
    print("RESULTADO FINAL - METEOGUESS")
    print("=" * 55)

    print("Estaciones:", len(ESTACIONES))
    print("Registros seleccionados:", calidad_etl["filas_despues_seleccion"])
    print("Duplicados eliminados:", calidad_etl["filas_eliminadas_por_duplicidad"])
    print("Registros fuera de rango:", calidad_etl["filas_fuera_rango"])
    print("Dataset limpio:", len(limpio))
    print("Dataset enriquecido:", len(final))
    print("Registros de agosto:", len(agosto))

    print(
        "Cobertura Open-Meteo:",
        calidad_union["cobertura_completa_pct"],
        "%"
    )

    print("\nArchivos generados en:", CARPETA_SALIDA)
    print("=" * 55)


if __name__ == "__main__":
    main()