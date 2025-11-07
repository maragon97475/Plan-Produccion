import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pyodbc

plantas = ["MALABRIGO", "CHIMBOTE", "SUPE", "VEGUETA", "CALLAO", "PISCO SUR"]
threshold_degradacion = 3
pozas = 10

cnxn_mio = pyodbc.connect('DRIVER={SQL Server};SERVER=srv-db-east-us003.database.windows.net;DATABASE=db_cfa_prd01;UID=userdbowner;PWD=$P4ssdbowner01#;')
cnxn_juv = pyodbc.connect('DRIVER={SQL Server};SERVER=srv-db-east-us-tasa-his-02.database.windows.net;DATABASE=db_bi_production_prd;UID=userpowerbi;PWD=#p4ssw0rdp0w3rb1#;')

def weighted_average(df, values, weights):
    return (df[values] * df[weights]).sum() / df[weights].sum()

def estimate_tvn_con_frio(tdc):
    """
    Estima el TVN de las embarcaciones con RCW.
    """
    if tdc is None:
        return None
    if tdc <= 22:
        return 0.0008 * tdc**3 - 0.0332 * tdc**2 + 0.6275 * tdc + 14
    else:
        return 14.5 * np.exp(0.0136 * tdc) + 0.7


def estimate_tvn_sin_frio(tdc):
    """
    Estima el TVN de las embarcaciones sin RCW.
    """
    if tdc is None:
        return None
    if tdc <= 20:
        return 13.573 * np.exp(0.0386 * tdc) + 0.8
    else:
        return 14.573 * np.exp(0.043 * tdc) - 4.4

def process_discharge_plan_data(df):
    """
    Process the Discharge Plan data to extract relevant information.

    Parameters:
    df (DataFrame): The input DataFrame containing Discharge Plan data.

    Returns:
    DataFrame: A DataFrame with the processed Discharge Plan data.
    """
    df["ChataLinea"] = df["Chata"] + "-" + df["Linea"]
    df = df.sort_values(by=["Fecha_Inicio_Succion"]) 
    return df

def process_limpieza_tuberia_data(df, planta_selected):

    """
    Process the Limpieza Tuberia data to extract relevant information.

    Parameters:
    df (DataFrame): The input DataFrame containing Limpieza Tuberia data.

    Returns:
    DataFrame: A DataFrame with the processed Limpieza Tuberia data.
    """
    # Filter for specific plants
    to_rename = {
        "PLANTA": "Planta",
        "FlujoIC_operacion_m3":"ValorFlujoIngreso",
        "LimpiezaTuberia_m3":"LimpiezaTuberia",
        "FactorCorreccionAguaRecirculacion":"Factor de Corrección Agua Recirculación",
        "FCSolidosGrasayAguaBombeo":"FCSolidosGrasayAguaBombeo"
    }
    df = df.rename(columns=to_rename)
    df["Planta"] = df["Planta"].str.upper()
    mask = df["Planta"].isin([planta_selected])
    return df[mask].reset_index(drop=True)

def group_intervals(intervalos_df, stock_minimo_pama):

    resultados_pp_grouped = intervalos_df.groupby(['Intervalo Chata', "Intervalo Poza", "NumHora"]).apply(
        lambda x: pd.Series({
            'VolumenTotal': x['Volumen'].sum(),
            'VolumenTotalAB': x['VolumenTotalAB'].sum(),
            'TVNPonderado': weighted_average(x, 'TVNDescarga', 'Volumen'),
            'RAPDescarga': weighted_average(x, 'RAP', 'Volumen'),
            'RAPPonderado': weighted_average(x, 'RAP_PAMA', 'Volumen'),
        })
    ).reset_index()
    # resultados_pp_grouped["Intervalo Poza"] = intervalos_df[""]
    # resultados_pp_grouped = pd.merge(resultados_pp_grouped, intervalos_df[['Intervalo Chata', 'Intervalo Poza']], left_on='Intervalo Chata', right_on="Intervalo Chata", how='left')
    resultados_pp_grouped["VolumenAcumulado"] = resultados_pp_grouped["VolumenTotal"].cumsum()
    resultados_pp_grouped["VolumenPama"] = resultados_pp_grouped["VolumenTotalAB"].cumsum()
    # resultados_pp_grouped["VolumenPama"] = resultados_pp_grouped["RAPPonderado"] * resultados_pp_grouped["VolumenAcumulado"]
    resultados_pp_grouped["flg_min"] = 0
    mask = resultados_pp_grouped["VolumenPama"] > stock_minimo_pama
    resultados_pp_grouped.loc[mask, 'flg_min'] = 1

    return resultados_pp_grouped

def process_calidades(UmbralTVN_Calidades):
    """
    Process the Calidades data to extract relevant information.

    Parameters:
    df (DataFrame): The input DataFrame containing Calidades data.

    Returns:
    DataFrame: A DataFrame with the processed Calidades data.
    """
    # Filter for specific plants
    UmbralTVN_Calidades_copy = UmbralTVN_Calidades.copy()
    to_rename = {
        "PISCO_SUR": "PISCO SUR",
    }
    UmbralTVN_Calidades_copy = UmbralTVN_Calidades_copy.rename(columns=to_rename)
    return UmbralTVN_Calidades_copy

def process_chatas_lineas_data(df, planta_selected):
    """
    Process the Chatas Lineas data to extract relevant information.

    Parameters:
    df (DataFrame): The input DataFrame containing Chatas Lineas data.

    Returns:
    DataFrame: A DataFrame with the processed Chatas Lineas data.
    """
    # Filter for specific plants
    df["id_planta"] = df["id_planta"].str.upper()
    mask = df["id_planta"].isin([planta_selected])
    df["ChataLinea"] = df["name"] + "-" + df["id_linea"]
    return df[mask].reset_index(drop=True)

def process_condiciones_rap(df, planta_selected):
    """
    Process the Chatas Lineas data to extract relevant information.

    Parameters:
    df (DataFrame): The input DataFrame containing Chatas Lineas data.

    Returns:
    DataFrame: A DataFrame with the processed Chatas Lineas data.
    """
    df["Planta"] = df["Planta"].str.upper()    
    mask = df["Planta"].isin([planta_selected])

    df["UmbralTDC_Min"] = df["UmbralTDC_Min"].fillna(df["UmbralTDC_Max"])
    df["UmbralTDC_Max"] = df["UmbralTDC_Max"].fillna(df["UmbralTDC_Min"])
    return df[mask].reset_index(drop=True)

def process_velocidades(velocidades_volumen_df, planta):
    """
    Process the Velocidades data to extract relevant information.

    Parameters:
    df (DataFrame): The input DataFrame containing Velocidades data.

    Returns:
    DataFrame: A DataFrame with the processed Velocidades data.
    """
    # Filter for specific plants
    to_rename = {
        "GRASA_MP": "GRASA MP",
        "VOLUMEN_MIN_ARRANQUE":"VOLUMEN MIN ARRANQUE",
        "MIN_TM":"MIN - TM",
        "MAX_TM":"MAX - TM",
        "VELOCIDAD_ARRANQUE":"VELOCIDAD DE ARRANQUE",
        "VELOCIDAD_MAXIMA":"VELOCIDAD MAXIMA",
        "VELOCIDAD_CIERRE":"VELOCIDAD DE CIERRE",
    }
    velocidades_volumen_df = velocidades_volumen_df.rename(columns=to_rename)
    velocidades_volumen_df["PLANTA"] = velocidades_volumen_df["PLANTA"].str.upper()
    mask = velocidades_volumen_df["PLANTA"].isin([planta])
    return velocidades_volumen_df[mask].reset_index(drop=True)

def process_balance_ksa_data(balance_ksa, planta):
    """
    Process the Balance KSA data to extract relevant information.

    Parameters:
    df (DataFrame): The input DataFrame containing Balance KSA data.

    Returns:
    DataFrame: A DataFrame with the processed Balance KSA data.
    """
    balance_ksa_df = balance_ksa.copy()
    to_rename = {
        "PorcS_Flujo_licor":"%S Flujo de licor",
        "PorcS_Flujo_lodo":"%S Flujo de lodo",
        "PorcH_Flujo_KSA":"%H Flujo de KSA",
        "PorcS_Flujo_efluente":"%S Flujo de efluente",
    }
    balance_ksa_df = balance_ksa_df.rename(columns=to_rename)
    mask = balance_ksa_df["PLANTA"] == 'Pisco'
    balance_ksa_df.loc[mask, "PLANTA"] = 'PISCO SUR'
    # Filter for specific plants
    balance_ksa_df["PLANTA"] = balance_ksa_df["PLANTA"].str.upper()
    mask = balance_ksa_df["PLANTA"].isin([planta])
    return balance_ksa_df[mask].reset_index(drop=True)

def hallar_volumen_minimo_pama(velocidades_volumen_df, grasa_h75):
    mask = (np.vectorize(eval)(str(grasa_h75) + velocidades_volumen_df["GRASA MP"]))
    stock_agua_minimo = velocidades_volumen_df.loc[mask, "VOLUMEN MIN ARRANQUE"].values[0]
    return stock_agua_minimo

def evaluar_condicion(row, df_adicional):
    condicion_tdc_min = (np.vectorize(eval)((str(row["TDC_Descarga_Actual"]) + df_adicional["UmbralTDC_Min"])))
    condicion_tdc_max = (np.vectorize(eval)((str(row["TDC_Descarga_Actual"]) + df_adicional["UmbralTDC_Max"])))
    condicion_juveniles = (np.vectorize(eval)((str(row["JuvenilesPonderado"]) + df_adicional["RangoJuvenilesPonderado"])))
    mask = (df_adicional["EstadoDeFrio"] == row["TipoEP"]) & (np.vectorize(eval)((str(row["Volumen"]) + df_adicional["RangoToneladasDeclaradas"]))) & (df_adicional["SistemaAbsorbente"] == row["sistema_absorbente"]) & condicion_tdc_min & condicion_tdc_max & condicion_juveniles
    if len(df_adicional[mask]) > 0:
        return df_adicional.loc[mask, 'RAP'].values[0]
    else:
        return df_adicional.loc[df_adicional["EstadoDeFrio"] == row["TipoEP"], 'RAP'].mean()
        
def add_rap(pd_df, chatas_lineas_df, condiciones_rap, limpieza_tuberia_df):
    """
    Add RAP to the DataFrame.
    """
    # Assuming you have a DataFrame named df with the necessary columns
    # Calculate RAP based on the conditions provided
    pd_updated_df = pd.merge(pd_df, chatas_lineas_df[["ChataLinea", "sistema_absorbente"]], on="ChataLinea", how="left")
    
    pd_updated_df["RAP"] = pd_updated_df.apply(evaluar_condicion, axis=1, df_adicional=condiciones_rap)

    volumenes = pd_updated_df.groupby("Embarcacion", as_index=False).agg(
        VolTotal=("Volumen", "sum")
    )
    pd_updated_df = pd.merge(pd_updated_df, volumenes, on="Embarcacion", how='left')
    pd_updated_df["PctVol"] = pd_updated_df["Volumen"] / pd_updated_df["VolTotal"]
    pd_updated_df["LimpiezaTuberia"] = pd_updated_df["PctVol"] * limpieza_tuberia_df["LimpiezaTuberia"].values[0]
    pd_updated_df["FactorCorreccion"] = limpieza_tuberia_df["Factor de Corrección Agua Recirculación"].values[0]
    pd_updated_df["FCSolidosGrasayAguaBombeo"] = limpieza_tuberia_df["FCSolidosGrasayAguaBombeo"].values[0]
    return pd_updated_df

def generate_intervalo_horas(pd_df, fecha_inicio_name="Fecha_Inicio_Succion", fecha_fin_name="Fecha_Fin_Succion", nombre_intervalo_chata="Intervalo Chata", tolva_inicio_name="Inicio_Descarga_PD", tolva_fin_name="Fin_Descarga_PD", nombre_intervalo_poza="Intervalo Poza", hours_adicional=20):
    """
    Generate intervals of hours for the given DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing Limpieza Tuberia data.

    Returns:
    DataFrame: A DataFrame with the generated intervals of hours.
    """
    horas_dia = pd.date_range(
        start=pd_df[fecha_inicio_name].min().floor('10min'),
        end=(pd_df[fecha_fin_name] + pd.Timedelta(hours=hours_adicional)).max().ceil("30min"),
        freq="H",
    )

    horas_dia_poza = pd.date_range(
        start=pd_df[tolva_inicio_name].min().floor('10min'),
        end=(pd_df[tolva_fin_name] + pd.Timedelta(hours=hours_adicional)).max().ceil("30min"),
        freq="H",
    )
    volumen_por_hora = pd.DataFrame(data=horas_dia, columns=[nombre_intervalo_chata])
    volumen_por_hora_poza = pd.DataFrame(data=horas_dia_poza, columns=[nombre_intervalo_poza])

    intervalos_df = pd.concat([volumen_por_hora, volumen_por_hora_poza], axis=1)
    intervalos_df["NumHora"] = np.arange(1, len(intervalos_df) + 1)
    return intervalos_df

def generar_perfil_velocidades(
    volumen: float,
    velocidades: Dict[str, float],
    duracion_paso_horas: float = 1.0
):
    """
    Genera un vector de velocidades (por hora) para cocinar todo el volumen,
    asegurando que el paso final (remanente) sea siempre menor que la velocidad de cierre.

    Parámetros:
    - volumen: Volumen total a cocinar (continuo).
    - velocidades: Diccionario con las cinco velocidades (ton/h):
        {
            'inicio': float,
            'subida': float,
            'crucero': float,
            'bajada': float,
            'cierre': float
        }
    - duracion_paso_horas: Duración de cada paso en horas (por defecto 1h).
    """
    # Convertimos a volumen procesado por paso
    v_ini = velocidades['Velocidad Inicio'] * duracion_paso_horas
    v_sub = velocidades['Velocidad Intermedia1'] * duracion_paso_horas
    v_cru = velocidades['Velocidad Maxima'] * duracion_paso_horas
    v_baj = velocidades['Velocidad Intermedia2'] * duracion_paso_horas
    v_cie = velocidades['Velocidad Final'] * duracion_paso_horas

    perfil: List[float] = []
    perfil_lbls: List[str]   = []
    rem = volumen

    # 1) Dos pasos de inicio (2h) si hay suficiente volumen
    if rem >= 2 * v_ini:
        perfil += [v_ini, v_ini]
        rem -= 2 * v_ini
        perfil_lbls += ['Velocidad Inicio', 'Velocidad Inicio']
    else:
        perfil.append(rem)
        perfil_lbls.append('Velocidad Final')
        return perfil, perfil_lbls

    # 2) Un paso de subida
    if rem >= v_sub:
        perfil.append(v_sub)
        perfil_lbls.append('Velocidad Intermedia1')
        rem -= v_sub
    else:
        perfil.append(rem)
        perfil_lbls.append('Velocidad Final')
        return perfil, perfil_lbls

    # 3) Reservamos bajada y cierre
    if rem < (v_baj + v_cie):
        # Si no alcanza para ambos completos, primero bajada parcial
        perfil.append(min(rem, v_baj))
        perfil_lbls.append('Velocidad Intermedia2')
        rem -= perfil[-1]
        if rem > 0:
            perfil.append(rem)  # remanente < v_cie necesariamente
            perfil_lbls.append('Velocidad Final')
        return perfil, perfil_lbls

    # Hay volumen suficiente para bajar y cerrar completos; los restamos para calcular crucero
    rem -= (v_baj + v_cie)

    # 4) Pasos completos de crucero
    n_cru = int(rem // v_cru)
    perfil += [v_cru] * n_cru
    perfil_lbls += ['Velocidad Maxima'] * n_cru
    rem -= n_cru * v_cru

    # 5) Pasos fijos: bajada y cierre
    perfil += [v_baj, v_cie]
    perfil_lbls += ['Velocidad Intermedia2', 'Velocidad Final']

    # 6) Ajuste de remanente para garantizar < v_cie
    if rem > 0:
        # Si rem ≥ v_cie, desglosamos en cierres completos + rem final
        if rem >= v_cie:
            n_cie_extra = int(rem // v_cie)
            perfil += [v_cie] * n_cie_extra
            perfil_lbls += ['Velocidad Final'] * n_cie_extra
            rem -= n_cie_extra * v_cie
        # Al final, rem < v_cie
        if rem > 0:
            perfil.append(rem)
            perfil_lbls.append('Velocidad Final')

    return perfil, perfil_lbls


def process_valores_fijos(ValoresFijosPP, planta):
    """
    Process the Valores Fijos data to extract relevant information.

    Parameters:
    df (DataFrame): The input DataFrame containing Valores Fijos data.

    Returns:
    DataFrame: A DataFrame with the processed Valores Fijos data.
    """
    to_rename = {
        "PROTEINA_HARINA": "PROTEINA-HARINA",
        "Ceniza_KK_PRENSA":"%Ceniza- KK PRENSA",
        "Grasa_KK_PRENSA":"%Grasa- KK PRENSA",
        "Humedad_KK_PRENSA":"%Humedad- KK PRENSA",
        "Proteina_KK_PRENSA":"% Proteina- KK PRENSA",
        "Grasa_KK_SEPARADORA":"%Grasa- KK SEPARADORA",
        "Ceniza_KK_SEPARADORA":"%Ceniza- KK SEPARADORA",
        "Proteina_KK_SEPARADORA":"% Proteina- KK SEPARADORA",
        "Humedad_KK_SEPARADORA":"%Humedad- KK SEPARADORA",
        "Grasa_CONCENTRADO":"%Grasa- CONCENTRADO",
        "Ceniza_CONCENTRADO":"%Ceniza- CONCENTRADO",
        "Proteina_CONCENTRADO":"% Proteina-CONCENTRADO",
        "Humedad_CONCENTRADO":"%Humedad- CONCENTRADO",
        "Ceniza_KSA":"%Ceniza- KSA",
        "Proteina_KSA":"% Proteina- KSA",
        "Humedad_KSA":"%Humedad- KSA",
    }
    ValoresFijosPP_copy = ValoresFijosPP.copy()
    ValoresFijosPP_copy = ValoresFijosPP_copy.rename(columns=to_rename)
    ValoresFijosPP_copy["PLANTA"] = ValoresFijosPP_copy["PLANTA"].str.upper()
    mask = ValoresFijosPP_copy["PLANTA"].isin([planta])
    return ValoresFijosPP_copy[mask].reset_index(drop=True)

def process_arranque(df, planta, fecha_batch):
    arranque_df = df.copy()
    to_rename = {
        "Planta":"PLANTA"
    }
    arranque_df = arranque_df.rename(to_rename, axis=1)
    arranque_df["PLANTA"] = arranque_df["PLANTA"].str.upper()
    mask = (arranque_df["PLANTA"].isin([planta])) & (arranque_df["FechaBatch"].astype(str) == fecha_batch)
    return arranque_df.loc[mask, "FechaAjustada"]

def hallar_velocidades_base(velocidades_volumen_df, grasa_h75, volumen_materia_prima):
    mask = (np.vectorize(eval)(str(grasa_h75) + velocidades_volumen_df["GRASA MP"])) & (velocidades_volumen_df["MIN - TM"] <= volumen_materia_prima) & (velocidades_volumen_df["MAX - TM"] >= volumen_materia_prima)
    v_inicial = velocidades_volumen_df.loc[mask, "VELOCIDAD DE ARRANQUE"].values[0]
    v_medio = velocidades_volumen_df.loc[mask, "VELOCIDAD MAXIMA"].values[0]
    v_intermedia_inicial =  velocidades_volumen_df.loc[mask, "VEL_INTERMEDIA1"].values[0]
    v_final = velocidades_volumen_df.loc[mask, "VELOCIDAD DE CIERRE"].values[0]
    v_intermedia_final = velocidades_volumen_df.loc[mask, "VEL_INTERMEDIA2"].values[0]
    return v_inicial, v_intermedia_inicial, v_medio, v_intermedia_final, v_final

def fill_intervals(intervalos_df, plan_descarga_df, cantidad_pozas=10, volumen_por_poza=250):

    intervalos_df_copy = intervalos_df.copy()
    for i in range(cantidad_pozas):
        intervalos_df_copy[f"Poza_{i+1}"] = 0
    plan_descarga_df_copy = plan_descarga_df.copy()
    for index, row in intervalos_df_copy.iterrows():
        start1 = row['Intervalo Chata']
        end1 = start1 + pd.Timedelta(hours=1)
        # print(index)
        start2 = plan_descarga_df_copy['Fecha_Inicio_Succion'].item()
        end2 = plan_descarga_df_copy['Fecha_Fin_Succion'].item()
        tdc = plan_descarga_df_copy['TDC_Descarga'].item()
        volumen_estimado = plan_descarga_df_copy['VolumenEstTM'].item()
        # velocidad = plan_descarga_df_copy['Velocidad'].item()
        velocidad = volumen_estimado / ((end2 - start2).total_seconds() / 3600)
        chatalinea = plan_descarga_df_copy['ChataLinea'].item()
        ep = plan_descarga_df_copy['Embarcacion'].item()
        tipoep = plan_descarga_df_copy['TIPO'].item()
        juveniles_ponderado = plan_descarga_df_copy['JuvenilesPonderado'].item()
        interseccion_inicio = max(start2, start1)
        interseccion_fin = min(end2, end1)
        
        # Calcular la duración de la intersección
        duracion_interseccion = max((interseccion_fin - interseccion_inicio).total_seconds(), 0)
        
        # Calcular la duración del rango original
        duracion_rango_original = (end1 - start1).total_seconds()
        
        # Calcular el porcentaje
        porcentaje = (duracion_interseccion / duracion_rango_original) if duracion_rango_original > 0 else 0

        intervalos_df_copy.at[index, 'Duracion'] = porcentaje
        intervalos_df_copy.at[index, 'Volumen'] = porcentaje * velocidad
        intervalos_df_copy.at[index, 'VelDesc'] = velocidad
        intervalos_df_copy.at[index, 'ChataLinea'] = chatalinea
        intervalos_df_copy.at[index, 'Embarcacion'] = ep
        intervalos_df_copy.at[index, 'TipoEP'] = tipoep
        intervalos_df_copy.at[index, 'TDC_Descarga_Inicial'] = tdc
        tvn_estimado = estimate_tvn_sin_frio(tdc) if tipoep == 'SF' else estimate_tvn_con_frio(tdc)
        intervalos_df_copy.at[index, 'TVN_Descarga_Inicial'] = tvn_estimado
        intervalos_df_copy.at[index, 'JuvenilesPonderado'] = juveniles_ponderado

        # intervalos_df_copy = almacenar_en_poza(intervalos_df_copy, tvn_estimado, porcentaje * velocidad)
    return intervalos_df_copy
    
class ParametrosSistema:
    """Parámetros constantes del sistema completo"""
    
    # Materia Prima - Pescado
    PESCADO_HUMEDAD = 75
    PESCADO_CENIZAS = 4
    
    # Materia Prima - Camaroncillo
    CAMARONCILLO_HUMEDAD = 82.47
    CAMARONCILLO_GRASA = 3.19
    CAMARONCILLO_CENIZAS = 4.65
    
    # Materia Prima - Escama
    ESCAMA_HUMEDAD = 89
    ESCAMA_GRASA = 0.19
    ESCAMA_CENIZAS = 9
    
    # Harina
    HARINA_HUMEDAD = 7
    HARINA_CENIZA = 0.07
    HARINA_GRASA = 0.08
    HARINA_PROTEINA = 0.65
    
    # KK Prensa
    KK_PRENSA_HUMEDAD = 0.42
    KK_PRENSA_CENIZA = 0.115
    KK_PRENSA_GRASA = 0.038
    KK_PRENSA_PROTEINA = 1 - 0.42 - 0.115 - 0.038
    
    # KK Separadora
    KK_SEP_HUMEDAD = 0.62
    KK_SEP_CENIZA = 0.04
    KK_SEP_GRASA = 0.025
    KK_SEP_PROTEINA = 1 - 0.62 - 0.04 - 0.025
    
    # KK CC
    KK_CC_HUMEDAD = 0.654
    KK_CC_CENIZA = 0.066
    KK_CC_GRASA = 0.035
    KK_CC_PROTEINA = 1 - 0.654 - 0.066 - 0.035
    
    # KSA
    KSA_HUMEDAD = 0.75
    KSA_CENIZA = 0.07
    
    # Porcentajes de receta por calidad
    PORCENTAJES_RECETA = {
        'A': 0.06,
        'B': 0.07,
        'C': 0.10,
        'D': 0.10
    }
    
    def get_porcentaje_receta(self, calidad: str) -> float:
        return self.PORCENTAJES_RECETA.get(calidad, 0.06)

class BalanceGeneralVectorizadoCompleto:
    def __init__(self):
        self.data = {}
        self._longitud_principal = 0
        self._parametros = ParametrosSistema()
        
    def _obtener_longitud_principal(self) -> int:
        """Obtiene la longitud del vector principal para referencia"""
        if 'intervalo_chata' in self.data and len(self.data['intervalo_chata']) > 0:
            return len(self.data['intervalo_chata'])
        return 0
    
    def _crear_vector_vacio(self, longitud: int, dtype=object) -> np.ndarray:
        """Crea un vector vacío del tamaño adecuado con NaN"""
        if dtype == object:
            return np.full(longitud, None, dtype=object)
        else:
            return np.full(longitud, np.nan, dtype=dtype)
    
    def _asegurar_longitud_vector(self, nombre_vector: str, longitud: int, dtype=object):
        """Asegura que un vector tenga la longitud correcta, creándolo si no existe"""
        if nombre_vector not in self.data:
            self.data[nombre_vector] = self._crear_vector_vacio(longitud, dtype)
        elif len(self.data[nombre_vector]) < longitud:
            vector_actual = self.data[nombre_vector]
            extension_dtype = vector_actual.dtype if vector_actual.dtype != object else dtype
            nan_extension = self._crear_vector_vacio(longitud - len(vector_actual), extension_dtype)
            self.data[nombre_vector] = np.concatenate([vector_actual, nan_extension])
        elif len(self.data[nombre_vector]) > longitud:
            self.data[nombre_vector] = self.data[nombre_vector][:longitud]
    
    def _asignar_datos_progresivos(self, columna: str, datos: list, inicio: int):
        """Asigna datos progresivamente a una columna manejando diferentes tipos"""
        if columna not in self.data:
            return
        
        longitud = len(self.data[columna])
        datos_a_asignar = min(len(datos), longitud - inicio)
        
        if datos_a_asignar > 0:
            try:
                datos_array = np.array(datos[:datos_a_asignar], dtype=self.data[columna].dtype)
                self.data[columna][inicio:inicio+datos_a_asignar] = datos_array
            except (ValueError, TypeError):
                for i in range(datos_a_asignar):
                    if inicio + i < longitud:
                        self.data[columna][inicio + i] = datos[i]
    
    def cargar_datos_completos_desde_excel(self, datos_excel: Dict[str, np.ndarray]):
        """Carga todos los datos """
        for key, value in datos_excel.items():
            self.data[key] = value
        
        self._longitud_principal = self._obtener_longitud_principal()
        
        for key in list(self.data.keys()):
            dtype_actual = self.data[key].dtype
            self._asegurar_longitud_vector(key, self._longitud_principal, dtype_actual)
    
    def inicializar_vectores_proceso(self):
        """Inicializa todos los vectores del proceso"""
        longitud = self._longitud_principal
        if longitud == 0:
            return
        
        # DEFINIR TODAS LAS COLUMNAS
        todas_las_columnas = [
            # Columnas básicas
            'intervalo_chata', 'intervalo_poza', 'descarga', 'avance_descarga',
            'grasa_mp', 'porcentaje_camaroncillo', 'velocidad',
            'velocidad_corregida_sa', 'velocidad_corregida_clarificador', 
            'velocidad_corregida_dosificacion',
            
            # Materia Prima - Pescado
            'pescado_tn2', 'pescado_hum_tn', 'pescado_porc_hum', 'pescado_gra_tn',
            'pescado_porc_gra', 'pescado_cen_tn', 'pescado_porc_cen',
            
            # Materia Prima - Camaroncillo
            'camaroncillo_tn', 'camaroncillo_hum_tn', 'camaroncillo_porc_hum',
            'camaroncillo_gra_tn', 'camaroncillo_porc_gra', 'camaroncillo_cen_tn',
            'camaroncillo_porc_cen',
            
            # Materia Prima - Escama
            'escama_tn', 'escama_hum_tn', 'escama_porc_hum', 'escama_gra_tn',
            'escama_porc_gra', 'escama_cen_tn', 'escama_porc_cen',
            
            # Materia Prima IC
            'mp_ic_tn', 'mp_ic_hum_tn', 'mp_ic_porc_hum', 'mp_ic_gra_tn',
            'mp_ic_porc_gra', 'mp_ic_cen_tn', 'mp_ic_porc_cen',
            
            # Acumulados y Stock
            'acum_alimentacion', 'acum_alimentacion_corregido_pama',
            'stock_pozas', 'stock_pozas_corregido_pama',
            
            # TVN y Sólidos
            'tvn_descarga', 'solidos_ab', 'grasa_ab', 'tvn_pozas',
            'tvn_pozas_corregido', 'tvn_ingreso_cocinas', 'tvn_ingreso_cocinas_corregido',
            'solidos_ab_ic', 'grasa_ab_ic', 'solidos_ab_ic_corregido_pama',
            'grasa_ab_corregido_pama',
            
            # RAP PAMA
            'rap_pama_desc', 'rap_pama',
            
            # Volúmenes generados
            'volumen_generado_ab_acumulado',
            
            # Calidad y Receta
            'calidad_harina', 'porcentaje_receta',
            
            # Flujos Teóricos
            'flujo_ksa_teorico', 'base_seca_teorica', 'flujo_ic_teorico',
            
            # Factores de Corrección
            'fc', 'flujo_ic_corregido', 'flujo_ksa_corregido', 'base_seca_corregida',
            'receta_corregida',
            
            # Límites de capacidad
            'limite_base_seca', 'limite_flujo_ic',
            
            # Correcciones por capacidad
            'fc2', 'base_seca_corregida_sa', 'flujo_ksa_corregido_sa',
            'flujo_ic_corregido_sa', 'receta_corregida_sa',
            
            # Correcciones por clarificador
            'fc3', 'flujo_ic_capacidad_clarificador', 
            'base_seca_capacidad_clarificador', 'flujo_ksa_capacidad_clarificador',
            
            # Correcciones por dosificación
            'receta_corregida_dosificacion', 'base_seca_corregida_dosificacion',
            'flujo_ksa_corregido_dosificacion', 'flujo_ic_corregido_dosificacion',
            
            # Clarificador
            'porcentaje_grasa_ingreso_clarificador', 'flujo_efluente',
            'porcentaje_grasa_efluente', 'porcentaje_solidos_efluente',
            'flujo_lodo', 'porcentaje_grasa_lodo', 'porcentaje_solidos_lodo',
            'flujo_licor', 'porcentaje_grasa_licor', 'porcentaje_solidos_licor',
            
            # Flujo KSA
            'flujo_ksa', 'porcentaje_grasa_ksa', 'porcentaje_solidos_ksa',
            'porcentaje_humedad_ksa',
            
            # TVN A
            # 'tvn_a',
            
            # KK Prensa
            'kk_prensa_tn', 'kk_prensa_porc_humedad', 'kk_prensa_hum_tn',
            'kk_prensa_porc_ceniza', 'kk_prensa_cen_tn', 'kk_prensa_porc_grasa',
            'kk_prensa_gra_tn', 'kk_prensa_porc_proteina', 'kk_prensa_pro_tn',
            
            # KK Separadora
            'kk_separadora_tn', 'kk_separadora_porc_humedad', 'kk_sep_hum_tn',
            'kk_separadora_porc_ceniza', 'kk_sep_cen_tn', 'kk_separadora_porc_grasa',
            'kk_sep_gra_tn', 'kk_separadora_porc_proteina', 'kk_sep_pro_tn',
            
            # KK CC
            'kk_cc_tn', 'kk_cc_porc_humedad_2', 'kk_cc_porc_humedad', 'kk_cc_hum_tn',
            'kk_cc_porc_ceniza_2', 'kk_cc_porc_ceniza', 'kk_cc_cen_tn',
            'kk_cc_porc_grasa', 'kk_cc_gra_tn', 'kk_cc_porc_proteina', 'kk_cc_pro_tn',
            
            # KSA
            'ksa_tn', 'ksa_porc_humedad', 'ksa_hum_tn', 'ksa_porc_ceniza',
            'ksa_cen_tn', 'ksa_porc_grasa', 'ksa_gra_tn', 'ksa_porc_proteina',
            'ksa_pro_tn',
            
            # Dosificación KSA
            'porcentaje_dosificacion_ksa2',
            
            # KK Integral
            'kk_integral_tn', 'kk_integral_hum_tn', 'kk_integral_cen_tn',
            'kk_integral_gra_tn', 'kk_integral_pro_tn', 'kk_integral_porc_humedad',
            'kk_integral_porc_ceniza', 'kk_integral_porc_grasa', 'kk_integral_porc_proteina',
            
            # Aportantes
            'aportante_porc_kp', 'aportante_porc_ks', 'aportante_porc_cc',
            
            # Harina
            'harina_porc_humedad', 'harina_porc_ceniza', 'harina_porc_grasa',
            'harina_porc_proteina', 'harina_porc_humedad_2', 'harina_porc_ceniza_2',
            'harina_porc_grasa_2', 'harina_porc_proteina_2',
            
            # Resultados finales
            'har_ton', 'rendimiento_harina', 'aceite_tn', 'rendimiento_aceite'
        ]
        
        # Inicializar todas las columnas
        for columna in todas_las_columnas:
            if columna not in self.data:
                # Determinar tipo basado en el nombre
                if any(x in columna.lower() for x in ['intervalo', 'calidad']):
                    dtype = object
                else:
                    dtype = float
                self._asegurar_longitud_vector(columna, longitud, dtype)
    
    def simular_progreso_completo(self):
        """Simula el progreso completo del proceso"""
        longitud = self._longitud_principal
        if longitud == 0:
            return
        
        inicio_proceso = 2  # Los datos empiezan alrededor de la fila 3
        
        if longitud <= inicio_proceso:
            return
        
        # COLUMNAS CONSTANTES
        if 'grasa_mp' in self.data:
            self.data['grasa_mp'][inicio_proceso:] = 3.0
        
        if 'porcentaje_camaroncillo' in self.data:
            self.data['porcentaje_camaroncillo'][inicio_proceso:] = 2.0
        
        # VELOCIDAD - empieza vacía, luego se llena progresivamente
        velocidades = [120.0, 120.0, 226.0, 226.0, 226.0, 210.0, 210.0, 210.0, 80.0, 80.0, 40.0, np.nan, np.nan]
        # velocidades = 
        self._asignar_datos_progresivos('velocidad', velocidades, inicio_proceso)
        
        # VELOCIDADES CORREGIDAS - siguen el mismo patrón que velocidad
        for columna in ['velocidad_corregida_sa', 'velocidad_corregida_clarificador', 
                       'velocidad_corregida_dosificacion']:
            if columna in self.data:
                datos_velocidad = [x if not pd.isna(x) else np.nan for x in self.data['velocidad'][inicio_proceso:inicio_proceso+len(velocidades)]]
                self._asignar_datos_progresivos(columna, datos_velocidad, inicio_proceso)
        
        # TVN DESCARGA - datos progresivos
        tvn_descarga = [21.88, 21.88, 22.23, 22.25, 22.03, 23.03, 22.69, 23.03, 22.69, 23.15]
        self._asignar_datos_progresivos('tvn_descarga', tvn_descarga, inicio_proceso)

        # TVN POZAS - datos progresivos
        tvn_pozas = [21.88, 21.88, 22.23, 22.25, 22.03, 23.03, 22.69, 23.03, 22.69, 23.15]
        self._asignar_datos_progresivos('tvn_pozas', tvn_pozas, inicio_proceso)

        # TVN POZAS CORREGIDO - datos progresivos
        tvn_pozas_corregido = [21.88, 21.88, 22.23, 22.25, 22.03, 23.03, 22.69, 23.03, 22.69, 23.15]
        self._asignar_datos_progresivos('tvn_pozas_corregido', tvn_pozas_corregido, inicio_proceso)
        
        # TVN INGRESO COCINAS - similar a TVN descarga
        tvn_ingreso = [21.88, 22.33, 21.98, 22.15, 22.03, 22.15, 22.69, 23.15, 22.69, 23.15]
        self._asignar_datos_progresivos('tvn_ingreso_cocinas', tvn_ingreso, inicio_proceso)
        self._asignar_datos_progresivos('tvn_ingreso_cocinas_corregido', tvn_ingreso, inicio_proceso)
        
        # CALIDAD HARINA - valores categóricos (strings)
        calidades = ['A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'D', 'D', 'D', 'D']
        self._asignar_datos_progresivos('calidad_harina', calidades, inicio_proceso)
        
        # PORCENTAJE RECETA - basado en calidad harina
        if 'calidad_harina' in self.data and 'porcentaje_receta' in self.data:
            for i in range(inicio_proceso, min(longitud, inicio_proceso + len(calidades))):
                if (i < len(self.data['calidad_harina']) and 
                    self.data['calidad_harina'][i] is not None and 
                    not pd.isna(self.data['calidad_harina'][i])):
                    
                    calidad = self.data['calidad_harina'][i]
                    porcentaje = self._parametros.get_porcentaje_receta(calidad)
                    if i < len(self.data['porcentaje_receta']):
                        self.data['porcentaje_receta'][i] = porcentaje
        
        # ACUMULADOS - se van acumulando
        if 'descarga' in self.data and 'acum_alimentacion' in self.data:
            descargas_validas = []
            indices_validos = []
            
            for i in range(inicio_proceso, min(longitud, len(self.data['velocidad_corregida_dosificacion']))):
                if (not pd.isna(self.data['velocidad_corregida_dosificacion'][i]) and 
                    self.data['velocidad_corregida_dosificacion'][i] is not None):
                    descargas_validas.append(float(self.data['velocidad_corregida_dosificacion'][i]))
                    indices_validos.append(i)
            
            if descargas_validas:
                acumulados = np.cumsum(descargas_validas)
                for idx, acum in zip(indices_validos, acumulados):
                    if idx < len(self.data['acum_alimentacion']):
                        self.data['acum_alimentacion'][idx] = acum
                        self.data['acum_alimentacion_corregido_pama'][idx] = acum # ´TODO: Cambiar a valor corregido si es necesario
                        self.data['stock_pozas'][idx] = self.data['avance_descarga'][idx] - acum
                        self.data['stock_pozas_corregido_pama'][idx] = self.data['avance_descarga'][idx] - acum
            
        
        # CONSTANTES DEL SISTEMA
        constantes = {
            'rap_pama_desc': 1.84,
            'rap_pama': 1.84,
            'limite_base_seca': 4.5,
            'limite_flujo_ic': 550.0,
            'fc': 1.0,
            'fc2': 1.0,
            'fc3': 550.0/597.62,
            'aportante_porc_kp': 23.0,
            'aportante_porc_ks': 8.0,
            'aportante_porc_cc': 14.0,
            'harina_porc_humedad': 7.0,
            'harina_porc_ceniza': 0.07,
            'harina_porc_grasa': 0.08,
            'harina_porc_proteina': 0.65
        }
        
        for columna, valor in constantes.items():
            if columna in self.data:
                # Asignar a todas las filas desde inicio_proceso
                for i in range(inicio_proceso, longitud):
                    if i < len(self.data[columna]):
                        self.data[columna][i] = valor

    # CALCULAR VECTORES DE VOLUMEN GENERADO AB ACUMULADO TODO:
        for i in range(inicio_proceso, longitud):
            # if i < len(self.data[columna]):
                self.data["volumen_generado_ab_acumulado"][i] = self.data["rap_pama"][i] * self.data["avance_descarga"][i]
        
    def calcular_vectores_materia_prima(self):
        """Calcula todos los vectores de materia prima de manera segura"""
        longitud = self._longitud_principal
        if longitud == 0:
            return
        
        # Usar enfoque iterativo para evitar problemas de máscaras
        for i in range(longitud):
            if (i < len(self.data['velocidad_corregida_dosificacion']) and 
                not pd.isna(self.data['velocidad_corregida_dosificacion'][i]) and 
                self.data['velocidad_corregida_dosificacion'][i] > 0 and
                i < len(self.data['grasa_mp']) and
                not pd.isna(self.data['grasa_mp'][i]) and
                i < len(self.data['porcentaje_camaroncillo']) and
                not pd.isna(self.data['porcentaje_camaroncillo'][i])):
                
                try:
                    velocidad = float(self.data['velocidad_corregida_dosificacion'][i])
                    grasa_mp = float(self.data['grasa_mp'][i])
                    porc_camaroncillo = float(self.data['porcentaje_camaroncillo'][i])
                    
                    # PESCADO
                    if i < len(self.data['pescado_tn2']):
                        self.data['pescado_tn2'][i] = velocidad
                    
                    if i < len(self.data['pescado_hum_tn']):
                        self.data['pescado_hum_tn'][i] = velocidad * self._parametros.PESCADO_HUMEDAD / 100
                    
                    if i < len(self.data['pescado_gra_tn']):
                        self.data['pescado_gra_tn'][i] = velocidad * grasa_mp / 100
                    
                    if i < len(self.data['pescado_cen_tn']):
                        self.data['pescado_cen_tn'][i] = velocidad * self._parametros.PESCADO_CENIZAS / 100
                    
                    # Porcentajes Pescado
                    if i < len(self.data['pescado_porc_hum']):
                        self.data['pescado_porc_hum'][i] = self._parametros.PESCADO_HUMEDAD
                    
                    if i < len(self.data['pescado_porc_gra']):
                        self.data['pescado_porc_gra'][i] = grasa_mp
                    
                    if i < len(self.data['pescado_porc_cen']):
                        self.data['pescado_porc_cen'][i] = self._parametros.PESCADO_CENIZAS
                    
                    # CAMARONCILLO
                    camaroncillo_tn = velocidad * porc_camaroncillo / 100
                    
                    if i < len(self.data['camaroncillo_tn']):
                        self.data['camaroncillo_tn'][i] = camaroncillo_tn
                    
                    if i < len(self.data['camaroncillo_hum_tn']):
                        self.data['camaroncillo_hum_tn'][i] = camaroncillo_tn * self._parametros.CAMARONCILLO_HUMEDAD / 100
                    
                    if i < len(self.data['camaroncillo_gra_tn']):
                        self.data['camaroncillo_gra_tn'][i] = camaroncillo_tn * self._parametros.CAMARONCILLO_GRASA / 100
                    
                    if i < len(self.data['camaroncillo_cen_tn']):
                        self.data['camaroncillo_cen_tn'][i] = camaroncillo_tn * self._parametros.CAMARONCILLO_CENIZAS / 100
                    
                    # Porcentajes Camaroncillo
                    if i < len(self.data['camaroncillo_porc_hum']):
                        self.data['camaroncillo_porc_hum'][i] = self._parametros.CAMARONCILLO_HUMEDAD
                    
                    if i < len(self.data['camaroncillo_porc_gra']):
                        self.data['camaroncillo_porc_gra'][i] = self._parametros.CAMARONCILLO_GRASA
                    
                    if i < len(self.data['camaroncillo_porc_cen']):
                        self.data['camaroncillo_porc_cen'][i] = self._parametros.CAMARONCILLO_CENIZAS
                    
                    # ESCAMA
                    escama_tn = 0.02 * (camaroncillo_tn + velocidad)
                    
                    if i < len(self.data['escama_tn']):
                        self.data['escama_tn'][i] = escama_tn
                    
                    if i < len(self.data['escama_hum_tn']):
                        self.data['escama_hum_tn'][i] = escama_tn * self._parametros.ESCAMA_HUMEDAD / 100
                    
                    if i < len(self.data['escama_gra_tn']):
                        self.data['escama_gra_tn'][i] = escama_tn * self._parametros.ESCAMA_GRASA / 100
                    
                    if i < len(self.data['escama_cen_tn']):
                        self.data['escama_cen_tn'][i] = escama_tn * self._parametros.ESCAMA_CENIZAS / 100
                    
                    # Porcentajes Escama
                    if i < len(self.data['escama_porc_hum']):
                        self.data['escama_porc_hum'][i] = self._parametros.ESCAMA_HUMEDAD
                    
                    if i < len(self.data['escama_porc_gra']):
                        self.data['escama_porc_gra'][i] = self._parametros.ESCAMA_GRASA
                    
                    if i < len(self.data['escama_porc_cen']):
                        self.data['escama_porc_cen'][i] = self._parametros.ESCAMA_CENIZAS
                    
                    # MATERIA PRIMA INGRESO COCINAS
                    mp_ic_tn = velocidad + camaroncillo_tn + escama_tn
                    
                    if i < len(self.data['mp_ic_tn']):
                        self.data['mp_ic_tn'][i] = mp_ic_tn
                    
                    if i < len(self.data['mp_ic_hum_tn']):
                        self.data['mp_ic_hum_tn'][i] = (self.data['pescado_hum_tn'][i] + 
                                                      self.data['camaroncillo_hum_tn'][i] + 
                                                      self.data['escama_hum_tn'][i])
                    
                    if i < len(self.data['mp_ic_gra_tn']):
                        self.data['mp_ic_gra_tn'][i] = (self.data['pescado_gra_tn'][i] + 
                                                      self.data['camaroncillo_gra_tn'][i] + 
                                                      self.data['escama_gra_tn'][i])
                    
                    if i < len(self.data['mp_ic_cen_tn']):
                        self.data['mp_ic_cen_tn'][i] = (self.data['pescado_cen_tn'][i] + 
                                                      self.data['camaroncillo_cen_tn'][i] + 
                                                      self.data['escama_cen_tn'][i])
                    
                    # PORCENTAJES MP IC
                    if mp_ic_tn > 0:
                        if i < len(self.data['mp_ic_porc_hum']):
                            self.data['mp_ic_porc_hum'][i] = (self.data['mp_ic_hum_tn'][i] / mp_ic_tn) * 100
                        
                        if i < len(self.data['mp_ic_porc_gra']):
                            self.data['mp_ic_porc_gra'][i] = (self.data['mp_ic_gra_tn'][i] / mp_ic_tn) * 100
                        
                        if i < len(self.data['mp_ic_porc_cen']):
                            self.data['mp_ic_porc_cen'][i] = (self.data['mp_ic_cen_tn'][i] / mp_ic_tn) * 100
                            
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    # Silenciosamente continuar con el siguiente índice
                    print("ERROR en cálculo de materia prima en índice {}: {}".format(i, e))
                    continue

    def calcular_vectores_proceso(self):
        """Calcula vectores del proceso principal de manera segura"""
        longitud = self._longitud_principal
        if longitud == 0:
            return
        
        for i in range(longitud):
            try:
                # SÓLIDOS AB
                if (i < len(self.data['grasa_mp']) and 
                    i < len(self.data['tvn_descarga']) and
                    not pd.isna(self.data['grasa_mp'][i]) and
                    not pd.isna(self.data['tvn_descarga'][i])):
                    
                    tvn = float(self.data['tvn_descarga'][i])
                    grasa = float(self.data['grasa_mp'][i])
                    
                    solidos_ab = (-445.24 + tvn * 294 - 606.62 * grasa + 88.804 * grasa**2)
                    grasa_ab = -47.169 + 0.943 * solidos_ab
                    
                    if i < len(self.data['solidos_ab']):
                        self.data['solidos_ab'][i] = solidos_ab
                    if i < len(self.data['grasa_ab']):
                        self.data['grasa_ab'][i] = grasa_ab
                
                # SÓLIDOS AB IC
                if (i < len(self.data['grasa_mp']) and 
                    i < len(self.data['tvn_ingreso_cocinas']) and
                    not pd.isna(self.data['grasa_mp'][i]) and
                    not pd.isna(self.data['tvn_ingreso_cocinas'][i])):
                    
                    tvn_ic = float(self.data['tvn_ingreso_cocinas'][i])
                    grasa = float(self.data['grasa_mp'][i])
                    
                    solidos_ab_ic = (-445.24 + tvn_ic * 294 - 606.62 * grasa + 88.804 * grasa**2)
                    grasa_ab_ic = -47.169 + 0.943 * solidos_ab_ic
                    
                    if i < len(self.data['solidos_ab_ic']):
                        self.data['solidos_ab_ic'][i] = solidos_ab_ic
                    if i < len(self.data['grasa_ab_ic']):
                        self.data['grasa_ab_ic'][i] = grasa_ab_ic
                
                # SÓLIDOS AB IC CORREGIDO
                if (i < len(self.data['grasa_mp']) and 
                    i < len(self.data['tvn_ingreso_cocinas_corregido']) and
                    not pd.isna(self.data['grasa_mp'][i]) and
                    not pd.isna(self.data['tvn_ingreso_cocinas_corregido'][i])):
                    
                    tvn_ic_corr = float(self.data['tvn_ingreso_cocinas_corregido'][i])
                    grasa = float(self.data['grasa_mp'][i])
                    
                    solidos_ab_ic_corr = (-445.24 + tvn_ic_corr * 294 - 606.62 * grasa + 88.804 * grasa**2)
                    grasa_ab_corr = -47.169 + 0.943 * solidos_ab_ic_corr
                    
                    if i < len(self.data['solidos_ab_ic_corregido_pama']):
                        self.data['solidos_ab_ic_corregido_pama'][i] = solidos_ab_ic_corr
                    if i < len(self.data['grasa_ab_corregido_pama']):
                        self.data['grasa_ab_corregido_pama'][i] = grasa_ab_corr
                        
            except (TypeError, ValueError) as e:
                continue

    def calcular_vectores_flujos(self):
        """Calcula vectores de flujos de manera segura"""
        longitud = self._longitud_principal
        if longitud == 0:
            return
        
        for i in range(longitud):
            try:
                if (i < len(self.data['velocidad']) and 
                    i < len(self.data['porcentaje_receta']) and
                    i < len(self.data['solidos_ab_ic']) and
                    not pd.isna(self.data['velocidad'][i]) and
                    self.data['velocidad'][i] > 0 and
                    not pd.isna(self.data['porcentaje_receta'][i]) and
                    not pd.isna(self.data['solidos_ab_ic'][i])):
                    
                    velocidad = float(self.data['velocidad'][i])
                    porcentaje_receta = float(self.data['porcentaje_receta'][i])
                    solidos_ab_ic = float(self.data['solidos_ab_ic'][i])
                    fc = float(self.data['fc'][i]) if (i < len(self.data['fc']) and not pd.isna(self.data['fc'][i])) else 1.0
                    
                    # FLUJOS TEÓRICOS
                    flujo_ksa_teorico = porcentaje_receta * velocidad
                    base_seca_teorica = flujo_ksa_teorico / 4
                    
                    if i < len(self.data['flujo_ksa_teorico']):
                        self.data['flujo_ksa_teorico'][i] = flujo_ksa_teorico
                    if i < len(self.data['base_seca_teorica']):
                        self.data['base_seca_teorica'][i] = base_seca_teorica
                    
                    # FLUJO IC TEÓRICO
                    if solidos_ab_ic > 0:
                        flujo_ic_teorico = (base_seca_teorica * 1000000) / solidos_ab_ic
                        if i < len(self.data['flujo_ic_teorico']):
                            self.data['flujo_ic_teorico'][i] = flujo_ic_teorico
                        
                        # FLUJOS CORREGIDOS
                        flujo_ic_corregido = flujo_ic_teorico * fc
                        if i < len(self.data['flujo_ic_corregido']):
                            self.data['flujo_ic_corregido'][i] = flujo_ic_corregido
                        
                        # FLUJO KSA CORREGIDO
                        flujo_ksa_corregido = (flujo_ic_corregido * solidos_ab_ic * 4) / 1000000
                        base_seca_corregida = flujo_ksa_corregido / 4
                        receta_corregida = flujo_ksa_corregido / velocidad
                        
                        # FLUJOS CORREGIDOS PAMA                        
                        if i < len(self.data['flujo_ksa_corregido']):
                            self.data['flujo_ksa_corregido'][i] = flujo_ksa_corregido
                        if i < len(self.data['base_seca_corregida']):
                            self.data['base_seca_corregida'][i] = base_seca_corregida
                        if i < len(self.data['receta_corregida']):
                            self.data['receta_corregida'][i] = receta_corregida

                        # FLUJOS CORREGIDOS SA
                        if i < len(self.data['base_seca_corregida_sa']):
                            self.data['base_seca_corregida_sa'][i] = base_seca_corregida
                        if i < len(self.data['flujo_ksa_corregido_sa']):
                            self.data['flujo_ksa_corregido_sa'][i] = flujo_ksa_corregido 
                        if i < len(self.data['flujo_ic_corregido_sa']):
                            self.data['flujo_ic_corregido_sa'][i] = flujo_ic_corregido
                        if i < len(self.data['receta_corregida_sa']):
                            self.data['receta_corregida_sa'][i] = receta_corregida

                        # FLUJOS CORREGIDOS CLARIFICADOR
                        if i < len(self.data['flujo_ic_capacidad_clarificador']):
                            self.data['flujo_ic_capacidad_clarificador'][i] = flujo_ic_corregido 
                        if i < len(self.data['base_seca_capacidad_clarificador']):
                            self.data['base_seca_capacidad_clarificador'][i] = base_seca_corregida
                        if i < len(self.data['flujo_ksa_capacidad_clarificador']):
                            self.data['flujo_ksa_capacidad_clarificador'][i] = flujo_ksa_corregido   

                        # FLUJOS CORREGIDOS DOSIFICACIÓN
                        if i < len(self.data['receta_corregida_dosificacion']):
                            self.data['receta_corregida_dosificacion'][i] = receta_corregida
                        if i < len(self.data['base_seca_corregida_dosificacion']):
                            self.data['base_seca_corregida_dosificacion'][i] = base_seca_corregida
                        if i < len(self.data['flujo_ksa_corregido_dosificacion']):
                            self.data['flujo_ksa_corregido_dosificacion'][i] = flujo_ksa_corregido
                        if i < len(self.data['flujo_ic_corregido_dosificacion']):
                            self.data['flujo_ic_corregido_dosificacion'][i] = flujo_ic_corregido                         
                                                                                                                                                     
                            
            except (TypeError, ValueError, ZeroDivisionError) as e:
                continue

    def calcular_vectores_productos(self):
        """Calcula vectores de productos de manera segura"""
        longitud = self._longitud_principal
        if longitud == 0:
            return
        
        for i in range(longitud):
            try:
                if (i < len(self.data['mp_ic_tn']) and 
                    not pd.isna(self.data['mp_ic_tn'][i]) and 
                    self.data['mp_ic_tn'][i] > 0 and
                    i < len(self.data['aportante_porc_kp']) and
                    i < len(self.data['aportante_porc_ks']) and
                    i < len(self.data['aportante_porc_cc'])):
                    
                    mp_ic_tn = float(self.data['mp_ic_tn'][i])
                    aportante_kp = float(self.data['aportante_porc_kp'][i])
                    aportante_ks = float(self.data['aportante_porc_ks'][i])
                    aportante_cc = float(self.data['aportante_porc_cc'][i])
                    
                    # VECTORES PREVIOS TODO:
                    if i < len(self.data['porcentaje_grasa_ingreso_clarificador']):
                        self.data['porcentaje_grasa_ingreso_clarificador'][i] = 0.5 * self.data['grasa_ab_corregido_pama'][i]

                    self.data['porcentaje_solidos_efluente'][i] = 0.04 / 100
                    self.data['porcentaje_solidos_lodo'][i] = 0.08
                    self.data['porcentaje_solidos_licor'][i] = 0.1 / 100
                    self.data['porcentaje_grasa_licor'][i] = self.data['porcentaje_solidos_licor'][i] / 2
                    self.data['porcentaje_humedad_ksa'][i] = 0.75
                    
                    self.data['porcentaje_grasa_efluente'][i] = self.data['porcentaje_solidos_efluente'][i] / 2
                    self.data['flujo_lodo'][i] = self.data['flujo_ic_corregido_dosificacion'][i] * (self.data['solidos_ab_ic_corregido_pama'][i] * 0.000001 - self.data['porcentaje_solidos_efluente'][i]) / (self.data['porcentaje_solidos_lodo'][i] - self.data['porcentaje_solidos_efluente'][i])

                    self.data['flujo_efluente'][i] = self.data['flujo_ic_corregido_dosificacion'][i] - self.data['flujo_lodo'][i]
                    self.data['porcentaje_grasa_lodo'][i] = (self.data['flujo_ic_corregido_dosificacion'][i] * self.data['porcentaje_grasa_ingreso_clarificador'][i] / 1000000 - self.data['flujo_efluente'][i] * self.data['porcentaje_grasa_efluente'][i]) / self.data['flujo_lodo'][i]

                    self.data['flujo_ksa'][i] = self.data['flujo_lodo'][i] * ((1 - self.data['porcentaje_grasa_lodo'][i] - self.data['porcentaje_solidos_lodo'][i]) - (1 - self.data['porcentaje_grasa_licor'][i] - self.data['porcentaje_solidos_licor'][i]))/(self.data['porcentaje_humedad_ksa'][i] - (1 - self.data['porcentaje_grasa_licor'][i] - self.data['porcentaje_solidos_licor'][i]))
                    self.data['flujo_licor'][i] = self.data['flujo_lodo'][i] - self.data['flujo_ksa'][i]

                    self.data['porcentaje_grasa_ksa'][i] = (self.data['flujo_lodo'][i] * self.data['porcentaje_grasa_lodo'][i] - self.data['flujo_licor'][i] * self.data['porcentaje_solidos_licor'][i]) / self.data['flujo_ksa'][i]
                    self.data['porcentaje_solidos_ksa'][i] = 1 - self.data['porcentaje_grasa_ksa'][i] - self.data['porcentaje_humedad_ksa'][i]

                    # KK PRENSA
                    kk_prensa_tn = mp_ic_tn * aportante_kp / 100
                    
                    if i < len(self.data['kk_prensa_tn']):
                        self.data['kk_prensa_tn'][i] = kk_prensa_tn
                    
                    if i < len(self.data['kk_prensa_hum_tn']):
                        self.data['kk_prensa_hum_tn'][i] = kk_prensa_tn * self._parametros.KK_PRENSA_HUMEDAD
                    
                    if i < len(self.data['kk_prensa_cen_tn']):
                        self.data['kk_prensa_cen_tn'][i] = kk_prensa_tn * self._parametros.KK_PRENSA_CENIZA
                    
                    if i < len(self.data['kk_prensa_gra_tn']):
                        self.data['kk_prensa_gra_tn'][i] = kk_prensa_tn * self._parametros.KK_PRENSA_GRASA
                    
                    if i < len(self.data['kk_prensa_pro_tn']):
                        self.data['kk_prensa_pro_tn'][i] = kk_prensa_tn * self._parametros.KK_PRENSA_PROTEINA
                    
                    # Porcentajes KK Prensa
                    if i < len(self.data['kk_prensa_porc_humedad']):
                        self.data['kk_prensa_porc_humedad'][i] = self._parametros.KK_PRENSA_HUMEDAD * 100
                    
                    if i < len(self.data['kk_prensa_porc_ceniza']):
                        self.data['kk_prensa_porc_ceniza'][i] = self._parametros.KK_PRENSA_CENIZA * 100
                    
                    if i < len(self.data['kk_prensa_porc_grasa']):
                        self.data['kk_prensa_porc_grasa'][i] = self._parametros.KK_PRENSA_GRASA * 100
                    
                    if i < len(self.data['kk_prensa_porc_proteina']):
                        self.data['kk_prensa_porc_proteina'][i] = self._parametros.KK_PRENSA_PROTEINA * 100
                    
                    # KK SEPARADORA
                    kk_separadora_tn = mp_ic_tn * aportante_ks / 100
                    
                    if i < len(self.data['kk_separadora_tn']):
                        self.data['kk_separadora_tn'][i] = kk_separadora_tn
                    
                    if i < len(self.data['kk_sep_hum_tn']):
                        self.data['kk_sep_hum_tn'][i] = kk_separadora_tn * self._parametros.KK_SEP_HUMEDAD
                    
                    if i < len(self.data['kk_sep_cen_tn']):
                        self.data['kk_sep_cen_tn'][i] = kk_separadora_tn * self._parametros.KK_SEP_CENIZA
                    
                    if i < len(self.data['kk_sep_gra_tn']):
                        self.data['kk_sep_gra_tn'][i] = kk_separadora_tn * self._parametros.KK_SEP_GRASA
                    
                    if i < len(self.data['kk_sep_pro_tn']):
                        self.data['kk_sep_pro_tn'][i] = kk_separadora_tn * self._parametros.KK_SEP_PROTEINA
                    
                    # Porcentajes KK Separadora
                    if i < len(self.data['kk_separadora_porc_humedad']):
                        self.data['kk_separadora_porc_humedad'][i] = self._parametros.KK_SEP_HUMEDAD * 100
                    
                    if i < len(self.data['kk_separadora_porc_ceniza']):
                        self.data['kk_separadora_porc_ceniza'][i] = self._parametros.KK_SEP_CENIZA * 100
                    
                    if i < len(self.data['kk_separadora_porc_grasa']):
                        self.data['kk_separadora_porc_grasa'][i] = self._parametros.KK_SEP_GRASA * 100
                    
                    if i < len(self.data['kk_separadora_porc_proteina']):
                        self.data['kk_separadora_porc_proteina'][i] = self._parametros.KK_SEP_PROTEINA * 100
                    
                    # KK CC TODO: Tener cuidado con los valores en porcentajes
                    kk_cc_tn = mp_ic_tn * aportante_cc / 100
                    
                    if i < len(self.data['kk_cc_tn']):
                        self.data['kk_cc_tn'][i] = kk_cc_tn

                    if i < len(self.data['kk_cc_cen_tn']):
                        self.data['kk_cc_cen_tn'][i] = self.data['mp_ic_cen_tn'][i] - self.data['kk_prensa_cen_tn'][i] - self.data['kk_sep_cen_tn'][i]

                    if i < len(self.data['kk_cc_gra_tn']):
                        self.data['kk_cc_gra_tn'][i] = self._parametros.KK_CC_GRASA * kk_cc_tn

                    if i < len(self.data['kk_cc_pro_tn']):
                        self.data['kk_cc_pro_tn'][i] = (mp_ic_tn - self.data['mp_ic_cen_tn'][i] - self.data['mp_ic_hum_tn'][i] - self.data['mp_ic_gra_tn'][i]) - self.data['kk_prensa_pro_tn'][i] - self.data['kk_sep_pro_tn'][i]
                    
                    if i < len(self.data['kk_cc_hum_tn']):
                        self.data['kk_cc_hum_tn'][i] = kk_cc_tn - self.data['kk_cc_cen_tn'][i] - self.data['kk_cc_gra_tn'][i] - self.data['kk_cc_pro_tn'][i]

                    if i < len(self.data['kk_cc_porc_humedad_2']):
                        self.data['kk_cc_porc_humedad_2'][i] = self.data['kk_cc_hum_tn'][i] / kk_cc_tn
                        self.data['kk_cc_porc_humedad'][i] = self._parametros.KK_CC_HUMEDAD

                    if i < len(self.data['kk_cc_porc_ceniza_2']):
                        self.data['kk_cc_porc_ceniza_2'][i] = self.data['kk_cc_cen_tn'][i] / kk_cc_tn
                        self.data['kk_cc_porc_ceniza'][i] = self._parametros.KK_CC_CENIZA  

                    if i < len(self.data['kk_cc_porc_grasa']):
                        self.data['kk_cc_porc_grasa'][i] = self._parametros.KK_CC_GRASA 

                    if i < len(self.data['kk_cc_porc_proteina']):
                        self.data['kk_cc_porc_proteina'][i] = self._parametros.KK_CC_PROTEINA  

                    if i < len(self.data['ksa_porc_humedad']):
                        self.data['ksa_porc_humedad'][i] = self._parametros.KSA_HUMEDAD   

                    if i < len(self.data['ksa_porc_ceniza']):
                        self.data['ksa_porc_ceniza'][i] = self._parametros.KSA_CENIZA

                    if i < len(self.data['ksa_porc_grasa']):
                        self.data['ksa_porc_grasa'][i] = self.data['porcentaje_grasa_ksa'][i] 

                    if i < len(self.data['ksa_porc_proteina']):
                        self.data['ksa_porc_proteina'][i] = 1 - self.data['ksa_porc_grasa'][i] - self.data['ksa_porc_humedad'][i] - self.data['ksa_porc_ceniza'][i]

                    if i < len(self.data['ksa_gra_tn']):
                        self.data['ksa_gra_tn'][i] = self.data['ksa_porc_grasa'][i] * self.data['receta_corregida_dosificacion'][i] * mp_ic_tn

                    if i < len(self.data['ksa_pro_tn']):
                        self.data['ksa_pro_tn'][i] = self.data['ksa_porc_proteina'][i] * self.data['receta_corregida_dosificacion'][i] * mp_ic_tn

                    if i < len(self.data['porcentaje_dosificacion_ksa2']):
                        self.data['porcentaje_dosificacion_ksa2'][i] = self.data['receta_corregida_dosificacion'][i]
                  
                    
                    # KSA (diferencia)
                    ksa_tn = mp_ic_tn - kk_prensa_tn - kk_separadora_tn - kk_cc_tn
                    
                    if i < len(self.data['ksa_tn']):
                        self.data['ksa_tn'][i] = ksa_tn
                    
                    if i < len(self.data['ksa_hum_tn']):
                        self.data['ksa_hum_tn'][i] = ksa_tn * self._parametros.KSA_HUMEDAD
                    
                    if i < len(self.data['ksa_cen_tn']):
                        self.data['ksa_cen_tn'][i] = ksa_tn * self._parametros.KSA_CENIZA
                    
                    # KK INTEGRAL
                    kk_integral_tn = kk_prensa_tn + kk_separadora_tn + kk_cc_tn + ksa_tn
                    
                    if i < len(self.data['kk_integral_tn']):
                        self.data['kk_integral_tn'][i] = kk_integral_tn

                    if i < len(self.data['kk_integral_hum_tn']):
                        self.data['kk_integral_hum_tn'][i] = self.data['kk_prensa_hum_tn'][i] + self.data['kk_sep_hum_tn'][i] + self.data['kk_cc_hum_tn'][i] + self.data['ksa_hum_tn'][i]

                    if i < len(self.data['kk_integral_cen_tn']):
                        self.data['kk_integral_cen_tn'][i] = self.data['kk_prensa_cen_tn'][i] + self.data['kk_sep_cen_tn'][i] + self.data['kk_cc_cen_tn'][i] + self.data['ksa_cen_tn'][i]

                    if i < len(self.data['kk_integral_gra_tn']):
                        self.data['kk_integral_gra_tn'][i] = self.data['kk_prensa_gra_tn'][i] + self.data['kk_sep_gra_tn'][i] + self.data['kk_cc_gra_tn'][i] + self.data['ksa_gra_tn'][i]

                    if i < len(self.data['kk_integral_pro_tn']):
                        self.data['kk_integral_pro_tn'][i] = self.data['kk_prensa_pro_tn'][i] + self.data['kk_sep_pro_tn'][i] + self.data['kk_cc_pro_tn'][i] + self.data['ksa_pro_tn'][i]                        

                    if i < len(self.data['kk_integral_porc_humedad']):
                        self.data['kk_integral_porc_humedad'][i] = (self.data['kk_integral_hum_tn'][i] / kk_integral_tn)

                    if i < len(self.data['kk_integral_porc_ceniza']):
                        self.data['kk_integral_porc_ceniza'][i] = (self.data['kk_integral_cen_tn'][i] / kk_integral_tn)
                    
                    if i < len(self.data['kk_integral_porc_grasa']):
                        self.data['kk_integral_porc_grasa'][i] = (self.data['kk_integral_gra_tn'][i] / kk_integral_tn)
                    
                    if i < len(self.data['kk_integral_porc_proteina']):
                        self.data['kk_integral_porc_proteina'][i] = (self.data['kk_integral_pro_tn'][i] / kk_integral_tn)

                    if i < len(self.data['harina_porc_humedad_2']):
                        self.data['harina_porc_humedad_2'][i] = self._parametros.HARINA_HUMEDAD

                    if i < len(self.data['harina_porc_ceniza_2']):
                        self.data['harina_porc_ceniza_2'][i] = self.data['kk_integral_porc_ceniza'][i]/((self.data['kk_integral_porc_ceniza'][i]+self.data['kk_integral_porc_grasa'][i]+self.data['kk_integral_porc_proteina'][i])/(100-self.data['harina_porc_humedad_2'][i]))

                    if i < len(self.data['harina_porc_grasa_2']):
                        self.data['harina_porc_grasa_2'][i] = self.data['kk_integral_porc_grasa'][i]/((self.data['kk_integral_porc_ceniza'][i]+self.data['kk_integral_porc_grasa'][i]+self.data['kk_integral_porc_proteina'][i])/(100-self.data['harina_porc_humedad_2'][i]))

                    if i < len(self.data['harina_porc_proteina_2']):
                        self.data['harina_porc_proteina_2'][i] = self.data['kk_integral_porc_proteina'][i]/((self.data['kk_integral_porc_ceniza'][i]+self.data['kk_integral_porc_grasa'][i]+self.data['kk_integral_porc_proteina'][i])/(100-self.data['harina_porc_humedad_2'][i]))
                        
            except (TypeError, ValueError, ZeroDivisionError) as e:
                continue

    def calcular_vectores_resultados_finales(self):
        """Calcula los resultados finales del proceso"""
        longitud = self._longitud_principal
        if longitud == 0:
            return
        
        for i in range(longitud):
            try:
                # HAR TON
                if (i < len(self.data['kk_integral_cen_tn']) and 
                    i < len(self.data['kk_integral_gra_tn']) and 
                    i < len(self.data['kk_integral_pro_tn']) and
                    i < len(self.data['harina_porc_humedad']) and
                    not pd.isna(self.data['kk_integral_cen_tn'][i]) and
                    not pd.isna(self.data['kk_integral_gra_tn'][i]) and
                    not pd.isna(self.data['kk_integral_pro_tn'][i]) and
                    not pd.isna(self.data['harina_porc_humedad'][i])):
                    
                    componentes_secos = (self.data['kk_integral_cen_tn'][i] + 
                                       self.data['kk_integral_gra_tn'][i] + 
                                       self.data['kk_integral_pro_tn'][i])
                    
                    har_ton = componentes_secos / (1 - self.data['harina_porc_humedad'][i] / 100)
                    
                    if i < len(self.data['har_ton']):
                        self.data['har_ton'][i] = har_ton

                    aceite_tn = self.data['mp_ic_gra_tn'][i] - (self.data['kk_prensa_gra_tn'][i]+self.data['kk_sep_gra_tn'][i] + self.data['kk_cc_gra_tn'][i])
                    if i < len(self.data['aceite_tn']):
                        self.data['aceite_tn'][i] = aceite_tn

                # RENDIMIENTO HARINA
                if (i < len(self.data['mp_ic_tn']) and 
                    i < len(self.data['har_ton']) and
                    not pd.isna(self.data['mp_ic_tn'][i]) and 
                    not pd.isna(self.data['har_ton'][i]) and
                    self.data['mp_ic_tn'][i] > 0):
                    
                    rendimiento = self.data['mp_ic_tn'][i] / self.data['har_ton'][i]
                    
                    if i < len(self.data['rendimiento_harina']):
                        self.data['rendimiento_harina'][i] = rendimiento

                    rendimiento_aceite = self.data['aceite_tn'][i] / self.data['mp_ic_tn'][i]
                    if i < len(self.data['rendimiento_aceite']):
                        self.data['rendimiento_aceite'][i] = rendimiento_aceite
                        
            except (TypeError, ValueError, ZeroDivisionError) as e:
                continue

    def calcular_vectores_clarificador(self):
        """Calcula vectores del clarificador (placeholder)"""
        # Implementación simplificada - se puede expandir luego
        pass

    def obtener_dataframe_completo(self) -> pd.DataFrame:
        """Retorna DataFrame con todas las columnas"""
        if self._longitud_principal == 0:
            return pd.DataFrame()
        
        longitudes = [len(vector) for vector in self.data.values()]
        longitud_maxima = max(longitudes) if longitudes else 0
        
        df_data = {}
        for nombre_vector, vector in self.data.items():
            if len(vector) < longitud_maxima:
                if vector.dtype == object:
                    extension = np.array([None] * (longitud_maxima - len(vector)), dtype=object)
                else:
                    extension = np.full(longitud_maxima - len(vector), np.nan, dtype=vector.dtype)
                
                vector_extendido = np.concatenate([vector, extension])
                df_data[nombre_vector] = vector_extendido
            else:
                df_data[nombre_vector] = vector
        
        df = pd.DataFrame(df_data)
        
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].where(pd.notna(df[col]), None)
        
        return df

    def ejecutar_calculos_completos(self):
        """Ejecuta todos los cálculos del proceso completo"""
        try:
            print("🔧 Inicializando vectores...")
            self.inicializar_vectores_proceso()
            
            print("📊 Simulando progreso...")
            self.simular_progreso_completo()
            
            print("🌱 Calculando materia prima...")
            self.calcular_vectores_materia_prima()
            
            print("⚙️ Calculando proceso...")
            self.calcular_vectores_proceso()
            
            print("📈 Calculando flujos...")
            self.calcular_vectores_flujos()
            
            print("🏭 Calculando productos...")
            self.calcular_vectores_productos()
            
            print("📊 Calculando resultados finales...")
            self.calcular_vectores_resultados_finales()
            
            print("✅ Todos los cálculos completados exitosamente")
            return self.data
            
        except Exception as e:
            print(f"❌ Error durante los cálculos: {e}")
            import traceback
            traceback.print_exc()
            return self.data
    
def cargar_datos_plan_descarga_y_tablas_mio(id_pd):
    query_pd = f"""
    SELECT A.*, B.id_planta, 
    B.FlagPPyPZ,
    case when TE = 'Propio C/Frio' then 'RC'
    when TE = 'Propio S/Frio' then 'SF'
    else 'SF' end as TIPO,
    0.1 AS JuvenilesPonderado
    FROM PlanDescargaOutput A
    LEFT JOIN PlanDescargaModeloLog B
    ON A.IDLog=B.id
    WHERE B.FlagPPyPZ = 1 AND B.id = {id_pd};"""
    discharge_plan_df_original = process_discharge_plan_data(pd.read_sql(query_pd, cnxn_mio))

    terceros_frios = [
        "RICARDO",
        "OLGA",
        "PATRICIA",
        "ALESSANDRO",
        "CHAVELI II",
        "ISABELITA",
        "STEFANO",
        "DON LUCHO",
        "DANIELLA",
        "SEBASTIAN",
        "BAMAR II",
        "MODESTO 3",
        "MODESTO 6",
    ]
    mask = discharge_plan_df_original["Embarcacion"].isin(terceros_frios)
    discharge_plan_df_original.loc[mask, "TIPO"] = 'RC'

    query = """
    select  top 60 Marea,FECHA ,TRIM(PLANTA) PLANTA , TVN_DES ,DECLARA, DESCARG ,GRA_DESCAR , HUM_DESCAR 
    from sap.ZQM_DESC_MP_CHI2 zdmc2 
    where F_ARRIB  >cast('2024-04-21' as date) AND DESCARG >0 and GRA_DESCAR > 0
    order by Marea desc;
    """

    grasa_df_original = pd.read_sql(query, cnxn_juv)
    grasa_df = grasa_df_original.copy()
    grasa_df["PLANTA"] = grasa_df["PLANTA"].str.upper()
    grasa_df = grasa_df.groupby('PLANTA', as_index=False)["GRA_DESCAR"].mean()

    to_replace = {
        "MALABRIGO SUR": "MALABRIGO",
        "CALLAO NORTE": "CALLAO",
    }
    grasa_df["PLANTA"] = grasa_df["PLANTA"].replace(to_replace)

    query = """
    SELECT * FROM dbo.Parametros_proceso
    """
    limpieza_tuberia_df_orig = pd.read_sql(query, cnxn_mio)

    query = """
    SELECT * FROM dbo.Chatas_Lineas
    """
    chatas_lineas_df_orig = pd.read_sql(query, cnxn_mio)

    query = """
    SELECT * FROM dbo.Tipo_Descarga
    """
    condiciones_rap_orig = pd.read_sql(query, cnxn_mio)

    calidad_ic_df = limpieza_tuberia_df_orig[["PLANTA", "UmbralIC_TVN"]].copy()

    query = """
    SELECT * FROM dbo.Tabla_velocidades
    """
    velocidades_volumen_df_orig = pd.read_sql(query, cnxn_mio)

    query = """
    SELECT * FROM dbo.Dosificacion_KSA_calidades"""
    UmbralTVN_Calidades = process_calidades(pd.read_sql(query, cnxn_mio))

    query = """
    SELECT * FROM dbo.KK_calidades"""
    ValoresFijosPP_orig = pd.read_sql(query, cnxn_mio)

    query = """
    SELECT * FROM dbo.Balance_KSA"""
    balance_ksa_df_orig = pd.read_sql(query, cnxn_mio)

    query = """
    SELECT * FROM dbo.FECHA_AJUSTADO_PP"""
    hora_arranque_df = pd.read_sql(query, cnxn_mio)

    now = datetime.now()
    discharge_plan_df = discharge_plan_df_original.copy()
    fecha_batch = discharge_plan_df["Fecha"].unique()[0]
    try:
        planta_selected = discharge_plan_df["id_planta"].unique()[0]
    except:
        pass
    try:
        grasa_h75 = grasa_df[grasa_df["PLANTA"] == planta_selected]["GRA_DESCAR"].values[0]
    except:
        grasa_h75 = grasa_df["GRA_DESCAR"].mean()
        
    print("Grasa H75", grasa_h75)
    limpieza_tuberia_df = process_limpieza_tuberia_data(limpieza_tuberia_df_orig, planta_selected)
    limite_base_seca = limpieza_tuberia_df["LimiteCapacidadSeparadoraAmbiental"].values[0]
    print("Limite Base Seca", limite_base_seca)
    limite_flujo_ic = limpieza_tuberia_df["ValorFlujoIngreso"].values[0]
    print("Limite Flujo IC", limite_flujo_ic)
    chatas_lineas_df = process_chatas_lineas_data(chatas_lineas_df_orig, planta_selected)
    condiciones_rap = process_condiciones_rap(condiciones_rap_orig, planta_selected)
    velocidades_volumen_df = process_velocidades(velocidades_volumen_df_orig, planta_selected)
    balance_ksa_df = process_balance_ksa_data(balance_ksa_df_orig, planta_selected)
    stock_minimo_pama = hallar_volumen_minimo_pama(velocidades_volumen_df, grasa_h75)
    ValoresFijosPP = process_valores_fijos(ValoresFijosPP_orig, planta_selected)
    arranque = process_arranque(hora_arranque_df, planta_selected, fecha_batch)
    volumen_materia_prima = discharge_plan_df["VolumenEstTM"].sum()
    v_init, v_intermedia_inicial, v_cruise, v_intermedia_final, v_term = hallar_velocidades_base(velocidades_volumen_df, grasa_h75, volumen_materia_prima)
    
    # ------------------------------------------Procesamiento de datos ------------------------------------------------------------------#

    # Generación de Intervalos
    intervalos_df = generate_intervalo_horas(discharge_plan_df)

    # Convesion de velocidades a lista
    V_tot     = discharge_plan_df["VolumenEstTM"].sum()    # volumen total a procesar

    velocidades = {
        'Velocidad Inicio': v_init,
        'Velocidad Intermedia1': v_intermedia_inicial,
        'Velocidad Maxima': v_cruise,
        'Velocidad Intermedia2': v_intermedia_final,
        'Velocidad Final': v_term
    }
    # volumen = 260.0

    speeds_fixed, perfil_lbls = generar_perfil_velocidades(V_tot, velocidades)
    # Generación de Estados de Poza
    nombre_velocidad = []
    for ind, vel in enumerate(speeds_fixed):
        cont = 0
        for tipo in velocidades.keys():
            if vel == velocidades[tipo]:
                if tipo == "Velocidad Inicio" and ind > 3:
                    continue
                nombre_velocidad.append(tipo)
                cont = cont + 1
                break
        if cont == 0:
            nombre_velocidad.append("Velocidad Final")

    df_pozas = pd.DataFrame(np.arange(1, pozas+1), columns=["Poza"])
    intervalos_df['key'] = 0
    df_pozas['key'] = 0
    status_pozas = pd.merge(intervalos_df, df_pozas, on='key', how='outer')
    del status_pozas["key"]
    status_pozas["StockActual"] = 0
    status_pozas["TVN_Actual"] = 0
    status_pozas["StockAlimentado"] = 0
    status_pozas["TVNAlimentado"] = 0
    status_pozas["Capacidad_Pozas"] = 300

    # Completar intervalos con la data de descarga
    eps = discharge_plan_df["Embarcacion"].unique()
    mareas_actuales = discharge_plan_df["MareaId"].unique()
    discharge_plan_df["TDC_Descarga"] = discharge_plan_df["TDC_Descarga"].clip(upper=28)
    intervals_filled_df = pd.concat([fill_intervals(intervalos_df, discharge_plan_df[discharge_plan_df["MareaId"] == marea_id]) for marea_id in mareas_actuales])

    # Actualizar TDC a la Descarga
    intervals_filled_df["FlagHora"] = 0
    mask = intervals_filled_df["Duracion"] > 0
    intervals_filled_df.loc[mask, 'FlagHora'] = 1
    intervals_filled_df["to_update_tdc"] = intervals_filled_df.groupby(['Embarcacion'])['FlagHora'].cumsum() - 1
    intervals_filled_df["TDC_Descarga_Actual"] = np.where(intervals_filled_df["Volumen"] > 0, intervals_filled_df["TDC_Descarga_Inicial"] + intervals_filled_df["to_update_tdc"], 0)

    # Agregar el RAP a la descarga
    intervals_rap = add_rap(intervals_filled_df, chatas_lineas_df, condiciones_rap, limpieza_tuberia_df)

    # Calcula el RAP PAMA
    intervals_rap["VolumenRAP"] = intervals_rap["Volumen"] * intervals_rap["RAP"]
    intervals_rap["VolumenTromel"] = intervals_rap["Duracion"] * 8
    intervals_rap["RAP_Limpieza_Tromel"] = intervals_rap["VolumenRAP"] + (intervals_rap["LimpiezaTuberia"] * intervals_rap["Duracion"]) + intervals_rap["VolumenTromel"]
    intervals_rap["Limpieza_de_Tuberia"] = intervals_rap["LimpiezaTuberia"] * intervals_rap["Duracion"]
    intervals_rap["VolumenTotalAB"] = intervals_rap["RAP_Limpieza_Tromel"] * intervals_rap["FactorCorreccion"] / 100
    intervals_rap["RAP_PAMA"] = intervals_rap["VolumenTotalAB"] / intervals_rap["Volumen"]


    intervals_rap["TVNDescarga"] = np.where(intervals_rap["TipoEP"].isin(['SF']), np.vectorize(estimate_tvn_sin_frio)(intervals_rap['TDC_Descarga_Actual']), np.vectorize(estimate_tvn_con_frio)(intervals_rap['TDC_Descarga_Actual']))
    intervals_rap["TVNDescarga"] = np.where(intervals_rap["Volumen"] > 0, intervals_rap["TVNDescarga"], np.nan)
    resultados_pp_grouped = group_intervals(intervals_rap, stock_minimo_pama)
    resultados_pp_grouped["flg_min_acum"] = resultados_pp_grouped["flg_min"].cumsum()
    volumen_ab = resultados_pp_grouped["VolumenPama"].max()
    return resultados_pp_grouped, volumen_ab, arranque

# EJEMPLO COMPLETO Y FUNCIONAL
def ejemplo_completo_funcional():
    print("=== SIMULACIÓN COMPLETA DEL BALANCE GENERAL ===")
    
    balance = BalanceGeneralVectorizadoCompleto()
    id_prueba_pd = 7206
    datos_pd_mio, volumen_ab, arranque = cargar_datos_plan_descarga_y_tablas_mio(id_prueba_pd)

    if len(arranque) > 0:
        idx_min = (datos_pd_mio['Intervalo Poza'] - arranque.values[0]).abs().idxmin()
        cook_start_hour = datos_pd_mio.loc[idx_min]["NumHora"]
    else:
        cook_start_hour = datos_pd_mio.loc[index_arranque, 'NumHora'] + 1

    try:
        index_arranque = datos_pd_mio[datos_pd_mio['flg_min'] == 1].iloc[0].name
        hora_cook = datos_pd_mio.loc[index_arranque, 'NumHora'] + 1
        cook_start_hour_model = datos_pd_mio.loc[datos_pd_mio["NumHora"] == hora_cook, "Intervalo Poza"].values[0]
    except:   
        cook_start_hour_model = None
    
    # descargas = intervals_rap.copy()
    # status_init = status_pozas.copy()

    print(datos_pd_mio)
    datos_iniciales = {
        'intervalo_chata': datos_pd_mio["Intervalo Chata"].to_numpy(dtype=object),
        'intervalo_poza':  datos_pd_mio["Intervalo Poza"].to_numpy(dtype=object),
        'descarga': datos_pd_mio["VolumenTotal"].to_numpy(dtype=object),
        'avance_descarga': datos_pd_mio["VolumenAcumulado"].to_numpy(dtype=object)
    }
    
    balance.cargar_datos_completos_desde_excel(datos_iniciales)
    resultados = balance.ejecutar_calculos_completos()
    
    df_completo = balance.obtener_dataframe_completo()
    
    print(f"\n✅ DataFrame generado: {df_completo.shape[0]} filas x {df_completo.shape[1]} columnas")
    
    # Mostrar resultados clave
    print("\n=== RESULTADOS CLAVE ===")
    columnas_interes = ['intervalo_chata', 'velocidad', 'mp_ic_tn', 'solidos_ab', 'flujo_ksa_corregido', 'kk_integral_tn', 'har_ton']
    columnas_disponibles = [col for col in columnas_interes if col in df_completo.columns]

    # start_time = datetime.now()
    now = datetime.now()
    fecha_hora_actual = now.strftime('%Y-%m-%d_%H-%M-%S')
    # time_elapsed = now - start_time
    # print(f'Time elapsed (hh:mm:ss.ms) {time_elapsed}')

    if columnas_disponibles:
        # Mostrar solo filas con datos
        df_completo.to_excel(f"outputs/resultado_balance_general_{fecha_hora_actual}.xlsx", index=False)
        df_filtrado = df_completo[columnas_disponibles].dropna(how='all')
        print(df_filtrado.to_string())
    
    return balance, df_completo

if __name__ == "__main__":
    try:
        balance, df = ejemplo_completo_funcional()
        print("\n🎉 ¡Simulación completada exitosamente!")
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        import traceback
        traceback.print_exc()    
    
print("aaaaaaaaaaaaa")