import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import pyodbc
import os


plantas = ["MALABRIGO", "CHIMBOTE", "SUPE", "VEGUETA", "CALLAO", "PISCO SUR"]
threshold_degradacion = 3
pozas = 10
ambiente = 'prod'
print("Ambiente: ", ambiente)
import sqlalchemy
import urllib

connCap = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=srv-db-east-us003.database.windows.net;DATABASE=db_cfa_prd01;UID=userdbowner;PWD=$P4ssdbowner01#;'
db_params_pc = urllib.parse.quote_plus(connCap)
enginePC = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(db_params_pc))

def compute_solidos_ab_ic(tvn_pozas, grasa_h75=4, fc_solidos=1):
    """
    Calcula el factor sólidos_ab_ic para cada tvn.
    """
    return (-445.24 + 294 * tvn_pozas - 606.62 * grasa_h75 + 88.804 * grasa_h75**2) * fc_solidos

def fill_next_two_nulls(df, columns):
    for col in columns:
        last_valid = None
        null_count = 0
        for i in range(len(df)):
            if pd.notnull(df.at[i, col]):
                last_valid = df.at[i, col]
                null_count = 0
            else:
                null_count += 1
                if null_count <= 2 and last_valid is not None:
                    df.at[i, col] = last_valid
    return df

# Aplicar la función

def compute_flujos(velocidad, dosificacion, tvn_pozas, grasa_h75, fc_solidos):
    """
    Dado velocidad y dosificacion, calcula flujo_ksa, base_seca, solidos_ab_ic y flujo_ic.
    """
    velocidad = np.array(velocidad, dtype=float)
    dosificacion = np.array(dosificacion, dtype=float)
    tvn_pozas = np.array(tvn_pozas, dtype=float)

    flujo_ksa = velocidad * dosificacion
    base_seca = flujo_ksa / 4.0
    sol_ab = compute_solidos_ab_ic(tvn_pozas, grasa_h75, fc_solidos)
    flujo_ic = base_seca * 1e6 / sol_ab
    return flujo_ksa, base_seca, sol_ab, flujo_ic


def calibracion1(velocidad, dosificacion, tvn_pozas,
                 volumen_agua,
                 grasa_h75, fc_solidos):
    """
    Ajusta para que la suma de flujo_ic sea volumen_agua.
    Devuelve (velocidad, dosificacion, tvn_pozas, flujo_ksa, base_seca, sol_ab, flujo_ic).
    """
    # 1. Cálculo inicial de flujos
    flujo_ksa, base_seca, sol_ab, flujo_ic = compute_flujos(
        velocidad, dosificacion, tvn_pozas, grasa_h75, fc_solidos)

    # 2. Factor de corrección
    fc1 = volumen_agua / flujo_ic.sum()

    # 3. Ajuste de flujos y bases
    flujo_ic1 = flujo_ic * fc1
    base_seca1 = flujo_ic1 * sol_ab / 1e6
    flujo_ksa1 = 4 * base_seca1
    dosificacion1 = flujo_ksa1 / np.array(velocidad, dtype=float)

    return (velocidad, dosificacion1.tolist(), tvn_pozas,
            flujo_ksa1.tolist(), base_seca1.tolist(), sol_ab.tolist(), flujo_ic1.tolist())


def calibracion2(velocidad, dosificacion, tvn_pozas,
                 flujo_ksa, base_seca, sol_ab, flujo_ic,
                 volumen_agua, limite_base_seca,
                 V_total, velocidad_final):
    """
    Ajusta base_seca para no superar limite_base_seca.
    Redistribuye déficit o añade un elemento extra si es necesario.
    """
    velocidad = list(velocidad)
    dosificacion = list(dosificacion)
    tvn_pozas = list(tvn_pozas)

    flujo_ksa = np.array(flujo_ksa)
    base_seca = np.array(base_seca)
    sol_ab = np.array(sol_ab)
    flujo_ic = np.array(flujo_ic)

    # 1. Acotar valores superiores al límite
    mask_over = base_seca > limite_base_seca
    base_seca[mask_over] = limite_base_seca

    # 2. Recalcular flujos y dosis
    flujo_ksa = 4 * base_seca
    flujo_ic = base_seca * 1e6 / sol_ab
    dosificacion = flujo_ksa / np.array(velocidad, dtype=float)

    # 3. Déficit vs volumen_agua
    deficit = volumen_agua - flujo_ic.sum()

    # 4. Redistribución proporcional si hay cupo
    if deficit > 0:
        mask_under = base_seca < limite_base_seca
        total_under = base_seca[mask_under].sum()
        if total_under > 0:
            delta = deficit * (base_seca / total_under)
            # Solo aplicamos a los bajo límite
            base_seca[mask_under] += delta[mask_under]
            # No exceder límite
            base_seca[mask_under] = np.minimum(base_seca[mask_under], limite_base_seca)

            # Recalcular tras redistribuir
            flujo_ksa = 4 * base_seca
            flujo_ic = base_seca * 1e6 / sol_ab
            dosificacion = flujo_ksa / np.array(velocidad, dtype=float)

            deficit = volumen_agua - flujo_ic.sum()

    # 5. Si aún queda déficit, añadir elemento extra
    if deficit > 0:
        tvn_extra = tvn_pozas[-1]
        dos_extra = dosificacion[-1]
        # velocidad extra como remanente para llegar a V_total
        remanente = V_total - sum(velocidad)
        v_extra = min(remanente, velocidad_final * 0.99)

        # Añadir valores
        velocidad.append(v_extra)
        dosificacion.append(dos_extra)
        tvn_pozas.append(tvn_extra)

        # Calcular flujo para el extra
        flujo_ksa_extra = v_extra * dos_extra
        base_seca_extra = flujo_ksa_extra / 4.0
        sol_ab_extra = compute_solidos_ab_ic(tvn_extra)
        flujo_ic_extra = base_seca_extra * 1e6 / sol_ab_extra

        flujo_ksa = np.append(flujo_ksa, flujo_ksa_extra)
        base_seca = np.append(base_seca, base_seca_extra)
        sol_ab = np.append(sol_ab, sol_ab_extra)
        flujo_ic = np.append(flujo_ic, flujo_ic_extra)

    return (velocidad, dosificacion, tvn_pozas,
            flujo_ksa.tolist(), base_seca.tolist(), sol_ab.tolist(), flujo_ic.tolist())


def calibracion3(velocidad, dosificacion, tvn_pozas,
                 flujo_ksa, base_seca, sol_ab, flujo_ic,
                 volumen_agua, limite_flujo_ic,
                 V_total, velocidad_final):
    """
    Ajusta flujo_ic para no superar limite_flujo_ic.
    Redistribuye déficit o añade un elemento extra si es necesario.
    """
    velocidad = list(velocidad)
    dosificacion = list(dosificacion)
    tvn_pozas = list(tvn_pozas)

    flujo_ksa = np.array(flujo_ksa)
    base_seca = np.array(base_seca)
    sol_ab = np.array(sol_ab)
    flujo_ic = np.array(flujo_ic)

    # 1. Acotar flujo_ic
    mask_over = flujo_ic > limite_flujo_ic
    flujo_ic[mask_over] = limite_flujo_ic

    # 2. Recalcular base_seca y dosificación
    base_seca = flujo_ic * sol_ab / 1e6
    flujo_ksa = 4 * base_seca
    dosificacion = list(flujo_ksa / np.array(velocidad, dtype=float))

    # 3. Déficit vs volumen_agua
    deficit = volumen_agua - flujo_ic.sum()

    # 4. Redistribución proporcional
    if deficit > 0:
        mask_under = flujo_ic < limite_flujo_ic
        total_under = flujo_ic[mask_under].sum()
        if total_under > 0:
            delta_ic = deficit * (flujo_ic / total_under)
            flujo_ic[mask_under] += delta_ic[mask_under]
            flujo_ic[mask_under] = np.minimum(flujo_ic[mask_under], limite_flujo_ic)

            # Recalcular tras redistribuir
            base_seca = flujo_ic * sol_ab / 1e6
            flujo_ksa = 4 * base_seca
            dosificacion = list(flujo_ksa / np.array(velocidad, dtype=float))

            deficit = volumen_agua - flujo_ic.sum()

    # 5. Añadir elemento extra si persiste déficit
    if deficit > 0:
        tvn_extra = tvn_pozas[-1]
        dos_extra = dosificacion[-1]
        remanente = V_total - sum(velocidad)
        v_extra = min(remanente, velocidad_final * 0.99)

        velocidad.append(v_extra)
        dosificacion.append(dos_extra)
        tvn_pozas.append(tvn_extra)

        flujo_ksa_extra = v_extra * dos_extra
        base_seca_extra = flujo_ksa_extra / 4.0
        sol_ab_extra = compute_solidos_ab_ic(tvn_extra)
        flujo_ic_extra = base_seca_extra * 1e6 / sol_ab_extra

        flujo_ksa = np.append(flujo_ksa, flujo_ksa_extra)
        base_seca = np.append(base_seca, base_seca_extra)
        sol_ab = np.append(sol_ab, sol_ab_extra)
        flujo_ic = np.append(flujo_ic, flujo_ic_extra)

    return (velocidad, dosificacion, tvn_pozas,
            flujo_ksa.tolist(), base_seca.tolist(), sol_ab.tolist(), flujo_ic.tolist())


def calibrar(velocidad, dosificacion, tvn_pozas,
             volumen_agua, limite_base_seca, limite_flujo_ic,
             V_total, velocidad_final,
             grasa_h75, fc_solidos):
    """
    Ejecuta las tres calibraciones secuenciales.
    Devuelve todos los vectores finales.
    """
    # Calibración 1
    out1 = calibracion1(velocidad, dosificacion, tvn_pozas,
                        volumen_agua, grasa_h75, fc_solidos)
    vel1, dos1, tvn1, fksa1, bsec1, sol1, fic1 = out1

    # Cambio 17.11.2025: sOLO HACEMOS UNA CALIBRACION
    # # Calibración 2
    # out2 = calibracion2(vel1, dos1, tvn1, fksa1, bsec1, sol1, fic1,
    #                     volumen_agua, limite_base_seca,
    #                     V_total, velocidad_final)
    # vel2, dos2, tvn2, fksa2, bsec2, sol2, fic2 = out2

    # # Calibración 3
    # out3 = calibracion3(vel2, dos2, tvn2, fksa2, bsec2, sol2, fic2,
    #                     volumen_agua, limite_flujo_ic,
    #                     V_total, velocidad_final)
    # return out3
    return out1

def get_final_grasa_df(df):
    
    df_copy = df.copy()
    mask = (df_copy['GRA_DESCAR'] > 0) & (df_copy['HUM_DESCAR'] > 68)
    df_copy = df_copy[mask]

    df_copy["GRASA_H75"] = (df_copy["GRA_DESCAR"] * (100 - 75)) / (100 - df_copy["HUM_DESCAR"])
    df_copy["TMXGRAN"] = df_copy["DECLARA"] * df_copy["GRASA_H75"]
    df_copy["TMXGRAS"] = df_copy["DECLARA"] * df_copy["GRA_DESCAR"]

    df_copy_grouped = df_copy.groupby("PLANTA", as_index=False).agg(
        TM=('DECLARA', 'sum'),
        TMXGRAN=('TMXGRAN', 'sum'),
        TMXGRAS=('TMXGRAS', 'sum')
    )   
    df_copy_grouped["GRASA_H75"] = df_copy_grouped["TMXGRAN"] / df_copy_grouped["TM"]
    df_copy_grouped["GRASA"] = df_copy_grouped["TMXGRAS"] / df_copy_grouped["TM"]
    df_copy_grouped["PLANTA"] = df_copy_grouped["PLANTA"].str.upper()
    return df_copy_grouped[["PLANTA", "GRASA_H75", "GRASA"]]

def asignar_calidades(claves, valores, valor_comparativo):

    valores_finales = []
    for valor_comp in valor_comparativo:
        cont = 0
        for clave, valor in zip(claves, valores):
            if valor_comp <= clave:
                cont += 1
                valores_finales.append(valor)
                break   
        if cont == 0:
            valores_finales.append('D')
    return valores_finales

def asignar_receta(claves, valores, valor_comparativo):

    valores_finales = []
    max_valor = np.array(valores).max()
    for valor_comp in valor_comparativo:
        cont = 0
        for clave, valor in zip(claves, valores):
            if valor_comp <= clave:
                cont += 1
                valores_finales.append(valor)
                break   
        if cont == 0:
            valores_finales.append(max_valor)
    return valores_finales

def asignar_valores(claves, valores, valor_comparativo):

    valores_finales = []
    for valor_comp in valor_comparativo:
        for clave, valor in zip(claves, valores):
            if valor_comp == clave:
                valores_finales.append(valor)
                break   

    return np.array(valores_finales)

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

def weighted_average(df, values, weights):
    return (df[values] * df[weights]).sum() / df[weights].sum()

def evaluar_condicion(row, df_adicional):
    condicion_tdc_min = (np.vectorize(eval)((str(row["TDC_Descarga_Actual"]) + df_adicional["UmbralTDC_Min"])))
    condicion_tdc_max = (np.vectorize(eval)((str(row["TDC_Descarga_Actual"]) + df_adicional["UmbralTDC_Max"])))
    condicion_juveniles = (np.vectorize(eval)((str(row["JuvenilesPonderado"]) + df_adicional["RangoJuvenilesPonderado"])))
    mask = (df_adicional["EstadoDeFrio"] == row["TipoEP"]) & (np.vectorize(eval)((str(row["Volumen"]) + df_adicional["RangoToneladasDeclaradas"]))) & (df_adicional["SistemaAbsorbente"] == row["sistema_absorbente"]) & condicion_tdc_min & condicion_tdc_max & condicion_juveniles
    if len(df_adicional[mask]) > 0:
        return df_adicional.loc[mask, 'RAP'].values[0]
    else:
        return df_adicional.loc[df_adicional["EstadoDeFrio"] == row["TipoEP"], 'RAP'].mean()

def asignar_calidad(row, calidad_ic_df):
    """
    Asigna la calidad a la fila dada.
    """
    if row["TVN_IC"] > 0:
        return calidad_ic_df.loc[calidad_ic_df["Umbral TVN IC"] > row["TVN_IC"], "Calidad"].values[0]
    else:
        return None
    
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

def group_intervals(intervalos_df, stock_minimo_pama, stock_minimo_mp):

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
    
    # Cambio 17.11.2025: la logica incluye volumen pama y de materia prima
    resultados_pp_grouped["flg_min"] = 0

    mask = (
        (resultados_pp_grouped["VolumenPama"] >= stock_minimo_pama) &
        (resultados_pp_grouped["VolumenAcumulado"] >= stock_minimo_mp)
    )

    resultados_pp_grouped.loc[mask, "flg_min"] = 1
    
    # resultados_pp_grouped["flg_min"] = 0
    # mask = resultados_pp_grouped["VolumenPama"] > stock_minimo_pama
    # resultados_pp_grouped.loc[mask, 'flg_min'] = 1

    return resultados_pp_grouped

def cook(descargas, status_init, threshold_degradacion=3):
    # Estado inicial de pozas
    first_hour = status_init['NumHora'].min()
    initial_pools = status_init[status_init['NumHora'] == first_hour][
        ['Poza', 'StockActual', 'TVN_Actual', 'Capacidad_Pozas']
    ].copy()
    initial_pools.columns = ['Poza', 'Stock', 'TVN_Actual', 'Capacidad']
    initial_pools.set_index('Poza', inplace=True)

    # Preparar simulación
    pools = initial_pools.copy()
    records = []

    min_hora = descargas['NumHora'].min()
    max_hora = descargas['NumHora'].max()

    for hora in range(min_hora, max_hora + 1):
        pools['StockAlimentado'] = 0.0
        pools['sum_vol_tvn'] = 0.0
        
        desc_hora = descargas[descargas['NumHora'] == hora]
        for _, row in desc_hora.iterrows():
            remaining = row['Volumen']
            tvn_in = row['TDC_Descarga_Actual']
            
            while remaining > 0:
                # Determinar espacio disponible en cada poza
                pools['Disponible'] = pools['Capacidad'] - pools['Stock']
                available = pools[pools['Disponible'] > 0]
                if available.empty:
                    raise ValueError(f"No hay capacidad disponible en hora {hora}")
                
                # Filtrar candidatos según umbral de TVN
                candidatos = available[abs(available['TVN_Actual'] - tvn_in) <= threshold_degradacion]
                if candidatos.empty:
                    candidatos = available  # si ninguno cumple umbral, considerar todos
                
                # Selección por mínima diferencia de TVN
                diffs = abs(candidatos['TVN_Actual'] - tvn_in).fillna(threshold_degradacion + 1)
                selected = diffs.idxmin()
                
                # Volumen a asignar
                alloc = min(remaining, pools.at[selected, 'Disponible'])
                
                # Actualizar alimentación y TVN ponderado
                pools.at[selected, 'StockAlimentado'] += alloc
                pools.at[selected, 'sum_vol_tvn']     += alloc * tvn_in
                
                old_stock = pools.at[selected, 'Stock']
                old_tvn   = pools.at[selected, 'TVN_Actual']
                new_stock = old_stock + alloc
                new_tvn   = ((old_tvn * old_stock) + (tvn_in * alloc)) / new_stock if pd.notna(old_tvn) else tvn_in
                
                pools.at[selected, 'Stock']      = new_stock
                pools.at[selected, 'TVN_Actual'] = new_tvn
                
                remaining -= alloc
        
        # Calcular promedio TVN alimentado
        pools['TVNAlimentado'] = pools['sum_vol_tvn'] / pools['StockAlimentado']
        
        # Registrar estado
        for poza, datos in pools.iterrows():
            records.append({
                'NumHora': hora,
                'Poza': poza,
                'StockActual': datos['Stock'],
                'TVN_Actual': datos['TVN_Actual'],
                'StockAlimentado': datos['StockAlimentado'],
                'TVNAlimentado': datos['TVNAlimentado'],
                'Capacidad_Pozas': datos['Capacidad']
            })

    return pd.DataFrame(records)

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

def hallar_volumen_minimo_pama(velocidades_volumen_df, grasa_h75):
    mask = (np.vectorize(eval)(str(grasa_h75) + velocidades_volumen_df["GRASA MP"]))
    stock_agua_minimo = velocidades_volumen_df.loc[mask, "VOLUMEN MIN ARRANQUE"].values[0]
    return stock_agua_minimo

def hallar_velocidades_base(velocidades_volumen_df, grasa_h75, volumen_materia_prima):
    mask = (np.vectorize(eval)(str(grasa_h75) + velocidades_volumen_df["GRASA MP"])) & (velocidades_volumen_df["MIN - TM"] <= volumen_materia_prima) & (velocidades_volumen_df["MAX - TM"] >= volumen_materia_prima)
    v_inicial = velocidades_volumen_df.loc[mask, "VELOCIDAD DE ARRANQUE"].values[0]
    v_medio = velocidades_volumen_df.loc[mask, "VELOCIDAD MAXIMA"].values[0]
    v_intermedia_inicial =  velocidades_volumen_df.loc[mask, "VEL_INTERMEDIA1"].values[0]
    v_final = velocidades_volumen_df.loc[mask, "VELOCIDAD DE CIERRE"].values[0]
    v_intermedia_final = velocidades_volumen_df.loc[mask, "VEL_INTERMEDIA2"].values[0]
    return v_inicial, v_intermedia_inicial, v_medio, v_intermedia_final, v_final

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
from typing import Dict, List

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

cnxn_mio = pyodbc.connect('DRIVER={SQL Server};SERVER=srv-db-east-us003.database.windows.net;DATABASE=db_cfa_prd01;UID=userdbowner;PWD=$P4ssdbowner01#;')
cnxn_juv = pyodbc.connect('DRIVER={SQL Server};SERVER=srv-db-east-us-tasa-his-02.database.windows.net;DATABASE=db_bi_production_prd;UID=userpowerbi;PWD=#p4ssw0rdp0w3rb1#;')

query = """
SELECT id, fecha, ID_PLANTA, FECCREACION FROM dbo.PlanDescargaModeloLog 
WHERE FlagPPyPZ = 1 ORDER BY FECCREACION DESC;
"""
pd_logs = pd.read_sql(query, cnxn_mio).sort_values("FECCREACION", ascending=False).reset_index(drop=True)
pd_logs = pd_logs.drop_duplicates(subset=["fecha", "ID_PLANTA"], keep="first")
# ids_pds = set(pd.read_sql(query, cnxn_mio)["id"].tolist())
ids_pds = set(pd_logs["id"].tolist())

query = """
SELECT id_pd FROM dbo.PlanProduccionLog
"""
ids_pps = set(pd.read_sql(query, cnxn_mio)["id_pd"].tolist())

ids_nuevos_pds = list(ids_pds - ids_pps)

# Stock minimo de MP para arranque
stock_minimo_MP_planta = pd.DataFrame({
    'PLANTA': ['MALABRIGO', 'CHIMBOTE', 'SUPE', 'VEGUETA', 'CALLAO', 'PISCO SUR'],
    'STOCK_MINIMO_MP': [400, 365, 250, 325, 400, 325]
})

for id in ids_nuevos_pds:
# for id in [6241]:
    print("ID: ", id)
    # if id in [2631, 3310, 2289, 2291, 3315, 3316, 3904, 3893, 3894, 3895, 3897, 4507] or id < 3763: # Los 3 ultimos sale error de len
    #     continue
    try:
        query_pd = """
        SELECT A.*, B.id_planta, 
        B.FlagPPyPZ,
        case when TE = 'Propio C/Frio' then 'RC'
        when TE = 'Propio S/Frio' then 'SF'
        else 'SF' end as TIPO,
        0.1 AS JuvenilesPonderado
        FROM PlanDescargaOutput A
        LEFT JOIN PlanDescargaModeloLog B
        ON A.IDLog=B.id
        WHERE B.FlagPPyPZ = 1;"""
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


        # Cambio 17.11.2025: La grasa ahora se toma de las tablas
        # query = """
        # select  top 60 Marea,FECHA ,TRIM(PLANTA) PLANTA , TVN_DES ,DECLARA, DESCARG ,GRA_DESCAR , HUM_DESCAR 
        # from sap.ZQM_DESC_MP_CHI2 zdmc2 
        # where F_ARRIB  >cast('2024-04-21' as date) AND DESCARG >0 and GRA_DESCAR > 0
        # order by Marea desc;
        # """

        # grasa_df_original = pd.read_sql(query, cnxn_juv)
        # grasa_df = grasa_df_original.copy()
        # grasa_df["PLANTA"] = grasa_df["PLANTA"].str.upper()
        # grasa_df = get_final_grasa_df(grasa_df)[["PLANTA", "GRASA_H75"]]

        # to_replace = {
        #     "MALABRIGO SUR": "MALABRIGO",
        #     "CALLAO NORTE": "CALLAO",
        # }
        # grasa_df["PLANTA"] = grasa_df["PLANTA"].replace(to_replace)

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
        
        # Cambio 17.11.2025: La grasa se tomara a partir de KK_calidades
        grasa_df_original = ValoresFijosPP_orig[["PLANTA", "PROTEINA_HARINA", "Grasa_MP"]].copy()
        grasa_df = grasa_df_original.copy()

        query = """
        SELECT * FROM dbo.Balance_KSA"""
        balance_ksa_df_orig = pd.read_sql(query, cnxn_mio)

        query = """
        SELECT * FROM dbo.FECHA_AJUSTADO_PP"""
        hora_arranque_df = pd.read_sql(query, cnxn_mio)
        hora_arranque_df["Planta"] = np.where(hora_arranque_df["Planta"].isin(["Pisco", "PISCO"]), "Pisco sur" , hora_arranque_df["Planta"])

        print("ID: ", id)

        now = datetime.now()
        mask = discharge_plan_df_original["IdLog"] == id
        discharge_plan_df = discharge_plan_df_original[mask].copy()
        fecha_batch = discharge_plan_df["Fecha"].unique()[0]
        try:
            planta_selected = discharge_plan_df["id_planta"].unique()[0]
        except:
            continue
        print('PLAN DE PRODUCCIÓN ID:', id, planta_selected)
        
        # Cambio 17.11.2025: Grasa de las tablas
        # try:
        #     grasa_h75 = grasa_df[grasa_df["PLANTA"] == planta_selected]["GRASA_H75"].values[0]
        # except:
        #     grasa_h75 = grasa_df["GRASA_H75"].mean()
        # print("Grasa H75", grasa_h75)
        
        # PARAMETRO_INICIAL: La grasa H75 para la planta seleccionada
        try:
            grasa_h75 = grasa_df["Grasa_MP"][grasa_df["PLANTA"] == planta_selected].values[0]
        except:
            grasa_h75 = grasa_df["Grasa_MP"][grasa_df["PLANTA"] == planta_selected].mean()
            print("aaaaa")
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
        
        # PARAMETRO PROCESO: Stock minimo de MP segun planta
        stock_minimo_mp = stock_minimo_MP_planta["STOCK_MINIMO_MP"][stock_minimo_MP_planta["PLANTA"]==planta_selected].values[0]
        print("Stock Minimo MP:", stock_minimo_mp)
        
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
        resultados_pp_grouped = group_intervals(intervals_rap, stock_minimo_pama, stock_minimo_mp)
        volumen_ab = resultados_pp_grouped["VolumenPama"].max()
        print("Volumen AB", volumen_ab)
        
        # Definición de Hora de Arranque

        resultados_pp_grouped["flg_min_acum"] = resultados_pp_grouped["flg_min"].cumsum()
        
        print(arranque)
        
        try:
            if len(arranque) == 0:
                index_arranque = resultados_pp_grouped[resultados_pp_grouped['flg_min'] == 1].iloc[0].name
        except:
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            logs = pd.DataFrame({
                'id_pd': [id],
                'StatusOptimizacion': ['0'],
                'Mensaje': ['No se alcanzó volumen mínimo de arranque'],
                'Timestamp': [timestamp],
            })
            if ambiente == 'prod':
                print("INGRESANDO LOGS PP A SQL")
                logs.to_sql('PlanProduccionLog', enginePC, if_exists='append', index=False)
            else:
                print("No se alcanzó volumen mínimo de arranque")
            continue
        
        # Cambio 17.11.2025: La hora mas cercana debe ser seteada un valor hacia arriba
        if len(arranque) > 0:
            idx_min = (resultados_pp_grouped['Intervalo Poza'] - arranque.values[0]).abs().idxmin()
            # cook_start_hour = resultados_pp_grouped.loc[idx_min]["NumHora"]
            
            print("Se toma siempre la siguiente hora hacia arriba al ajuste del usuario")

            # hora objetivo
            target = arranque.values[0]

            # filtrar solo horas >= target
            mayores = resultados_pp_grouped[resultados_pp_grouped['Intervalo Poza'] >= target]

            if len(mayores) > 0:
                # primera hora hacia arriba
                idx = mayores['Intervalo Poza'].idxmin()
                cook_start_hour = resultados_pp_grouped.loc[idx, "NumHora"]
            else:
                # si no hay mayores, tomar la última disponible
                cook_start_hour = resultados_pp_grouped.loc[idx_min]["NumHora"] + 1
            
        else:
            cook_start_hour = resultados_pp_grouped.loc[index_arranque, 'NumHora'] + 1
            

        try:
            index_arranque = resultados_pp_grouped[resultados_pp_grouped['flg_min'] == 1].iloc[0].name
            hora_cook = resultados_pp_grouped.loc[index_arranque, 'NumHora'] + 1
            cook_start_hour_model = resultados_pp_grouped.loc[resultados_pp_grouped["NumHora"] == hora_cook, "Intervalo Poza"].values[0]
        except:   
            cook_start_hour_model = None
        
        descargas = intervals_rap.copy()
        status_init = status_pozas.copy()

        ## Calculo de descarga, almacenamiento y cocción

        first_hour = status_init['NumHora'].min()
        initial = (status_init[status_init['NumHora'] == first_hour]
                .dropna(subset=['Poza'])
                .copy())
        initial['Poza'] = initial['Poza'].astype(int)
        initial = initial.set_index('Poza')[['StockActual', 'TVN_Actual', 'Capacidad_Pozas']]
        initial.columns = ['Stock', 'TVN_Actual', 'Capacidad']

        # Copias para simulación
        pools = initial.copy()

        # Almacenar resultados
        records_desc = []
        cook_records = []

        def sim_descarga_coccion(pools_df, descargas_df, speeds_fixed, cook_start_hour, threshold_degradacion):
            pools = pools_df.copy()
            descargas = descargas_df.copy()
            # Rango total de horas a simular
            min_hora, max_descarga_hora = descargas['NumHora'].min(), descargas['NumHora'].max()
            max_coccion_hora = cook_start_hour + len(speeds_fixed) - 1
            max_hora = max(max_descarga_hora, max_coccion_hora)

            # Índice para speeds_fixed
            cook_idx = 0

            for hora in range(min_hora, max_hora + 1):
                # --- DESCARGA ---
                # Inicializar métricas de descarga para esta hora
                pools['StockAlimentado'] = 0.0
                pools['sum_vol_tvn'] = 0.0
                pools['TVN_Actual'].fillna(0, inplace=True)
                
                # Si hay descargas en esta hora, asignarlas
                for _, row in descargas[descargas['NumHora'] == hora].iterrows():
                    remaining, tvn_in = row['Volumen'], estimate_tvn_sin_frio(row['TDC_Descarga_Actual']) if row["TipoEP"] == 'SF' else estimate_tvn_con_frio(row['TDC_Descarga_Actual'])
                    while remaining > 0:
                        pools['Disponible'] = pools['Capacidad'] - pools['Stock']
                        avail = pools[pools['Disponible'] > 0]
                        if avail.empty:
                            raise ValueError(f"No hay capacidad disponible en hora {hora}")
                        cand = avail[abs(avail['TVN_Actual'] - tvn_in) <= threshold_degradacion]
                        if cand.empty:
                            cand = avail
                        sel = abs(cand['TVN_Actual'] - tvn_in).idxmin()
                        alloc = min(remaining, pools.at[sel, 'Disponible'])
                        # Actualizar descarga
                        pools.at[sel, 'StockAlimentado'] += alloc
                        pools.at[sel, 'sum_vol_tvn']     += alloc * tvn_in
                        old_s, old_t = pools.at[sel, 'Stock'], pools.at[sel, 'TVN_Actual']
                        new_s = old_s + alloc
                        pools.at[sel, 'Stock'] = new_s
                        pools.at[sel, 'TVN_Actual'] = ((old_t * old_s + tvn_in * alloc) / new_s
                                                    if old_s > 0 else tvn_in)
                        remaining -= alloc

                pools['TVNAlimentado'] = pools['sum_vol_tvn'] / pools['StockAlimentado']
                
                # Registrar estado de descarga de cada poza
                for p, d in pools.iterrows():
                    records_desc.append({
                        'NumHora': hora,
                        'Poza': p,
                        'StockActual': d['Stock'],
                        'TVN_Actual': d['TVN_Actual'],
                        'StockAlimentado': d['StockAlimentado'],
                        'TVNAlimentado': d['TVNAlimentado'],
                        'Capacidad_Pozas': d['Capacidad']
                    })
                
                # --- COCCIÓN (si corresponde) ---
                if hora >= cook_start_hour and cook_idx < len(speeds_fixed):
                    speed = speeds_fixed[cook_idx]
                    # Calcular TVN promedio ponderado
                    total_stock = pools['Stock'].sum()
                    avg_tvn = (pools['TVN_Actual'] * pools['Stock']).sum() / total_stock if total_stock > 0 else 0
                    pools['Group'] = np.where(pools['TVN_Actual'] >= avg_tvn, 2, 1)
                    order = pools.sort_values(['Group', 'TVN_Actual'], ascending=[True, False])
                    
                    rem = speed
                    for p, row in order.iterrows():
                        if rem <= 0:
                            break
                        pre_stock = row['Stock']
                        cook_vol = min(pre_stock, rem)
                        if cook_vol <= 0:
                            continue
                        
                        tvn_pre = row['TVN_Actual']
                        # Reducir stock
                        pools.at[p, 'Stock'] = pre_stock - cook_vol
                        # Incrementar TVN en 1 unidad si queda stock
                        if pools.at[p, 'Stock'] > 0:
                            pools.at[p, 'TVN_Actual'] = tvn_pre + 1
                        tvn_post = pools.at[p, 'TVN_Actual']
                        
                        cook_records.append({
                            'Hora': hora,
                            'Poza': p,
                            'Group': int(row['Group']),
                            'TVN_PreCoccion': tvn_pre,
                            'TVN_PostCoccion': tvn_post,
                            'StockPreCoccion': pre_stock,
                            'VolCocido': cook_vol,
                            'StockPostCoccion': pre_stock - cook_vol,
                            'Velocidad': speed
                        })
                        rem -= cook_vol
                    
                    cook_idx += 1
                
                # --- ENVEJECIMIENTO POR HORA ---
                mask = pools['Stock'] > 0
                pools.loc[mask, 'TVN_Actual'] += 1
            return pd.DataFrame(records_desc), pd.DataFrame(cook_records)


        status_pozas_sim, df_coccion = sim_descarga_coccion(pools, intervals_rap, speeds_fixed, cook_start_hour, threshold_degradacion)

        df_coccion_grouped = df_coccion.groupby(["Hora"], as_index=False).agg(
            TVN_Pozas=('TVN_PreCoccion', 'mean'),
            TVN_IC=('TVN_PostCoccion', 'mean'),
            VolCocido=('VolCocido', 'sum'),
            VelocidadCoccion=('Velocidad', 'mean')
        )
        tvn_pozas = df_coccion_grouped["TVN_Pozas"].to_numpy()
        tvn_ingreso_cocina = df_coccion_grouped["TVN_IC"].to_numpy()
        velocidad_proceso = df_coccion_grouped["VelocidadCoccion"].to_numpy()

        calidades = asignar_calidades(UmbralTVN_Calidades[planta_selected].tolist(), UmbralTVN_Calidades["CALIDAD"].tolist(), tvn_ingreso_cocina)
        receta = asignar_receta(UmbralTVN_Calidades[planta_selected].tolist(), UmbralTVN_Calidades["DOSIFICACION"].tolist(), tvn_ingreso_cocina)
        
        # ——— 3. Parámetros de proceso ———
        fc_solidos            = intervals_rap["FCSolidosGrasayAguaBombeo"].values[0]
        # pct_solidos_licor     = 0.001
        pct_solidos_licor     = balance_ksa_df["%S Flujo de licor"].values[0]
        # pct_solidos_lodo      = 0.08
        pct_solidos_lodo      = balance_ksa_df["%S Flujo de lodo"].values[0]
        # pct_humedad_ksa       = 0.73
        pct_humedad_ksa       = balance_ksa_df["%H Flujo de KSA"].values[0]
        # pct_solidos_efluente  = 0.0004
        pct_solidos_efluente  = balance_ksa_df["%S Flujo de efluente"].values[0]
        # humedad_harina        = 7
        humedad_harina = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["HumedadHarina"].tolist(), calidades)[0] * 100
        # grasa_h75             = 8
        # aportante_kp          = 24
        # aportante_ks          = 8
        # aportante_cc          = 14

        tvn_ponderado = resultados_pp_grouped.loc[resultados_pp_grouped["TVNPonderado"].notna(), "TVNPonderado"].values
        solidos_ab = (
            -445.24
            + tvn_ponderado * 294
            - 606.62 * grasa_h75
            +  88.804 * grasa_h75**2
        )
        grasa_ab = -47.169 + 0.943 * solidos_ab 

        # Pre‑cálculo de sólidos antes del clarificador (vector constante)
        solidos_ab_ic = (
            -445.24
            + tvn_pozas[:-2] * 294
            - 606.62 * grasa_h75
            +  88.804 * grasa_h75**2
        ) * fc_solidos

        n = velocidad_proceso.size
        EPS = 1e-8
        
        # ——— 4. Bounds para d_i: 0+ε ≤ d_i ≤ min(límite base_seca, límite flujo_ing) ———
        # d_max_base = limite_base_seca * 4 / velocidad_proceso
        # d_max_flow = (limite_flujo_ic * solidos_ab_ic * 4) / (1e6 * velocidad_proceso)
        # bounds    = [(1e-4, min(d_max_base[i], d_max_flow[i])) for i in range(n)]

        # ——— 5. Función que calcula el vector de proteína_harina dado d ———
        def compute_protein(d):
            # Flujos iniciales
            flujo_de_ksa = velocidad_proceso * d
            base_seca    = flujo_de_ksa / 4
            flujo_ing    = base_seca * 1e6 / (solidos_ab_ic + EPS)

            # Parámetros intermedios
            pct_gr_ef    = pct_solidos_efluente / 2
            grasa_ab_ic  = (-47.169 + 0.943 * solidos_ab_ic) * fc_solidos
            pct_gr_ic    = grasa_ab_ic * 0.5

            # Flujos de lodo y efluente
            flujo_lodo   = flujo_ing * ((solidos_ab_ic * 1e-6) - pct_solidos_efluente) / (pct_solidos_lodo - pct_solidos_efluente + EPS)
            flujo_ef     = flujo_ing - flujo_lodo
            pct_gr_li    = pct_solidos_licor / 2
            pct_gr_ld    = (flujo_ing * pct_gr_ic / 1e6 - flujo_ef * pct_gr_ef) / (flujo_lodo + EPS)

            # Flujos de KSA y licor
            flujo_ksa    = flujo_lodo * (
                            ((1 - pct_gr_ld - pct_solidos_lodo)
                            - (1 - pct_gr_li - pct_solidos_licor))
                        / (pct_humedad_ksa - (1 - pct_gr_li - pct_solidos_lodo) + EPS)
                        )
            flujo_licor  = flujo_lodo - flujo_ksa
            pct_gr_ksa   = (flujo_lodo * pct_gr_ld - flujo_licor * pct_solidos_licor) / (flujo_ksa + EPS)

            proteina_ksa = 1 - humedad_ksa - pct_gr_ksa - ceniza_ksa
            # Composición integral (KK + dosificación)
            denom             = aportante_kp + aportante_ks + aportante_cc + d * 100 + EPS
            grasa_integral    = (aportante_kp * grasa_kk_prensa
                            + aportante_ks * grasa_kk_separadora
                            + aportante_cc * grasa_kk_cc
                            + d * pct_gr_ksa * 100) / denom
            # ceniza_integral   = (aportante_kp * ceniza_kk_prensa
            #                 + aportante_ks * ceniza_kk_separadora
            #                 + aportante_cc * ceniza_kk_cc
            #                 + d * ceniza_ksa * 100) / denom
            proteina_integral = (aportante_kp * proteina_kk_prensa
                            + aportante_ks * proteina_kk_separadora
                            + aportante_cc * proteina_kk_cc
                            + d * proteina_ksa * 100) / denom

            # Proteína de harina final
            proteina_harina = proteina_integral / (
                (grasa_integral + ceniza_integral + proteina_integral) / (100 - humedad_harina) + EPS
            )
            return proteina_harina

        # ——— 6. Función objetivo y restricciones ———
        # def objective(d):
        #     return -compute_protein(d).sum()  # minimizamos el negativo para maximizar
        
        # constraints = [
        #     {'type': 'ineq',
        #     'fun': lambda d, i=i: compute_protein(d)[i] - compute_protein(d)[i+1]}
        #     for i in range(n-1)
        # ]
        # # 6.2 Suma de flujo_ingreso_clarificador 
        # def sum_flux(d):
        #     base_seca = velocidad_proceso * d / 4
        #     flujo     = base_seca * 1e6 / (solidos_ab_ic + EPS)
        #     return flujo.sum()

        # constraints.append({
        #     'type': 'eq',
        #     # 'fun': lambda d: 5000 - sum_flux(d)
        #     'fun': lambda d: sum_flux(d) - np.round(volumen_ab, 1)
        # })

        # constraints.append({
        #         'type': 'ineq',
        #         'fun': lambda d: compute_protein(d) - 65.001
        #     })

        # ——— 7. Ejecutar optimización ———
        # x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
        # tol = 1e-3
        # res = minimize(
        #     objective,
        #     x0,
        #     method='SLSQP',
        #     bounds=bounds,
        #     constraints=constraints,
        #     options={'maxiter': 1000, 'ftol': tol}
        # )


        # for i, constraint in enumerate(constraints):
        #     # value = constraint'fun'
        #     # pass
        #     value = constraint['fun'](res.x)
        #     print(constraint)
        #     print(value)
        #     # value = constraint['fun'](res.xf"Restricción {i+1}: valor = {value}, {'cumple' if value >= 0 else 'no cumple'}")
        #     print(f"Restricción {i+1}: valor = {value}, {'cumple' if value >= 0 else 'no cumple'}")

        # ——— 8. Calcular todas las variables con d_opt ———
        resultados_test = calibrar(
            velocidad_proceso[:-2], receta[:-2], tvn_pozas[:-2],
            volumen_ab, limite_base_seca, limite_flujo_ic,
            V_tot, v_term, grasa_h75, fc_solidos
        )
        print(resultados_test)
        # d_opt = np.array(receta)
        # f_ksa    = velocidad_proceso * d_opt
        # base_sec = f_ksa / 4
        # flujo_ing = base_sec * 1e6 / (solidos_ab_ic + EPS)
        
        velocidad_proceso = np.array(resultados_test[0])
        d_opt = np.array(resultados_test[1])
        f_ksa = np.array(resultados_test[3])
        base_sec = np.array(resultados_test[4])
        flujo_ing = np.array(resultados_test[6])

        n_lags = 2
        if len(solidos_ab_ic) < len(flujo_ing): # TODO: Mejorar esta parte
            solidos_ab_ic = np.append(solidos_ab_ic, solidos_ab_ic[-1])
            n_lags = 1
            
        flujo_lodo = flujo_ing * ((solidos_ab_ic * 1e-6) - pct_solidos_efluente) / (pct_solidos_lodo - pct_solidos_efluente + EPS)
        flujo_ef   = flujo_ing - flujo_lodo
        # flujo_ksa2 = flujo_lodo * (((1 - (flujo_lodo*0) - pct_solidos_lodo) - (1 - pct_solidos_licor - pct_solidos_lodo)) / (pct_humedad_ksa - (1 - pct_solidos_licor - pct_solidos_lodo) + EPS))
        grasa_ab_ic  = (-47.169 + 0.943 * solidos_ab_ic) * fc_solidos
        pct_grasa_ic = grasa_ab_ic * 0.5
        pct_grasa_efluente = pct_solidos_efluente / 2
        pct_grasa_lodo = (flujo_ing * pct_grasa_ic / 1e6 - flujo_ef * pct_grasa_efluente) / (flujo_lodo + EPS)

        pct_grasa_licor = pct_solidos_licor / 2
        flujo_ksa2 = flujo_lodo * (((1 - pct_grasa_lodo - pct_solidos_lodo) - (1 - pct_grasa_licor - pct_solidos_licor))/(pct_humedad_ksa - (1 - pct_grasa_licor - pct_solidos_licor) + EPS))
        flujo_lic  = flujo_lodo - flujo_ksa2
        # pct_gr_ksa = (flujo_lodo * ((flujo_ing * 0) - flujo_ef * (pct_solidos_efluente/2)) - flujo_lic * pct_solidos_licor) / (flujo_ksa2 + EPS)
        pct_gr_ksa   = (flujo_lodo * pct_grasa_lodo - flujo_lic * pct_solidos_licor) / (flujo_ksa2 + EPS)

        ceniza_kk_prensa = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Ceniza- KK PRENSA"].tolist(), calidades)
        ceniza_kk_prensa = ceniza_kk_prensa[:-n_lags]  # Exclude last two values for KSA
        grasa_kk_prensa = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Grasa- KK PRENSA"].tolist(), calidades)/100
        grasa_kk_prensa = grasa_kk_prensa[:-n_lags]  # Exclude last two values for KSA
        humedad_kk_prensa = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Humedad- KK PRENSA"].tolist(), calidades) / 100
        humedad_kk_prensa = humedad_kk_prensa[:-n_lags]  # Exclude last two values for KSA
        proteina_kk_prensa = 1 - ceniza_kk_prensa - grasa_kk_prensa - humedad_kk_prensa
        # proteina_kk_prensa = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["% Proteina- KK PRENSA"].tolist(), calidades) / 100
        # humedad_kk_prensa = 1 - ceniza_kk_prensa - grasa_kk_prensa - proteina_kk_prensa

        grasa_kk_separadora = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Grasa- KK SEPARADORA"].tolist(), calidades) / 100
        grasa_kk_separadora = grasa_kk_separadora[:-n_lags]  # Exclude last two values for KSA
        ceniza_kk_separadora = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Ceniza- KK SEPARADORA"].tolist(), calidades)
        ceniza_kk_separadora = ceniza_kk_separadora[:-n_lags]  # Exclude last two values for KSA
        # proteina_kk_separadora = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["% Proteina- KK SEPARADORA"].tolist(), calidades) / 100
        humedad_kk_separadora = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Humedad- KK SEPARADORA"].tolist(), calidades) / 100
        humedad_kk_separadora = humedad_kk_separadora[:-n_lags]  # Exclude last two values for KSA
        proteina_kk_separadora = 1 - ceniza_kk_separadora - grasa_kk_separadora - humedad_kk_separadora

        grasa_kk_cc = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Grasa- CONCENTRADO"].tolist(), calidades) / 100
        grasa_kk_cc = grasa_kk_cc[:-n_lags]  # Exclude last two values for KSA
        ceniza_kk_cc = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Ceniza- CONCENTRADO"].tolist(), calidades)
        ceniza_kk_cc = ceniza_kk_cc[:-n_lags]  # Exclude last two values for KSA
        # proteina_kk_cc = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["% Proteina-CONCENTRADO"].tolist(), calidades) / 100
        humedad_kk_cc = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Humedad- CONCENTRADO"].tolist(), calidades) / 100
        humedad_kk_cc = humedad_kk_cc[:-n_lags]  # Exclude last two values for KSA
        proteina_kk_cc = 1 - ceniza_kk_cc - grasa_kk_cc - humedad_kk_cc

        ceniza_ksa = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Ceniza- KSA"].tolist(), calidades)
        ceniza_ksa = ceniza_ksa[:-n_lags]
        # proteina_ksa = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["% Proteina- KSA"].tolist(), calidades) / 100
        humedad_ksa = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["%Humedad- KSA"].tolist(), calidades) / 100
        humedad_ksa = humedad_ksa[:-n_lags]

        aportante_kp = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["Aportantes_Porc_KP"].tolist(), calidades)[:-n_lags]
        aportante_ks = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["Aportantes_Porc_KS"].tolist(), calidades)[:-n_lags]
        aportante_cc = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["Aportantes_Porc_CC"].tolist(), calidades)[:-n_lags]

        ceniza_integral = asignar_valores(ValoresFijosPP["PROTEINA-HARINA"].tolist(), ValoresFijosPP["ceniza_kk_integral"].tolist(), calidades)[:-n_lags]

        proteina_harina = compute_protein(d_opt)

        # Calculo de otras variables finales

        denom             = aportante_kp + aportante_ks + aportante_cc + d_opt * 100 + EPS
        grasa_integral    = (aportante_kp * grasa_kk_prensa
                            + aportante_ks * grasa_kk_separadora
                            + aportante_cc * grasa_kk_cc
                            + d_opt * pct_gr_ksa * 100) / denom

        # ceniza_integral   = (aportante_kp * ceniza_kk_prensa
        #                     + aportante_ks * ceniza_kk_separadora
        #                     + aportante_cc * ceniza_kk_cc
        #                     + d_opt * ceniza_ksa * 100) / denom

        humedad_integral   = (aportante_kp * humedad_kk_prensa
                            + aportante_ks * humedad_kk_separadora
                            + aportante_cc * humedad_kk_cc
                            + d_opt * humedad_ksa * 100) / denom
        
        proteina_ksa = 1 - humedad_ksa - pct_gr_ksa - ceniza_ksa

        proteina_integral = (aportante_kp * proteina_kk_prensa
                            + aportante_ks * proteina_kk_separadora
                            + aportante_cc * proteina_kk_cc
                            + d_opt * proteina_ksa * 100) / denom

        ceniza_harina = ceniza_integral / (
            (grasa_integral + ceniza_integral + proteina_integral) / (100 - humedad_harina) + EPS
        )
        grasa_harina = grasa_integral / (
            (grasa_integral + ceniza_integral + proteina_integral) / (100 - humedad_harina) + EPS
        )
        # ——— 9. Crear DataFrame y exportar a Excel ———
        df_velocidades_asociados = pd.DataFrame({
            'velocidad_proceso':         df_coccion_grouped["VelocidadCoccion"].to_numpy(),
            'Calidad':                   calidades,
            "Grasa_h75":                 grasa_h75,
            })
        df = pd.DataFrame({
            # 'velocidad_proceso':         velocidad_proceso,
            # 'Calidad':                   calidades,
            # "Grasa_h75":                 grasa_h75,
            'dosificacion':              d_opt,
            # 'solidos_ab':                solidos_ab,
            # 'grasa_ab':                  grasa_ab,
            'solidos_ab_ic':             solidos_ab_ic,
            'grasa_ab_ic':               grasa_ab_ic,
            'flujo_de_ksa':              f_ksa,
            'base_seca':                 base_sec,
            'flujo_ingreso_clarificador':flujo_ing,
            'flujo_lodo':                flujo_lodo,
            'flujo_efluente':            flujo_ef,
            'flujo_ksa':                 flujo_ksa2,
            'flujo_licor':               flujo_lic,

            'grasa_kk_prensa':         grasa_kk_prensa,
            'ceniza_kk_prensa':     ceniza_kk_prensa,
            'humedad_kk_prensa': humedad_kk_prensa,
            'proteina_kk_prensa': proteina_kk_prensa,

            'grasa_kk_separadora':     grasa_kk_separadora,
            'ceniza_kk_separadora': ceniza_kk_separadora,
            'proteina_kk_separadora': proteina_kk_separadora,
            'humedad_kk_separadora': humedad_kk_separadora,

            'grasa_kk_cc':          grasa_kk_cc,
            'ceniza_kk_cc':      ceniza_kk_cc,
            'proteina_kk_cc':     proteina_kk_cc,
            'humedad_kk_cc':    humedad_kk_cc,

            'grasa_ksa':                 pct_gr_ksa,
            'ceniza_ksa':                ceniza_ksa,
            'proteina_ksa':             proteina_ksa,
            'humedad_ksa':              humedad_ksa,

            'grasa_integral':            grasa_integral,
            'ceniza_integral':           ceniza_integral,
            'proteina_integral':         proteina_integral,
            'proteina_harina':           proteina_harina,
            'ceniza_harina':           ceniza_harina,
            'grasa_harina':             grasa_harina,
        })

        df_velocidades_asociados["Hora"] = np.arange(cook_start_hour, cook_start_hour + len(df_velocidades_asociados.index))
        df["Hora"] = np.arange(cook_start_hour, cook_start_hour + len(df.index))
        df = pd.merge(df_velocidades_asociados, df, on="Hora", how="left")
        resultados_pp_grouped["solidos_ab"] = np.nan
        resultados_pp_grouped["grasa_ab"] = np.nan
        mask = resultados_pp_grouped["TVNPonderado"].notna()
        resultados_pp_grouped.loc[mask, "solidos_ab"] = solidos_ab
        resultados_pp_grouped.loc[mask, "grasa_ab"] = grasa_ab
        df_final = pd.merge(resultados_pp_grouped[["Intervalo Chata", "TVNPonderado", "VolumenTotal", "Intervalo Poza", "NumHora", "RAPPonderado", "RAPDescarga", "VolumenPama", "solidos_ab", "grasa_ab"]], df_coccion_grouped, left_on="NumHora", right_on="Hora",  how="left")
        df_final_pp = pd.merge(df_final, df, left_on="Hora", right_on="Hora", how="left")
        # print(f"Optimización concluida: {res.success}, mensaje: {res.message}")
        # print(f"Se han exportado todas las variables a: {output_path}")

        # Ruta donde se guardará el Excel}
        
        # df_final_pp["Timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
        # df_final_pp["StatusOptimizacion"] = res.success
        

        # print("Solución candidata:", res.x)
        # # 5) Comprobación detallada
        # for i, c in enumerate(constraints, start=1):
        #     val = np.atleast_1d(c['fun'](res.x))  # forzar array
        #     for j, v in enumerate(val, start=1):
        #         if c['type'] == 'ineq':
        #             if v < -tol:
        #                 print(f"❌ Restricción {i}[{j}] (ineq) violada: fun(x) = {v:.3e} (< 0)")
        #             else:
        #                 print(f"✅ Restricción {i}[{j}] (ineq) satisfecha: fun(x) = {v:.3e}")
        #         else:  # eq
        #             if abs(v) > tol:
        #                 print(f"❌ Restricción {i}[{j}] (eq) violada: |fun(x)| = {abs(v):.3e} (> tol)")
        #             else:
        #                 print(f"✅ Restricción {i}[{j}] (eq) satisfecha: fun(x) = {v:.3e}")

        # print(df_final_pp)
        # output_path = f'outputs/Plan_Produccion_{now.strftime("%Y_%m_%d-%I_%M_%S_%p")}_validation.xlsx'
        # df.to_excel(output_path, index=False)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        logs = pd.DataFrame({
            'id_pd': [id],
            'StatusOptimizacion': ["Prueba"],
            'Mensaje': ["Prueba"],
            'Timestamp': [timestamp],
        })
        df_final_pp["FECHA HORA"] = df_final_pp["Intervalo Poza"].copy()
        df_final_pp["PLANTA"] = planta_selected
        df_final_pp["AVANCE DESCARGA"] = df_final_pp["VolumenTotal"].cumsum()
        df_final_pp["ACUM_ALIMENTACION"] = df_final_pp["VelocidadCoccion"].cumsum()
        df_final_pp["STOCK POZAS"] = df_final_pp["AVANCE DESCARGA"] - df_final_pp["ACUM_ALIMENTACION"]
        df_final_pp["Tipo velocidad"] = np.nan
        df_final_pp['VOL AB'] = df_final_pp['VolumenPama'].diff().fillna(df_final_pp['VolumenPama'])
        df_final_pp["RAP PAMA ACUMULADO"] = df_final_pp["VolumenPama"] / df_final_pp["AVANCE DESCARGA"]
        mask = df_final_pp["Hora"].notna()
        df_final_pp.loc[mask, "Tipo velocidad"] = perfil_lbls
        df_final_pp["FECHA HORA INICIO PROCESO"] = df_final_pp.loc[df_final_pp["STOCK POZAS"].notna(), "Intervalo Poza"].reset_index(drop=True)[0]
        df_final_pp["FECHA HORA FIN PROCESO"] = df_final_pp.loc[df_final_pp["STOCK POZAS"].notna(), ["Intervalo Poza"]].sort_values("Intervalo Poza", ascending=False).reset_index(drop=True)["Intervalo Poza"][0]

        # Filtrar los valores no nulos
        non_null_values = df_final_pp['VelocidadCoccion'].dropna()

        # Obtener el penúltimo valor no nulo
        penultimate_non_null_value = non_null_values.iloc[-2]

        last_valid_index = df_final_pp['VelocidadCoccion'].last_valid_index()
        last_non_null_value = df_final_pp.loc[last_valid_index, 'VelocidadCoccion']
        df_final_pp["FECHA HORA FIN PROCESO"] = df_final_pp["FECHA HORA FIN PROCESO"] + pd.Timedelta(minutes=(last_non_null_value / penultimate_non_null_value) * 60)
        mask = df_final_pp["VelocidadCoccion"].isna()
        df_final_pp["STOCK MIN ARRANQUE"] = df_final_pp.loc[mask, "VolumenTotal"].sum()
        df_final_pp["DURACION BATCH"] = (df_final_pp["FECHA HORA FIN PROCESO"] - df_final_pp["FECHA HORA INICIO PROCESO"]).dt.total_seconds() / 3600

        if ambiente == 'prod':
            print("INGRESANDO LOGS PP A SQL")
            logs.to_sql('PlanProduccionLog', enginePC, if_exists='append', index=False)
        else:
            print("Logs no ingresados a SQL en entorno de prueba")

        print("LOGS PP A SQL")
        id_pp = pd.read_sql("SELECT MAX(id) AS MAX_ID FROM PlanProduccionLog", cnxn_mio)["MAX_ID"].values[0]
        df_final_pp["id_pp"] = id_pp
        df_final_pp["Timestamp"] = timestamp
        df_final_pp["FECHA HORA INICIO PROCESO MODELO"] = cook_start_hour_model
        df_final_pp["flujo_ksa_mm"] = df_final_pp["flujo_ksa"].rolling(window=4, min_periods=1).mean()
        df_final_pp["flujo_ingreso_clarificador_mm"] = df_final_pp["flujo_ingreso_clarificador"].rolling(window=4, min_periods=1).mean()
        to_rename = {
            "Intervalo Chata":"INTERVALO CHATA",
            "Intervalo Poza":"INTERVALO POZA",
            "FECHA HORA":"FECHA HORA",
            "Intervalo Chata":"FECHA BATCH",
            "NumHora":"HORAS PROCESO",
            "PLANTA":"PLANTA",
            "VolumenTotal":"DESCARGA",
            "AVANCE DESCARGA":"AVANCE DESCARGA",
            "Grasa_h75":"GRASA H75",
            "VelocidadCoccion":"VELOCIDAD",
            "Tipo velocidad":"Tipo velocidad",
            "ACUM_ALIMENTACION":"ACUM_ALIMENTACION",
            "STOCK POZAS":"STOCK POZAS",
            "TVNPonderado":"TVN DESCARGA",
            "solidos_ab":"SOLIDOS AB",
            "grasa_ab":"GRASA AB",
            "TVN_Pozas":"TVN POZAS",
            "TVN_IC":"TVN INGRESO COCINAS",
            "solidos_ab_ic":"SOLIDOS AB IC",
            'grasa_ab_ic':"GRASA AB IC",
            "RAPPonderado":"RAP PAMA",
            "VOL AB":"VOL AB",
            "VolumenPama":"VOLUMEN AB ACUMULADO",
            "RAP PAMA ACUMULADO":"RAP PAMA ACUMULADO",
            "Calidad":"CALIDAD HARINA",
            "dosificacion":"RECETA CORREGIDA",
            "base_seca":"BASE SECA CORREGIDA",
            "flujo_de_ksa":"FLUJO DE KSA CORREGIDO",
            "flujo_ingreso_clarificador":"FLUJO DE INGRESO AL CLARIFICADOR",
            "flujo_ingreso_clarificador_mm":"FLUJO DE INGRESO AL CLARIFICADOR MM",
            "flujo_ksa":"FLUJO KSA",
            "flujo_ksa_mm":"FLUJO KSA MM",
            'grasa_integral':"KK INTEGRAL GRASA",
            'ceniza_integral':"KK INTEGRAL CENIZA",
            'proteina_integral':"KK INTEGRAL PROTEINA",
            'proteina_harina':"HARINA PROTEINA", 
            'ceniza_harina':"HARINA CENIZA", 
            'grasa_harina':"HARINA GRASA",
            "FECHA HORA INICIO PROCESO":"FECHA HORA INICIO PROCESO",
            "FECHA HORA FIN PROCESO":"FECHA HORA FIN PROCESO",
            "FECHA HORA INICIO PROCESO MODELO":"FECHA HORA INICIO PROCESO MODELO",
            "STOCK MIN ARRANQUE":"STOCK MIN ARRANQUE",
            "DURACION BATCH":"DURACION BATCH",
            "id_pp":"ID_PP",
            "Timestamp":"TIMESTAMP"
        }
        df_final_pp = df_final_pp.rename(to_rename, axis=1)

        columns_to_repeat = [
            'HARINA PROTEINA',
        'HARINA CENIZA', 'HARINA GRASA'
        ]
        last_non_empty_row = df_final_pp[columns_to_repeat].dropna().iloc[-1]
        
        # df_repeated = pd.concat([df_final_pp, pd.DataFrame([last_non_empty_row] * 2, columns=columns_to_repeat)], ignore_index=True)

        df_final_pp = fill_next_two_nulls(df_final_pp, columns_to_repeat)

        # print("LOGS PP A SQL")
        # id_pp = pd.read_sql("SELECT MAX(id) AS MAX_ID FROM PlanProduccionLog", cnxn_mio)["MAX_ID"].values[0]
        # output_path = f'outputs/Plan_Produccion_{now.strftime("%Y_%m_%d-%I_%M_%S_%p")}_{planta_selected}_validation.xlsx'
        # df_final_pp.to_excel(output_path, index=False)
        if ambiente == 'prod':
            print("INGRESANDO PP A SQL")
            print(df_final_pp[to_rename.values()])
            
            # df_final_pp.to_csv("df_final_pp.csv")
            
            df_final_pp[to_rename.values()].to_sql('PlanProduccionOutput', enginePC, if_exists='append', index=False)
            print("PP INGRESADO A SQL")
        else:
            now = datetime.now()
            output_path = f'outputs/Plan_Produccion_{now.strftime("%Y_%m_%d-%I_%M_%S_%p")}_val.xlsx'
            df_final_pp[to_rename.values()].to_excel(output_path, index=False)
            print("PP exportado")
    except Exception as e:
        print(f"Error: {e}")
        logs = pd.DataFrame({
            'id_pd': [id],
            'StatusOptimizacion': ["Error"],
            'Mensaje': [str(e)],
            'Timestamp': [now.strftime("%Y-%m-%d %H:%M:%S")],
        })
        if ambiente == 'prod':
            logs.to_sql('PlanProduccionLog', enginePC, if_exists='append', index=False)
            print("Error registrado en SQL")

cnxn_mio.close()
cnxn_juv.close()
enginePC.dispose()