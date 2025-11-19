import sys
import os
from time import time, sleep
from datetime import datetime
from dotenv import load_dotenv
from fp_utils import *
from data import insert_data_to_db, get_data_from_db, db_connection, get_dumped_data, insert_data_to_db_nmd, get_data_from_db_nmd
import opt_utils
import traceback
import logging
import requests

import pandas as pd

from flask import Flask
from flask import request
from flask_cors import CORS
import routes as api_routes
from datetime import timedelta

def main_fp():
    
    # --------------------------------------------- OBTENCIÓN DE LA DATA --------------------------------------------------------------
    get_data = get_data_from_db
    get_data_from_db.import_connection()
    
    inputs_plan_descarga = get_data.get_inputs_plan_descarga().sort_values('fecCreacion', ascending=False).reset_index(drop=True)
    id_log = inputs_plan_descarga.loc[0, 'id']
    planta_seleccionada = inputs_plan_descarga.loc[0, 'ID_PLANTA']
    selected_date_anterior = inputs_plan_descarga.loc[0, 'fecha']
    df_inicio_plantas = get_data.get_requerimiento_planta()
    buffer_time_df = get_data.buffer_time()

    chatas_lineas_habilitadas_df = get_data.get_lineas_velocidad_descarga()
    mask = (chatas_lineas_habilitadas_df["id_planta"] == planta_seleccionada) & (chatas_lineas_habilitadas_df["UsoLinea"].notna())
    chatas_lineas_habilitadas_df = chatas_lineas_habilitadas_df[mask].reset_index(drop=True)
    chatas_lineas_habilitadas_df["CHATA-LINEA"] = chatas_lineas_habilitadas_df["name"] + "-" + chatas_lineas_habilitadas_df["id_linea"]
    lineas_habilitadadas_list = chatas_lineas_habilitadas_df["CHATA-LINEA"].tolist()
    chatas_lineas_habilitadas_df["NumLinea"] = chatas_lineas_habilitadas_df["UsoLinea"].copy()

    mask = df_inicio_plantas["id"] == planta_seleccionada
    selected_time = df_inicio_plantas.loc[mask, 'hora_inicio'].values[0]
    hora_24_format = selected_time.hour
    selected_date = datetime.combine(selected_date_anterior, selected_time)
    if hora_24_format < 5:
        selected_date = selected_date - timedelta(hours=5)
        selected_date = selected_date + timedelta(days=1)
        selected_date = selected_date + timedelta(hours=5)
    stock_minimo = inputs_plan_descarga.loc[0, 'stockMin']
    ratio_declarado_descargado = inputs_plan_descarga.loc[0, 'ratio']
    intervalo_zarpe = inputs_plan_descarga.loc[0, 'esperaZarpe']
    df_current_eps = get_data.get_data_plan_descarga()
    mask = df_current_eps["idLog"] == id_log
    df_current_eps = df_current_eps[mask].reset_index(drop=True)
    mask = df_current_eps["TON_DECLARADAS"] > 0
    df_current_eps = df_current_eps[mask].reset_index(drop=True)

    mask = df_current_eps["seleccionado"] > 0
    df_current_eps = df_current_eps[mask].reset_index(drop=True)

    df_current_eps['juvenil_ponderado'] = df_current_eps['juvenil_ponderado'].fillna(0)

    df_cluster = get_data.get_cluster()

    mask = df_cluster["Int_InicioHoraPico"].isna()
    df_cluster.loc[mask, 'Int_InicioHoraPico'] = df_cluster.loc[mask, 'Int_HoraIniCluster']

    mask = df_cluster["Int_FinHoraPico"].isna()
    df_cluster.loc[mask, 'Int_FinHoraPico'] = df_cluster.loc[mask, 'Int_HoraFinCluster']

    df_velocidad_eps = get_data.get_info_static_eps() # TODO: Actualizar tabla de velocidadeds para obtener el tipo de EP
    conditions_df = get_data.get_tipo_descarga()

    conditions_df["Planta"] = conditions_df["Planta"].str.upper()
    mask = conditions_df["Planta"] == planta_seleccionada
    conditions_df = conditions_df[mask].reset_index(drop=True).reset_index()
    to_rename = {
        "EstadoDeFrio":"Boat_Type",
        "SistemaAbsorbente":"Line_Type",
        "RangoToneladasDeclaradas":"Vol_Threshold",
        "RangoJuvenilesPonderado":"Juvenil_Threshold",
        "UmbralTDC_Min":"tdc_a",
        "UmbralTDC_Max":"tdc_b",
        "TipoDescarga":"Multiplier",
        "index":"Orden"

    }
    conditions_df = conditions_df.rename(to_rename, axis=1)
    conditions_df["tdc_a"] = conditions_df["tdc_a"].fillna(conditions_df["tdc_b"])
    conditions_df["tdc_b"] = conditions_df["tdc_b"].fillna(conditions_df["tdc_a"])

    df_current_eps['VolumenEstTM'] = df_current_eps['TON_DECLARADAS'] * ratio_declarado_descargado
    df_current_eps["TIPO"] = 'SF'
    mask = df_current_eps["ESTADOFRIO"].isin(["RC", "GF"])
    df_current_eps.loc[mask, 'TIPO'] = 'RC'
    if df_current_eps["ORDENPD"].sum() == 0:
        df_current_eps = df_current_eps.sort_values(by='FECHADELLEGADAPD').reset_index(drop=True)
        orden_arribo = orden_asignado = (df_current_eps.index + 1).tolist()
    else:
        df_current_eps = df_current_eps.sort_values(by='ORDENPD').reset_index(drop=True)
        orden_asignado = (df_current_eps.index + 1).tolist()
        orden_arribo = (df_current_eps.sort_values(by='FECHADELLEGADAPD').reset_index().sort_values("ORDENPD").index + 1).tolist()
    
    df_current_eps["OrAsig"] = orden_asignado
    df_current_eps = pd.merge(df_current_eps, chatas_lineas_habilitadas_df[["CHATA-LINEA", "NumLinea"]], how='left', left_on="LineaDescargaAuxiliar", right_on="CHATA-LINEA")
    df_current_eps['NumLinea'] = df_current_eps['NumLinea'].apply(lambda x: None if pd.isna(x) else x)
    df_current_eps['NumLinea'] = np.where(df_current_eps["NumLinea"] > 0, df_current_eps["NumLinea"], None)
    df_current_eps["velocidadAjustadaManual"] = df_current_eps["velocidadAjustadaManual"].fillna(np.nan)
    boats, dia_hora_cero = convert_df_to_instances_boats(df_current_eps)

    # mask = (chatas_lineas_habilitadas_df["id_planta"] == planta_seleccionada) #& (chatas_lineas_habilitadas_df["CHATA-LINEA"].isin(lineas_habilitadadas_list))    
    chatas_lineas_habilitadas_df["inicio_succion_time"] = buffer_time_df.loc[buffer_time_df["plant"] == planta_seleccionada, "Acodere_inicio_succion_limite_PD"].item()
    chatas_lineas_habilitadas_df["fin_succion_time"] = buffer_time_df.loc[buffer_time_df["plant"] == planta_seleccionada, "fin_succion_desacodere_limite_pd"].item()
    chatas_lineas_habilitadas_df["time_entre_barcos"] = buffer_time_df.loc[buffer_time_df["plant"] == planta_seleccionada, "time_acoderar_limite_PD"].item()


    lines = convert_df_to_instances_lines(chatas_lineas_habilitadas_df)

    simulation_end = 48 + 24         # 48 horas (2 días)
    start_unloading_time = (selected_date - dia_hora_cero).total_seconds() / 3600   # Inicio mínimo global de descarga
    time_to_line = 0.2           # Tiempo (en horas) para trasladarse a la línea

    unloading_order = simulate_unloading(boats, lines, conditions_df,
                                           simulation_end=simulation_end, 
                                           start_unloading_time=start_unloading_time, 
                                           time_to_line=time_to_line)

    unloading_order_sorted = sorted(unloading_order, key=lambda b: b.operator_order)
    data = []
    order = 1
    for boat in unloading_order_sorted:
        waiting_time = boat.start_time - boat.arrival_time if boat.start_time is not None else None
        data.append({
            "Orden de Descarga": order,
            "MareaId": boat.id,
            "Línea de Descarga": boat.assigned_line,
            "Volumen Est. (TM)": boat.tonnage,
            "FECHADELLEGADAPD": dia_hora_cero + pd.Timedelta(hours=boat.arrival_time),
            "DecVelocidadProyectada": boat.speed_discharge_ship if boat.speed_discharge_ship is not None else "No asignado",
            "DecVelocidadCamaroncillo": boat.speed_camaron if boat.speed_camaron is not None else None,
            "Tipo Velocidad": boat.type_discharge_ship if boat.type_discharge_ship is not None else "No asignado",
            "TDC_Descarga": boat.tdc_discharge if boat.tdc_discharge is not None else "No asignado",
            "FechaAcodere": dia_hora_cero + pd.Timedelta(hours=boat.start_time) if boat.start_time is not None else "No asignado",
            "FechaInicioSuccion": dia_hora_cero + pd.Timedelta(hours=boat.start_succion_time) if boat.start_succion_time is not None else "No asignado",
            "FechaFinSuccion": dia_hora_cero + pd.Timedelta(hours=boat.finish_time) if boat.finish_time is not None else "No asignado",
            "FechaDesacodere": dia_hora_cero + pd.Timedelta(hours=boat.desacodere_time) if boat.desacodere_time is not None else "No asignado",
            "Tiempo de Espera Total": f"{waiting_time:.2f}h" if waiting_time is not None else "N/A"
        })
        order += 1

    tabla_simulation_descarga = pd.DataFrame(data)
    tabla_simulation_descarga = pd.merge(tabla_simulation_descarga, chatas_lineas_habilitadas_df[["NumLinea", "CHATA-LINEA", "coordslatitude", "coordslongitude", "tiempoEmpuje"]], left_on="Línea de Descarga", right_on="NumLinea", how='left')
    tabla_simulation_descarga = pd.merge(tabla_simulation_descarga, df_current_eps[["MareaId", "Embarcacion"]], how='left', on="MareaId")
    tabla_simulation_descarga["Volumen (TM)"] = tabla_simulation_descarga["Volumen Est. (TM)"] / ratio_declarado_descargado
    tabla_simulation_descarga["LineaDescarga"] = tabla_simulation_descarga["NumLinea"].copy()
    eps_frio = [
        "TASA 71",
        "TASA 61",
        "TASA 52",
        "TASA 51",
        "TASA 57",
        "TASA 58",
        "TASA 54",
        "TASA 59",
        "TASA 53",
        "TASA 55",
        "TASA 56",
        "TASA 41",
        "TASA 42",
        "TASA 427",
        "TASA 425",
        "TASA 44",
        "TASA 43",
        "TASA 419",
        "TASA 45",
        "TASA 450"
    ] # TODO: Debería estar en la tabla de EPs
    mask = tabla_simulation_descarga["Embarcacion"].isin(eps_frio)
    
    tabla_simulation_descarga["TE"] = 'Tercero'
    tabla_simulation_descarga.loc[mask, 'TE'] = 'Propio C/Frio'

    mask = (tabla_simulation_descarga["Embarcacion"].str.startswith("TASA")) & (tabla_simulation_descarga["TE"] == "Tercero") 
    tabla_simulation_descarga.loc[mask, 'TE'] = 'Propio S/Frio'
    tabla_simulation_descarga["Or. Asig."] = orden_asignado
    tabla_simulation_descarga["Or. Arri"] = orden_arribo
    tabla_simulation_descarga["Línea"] = tabla_simulation_descarga["CHATA-LINEA"].copy()

    tabla_simulation_descarga["TVN_Descarga"] = np.where(tabla_simulation_descarga["TE"].isin(['Propio S/Frio', 'Tercero']), np.vectorize(estimate_tvn_sin_frio)(tabla_simulation_descarga['TDC_Descarga']), np.vectorize(estimate_tvn_con_frio)(tabla_simulation_descarga['TDC_Descarga']))
    # tabla_simulation_descarga.to_excel("outputs/tabla_simulation_descarga_v2.xlsx", index=False)
    # ---------------------------------------------- VOLUMEN ARRIBO + POZAS -----------------------------------------------------------------

    tabla_simulation_descarga["Inicio Descarga PD"] = pd.to_datetime(tabla_simulation_descarga["FechaInicioSuccion"]) + pd.to_timedelta(tabla_simulation_descarga['tiempoEmpuje'], unit='m')
    tabla_simulation_descarga["Fin Descarga PD"] = pd.to_datetime(tabla_simulation_descarga["FechaFinSuccion"]) + pd.to_timedelta(tabla_simulation_descarga['tiempoEmpuje'], unit='m')

    fecha_inicial = tabla_simulation_descarga["Inicio Descarga PD"].min()
    CADA_MINUTOS = 8
    fechas_horas = [fecha_inicial + timedelta(minutes=i * CADA_MINUTOS) for i in range((20 * 60) // CADA_MINUTOS)]  # 12 horas * 60 minutos

    df_horas = pd.DataFrame(fechas_horas, columns=['FechaHora'])

    lista_barcos = tabla_simulation_descarga["Embarcacion"].unique()

    for barco in lista_barcos:
        mask = tabla_simulation_descarga["Embarcacion"] == barco
        rango_inicio = tabla_simulation_descarga.loc[mask, "Inicio Descarga PD"].reset_index(drop=True)[0]#.item()
        rango_fin = tabla_simulation_descarga.loc[mask, "Fin Descarga PD"].reset_index(drop=True)[0]#.item()
        # velocidad = tabla_simulation_descarga.loc[mask, "Velocidad"].reset_index(drop=True)[0]#.item()
        volumen = tabla_simulation_descarga.loc[mask, "Volumen Est. (TM)"].reset_index(drop=True)[0]#.item()
        df_horas[barco] = 0.0
        mask = (df_horas['FechaHora'] > rango_inicio) & (df_horas['FechaHora'] <= rango_fin)
        df_horas.loc[mask, barco] = volumen / mask.sum()

    df_horas["Stock MP Actual"] = df_horas[lista_barcos].sum(axis=1)
    df_horas["Stock MP Acumulado"] = np.cumsum(df_horas["Stock MP Actual"])

    df_horas["Stock Mínimo"] = stock_minimo
    
    df_horas["FLAG_MINIMO"] = 0
    mask = df_horas["Stock MP Acumulado"] > stock_minimo
    df_horas.loc[mask, 'FLAG_MINIMO'] = 1

    mask = df_horas["FLAG_MINIMO"] == 1
    if len(df_horas.loc[mask, 'FechaHora'].index) > 0:
        hora_arranque = round_to_nearest_5_minutes(df_horas.loc[mask, 'FechaHora'].min())
    else:
        hora_arranque =  df_horas["FechaHora"].max()
    
    df_horas["Tipo"] = 'Vol. Pozas'
    df_densidad_arribos = calculate_volumen_por_hora(tabla_simulation_descarga)
    df_densidad_arribos["Tipo"] = 'Vol. Arribo'
    
    
    # ------------------------------------------------------ ZONAS --------------------------------------------------------------------------
    now = datetime.now()
    df_destinos = generate_destinos_table(tabla_simulation_descarga, df_cluster, df_velocidad_eps, intervalo_zarpe)
    grouped = df_destinos.groupby('MareaId')
    top_3_per_group = grouped.apply(rank_within_group).query('CLUSTER <= 3')
    top_3_per_group = top_3_per_group.reset_index(drop=True)
    top_3_per_group['CLUSTER_COLUMNA'] = 'CLUSTER_' + top_3_per_group['CLUSTER'].astype(str)
    top_3_per_group["Fecha"] = selected_date_anterior

    to_rename = {
        "Int_OrdenCluster":"Cluster",
        "Int_InicioHoraPico":"Inicio Hora Pico",
        "Int_FinHoraPico":"Fin Hora Pico",
    }
    top_3_per_group = top_3_per_group.rename(to_rename, axis=1)
    
    date_objetivo = (selected_date + pd.Timedelta(days=1)).date()
    top_3_per_group["DIA Objetivo"] = pd.to_datetime(date_objetivo)
    top_3_per_group["DIA INICIO PICO"] = top_3_per_group["DIA Objetivo"] + pd.to_timedelta(top_3_per_group["Inicio Hora Pico"], unit='h')
    top_3_per_group["DIA FIN PICO"] = top_3_per_group["DIA Objetivo"] + pd.to_timedelta(top_3_per_group["Fin Hora Pico"], unit='h')

    top_3_per_group["ESTATUS DE LLEGADA"] = None
    mask = ((pd.to_datetime(top_3_per_group["FECHA_LLEGADA_MAX"]) - pd.Timedelta(hours=5)) < top_3_per_group["DIA INICIO PICO"])
    top_3_per_group.loc[mask, "ESTATUS DE LLEGADA"] = 'Antes'

    mask = (pd.to_datetime(top_3_per_group["FECHA_LLEGADA_MAX"]) - pd.Timedelta(hours=5) > top_3_per_group["DIA INICIO PICO"]) & (pd.to_datetime(top_3_per_group["FECHA_LLEGADA_MAX"]) - pd.Timedelta(hours=5) < top_3_per_group["DIA FIN PICO"])
    top_3_per_group.loc[mask, "ESTATUS DE LLEGADA"] = 'Durante'

    mask = (pd.to_datetime(top_3_per_group["FECHA_LLEGADA_MAX"]) - pd.Timedelta(hours=5) > top_3_per_group["DIA FIN PICO"])
    top_3_per_group.loc[mask, "ESTATUS DE LLEGADA"] = 'Después'
    top_3_per_group["RANGO HORAS PICO"] = top_3_per_group["Inicio Hora Pico"].astype(str) + '-' + top_3_per_group["Fin Hora Pico"].astype(str)

    to_rename = {
        "Vch_ZonaCaleta":"PUERTO-CALETA",
        "Int_OrdenCluster":"CLUSTER"
    }
    top_3_per_group = top_3_per_group.rename(to_rename, axis=1)

    values_cols = [
       'CLUSTER', 
       'RANGO HORAS PICO', 'PUERTO-CALETA', "FECHA_LLEGADA_MAX", "ESTATUS DE LLEGADA", "TIEMPO_EN_MAX"
    ]   
    pivot_table = top_3_per_group.pivot_table(index="MareaId", columns='CLUSTER_COLUMNA', values=values_cols, aggfunc='first')
    pivot_table.columns = pivot_table.columns.map('_'.join)
    pivot_table = pivot_table.reset_index()

    index_cols = [
        "Fecha",
        "MareaId",
        "TE",
        "Embarcacion",
        "FechaDesacodere",
        "ZARPE PROYECTADO"
    ]
    df_dz = pd.merge(top_3_per_group[index_cols].drop_duplicates("MareaId"), pivot_table, how='left', left_on="MareaId", right_on="MareaId")
    df_dz["Ultima Actualización"] = now
    to_rename = {
        "Embarcacion":"EMBARCACION",
        "FechaDesacodere":"DESACODERA CHATA DZ"
    }
    top_3_per_group = top_3_per_group.rename(to_rename, axis=1)

    tabla_simulation_descarga["Stock Arranque"] = stock_minimo
    tabla_simulation_descarga["Hora Arranque"] = hora_arranque
    tabla_simulation_descarga["Ultima_Actualizacion"] = now

    final_cols = [
        "Or. Asig.",
        "Or. Arri",
        "Embarcacion",
        "MareaId",
        # "Bloque",
        'Volumen Est. (TM)',
        "Volumen (TM)",
        # "Velocidad",
        "Tipo Velocidad",
        "FECHADELLEGADAPD",
        "FechaAcodere",
        "FechaInicioSuccion",
        "FechaFinSuccion",
        "FechaDesacodere",
        "TDC_Descarga",
        "TVN_Descarga",
        "Línea",
        "Inicio Descarga PD",
        "Fin Descarga PD",
        "Stock Arranque",
        "Hora Arranque",
        "Ultima_Actualizacion",
        "TE",
        "DecVelocidadCamaroncillo",
        "LineaDescarga",
        "DecVelocidadProyectada"
    ]

    tabla_simulation_descarga = tabla_simulation_descarga[final_cols]
    # tabla_simulation_descarga["Chata"] = tabla_simulation_descarga["Línea"].apply(lambda x: x.split('-')[0].strip() if isinstance(x, str) and '-' in x else x)
    # tabla_simulation_descarga["Línea"] = tabla_simulation_descarga["Línea"].apply(lambda x: x.split('-')[-1].strip() if isinstance(x, str) and '-' in x else x)
    tabla_simulation_descarga[["Chata", "Línea"]] = tabla_simulation_descarga["Línea"].str.rsplit('-', n=1, expand=True)
    tabla_simulation_descarga["Fecha"] = selected_date_anterior
    tabla_simulation_descarga['IdLog'] = id_log

    to_rename = {
        'Or. Asig.':"OrAsig",
        'Or. Arri':"OrArri",
        'Embarcacion':"Embarcacion",
        'MareaId':"MareaId",
        "Fecha":"Fecha",
        'BLOQUE':"BLOQUE",
        'Volumen Est. (TM)':"VolumenEstTM",
        'Volumen (TM)':"VolumenTM",
        'DecVelocidadProyectada':"Velocidad",
        'Tipo Velocidad':"TipoVelocidad",
        'FECHADELLEGADAPD':"Fecha_Llegada",
        'FechaAcodere':"Fecha_Acodere",
        'FechaInicioSuccion':"Fecha_Inicio_Succion",
        'FechaFinSuccion':"Fecha_Fin_Succion",
        'FechaDesacodere':"Fecha_Desacodere",
        'TDC_Descarga':"TDC_Descarga",
        'TVN_Descarga':"TVN_Descarga",
        'Línea':"Linea",
        'Chata':"Chata",
        'TE':"TE",
        'Inicio Descarga PD':"Inicio_Descarga_PD",
        'Fin Descarga PD':"Fin_Descarga_PD",
        'Stock Arranque':"Stock_Arranque",
        'Hora Arranque':"Hora_Arranque",
        'Ultima_Actualizacion':"Ultima_Actualizacion",
        'IdLog':'IdLog'
    }
    tabla_simulation_descarga["TVN_Descarga"] = tabla_simulation_descarga["TVN_Descarga"].clip(upper=120.0)
    tabla_simulation_descarga = tabla_simulation_descarga.rename(to_rename, axis=1)
    # tabla_simulation_descarga["LineaDescarga"] = np.random.randint(1, 5, size=len(tabla_simulation_descarga.index))
    # tabla_simulation_descarga["DecVelocidadProyectada"] = np.random.randint(100, 120, size=len(tabla_simulation_descarga.index))
    # tabla_simulation_descarga["DecVelocidadCamaroncillo"] = np.random.randint(100, 120, size=len(tabla_simulation_descarga.index))
    print(tabla_simulation_descarga.head())

    to_rename_dz = {
        "IdLogEjecucionModelo":"IdLogEjecucionModelo",
        "Fecha":"Fecha",
        "Embarcacion":"Embarcacion",
        "TipoBodega":"TipoBodega",
        "FechaDesacodere":"DesacoderaChataPZ",
        "ZARPE PROYECTADO":"ZarpeProyectado",
        "CLUSTER_CLUSTER_1":"Cluster1",
        "PUERTO-CALETA_CLUSTER_1":"PuertoCaletaC1",
        "RANGO HORAS PICO_CLUSTER_1":"RangoHorasPicoC1",
        "TIEMPO_EN_MAX_CLUSTER_1":"HorasNavegacionC1",
        "FECHA_LLEGADA_MAX_CLUSTER_1":"FechaProyectadaLlegadaC1",
        "ESTATUS DE LLEGADA_CLUSTER_1":"EstatusLlegadaC1",

        "CLUSTER_CLUSTER_2":"Cluster2",
        "PUERTO-CALETA_CLUSTER_2":"PuertoCaletaC2",
        "RANGO HORAS PICO_CLUSTER_2":"RangoHorasPicoC2",
        "TIEMPO_EN_MAX_CLUSTER_2":"HorasNavegacionC2",
        "FECHA_LLEGADA_MAX_CLUSTER_2":"FechaProyectadaLlegadaC2",
        "ESTATUS DE LLEGADA_CLUSTER_2":"EstatusLlegadaC2",

        "CLUSTER_CLUSTER_3":"Cluster3",
        "PUERTO-CALETA_CLUSTER_3":"PuertoCaletaC3",
        "RANGO HORAS PICO_CLUSTER_3":"RangoHorasPicoC3",
        "TIEMPO_EN_MAX_CLUSTER_3":"HorasNavegacionC3",
        "FECHA_LLEGADA_MAX_CLUSTER_3":"FechaProyectadaLlegadaC3",
        "ESTATUS DE LLEGADA_CLUSTER_3":"EstatusLlegadaC3",
        "UltimaActualización":"UltimaActualizacion",
        "TE":"TipoBodega"

    }
    df_dz = df_dz.rename(to_rename_dz, axis=1)
    df_dz["IdLogEjecucionModelo"] = id_log
    df_dz["UltimaActualizacion"] = now
    # df_dz["TipoBodega"] = df_dz["TE"]
    print(df_dz.head())

    to_rename_hours = {
        "IdLogEjecucionModelo":"IdLogEjecucionModelo",
        "FechaHora":"FechaHora",
        "Stock MP Actual":"StockMPActual",
        "Stock MP Acumulado":"StockMPAcumulado",
        "Stock Mínimo":"StockMinimo",
        "FLAG_MINIMO":"FLAG_MINIMO",
        "Tipo":"Tipo",
    }
    df_horas = df_horas.rename(to_rename_hours, axis=1)
    df_horas["IdLogEjecucionModelo"] = id_log
    print(df_horas.head())


    to_rename_arribos = {
        "IdLogEjecucionModelo":"IdLogEjecucionModelo",
        "FechaHora":"FechaHora",
        "Stock MP":"StockMP",
        "Stock MP Acumulado":"StockMPAcumulado",
        "Tipo":"Tipo",
    }
    df_densidad_arribos = df_densidad_arribos.rename(to_rename_arribos, axis=1)
    df_densidad_arribos["IdLogEjecucionModelo"] = id_log
    
    print(df_densidad_arribos.head())
    # cols_dz = list(to_rename_dz.values())
    # for col in cols_dz:
    #     if col not in df_dz.columns.tolist():
    #         df_dz[col] = None
        
    if "RangoHorasPicoC2" not in df_dz.columns.tolist():
         df_dz["RangoHorasPicoC2"] = df_dz["RangoHorasPicoC1"].copy()

    if "HorasNavegacionC2" not in df_dz.columns.tolist():
         df_dz["HorasNavegacionC2"] = df_dz["HorasNavegacionC1"].copy()

    if "PuertoCaletaC2" not in df_dz.columns.tolist():
         df_dz["PuertoCaletaC2"] = df_dz["PuertoCaletaC1"].copy()

    if "FechaProyectadaLlegadaC2" not in df_dz.columns.tolist():
         df_dz["FechaProyectadaLlegadaC2"] = df_dz["FechaProyectadaLlegadaC1"].copy()

    if "EstatusLlegadaC2" not in df_dz.columns.tolist():
         df_dz["EstatusLlegadaC2"] = df_dz["EstatusLlegadaC1"].copy()

    if "Cluster2" not in df_dz.columns.tolist():
         df_dz["Cluster2"] = df_dz["Cluster1"].copy()

    ## ---

    if "RangoHorasPicoC3" not in df_dz.columns.tolist():
         df_dz["RangoHorasPicoC3"] = df_dz["RangoHorasPicoC1"].copy()

    if "HorasNavegacionC3" not in df_dz.columns.tolist():
         df_dz["HorasNavegacionC3"] = df_dz["HorasNavegacionC1"].copy()

    if "PuertoCaletaC3" not in df_dz.columns.tolist():
         df_dz["PuertoCaletaC3"] = df_dz["PuertoCaletaC1"].copy()

    if "FechaProyectadaLlegadaC3" not in df_dz.columns.tolist():
         df_dz["FechaProyectadaLlegadaC3"] = df_dz["FechaProyectadaLlegadaC1"].copy()

    if "EstatusLlegadaC3" not in df_dz.columns.tolist():
         df_dz["EstatusLlegadaC3"] = df_dz["EstatusLlegadaC1"].copy()

    if "Cluster3" not in df_dz.columns.tolist():
         df_dz["Cluster3"] = df_dz["Cluster1"].copy()         


    # tabla_simulation_descarga["LineaDescarga"]  = 1
    tabla_simulation_descarga["DecVelocidadProyectada"] = tabla_simulation_descarga["Velocidad"].copy()
    # tabla_simulation_descarga["DecVelocidadCamaroncillo"] = 120
    tabla_simulation_descarga["Bloque"] = 'A'
    # tabla_simulation_descarga["Velocidad"] = 10
    df_dz["EstatusLlegadaC1"] = df_dz["EstatusLlegadaC1"].fillna('Antes')
    #
    return tabla_simulation_descarga, df_dz[list(to_rename_dz.values())], df_horas[list(to_rename_hours.values())], df_densidad_arribos[list(to_rename_arribos.values())]




def main_disponibilidad_eps():

    get_data = get_data_from_db
    get_data_from_db.import_connection()
    df_disponibilidad_eps = get_data.get_disponibilidad_eps()
    df_destinos_dz = get_data.destinos_dz()
    id_pd = df_disponibilidad_eps["idLog"].max()    
    mask = (df_disponibilidad_eps["finSuccion"] > '2020-01-01') & (df_disponibilidad_eps["desacoderaChata"] > '2020-01-01') & (df_disponibilidad_eps["idLog"] == id_pd)
    fecha_nuevo_inicio = df_disponibilidad_eps.loc[mask, "desacoderaChata"].max()
    mareas_descargadas = df_disponibilidad_eps.loc[mask, "idmarea"].unique().tolist()
    fec_crea = df_disponibilidad_eps["feccrea"].max()
    fec_crea_dz = df_destinos_dz["feccrea"].max()
    inputs_plan_descarga = get_data.get_inputs_plan_descarga().sort_values('fecCreacion', ascending=False).reset_index(drop=True)
    inputs_plan_descarga = inputs_plan_descarga[inputs_plan_descarga["id"] == id_pd].reset_index(drop=True)
    planta_seleccionada = inputs_plan_descarga.loc[0, 'ID_PLANTA']
    df_current_eps = get_data.get_data_plan_descarga_filtered(id_pd)
    df_current_eps = df_current_eps[df_current_eps["idLog"] == id_pd]
    df_current_eps = df_current_eps[~df_current_eps["MareaId"].isin(mareas_descargadas)]

    df_disponibilidad_eps_actual = df_disponibilidad_eps.loc[df_disponibilidad_eps["idmarea"].isin(mareas_descargadas), ["idLog", "idmarea", "embaracion", "TMdescargado", "eta", "arriboPlanta", "tdcActual", "ordenDescargaAsig", "acodereChata", "inicioSuccion", "finSuccion", "desacoderaChata"]]
    
# ---------------------------------------- DISIPO---------------------

    df_inicio_plantas = get_data.get_requerimiento_planta()
    buffer_time_df = get_data.buffer_time()
    chatas_lineas_habilitadas_df = get_data.get_lineas_velocidad_descarga()
    mask = (chatas_lineas_habilitadas_df["id_planta"] == planta_seleccionada) & (chatas_lineas_habilitadas_df["UsoLinea"].notna())
    chatas_lineas_habilitadas_df = chatas_lineas_habilitadas_df[mask].reset_index(drop=True)
    chatas_lineas_habilitadas_df["CHATA-LINEA"] = chatas_lineas_habilitadas_df["name"] + "-" + chatas_lineas_habilitadas_df["id_linea"]
    lineas_habilitadadas_list = chatas_lineas_habilitadas_df["CHATA-LINEA"].tolist()
    chatas_lineas_habilitadas_df["NumLinea"] = chatas_lineas_habilitadas_df["UsoLinea"].copy()

    mask = df_inicio_plantas["id"] == planta_seleccionada
    selected_time = df_inicio_plantas.loc[mask, 'hora_inicio'].item()
    # hora_24_format = selected_time.hour
    selected_date = fecha_nuevo_inicio
    # stock_minimo = inputs_plan_descarga.loc[0, 'stockMin']
    ratio_declarado_descargado = inputs_plan_descarga.loc[0, 'ratio']
    # intervalo_zarpe = inputs_plan_descarga.loc[0, 'esperaZarpe']
    # df_current_eps = get_data.get_data_plan_descarga()
    mask = df_current_eps["TON_DECLARADAS"] > 0
    df_current_eps = df_current_eps[mask].reset_index(drop=True)

    mask = df_current_eps["seleccionado"] > 0
    df_current_eps = df_current_eps[mask].reset_index(drop=True)

    df_current_eps['juvenil_ponderado'] = df_current_eps['juvenil_ponderado'].fillna(0)

    df_cluster = get_data.get_cluster()

    mask = df_cluster["Int_InicioHoraPico"].isna()
    df_cluster.loc[mask, 'Int_InicioHoraPico'] = df_cluster.loc[mask, 'Int_HoraIniCluster']

    mask = df_cluster["Int_FinHoraPico"].isna()
    df_cluster.loc[mask, 'Int_FinHoraPico'] = df_cluster.loc[mask, 'Int_HoraFinCluster']

    # df_velocidad_eps = get_data.get_info_static_eps() # TODO: Actualizar tabla de velocidadeds para obtener el tipo de EP
    conditions_df = get_data.get_tipo_descarga()

    conditions_df["Planta"] = conditions_df["Planta"].str.upper()
    mask = conditions_df["Planta"] == planta_seleccionada
    conditions_df = conditions_df[mask].reset_index(drop=True).reset_index()
    to_rename = {
        "EstadoDeFrio":"Boat_Type",
        "SistemaAbsorbente":"Line_Type",
        "RangoToneladasDeclaradas":"Vol_Threshold",
        "RangoJuvenilesPonderado":"Juvenil_Threshold",
        "UmbralTDC_Min":"tdc_a",
        "UmbralTDC_Max":"tdc_b",
        "TipoDescarga":"Multiplier",
        "index":"Orden"

    }
    conditions_df = conditions_df.rename(to_rename, axis=1)
    conditions_df["tdc_a"] = conditions_df["tdc_a"].fillna(conditions_df["tdc_b"])
    conditions_df["tdc_b"] = conditions_df["tdc_b"].fillna(conditions_df["tdc_a"])

    df_current_eps['VolumenEstTM'] = df_current_eps['TON_DECLARADAS'] * ratio_declarado_descargado
    df_current_eps["TIPO"] = 'SF'
    mask = df_current_eps["ESTADOFRIO"].isin(["RC", "GF"])
    df_current_eps.loc[mask, 'TIPO'] = 'RC'
    if df_current_eps["ORDENPD"].sum() == 0:
        df_current_eps = df_current_eps.sort_values(by='FECHADELLEGADAPD').reset_index(drop=True)
        orden_arribo = orden_asignado = (df_current_eps.index + 1).tolist()
    else:
        df_current_eps = df_current_eps.sort_values(by='ORDENPD').reset_index(drop=True)
        orden_asignado = (df_current_eps.index + 1).tolist()
        orden_arribo = (df_current_eps.sort_values(by='FECHADELLEGADAPD').reset_index().sort_values("ORDENPD").index + 1).tolist()
    
    df_current_eps["OrAsig"] = orden_asignado
    df_current_eps = pd.merge(df_current_eps, chatas_lineas_habilitadas_df[["CHATA-LINEA", "NumLinea"]], how='left', left_on="LineaDescarga", right_on="CHATA-LINEA")
    df_current_eps['NumLinea'] = df_current_eps['NumLinea'].apply(lambda x: None if pd.isna(x) else x)
    df_current_eps['NumLinea'] = np.where(df_current_eps["NumLinea"] > 0, df_current_eps["NumLinea"], None)
    df_current_eps['munida_ponderado'] = df_current_eps['munida_ponderado'] * 0
    df_current_eps["velocidadAjustadaManual"] = df_current_eps["velocidadAjustadaManual"].fillna(np.nan)
    boats, dia_hora_cero = convert_df_to_instances_boats(df_current_eps)

    chatas_lineas_habilitadas_df["inicio_succion_time"] = buffer_time_df.loc[buffer_time_df["plant"] == planta_seleccionada, "Acodere_inicio_succion_limite_PD"].item()
    chatas_lineas_habilitadas_df["fin_succion_time"] = buffer_time_df.loc[buffer_time_df["plant"] == planta_seleccionada, "fin_succion_desacodere_limite_pd"].item()
    chatas_lineas_habilitadas_df["time_entre_barcos"] = buffer_time_df.loc[buffer_time_df["plant"] == planta_seleccionada, "time_acoderar_limite_PD"].item()

    # mask = (chatas_lineas_habilitadas_df["id_planta"] == planta_seleccionada) & (chatas_lineas_habilitadas_df["CHATA-LINEA"].isin(lineas_habilitadadas_list))    
    lines = convert_df_to_instances_lines(chatas_lineas_habilitadas_df)

    simulation_end = 48          # 48 horas (2 días)
    start_unloading_time = (selected_date - dia_hora_cero).total_seconds() / 3600   # Inicio mínimo global de descarga
    time_to_line = 0.2           # Tiempo (en horas) para trasladarse a la línea

    unloading_order = simulate_unloading(boats, lines, conditions_df,
                                           simulation_end=simulation_end, 
                                           start_unloading_time=start_unloading_time, 
                                           time_to_line=time_to_line)

    unloading_order_sorted = sorted(unloading_order, key=lambda b: b.operator_order)
    data = []
    order = 1
    for boat in unloading_order_sorted:
        waiting_time = boat.start_time - boat.arrival_time if boat.start_time is not None else None
        data.append({
            "Orden de Descarga": order,
            "MareaId": boat.id,
            "Línea de Descarga": boat.assigned_line,
            "Volumen Est. (TM)": boat.tonnage,
            "FECHADELLEGADAPD": dia_hora_cero + pd.Timedelta(hours=boat.arrival_time),
            "DecVelocidadProyectada": boat.speed_discharge_ship if boat.speed_discharge_ship is not None else "No asignado",
            "DecVelocidadCamaroncillo": boat.speed_camaron if boat.speed_camaron is not None else None,
            "Tipo Velocidad": boat.type_discharge_ship if boat.type_discharge_ship is not None else "No asignado",
            "TDC_Descarga": boat.tdc_discharge if boat.tdc_discharge is not None else "No asignado",
            "FechaAcodere": dia_hora_cero + pd.Timedelta(hours=boat.start_time) if boat.start_time is not None else "No asignado",
            "FechaInicioSuccion": dia_hora_cero + pd.Timedelta(hours=boat.start_succion_time) if boat.start_succion_time is not None else "No asignado",
            "FechaFinSuccion": dia_hora_cero + pd.Timedelta(hours=boat.finish_time) if boat.finish_time is not None else "No asignado",
            "FechaDesacodere": dia_hora_cero + pd.Timedelta(hours=boat.desacodere_time) if boat.desacodere_time is not None else "No asignado",
            "Tiempo de Espera Total": f"{waiting_time:.2f}h" if waiting_time is not None else "N/A"
        })
        order += 1

    tabla_simulation_descarga = pd.DataFrame(data)
    if len(tabla_simulation_descarga.index) == 0:
         tabla_simulation_descarga = pd.DataFrame(columns=[
            "Orden de Descarga", "MareaId", "Línea de Descarga", "Volumen Est. (TM)", 
            "FECHADELLEGADAPD", "DecVelocidadProyectada", "DecVelocidadCamaroncillo", 
            "Tipo Velocidad", "TDC_Descarga", "FechaAcodere", "FechaInicioSuccion", 
            "FechaFinSuccion", "FechaDesacodere", "Tiempo de Espera Total"])
    tabla_simulation_descarga = pd.merge(tabla_simulation_descarga, chatas_lineas_habilitadas_df[["NumLinea", "CHATA-LINEA", "coordslatitude", "coordslongitude", "tiempoEmpuje"]], left_on="Línea de Descarga", right_on="NumLinea", how='left')
    tabla_simulation_descarga = pd.merge(tabla_simulation_descarga, df_current_eps[["MareaId", "Embarcacion"]], how='left', on="MareaId")
    tabla_simulation_descarga["Volumen (TM)"] = tabla_simulation_descarga["Volumen Est. (TM)"] / ratio_declarado_descargado
    tabla_simulation_descarga["LineaDescarga"] = tabla_simulation_descarga["NumLinea"].copy()
    eps_frio = [
        "TASA 71",
        "TASA 61",
        "TASA 52",
        "TASA 51",
        "TASA 57",
        "TASA 58",
        "TASA 54",
        "TASA 59",
        "TASA 53",
        "TASA 55",
        "TASA 56",
        "TASA 41",
        "TASA 42",
        "TASA 427",
        "TASA 425",
        "TASA 44",
        "TASA 43",
        "TASA 419",
        "TASA 45",
        "TASA 450"
    ] # TODO: Debería estar en la tabla de EPs
    mask = tabla_simulation_descarga["Embarcacion"].isin(eps_frio)
    
    tabla_simulation_descarga["TE"] = 'Tercero'
    tabla_simulation_descarga.loc[mask, 'TE'] = 'Propio C/Frio'

    mask = (tabla_simulation_descarga["Embarcacion"].str.startswith("TASA")) & (tabla_simulation_descarga["TE"] == "Tercero") 
    tabla_simulation_descarga.loc[mask, 'TE'] = 'Propio S/Frio'
    tabla_simulation_descarga["ordenPD"] = orden_asignado
    tabla_simulation_descarga["ordenLLegada"] = orden_arribo
    tabla_simulation_descarga["Línea"] = tabla_simulation_descarga["CHATA-LINEA"].copy()
    if len(tabla_simulation_descarga.index) == 0:
         tabla_simulation_descarga["TVN_Descarga"] = 0
    else:
        tabla_simulation_descarga["TVN_Descarga"] = np.where(tabla_simulation_descarga["TE"].isin(['Propio S/Frio', 'Tercero']), np.vectorize(estimate_tvn_sin_frio)(tabla_simulation_descarga['TDC_Descarga']), np.vectorize(estimate_tvn_con_frio)(tabla_simulation_descarga['TDC_Descarga']))

    # ---------------------------------------------- VOLUMEN ARRIBO + POZAS -----------------------------------------------------------------

    tabla_simulation_descarga["Inicio Descarga PD"] = pd.to_datetime(tabla_simulation_descarga["FechaInicioSuccion"]) + pd.to_timedelta(tabla_simulation_descarga['tiempoEmpuje'], unit='m')
    tabla_simulation_descarga["Fin Descarga PD"] = pd.to_datetime(tabla_simulation_descarga["FechaFinSuccion"]) + pd.to_timedelta(tabla_simulation_descarga['tiempoEmpuje'], unit='m')

    tabla_simulation_descarga["idLog"] = id_pd
    
    to_rename = {
         "Línea de Descarga":"lineaDeDescargaAj",
         "Tipo Velocidad":"tipoVelocidadAj",
         "FECHADELLEGADAPD":"fechaLLegadaAj",
         "Volumen Est. (TM)":"descargadoProyectadoAJ",
        #  "DecVelocidadProyectada":"TDCDescargaAj",
        "TDC_Descarga":"TDCDescargaAj",
        "FechaAcodere":"acodereChataAj",
        "FechaInicioSuccion":"inicioSuccionAj",
        "FechaFinSuccion":"finSuccionAj",
        "FechaDesacodere":"desacoderaChataAj",
        "Tiempo de Espera Total":"tiempoEsperaAj",
        "TVN_Descarga":"TVNDescargaAj",
        "Embarcacion":"embarcacion",

    }
    tabla_simulation_descarga = tabla_simulation_descarga.rename(to_rename, axis=1)
    tabla_simulation_descarga["feccrea"] = fec_crea
    tabla_simulation_descarga["fl_real"] = 1
    
    to_rename_disp = {
         "ordenDescargaAsig":"ordenPD",
         "embaracion":"embarcacion",
         "TMdescargado":"descargadoProyectadoAJ",
         "arriboPlanta":"fechaLLegadaAj",
         "tdcActual":"TDCDescargaAj",
         "acodereChata":"acodereChataAj",
         "inicioSuccion":"inicioSuccionAj",
         "finSuccion":"finSuccionAj",
         "desacodereChata":"desacoderaChataAj",
        #  "lineaDeDescargaAj"
    }
    df_disponibilidad_eps_actual = df_disponibilidad_eps_actual.rename(to_rename_disp, axis=1)
    df_disponibilidad_eps_actual["fl_real"] = 0
    df_disponibilidad_eps_actual["feccrea"] = fec_crea
    barcos_descargados = len(df_disponibilidad_eps_actual.index)
    tabla_simulation_descarga["ordenPD"] = tabla_simulation_descarga["ordenPD"] + barcos_descargados
    # df_disponibilidad_eps_actual["ordenLLegada"] = df_disponibilidad_eps_actual["ordenLLegada"] + barcos_descargados
    # TVNDescargaAj
    # tipoVelocidadAj
    tabla_simulation_descarga = pd.concat([tabla_simulation_descarga, df_disponibilidad_eps_actual], ignore_index=True)
    return tabla_simulation_descarga[["idLog", "ordenLLegada", "ordenPD", "embarcacion", "descargadoProyectadoAJ", "fechaLLegadaAj", "TDCDescargaAj", "TVNDescargaAj", "tipoVelocidadAj", "acodereChataAj", "inicioSuccionAj", "finSuccionAj", "desacoderaChataAj", "tiempoEsperaAj", "lineaDeDescargaAj", "feccrea", "fl_real"]]
