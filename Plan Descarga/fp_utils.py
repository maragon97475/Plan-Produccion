import pandas as pd
from datetime import datetime, timedelta, time
import io
import base64
import re
import numpy as np
import dateutil.relativedelta
from itertools import permutations
from math import radians, cos, sin, asin, sqrt, isnan

# ------------------------------------------ Get Data -----------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------

def rank_within_group(group):
    group = group.sort_values(by='TM_MILLAS', ascending=False)
    group['CLUSTER'] = range(1, len(group) + 1)
    return group

def distance(lat_1, lng_1, lat_2, lng_2):
    """
    The function takes two points on the Earth's surface, and returns the distance between them in
    marine miles

    :param lat_1: Latitude of the first point
    :param lng_1: longitude of the first point
    :param lat_2: Latitude of the second point
    :param lng_2: longitude of the second point
    :return: The distance between two points.
    """
    lon2 = radians(lng_2)
    lat2 = radians(lat_2)
    lon1 = radians(lng_1)
    lat1 = radians(lat_1)
    R = 6371
    x = (lon2 - lon1) * cos(0.5 * (lat2 + lat1))
    y = lat2 - lat1
    d = R * sqrt(x * x + y * y)
    return d / 1.852


def distancer_cluster(
    row,
    zona_latitud_name="Dec_LatitudCentroide",
    zona_longitud_name="Dec_LongitudCentroide",
    origen_latitude_name="coordslatitude",
    origen_longitude_name="coordslongitude",
):
    return distance(
        row[origen_latitude_name],
        row[origen_longitude_name],
        row[zona_latitud_name],
        row[zona_longitud_name],
    )


def generate_destinos_table(
    tabla_simulation_descarga,
    df_posicion_zonas_pesca,
    df_velocidad_eps,
    intervalo_zarpe,
    nombre_ep="Embarcacion",
    marea_id="MareaId",
    nombre_ep_velocidad="EP",
    origen_latitude_name="coordslatitude",
    origen_longitude_name="coordslongitude",
    desacodere_name="FechaDesacodere",
    fc_distancer=distancer_cluster,
):
    tabla_simulation_descarga['tmp'] = 1
    df_posicion_zonas_pesca['tmp'] = 1
    df_destinos = pd.merge(
    tabla_simulation_descarga[
        [nombre_ep, 'TE',marea_id, origen_latitude_name, origen_longitude_name, desacodere_name, 'tmp']
    ],
    df_posicion_zonas_pesca,
    # how="cross",
    on='tmp'
    )
    df_destinos = df_destinos.drop('tmp', axis=1)

    df_destinos["DISTANCIA_MILLAS_MARINA"] = df_destinos.apply(fc_distancer, axis=1)

    df_destinos = pd.merge(
        df_destinos,
        df_velocidad_eps,
        left_on=nombre_ep,
        right_on=nombre_ep_velocidad,
        how="left",
    )
    mask = df_destinos[nombre_ep_velocidad].isna()
    df_destinos.loc[mask, ["Maximo"]] = 10
    df_destinos["TIPO"] = "PROPIA"
    df_destinos.loc[mask, ["TIPO"]] = "TERCERO"

    df_destinos["TIEMPO_EN_MAX"] = (
        df_destinos["DISTANCIA_MILLAS_MARINA"] / df_destinos["Maximo"]
    )
    df_destinos["INTERVALO ZARPE"] = intervalo_zarpe
    df_destinos["ZARPE PROYECTADO"] = df_destinos[desacodere_name] + pd.to_timedelta(df_destinos["INTERVALO ZARPE"], unit="h")

    df_destinos["FECHA_LLEGADA_MAX"] = (
        df_destinos["ZARPE PROYECTADO"]
        + pd.to_timedelta(df_destinos["TIEMPO_EN_MAX"], unit="h")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    df_destinos["TM_MILLAS"] = df_destinos["Dec_ToneladasDeclaradas"] / df_destinos["DISTANCIA_MILLAS_MARINA"]
    return df_destinos


def round_to_nearest_5_minutes(dt: datetime) -> time:
    # Extraer el minuto y segundo de la fecha y hora
    minute = dt.minute
    second = dt.second

    # Calcular cuántos minutos hay en total desde la hora
    total_minutes = minute + second / 60.0

    # Redondear al múltiplo de 5 más cercano
    rounded_minutes = round(total_minutes / 5) * 5

    # Crear la nueva fecha y hora con los minutos redondeados
    rounded_dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(
        minutes=rounded_minutes
    )

    return rounded_dt


def calculate_volumen_por_hora(
    df,
    hora_name="FECHADELLEGADAPD",
    volumen_name="Volumen (TM)",
    name_stock="Stock MP",
    name_stock_acumulado="Stock MP Acumulado",
    x_value_time="FechaHora",
):
    # Crear un rango de horas para el día
    horas_dia = pd.date_range(
        start=df[hora_name].min().floor("H"),
        end=df[hora_name].max().ceil("H"),
        freq="H",
    )

    # Inicializar un DataFrame para almacenar el volumen por hora
    volumen_por_hora = pd.DataFrame(index=horas_dia, columns=[name_stock], data=0)

    # Sumar el volumen descargado por cada hora
    for idx, row in df.iterrows():
        hora_llegada = row[hora_name]
        volumen = row[volumen_name]
        volumen_por_hora.loc[hora_llegada.floor("H"), name_stock] += volumen

    volumen_por_hora[name_stock_acumulado] = np.cumsum(volumen_por_hora[name_stock])
    volumen_por_hora = volumen_por_hora.reset_index()
    to_rename = {"index": x_value_time}
    volumen_por_hora = volumen_por_hora.rename(to_rename, axis=1)
    return volumen_por_hora


def dividir_dataframe_por_indice(df, valor_indice):
    """
    Divide un DataFrame en dos DataFrames y los coloca en una lista
    basándose en el valor de su índice.

    Args:
    - df: DataFrame a dividir.
    - valor_indice: Valor del índice que marca la división.

    Returns:
    - lista de dos DataFrames: El primer DataFrame contiene todas las filas
      hasta el valor del índice (inclusive), el segundo DataFrame contiene
      todas las filas después del valor del índice.
    """
    # Dividir el DataFrame en dos partes
    df1 = df[df.index < valor_indice]  # Incluye el valor del índice
    df2 = df[df.index >= valor_indice]  # Excluye el valor del índice

    # Colocar los DataFrames en una lista y devolverla
    return [df1, df2]


def convertir_latitud(latitud_str):
    partes = re.split(r"[°'']+", latitud_str.strip())
    grados = float(partes[0])
    minutos = float(partes[1])
    segundos = float(partes[2])

    direccion = partes[3].strip().upper()
    if direccion == "S":
        return -(grados + minutos / 60 + segundos / 3600)
    elif direccion == "N":
        return grados + minutos / 60 + segundos / 3600
    else:
        raise ValueError("Dirección de latitud no válida")


def convertir_longitud(longitud_str):
    partes = re.split(r"[°'']+", longitud_str.strip())
    grados = float(partes[0])
    minutos = float(partes[1])
    segundos = float(partes[2])
    direccion = partes[3].strip().upper()
    if direccion == "W":
        return -(grados + minutos / 60 + segundos / 3600)
    elif direccion == "E":
        return grados + minutos / 60 + segundos / 3600
    else:
        raise ValueError("Dirección de longitud no válida")


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


def generar_valor_correlativo(df, columna_origen, id_bloque_name="ID_Bloque"):
    """
    Genera una nueva columna en un DataFrame pandas con valores correlativos
    dependiendo de cada valor distinto al anterior encontrado en otra columna.

    Args:
    - df: DataFrame pandas.
    - columna_origen: Nombre de la columna en el DataFrame donde se buscarán los valores.

    Returns:
    - df: DataFrame pandas con la nueva columna agregada.
    """

    # Inicializar variables
    valor_anterior = None
    contador = 0
    lista_valores_correlativos = []

    # Recorrer los valores de la columna de origen
    for valor in df[columna_origen]:
        # Si es el primer valor o es diferente al anterior, incrementar el contador
        if valor_anterior is None or valor != valor_anterior:
            contador += 1
        # Agregar el contador a la lista de valores correlativos
        lista_valores_correlativos.append(contador)
        # Actualizar el valor anterior
        valor_anterior = valor

    # Agregar la lista de valores correlativos como una nueva columna al DataFrame
    df[id_bloque_name] = lista_valores_correlativos

    return df


def assign_alphabet_letters(df):
    """
    Asigna letras del abecedario a cada fila de un DataFrame de forma ordenada ascendente.

    Args:
        df (pandas.DataFrame): El DataFrame al que se le asignarán las letras.

    Returns:
        pandas.DataFrame: El DataFrame con una nueva columna que contiene las letras del abecedario.
    """
    # Calcula el número de filas en el DataFrame
    num_rows = len(df)

    # Genera una lista de letras del abecedario
    alphabet = []
    for i in range(num_rows):
        if i < 26:
            alphabet.append(chr(i + 65))  # Letras A-Z
        else:
            prefix = chr((i // 26) + 64)  # Prefijo: A, B, C, ...
            suffix = chr((i % 26) + 65)  # Sufijo: A, B, C, ...
            alphabet.append(prefix + suffix)

    # Asigna las letras al DataFrame
    df["Bloque"] = alphabet

    return df


def simulate_discharge_of_ships(
    df_eps_orig,
    num_lineas,
    df_velocidad_lineas,
    linea_fin_tiempo_orig,
    barcos_iniciales,
):
    # Ordenar el DataFrame de clientes por fecha de llegada
    df_eps = df_eps_orig.copy()
    linea_fin_tiempo = linea_fin_tiempo_orig.copy()

    df_velocidad_lineas_copy = df_velocidad_lineas.sort_values(
        "ORDEN-CHATA-LINEA"
    ).reset_index(drop=True)
    # velocidad_lineas = df_velocidad_lineas_copy["Vel. T1"].tolist()
    umbrales_tdc = df_velocidad_lineas_copy["UmbralTDC"].tolist()
    velocidades_tipo_1 = df_velocidad_lineas_copy["TIPO1"].tolist()
    velocidades_tipo_2 = df_velocidad_lineas_copy["TIPO2"].tolist()
    tiempo_empuje_tolvas = df_velocidad_lineas_copy["TiempoEmpuje"].tolist()
    nombre_lineas = df_velocidad_lineas_copy["ORDEN-CHATA-LINEA"].astype(str).tolist()
    tiempo_inicio_succion = df_velocidad_lineas_copy["InicioSuccion"].tolist()[0]
    marea_list = df_eps["MareaId"].tolist()
    bloque_list = df_eps["BLOQUE"].tolist()
    nombre_ep_list = df_eps["Embarcacion"].tolist()
    fecha_llegada_list = df_eps["FECHADELLEGADAPD"].tolist()
    volumen_list = df_eps["TON_ESTIMADAS"].tolist()
    volumen_declarado_list = df_eps["TON_DECLARADAS"].tolist()
    primera_cala_list = df_eps["FECHAPRIMERACALA"].tolist()
    estado_sistema_frio_list = df_eps["Estado de sistema frio"].tolist()

    # Crear listas para las columnas del DataFrame
    fecha_acodere_list = []
    fecha_inicio_succion_list = []
    fecha_final_succion_list = []
    fecha_desacodere_list = []
    linea_list = []
    tdc_final_list = []
    tvn_final_list = []
    velocidad_list = []
    tipo_velocidades_list = []
    tiempos_empuje_tolvas_list = []
    # fecha_llegada_barcos = df_eps.reset_index(drop=True).loc[0, "arriboReal"]

    # Simular la descarga de cada línea
    for nombre_barco, fecha_llegada, tiempo_atencion, primera_cala, sistema_frio in zip(
        nombre_ep_list,
        fecha_llegada_list,
        volumen_list,
        primera_cala_list,
        estado_sistema_frio_list,
    ):  
        fecha_llegada_original = fecha_llegada
        # Verificar si alguna línea está disponible
        linea_disponible = False
        for linea_index in range(num_lineas):
            if linea_fin_tiempo[linea_index] <= fecha_llegada:
                linea_disponible = True
                break

        # Si ningún línea está disponible, encontrar el línea que se desocupe primero
        if not linea_disponible:
            min_fin_tiempo = min(linea_fin_tiempo)
            linea_index = np.argmin(linea_fin_tiempo)

            # Esperar al línea desocupado más temprano
            fecha_llegada = max(fecha_llegada, min_fin_tiempo)

        if nombre_barco in barcos_iniciales:
            fecha_inicio_succion = max(fecha_llegada, linea_fin_tiempo[linea_index])
            if fecha_inicio_succion == fecha_llegada_original:
                fecha_inicio_succion = fecha_llegada_original + pd.Timedelta(minutes=13) + pd.Timedelta(minutes=tiempo_inicio_succion)
                fecha_acodere = fecha_llegada_original + pd.Timedelta(minutes=13)
            else:
                fecha_acodere = max(
                    fecha_llegada_original + pd.Timedelta(minutes=13),
                    fecha_inicio_succion - pd.Timedelta(minutes=tiempo_inicio_succion),
                )

        else:
            fecha_acodere = max(
                fecha_llegada + pd.Timedelta(minutes=13), linea_fin_tiempo[linea_index] + pd.Timedelta(minutes=12)
            )  
            fecha_inicio_succion = fecha_acodere + pd.Timedelta(
                minutes=tiempo_inicio_succion
            )
        # Calcular las fechas de descarga según la velocidad del línea
        # velocidad_actual = velocidad_lineas[linea_index]
        # fecha_acodere = max(fecha_llegada, linea_fin_tiempo[linea_index]) + pd.Timedelta(minutes=16)
        # fecha_inicio_succion = fecha_acodere + pd.Timedelta(minutes=tiempo_inicio_succion)
        # tdc_descarga = (primera_cala) / 1
        tdc_descarga = (fecha_inicio_succion - primera_cala).total_seconds() / 3600
        # nombre_actual_linea = nombre_lineas[linea_index]
        umbral_actual_tdc = umbrales_tdc[linea_index]
        if tdc_descarga <= umbral_actual_tdc:
            velocidad_actual = velocidades_tipo_1[linea_index]
            tipo_velocidad = "Tipo 1"
        else:
            velocidad_actual = velocidades_tipo_2[linea_index]
            tipo_velocidad = "Tipo 2"

        if sistema_frio == 1:
            tvn_descarga = estimate_tvn_con_frio(tdc_descarga)
            velocidad_actual = velocidades_tipo_1[linea_index]
            tipo_velocidad = "Tipo 1"
        else:
            tvn_descarga = estimate_tvn_sin_frio(tdc_descarga)

        # if df_eps["ID_PLANTA"].unique()[0] == 'CHIMBOTE': # TODO: Eliminar
        #     velocidad_actual = velocidades_tipo_2[linea_index]
        #     tipo_velocidad = "Tipo 2"            

        tiempo_descarga_linea = tiempo_atencion / velocidad_actual

        fecha_final_succion = (
            fecha_inicio_succion + pd.Timedelta(hours=tiempo_descarga_linea)
        ).strftime("%Y-%m-%d %H:%M:%S")
        # fecha_final_descarga_anterior = fecha_inicio_succion + pd.Timedelta(hours=tiempo_descarga_linea) + pd.Timedelta(minutes=tiempo_limpieza[linea_index]) #+ pd.Timedelta(minutes=10)
        fecha_desacodere = (
            fecha_inicio_succion
            + pd.Timedelta(hours=tiempo_descarga_linea)
            + pd.Timedelta(minutes=5) # TODO: Revisar 7 minutos
        )

        # Actualizar el linea_fin_tiempo para reflejar el tiempo de finalización del línea actual
        linea_fin_tiempo[linea_index] = fecha_desacodere
        tiempo_empuje_actual = tiempo_empuje_tolvas[linea_index]

        # Agregar los datos a las listas
        fecha_acodere_list.append(fecha_acodere)
        fecha_inicio_succion_list.append(fecha_inicio_succion)
        fecha_final_succion_list.append(fecha_final_succion)
        fecha_desacodere_list.append(fecha_desacodere)
        linea_list.append(nombre_lineas[linea_index])
        tdc_final_list.append(tdc_descarga)
        tvn_final_list.append(tvn_descarga)
        velocidad_list.append(velocidad_actual)
        tipo_velocidades_list.append(tipo_velocidad)
        tiempos_empuje_tolvas_list.append(tiempo_empuje_actual)

    # Crear un DataFrame a partir de las listas
    df = pd.DataFrame(
        {
            "Embarcacion": nombre_ep_list,
            "MareaId": marea_list,
            "Bloque": bloque_list,
            "Volumen Est. (TM)": volumen_list,
            "Volumen (TM)": volumen_declarado_list,
            "Velocidad": velocidad_list,
            "Tipo Velocidad": tipo_velocidades_list,
            "arriboReal": fecha_llegada_list,
            "FechaAcodere": fecha_acodere_list,
            "FechaInicioSuccion": fecha_inicio_succion_list,
            "FechaFinSuccion": fecha_final_succion_list,
            "FechaDesacodere": fecha_desacodere_list,
            "TDC_Descarga": tdc_final_list,
            "TVN_Descarga": tvn_final_list,
            "Línea": linea_list,
            "TiempoEmpuje": tiempos_empuje_tolvas_list,
        }
    )

    return df


def generate_plan_batch(
    df_eps_actual,
    planta_seleccionada,
    df_caracteristicas_lineas,
    date_apertura,
    barcos_iniciales,
    df_bloques,
    bloque_name="BLOQUE",
    id_bloque_name="ID_Bloque",
    id_planta_name="ID_PLANTA",
    desacodere_name="FechaDesacodere",
):
    df_eps_updated = generar_valor_correlativo(df_eps_actual, bloque_name)

    # mask = (
    #     df_bloques["ID_PLANTA"] == planta_seleccionada
    # )  # Revisar el nombre de la columna planta
    # df_eps_updated = pd.merge(df_eps_updated, df_bloques[mask], on=bloque_name, how="left")

    bloques = df_eps_updated[id_bloque_name].unique()
    tabla_simulacion = pd.DataFrame()

    mask = df_caracteristicas_lineas[id_planta_name] == planta_seleccionada
    df_status_lineas = df_caracteristicas_lineas.loc[
        mask, [id_planta_name, "CHATA-LINEA"]
    ].copy()
    df_status_lineas["FIN_DESCARGA"] = date_apertura
    lineas_totales = df_status_lineas["CHATA-LINEA"].tolist()
    lineas_para_actualizar = df_status_lineas["CHATA-LINEA"].tolist()
    valor_minimo_ultimo = df_status_lineas["FIN_DESCARGA"].min()
    for bloque in bloques:

        mask_bloque = df_eps_updated[id_bloque_name] == bloque
        df_eps_temp = df_eps_updated[mask_bloque]

        for total_linea in lineas_totales:
            if ((total_linea not in lineas_para_actualizar)):
                mask = df_status_lineas["CHATA-LINEA"] == total_linea
                llegada_nueva = (df_eps_temp["arriboReal"] - pd.Timedelta(minutes=12)).min()
                df_status_lineas.loc[mask, "FIN_DESCARGA"] = max(max(llegada_nueva, valor_minimo_ultimo), df_status_lineas.loc[mask, "FIN_DESCARGA"].reset_index(drop=True)[0])
                
        option_selected = df_eps_temp["COMBINACION"].unique()[0].split(",")
        mask = df_caracteristicas_lineas["CHATA-LINEA"].isin(option_selected)
        df_velocidad_lineas = (
            df_caracteristicas_lineas[mask].reset_index(drop=True).copy()
        )
        df_velocidad_lineas["ORDEN-CHATA-LINEA"] = pd.Categorical(
            df_velocidad_lineas["CHATA-LINEA"],
            option_selected,
        )
        num_lineas = len(option_selected)
        if bloque == 1:
            linea_fin_tiempo = [date_apertura] * num_lineas
        else:
            mask = df_status_lineas["CHATA-LINEA"].isin(option_selected)
            df_copy_status_lineas = df_status_lineas[mask].reset_index(drop=True)

            df_copy_status_lineas["ORDEN-CHATA-LINEA"] = pd.Categorical(
                df_copy_status_lineas["CHATA-LINEA"],
                option_selected,
            )
            df_copy_status_lineas = df_copy_status_lineas.sort_values(
                "ORDEN-CHATA-LINEA"
            ).reset_index(drop=True)
            linea_fin_tiempo = df_copy_status_lineas["FIN_DESCARGA"].tolist()
            linea_fin_tiempo = [linea.to_pydatetime() for linea in linea_fin_tiempo]

        tabla_sim_des_temp = simulate_discharge_of_ships(
            df_eps_temp,
            num_lineas,
            df_velocidad_lineas,
            linea_fin_tiempo,
            barcos_iniciales,
        )

        tabla_simulacion = pd.concat(
            [tabla_simulacion, tabla_sim_des_temp], ignore_index=True
        )

        df_linea_fin_tiempo = tabla_sim_des_temp.groupby(["Línea"], as_index=False)[
            desacodere_name
        ].max()
        lineas_para_actualizar = df_linea_fin_tiempo["Línea"].tolist()

        for linea_actual in lineas_para_actualizar:
            mask = df_linea_fin_tiempo["Línea"] == linea_actual
            fecha_fin = df_linea_fin_tiempo.loc[mask, desacodere_name].reset_index(drop=True)[0]#.item()

            mask = df_status_lineas["CHATA-LINEA"] == linea_actual
            df_status_lineas.loc[mask, "FIN_DESCARGA"] = fecha_fin
        valor_minimo_ultimo = df_status_lineas.loc[df_status_lineas["CHATA-LINEA"].isin(lineas_para_actualizar), 'FIN_DESCARGA'].min()
        # for total_linea in lineas_totales:
        #     if ((total_linea not in lineas_para_actualizar) and len(option_selected) > 1):
        #         mask = df_status_lineas["CHATA-LINEA"] == total_linea
        #         df_status_lineas.loc[mask, "FIN_DESCARGA"] = df_status_lineas["FIN_DESCARGA"].max()

    mask = df_velocidad_lineas[id_planta_name] == planta_seleccionada
    coordslatitude = df_velocidad_lineas.loc[mask, "coordslatitude"].unique()[0]
    coordslongitude = df_velocidad_lineas.loc[mask, "coordslongitude"].unique()[0]
    tabla_simulacion["coordslatitude"] = coordslatitude
    tabla_simulacion["coordslongitude"] = coordslongitude
    # tabla_simulacion["TE"] = (
    #     tabla_simulacion["Fecha_Inicio_Succión"] - tabla_simulacion["arriboReal"]
    # ).dt.total_seconds() / (60 * 60)

    # tabla_simulacion["TVN/TDC"] = np.round(
    #     tabla_simulacion["TVN_Descarga"] / tabla_simulacion["TDC_Descarga"], 1
    # )
    return tabla_simulacion


## 
# ------------------------------------------------- 2025-I (Clases Forecast Planning)---------------------------------------------------------------------
##
class Boat:
    def __init__(self, id, arrival_time, tonnage, degradation, boat_type, juveniles, camaroncillo, incremento_camaron = 1,
                 preferred_line=None, operator_order=None, new_start_times=None, manual_speed=None):
        """
        id: identificador del barco.
        arrival_time: hora de llegada (en horas, ej. 6.5, 23.75, etc.).
        tonnage: cantidad de anchoveta en toneladas.
        degradation: nivel de degradación del pescado.
        boat_type: tipo de barco ("pequeño", "mediano", "grande").
        preferred_line: línea de descarga preferente (si se asigna manualmente).
        operator_order: prioridad asignada por el operador (menor valor = mayor prioridad).
                        Se asume que siempre se define.
        new_start_times: lista (posiblemente vacía) de nuevos inicios.
                         Si existe al menos uno, el barco deberá iniciar la descarga a partir del primer nuevo inicio.
        """
        self.id = id
        self.arrival_time = arrival_time
        self.tonnage = tonnage
        self.degradation = degradation
        self.boat_type = boat_type
        self.preferred_line = preferred_line
        self.operator_order = operator_order
        self.new_start_times = sorted(new_start_times) if new_start_times else []
        self.start_time = None    # Hora efectiva en que inicia la descarga (después del traslado) (Acodere)
        self.start_succion_time = None # Inicio de succión
        self.finish_time = None # Fin de succion
        self.desacodere_time = None # Desacodere
        self.assigned_line = None # Línea a la que se asigna el barco
        self.juveniles = juveniles
        self.camaroncillo = camaroncillo
        self.type_discharge_ship = None
        self.speed_discharge_ship = None
        self.tdc_discharge = None
        self.speed_camaron = None
        self.incremento_camaron = incremento_camaron
        self.manual_speed = manual_speed

class DischargeLine:
    def __init__(self, id, base_speed, line_type, start_succion, fin_succion, time_entre_barcos):
        """
        id: identificador de la línea.
        base_speed: velocidad base de descarga (ton/hora).
        line_type: tipo de línea (por ejemplo "A", "B", "C").
        """
        self.id = id
        self.base_speed = base_speed
        self.line_type = line_type
        self.start_succion = start_succion
        self.fin_succion = fin_succion
        self.time_entre_barcos = time_entre_barcos
        self.busy = False          # Indica si la línea está en descarga.
        self.current_boat = None   # Barco que se descarga actualmente.
        self.free_time = 0         # Tiempo en el que la línea quedará libre.

# -------------------------
# Cálculo de velocidad efectiva usando condiciones (input en DataFrame)
# -------------------------

def effective_speed(line, boat, conditions_df, tonnage_threshold=100.0):
    """
    Calcula la velocidad efectiva de descarga evaluando las condiciones del DataFrame 'conditions_df'
    (ordenadas por la columna "Orden").

    Cada fila debe tener:
      - Orden: número de orden de evaluación.
      - Line_Type: valor esperado para el atributo line_type.
      - Boat_Type: valor esperado para el atributo boat_type.
      - Degradation_Condition: condición (ej. "< 2" o ">= 2") a evaluar sobre boat.degradation; 
                               puede ser None si no se evalúa.
      - Multiplier: multiplicador a aplicar al base_speed.
    
    Si ninguna condición se cumple, se usa un multiplicador de 1.0.
    Además, si el tonaje del barco supera 'tonnage_threshold', se ajusta la velocidad proporcionalmente.
    """
    conditions_sorted = conditions_df.sort_values(by="Orden")
    multiplier = None
    velocidad = None
    for _, row in conditions_sorted.iterrows():
        condition_str = f"{boat.degradation + boat.start_time} {row['tdc_a']}"
        condition_str_b = f"{boat.degradation + boat.start_time} {row['tdc_b']}"
        if line.line_type == row["Line_Type"] and boat.boat_type == row["Boat_Type"] and eval(f"{boat.tonnage} {row['Vol_Threshold']}") and eval(f"{boat.juveniles / 100} {row['Juvenil_Threshold']}") and eval(condition_str) and eval(condition_str_b):
            multiplier = row["Multiplier"]
            velocidad = float(row["Tipo{}".format(row["Multiplier"])])
            break
            # if pd.notnull(row["tdc_a"]):

            #     if eval(condition_str) and eval(condition_str_b):
            #         multiplier = row["Multiplier"]
            #         velocidad = row["TIPO {}".format(row["Multiplier"])]
            #         break
            # else:
            #     multiplier = row["Multiplier"]
            #     velocidad = row["TIPO {}".format(row["Multiplier"])]
            #     break
        # else:
        #     print("No se cumplieron las condiciones")
    
    if multiplier is None:
        print("No se cumplieron las condiciones")
        multiplier = 1.0
        velocidad = 100

    # base_effective_speed = line.base_speed * multiplier
    base_effective_speed = line.base_speed * velocidad / line.base_speed
    boat.type_discharge_ship = multiplier
    boat.speed_discharge_ship = velocidad
    boat.tdc_discharge = boat.degradation + boat.start_time
    if boat.camaroncillo > 0.1:
        boat.speed_camaron = boat.incremento_camaron * base_effective_speed
        base_effective_speed = boat.speed_camaron
    if boat.manual_speed > 0:
        base_effective_speed = boat.manual_speed
    # if boat.tonnage > tonnage_threshold:
    #     base_effective_speed *= (tonnage_threshold / boat.tonnage)
    return base_effective_speed

# -------------------------
# Función para calcular el tiempo final de descarga sin partirla
# -------------------------

def compute_finish_time(boat, speed, effective_start, line):
    """
    Calcula la hora final de descarga del barco.
    
    Si el barco tiene definido al menos un "nuevo inicio" (new_start_times) y dicho valor es mayor
    que el effective_start, se fuerza que la descarga inicie a partir de ese nuevo inicio (sin partir la descarga).
    Luego, se descarga de forma continua.
    """
    if boat.new_start_times[0] > effective_start:
        effective_start = boat.new_start_times[0]
        boat.start_succion_time = effective_start
        boat.start_time = max(boat.start_succion_time - line.start_succion / 60, boat.arrival_time)
    finish_time = effective_start + (boat.tonnage / speed)
    return finish_time

# -------------------------
# Funciones auxiliares
# -------------------------

def format_time(time_in_hours):
    """
    Convierte un tiempo (en horas) a formato "Day X HH:MM".
    Se asume que el tiempo 0 es el inicio del Día 1.
    """
    days = int(time_in_hours // 24) + 1
    hours = int(time_in_hours % 24)
    minutes = int(round((time_in_hours % 1) * 60))
    return f"Day {days} {hours:02d}:{minutes:02d}"

def get_priority(boat):
    """
    Retorna la prioridad del barco usando exclusivamente operator_order.
    Se asume que siempre está definido.
    """
    return boat.operator_order

# -------------------------
# Simulación de la descarga
# -------------------------

def simulate_unloading(boats, lines, conditions_df, simulation_end=48, start_unloading_time=8.0, time_to_line=0.2):
    """
    Simula la descarga de barcos hasta simulation_end horas.
    
    Parámetros:
      - boats: lista de objetos Boat.
      - lines: lista de objetos DischargeLine.
      - conditions_df: DataFrame con las condiciones para calcular la velocidad efectiva.
      - simulation_end: hora final de simulación (puede ser >24 para varios días).
      - start_unloading_time: inicio mínimo global de descarga.
      - time_to_line: tiempo (en horas) para trasladarse a la línea.
    
    Devuelve la lista de barcos en el orden en que inician la descarga.
    """
    current_time = 0.0
    waiting_boats = []  # Cola de barcos en espera
    boats = sorted(boats, key=lambda b: b.operator_order)
    boat_index = 0
    unloading_order = []
    # global_next_start: ningún barco podrá iniciar antes de esta hora, actualizada cuando un barco tiene nuevo inicio
    global_next_start = start_unloading_time

    while (current_time < simulation_end) or waiting_boats or any(line.busy for line in lines):
        next_arrival_time = boats[boat_index].arrival_time if boat_index < len(boats) else float('inf')
        busy_lines = [line for line in lines if line.busy]
        next_finish_time = min([line.free_time for line in busy_lines], default=float('inf'))
        next_event_time = min(next_arrival_time, next_finish_time)
        if next_event_time == float('inf'):
            break
        current_time = next_event_time

        # Procesar llegadas de barcos
        while boat_index < len(boats) and boats[boat_index].arrival_time <= current_time:
            waiting_boats.append(boats[boat_index])
            # print(f"{format_time(boats[boat_index].arrival_time)}: Barco {boats[boat_index].id} llega a la planta")
            boat_index += 1

        # Liberar líneas que hayan finalizado descarga
        for line in lines:
            if line.busy and line.free_time <= current_time:
                # print(f"{format_time(current_time)}: Línea {line.id} finaliza descarga del Barco {line.current_boat.id}")
                line.busy = False
                line.current_boat = None

        free_lines = [line for line in lines if not line.busy]
        # Ordenar los barcos en espera por operator_order (siempre definido)
        sorted_waiting = sorted(waiting_boats, key=get_priority)
        for boat in sorted_waiting.copy():
            if not free_lines:
                break  # No hay líneas libres
            # Si el barco tiene línea preferida, asignarla solo si está libre
            if boat.preferred_line is not None:
                line = next((l for l in free_lines if l.id == boat.preferred_line), None)
                if line is None:
                    continue  # Esperar a que su línea preferida esté libre
            else:
                # Para barcos sin preferencia, se asigna la primera línea libre (ordenada por id)
                line = sorted(free_lines, key=lambda l: l.id)[0]
            # La hora efectiva de inicio es el máximo entre current_time y global_next_start, más el tiempo de traslado
            if boats[boat_index - 1].operator_order > 1:
                effective_start = max(current_time, global_next_start) #+ time_to_line
                boat.start_time = effective_start
                boat.start_succion_time = boat.start_time + line.start_succion / 60
                boat.assigned_line = line.id
                speed = effective_speed(line, boat, conditions_df)
                finish_time = compute_finish_time(boat, speed, boat.start_succion_time, line)
                boat.finish_time = finish_time
                boat.desacodere_time = boat.finish_time + line.fin_succion / 60 

            else:
                effective_start = max(current_time, global_next_start)
                boat.start_succion_time = effective_start
                boat.start_time = max(boat.start_succion_time - line.start_succion / 60, boat.arrival_time)
                boat.assigned_line = line.id
                speed = effective_speed(line, boat, conditions_df)
                finish_time = compute_finish_time(boat, speed, boat.start_succion_time, line)
                boat.finish_time = finish_time
                boat.desacodere_time = boat.finish_time + line.fin_succion / 60 
            # Si el barco tiene nuevos inicios, actualizar global_next_start con el primero (ya que se fuerza su inicio)
            if boat.new_start_times:
                global_next_start = max(global_next_start, boat.new_start_times[0])
            # Actualizar la línea asignada
            line.busy = True
            line.current_boat = boat
            line.free_time = boat.desacodere_time + line.time_entre_barcos / 60  # Convertir a horas
            unloading_order.append(boat)
            waiting_boats.remove(boat)
            free_lines.remove(line)
            # pref_text = "línea preferida" if boat.preferred_line is not None else "auto"
            # print(f"{format_time(current_time)}: Barco {boat.id} ({pref_text}, línea {line.id}) asignado; inicia descarga a las {format_time(boat.start_time)} y finaliza a las {format_time(boat.finish_time)}")

    return unloading_order


def convert_df_to_instances_boats(df):
    df_copy = df.copy()
    df_copy["dia_hora_cero"] = pd.to_datetime(df_copy["FECHADELLEGADAPD"].dt.date).min()
    df_copy["arrival_time"] = (df_copy["FECHADELLEGADAPD"] - df_copy["dia_hora_cero"]).dt.total_seconds() / 3600
    df_copy["first_cala_time"] = (df_copy["dia_hora_cero"] - df_copy["FECHAPRIMERACALA"]).dt.total_seconds() / 3600 # Revisar signo y luego sumar o restar con el tiempo de espera
    df_copy["new_start_time"] = (pd.to_datetime(df_copy["FECHAPROGRAMADA"]) - df_copy["dia_hora_cero"]).dt.total_seconds() / 3600

    lista_barcos = []
    for index, row in df_copy.iterrows():
        persona = Boat(id=row['MareaId'], arrival_time=row['arrival_time'], tonnage=row['VolumenEstTM'], 
                       degradation=row['first_cala_time'], boat_type=row['TIPO'], 
                       operator_order=row['OrAsig'], new_start_times=[row['new_start_time']], juveniles=row['juvenil_ponderado'], camaroncillo=row['munida_ponderado'], preferred_line=row['NumLinea'], manual_speed=row['velocidadAjustadaManual'])
        lista_barcos.append(persona)
    return lista_barcos, df_copy["dia_hora_cero"].min()

def convert_df_to_instances_lines(df):
    df_copy = df.copy()

    lista_lineas = []
    for index, row in df_copy.iterrows():
        persona = DischargeLine(id=row['NumLinea'], base_speed=row['Tipo1'], line_type=row['sistema_absorbente'], start_succion=row["inicio_succion_time"], fin_succion=row["fin_succion_time"], time_entre_barcos=row["time_entre_barcos"])
        lista_lineas.append(persona)
    return lista_lineas