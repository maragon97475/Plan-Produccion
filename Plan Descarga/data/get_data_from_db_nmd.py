# Third party imports
import pandas as pd


def get_plantas_habilitadas():
    query = """select * from PlantasHabilitadasTemporada WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_active_mareas_with_location_and_static_data():
    query = """SELECT UPPER(REPLACE(boat_name, ' ', '')) as boat_name_trim,RN = DENSE_RANK() OVER(PARTITION BY boat_name ORDER BY departure_port_date DESC),* FROM SPActiveMareasJoinedRecoms WITH(NOLOCK)
        WHERE marea_motive = 2"""
    return pd.read_sql(query, connection)

##revisar DONE
def get_pozas_estado():
    query = """EXEC descargas_pozas_cocinas_NMD 32"""
    return pd.read_sql(query, connection)


def get_ubicacion_capacidad_pozas():
    query = """select
            --A.ID_CONEXION, 
            --A.CHATA_ID, 
            A.NOM_CHATA, A.NOM_CHATA_COMPLETO id_chata, 
            --A.LINEA_ID, 
            A.NOM_LINEA id_linea, 
            --A.PLANTA_ID, 
            A.NOM_PLANTA id_planta, 
            --A.POZA_ID	
            A.NUM_POZA poza_number,
            B.NUM_COORDENADA_LATITUD coordslatitude, B.NUM_COORDENADA_LONGITUD coordslongitude,
            C.CTD_CAPACIDAD pozaCapacity
        from MA_CONEXION_LINEA_POZA A
        LEFT JOIN MA_CHATA B ON A.ID_CHATA=B.ID_CHATA
        LEFT JOIN MA_LINEA D ON A.ID_LINEA=D.ID_LINEA
        LEFT JOIN MA_POZA C ON A.ID_POZA=C.ID_POZA
        WHERE	D.FLG_HABILITADO=1"""
    return pd.read_sql(query, connection)

    

#REVISAR DONE
def get_lineas_velocidad_descarga():
    query = """select
            MAC.NOM_COMPLETO id_chata, MAP.NOM_PLANTA id_planta, MAC.NOM_CHATA name, MAC.NUM_COORDENADA_LATITUD coordslatitude, MAC.NUM_COORDENADA_LONGITUD coordslongitude, 
            MAC.CTD_MT_DISTANCIA_PLAYA distanciaChataPlayaEnMetros, MAL.NUM_BOMBAS_COMBUSTIBLE numberFuelPumps, MAL.CTD_SEGUNDOS_CONFIGURACION fuelPumpSideChangeSetupTimeSeconds, 
            MAL.NOM_LINEA id_linea, MAL.CTD_GALONES_MINUTO lados_fuelSpeedGallonsPerMinute, 
            MAL.TIP_SISTEMA_ABOSERVENTE sistema_absorbente, MAL.CTD_VELOCIDAD_100_TONS velocidad_0_100_tons, 
            MAL.CTD_VELOCIDAD_300_TONS velocidad_100_300_tons, MAL.CTD_VELOCIDAD_TONS velocidad_300_mas_tons, MAL.FLG_HABILITADO habilitado, 
            MAL.FEH_ULTIMA_MODIFICACION last_modification_user, MAL.NOM_USUARIO_MODIFICACION last_modification_date, MAL.FLG_HABILITADO_RETORNO habilitado_retorno
        from MA_LINEA MAL
        LEFT JOIN MA_CHATA MAC ON MAL.ID_CHATA=MAC.ID_CHATA
        LEFT JOIN MA_PLANTA MAP ON MAC.ID_PLANTA=MAP.ID_PLANTA
        WHERE MAL.FLG_HABILITADO=1"""
    return pd.read_sql(query, connection)

#REVISAR DONE
# Este stored procedure incluye info estatica de la planta joineado con el requerimiento, su hora de apertura, velocidad de planta actual, limites de tvn, etc
def get_requerimiento_planta():
    df_requerimiento_plantas = pd.read_sql("""EXEC plantas_joined_reqs_hora_apertura""", connection)
    df_requerimiento_plantas['requerimiento'] = df_requerimiento_plantas['daily_requirement']
    df_requerimiento_plantas.loc[df_requerimiento_plantas['daily_requirement'].isnull(),'requerimiento'] = df_requerimiento_plantas['requerimiento_por_defecto']
    return df_requerimiento_plantas

#REVISAR DONE
def get_lineas_reservadas_terceros():
    query = """select
                MAP.NOM_PLANTA id_planta, MAC.NOM_COMPLETO id_chata, MAL.NOM_LINEA id_linea, MAL.FLG_HABILITADO_TERCERO reserv_terc,MAL.FEH_ULTIMA_MODIFICACION last_modification_user, MAL.NOM_USUARIO_MODIFICACION last_modification_date
            from MA_LINEA MAL
            LEFT JOIN MA_CHATA MAC ON MAL.ID_CHATA=MAC.ID_CHATA
            LEFT JOIN MA_PLANTA MAP ON MAC.ID_PLANTA=MAP.ID_PLANTA
            WHERE MAL.FLG_HABILITADO=1"""
    return pd.read_sql(query, connection)


def get_plantas_velocidades():
    query = """select * from PlantasVelocidad WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

#REVISAR DONE
# Includes pozas data: number, plant, hermanada, priority
def get_pozas_hermanadas():
    query = """
            SELECT 
                MAP.NOM_PLANTA id_planta,
                MPO.NUM_POZA pozaNumber, MPO.CTD_CAPACIDAD pozaCapacity, MPO.NUM_PRIORIDAD poza_use_priority, 
                MPO.FLG_HABILITADO deshabilitado, MPO.FLG_VARIADOR tiene_variador, MPO.FLG_HERMANADA es_hermanada, 
                MPO.NUM_POZA_HERMANADA poza_hermanda
            FROM MA_POZA MPO
            LEFT JOIN MA_PLANTA MAP ON MPO.ID_PLANTA=MAP.ID_PLANTA
        """
    return pd.read_sql(query, connection)


def get_minimo_perc_bodega_recom_retorno():
    query = """select * from MinimoPercBodegaRecomRetorno WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_calidades_precio_venta():
    query = """select * from CalidadesPrecioVenta WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_costo_combustible():
    query = """select * from Costo_Combustible WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def get_data_restricciones():
    query="""select * from RestriccionesDF WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def get_minimos_planta():
    query="""select * from RestriccionesPlanta WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def get_average_discharge_time_in_pozas():
    query = """select * from AverageTimeDischargeInPoza WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

#REVISAR
def get_marea_web_service():
    query="""select * from MareasWebService WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_priorizacion_linea():
    query="""select * from PriorizacionLinea WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def get_hora_inicio():
    query="""select * from HoraInicioProduccion WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

#REVISAR DONE
def get_mareas_cerradas():
    query="""select * from SPMareasJoinedRecomsHistorico WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

#REVISAR DONE
def get_mareas_acodere():
    query=""" 
        SELECT	
            OPM.ID_MAREA marea_id, 
            MAL.NOM_LINEA linea_descarga_acodere, 
            OPD.TIP_MA_MOTIVO_LINEA motive_linea_descarga, 
            MAC.NOM_CHATA acodera_chata, 
            OPD.FEH_MA_INICIO_SUCCION inicio_succion, 
            OPD.FEH_MA_FIN_SUCCION termino_succion,  
            OPD.FEH_MA_DESACODERE desacodera_chata, 
            OPD.CTD_MINUTO_ACODERE_SUCCION acodera_inicio_succion, 
            OPD.TIP_MA_MOTIVO_ACODERE_INICIO motive_acodere_inicio, 
            OPD.CTD_MINUTO_FSUCCION_DESACODERE fin_succion_desacodere, 
            OPD.TIP_MA_MOTIVO_ACODERE_FIN motive_acodere_fin, 
            OPD.CTD_MINUTO_ARRIBO_ACODERE acoderar, 
            OPD.TIP_MA_MOTIVO_ACODERE motive_acodere, 
            OPP.NUM_POZA_UNO poza_1, OPP.NUM_POZA_DOS poza_2, OPP.NUM_POZA_TRES poza_3, OPP.NUM_POZA_CUATRO poza_4,
            OPP.TIP_MOTIVO_POZA motive_poza, OPD.FEH_MA_ULTIMA_MODIFICACION timestamp, OPD.NOM_USUARIO_MODIFICACION by_user, 
            OPD.NUM_MA_ORDEN_DESCARGA orden_descarga, 
            OPD.FEH_MA_FECHA_DESCARGA discharge_start_date, 
            MAC.NOM_COMPLETO chata_descarga_acodere,
            OPD.TIP_MA_MOTIVO_ACODERE_ORDEN motive_acodere_order, 
            OPP.FEH_INICIO_POZA_DOS discharge_start_date_poza_2, OPP.FEH_INICIO_POZA_TRES discharge_start_date_poza_3, OPP.FEH_INICIO_POZA_CUATRO discharge_start_date_poza_4
        FROM OP_MAREA OPM
        LEFT JOIN OP_DESCARGA OPD ON OPM.ID_MAREA=OPD.ID_MAREA
        LEFT JOIN MA_LINEA MAL ON OPD.ID_LINEA=MAL.ID_LINEA
        LEFT JOIN MA_CHATA MAC ON OPD.ID_CHATA=MAC.ID_CHATA
        LEFT JOIN MA_PLANTA MAP ON OPM.ID_PLANTA=MAP.ID_PLANTA
        LEFT JOIN (
        SELECT 
            ID_DESCARGA, 
            MAX(CASE WHEN NUM_ORDEN_POZA=1 THEN OPA.ID_POZA END) ID_POZA_UNO,
            MAX(CASE WHEN NUM_ORDEN_POZA=1 THEN FEH_INICIO_POZA END) FEH_INICIO_POZA_UNO,
            MAX(CASE WHEN NUM_ORDEN_POZA=1 THEN FEH_INICIO_POZA END) NUM_POZA_UNO,
            MAX(CASE WHEN NUM_ORDEN_POZA=2 THEN OPA.ID_POZA END) ID_POZA_DOS,
            MAX(CASE WHEN NUM_ORDEN_POZA=2 THEN FEH_INICIO_POZA END) FEH_INICIO_POZA_DOS,
            MAX(CASE WHEN NUM_ORDEN_POZA=2 THEN FEH_INICIO_POZA END) NUM_POZA_DOS,
            MAX(CASE WHEN NUM_ORDEN_POZA=3 THEN OPA.ID_POZA END) ID_POZA_TRES,
            MAX(CASE WHEN NUM_ORDEN_POZA=3 THEN FEH_INICIO_POZA END) FEH_INICIO_POZA_TRES,
            MAX(CASE WHEN NUM_ORDEN_POZA=3 THEN FEH_INICIO_POZA END) NUM_POZA_TRES,
            MAX(CASE WHEN NUM_ORDEN_POZA=4 THEN OPA.ID_POZA END) ID_POZA_CUATRO,
            MAX(CASE WHEN NUM_ORDEN_POZA=4 THEN FEH_INICIO_POZA END) FEH_INICIO_POZA_CUATRO,
            MAX(CASE WHEN NUM_ORDEN_POZA=4 THEN FEH_INICIO_POZA END) NUM_POZA_CUATRO,
            MAX(TIP_MOTIVO_POZA) TIP_MOTIVO_POZA
        FROM OP_ALMACENAMIENTO OPA
        LEFT JOIN MA_POZA MPZ ON OPA.ID_POZA=MPZ.ID_POZA
        GROUP BY ID_DESCARGA 
        ) AS OPP ON OPD.ID_DESCARGA=OPP.ID_DESCARGA WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

#REVISAR
def get_chata_linea():
    query="""select
            MAC.NOM_COMPLETO id_chata, MAP.NOM_PLANTA id_planta, MAC.NOM_CHATA name, MAC.NUM_COORDENADA_LATITUD coordslatitude, MAC.NUM_COORDENADA_LONGITUD coordslongitude, 
            MAC.CTD_MT_DISTANCIA_PLAYA distanciaChataPlayaEnMetros, MAL.NUM_BOMBAS_COMBUSTIBLE numberFuelPumps, MAL.CTD_SEGUNDOS_CONFIGURACION fuelPumpSideChangeSetupTimeSeconds, 
            MAL.NOM_LINEA id_linea, MAL.CTD_GALONES_MINUTO lados_fuelSpeedGallonsPerMinute, 
            MAL.TIP_SISTEMA_ABOSERVENTE sistema_absorbente, MAL.CTD_VELOCIDAD_100_TONS velocidad_0_100_tons, 
            MAL.CTD_VELOCIDAD_300_TONS velocidad_100_300_tons, MAL.CTD_VELOCIDAD_TONS velocidad_300_mas_tons, MAL.FLG_HABILITADO habilitado, 
            MAL.FEH_ULTIMA_MODIFICACION last_modification_user, MAL.NOM_USUARIO_MODIFICACION last_modification_date, MAL.FLG_HABILITADO_RETORNO habilitado_retorno
        from MA_LINEA MAL
        LEFT JOIN MA_CHATA MAC ON MAL.ID_CHATA=MAC.ID_CHATA
        LEFT JOIN MA_PLANTA MAP ON MAC.ID_PLANTA=MAP.ID_PLANTA
        WHERE MAL.FLG_HABILITADO=1 WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_contingencia_max_tdc_limit_discharge_tradicionales():
    return 36


def get_contingencia_max_tdc_limit_cocina_tradicionales():
    return 36


def get_contingencia_max_tvn_limit_cocina_all():
    return 60

# Traer la data de las recomendaciones del ultimo dia
#REVISAR DONE
def get_recomendaciones_ultimodia():
    query="""SELECT * FROM VW_RETORNO_RECOMENDACION_HIST WITH(NOLOCK)
            WHERE DATEDIFF(DD,last_modification,GETUTCDATE()) BETWEEN 0 AND 1"""
    return pd.read_sql(query, connection)

# Traer los dos ultimos registros de stock de pozas para la logica de acopio en pozas
def get_stock_pozas_recientes():
    query="""SELECT * 
            FROM(
                	SELECT id_planta,numero_poza,stock,update_date,report_date,report_hour,not_report_label,last_update,
                	MAX(update_date) OVER(PARTITION BY id_planta,numero_poza) previous_update
                	FROM(
                		SELECT id_planta,numero_poza,stock,update_date,report_date,report_hour,not_report_label,
                		MAX(update_date) OVER(PARTITION BY id_planta,numero_poza) last_update
                		FROM HistoricoStockPozas WITH(NOLOCK)
                		WHERE DATEDIFF(MM,update_date,GETUTCDATE()) BETWEEN 0 AND 6)ACT
                	WHERE update_date<last_update)FILTRO
            WHERE previous_update=update_date"""
    return pd.read_sql(query, connection)


# Traer los ultimos registros mayores a cero de la velocidad de planta
def get_plantas_velocidades_historico():
    query="""SELECT * 
            FROM(
                	SELECT *,MAX(last_modification) OVER(PARTITION BY id_planta) FechaMax
                	FROM PlantasVelocidadHistorico WITH(NOLOCK)
                	WHERE velocidad>0)RES
            WHERE last_modification=FechaMax"""
    return pd.read_sql(query, connection)

# Traer el ultimo id utilidad registrado
def get_id_utilidad():
    query = """SELECT MAX(id_utilidad) id_utilidad FROM CocinaRecomTablaUtilidadCabecera WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

# Traer la ultima tabla de utilidad de cada planta del historico
def get_tabla_utilidad_plantas():
    query = """SELECT REC.* 
            FROM CocinaRecomTablaUtilidadHistorico REC WITH(NOLOCK)
            INNER JOIN(
                	SELECT Planta,MAX(id_utilidad) UltimoId
                	FROM CocinaRecomTablaUtilidadHistorico WITH(NOLOCK)
                	GROUP BY Planta)HIST ON HIST.Planta=REC.Planta AND HIST.UltimoID=REC.id_utilidad"""
    return pd.read_sql(query, connection)

#REVISAR DONE-ELIMINADO
# def get_id_utilidad_chat_lin_poza():
#     query = """SELECT MAX(id_utilidad) id_utilidad FROM ChataLineaPozaRecomTablaUtilidadCabecera WITH(NOLOCK)"""
#     return pd.read_sql(query, connection)

# Traer la ultima tabla de utilidad de cada planta del historico
#REVISAR DONE-ELIMINADO
# def get_tabla_utilidad_plantas_chat_lin_poza():
#     query = """SELECT REC.* 
#             FROM ChataLineaPozaRecomTablaUtilidadHistorico REC WITH(NOLOCK)
#             INNER JOIN(
#                 	SELECT Planta,MAX(id_utilidad) UltimoId
#                 	FROM ChataLineaPozaRecomTablaUtilidadHistorico WITH(NOLOCK)
#                 	GROUP BY Planta)HIST ON HIST.Planta=REC.Planta AND HIST.UltimoID=REC.id_utilidad"""
#     return pd.read_sql(query, connection)

#REVISAR DONE-POSIBLE A ELIMINAR
# Traer el ultimo id de ejecucion de la nueva tabla de utilidad chata-linea-poza
def get_id_utilidad_chata_linea_poza():
    query = """SELECT MAX(id_ejecucion) id_ejecucion FROM ChataLineaPozaUtilidadesEjec WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

# Leer comentarios de flags
def get_category_flags():
    query = """SELECT * FROM CategoriaFlagUtilidad WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

# def get_contingencia_max_hours_in_poza_for_frio_with_discharge_over_24_hours():
#     return 2
#nuevas maestras MD
def get_mae_embarcacion():
    query='SELECT * FROM MA_EMBARCACION WITH(NOLOCK)'
    return pd.read_sql(query, connection)
	
def get_mae_planta():
    query='SELECT * FROM MA_PLANTA WITH(NOLOCK)'
    return pd.read_sql(query, connection)

def get_mae_planta_zarpe():
    query='SELECT * FROM MA_PLANTA_ZARPE WITH(NOLOCK)'
    return pd.read_sql(query, connection)

def get_mae_linea():
    query='SELECT * FROM MA_LINEA WITH(NOLOCK)'
    return pd.read_sql(query, connection)

def get_mae_chata():
    query='SELECT * FROM MA_CHATA WITH(NOLOCK)'
    return pd.read_sql(query, connection)

def get_mae_poza():
    query='SELECT * FROM MA_POZA WITH(NOLOCK)'
    return pd.read_sql(query, connection)

#REVISAR
def get_season_tdc_tvn_data(season_start_date, season_end_date):
    query = """select 
                OPM.ID_MAREA marea_id, 
                MAE.NOM_EMBARCACION boat_name, 
                MAE.TIP_BODEGA tipo_bodega, 
                MAE.TIP_EMBARCACION owner_group, 
                OPM.TIP_ESTADO_FRIO frio_system_state, 
                OPD.CTD_TDC_ARRIBO tdc_arrival, 
                OPD.CTD_TVN_DESCARGA tvn_discharge, 
                OPM.FEH_ZARPE departure_port_date 
            from OP_MAREA OPM --MareasWebService
            LEFT JOIN OP_DESCARGA OPD ON OPM.ID_MAREA=OPD.ID_MAREA
            LEFT JOIN MA_EMBARCACION MAE ON OPM.ID_EMBARCACION=MAE.ID_EMBARCACION
            WHERE OPD.CTD_TVN_DESCARGA is not null and OPD.CTD_TDC_ARRIBO is not null
                and OPM.FEH_ZARPE > '{}' and OPM.FEH_ZARPE < '{}'
            order by OPM.FEH_ZARPE""".format(season_start_date, season_end_date)
    print(query)
    return pd.read_sql(query, connection)


# This function imports the connection object to this file globally
def import_connection():
    print('Importing connection to get_data_from_db_nmd')

    global connection
    from data.db_connection import connection
