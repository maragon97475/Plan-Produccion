# Third party imports
import pandas as pd


def get_plantas_habilitadas():
    query = """select * from PlantasHabilitadasTemporada WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_active_mareas_with_location_and_static_data():
    query = """SELECT UPPER(REPLACE(boat_name, ' ', '')) as boat_name_trim,RN = DENSE_RANK() OVER(PARTITION BY boat_name ORDER BY departure_port_date DESC),* FROM SPActiveMareasJoinedRecoms WITH(NOLOCK)
            WHERE marea_motive = 2"""
    return pd.read_sql(query, connection)


def get_pozas_estado():
    query = """EXEC descargas_pozas_cocinas 32"""
    return pd.read_sql(query, connection)


def get_ubicacion_capacidad_pozas():
    query = """select ConexionLineasPozas.*, Chatas_Lineas.coordslatitude as coordslatitude, Chatas_Lineas.coordslongitude as coordslongitude, Pozas.pozaCapacity  from ConexionLineasPozas WITH(NOLOCK)
    left join Chatas_Lineas WITH(NOLOCK) on ConexionLineasPozas.id_chata = Chatas_Lineas.id_chata and ConexionLineasPozas.id_linea = Chatas_Lineas.id_linea
    left join Pozas WITH(NOLOCK) on ConexionLineasPozas.id_planta = Pozas.id_planta and ConexionLineasPozas.poza_number = Pozas.pozaNumber
    where Chatas_Lineas.habilitado = 1"""
    return pd.read_sql(query, connection)


def get_lineas_velocidad_descarga():
    query = """select * from Chatas_Lineas WITH(NOLOCK) where habilitado = 1"""
    return pd.read_sql(query, connection)


# Este stored procedure incluye info estatica de la planta joineado con el requerimiento, su hora de apertura, velocidad de planta actual, limites de tvn, etc
def get_requerimiento_planta():
    df_requerimiento_plantas = pd.read_sql("""EXEC plantas_joined_reqs_hora_apertura""", connection)
    df_requerimiento_plantas['requerimiento'] = df_requerimiento_plantas['daily_requirement']
    df_requerimiento_plantas.loc[df_requerimiento_plantas['daily_requirement'].isnull(),'requerimiento'] = df_requerimiento_plantas['requerimiento_por_defecto']
    return df_requerimiento_plantas


def get_lineas_reservadas_terceros():
    query = """select Lineas_Reserv_Terc.* from Chatas_Lineas WITH(NOLOCK)
                left join Lineas_Reserv_Terc WITH(NOLOCK) on Chatas_Lineas.id_chata = Lineas_Reserv_Terc.id_chata and Chatas_Lineas.id_linea = Lineas_Reserv_Terc.id_linea
                where Chatas_Lineas.habilitado = 1"""
    return pd.read_sql(query, connection)


def get_plantas_velocidades():
    query = """select * from PlantasVelocidad WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


# Includes pozas data: number, plant, hermanada, priority
def get_pozas_hermanadas():
    query = """select * from pozas WITH(NOLOCK)"""
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


def get_marea_web_service():
    query="""select * from MareasWebService WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_priorizacion_linea():
    query="""select * from PriorizacionLinea WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_hora_inicio():
    query="""select * from HoraInicioProduccion WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_mareas_cerradas():
    query="""select * from SPMareasJoinedRecomsHistorico WITH(NOLOCK)
            WHERE DATEDIFF(MM,last_modification,GETUTCDATE()) BETWEEN 0 AND 4"""
    return pd.read_sql(query, connection)


def get_mareas_acodere():
    query="""select * from MareasAcodere WITH(NOLOCK)
            WHERE 
            DATEDIFF(MM,timestamp,GETUTCDATE()) BETWEEN 0 AND 4"""
    return pd.read_sql(query, connection)


def get_chata_linea():
    query="""SELECT * FROM Chatas_Lineas WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_contingencia_max_tdc_limit_discharge_tradicionales():
    return 36


def get_contingencia_max_tdc_limit_cocina_tradicionales():
    return 36


def get_contingencia_max_tvn_limit_cocina_all():
    return 60


# Traer la data de las recomendaciones del ultimo dia
def get_recomendaciones_ultimodia():
    query="""SELECT * FROM RetornoRecomendacionHistorico WITH(NOLOCK)
            WHERE DATEDIFF(DD,last_modification,GETUTCDATE()) BETWEEN 0 AND 1"""
    return pd.read_sql(query, connection)


# Traer los dos ultimos registros de stock de pozas para la logica de acopio en pozas
def get_stock_pozas_recientes():
    query="""SELECT * 
            FROM(
            	SELECT id_planta,numero_poza,stock,update_date,report_date,report_hour,not_report_label,
            	ROW_NUMBER() OVER(PARTITION BY id_planta,numero_poza ORDER BY update_date DESC) Lecturas
            	FROM HistoricoStockPozas WITH(NOLOCK)
            	WHERE DATEDIFF(MM,update_date,GETUTCDATE()) BETWEEN 0 AND 3)TOTAL
            WHERE Lecturas<=3"""
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


def get_id_utilidad_chat_lin_poza():
    query = """SELECT MAX(id_utilidad) id_utilidad FROM ChataLineaPozaRecomTablaUtilidadCabecera WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


# Traer la ultima tabla de utilidad de cada planta del historico
def get_tabla_utilidad_plantas_chat_lin_poza():
    query = """SELECT REC.* 
            FROM ChataLineaPozaRecomTablaUtilidadHistorico REC WITH(NOLOCK)
            INNER JOIN(
                	SELECT Planta,MAX(id_utilidad) UltimoId
                	FROM ChataLineaPozaRecomTablaUtilidadHistorico WITH(NOLOCK)
                	GROUP BY Planta)HIST ON HIST.Planta=REC.Planta AND HIST.UltimoID=REC.id_utilidad"""
    return pd.read_sql(query, connection)


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


def get_season_tdc_tvn_data(season_start_date, season_end_date):
    query = """select marea_id, MareasWebService.boat_name, EPI.tipo_bodega, owner_group, frio_system_state, tdc_arrival, tvn_discharge, departure_port_date from MareasWebService
                left join Embarcaciones_Propias_Info EPI on MareasWebService.boat_name = EPI.boat_name
                where tvn_discharge is not null and tdc_arrival is not null
                  and departure_port_date > '{}' and departure_port_date < '{}'
                order by departure_port_date""".format(season_start_date, season_end_date)
    print(query)
    return pd.read_sql(query, connection)


def get_master_fajas():
    query = """SELECT * FROM MA_FAJAS WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_prioridad_pozas():
    query_prioridad_pozas = """SELECT CON.NOM_PLANTA,CON.NOM_CHATA,CON.NOM_LINEA,PRI.* 
                            FROM MA_PRIORIDAD_DESCARGA_POZAS PRI WITH(NOLOCK) 
                            INNER JOIN(
                            	SELECT L.ID_LINEA,C.*,L.NOM_LINEA,P.NOM_PLANTA
                            	FROM MA_CHATA C WITH(NOLOCK)
                            	INNER JOIN MA_LINEA L WITH(NOLOCK) ON L.ID_CHATA=C.ID_CHATA
                            	INNER JOIN MA_PLANTA P WITH(NOLOCK) ON P.ID_PLANTA=C.ID_PLANTA
                            )CON ON PRI.ID_CHATA=CON.ID_CHATA AND PRI.ID_LINEA=CON.ID_LINEA"""
    return pd.read_sql(query_prioridad_pozas, connection)


def get_id_planta():
    query = """SELECT ID_PLANTA, NOM_PLANTA FROM MA_PLANTA WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_velocidad_descarga_chata():
    query = """SELECT * FROM MA_VELOCIDAD_DESCARGA_VARIABLE WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_master_planta():
    query = """SELECT * FROM MA_PLANTA WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_master_chata():
    query = """SELECT * FROM MA_CHATA WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_master_linea():
    query = """SELECT * FROM MA_LINEA WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_master_embarcacion():
    query = """SELECT * FROM MA_EMBARCACION WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_master_poza():
    query = """SELECT * FROM MA_POZA WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_ejecucion_modelo():
    query = """SELECT TOP 1 * FROM PA_LOG_MODELO WITH(NOLOCK) ORDER BY ID_EJECUCION DESC"""
    return pd.read_sql(query, connection)
# FALTABA FROM 
def get_inputs_plan_descarga():
    query = """SELECT * FROM PlanDescargaModeloLog"""
    return pd.read_sql(query, connection)

def get_data_plan_descarga():
    query = """SELECT * FROM PlanDescargaInputsModel WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def get_data_plan_descarga_filtered(id):
    query = """SELECT * FROM PlanDescargaInputsModel WITH(NOLOCK) WHERE IDLOG = {}""".format(id)
    return pd.read_sql(query, connection)

def get_caracteristicas_lineas():
    query = """SELECT * FROM Chatas_Lineas WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def get_info_static_eps():
    query = """SELECT * FROM VelocidadEPS"""
    return pd.read_sql(query, connection)

def get_cluster():
    query = """SELECT * FROM DA_FP_CLUSTER_ZONAS WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def get_bloques():
    query = """SELECT * FROM PlantaCombinacionBloque WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def get_tipo_descarga():
    query = """SELECT * FROM Tipo_Descarga WITH(NOLOCK)"""
    return pd.read_sql(query, connection)


def get_disponibilidad_eps():
    query = """SELECT * FROM planDescargaInputsAjustadoPD WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def buffer_time():
    query = """SELECT * FROM limitMinutesAcode WITH(NOLOCK)"""
    return pd.read_sql(query, connection)

def destinos_dz():
    query = """SELECT * FROM ProyeccionZonaInputPD WITH(NOLOCK)"""
    return pd.read_sql(query, connection) 

# This function imports the connection object to this file globally
def import_connection():
    print('Importing connection to get_data_from_db')

    global connection
    from data.db_connection import connection
