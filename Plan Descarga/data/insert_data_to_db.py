# Third party imports
import pandas as pd
import sqlalchemy
import numpy as np

# Internal imports
from data import get_data_from_db


def insert_execution_model(df_execution_time):
    df_execution_time.to_sql('PA_LOG_MODELO', connection, if_exists='append', index=False)
    print('End saving current execution')


def update_execution_model(return_optimization_end, current_execution):
    connection.execute("""update PA_LOG_MODELO set FEH_FIN = ? where ID_EJECUCION = ?""", return_optimization_end, current_execution)


def insert_return_optimization_result(return_recoms_df, return_opt_date):
    return_recoms_df['retorno_recom_id'] = return_recoms_df['marea_id'].astype(str) + ' - ' + return_opt_date.strftime("%m/%d/%Y_%H:%M:%S")
    return_recoms_df['last_modification'] = return_opt_date

    print('return_recoms_df \n', return_recoms_df)
    return_recoms_df.to_sql('RetornoRecomendacionActual', connection, if_exists='replace', index=False,
                            dtype={
                                'retorno_recom_id': sqlalchemy.types.VARCHAR(length=100),
                                'marea_id': sqlalchemy.types.INTEGER(),
                                'velocidad_retorno': sqlalchemy.types.VARCHAR(length=3),
                                'planta_retorno': sqlalchemy.types.VARCHAR(length=20),
                                'chata_descarga': sqlalchemy.types.VARCHAR(length=50),
                                'linea_descarga': sqlalchemy.types.VARCHAR(length=1),
                                'orden_descarga': sqlalchemy.types.INTEGER(),
                                'orden_descarga_global':  sqlalchemy.types.INTEGER(),
                                'poza_descarga_1': sqlalchemy.types.INTEGER(),
                                'poza_descarga_2': sqlalchemy.types.INTEGER(),
                                'poza_descarga_3': sqlalchemy.types.INTEGER(),
                                'tons_poza_descarga_1': sqlalchemy.types.FLOAT(),
                                'tons_poza_descarga_2': sqlalchemy.types.FLOAT(),
                                'tons_poza_descarga_3': sqlalchemy.types.FLOAT(),
                                'last_modification': sqlalchemy.types.DateTime(),
                                'flag_planta_llena': sqlalchemy.types.INTEGER()
                            }, method='multi')
    return_recoms_df.to_sql('RetornoRecomendacionHistorico', connection, if_exists='append', index=False, method='multi')
    connection.execute("""update UltActualizacionFechas set fecha = ? where tipo = ?""", return_opt_date, 'optimization')
    
    # Guardar id ejecucion para tabla de utilidades
    # return_esp_desc = return_recoms_df[return_recoms_df['orden_descarga_global']>0].copy()
    # if len(return_esp_desc)>0:
    #     df_ejecucion_cabecera = pd.DataFrame(columns=['timestamp'])
    #     df_ejecucion_cabecera.loc[0,'timestamp'] = return_opt_date
    
    #     # Tambien insertar la ejecucion en la tabla de utilidades chata linea poza
    #     df_ejecucion_cabecera.to_sql('ChataLineaPozaUtilidadesEjec', connection, if_exists='append', index=False)
    
    print("Finished inserting tu current recoms table")
    return 'success'


def insert_return_optimization_errors(return_recoms_errors, return_opt_date):
    return_recoms_errors['recom_date'] = return_opt_date
    print(return_recoms_errors)
    mask = return_recoms_errors['error_category_id'] == 'eta null'
    return_recoms_errors.loc[mask, 'error_category_id'] = 10
    return_recoms_errors.to_sql('RetornoRecomendacionErrores', connection, if_exists='append', index=False, method='multi')
    print('Inserting return optimization errors')


def insert_return_flags_and_ordenes(return_flags_head, return_flags_ordenes, return_opt_date, return_utility_table):
    
    if len(return_flags_head)>0:
        
        # Insertar la cabecera de ejecuciones
        df_ejecucion_cabecera = pd.DataFrame(columns=['timestamp'])
        df_ejecucion_cabecera.loc[0,'timestamp'] = return_opt_date
    
        # Tambien insertar la ejecucion en la tabla de utilidades chata linea poza
        df_ejecucion_cabecera.to_sql('ChataLineaPozaUtilidadesEjec', connection, if_exists='append', index=False)
        
        # Insertar primero la cabecera de flags
        tsql_chunksize = 2097 // len(return_flags_head.columns)
        # cap at 1000 (limit for number of rows inserted by table-value constructor)
        tsql_chunksize = 1000 if tsql_chunksize > 1000 else tsql_chunksize
        
        # Antes reemplazar timestamp por la fecha de ejecucion del algoritmo de retorno (no siempre son las mismas)
        return_flags_head['timestamp'] = return_opt_date
        # Insertar el orden de la tabla de ejecuciones
        last_execution = get_data_from_db.get_id_utilidad_chata_linea_poza()
        return_flags_head['id_ejecucion'] = int(last_execution['id_ejecucion'])

        return_flags_head.to_sql('ChataLineaPozaUtilidadCabecera', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize)
        
        # Insertar los ordenes + flags
        tsql_chunksize_order = 2097 // len(return_flags_ordenes.columns)
        tsql_chunksize_order = 1000 if tsql_chunksize_order > 1000 else tsql_chunksize_order
        
        # Aca tmb reemplazar el timestamp por la fecha de ejecucion de modelo retorno
        return_flags_ordenes['timestamp'] = return_opt_date
        del return_flags_ordenes['flag_planta_llena']
        return_flags_ordenes.to_sql('ChataLineaPozaRecomUtilidad', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_order)
        
        # Insertar la tabla de utilidades
        # Completar id ejecucion
        last_execution = get_data_from_db.get_id_utilidad_chata_linea_poza()
        return_utility_table['id_ejecucion'] = int(last_execution['id_ejecucion'])
        # df_tabla_utilidad['id_ejecucion'] = np.nan
        cols_final = return_utility_table.columns.tolist()
        cols_final = cols_final[-1:] + cols_final[:-1]
        # df_tabla_utilidad = df_tabla_utilidad[cols_final]
        return_utility_table = return_utility_table[cols_final]
        
        # Completar timestamp
        return_utility_table['timestamp'] = return_opt_date
        
        tsql_chunksize_util = 2097 // len(return_utility_table.columns)
        tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
        return_utility_table['tvn_pozas_ponderado'] = return_utility_table['tvn_pozas_ponderado'].clip(lower=0, upper=1200)
        return_utility_table.to_sql('ChataLineaPozaUtilidadActual', connection, if_exists='replace', index=False, method='multi', chunksize=tsql_chunksize_util)


def insert_plant_return_flags(return_flags_head, return_flags_ordenes, return_opt_date, return_utility_table):
    
    if len(return_flags_head)>0:
        
        # Insertar la cabecera de ejecuciones
        df_ejecucion_cabecera = pd.DataFrame(columns=['timestamp'])
        df_ejecucion_cabecera.loc[0,'timestamp'] = return_opt_date
    
        # Tambien insertar la ejecucion en la tabla de utilidades chata linea poza
        df_ejecucion_cabecera.to_sql('ChataLineaPozaUtilidadesEjec', connection, if_exists='append', index=False)
        
        # Insertar primero la cabecera de flags
        tsql_chunksize = 2097 // len(return_flags_head.columns)
        # cap at 1000 (limit for number of rows inserted by table-value constructor)
        tsql_chunksize = 1000 if tsql_chunksize > 1000 else tsql_chunksize
        
        # Antes reemplazar timestamp por la fecha de ejecucion del algoritmo de retorno (no siempre son las mismas)
        return_flags_head['timestamp'] = return_opt_date
        # Insertar el orden de la tabla de ejecuciones
        last_execution = get_data_from_db.get_id_utilidad_chata_linea_poza()
        return_flags_head['id_ejecucion'] = int(last_execution['id_ejecucion'])

        return_flags_head.to_sql('ChataLineaPozaUtilidadCabecera', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize)
        
        # Insertar los ordenes + flags
        tsql_chunksize_order = 2097 // len(return_flags_ordenes.columns)
        tsql_chunksize_order = 1000 if tsql_chunksize_order > 1000 else tsql_chunksize_order
        
        # Aca tmb reemplazar el timestamp por la fecha de ejecucion de modelo retorno
        return_flags_ordenes['timestamp'] = return_opt_date
        
        return_flags_ordenes.to_sql('ChataLineaPozaRecomUtilidad', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_order)
        
        # Insertar la tabla de utilidades
        # Completar id ejecucion
        last_execution = get_data_from_db.get_id_utilidad_chata_linea_poza()
        return_utility_table['id_ejecucion'] = int(last_execution['id_ejecucion'])
        # df_tabla_utilidad['id_ejecucion'] = np.nan
        cols_final = return_utility_table.columns.tolist()
        cols_final = cols_final[-1:] + cols_final[:-1]
        # df_tabla_utilidad = df_tabla_utilidad[cols_final]
        return_utility_table = return_utility_table[cols_final]
        
        # Completar timestamp
        return_utility_table['timestamp'] = return_opt_date
        
        tsql_chunksize_util = 2097 // len(return_utility_table.columns)
        tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
        return_utility_table.to_sql('ChataLineaPozaUtilidadActual', connection, if_exists='replace', index=False, method='multi', chunksize=tsql_chunksize_util)


# Recoms with a associated utility table will have id_cocina_recom, the others won't
def add_id_cocina_recom_for_missing(row, cocina_opt_date):
    if pd.isnull(row['id_cocina_recom']):
        return row['id_planta'] + ' - ' + str(row['recom_type']) + ' - ' + cocina_opt_date.strftime("%m/%d/%Y_%H:%M:%S")
    else:
        return row['id_cocina_recom']


def add_id_chat_lin_poza_recom_for_missing(row, chat_lin_poza_opt_date):
    if pd.isnull(row['id_ChataLineaPoza_recom']):
        return row['id_planta'] + ' - ' + str(row['recom_type']) + ' - ' + chat_lin_poza_opt_date.strftime("%m/%d/%Y_%H:%M:%S")
    else:
        return row['id_ChataLineaPoza_recom']


def insert_cocina_optimization_result(cocina_recoms_df, df_utility_table_agg, cocina_opt_date):

    if 'recom_type' not in cocina_recoms_df:
        cocina_recoms_df['recom_type'] = 'None'

    # Solo agregar id_cocina_recom para los que no tienen. Los que ya tienen es porque su id se genero antes para que coincida con el de utility table
    if len(cocina_recoms_df.apply(add_id_cocina_recom_for_missing, args=(cocina_opt_date,), axis=1)) > 0:
        cocina_recoms_df['id_cocina_recom'] = cocina_recoms_df.apply(add_id_cocina_recom_for_missing, args=(cocina_opt_date,), axis=1)        
        # cocina_recoms_df['id_cocina_recom'] = cocina_recoms_df.apply(add_id_cocina_recom_for_missing, args=(cocina_opt_date,), axis=1)
        cocina_recoms_df['has_utility_table'].fillna(value=False, inplace=True)
        cocina_recoms_df['last_modification'] = cocina_opt_date


    unique_recoms_ids = cocina_recoms_df['id_cocina_recom'].unique()
    cocina_recoms_header_data = []
    for recom_id in unique_recoms_ids:
        cocina_recoms_header_data.append({'id_recom': recom_id, 'last_modification': cocina_opt_date})
    cocina_recoms_header_df = pd.DataFrame(cocina_recoms_header_data, columns=['id_recom', 'last_modification'])

    # print('cocina_recoms_header_df \n', cocina_recoms_header_df)
    # print('cocina_recoms_df \n', cocina_recoms_df)

    cocina_recoms_header_df.to_sql('CocinasRecomendacionHistoricoCabecera', connection, if_exists='append', index=False, method='multi')
    cocina_recoms_df.to_sql('CocinasRecomendacionHistoricoDetalle', connection, if_exists='append', index=False, method='multi')
    cocina_recoms_df.to_sql('CocinasRecomendacionActual', connection, if_exists='replace', index=False,
                            dtype={
                                'id_cocina_recom': sqlalchemy.types.VARCHAR(length=100),
                                'id_planta': sqlalchemy.types.VARCHAR(length=20),
                                'numero_poza': sqlalchemy.types.INTEGER(),
                                'stock_inicial': sqlalchemy.types.FLOAT(),
                                'tdc_mas_antiguo_en_poza': sqlalchemy.types.FLOAT(),
                                'tvn_estimado': sqlalchemy.types.FLOAT(),
                                'last_modification': sqlalchemy.DateTime(),
                                'priorizacion': sqlalchemy.types.INTEGER(),
                                'utilidad': sqlalchemy.types.FLOAT(),
                                'contingencia': sqlalchemy.types.Boolean(),
                                'linea_frio': sqlalchemy.types.Boolean(),
                                'recommended': sqlalchemy.types.Boolean(),
                                'aprovechamiento_frio': sqlalchemy.types.Boolean(),
                                'recom_type': sqlalchemy.types.VARCHAR(length=20),
                                'mix_with_cooking_poza': sqlalchemy.types.INTEGER(),
                                'has_utility_table': sqlalchemy.types.Boolean(),
                            }, method='multi')

    # Antigua logica tabal de utilidades
    # df_utility_table_agg.drop(['weighted_tvn', 'multiple_poza'], inplace=True, axis=1)
    # df_utility_table_agg['last_modification'] = cocina_opt_date
    # df_utility_table_agg.to_sql('CocinaRecomendacionUtilityTable', connection, if_exists='append', index=False)

    # Nueva logica tabla utilidades
    if len(df_utility_table_agg)>0:
    
        # Insertar Tabla Utilidad Cabecera
        df_utilidad_cabecera = pd.DataFrame(columns=['last_modification'])
        df_utilidad_cabecera.loc[0,'last_modification'] = cocina_opt_date
        df_utilidad_cabecera.to_sql('CocinaRecomTablaUtilidadCabecera', connection, if_exists='append', index=False, method='multi')
        
        # Insertar Tabla Utilidad Historico
        last_id_utilidad = get_data_from_db.get_id_utilidad()
        updated_id_utilidad = int(last_id_utilidad['id_utilidad'])
    
        df_utility_table_agg.drop(['weighted_tvn', 'multiple_poza'], inplace=True, axis=1)
        df_utility_table_agg['last_modification'] = cocina_opt_date
        df_id_utilidad = df_utility_table_agg['sort_order'].copy()
        df_id_utilidad = df_id_utilidad.to_frame()
        df_id_utilidad = df_id_utilidad.rename(columns ={'sort_order':'id_utilidad'})
        df_id_utilidad['id_utilidad'] = updated_id_utilidad
        df_id_utilidad = pd.concat([df_id_utilidad,df_utility_table_agg['id_cocina_recom']], axis=1)

        # Drop el id de la tabla original de utilidad
        del df_utility_table_agg['id_cocina_recom']
        
        df_utility_table_agg = pd.concat([df_id_utilidad, df_utility_table_agg], axis=1) 
        
        df_utility_table_agg['tiempo_0'] = df_utility_table_agg['tiempo_0'].astype(str)
        df_utility_table_agg['tiempo_1'] = df_utility_table_agg['tiempo_1'].astype(str)
        df_utility_table_agg['tiempo_2'] = df_utility_table_agg['tiempo_2'].astype(str)
        df_utility_table_agg['tiempo_3'] = df_utility_table_agg['tiempo_3'].astype(str)

        df_utility_table_agg['utilidad'] = df_utility_table_agg['utilidad'].astype(float)
        df_utility_table_agg['utilidad_0'] = df_utility_table_agg['utilidad_1'].astype(float)
        df_utility_table_agg['utilidad_1'] = df_utility_table_agg['utilidad_1'].astype(float)
        df_utility_table_agg['utilidad_2'] = df_utility_table_agg['utilidad_2'].astype(float)
        df_utility_table_agg['utilidad_3'] = df_utility_table_agg['utilidad_3'].astype(float)

        df_utility_table_agg['hours_to_wait_for_next_cook_0'] = df_utility_table_agg['hours_to_wait_for_next_cook_0'].astype(float)
        df_utility_table_agg['hours_to_wait_for_next_cook_1'] = df_utility_table_agg['hours_to_wait_for_next_cook_1'].astype(float)
        df_utility_table_agg['hours_to_wait_for_next_cook_2'] = df_utility_table_agg['hours_to_wait_for_next_cook_2'].astype(float)
        df_utility_table_agg['hours_to_wait_for_next_cook_3'] = df_utility_table_agg['hours_to_wait_for_next_cook_3'].astype(float)

        df_utility_table_agg['pozas_cook_weighted_tvn_0'] = df_utility_table_agg['pozas_cook_weighted_tvn_0'].astype(float)
        df_utility_table_agg['pozas_cook_weighted_tvn_1'] = df_utility_table_agg['pozas_cook_weighted_tvn_1'].astype(float)
        df_utility_table_agg['pozas_cook_weighted_tvn_2'] = df_utility_table_agg['pozas_cook_weighted_tvn_2'].astype(float)
        df_utility_table_agg['pozas_cook_weighted_tvn_3'] = df_utility_table_agg['pozas_cook_weighted_tvn_3'].astype(float)

        df_utility_table_agg['mix_quality_0'] = df_utility_table_agg['mix_quality_0'].astype(str)
        df_utility_table_agg['mix_quality_1'] = df_utility_table_agg['mix_quality_1'].astype(str)
        df_utility_table_agg['mix_quality_2'] = df_utility_table_agg['mix_quality_2'].astype(str)  
        df_utility_table_agg['mix_quality_3'] = df_utility_table_agg['mix_quality_3'].astype(str)
    
        df_utility_table_agg['pozas_cook_tons_0'] = df_utility_table_agg['pozas_cook_tons_0'].astype(float) 
        df_utility_table_agg['pozas_cook_tons_1'] = df_utility_table_agg['pozas_cook_tons_1'].astype(float)  
        df_utility_table_agg['pozas_cook_tons_2'] = df_utility_table_agg['pozas_cook_tons_2'].astype(float)
        df_utility_table_agg['pozas_cook_tons_3'] = df_utility_table_agg['pozas_cook_tons_3'].astype(float)
    
        df_utility_table_agg['sort_order'] = df_utility_table_agg['sort_order'].astype('int64')
         
        import numpy as np
        df_utility_table_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_utility_table_agg.fillna(np.nan, inplace=True)        
        
        print('df_utility_table_agg \n', df_utility_table_agg)
        
        tsql_chunksize = 2097 // len(df_utility_table_agg.columns)
        # cap at 1000 (limit for number of rows inserted by table-value constructor)
        tsql_chunksize = 1000 if tsql_chunksize > 1000 else tsql_chunksize
        
        df_utility_table_agg.to_sql('CocinaRecomTablaUtilidadHistorico', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize)
        
        # Insertar tabla de utilidad actual
        # Preparar logica para traer ultima tabla de utilidad para todas las plantas
    
        # plantas_con_utilidad = df_utility_table_agg['planta'].unique()
        
        # Buscar ultima tabla de utilidad de las plantas
        last_utility_table = get_data_from_db.get_tabla_utilidad_plantas()
        historico_plantas = df_utility_table_agg[~df_utility_table_agg.planta.isin(last_utility_table['planta'].unique())]
        
        if len(historico_plantas)>0:
            df_utilidad_actual = pd.concat([df_utility_table_agg,historico_plantas], axis=0)
        else:
            df_utilidad_actual = df_utility_table_agg.copy()
        
        tsql_chunksize_act = 2097 // len(df_utilidad_actual.columns)
        # cap at 1000 (limit for number of rows inserted by table-value constructor)
        tsql_chunksize_act = 1000 if tsql_chunksize_act > 1000 else tsql_chunksize_act
        
        
        df_utilidad_actual.to_sql('CocinaRecomTablaUtilidadActual', connection, if_exists='replace', index=False, method='multi', chunksize=tsql_chunksize_act)
        
    connection.execute("""update UltActualizacionFechas set fecha = ? where tipo = ?""", cocina_opt_date, 'cocina_optimization')

    print('Finished inserting cocina recommendation')


def insert_chata_linea_poza_optimization_result_new(df_utility_table_agg, chat_lin_poza_opt_date):

    if len(df_utility_table_agg)>0:
    
        # Insertar Tabla Utilidad Cabecera
        df_utilidad_cabecera = pd.DataFrame(columns=['last_modification'])
        df_utilidad_cabecera.loc[0,'last_modification'] = chat_lin_poza_opt_date
        df_utilidad_cabecera.to_sql('ChataLineaPozaRecomTablaUtilidadCabecera', connection, if_exists='append', index=False, method='multi')
        
        # Insertar Tabla Utilidad Historico
        last_id_utilidad = get_data_from_db.get_id_utilidad_chat_lin_poza()
        updated_id_utilidad = int(last_id_utilidad['id_utilidad'])
    
        # df_utility_table_agg.drop(['weighted_tvn', 'multiple_poza'], inplace=True, axis=1)
        df_utility_table_agg['last_modification'] = chat_lin_poza_opt_date
        df_id_utilidad = df_utility_table_agg['id_marea'].copy()
        df_id_utilidad = df_id_utilidad.to_frame()
        df_id_utilidad = df_id_utilidad.rename(columns ={'id_marea':'id_utilidad'})
        df_id_utilidad['id_utilidad'] = updated_id_utilidad
        df_id_utilidad = pd.concat([df_id_utilidad,df_utility_table_agg['id_ChataLineaPoza_recom']], axis=1)

        # Drop el id de la tabla original de utilidad
        del df_utility_table_agg['id_ChataLineaPoza_recom']
        
        df_utility_table_agg = pd.concat([df_id_utilidad, df_utility_table_agg], axis=1)    
        
        df_utility_table_agg.to_sql('ChataLineaPozaRecomTablaUtilidadHistorico', connection, if_exists='append', index=False, method='multi')
        
        # Insertar tabla de utilidad actual
        # Preparar logica para traer ultima tabla de utilidad para todas las plantas
    
        # plantas_con_utilidad = df_utility_table_agg['planta'].unique()
        
        # Buscar ultima tabla de utilidad de las plantas
        last_utility_table = get_data_from_db.get_tabla_utilidad_plantas_chat_lin_poza()
        historico_plantas = df_utility_table_agg[~df_utility_table_agg.planta.isin(last_utility_table['planta'].unique())]
        
        if len(historico_plantas)>0:
            df_utilidad_actual = pd.concat([df_utility_table_agg, historico_plantas], axis=0)
        else:
            df_utilidad_actual = df_utility_table_agg.copy()
        
        df_utilidad_actual.to_sql('ChataLineaPozaRecomTablaUtilidadActual', connection, if_exists='append', index=False, method='multi')
                
    print('Finished inserting Chata Linea Poza Utilidad')


def add_discharge_tvn_to_db(df_pozas_estado_with_tvn, insert_date):
    df_pozas_estado_with_tvn = df_pozas_estado_with_tvn[['marea_id', 'estimated_discharge_tvn']]
    df_pozas_estado_with_tvn['update_date'] = insert_date

    query = "SELECT * from HistoricDischargeTVN"
    mareas_with_discharge_tvn_in_db = pd.read_sql(query, connection)
    mareas_with_discharge_tvn_in_db = mareas_with_discharge_tvn_in_db['marea_id'].tolist()

    mareas_to_insert = df_pozas_estado_with_tvn[df_pozas_estado_with_tvn['marea_id'].isin(mareas_with_discharge_tvn_in_db) == 0]
    mareas_to_update = df_pozas_estado_with_tvn[df_pozas_estado_with_tvn['marea_id'].isin(mareas_with_discharge_tvn_in_db) == 1]

    print('mareas_to_insert', mareas_to_insert)
    # print('mareas_to_update',  mareas_to_update)

    mareas_to_insert.to_sql('HistoricDischargeTVN', connection, if_exists='append', index=False, method='multi')

    # cursor = cnxn.cursor()
    # cursor.fast_executemany = True
    # for index, marea in mareas_to_update.iterrows():
    #     cursor.execute("""UPDATE HistoricDischargeTVN SET discharge_tvn = ?, is_real_value = ? , update_date = ? where marea_id = ?""",
    #                    marea['discharge_tvn'], marea['is_real_value'], marea['update_date'], marea['marea_id'])
    #
    # cnxn.commit()

    return 'success'


def insert_return_plant_utility(df_retorno_utilidad, return_opt_date):
    df_retorno_utilidad_copy = df_retorno_utilidad.copy()
    if len(df_retorno_utilidad_copy.index)>0:
        
        # Insertar primero la cabecera de flags
        tsql_chunksize = 2097 // len(df_retorno_utilidad_copy.columns)
        # cap at 1000 (limit for number of rows inserted by table-value constructor)
        tsql_chunksize = 1000 if tsql_chunksize > 1000 else tsql_chunksize
        
        cols_to_drop = [
            "marea_status",
            "boat_name",
            "planta_retorno",
            "velocidad_retorno",
            "toneladas"
        ]
        df_retorno_utilidad_copy.drop(cols_to_drop, axis=1, inplace=True)
        # Antes reemplazar timestamp por la fecha de ejecucion del algoritmo de retorno (no siempre son las mismas)
        df_retorno_utilidad_copy['timestamp'] = return_opt_date
        # df_retorno_utilidad_copy['COD_RECOM'] = df_retorno_utilidad_copy['timestamp'].astype(str) + df_retorno_utilidad_copy['marea_id'].astype(str)
        df_retorno_utilidad_copy['COD_RECOM'] = df_retorno_utilidad_copy['marea_id'].astype(str) + ' - ' + return_opt_date.strftime("%m/%d/%Y_%H:%M:%S")
        df_master_planta = get_data_from_db.get_id_planta()
        df_retorno_utilidad_copy = pd.merge(df_retorno_utilidad_copy, df_master_planta, how='left', left_on=["PLANTA_1"], right_on=["NOM_PLANTA"])
        to_rename = {
            "ID_PLANTA":"ID_PLANTA_UNO",
            "NOM_PLANTA":"NOM_PLANTA_UNO",
        }
        df_retorno_utilidad_copy.rename(columns=to_rename, inplace=True)
        df_retorno_utilidad_copy = pd.merge(df_retorno_utilidad_copy, df_master_planta, how='left', left_on=["PLANTA_2"], right_on=["NOM_PLANTA"])
        to_rename = {
            "ID_PLANTA":"ID_PLANTA_DOS",
            "NOM_PLANTA":"NOM_PLANTA_DOS",
        }
        df_retorno_utilidad_copy.rename(columns=to_rename, inplace=True)
        df_retorno_utilidad_copy = pd.merge(df_retorno_utilidad_copy, df_master_planta, how='left', left_on=["PLANTA_3"], right_on=["NOM_PLANTA"])
        to_rename = {
            "ID_PLANTA":"ID_PLANTA_TRES",
            "NOM_PLANTA":"NOM_PLANTA_TRES",
        }
        df_retorno_utilidad_copy.rename(columns=to_rename, inplace=True)
        
        to_rename = {
            "marea_id":"ID_MAREA",
            "distancia_1":"CTD_DISTANCIA_UNO",
            "distancia_2":"CTD_DISTANCIA_DOS",
            "distancia_3":"CTD_DISTANCIA_TRES",
            "gph_opt":"CTD_GPH_OPTIMO",
            "gph_max":"CTD_GPH_MAX",
            "SPEED_OPT_km":"CTD_SPEED_OPTIMO",
            "SPEED_MAX_km":"CTD_SPEED_MAX",
            "combustible_opt_1":"CTD_COMBUSTIBLE_OPT_UNO",
            "combustible_opt_2":"CTD_COMBUSTIBLE_OPT_DOS",
            "combustible_opt_3":"CTD_COMBUSTIBLE_OPT_TRES",
            "combustible_max_1":"CTD_COMBUSTIBLE_MAX_UNO",
            "combustible_max_2":"CTD_COMBUSTIBLE_MAX_DOS",
            "combustible_max_3":"CTD_COMBUSTIBLE_MAX_TRES",
            "stock_poza_planta_1":"CTD_STOCK_POZA_UNO",
            "stock_poza_planta_2":"CTD_STOCK_POZA_DOS",
            "stock_poza_planta_3":"CTD_STOCK_POZA_TRES",
            "tvn_poza_planta_1":"CTD_TVN_POZA_UNO",
            "tvn_poza_planta_2":"CTD_TVN_POZA_DOS",
            "tvn_poza_planta_3":"CTD_TVN_POZA_TRES",
            "valor_marea_1":"IMP_VALOR_MAREA_UNO",
            "valor_marea_2":"IMP_VALOR_MAREA_DOS",
            "valor_marea_3":"IMP_VALOR_MAREA_TRES",
            "costo_consumo_comb_1":"IMP_COSTO_CONSUMO_COMB_UNO",
            "costo_consumo_comb_2":"IMP_COSTO_CONSUMO_COMB_DOS",
            "costo_consumo_comb_3":"IMP_COSTO_CONSUMO_COMB_TRES",
            "tradeoff_1":"IMP_TRADEOFF_UNO",
            "tradeoff_2":"IMP_TRADEOFF_DOS",
            "tradeoff_3":"IMP_TRADEOFF_TRES",
            "prioridad":"NUM_PRIORIDAD",
            "eta":"FEH_ETA",
            "eta_plant":"NOM_ETA_PLANTA",
        }
        df_retorno_utilidad_copy.rename(columns=to_rename, inplace=True)
        df_retorno_utilidad_copy['FEH_ULTIMA_MODIFICACION'] = return_opt_date
        # df_retorno_utilidad_copy['FEH_ARRIBO_ESTIMADA'] = np.nan
        # df_retorno_utilidad_copy['FEH_DESCARGA_ESTIMADA'] = np.nan
        cols = [
            "COD_RECOM",
            "ID_MAREA",
            "ID_PLANTA_UNO",
            "NOM_PLANTA_UNO",
            "ID_PLANTA_DOS",
            "NOM_PLANTA_DOS",
            "ID_PLANTA_TRES",
            "NOM_PLANTA_TRES",
            "CTD_DISTANCIA_UNO",
            "CTD_DISTANCIA_DOS",
            "CTD_DISTANCIA_TRES",
            "CTD_GPH_OPTIMO",
            "CTD_GPH_MAX",
            "CTD_SPEED_OPTIMO",
            "CTD_SPEED_MAX",
            "CTD_COMBUSTIBLE_OPT_UNO",
            "CTD_COMBUSTIBLE_OPT_DOS",
            "CTD_COMBUSTIBLE_OPT_TRES",
            "CTD_COMBUSTIBLE_MAX_UNO",
            "CTD_COMBUSTIBLE_MAX_DOS",
            "CTD_COMBUSTIBLE_MAX_TRES",
            "CTD_STOCK_POZA_UNO",
            "CTD_STOCK_POZA_DOS",
            "CTD_STOCK_POZA_TRES",
            "CTD_TVN_POZA_UNO",
            "CTD_TVN_POZA_DOS",
            "CTD_TVN_POZA_TRES",
            "IMP_VALOR_MAREA_UNO",
            "IMP_VALOR_MAREA_DOS",
            "IMP_VALOR_MAREA_TRES",
            "IMP_COSTO_CONSUMO_COMB_UNO",
            "IMP_COSTO_CONSUMO_COMB_DOS",
            "IMP_COSTO_CONSUMO_COMB_TRES",
            "IMP_TRADEOFF_UNO",
            "IMP_TRADEOFF_DOS",
            "IMP_TRADEOFF_TRES",
            "NUM_PRIORIDAD",
            "FEH_ARRIBO_ESTIMADA",
            "FEH_DESCARGA_ESTIMADA",
            "FEH_ETA",
            "NOM_ETA_PLANTA",
            "FEH_ULTIMA_MODIFICACION",
        ]

        # Insertar los ordenes + flags
        tsql_chunksize_order = 2097 // len(df_retorno_utilidad_copy.columns)
        tsql_chunksize_order = 1000 if tsql_chunksize_order > 1000 else tsql_chunksize_order
        
        # Aca tmb reemplazar el timestamp por la fecha de ejecucion de modelo retorno        
        df_retorno_utilidad_copy[cols].sort_values("NUM_PRIORIDAD").to_sql('OP_RECOM_RETORNO_UTILIDAD_ACT', connection, if_exists='replace', index=False, method='multi', chunksize=tsql_chunksize_order)
              
        tsql_chunksize_util = 2097 // len(df_retorno_utilidad_copy.columns)
        tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
        df_retorno_utilidad_copy[cols].sort_values("NUM_PRIORIDAD").to_sql('OP_RECOM_RETORNO_UTILIDAD', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_util)


def insert_fp_densidad_arribo(df_densidad_arribo):
    tsql_chunksize_util = 2097 // len(df_densidad_arribo.columns)
    tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
    df_densidad_arribo.to_sql('DensidadArriboOutputPD', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_util)
    print('End saving densidad arribo')

def insert_fp_horas(df_horas):
    tsql_chunksize_util = 2097 // len(df_horas.columns)
    tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
    df_horas.to_sql('HorasOutputPD', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_util)
    print('End saving horas')

def insert_fp_pd(df_plan_descarga):
    tsql_chunksize_util = 2097 // len(df_plan_descarga.columns)
    tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
    df_plan_descarga.to_sql('PlanDescargaOutput', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_util)
    print('End saving plan de descarga')

def insert_fp_dz(df_descarga_zona):
    tsql_chunksize_util = 2097 // len(df_descarga_zona.columns)
    tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
    df_descarga_zona.to_sql('ProyeccionZonaOutputPD', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_util)
    print('End saving descarga zona')

def insert_fp_dz_backup(df_descarga_zona):
    tsql_chunksize_util = 2097 // len(df_descarga_zona.columns)
    tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
    df_descarga_zona.to_sql('ProyeccionZonaOutputPDBU', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_util)
    print('End saving descarga zona')

def insert_panel_disponibilidad(df_panel):
    tsql_chunksize_util = 2097 // len(df_panel.columns)
    tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
    df_panel.to_sql('PlanDescargaOutputsModelPDAjustado', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_util)
    print('End saving descarga zona')

# This function imports the connection object to this file globally
def import_connection():
    print('Importing connection to insert_data_to_db')

    global connection
    from data.db_connection import connection
