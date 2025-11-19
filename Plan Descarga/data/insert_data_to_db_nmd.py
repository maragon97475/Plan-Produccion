# Third party imports
import pandas as pd
import sqlalchemy
import numpy as np

# Internal imports
from data import get_data_from_db

def insert_recom_return(return_recoms_df, return_opt_date):
    return_recoms_df['retorno_recom_id'] = return_recoms_df['marea_id'].astype(str) + ' - ' + return_opt_date.strftime("%m/%d/%Y_%H:%M:%S")
    return_recoms_df['last_modification'] = return_opt_date

    print('return_recoms_df \n', return_recoms_df)
    #Nueva estructura del MD MIO - Recomendaciones
    ## Tablas maestras
    master_planta_df = get_data_from_db.get_master_planta()
    master_chata_df = get_data_from_db.get_master_chata()
    master_linea_df = get_data_from_db.get_master_linea()
    ## Cruce de tablas para obtener los ID's
    return_recoms_df['COD_CHATA_F'] = return_recoms_df['chata_descarga'].str.replace('CHATA ','')
    return_recoms_df['COD_LINEA_F'] = np.where(return_recoms_df.linea_descarga.isin([None, '', '0', 'S', 'Sur', 'SUR']), 'S',
                                     np.where(return_recoms_df.linea_descarga.isin(['N', 'Norte', 'NORTE']), 'N', 'S'))

    aux = pd.merge(return_recoms_df, master_planta_df, left_on= 'planta_retorno', right_on = 'NOM_PLANTA', how='left')
    aux2 = pd.merge(aux, master_chata_df, left_on= ['COD_CHATA_F','ID_PLANTA'], right_on = ['NOM_CHATA','ID_PLANTA'], how='left')
    aux3 = pd.merge(aux2, master_linea_df, left_on= ['ID_CHATA','COD_LINEA_F'], right_on = ['ID_CHATA','NOM_LINEA'], how='left')

    ## Renombre de la tabla
    return_recoms_df = aux3[['retorno_recom_id', 'marea_id', 'velocidad_retorno', 
        'ID_PLANTA', 'planta_retorno', 'ID_CHATA', 'chata_descarga', 'ID_LINEA', 'linea_descarga',
        'last_modification']].rename(columns = {
        'retorno_recom_id':'COD_RECOM', 
        'marea_id':'ID_MAREA',
        'velocidad_retorno':'DES_VELOCIDAD_RETORNO',
        'ID_PLANTA':'ID_PLANTA_RETORNO',
        'planta_retorno':'NOM_PLANTA_RETORNO',
        'ID_CHATA':'ID_CHATA_RETORNO',
        'chata_descarga':'NOM_CHATA_RETORNO',
        'ID_LINEA':'ID_LINEA_RETORNO',
        'linea_descarga':'NOM_LINEA_RETORNO',
        'last_modification':'FEH_ULTIMA_MODIFICACION'
        })
    # Guardar la data con nueva estructura
    if(len(return_recoms_df)>0):
        return_recoms_df.to_sql('OP_RECOM_RETORNO_ACT', connection, if_exists='replace', index=False, method='multi')
        return_recoms_df.to_sql('OP_RECOM_RETORNO', connection, if_exists='append', index=False, method='multi')
    # Actualizar la tabla de fechas de recomendaciones
    #connection.execute("""update UltActualizacionFechas set fecha = ? where tipo = ?""", return_opt_date, 'optimization')


def insert_recom_discharge(return_recoms_df, return_opt_date):
    return_recoms_df['retorno_recom_id'] = return_recoms_df['marea_id'].astype(str) + ' - ' + return_opt_date.strftime("%m/%d/%Y_%H:%M:%S")
    return_recoms_df['last_modification'] = return_opt_date
    #Nueva estructura del MD MIO - Recomendaciones
    ## Tablas maestras
    master_planta_df = get_data_from_db.get_master_planta()
    master_chata_df = get_data_from_db.get_master_chata()
    master_linea_df = get_data_from_db.get_master_linea()
    master_poza_df = get_data_from_db.get_master_poza()
    master_poza_df_1 = master_poza_df.rename(columns={'ID_POZA': 'ID_POZA_UNO', 'NUM_POZA':'NUM_POZA_UNO', 'ID_PLANTA':'ID_PLANTA_UNO'})
    master_poza_df_2 = master_poza_df.rename(columns={'ID_POZA': 'ID_POZA_DOS', 'NUM_POZA':'NUM_POZA_DOS', 'ID_PLANTA':'ID_PLANTA_DOS'})
    master_poza_df_3 = master_poza_df.rename(columns={'ID_POZA': 'ID_POZA_TRES', 'NUM_POZA':'NUM_POZA_TRES', 'ID_PLANTA':'ID_PLANTA_TRES'})
    master_poza_df_4 = master_poza_df.rename(columns={'ID_POZA': 'ID_POZA_CUATRO', 'NUM_POZA':'NUM_POZA_CUATRO', 'ID_PLANTA':'ID_PLANTA_CUATRO'})

    ## Cruce de tablas para obtener los ID's
    return_recoms_df['COD_CHATA_F'] = return_recoms_df['chata_descarga'].str.replace('CHATA ','')
    return_recoms_df['COD_LINEA_F'] = np.where(return_recoms_df.linea_descarga.isin([None, '', '0', 'S', 'Sur', 'SUR']), 'S',
                                     np.where(return_recoms_df.linea_descarga.isin(['N', 'Norte', 'NORTE']), 'N', 'S'))
    mask = return_recoms_df['poza_descarga_1'].notna()
    aux = pd.merge(return_recoms_df[mask], master_planta_df, left_on= 'planta_retorno', right_on = 'NOM_PLANTA', how='left')
    aux2 = pd.merge(aux, master_chata_df, left_on= ['COD_CHATA_F','ID_PLANTA'], right_on = ['NOM_CHATA','ID_PLANTA'], how='left')
    aux3 = pd.merge(aux2, master_linea_df, left_on= ['ID_CHATA','COD_LINEA_F'], right_on = ['ID_CHATA','NOM_LINEA'], how='left')
    aux3['poza_descarga_1'] = aux3['poza_descarga_1'].astype(int)
    aux4 = pd.merge(aux3, master_poza_df_1, left_on= ['poza_descarga_1','ID_PLANTA'], right_on = ['NUM_POZA_UNO','ID_PLANTA_UNO'], how='left')
    mask = aux4['poza_descarga_2'].isna()
    aux4.loc[mask, 'poza_descarga_2'] = 29
    aux5 = pd.merge(aux4, master_poza_df_2, left_on= ['poza_descarga_2','ID_PLANTA'], right_on = ['NUM_POZA_DOS','ID_PLANTA_DOS'], how='left')
    mask = aux5['poza_descarga_3'].isna()
    aux5.loc[mask, 'poza_descarga_3'] = 29
    aux6 = pd.merge(aux5, master_poza_df_3, left_on= ['poza_descarga_3','ID_PLANTA'], right_on = ['NUM_POZA_TRES','ID_PLANTA_TRES'], how='left')
    mask = aux6['poza_descarga_4'].isna()
    aux6.loc[mask, 'poza_descarga_4'] = 29
    aux7 = pd.merge(aux6, master_poza_df_4, left_on= ['poza_descarga_4','ID_PLANTA'], right_on = ['NUM_POZA_CUATRO','ID_PLANTA_CUATRO'], how='left')

    ## Renombre de la tabla
    return_recoms_df = aux7[['retorno_recom_id','marea_id','ID_PLANTA','planta_retorno','ID_CHATA',
        'chata_descarga','ID_LINEA','linea_descarga','orden_descarga','orden_descarga_global',
        'ID_POZA_UNO','poza_descarga_1','tons_poza_descarga_1','ID_POZA_DOS','poza_descarga_2',
        'tons_poza_descarga_2','ID_POZA_TRES','poza_descarga_3','tons_poza_descarga_3','ID_POZA_CUATRO',
        'poza_descarga_4','tons_poza_descarga_4','last_modification']].rename(columns = {
        'retorno_recom_id':'COD_RECOM',
        'marea_id':'ID_MAREA',
        'ID_PLANTA':'ID_RECOM_PLANTA',
        'planta_retorno':'NOM_RECOM_PLANTA',
        'ID_CHATA':'ID_RECOM_CHATA',
        'chata_descarga':'NOM_RECOM_CHATA',
        'ID_LINEA':'ID_RECOM_LINEA',
        'linea_descarga':'NOM_RECOM_LINEA',
        'orden_descarga':'NUM_ORDEN_DESC',
        'orden_descarga_global':'NUM_ORDEN_DESC_GLOBAL',
        'ID_POZA_UNO':'ID_RECOM_POZA_UNO',
        'poza_descarga_1':'NUM_RECOM_POZA_UNO',
        'tons_poza_descarga_1':'VOL_RECOM_POZA_UNO',
        'ID_POZA_DOS':'ID_RECOM_POZA_DOS',
        'poza_descarga_2':'NUM_RECOM_POZA_DOS',
        'tons_poza_descarga_2':'VOL_RECOM_POZA_DOS',
        'ID_POZA_TRES':'ID_RECOM_POZA_TRES',
        'poza_descarga_3':'NUM_RECOM_POZA_TRES',
        'tons_poza_descarga_3':'VOL_RECOM_POZA_TRES',
        'ID_POZA_CUATRO':'ID_RECOM_POZA_CUATRO',
        'poza_descarga_4':'NUM_RECOM_POZA_CUATRO',
        'tons_poza_descarga_4':'VOL_RECOM_POZA_CUATRO',
        'last_modification':'FEH_ULTIMA_MODIFICACION'})

    print('return_recoms_df \n', return_recoms_df)
    if(len(return_recoms_df)>0):
        print('entro')
        return_recoms_df.to_sql('OP_RECOM_DESCARGA_ACT', connection, if_exists='replace', index=False, method='multi')
        print('Insert recoms discharge act')
        return_recoms_df.to_sql('OP_RECOM_DESCARGA', connection, if_exists='append', index=False, method='multi')
        print('Insert recoms discharge historical')
    # Actualizar la tabla de fechas de recomendaciones
    # connection.execute("""update UltActualizacionFechas set fecha = ? where tipo = ?""", return_opt_date, 'optimization')


def insert_return_optimization_errors(return_recoms_errors, return_opt_date):
    return_recoms_errors['recom_date'] = return_opt_date
    print(return_recoms_errors)
    mask = return_recoms_errors['error_category_id'] == 'eta null'
    return_recoms_errors.loc[mask, 'error_category_id'] = 10
    master_embarcacion_df = get_data_from_db.get_master_embarcacion()
    aux = pd.merge(return_recoms_errors, master_embarcacion_df, left_on= 'boat_name', right_on = 'NOM_EMBARCACION', how='left')
    ## Renombre de la tabla
    return_recoms_errors = aux[['error_is_global', 'global_error_description', 'marea_id', 'ID_EMBARCACION', 'boat_name', 'recom_state', 'owner_group', 'error_category_id', 'recom_date',
        'recom_date']].rename(columns = {
        'error_is_global':'FLG_GLOBAL', 
        'global_error_description':'DES_GLOBAL_ERROR',
        'marea_id':'ID_MAREA',
        'ID_EMBARCACION':'ID_EMBARCACION',
        'boat_name':'NOM_EMBARCACION',
        'recom_state':'DES_ESTADO_RECOM',
        'owner_group':'TIP_EMBARCACION',
        'error_category_id':'TIP_CATEGORIA',
        'recom_date':'FEH_RECOMENDACION'
        })
        
    if(len(return_recoms_errors)>0):
        return_recoms_errors.to_sql('OP_RECOM_ERRORES', connection, if_exists='append', index=False, method='multi')
    print('Inserting return optimization errors')

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
        df_retorno_utilidad_copy['COD_RECOM'] = df_retorno_utilidad_copy['timestamp'].astype(str) + df_retorno_utilidad_copy['marea_id'].astype(str)

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
            "prioridad":"NUM_PRIORIDAD"
        }
        df_retorno_utilidad_copy.rename(columns=to_rename, inplace=True)
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
            "NUM_PRIORIDAD"
        ]
        # Insertar los ordenes + flags
        tsql_chunksize_order = 2097 // len(df_retorno_utilidad_copy.columns)
        tsql_chunksize_order = 1000 if tsql_chunksize_order > 1000 else tsql_chunksize_order
        
        # Aca tmb reemplazar el timestamp por la fecha de ejecucion de modelo retorno        
        df_retorno_utilidad_copy[cols].sort_values("NUM_PRIORIDAD").to_sql('OP_RECOM_RETORNO_UTILIDAD_ACT', connection, if_exists='replace', index=False, method='multi', chunksize=tsql_chunksize_order)
              
        tsql_chunksize_util = 2097 // len(df_retorno_utilidad_copy.columns)
        tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
        df_retorno_utilidad_copy[cols].sort_values("NUM_PRIORIDAD").to_sql('OP_RECOM_RETORNO_UTILIDAD', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_util)


def insert_discharge_utility(return_flags_head, return_flags_ordenes, return_opt_date, return_utility_table):
    
    # if len(return_flags_head)>0:
    mask = return_flags_ordenes['chata_rec_orden'] == 'CHATA CHILLON'
    return_flags_ordenes.loc[mask, 'chata_rec_orden'] = 'CHATA TASA CALLAO'
    # Insertar la cabecera de ejecuciones
    df_ejecucion_cabecera = pd.DataFrame(columns=['timestamp'])
    df_ejecucion_cabecera.loc[0,'timestamp'] = return_opt_date

    # Insertar primero la cabecera de flags
    # Antes reemplazar timestamp por la fecha de ejecucion del algoritmo de retorno (no siempre son las mismas)
    return_flags_head['retorno_recom_id'] = return_flags_head['marea_id'].astype(str) + ' - ' + return_opt_date.strftime("%m/%d/%Y_%H:%M:%S")
    return_flags_head['timestamp'] = return_opt_date
    # return_flags_head.fillna(np.nan, inplace=True)
    # Insertar el orden de la tabla de ejecuciones

    #Nueva estructura del MD MIO - Recomendaciones

    ## Obtener las tablas maestras
    master_planta_df = get_data_from_db.get_master_planta()
    master_chata_df = get_data_from_db.get_master_chata()
    master_linea_df = get_data_from_db.get_master_linea()
    master_poza_df = get_data_from_db.get_master_poza()
    master_poza_df_1 = master_poza_df.rename(columns={'ID_POZA': 'ID_POZA_UNO', 'NUM_POZA':'NUM_POZA_UNO', 'ID_PLANTA':'ID_PLANTA_UNO'})
    master_poza_df_2 = master_poza_df.rename(columns={'ID_POZA': 'ID_POZA_DOS', 'NUM_POZA':'NUM_POZA_DOS', 'ID_PLANTA':'ID_PLANTA_DOS'})
    master_poza_df_3 = master_poza_df.rename(columns={'ID_POZA': 'ID_POZA_TRES', 'NUM_POZA':'NUM_POZA_TRES', 'ID_PLANTA':'ID_PLANTA_TRES'})
    master_poza_df_4 = master_poza_df.rename(columns={'ID_POZA': 'ID_POZA_CUATRO', 'NUM_POZA':'NUM_POZA_CUATRO', 'ID_PLANTA':'ID_PLANTA_CUATRO'})

    ## Cruce de tablas para obtener los ID's
    return_flags_head['COD_CHATA_F'] = return_flags_head['discharge_chata_name'].str.replace('CHATA ','')
    return_flags_head['COD_LINEA_F'] = np.where(return_flags_head.discharge_line_name.isin([None, '', '0', 'S', 'Sur', 'SUR']), 'S',
                                    np.where(return_flags_head.discharge_line_name.isin(['N', 'Norte', 'NORTE']), 'N', 'S'))
    # mask = return_flags_head['discharge_poza_1'].notna()
    aux = pd.merge(return_flags_head, master_planta_df, left_on= 'discharge_plant_name', right_on = 'NOM_PLANTA', how='left')
    aux2 = pd.merge(aux, master_chata_df, left_on= ['COD_CHATA_F','ID_PLANTA'], right_on = ['NOM_CHATA','ID_PLANTA'], how='left')
    aux3 = pd.merge(aux2, master_linea_df, left_on= ['ID_CHATA','COD_LINEA_F'], right_on = ['ID_CHATA','NOM_LINEA'], how='left')
    mask = aux3['discharge_poza_1'].isna()
    aux3.loc[mask, 'discharge_poza_1'] = 29
    aux4 = pd.merge(aux3, master_poza_df_1, left_on= ['discharge_poza_1','ID_PLANTA'], right_on = ['NUM_POZA_UNO','ID_PLANTA_UNO'], how='left')
    mask = aux4['discharge_poza_2'].isna()
    aux4.loc[mask, 'discharge_poza_2'] = 29  
    aux5 = pd.merge(aux4, master_poza_df_2, left_on= ['discharge_poza_2','ID_PLANTA'], right_on = ['NUM_POZA_DOS','ID_PLANTA_DOS'], how='left')
    mask = aux5['discharge_poza_3'].isna()
    aux5.loc[mask, 'discharge_poza_3'] = 29  
    aux6 = pd.merge(aux5, master_poza_df_3, left_on= ['discharge_poza_3','ID_PLANTA'], right_on = ['NUM_POZA_TRES','ID_PLANTA_TRES'], how='left')
    mask = aux6['discharge_poza_4'].isna()
    aux6.loc[mask, 'discharge_poza_4'] = 29  
    aux7 = pd.merge(aux6, master_poza_df_4, left_on= ['discharge_poza_4','ID_PLANTA'], right_on = ['NUM_POZA_CUATRO','ID_PLANTA_CUATRO'], how='left')

    for col in ['discharge_poza_1', 'discharge_poza_2', 'discharge_poza_3', 'discharge_poza_4']:
        mask = aux7[col] == 29
        aux7.loc[mask, col] = np.nan
    ## Renombre de la tabla
    return_flags_head = aux7[['retorno_recom_id','timestamp','marea_id','ID_PLANTA','discharge_plant_name',
        'ID_CHATA','discharge_chata_name','ID_LINEA','discharge_line_name','discharge_start_date',
        'tdc_arrival','tvn_discharge','ID_POZA_UNO','discharge_poza_1','ID_POZA_DOS','discharge_poza_2',
        'ID_POZA_TRES','discharge_poza_3','ID_POZA_CUATRO','discharge_poza_4',
        'tipo_bodega','frio_system_state','bodega_frio','restriccion_ep','chata_linea_ingresada']].rename(columns = {
        'retorno_recom_id':'COD_RECOM',
        'timestamp':'FEH_ULTIMA_MODIFICACION',
        'marea_id':'ID_MAREA',
        'ID_PLANTA':'ID_PLANTA_DESCARGA',
        'discharge_plant_name':'NOM_PLANTA_DESCARGA',
        'chata_descarga':'NOM_RECOM_CHATA',
        'ID_CHATA':'ID_CHATA_DESCARGA',
        'discharge_chata_name':'NOM_CHATA_DESCARGA',
        'ID_LINEA':'ID_LINEA_DESCARGA',
        'discharge_line_name':'NOM_LINEA_DESCARGA',
        'discharge_start_date':'FEH_INICIO_DESCARGA',
        'tdc_arrival':'CTD_TDC_DESCARGA',
        'tvn_discharge':'CTD_TVN_DESCARGA',
        'ID_POZA_UNO':'ID_POZA_UNO',
        'discharge_poza_1':'NUM_POZA_UNO',
        'ID_POZA_DOS':'ID_POZA_DOS',
        'discharge_poza_2':'NUM_POZA_DOS',
        'ID_POZA_TRES':'ID_POZA_TRES',
        'discharge_poza_3':'NUM_POZA_TRES',
        'ID_POZA_CUATRO':'ID_POZA_CUATRO',
        'discharge_poza_4':'NUM_POZA_CUATRO',
        'tipo_bodega':'TIP_BODEGA',
        'frio_system_state':'TIP_ESTADO_FRIO',
        'bodega_frio':'TIP_BODEGA_FRIO',
        'restriccion_ep':'FLG_RESTRICCION_EP',
        'chata_linea_ingresada':'FLG_CHATA_LINEA_INGRESADA'
        })

    tsql_chunksize = 2097 // len(return_flags_head.columns)
    tsql_chunksize = 1000 if tsql_chunksize > 1000 else tsql_chunksize
    if(len(return_flags_head)>0):
        return_flags_head.to_sql('OP_RECOM_DESCARGA_UTILIDAD_CAB', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize)
    print('Inserting discharge optimization utility cab')
    
    ## Cruce de tablas para obtener los ID's
    return_flags_ordenes['retorno_recom_id'] = return_flags_ordenes['marea_id'].astype(str) + ' - ' + return_opt_date.strftime("%m/%d/%Y_%H:%M:%S")

    return_flags_ordenes['COD_CHATA_F'] = return_flags_ordenes['chata_rec_orden'].str.replace('CHATA ','')
    return_flags_ordenes['COD_LINEA_F'] = np.where(return_flags_ordenes.linea_rec_orden.isin([None, '', '0', 'S', 'Sur', 'SUR']), 'S',
                                    np.where(return_flags_ordenes.linea_rec_orden.isin(['N', 'Norte', 'NORTE']), 'N', 'S'))

    # aux = pd.merge(return_flags_ordenes, master_planta_df, left_on= 'discharge_plant_name', right_on = 'NOM_PLANTA', how='left')
    aux2 = pd.merge(return_flags_ordenes, master_chata_df, left_on= ['COD_CHATA_F'], right_on = ['NOM_CHATA'], how='left')
    aux3 = pd.merge(aux2, master_linea_df, left_on= ['ID_CHATA','COD_LINEA_F'], right_on = ['ID_CHATA','NOM_LINEA'], how='left')
    aux3['poza1_rec_orden'] = aux3['poza1_rec_orden'].fillna(29).astype(int)
    aux4 = pd.merge(aux3, master_poza_df_1, left_on= ['poza1_rec_orden','ID_PLANTA'], right_on = ['NUM_POZA_UNO','ID_PLANTA_UNO'], how='left')
    aux4['poza2_rec_orden'] = aux4['poza2_rec_orden'].fillna(29).astype(int)
    aux5 = pd.merge(aux4, master_poza_df_2, left_on= ['poza2_rec_orden','ID_PLANTA'], right_on = ['NUM_POZA_DOS','ID_PLANTA_DOS'], how='left')
    aux5['poza3_rec_orden'] = aux5['poza3_rec_orden'].fillna(29).astype(int)
    aux6 = pd.merge(aux5, master_poza_df_3, left_on= ['poza3_rec_orden','ID_PLANTA'], right_on = ['NUM_POZA_TRES','ID_PLANTA_TRES'], how='left')
    aux6['poza4_rec_orden'] = aux6['poza4_rec_orden'].fillna(29).astype(int)
    aux7 = pd.merge(aux6, master_poza_df_4, left_on= ['poza4_rec_orden','ID_PLANTA'], right_on = ['NUM_POZA_CUATRO','ID_PLANTA_CUATRO'], how='left')
    return_flags_ordenes['timestamp'] = return_opt_date
    ## Renombre de la tabla
    return_flags_ordenes = aux7[['id_orden','retorno_recom_id','timestamp','marea_id','tdc_ponderado',
        'capacidad_pozas','tipo_poza','flag_preservante','flag_presion_vacio','dif_tdc_ep_poza',
        'tvn_previo_desc_ep','tvn_poza','flag_limite_emb','flag_limite_tiempo','flag_balanceo','tdc_previo_desc_ep',
        'orden_emb','es_pama','ID_CHATA','chata_rec_orden',
        'ID_LINEA','linea_rec_orden','ID_POZA_UNO','poza1_rec_orden','ID_POZA_DOS','poza2_rec_orden',
        'ID_POZA_TRES','poza3_rec_orden','ID_POZA_CUATRO', 'poza4_rec_orden'
        ]].rename(columns = {
        'id_orden':'ID_ORDEN',
        'retorno_recom_id':'COD_RECOM',
        'timestamp':'FEH_ULTIMA_MODIFICACION',
        'marea_id':'ID_MAREA',
        'tdc_ponderado':'CTD_TDC_PONDERADO',
        'capacidad_pozas':'CTD_CAPACIDAD_POZA',
        'tipo_poza':'FLG_DESCARGA_FRIA',
        'flag_preservante':'FLG_PRESERVANTE',
        'flag_presion_vacio':'FLG_PRESION_VACIO',
        'dif_tdc_ep_poza':'CTD_DIFERENCIA_POZA',
        'tvn_previo_desc_ep':'CTD_TVN_PREVIO_DESCARGA_EP',
        'tvn_poza':'CTD_TVN_POZA',
        'flag_limite_emb':'FLG_LIMITE_EMB',
        'flag_limite_tiempo':'FLG_LIMITE_TIEMPO',
        'flag_balanceo':'FLG_BALANCEO',
        'tdc_previo_desc_ep':'CTD_TDC_PREVIO_DESCARGA_EP',
        'orden_emb':'NUM_ORDEN_EP',
        'es_pama':'FLG_PAMA',
        'ID_CHATA':'ID_CHATA_ORDEN',
        'chata_rec_orden':'DES_CHATA_ORDEN',
        'ID_LINEA':'ID_LINEA_ORDEN',
        'linea_rec_orden':'DES_LINEA_ORDEN',
        'ID_POZA_UNO':'ID_POZA_ORDEN_UNO',
        'poza1_rec_orden':'NUM_POZA_ORDEN_UNO',
        'ID_POZA_DOS':'ID_POZA_ORDEN_DOS',
        'poza2_rec_orden':'NUM_POZA_ORDEN_DOS',
        'ID_POZA_TRES':'ID_POZA_ORDEN_TRES',
        'poza3_rec_orden':'NUM_POZA_ORDEN_TRES',
        'ID_POZA_CUATRO':'ID_POZA_ORDEN_CUATRO',
        'poza4_rec_orden':'NUM_POZA_ORDEN_CUATRO'
        })

    # Insertar los ordenes + flags
    tsql_chunksize_order = 2097 // len(return_flags_ordenes.columns)
    tsql_chunksize_order = 1000 if tsql_chunksize_order > 1000 else tsql_chunksize_order
    
    # Aca tmb reemplazar el timestamp por la fecha de ejecucion de modelo retorno
    # return_flags_ordenes['timestamp'] = return_opt_date
    if(len(return_flags_ordenes)>0):
        return_flags_ordenes.to_sql('OP_RECOM_DESCARGA_UTILIDAD', connection, if_exists='append', index=False, method='multi', chunksize=tsql_chunksize_order)
    print('Inserting discharge optimization utility')

    # Insertar la tabla de utilidades
    # df_tabla_utilidad['id_ejecucion'] = np.nan
    cols_final = return_utility_table.columns.tolist()
    cols_final = cols_final[-1:] + cols_final[:-1]
    # df_tabla_utilidad = df_tabla_utilidad[cols_final]
    return_utility_table = return_utility_table[cols_final]
    
    # Completar timestamp
    return_utility_table['timestamp'] = return_opt_date

    return_utility_table.rename(columns = {
        'discharge_plant_name':'NOM_PLANTA_DESCARGA',
        'marea_id':'ID_MAREA',
        'boat_name':'NOM_EMBARCACION',
        'discharge_plant_arrival_date':'FEH_ARRIBO_PLANTA',
        'id_orden':'ID_ORDEN',
        'tvn_pozas_ponderado':'CTD_TVN_POZAS_PONDERADO',
        'orden_emb':'NUM_ORDEN_EMBARCACIONES',
        'chata_rec_orden':'NOM_CHATA_RECOM_ORDEN',
        'linea_rec_orden':'NOM_LINEA_RECOM_ORDEN',
        'poza1_rec_orden':'NOM_POZA1_RECOM_ORDEN',
        'poza2_rec_orden':'NOM_POZA2_RECOM_ORDEN',
        'poza3_rec_orden':'NOM_POZA3_RECOM_ORDEN',
        'poza4_rec_orden':'NOM_POZA4_RECOM_ORDEN',
        'tdc_previo_desc_ep':'CTD_TDC_PREVIO_DESC',
        'tvn_previo_desc_ep':'CTD_TVN_PREVIO_DESC',
        'tvn_poza':'TVN_POZA',
        'tipo_poza':'TIP_POZA',
        'dif_tdc_ep_poza':'CTD_DIFERENCIA_TDC_EP_POZA',
        'flag_preservante':'FLG_PRESERVANTE',
        'flag_presion_vacio':'FLG_PRESION_VACIO',
        'flag_balanceo':'FLG_BALANCEO',
        'msg_tipo_poza':'DES_MSG_TIPO_POZA',
        'msg_dif_tdc':'DES_MSG_DIF_TDC',
        'msg_preservante':'DES_MSG_PRESERVANTE',
        'msg_presion_vacio':'DES_MSG_PRESION_VACIO',
        'msg_balanceo':'DES_MSG_BALANCEO',
        'timestamp':'FEH_ACTUALIZACION'
        })
    
    tsql_chunksize_util = 2097 // len(return_utility_table.columns)
    tsql_chunksize_util = 1000 if tsql_chunksize_util > 1000 else tsql_chunksize_util
    return_utility_table.to_sql('OP_RECOM_DESCARGA_UTILIDAD_ACT', connection, if_exists='replace', index=False, method='multi', chunksize=tsql_chunksize_util)

def add_discharge_tvn_to_db(df_pozas_estado_with_tvn, insert_date):
    
    df_pozas_estado_with_tvn = df_pozas_estado_with_tvn[['marea_id', 'estimated_discharge_tvn']]
    df_pozas_estado_with_tvn['FEH_MODIFICACION'] = insert_date

    df_pozas_estado_with_tvn.rename(columns = {
        'marea_id':'ID_MAREA',
        'estimated_discharge_tvn':'CTD_TVN_ESTIMADO',
        'FEH_MODIFICACION':'FEH_MODIFICACION'
    })

    query = "SELECT ID_MAREA, CTD_TVN_ESTIMADO, FEH_MODIFICACION from MA_HISTORICO_DESCARGA_TVN"
    mareas_with_discharge_tvn_in_db = pd.read_sql(query, connection)
    mareas_with_discharge_tvn_in_db = mareas_with_discharge_tvn_in_db['ID_MAREA'].tolist()

    mareas_to_insert = df_pozas_estado_with_tvn[df_pozas_estado_with_tvn['ID_MAREA'].isin(mareas_with_discharge_tvn_in_db) == 0]
    mareas_to_update = df_pozas_estado_with_tvn[df_pozas_estado_with_tvn['ID_MAREA'].isin(mareas_with_discharge_tvn_in_db) == 1]

    print('mareas_to_insert', mareas_to_insert)
    # print('mareas_to_update',  mareas_to_update)
    # [MA_HISTORICO_DESCARGA_TVN]
    mareas_to_insert.to_sql('MA_HISTORICO_DESCARGA_TVN', connection, if_exists='append', index=False, method='multi')

    # cursor = cnxn.cursor()
    # cursor.fast_executemany = True
    # for index, marea in mareas_to_update.iterrows():
    #     cursor.execute("""UPDATE HistoricDischargeTVN SET discharge_tvn = ?, is_real_value = ? , update_date = ? where marea_id = ?""",
    #                    marea['discharge_tvn'], marea['is_real_value'], marea['update_date'], marea['marea_id'])
    #
    # cnxn.commit()

    return 'success'


# This function imports the connection object to this file globally
def import_connection():
    print('Importing connection to insert_data_to_db_nmd')

    global connection
    from data.db_connection import connection
