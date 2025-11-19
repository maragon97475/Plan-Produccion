## IMPORTS
# Third party imports
import itertools
import pandas as pd
import random
import datetime
import math
import sys
from math import radians, cos, sin, asin, sqrt, isnan
import numpy as np
import traceback
import time
import os
import ast
from ast import literal_eval

from data.get_dumped_data import get_dumped_data_date
from models.tvn_prediction import get_return_model_polynomial, get_tvn_increase
from itertools import chain


def run_return_optimization(get_data):
    print('Started to run return optimization: ' + os.getenv("ENVIRONMENT"))

    ## CARGAR DATOS

    # Planta habilitada o no (según vista Ingreso datos)
    df_plantas_habilitadas = get_data.get_plantas_habilitadas()

    # Info de mareas activas según lo leído de sap, info estática de EPs (tipo de bodega, velocidades, GPH) y ubicación de PC
    df_embarcaciones = get_data.get_active_mareas_with_location_and_static_data()
    # df_embarcaciones['feh_arribo_geocerca'] = np.nan
    # df_embarcaciones['feh_arribo_geocerca'] = pd.to_datetime(df_embarcaciones['feh_arribo_geocerca'])
    df_embarcaciones['first_cala_start_date']=pd.to_datetime(df_embarcaciones['first_cala_start_date'])

    # Actualizar fecha de arribo con geocerca
    df_embarcaciones['discharge_plant_name'] = np.where(~df_embarcaciones['discharge_plant_arrival_date'].notnull(), df_embarcaciones['eta_plant'], df_embarcaciones['discharge_plant_name'])
    df_embarcaciones['discharge_plant_arrival_date'] = np.where(~df_embarcaciones['discharge_plant_arrival_date'].notnull(), df_embarcaciones['feh_arribo_geocerca'], df_embarcaciones['discharge_plant_arrival_date'])
    # Actualizar estado de embarcacion
    df_embarcaciones['marea_status'] = np.where(df_embarcaciones['feh_arribo_geocerca'].notnull() & ((df_embarcaciones['marea_status']=='PESCANDO') | (df_embarcaciones['marea_status']=='RETORNANDO A PUERTO')), 'ESPERANDO DESCARGA', df_embarcaciones['marea_status'])
    
    df_embarcaciones['discharge_plant_arrival_date'] = pd.to_datetime(df_embarcaciones['discharge_plant_arrival_date'])
    # df_embarcaciones['feh_arribo_geocerca'] = pd.to_datetime(df_embarcaciones['feh_arribo_geocerca'])
    

    # Detalle de mareas que han descargado en poza (si ha sido alimentada, su TDC, su bodega, su declarado y descargado)
    df_pozas_estado = get_data.get_pozas_estado()
    df_pozas_estado_cocina = df_pozas_estado.copy()
    df_pozas_estado_leido = df_pozas_estado.copy()
    
    mask_poza = (df_pozas_estado['id_planta']=='CALLAO') & (df_pozas_estado['pozaNumber']==9)
    cap_poza_9 = df_pozas_estado.loc[mask_poza,'pozaCapacity'].unique().item()
    df_pozas_estado.loc[mask_poza,'pozaCapacity'] = 0
    mask_poza = (df_pozas_estado['id_planta']=='CALLAO') & (df_pozas_estado['pozaNumber']==8)
    cap_poza_8 = df_pozas_estado.loc[mask_poza,'pozaCapacity'].unique().item()
    df_pozas_estado.loc[mask_poza,'pozaCapacity'] = cap_poza_8 + cap_poza_9
    
    # Listado de pozas por planta con información estática (hermanadas y capacidad)
    df_pozas_ubicacion_capacidad = get_data.get_ubicacion_capacidad_pozas()
    df_pozas_ubicacion_capacidad['fec_deshabilitado'] = pd.to_datetime(df_pozas_ubicacion_capacidad['fec_deshabilitado'])

    # Velocidades promedios de descarga por línea según rango de declarado y si está habilidad la línea
    df_tiempo_descarga = get_data.get_lineas_velocidad_descarga()

    # Requerimiento del día de la plata (8-8)
    df_requerimiento_plantas = get_data.get_requerimiento_planta()
    df_requerimiento_plantas['requerimiento'] = df_requerimiento_plantas['daily_requirement']
    df_requerimiento_plantas['requerimiento']=np.where(df_requerimiento_plantas['daily_requirement'].isnull(),df_requerimiento_plantas['requerimiento_por_defecto'],df_requerimiento_plantas['requerimiento'])
    df_requerimiento_plantas['date']=pd.to_datetime((df_requerimiento_plantas['hora_inicio_last_modification'] - pd.Timedelta(hours= 5)).dt.date)
    df_requerimiento_plantas['hora_inicio']=pd.to_datetime(df_requerimiento_plantas['date'].astype(str)+' '+df_requerimiento_plantas['hora_inicio'].astype(str))

    # Líneas reservadas para terceros (según vista Ingreso datos)
    df_lineas_reservada_terceros = get_data.get_lineas_reservadas_terceros()
    
    # Velocidad de planta (cocina) según lo ingresado por operario
    df_plantas_velocidad_limites = get_data.get_requerimiento_planta()
    df_plantas_velocidad_limites = df_plantas_velocidad_limites[~df_plantas_velocidad_limites['id'].duplicated()].reset_index(drop=True)
    df_planta_velocidad_anterior = get_data.get_plantas_velocidades_historico()
    df_calidades_precio_venta = get_data.get_calidades_precio_venta()

    # Mínimo porcentaje de llenado de bodega para recomendar retorno (según vista Ingreso datos)
    df_min_perc_bodego_recom_retorno = float(get_data.get_minimo_perc_bodega_recom_retorno()['perc'])

    # Info estática del costo de combustible por galón (es un valor único)
    costo_combustible = float(get_data.get_costo_combustible()['price_per_gallon'])

    # Info de cada poza: hermanadas, orden de prioridad
    # df_pozas_hermanadas_and_priority = get_data.get_pozas_hermanadas()

    #Tabla estática de restricciones y minimos_planta
    df_restricciones=get_data.get_data_restricciones()
    df_minimos_planta=get_data.get_minimos_planta()
    df_minimos_planta.rename(columns={'planta':'code','tn_min':'minimo_arranque'},inplace=True)

    df_plantas_velocidad_limites['requerimiento'] = df_plantas_velocidad_limites['daily_requirement']
    df_plantas_velocidad_limites['requerimiento']=np.where(df_plantas_velocidad_limites['daily_requirement'].isnull(),df_plantas_velocidad_limites['requerimiento_por_defecto'],df_plantas_velocidad_limites['requerimiento'])
    df_plantas_velocidad_limites=df_plantas_velocidad_limites.merge(df_minimos_planta,on='code',how='left')

    # df_marea_web_services=get_data.get_marea_web_service()
    # df_marea_web_services['discharge_plant_name'].replace('CHICAMA', 'MALABRIGO',inplace=True)
    # df_marea_web_services['discharge_plant_name'].replace('MOLLENDO', 'MATARANI',inplace=True)


    df_priorizacion_linea=get_data.get_priorizacion_linea()
    df_priorizacion_linea['chata'].replace('CHATA EX-ABA', 'CHATA EXABA',inplace=True)
    df_priorizacion_linea['chata-linea']=df_priorizacion_linea['chata']+'-'+df_priorizacion_linea['linea']


    df_horas_produccion=get_data.get_hora_inicio()
    df_horas_produccion['hora']=pd.to_datetime(df_horas_produccion['hour_production'].astype(str))
    df_horas_produccion['date_production']=pd.to_datetime(df_horas_produccion['date_production'])

    df_mareas_cerradas=get_data.get_mareas_cerradas()
    df_mareas_cerradas['discharge_plant_name'].replace('CHICAMA', 'MALABRIGO',inplace=True)
    df_mareas_cerradas['discharge_plant_name'].replace('MOLLENDO', 'MATARANI',inplace=True)
    to_replace = {'PISCO SUR':'PISCO'}
    df_mareas_cerradas.replace({'discharge_plant_name':to_replace}, inplace=True)
    df_mareas_cerradas['discharge_plant_arrival_date']=pd.to_datetime(df_mareas_cerradas['discharge_plant_arrival_date'])

    # Comentarios de categorias de flags de condiciones algoritmo esperando descarga
    df_comentarios = get_data.get_category_flags()

    df_mareas_acodere=get_data.get_mareas_acodere()
    
    # Renombrar las  mareas que tengan termino de succion pero no fin de descarga
    df_desc_no_validas = df_embarcaciones[['marea_id','discharge_end_date']].merge(df_mareas_acodere[['marea_id','termino_succion']], how='left', on='marea_id')
    df_desc_no_validas = df_desc_no_validas[(df_desc_no_validas['discharge_end_date'].isnull()) & (df_desc_no_validas['termino_succion'].notnull())]
    df_embarcaciones['marea_status'] = np.where(df_embarcaciones['marea_id'].isin(df_desc_no_validas['marea_id']), 'EP ESPERANDO ZARPE', df_embarcaciones['marea_status'])
    df_desc_no_validas = df_embarcaciones[df_embarcaciones['marea_id'].isin(df_desc_no_validas['marea_id'])]
    
    # Renombrar las mareas que tengan inicio de succion pero no inicio de descarga
    mask_error_desc = (df_embarcaciones['discharge_start_date'].isnull()) & (df_embarcaciones['inicio_succion'].notnull())
    df_error_desc = df_embarcaciones[mask_error_desc]
    df_embarcaciones['marea_status'] = np.where(df_embarcaciones['marea_id'].isin(df_error_desc['marea_id']), 'DESCARGANDO', df_embarcaciones['marea_status'])
    df_desc_no_validas = pd.concat([df_desc_no_validas, df_error_desc], axis=0)
    
    df_prioridad_pozas = get_data.get_prioridad_pozas()
    df_prioridad_pozas['NOM_CHATA'] = 'CHATA ' + df_prioridad_pozas['NOM_CHATA']
    df_prioridad_pozas['NOM_CHATA'] = df_prioridad_pozas['NOM_CHATA'].str.replace('TASA CALLAO','CHILLON')
    df_prioridad_pozas['COD_UNICO'] = df_prioridad_pozas['NOM_CHATA'] + '-' + df_prioridad_pozas['NOM_LINEA']
    df_velocidad_descarga = get_data.get_velocidad_descarga_chata()
    df_velocidad_descarga['NOM_CHATA_COMPLETO'] = 'CHATA ' + df_velocidad_descarga['NOM_CHATA_DESCARGA']
    
    #Procedimiento para cambiar el status de las embarcaciones descargando cuya última recomendación fue nula a esperando descarga
    df_ultimas_recomendaciones=get_data.get_recomendaciones_ultimodia()
    df_ultimas_recomendaciones_orig = df_ultimas_recomendaciones.copy()
    set_embarcaciones_descargando=set(df_embarcaciones[df_embarcaciones.marea_status=='DESCARGANDO'].marea_id)
    idx = df_ultimas_recomendaciones.groupby(['marea_id'])['last_modification'].transform(max) == df_ultimas_recomendaciones['last_modification']
    df_ultimas_recomendaciones=df_ultimas_recomendaciones[idx].reset_index(drop=True).copy()
    set_emb_desc_con_recom=set(df_ultimas_recomendaciones[(df_ultimas_recomendaciones.marea_id.isin(set_embarcaciones_descargando))&(df_ultimas_recomendaciones.orden_descarga_global.notnull())].marea_id)
    emb_desc_sin_recom=set_embarcaciones_descargando-set_emb_desc_con_recom
    df_embarcaciones.loc[df_embarcaciones.marea_id.isin(emb_desc_sin_recom),'marea_status']='ESPERANDO DESCARGA'
    
    df_chatas_lineas=get_data.get_chata_linea()
    df_chatas_lineas=df_chatas_lineas.replace(to_replace ="CHATA EX-ABA", value ="CHATA EXABA")
    df_chatas_lineas['chatalinea']=df_chatas_lineas['id_chata']+'-'+df_chatas_lineas['id_linea']

    df_master_fajas = get_data.get_master_fajas()
    mask = df_master_fajas['NOM_FAJA'] == 'np.nan'
    df_master_fajas.loc[mask, 'NOM_FAJA'] = np.nan

    # mask = df_master_fajas['NOM_LINEA'].isin(['CHATA TAMAKUN-S', 'CHATA TAMAKUN-N', 'CHATA TANGARARA-S', 'CHATA TANGARARA-N'])
    # df_master_fajas = df_master_fajas[mask]

    print('Finalizo carga de datos')

    # def get_return_model_polynomial(temp):
    #     temp = temp.reset_index(drop=True)
    #     limite_tdc_frio = 22
    #     limite_tdc_trad = 20
    #     slope_frio = 0.2
    #     slope_trad = 0.9
    #     lista_tvn = []
    #     for index, row in temp.iterrows():
    #         if row['TDC-Desc'] is None:
    #             lista_tvn.append(0)
    #         else:
    #             if (row['Bodega_Frio'] == 1) & (row['TDC-Desc']>=limite_tdc_frio):
    #                 tvn_inicial = 1.46184258e-05*limite_tdc_frio**5 -1.38737645e-03*limite_tdc_frio**4 + 4.92813314e-02*limite_tdc_frio**3 -8.12249406e-01*limite_tdc_frio**2 + 6.34283538e+00*limite_tdc_frio
    #                 tdc_adicional = row['TDC-Desc'] - limite_tdc_frio
    #                 lista_tvn.append((tvn_inicial + tdc_adicional*slope_frio))
    #             elif (row['Bodega_Frio'] == 0) & (row['TDC-Desc']>=limite_tdc_trad):
    #                 tvn_inicial = 1.46184258e-05* limite_tdc_trad**5 -1.38737645e-03*limite_tdc_trad**4 + 4.92813314e-02*limite_tdc_trad**3 -8.12249406e-01*limite_tdc_trad**2 + 6.34283538e+00*limite_tdc_trad
    #                 tdc_adicional = row['TDC-Desc'] - limite_tdc_trad
    #                 lista_tvn.append((tvn_inicial + tdc_adicional*slope_trad))
    #             elif (row['Bodega_Frio'] == 2) & (row['TDC-Desc']>=limite_tdc_frio):
    #                 tvn_inicial = 1.46184258e-05*limite_tdc_frio**5 -1.38737645e-03*limite_tdc_frio**4 + 4.92813314e-02*limite_tdc_frio**3 -8.12249406e-01*limite_tdc_frio**2 + 6.34283538e+00*limite_tdc_frio
    #                 tdc_adicional = row['TDC-Desc'] - limite_tdc_frio
    #                 lista_tvn.append((tvn_inicial + tdc_adicional*slope_frio))
    #             else:
    #                 lista_tvn.append((1.46184258e-05* row['TDC-Desc']**5 -1.38737645e-03*row['TDC-Desc']**4 + 4.92813314e-02*row['TDC-Desc']**3 -8.12249406e-01*row['TDC-Desc']**2 + 6.34283538e+00*row['TDC-Desc'])*row['Bodega_Frio']+(1.46184258e-05* row['TDC-Desc']**5 -1.38737645e-03*row['TDC-Desc']**4 + 4.92813314e-02*row['TDC-Desc']**3 -8.12249406e-01*row['TDC-Desc']**2 + 6.34283538e+00*row['TDC-Desc'])*(row['Bodega_Tradicional'])) # Se elimina la bodega estanca
    #     return pd.Series(lista_tvn)

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
        
    estimate_tvn_sin_frio_vect = np.vectorize(estimate_tvn_sin_frio, otypes=[object]) 

    def estimate_tvn_golpe_frio(tdc):
        """
        Estima el TVN de las embarcaciones con golpe de frío.
        """
        if tdc is None:
            return None
        if tdc <= 21:
            return 0.0009 * tdc ** 3 - 0.0312 * tdc ** 2 + 0.7375 * tdc + 14
        else:
            return 14 * np.exp(0.0350 * tdc) - 5.15

    estimate_tvn_golpe_frio_vect = np.vectorize(estimate_tvn_golpe_frio) 

    def estimate_tvn_con_frio(tdc):
        """
        Estima el TVN de las embarcaciones con RCW.
        """
        if tdc is None:
            return None
        if tdc <= 22:
            return 0.0008 * tdc ** 3 - 0.0332 * tdc ** 2 + 0.6275 * tdc + 14
        else:
            return 14.5 * np.exp(0.0136 * tdc) + 0.7

    estimate_tvn_con_frio_vect = np.vectorize(estimate_tvn_con_frio, otypes=[object])  
    
    def get_return_model_polynomial(temp):
        temp_copy = temp.reset_index(drop=True)
        temp_copy['tvn_estimado'] = 0
        
        mask = temp_copy['Bodega_Frio'] == 0
        temp_copy.loc[mask, 'tvn_estimado'] = estimate_tvn_sin_frio_vect(temp_copy.loc[mask, 'TDC-Desc'])

        mask = temp_copy['Bodega_Frio'] == 1
        temp_copy.loc[mask, 'tvn_estimado'] = estimate_tvn_con_frio_vect(temp_copy.loc[mask, 'TDC-Desc'])

        mask = temp_copy['Bodega_Frio'] == 2
        temp_copy.loc[mask, 'tvn_estimado'] = estimate_tvn_con_frio_vect(temp_copy.loc[mask, 'TDC-Desc'])

        return pd.Series(temp_copy['tvn_estimado'].tolist())
    
    def get_tvn_increase_in_poza_for_marea_opt(marea_row):
        return get_tvn_increase(marea_row['tiempo_consumo_cocina_opt'], marea_row['bodega_frio'], False, marea_row['id_planta'])

    def get_tvn_increase_in_poza_for_marea_max(marea_row):
        return get_tvn_increase(marea_row['tiempo_consumo_cocina_max'], marea_row['bodega_frio'], False, marea_row['id_planta'])
    
    def get_tvn_increase(residence_hours, working_frio, con_hielo, planta):
        if not working_frio and not con_hielo:
            if residence_hours < 6.25:
                return 1.38 * residence_hours
            else:
                return 1.47 * residence_hours
    
        elif not working_frio and con_hielo:
            if planta=='CHIMBOTE':
                ratio_chimbote = 1
                return ratio_chimbote * residence_hours
            elif planta=='VEGUETA':
                ratio_vegueta = 0.5
                return ratio_vegueta * residence_hours
            elif residence_hours < 12:
                return 1.04 * residence_hours
            else:
                return 1.04 * 12 + 1.15 * (residence_hours - 12)
    
        elif working_frio and not con_hielo:
            if residence_hours < 7.6:
                return 1.09 * residence_hours
            else:
                return 1.09 * 7.6 + 1.23 * (residence_hours - 7.6)
    
        elif working_frio and con_hielo :
            if planta=='CHIMBOTE':
                ratio_chimbote = 1
                return ratio_chimbote * residence_hours
            elif planta=='VEGUETA':
                ratio_vegueta = 0.5
                return ratio_vegueta * residence_hours
            elif residence_hours < 7.7:
                return 0.54 * residence_hours
            else:
                return 0.54 * 7.7 + 0.57 * (residence_hours - 7.7)
    
    def replace_percentage_of_bodega_for_thirds(marea_row):
        if marea_row['owner_group'] == 'T':
            marea_row['pp_llenado_bodega'] = 75
            marea_row['tipo_bodega'] = 'Tradicional'
            return marea_row
        else:
            return marea_row

    ## combinar data de pozas y definir nivel y tvn
    def data_pozas(df_pozas_ubicacion_capacidad, df_pozas_estado):
        #Se crea la variable con_hielo en caso no exista
        if not ('con_hielo' in df_pozas_estado.columns):
            df_pozas_estado['con_hielo']=np.where(df_pozas_estado.frio_system_state=='CH',True,False)

        ## obtener informacion de: 'id_planta','id_chata','id_linea','poza_number','coordslatitude','coordslongitude'
        df_pozas_ubicacion_capacidad = df_pozas_ubicacion_capacidad[['id_planta','id_chata','id_linea','poza_number','coordslatitude','coordslongitude']].copy()

        df_pozas_ubicacion_capacidad = df_pozas_ubicacion_capacidad[df_pozas_ubicacion_capacidad['poza_number'].notnull()].copy()
        df_pozas_ubicacion_capacidad['id'] = df_pozas_ubicacion_capacidad['id_planta'] + '-' + df_pozas_ubicacion_capacidad['poza_number'].map(int).map(str)

        df_pozas_estado = df_pozas_estado.apply(replace_percentage_of_bodega_for_thirds, axis=1)
        df_pozas_estado.loc[df_pozas_estado['owner_group']=='T','tipo_bodega']='Tradicional'

        ## agregar informacion de las descargas en la poza
        df_pozas_estado['id'] = df_pozas_estado['id_planta'] + '-' + df_pozas_estado['pozaNumber'].map(int).map(str)
        # Mantener stock actual tal cual se registro
        # df_pozas_estado['fin_cocina'] = np.nan
        lista = list(df_pozas_estado[df_pozas_estado['deshabilitado']==0]['id'].unique())
        df_temp = pd.DataFrame(columns=['id','id_planta','pozaNumber','pozaCapacity','nivel','tvn','tipo_conservacion','con_hielo','frio_system_state'])
        for poz in lista:
            temp = df_pozas_estado[(df_pozas_estado['id']==poz)&(df_pozas_estado['fin_cocina'].isnull())&(df_pozas_estado['stock_actual']>0)].reset_index(drop=True).copy()
            if len(temp) == 0:
                fin_cocina=df_pozas_estado[(df_pozas_estado['id']==poz)&(df_pozas_estado['fin_cocina'].notnull())].fin_cocina.max()
                t = pd.DataFrame(columns=['id','id_planta','pozaNumber','pozaCapacity','nivel','tvn','tipo_conservacion','con_hielo','frio_system_state'])
                t.loc[0,'id'] = poz
                t.loc[0,'id_planta'] = df_pozas_estado[df_pozas_estado['id']==poz].reset_index(drop=True).loc[0,'id_planta']
                t.loc[0,'pozaNumber'] = df_pozas_estado[df_pozas_estado['id']==poz].reset_index(drop=True).loc[0,'pozaNumber']
                t.loc[0,'pozaCapacity'] = df_pozas_estado[df_pozas_estado['id']==poz].reset_index(drop=True).loc[0,'pozaCapacity']
                t.loc[0,'nivel'] = 0
                t.loc[0,'tvn'] = 0
                t.loc[0,'tipo_conservacion'] = np.nan
                t.loc[0,'con_hielo'] = df_pozas_estado[df_pozas_estado['id']==poz].reset_index(drop=True).loc[0,'con_hielo']
                t.loc[0,'frio_system_state'] = df_pozas_estado[df_pozas_estado['id']==poz].reset_index(drop=True).loc[0,'frio_system_state']
                t.loc[0,'fin_cocina']=fin_cocina
                # df_temp = df_temp.append(t)
                df_temp = pd.concat([df_temp, t], ignore_index=True)
            else:
                temp['% Llenado'] = temp['pp_llenado_bodega']
                temp['Descarg.'] = temp['declared_ton']
                temp['TDC-Desc'] = temp['tdc_discharge']
                temp['Bodega_Estanca'] = 0
                temp['Bodega_Frio'] = 0
                temp['Bodega_Tradicional'] = 0
                temp.loc[temp['tipo_bodega']=='Tradicional','Bodega_Tradicional']=1
                temp.loc[temp['tipo_bodega']=='Frio','Bodega_Frio']=1
                temp.loc[temp['tipo_bodega']=='Estanca','Bodega_Estanca']=1
                try:
                    temp['TDC_2'] = np.power(temp['TDC-Desc'],2)
                except:
                    temp['TDC_2'] = 0
                try:
                    temp['tvn'] = np.where((temp['stock_actual']>0) & (temp['tvn_discharge'].isnull()), get_return_model_polynomial(temp[['% Llenado', 'Descarg.', 'TDC-Desc', 'TDC_2', 'Bodega_Estanca', 'Bodega_Frio', 'Bodega_Tradicional']]), temp['tvn_discharge'])
                    temp['tvn'] = temp['tvn'].fillna(0)
                    temp['tvn'] = temp['tvn'].astype(float)
                except:
                    temp['tvn']=0
                temp.loc[temp.declared_ton.isnull(),'declared_ton']=0
                # temp = temp.groupby(['id']).agg({'id_planta':'first','pozaNumber':'first','pozaCapacity':'first','declared_ton':'sum','tvn':'mean','tipo_bodega':'first','con_hielo':'first','frio_system_state':'first'}).reset_index()
                temp = temp.groupby(['id']).agg({'id_planta':'first','pozaNumber':'first','pozaCapacity':'first','stock_actual':'max','tvn':'mean','tipo_bodega':'first','con_hielo':'first','frio_system_state':'first'}).reset_index()
                temp = temp.rename(columns={'stock_actual': 'nivel','tipo_bodega':'tipo_conservacion'})
                # df_temp = df_temp.append(temp)
                df_temp = pd.concat([df_temp, temp], ignore_index=True)
        df_temp = df_temp.reset_index(drop=True)
        df_pozas_estado = pd.merge(df_temp,df_pozas_ubicacion_capacidad[['id','id_chata','id_linea','coordslatitude','coordslongitude']], how='left', on ='id')
        df_pozas_estado = df_pozas_estado[df_pozas_estado['coordslatitude'].notnull()].copy()
        df_pozas_estado = df_pozas_estado[df_pozas_estado['coordslongitude'].notnull()].copy()
        df_pozas_estado = df_pozas_estado.rename(columns={'pozaNumber': 'poza_number'})
        df_pozas_estado.loc[df_pozas_estado['nivel']>df_pozas_estado['pozaCapacity'],'nivel']=df_pozas_estado['pozaCapacity']
        df_pozas_estado = df_pozas_estado.reset_index(drop=True)
        df_pozas_estado.loc[df_pozas_estado['tvn'].isnull(),'tvn'] = 0

        ## cambiar nombre chata
        return df_pozas_estado

    def limpieza_df_embarcaciones(df_embarcaciones, df_ultimas_recomendaciones_orig):
        # df_embarcaciones = df_embarcaciones.replace(to_replace =['PISCO',"PISCO SUR",'PISCO NORTE'],value ="PISCO SUR") 
        df_embarcaciones['discharge_plant_name'] = df_embarcaciones['discharge_plant_name'].replace(to_replace =['PISCO',"PISCO SUR",'PISCO NORTE'],value ="PISCO SUR") 
        df_embarcaciones.loc[df_embarcaciones['owner_group']=='T','tipo_bodega']='Tradicional'
        df_embarcaciones = df_embarcaciones.replace(to_replace ="CHATA EX-ABA", value ="CHATA EXABA")
        df_embarcaciones.loc[(df_embarcaciones['discharge_chata_name'] == 'CHATA SARIMON')&(df_embarcaciones['discharge_line_name'] == 'S'), 'discharge_line_name'] = 'N'
        df_embarcaciones['discharge_plant_name'].replace('CHICAMA', 'MALABRIGO', inplace=True)
        df_embarcaciones['discharge_plant_name'].replace('MOLLENDO', 'MATARANI', inplace=True)
        df_embarcaciones['lineaname'] = df_embarcaciones['discharge_plant_name'].map(str) + '-' + df_embarcaciones['discharge_chata_name'].map(str) + '-' + df_embarcaciones['discharge_line_name'].map(str)
        df_embarcaciones['discharge_start_date'] = pd.to_datetime(df_embarcaciones['discharge_start_date'])
        df_embarcaciones['inicio_succion'] = pd.to_datetime(df_embarcaciones['inicio_succion'])
        # df_embarcaciones['discharge_start_date'] = np.minimum(df_embarcaciones['inicio_succion'], df_embarcaciones['discharge_start_date'])
        df_embarcaciones['discharge_start_date'] = np.where(df_embarcaciones['inicio_succion']<df_embarcaciones['discharge_start_date'], df_embarcaciones['inicio_succion'], df_embarcaciones['discharge_start_date'])
        df_embarcaciones['discharge_start_date'] = df_embarcaciones['discharge_start_date'].fillna(df_embarcaciones['inicio_succion'])
        df_embarcaciones['fish_zone_arrival_date'] = pd.to_datetime(df_embarcaciones['fish_zone_arrival_date'])
        df_embarcaciones['first_cala_start_date'] = pd.to_datetime(df_embarcaciones['first_cala_start_date'])
        df_embarcaciones['discharge_plant_arrival_date'] = pd.to_datetime(df_embarcaciones['discharge_plant_arrival_date'])
        df_embarcaciones['eta'] = pd.to_datetime(df_embarcaciones['eta'])

        # Completar recomendaciones de EPs descargando con ultima recomendacion registrada si es que no se registro
        mask_chata_asig = df_embarcaciones['discharge_chata_name'].isna()
        mask_poza_asig = df_embarcaciones['discharge_poza_1'].isna()
        mask_desc = df_embarcaciones['marea_status']=='DESCARGANDO'
        df_emb_desc_faltante = df_embarcaciones[(mask_chata_asig | mask_poza_asig) & mask_desc].reset_index(drop=True)
        df_ultimas_recomendaciones_orig['last_modification'] = pd.to_datetime(df_ultimas_recomendaciones_orig['last_modification'])
        df_embarcaciones['acodera_chata'] = pd.to_datetime(df_embarcaciones['acodera_chata'])
        
        for index_marea,row_marea in df_emb_desc_faltante.iterrows():
            fecha_descarga = max(row_marea['discharge_start_date'],row_marea['inicio_succion'])
            # fecha_acodere = df_embarcaciones.loc[df_embarcaciones['marea_id']==row_marea['marea_id'],'acodera_chata'].item()
            fecha_acodere = max(df_embarcaciones.loc[df_embarcaciones['marea_id']==row_marea['marea_id'],'acodera_chata'])
            fecha_filtro = min(fecha_descarga,fecha_acodere)
    
            mask_marea = df_ultimas_recomendaciones_orig['marea_id']==row_marea['marea_id']
            mask_fecha = df_ultimas_recomendaciones_orig['last_modification']<=fecha_filtro
            df_last_recom = df_ultimas_recomendaciones_orig[mask_marea & mask_fecha].sort_values(['last_modification'], ascending=False)
            df_last_recom = df_last_recom.head(1).reset_index(drop=True)
            try:
                df_embarcaciones.loc[df_embarcaciones['marea_id']==row_marea['marea_id'],'discharge_chata_name'] = df_last_recom['chata_descarga'].item()
                df_embarcaciones.loc[df_embarcaciones['marea_id']==row_marea['marea_id'],'discharge_line_name'] = df_last_recom['linea_descarga'].item()
                df_embarcaciones.loc[df_embarcaciones['marea_id']==row_marea['marea_id'],'discharge_poza_1'] = df_last_recom['poza_descarga_1'].item()
            except:
                pass
        df_embarcaciones['PORC_CAPACIDAD'] = df_embarcaciones['declared_ton'] / df_embarcaciones['capacidad_bodega_real']
        df_embarcaciones['PORC_CAPACIDAD'] = df_embarcaciones['PORC_CAPACIDAD'] * 100
        
        df_embarcaciones.loc[df_embarcaciones['owner_group']=='T','PORC_CAPACIDAD']=50
        df_embarcaciones.loc[df_embarcaciones['owner_group']=='T','tipo_bodega']='Tradicional'

        df_embarcaciones['SPEED_OPT_km'] =  df_embarcaciones['speed_opt'] * 1.852
        df_embarcaciones['SPEED_MAX_km'] =  df_embarcaciones['speed_max'] * 1.852

        ## por ahora
        df_embarcaciones['first_cala_start_date'] = df_embarcaciones['first_cala_start_date'].fillna(df_embarcaciones['fish_zone_arrival_date'])
        ## change year
        df_embarcaciones = df_embarcaciones.reset_index(drop=True)
        ##df_embarcaciones['first_cala_start_date'] = df_embarcaciones['first_cala_start_date'].apply(lambda dt: dt.replace(year=2021)) 

        return df_embarcaciones

    def limpieza_df_tiempo_descarga(df_tiempo_descarga):
        df_tiempo_descarga = df_tiempo_descarga.replace(to_replace ="CHATA EX-ABA", value ="CHATA EXABA") 
        return df_tiempo_descarga

    def limpieza_df_pozas_estado(df_pozas_estado, df_plantas_estado_habilitadas, df_requerimiento_plantas,df_tiempo_descarga):
        lista_requerimiento_planta_cero = list(df_requerimiento_plantas[df_requerimiento_plantas['requerimiento']==0]['id'])
        df_plantas_estado_habilitadas.loc[df_plantas_estado_habilitadas['planta'].isin(lista_requerimiento_planta_cero),'habilitada_general'] = False
        lista_df_plantas_estado_habilitadas = list(df_plantas_estado_habilitadas[df_plantas_estado_habilitadas['habilitada_general']==True]['planta'].unique())
        df_pozas_estado = df_pozas_estado[df_pozas_estado['id_planta'].isin(lista_df_plantas_estado_habilitadas)]
        df_pozas_estado = df_pozas_estado.replace(to_replace =["PISCO","PISCO SUR",'PISCO NORTE'],value ="PISCO SUR") 
        df_pozas_estado = df_pozas_estado.replace(to_replace ="CHATA EX-ABA", value ="CHATA EXABA")
        df_pozas_estado=df_pozas_estado.merge(df_tiempo_descarga[['id_planta','id_chata','id_linea','sistema_absorbente']],how='left',left_on=['id_planta','id_chata','id_linea'],right_on=['id_planta','id_chata','id_linea'])
        return df_pozas_estado

    ## dividir datos por estados y extraer la informacion relevante
    def dividir_embarcaciones(df_embarcaciones, timestamp, porcentaje_llenado_recom, df_pozas_estado):
        
        df_embarcaciones_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
    
        ##barcos descargando
        df_embarcaciones_descargando = df_embarcaciones[df_embarcaciones['marea_status']=='DESCARGANDO'][['marea_id','boat_name','PORC_CAPACIDAD','first_cala_start_date','discharge_start_date','declared_ton','discharge_plant_name','discharge_chata_name','discharge_line_name','tipo_bodega','frio_system_state','owner_group','discharge_poza_1','discharge_poza_2','discharge_poza_3','marea_status', 'inicio_succion', 'termino_succion']].reset_index(drop=True)  
    
        # errores toneladas 0 o menor 0
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_desc_errores = df_embarcaciones_descargando[df_embarcaciones_descargando['declared_ton']<=0].copy()
        # df_temp_errores['marea_id'] = df_embarcaciones_descargando[df_embarcaciones_descargando['declared_ton']<=0]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_desc_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_desc_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'descargando'
        df_temp_errores['tipo_error'] = 'toneladas 0 o menor a 0'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_descargando = df_embarcaciones_descargando[df_embarcaciones_descargando['declared_ton']>0]
    
        # errores discharge_start_date null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_desc_errores = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_start_date'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_start_date'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_desc_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_desc_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'descargando'
        df_temp_errores['tipo_error'] = 'discharge_start_date null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_descargando = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_start_date'].notnull()]
    
        # errores discharge_plant_name null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_desc_errores = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_plant_name'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_plant_name'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_desc_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_desc_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'descargando'
        df_temp_errores['tipo_error'] = 'discharge_plant_name null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_descargando = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_plant_name'].notnull()]
    
        # errores discharge_chata_name null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_desc_errores = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_chata_name'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_chata_name'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_desc_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_desc_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'descargando'
        df_temp_errores['tipo_error'] = 'discharge_chata_name null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_descargando = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_chata_name'].notnull()]
    
        # errores discharge_line_name null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_desc_errores = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_chata_name'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_chata_name'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_desc_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_desc_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'descargando'
        df_temp_errores['tipo_error'] = 'discharge_line_name null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_descargando = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_line_name'].notnull()]
    
        # errores discharge_poza_1 null # TODO: Agregar if en optimización esperando 
        # df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp']) 
        # bd_desc_errores = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_poza_1'].isnull()].copy()
        # # df_temp_errores['embarcaciones'] = list(df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_poza_1'].isnull()]['boat_name'].unique())
        # df_temp_errores['marea_id'] = bd_desc_errores['marea_id']
        # df_temp_errores['embarcaciones'] = bd_desc_errores['boat_name']
        # df_temp_errores['tipo_emb'] = 'descargando'
        # df_temp_errores['tipo_error'] = 'discharge_poza_1 null'
        # df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        # df_embarcaciones_descargando = df_embarcaciones_descargando[df_embarcaciones_descargando['discharge_poza_1'].notnull()]
        df_embarcaciones_descargando = df_embarcaciones_descargando.reset_index(drop=True)
    
        ## barcos esperando descarga
        df_embarcaciones_esperando_descarga = df_embarcaciones[df_embarcaciones['marea_status']=='ESPERANDO DESCARGA']
    
        # errores toneladas 0 o menor 0
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])        
        bd_esp_desc_errores = df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga['declared_ton']<=0].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga['declared_ton']<=0]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_esp_desc_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_esp_desc_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'esperando_descarga'
        df_temp_errores['tipo_error'] = 'toneladas 0 o menor a 0'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_esperando_descarga = df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga['declared_ton']>0]
    
        # errores discharge_plant_name null
        df_embarcaciones_esperando_descarga['discharge_plant_name'].replace('CHICAMA', 'MALABRIGO',inplace=True)
        df_embarcaciones_esperando_descarga['discharge_plant_name'].replace('MOLLENDO', 'MATARANI',inplace=True)
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_esp_desc_errores = df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga['discharge_plant_name'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga['discharge_plant_name'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_esp_desc_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_esp_desc_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'esperando_descarga'
        df_temp_errores['tipo_error'] = 'discharge_plant_name null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_esperando_descarga = df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga['discharge_plant_name'].notnull()]
        df_embarcaciones_esperando_descarga = df_embarcaciones_esperando_descarga.reset_index(drop=True)
    
        # Eliminar logica de no ingresar las ep con acodere a la optimizacion
        # mask_order_1 = df_embarcaciones_esperando_descarga['discharge_chata_name'].notna() & df_embarcaciones_esperando_descarga['acodera_chata'].notna()
        # mask_order_2 = df_embarcaciones_esperando_descarga['discharge_chata_name'].isna() & df_embarcaciones_esperando_descarga['acodera_chata'].notna()

        # df_adic = df_embarcaciones_esperando_descarga[(mask_order_1 | mask_order_2)]
        # df_embarcaciones_esperando_descarga = df_embarcaciones_esperando_descarga[~(mask_order_1 | mask_order_2)]
        
        ## barcos calando y retornando a puerto
        porcentaje_llenado_recom = porcentaje_llenado_recom
        df_embarcaciones_retornando = df_embarcaciones[((df_embarcaciones['marea_status']=='PESCANDO')&(df_embarcaciones['PORC_CAPACIDAD']>=porcentaje_llenado_recom))|(df_embarcaciones['marea_status']=='RETORNANDO A PUERTO')]
        df_embarcaciones_retornando = df_embarcaciones_retornando[df_embarcaciones_retornando['owner_group']=='P']
    
        # errores toneladas 0 o menor 0
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_retorno_errores = df_embarcaciones_retornando[df_embarcaciones_retornando['declared_ton']<=0].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_retornando[df_embarcaciones_retornando['declared_ton']<=0]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_retorno_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_retorno_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'asignar'
        df_temp_errores['tipo_error'] = 'toneladas 0 o menor a 0'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_retornando = df_embarcaciones_retornando[df_embarcaciones_retornando['declared_ton']>0]
    
        # errores Latitud null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_retorno_errores = df_embarcaciones_retornando[df_embarcaciones_retornando['Latitud'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_retornando[df_embarcaciones_retornando['Latitud'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_retorno_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_retorno_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'asignar'
        df_temp_errores['tipo_error'] = 'Latitud null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_retornando = df_embarcaciones_retornando[df_embarcaciones_retornando['Latitud'].notnull()]
    
        # errores Longitud null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_retorno_errores = df_embarcaciones_retornando[df_embarcaciones_retornando['Longitud'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_retornando[df_embarcaciones_retornando['Longitud'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_retorno_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_retorno_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'asignar'
        df_temp_errores['tipo_error'] = 'Longitud null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_retornando = df_embarcaciones_retornando[df_embarcaciones_retornando['Longitud'].notnull()]
    
        # errores first_cala_start_date null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_retorno_errores = df_embarcaciones_retornando[df_embarcaciones_retornando['first_cala_start_date'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_retornando[df_embarcaciones_retornando['first_cala_start_date'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_retorno_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_retorno_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'asignar'
        df_temp_errores['tipo_error'] = 'first_cala_start_date null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_retornando = df_embarcaciones_retornando[df_embarcaciones_retornando['first_cala_start_date'].notnull()]
    
        # errores horas desde primera cala mayor a 72
        df_embarcaciones_retornando['horas_desde_primera_cala'] = (timestamp - df_embarcaciones_retornando['first_cala_start_date'])/ np.timedelta64(1, 'h')
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_retorno_errores = df_embarcaciones_retornando[df_embarcaciones_retornando['horas_desde_primera_cala']>=72].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_retornando[df_embarcaciones_retornando['horas_desde_primera_cala']>=72]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_retorno_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_retorno_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'asignar'
        df_temp_errores['tipo_error'] = 'horas desde primera cala mayor a 72'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_retornando = df_embarcaciones_retornando[df_embarcaciones_retornando['horas_desde_primera_cala']<72]
    
        df_embarcaciones_retornando = df_embarcaciones_retornando.reset_index(drop=True)
    
        ## barcos terceros retornando
        emb_errores_terceros = []
        df_embarcaciones_terceros = df_embarcaciones[(df_embarcaciones['marea_status']=='RETORNANDO A PUERTO')&(df_embarcaciones['owner_group']=='T')]
    
        ## errores eta null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_terceros_errores = df_embarcaciones_terceros[df_embarcaciones_terceros['eta'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_terceros[df_embarcaciones_terceros['eta'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_terceros_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_terceros_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'terceros retornando'
        df_temp_errores['tipo_error'] = 'eta null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_terceros = df_embarcaciones_terceros[df_embarcaciones_terceros['eta'].notnull()]
    
        ## errores eta_plant null
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_terceros_errores = df_embarcaciones_terceros[df_embarcaciones_terceros['eta_plant'].isnull()].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_terceros[df_embarcaciones_terceros['eta_plant'].isnull()]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_terceros_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_terceros_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'terceros retornando'
        df_temp_errores['tipo_error'] = 'eta_plant null'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_terceros = df_embarcaciones_terceros[df_embarcaciones_terceros['eta_plant'].notnull()]
    
        # errores de tiempos negativos o mas de 0.5 horas
        df_embarcaciones_terceros['delta_tiempo'] = (df_embarcaciones_terceros['eta'] - timestamp) / np.timedelta64(1, 'h')
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_terceros_errores = df_embarcaciones_terceros[df_embarcaciones_terceros['delta_tiempo']<0].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_terceros[df_embarcaciones_terceros['delta_tiempo']<0]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_terceros_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_terceros_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'terceros retornando'
        df_temp_errores['tipo_error'] = 'tiempo negativo'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_terceros = df_embarcaciones_terceros[df_embarcaciones_terceros['delta_tiempo']>=0]
    
        # mas de 0.5 horas
        df_embarcaciones_terceros['delta_tiempo'] = (df_embarcaciones_terceros['eta'] - timestamp) / np.timedelta64(1, 'h')
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_terceros_errores = df_embarcaciones_terceros[df_embarcaciones_terceros['delta_tiempo']>12].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_terceros[df_embarcaciones_terceros['delta_tiempo']>0.5]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_terceros_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_terceros_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'terceros retornando'
        df_temp_errores['tipo_error'] = 'mas de 0.5 horas'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_terceros = df_embarcaciones_terceros[df_embarcaciones_terceros['delta_tiempo']<=12]
    
        # errores toneladas 0 o menor a 0
        df_temp_errores = pd.DataFrame(columns=['embarcaciones','tipo_emb','tipo_error','timestamp'])
        bd_terceros_errores = df_embarcaciones_terceros[df_embarcaciones_terceros['declared_ton']<=0].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_terceros[df_embarcaciones_terceros['declared_ton']<=0]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_terceros_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_terceros_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'terceros retornando'
        df_temp_errores['tipo_error'] = 'toneladas 0 o menor a 0'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_terceros = df_embarcaciones_terceros[df_embarcaciones_terceros['declared_ton']>0]
    
        df_embarcaciones_terceros['eta_plant'].replace('CHICAMA', 'MALABRIGO',inplace=True)
        df_embarcaciones_terceros['eta_plant'].replace('MOLLENDO', 'MATARANI',inplace=True)
        df_embarcaciones_terceros = df_embarcaciones_terceros.reset_index(drop=True)

        # Errores eta_plant - planta con requerimiento 0
        df_temp_errores = pd.DataFrame(columns=['marea_id','embarcaciones','tipo_emb','tipo_error','timestamp'])
        plants_habilitadas = df_pozas_estado['id_planta'].unique().tolist()
        bd_retorno_errores = df_embarcaciones_retornando[~df_embarcaciones_retornando['eta_plant'].isin(plants_habilitadas)].copy()
        # df_temp_errores['embarcaciones'] = list(df_embarcaciones_retornando[df_embarcaciones_retornando['horas_desde_primera_cala']>=72]['boat_name'].unique())
        df_temp_errores['marea_id'] = bd_retorno_errores['marea_id']
        df_temp_errores['embarcaciones'] = bd_retorno_errores['boat_name']
        df_temp_errores['tipo_emb'] = 'asignar'
        df_temp_errores['tipo_error'] = 'eta plant no habilitada'
        df_temp_errores['timestamp'] = timestamp
        # df_embarcaciones_errores = df_embarcaciones_errores.append(df_temp_errores)
        df_embarcaciones_errores = pd.concat([df_embarcaciones_errores, df_temp_errores], ignore_index=True)
        df_embarcaciones_retornando = df_embarcaciones_retornando[df_embarcaciones_retornando['eta_plant'].isin(plants_habilitadas)]
    
        df_embarcaciones_retornando = df_embarcaciones_retornando.reset_index(drop=True)
    
        return df_embarcaciones_descargando, df_embarcaciones_esperando_descarga, df_embarcaciones_retornando, df_embarcaciones_terceros, df_embarcaciones_errores




    def dividir_lineas_plantas(df_pozas_estado):
        ## transformar data de pozas a lineas
        df_pozas_estado['lineaname'] = df_pozas_estado['id_planta'] + '-' + df_pozas_estado['id_chata'] + '-' + df_pozas_estado['id_linea']
        df_pozas_estado['pozaname'] = df_pozas_estado['id_planta'] + '-' + df_pozas_estado['poza_number'].map(str)
        df_lineas_estado = df_pozas_estado.groupby(['lineaname']).agg({'id_planta':'first',
                                                'id_chata':'first',
                                                'id_linea':'first',
                                                'coordslatitude':'first',
                                                'coordslongitude':'first',
                                                'pozaCapacity':'sum',
                                                'nivel':'sum'}).reset_index(drop=True)


        ## transformar data de pozas a planta
        df_plantas_estado = df_pozas_estado.groupby(['id_planta']).agg({'coordslatitude':'first',
                                                    'coordslongitude':'first',
                                                    'pozaCapacity':'sum',
                                                    'nivel':'sum'}).reset_index()
        return df_pozas_estado, df_lineas_estado, df_plantas_estado

    def predecir_tvn(df):
        llenado = df['PORC_CAPACIDAD']
        ton = df['declared_ton']
        tdc = (df['discharge_start_date'] - df['first_cala_start_date']) / np.timedelta64(1, 'h')
        df['tipo_bodega'] = np.where(df['frio_system_state']=='GF','Golpe Frio',df['tipo_bodega'])
        tipo_bodega = df['tipo_bodega']
        if tipo_bodega == 'Estanca':
            lista_bodega = [1,0,0]
        elif tipo_bodega == 'Frio':
            lista_bodega = [0,1,0]
        elif tipo_bodega == 'Golpe Frio':
            lista_bodega = [0,2,0]
        else:
            lista_bodega = [0,0,1]
        lista = [[llenado,ton,tdc,np.power(tdc,2)] + lista_bodega]
        df_tdc=pd.DataFrame(lista,columns=['% Llenado', 'Descarg.', 'TDC-Desc', 'TDC_2', 'Bodega_Estanca', 'Bodega_Frio', 'Bodega_Tradicional'])
        return get_return_model_polynomial(df_tdc)[0]

    # funcion para calcular cuando terminaran de descargar las embarcaciones que estan descargando
    def fin_descarga_temp(x,df_tiempo_descarga,timestamp):
        if x['marea_status']=='DESCARGANDO':
            ton = float(x['declared_ton'])
            chata = str(x['discharge_chata_name'])
            linea = str(x['discharge_line_name'])
        else:
            ton = float(x['toneladas'])
            chata = str(x['id_chata'])
            linea = str(x['id_linea'])
        if ton < 100: # antes id_chata
            vel_encontrada = df_tiempo_descarga[(df_tiempo_descarga['name']==chata)&(df_tiempo_descarga['id_linea']==linea)]['velocidad_0_100_tons']
            if len(vel_encontrada) > 0:
                vel = float(vel_encontrada)
            else:
                vel = 90.0
        elif ton < 300: # antes id_chata
            vel_encontrada = df_tiempo_descarga[(df_tiempo_descarga['name']==chata)&(df_tiempo_descarga['id_linea']==linea)]['velocidad_100_300_tons']
            if len(vel_encontrada) > 0:
                vel = float(vel_encontrada)
            else:
                vel = 100.0
        else: # antes id_chata
            vel_encontrada = df_tiempo_descarga[(df_tiempo_descarga['name']==chata)&(df_tiempo_descarga['id_linea']==linea)]['velocidad_300_mas_tons']
            if len(vel_encontrada) > 0:
                vel = float(vel_encontrada)
            else:
                vel = 120.0
        horas = ton/vel
        ## tiempo cuando empezo + cuanto se demorara el descargar
        ## print(x['marea_status'])
        if x['marea_status']=='DESCARGANDO':
            tiempo = x['discharge_start_date'] + datetime.timedelta(hours=horas)
            ## si tiempo es mayor que ahora
            if tiempo > timestamp:
                return tiempo
            else:
                return timestamp
        else:
            tiempo = x['tiempo_fin_descarga'] + datetime.timedelta(hours=horas)
            return tiempo

    ## calcular cuando terminara descarga y tvn de la embarcacion
    def opt_emb_descargando(df_embarcaciones_descargando, df_pozas_estado,df_tiempo_descarga,timestamp,df_pozas_estado_leido):

        df_embarcaciones_descargando['tiempo_fin_descarga'] = df_embarcaciones_descargando.apply(fin_descarga_temp,df_tiempo_descarga=df_tiempo_descarga,timestamp=timestamp, axis=1)
        mask = df_embarcaciones_descargando['termino_succion'].notna()
        df_embarcaciones_descargando.loc[mask, 'tiempo_fin_descarga'] = pd.to_datetime(df_embarcaciones_descargando.loc[mask, 'termino_succion']).copy()
        df_embarcaciones_descargando['tvn_emb_descargando'] = df_embarcaciones_descargando.apply(predecir_tvn, axis=1)

        # df_embarcaciones_descargando.loc[df_embarcaciones_descargando['discharge_poza_1'].notna(), 'discharge_poza_1'] = df_embarcaciones_descargando.loc[df_embarcaciones_descargando['discharge_poza_1'].notna(), 'discharge_poza_1'].map(int) 
        pozas_max = df_pozas_estado.groupby(['id_planta'], as_index=False)[['pozaCapacity', 'poza_number']].max()
        pozas_max.sort_values('id_planta', inplace=True)
        mask = df_embarcaciones_descargando['discharge_poza_1'].isna()
        df_embarcaciones_descargando.loc[mask, 'discharge_poza_1'] = pozas_max.loc[pozas_max['id_planta'].isin(df_embarcaciones_descargando.loc[mask, 'discharge_plant_name'].sort_values().unique().tolist()), 'poza_number'].to_numpy()
        df_embarcaciones_descargando['discharge_poza_1'] = df_embarcaciones_descargando['discharge_poza_1'].map(int) 

        # en el dataframe de pozas y poner los valores de tvn y nivel de la poza
        df_embarcaciones_descargando['id_poza'] = df_embarcaciones_descargando['discharge_plant_name'].map(str) + '-' + df_embarcaciones_descargando['discharge_poza_1'].map(str)
        # df_pozas_estado = pd.merge(df_pozas_estado, df_embarcaciones_descargando.groupby(['id_poza']).agg({'discharge_plant_name':'min','discharge_poza_1':'min','declared_ton':'sum','tvn_emb_descargando':'mean'}).reset_index(),  how='left', left_on=['id_planta','poza_number'], right_on = ['discharge_plant_name','discharge_poza_1'])
        
        # Obtener tm inicial
        df_pozas_res_inicial = df_pozas_estado.groupby(['id_planta','poza_number']).agg(TM=('nivel','max'))
        df_pozas_res_inicial.reset_index(inplace=True)
        df_pozas_res_inicial['cod_id_poza'] = df_pozas_res_inicial['id_planta'] + '-' + df_pozas_res_inicial['poza_number'].map(int).map(str)
        
        # Cruzar mareas descargando que estén asociadas a pozas sin stock (se debe actualizar el stock y marcar tipo de descarga)
        lista_desc = list(df_embarcaciones_descargando.marea_id)
        df_pozas_estado_leido['discharge_start_date'] = pd.to_datetime(df_pozas_estado_leido['discharge_start_date'])
        df_pozas_estado_leido['discharge_end_date'] = pd.to_datetime(df_pozas_estado_leido['discharge_end_date'])
        df_ajuste = df_pozas_estado_leido[(df_pozas_estado_leido['marea_id'].isin(lista_desc)) & (df_pozas_estado_leido['stock_actual']==0)]
        lista_desc_review = list(df_ajuste.marea_id.unique())
        df_pozas_asig = df_pozas_estado_leido[df_pozas_estado_leido['marea_id'].isin(lista_desc_review)]
        df_pozas_asig = df_pozas_asig.sort_values(['marea_id','discharge_start_date','pozaNumber'])
        
        # Mareas descargando que no tienen registrado una poza asignada (se considero la ultima recomendacion)
        df_ajuste_res = df_embarcaciones_descargando[~df_embarcaciones_descargando['marea_id'].isin(df_ajuste['marea_id'])].reset_index(drop=True)
        df_ajuste_res['tipo_conservacion'] = np.where((df_ajuste_res['tipo_bodega']=='Frio') & ((df_ajuste_res['frio_system_state']=='RC') | (df_ajuste_res['frio_system_state']=='GF')),'Frio','Tradicional')
        # Obtener tm a agregar
        df_ajuste_res = df_ajuste_res.groupby(['discharge_plant_name','discharge_poza_1','tipo_conservacion']).agg(TM=('declared_ton','sum'))
        df_ajuste_res.reset_index(inplace=True)
        df_ajuste_res['cod_id_poza'] = df_ajuste_res['discharge_plant_name'] + '-' + df_ajuste_res['discharge_poza_1'].map(int).map(str)
        
        # Podria haber un problema cuando hay mas de 1 poza sin inicio de descarga
        df_ajuste = df_ajuste.sort_values(['marea_id','discharge_start_date','pozaNumber']).reset_index(drop=True)
        # Agregar tipo de conservacion
        df_ajuste['tipo_conservacion'] = np.where((df_ajuste['tipo_bodega']=='Frio') & ((df_ajuste['frio_system_state']=='RC') | (df_ajuste['frio_system_state']=='GF')),'Frio','Tradicional')
        df_embarcaciones_descargando['tipo_conservacion'] = np.where((df_embarcaciones_descargando['tipo_bodega']=='Frio') & ((df_embarcaciones_descargando['frio_system_state']=='RC') | (df_embarcaciones_descargando['frio_system_state']=='GF')),'Frio','Tradicional')
        df_pozas_estado['cap_disponible'] = df_pozas_estado['pozaCapacity'] - df_pozas_estado['nivel']
        
        # Para el resto de pozas
        for index,row in df_ajuste.iterrows():
            poza_ajuste = row['pozaNumber']
            df_mareas_ajuste = df_embarcaciones_descargando[df_embarcaciones_descargando['marea_id']==row['marea_id']]
            tipo_conservacion_emb = df_mareas_ajuste['tipo_conservacion'].unique().item()
            
            # Para la primera poza
            # Se considera que la poza asignada nunca tendra registrada la descarga de la poza
            # Datos primera poza asignada
            mask_poza1 = (df_pozas_estado['poza_number']==df_mareas_ajuste['discharge_poza_1'].item()) & (df_pozas_estado['id_planta']==df_mareas_ajuste['discharge_plant_name'].item())
            
            # Para cuando hay mareas con pozas asignadas que en este momento estan deshabilitadas, skip a la siguiente iteracion
            try:
                cap_poza_1 = df_pozas_estado.loc[mask_poza1,'cap_disponible'].unique().item()
            except:
                continue
            tm_asignado1 = min(df_mareas_ajuste['declared_ton'].item(), cap_poza_1)
            tm_restante1 = df_mareas_ajuste['declared_ton'].item() - tm_asignado1
                
            # Actualizar stock
            mask_marea = (df_pozas_estado['poza_number']==df_mareas_ajuste['discharge_poza_1'].item()) & (df_pozas_estado['id_planta']==df_mareas_ajuste['discharge_plant_name'].item())
            df_pozas_estado.loc[mask_marea,'nivel'] = tm_asignado1 + df_pozas_estado.loc[mask_marea,'nivel']
            
            # Actualizar tipo de descarga, si stock es 0
            stock_actual = df_pozas_estado.loc[mask_marea,'nivel'].unique().item()
            tipo_conservacion_actual = df_pozas_estado.loc[mask_marea,'tipo_conservacion'].unique().item()
            # sistema_frio_actual = df_pozas_estado.loc[mask_marea,'frio_system_state'].unique().item()
            df_pozas_estado.loc[mask_marea,'tipo_conservacion'] = np.where(stock_actual==0, tipo_conservacion_emb, tipo_conservacion_actual)
            
            if df_mareas_ajuste['discharge_poza_2'].item()==poza_ajuste:
                # Asignar la cantidad minima entre la capacidad de la poza y el restante
                tm_asignado2 = min(tm_restante1, row['pozaCapacity'])
                tm_restante2 = tm_restante1 - tm_asignado2
                
                # Actualizar stock
                mask_poza2 = (df_pozas_estado['poza_number']==df_mareas_ajuste['discharge_poza_2'].item()) & (df_pozas_estado['id_planta']==df_mareas_ajuste['discharge_plant_name'].item())
                df_pozas_estado.loc[mask_poza2,'nivel'] = tm_asignado2
                
                # Actualizar tipo de descarga, si stock es 0
                stock_actual_p2 = df_pozas_estado.loc[mask_poza2,'nivel']
                # tipo_conservacion_p2 = df_pozas_estado.loc[mask_poza2,'tipo_conservacion'].unique().item()
                df_pozas_estado.loc[mask_poza2,'tipo_conservacion'] = np.where(stock_actual_p2==0, tipo_conservacion_emb, tipo_conservacion_actual)
                    
            if df_mareas_ajuste['discharge_poza_3'].item()==poza_ajuste:
                # Asignar la cantidad minima entre la capacidad de la poza y el restante
                tm_asignado3 = min(tm_restante2, row['pozaCapacity'])
                    
                # Actualizar stock
                mask_poza3 = (df_pozas_estado['poza_number']==df_mareas_ajuste['discharge_poza_3'].item()) & (df_pozas_estado['id_planta']==df_mareas_ajuste['discharge_plant_name'].item())
                df_pozas_estado.loc[mask_poza3,'stock_actual'] = tm_asignado3
                
                # Actualizar tipo de descarga, si stock es 0
                stock_actual_p3 = df_pozas_estado.loc[mask_poza3,'nivel'].item()
                # tipo_conservacion_p3 = df_pozas_estado.loc[mask_poza3,'tipo_conservacion'].unique().item()
                df_pozas_estado.loc[mask_poza3,'tipo_conservacion'] = np.where(stock_actual_p3==0, tipo_conservacion_emb, tipo_conservacion_actual)
        
        # Eliminar columna adicional en df_pozas_estado
        del df_pozas_estado['cap_disponible']

        # # actualizar los valores de tvn, nivel de la poza
        # df_pozas_estado.loc[df_pozas_estado['tvn_emb_descargando'].notnull(),'tvn']=((df_pozas_estado['tvn']*df_pozas_estado['nivel'])+(df_pozas_estado['tvn_emb_descargando']*df_pozas_estado['declared_ton']))/(df_pozas_estado['nivel']+df_pozas_estado['declared_ton'])
        # df_pozas_estado.loc[df_pozas_estado['tvn_emb_descargando'].notnull(),'nivel'] = df_pozas_estado['declared_ton'] + df_pozas_estado['nivel']
        # del df_pozas_estado['tvn_emb_descargando']
        # del df_pozas_estado['declared_ton']
        # del df_pozas_estado['discharge_plant_name']

        # actualizar el valor de fin de descarga para toda la linea
        df_embarcaciones_descargando['discharge_chata_name'].replace('TASA CALLAO', 'CHILLON',inplace=True)
        df_embarcaciones_descargando['discharge_chata_name'].replace('EX-ABA','EXABA',inplace=True)
        df_embarcaciones_descargando['discharge_chata_name'] = 'CHATA ' + df_embarcaciones_descargando['discharge_chata_name']
        df_embarcaciones_descargando['id_temp'] = df_embarcaciones_descargando['discharge_plant_name'] + '-' + df_embarcaciones_descargando['discharge_chata_name'].map(str) + '-' + df_embarcaciones_descargando['discharge_line_name'].map(str)
        df_embarcaciones_descargando = df_embarcaciones_descargando.groupby(['id_temp']).agg({'discharge_plant_name':'first','discharge_chata_name':'first','discharge_line_name':'first','tiempo_fin_descarga':'max'}).reset_index(drop=True)
        df_pozas_estado = pd.merge(df_pozas_estado, df_embarcaciones_descargando,  how='left', left_on=['id_planta','id_chata','id_linea'], right_on = ['discharge_plant_name','discharge_chata_name','discharge_line_name'])
        df_pozas_estado['tiempo_fin_descarga'] = df_pozas_estado['tiempo_fin_descarga'].fillna(timestamp)
        df_embarcaciones_descargando['discharge_chata_name'] = df_embarcaciones_descargando['discharge_chata_name'].str[6:]
        del df_pozas_estado['discharge_plant_name']
        del df_pozas_estado['discharge_chata_name']
        del df_pozas_estado['discharge_line_name']
        
        df_pozas_estado_res = df_pozas_estado.groupby(['id_planta','poza_number']).agg(TM=('nivel','max'))
        df_pozas_estado_res.reset_index(inplace=True)
        df_pozas_estado_res['cod_id_poza'] = df_pozas_estado_res['id_planta'] + '-' + df_pozas_estado_res['poza_number'].map(int).map(str)
        df_pozas_aux = df_pozas_estado[['id','tipo_conservacion']].drop_duplicates()
        df_pozas_aux['tipo_conserv'] = df_pozas_aux.groupby(['id']).tipo_conservacion.transform('first')
        del df_pozas_aux['tipo_conservacion']
        df_pozas_aux.rename(columns={'tipo_conserv':'tipo_conservacion'}, inplace=True)
        df_pozas_estado_res = df_pozas_estado_res.merge(df_pozas_aux[['id','tipo_conservacion']], how='left', left_on='cod_id_poza', right_on='id')
        del df_pozas_estado_res['id']
        df_ajuste_res = df_ajuste_res.merge(df_pozas_res_inicial[['cod_id_poza','TM']], how='left', on='cod_id_poza')
        df_ajuste_res = df_ajuste_res.merge(df_pozas_estado[['id','pozaCapacity']], how='left', left_on='cod_id_poza', right_on='id')
        df_ajuste_res['TM_final'] = df_ajuste_res['TM_x'] + df_ajuste_res['TM_y']    
        df_ajuste_res = df_ajuste_res.drop_duplicates()
        # Agregar lo que ya estaba asignado por match en mareas en df pozas estado
        df_ajuste['cod_id_poza'] = df_ajuste['id_planta'] + '-' + df_ajuste['pozaNumber'].map(int).map(str)
        df_ajuste_group = df_ajuste.groupby(['cod_id_poza']).agg(TM_match=('declared_ton','sum'))
        df_ajuste_group.reset_index(inplace=True)
        df_ajuste_res = df_ajuste_res.merge(df_ajuste_group, how='left', on='cod_id_poza')
        df_ajuste_res['TM_match'] = df_ajuste_res['TM_match'].fillna(0)
        df_ajuste_res['TM_final'] = df_ajuste_res['TM_final'] + df_ajuste_res['TM_match']
        df_ajuste_res['TM_final'] = np.where(df_ajuste_res['TM_final']>df_ajuste_res['pozaCapacity'], df_ajuste_res['pozaCapacity'], df_ajuste_res['TM_final'])
        df_pozas_estado_res = df_pozas_estado_res.merge(df_ajuste_res[['cod_id_poza','TM_final','tipo_conservacion']], how='left', on='cod_id_poza')
        df_pozas_estado_res['TM_final'] = np.where(~df_pozas_estado_res['TM_final'].notnull(), df_pozas_estado_res['TM'], df_pozas_estado_res['TM_final'])
        df_pozas_estado_res['tipo_conservacion_x'] =  np.where(~df_pozas_estado_res['tipo_conservacion_x'].isin(['Tradicional','Frio']), df_pozas_estado_res['tipo_conservacion_y'], df_pozas_estado_res['tipo_conservacion_x'])
        df_pozas_estado = df_pozas_estado.merge(df_pozas_estado_res[['cod_id_poza','TM_final','tipo_conservacion_x']], how='left', left_on='id', right_on='cod_id_poza')
        df_pozas_estado['nivel'] = np.where(df_pozas_estado['nivel']<df_pozas_estado['TM_final'], df_pozas_estado['TM_final'], df_pozas_estado['nivel'])
        df_pozas_estado['tipo_conservacion'] = np.where(~df_pozas_estado['tipo_conservacion'].isin(['Tradicional','Frio']), df_pozas_estado['tipo_conservacion_x'], df_pozas_estado['tipo_conservacion'])
        del df_pozas_estado['TM_final']
        del df_pozas_estado['cod_id_poza']
        del df_pozas_estado_res['tipo_conservacion_x']
            
        return df_pozas_estado

    def distancia(lat_1, lng_1, lat_2, lng_2):
        """
        The function takes two points on the Earth's surface, and returns the distance between them in
        kilometers
        
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
        x = (lon2 - lon1) * cos( 0.5*(lat2+lat1))
        y = lat2 - lat1
        d = R * sqrt( x*x + y*y )
        return d

    def limpiar_recomendacion_planta(df_sol_opt, df_pozas_estado):
        df_plantas = df_pozas_estado[['id_planta','poza_number','pozaname','pozaCapacity','nivel','coordslatitude','coordslongitude']].groupby('pozaname').head(1)
        df_plantas['cap_usar'] = df_plantas['pozaCapacity'] - df_plantas['nivel']
        df_plantas = df_plantas.groupby('id_planta').agg({'coordslatitude':'first','coordslongitude':'first','cap_usar':'sum'}).reset_index()
        df_plantas = df_plantas[df_plantas['cap_usar']>0].reset_index(drop=True)
        for index, row in df_sol_opt.iterrows():
            tipo = row['tipo']
            if tipo == 'asignar':
                lat = row['Latitud']
                lon = row['Longitud']
                planta_recomendacion = row['id_planta']
                marea = row['marea_id']
                temp1 = pd.DataFrame(columns=['planta','distancia'])
                cont = 0
                for index, row in df_plantas.iterrows():
                    lat1 = row['coordslatitude']
                    lon1 = row['coordslongitude']
                    planta = row['id_planta']
                    temp1.loc[cont] = [planta,distancia(lat1, lon1, lat, lon)]
                    cont = cont + 1
                temp1 = temp1.sort_values('distancia').reset_index(drop=True)
                lista = list(temp1[:2]['planta'])
                if planta_recomendacion not in lista:
                    df_sol_opt.loc[df_sol_opt['marea_id']==marea,'id_planta'] = lista[0]
                    chata =  df_pozas_estado[(df_pozas_estado['id_planta']==lista[0])].groupby('id_chata').sum().sort_values('cap_usar').reset_index()['id_chata'][0]
                    df_sol_opt.loc[df_sol_opt['marea_id']==marea,'id_chata'] = chata
                    df_sol_opt.loc[df_sol_opt['marea_id']==marea,'id_linea'] = np.nan
        return df_sol_opt

    def limpiar_eta(df_sol_opt,df_embarcaciones,df_pozas_estado_original,df_chatas_lineas):
        df_sol_opt2=df_sol_opt.copy()
        df_sol_opt3=df_sol_opt.copy()
        df_sol_opt3['chatalinea']=df_sol_opt3['chata_descarga']+'-'+df_sol_opt3['linea_descarga']
        df_pozas_estado_original['chatalinea']=df_pozas_estado_original['id_chata']+'-'+df_pozas_estado_original['id_linea']
        df_sol_opt2 = pd.merge(df_sol_opt2,df_embarcaciones[['marea_id','eta','eta_plant', 'marea_status']],how='left',on='marea_id').reset_index(drop=True).copy()
        df_sol_opt2['chatalinea']=df_sol_opt2['chata_descarga']+'-'+df_sol_opt2['linea_descarga']
        df_sol_opt2=df_sol_opt2[df_sol_opt2.velocidad_retorno.notnull()].copy()
        df_sol_opt2.sort_values(by=['planta_retorno','eta'],inplace=True)
        df_sol_opt2.reset_index(inplace=True, drop=True)
        #lineas_ocupadas_recomendadas=list(df_sol_opt2[(df_sol_opt2.planta_retorno==df_sol_opt2.eta_plant)&(df_sol_opt2.velocidad_retorno.notnull())].chatalinea)
        embarcaciones_cambio=[]
        for index,row in df_sol_opt2.iterrows():
            if pd.notnull(row['eta']) & pd.notnull(row['eta_plant']) & (row['marea_status'] in ['RETORNANDO A PUERTO', 'PESCANDO']):
                row['orden_descarga_global'] = np.nan
                if (row['eta_plant']!=row['planta_retorno']) & (not pd.notnull(row['orden_descarga_global'])):
                    try:
                        chatalineaanterior=list(df_sol_opt2[(df_sol_opt2.eta_plant==row['eta_plant'])&(df_sol_opt2.index==index-1)].chatalinea.values[0])
                    except:
                        chatalineaanterior=[]
                    chatalineasposibles=pd.DataFrame(df_chatas_lineas[(df_chatas_lineas.id_planta==row['eta_plant']) & (df_chatas_lineas.habilitado==True)].chatalinea.value_counts().reset_index())
                    chatalineasposibles.columns=['chatalinea','index']
                    valuecounts=pd.DataFrame(df_sol_opt3[(df_sol_opt3.planta_retorno==row['eta_plant'])].chatalinea.value_counts().reset_index())
                    valuecounts.columns=['chatalinea','cuenta']
                    posibilidades=chatalineasposibles.merge(valuecounts,how='left',on='chatalinea')
                    if len(chatalineaanterior)>0:
                        posibilidades=posibilidades[posibilidades.chatalinea!=chatalineaanterior[0]]
                    posibilidades=posibilidades.sort_values(by='cuenta').reset_index(drop=True).copy()
                    chatalineaelegida=posibilidades.loc[0,'chatalinea']
                    chata=df_chatas_lineas[df_chatas_lineas.chatalinea==chatalineaelegida].id_chata.values[0]
                    linea=df_chatas_lineas[df_chatas_lineas.chatalinea==chatalineaelegida].id_linea.values[0]


                    #chata_linea_libres=df_pozas_estado_original[(df_pozas_estado_original.id_planta==row['eta_plant'])&
                    #                                            ~(df_pozas_estado_original.chatalinea.isin(lineas_ocupadas_recomendadas))].copy()
                    #df_sol_opt2.loc[index,'planta_retorno'] = row['eta_plant']
                    #if len(chata_linea_libres)>0:
                    #    chata=chata_linea_libres.sort_values(by='tiempo_fin_descarga').reset_index(drop=True).iloc[0].id_chata
                    #    linea=chata_linea_libres.sort_values(by='tiempo_fin_descarga').reset_index(drop=True).iloc[0].id_linea
                    #    lineas_ocupadas_recomendadas.append(chata+'-'+linea)
                    #else:    
                    #    chata =  df_pozas_estado_original[(df_pozas_estado_original['id_planta']==row['eta_plant'])].groupby(by=['id_chata','id_linea']).sum().sort_values('cap_usar').reset_index()['id_chata'][0]
                    #    linea=df_pozas_estado_original[(df_pozas_estado_original['id_planta']==row['eta_plant'])].groupby(by=['id_chata','id_linea']).sum().sort_values('cap_usar').reset_index()['id_linea'][0]
                    #eta=df_sol_opt.loc[index,'eta']
                    #nuevo_orden=df_sol_opt[(df_sol_opt.chata_descarga==chata)&(df_sol_opt.linea_descarga==linea)&(df_sol_opt.eta<=eta)].orden_descarga.max()+1
                    #df_sol_opt.loc[(df_sol_opt.chata_descarga==chata)&(df_sol_opt.linea_descarga==linea)&(df_sol_opt.orden_descarga>=nuevo_orden),'orden_descarga']=df_sol_opt.loc[(df_sol_opt.chata_descarga==chata)&(df_sol_opt.linea_descarga==linea)&(df_sol_opt.orden_descarga>=nuevo_orden),'orden_descarga']+1
                    #df_sol_opt.loc[index,'orden_descarga']=nuevo_orden
                    marea=row.marea_id
                    df_sol_opt3.loc[df_sol_opt3.marea_id==marea,'chata_descarga'] = chata
                    df_sol_opt3.loc[df_sol_opt3.marea_id==marea,'linea_descarga'] = linea
                    df_sol_opt3.loc[df_sol_opt3.marea_id==marea,'chatalinea']=chata+'-'+linea
                    #embarcaciones_cambio.append(row.marea_id)
        try:
            del df_sol_opt3['eta']
        except:
            pass

        try:
            del df_sol_opt3['eta_plant']
        except:
            pass

        try:
            del df_sol_opt3['chatalinea']
        except:
            pass

        return df_sol_opt3

    def distancia_velocidad(mareaid,planta,df_embarcaciones,df_pozas_estado):
        lat_emb = float(df_embarcaciones[df_embarcaciones['marea_id']==mareaid]['Latitud'])
        lon_emb = float(df_embarcaciones[df_embarcaciones['marea_id']==mareaid]['Longitud'])
        lat_pla = float(df_pozas_estado[df_pozas_estado['id_planta']==planta].reset_index(drop=True)['coordslatitude'][0])
        lon_pla = float(df_pozas_estado[df_pozas_estado['id_planta']==planta].reset_index(drop=True)['coordslongitude'][0])
        return distancia(lat_emb, lon_emb, lat_pla, lon_pla)

    def limpiar_velocidades_max(df_sol_opt,df_pozas_estado_vel,timestamp,df_embarcaciones):
        for index, row in df_sol_opt[df_sol_opt['velocidad_retorno'].notnull()].iterrows():
            marea = row['marea_id']
            planta = row['planta_retorno']
            chata = row['chata_descarga']
            hora_fin_chata = df_pozas_estado_vel[(df_pozas_estado_vel['id_planta']==planta)&(df_pozas_estado_vel['id_chata']==chata)]['tiempo_fin_descarga'].max() + datetime.timedelta(hours=0.5)
            dist = distancia_velocidad(marea,planta)
            velocidad_emb = float(df_embarcaciones[df_embarcaciones['marea_id']==marea]['SPEED_MAX_km'])
            tiempo_extra = dist / velocidad_emb
            tiempo_llegada = timestamp + datetime.timedelta(hours=tiempo_extra)
            if tiempo_llegada>hora_fin_chata:
                df_sol_opt.at[index, 'velocidad_retorno'] = 'MAX'
            else:
                df_sol_opt.at[index, 'velocidad_retorno'] = 'OPT'
        return df_sol_opt

    def get_current_tdc_from_first_cala():
        env_argument_param = sys.argv[1]
        if env_argument_param == 'dumped':
            date_now = datetime.datetime.strptime(os.getenv("DUMPED_TIME_REFERENCE_FOR_TDC"), '%d/%m/%y %H:%M:%S')
        else:
            date_now = datetime.datetime.utcnow()
        return date_now


    # def get_return_model_polynomial_value(tdc,bodega_frio):
    #     limite_tdc_frio = 22
    #     limite_tdc_trad = 20
    #     slope_frio = 0.2
    #     slope_trad = 0.9
    #     lista_tvn = []
    #     if (bodega_frio == 1) & (tdc>=limite_tdc_frio):
    #         tvn_inicial = 1.46184258e-05*limite_tdc_frio**5 -1.38737645e-03*limite_tdc_frio**4 + 4.92813314e-02*limite_tdc_frio**3 -8.12249406e-01*limite_tdc_frio**2 + 6.34283538e+00*limite_tdc_frio
    #         tdc_adicional = tdc - limite_tdc_frio
    #         return tvn_inicial + tdc_adicional*slope_frio
    #     elif (bodega_frio == 0) & (tdc>=limite_tdc_trad):
    #         tvn_inicial = 1.46184258e-05* limite_tdc_trad**5 -1.38737645e-03*limite_tdc_trad**4 + 4.92813314e-02*limite_tdc_trad**3 -8.12249406e-01*limite_tdc_trad**2 + 6.34283538e+00*limite_tdc_trad
    #         tdc_adicional = tdc - limite_tdc_trad
    #         return tvn_inicial + tdc_adicional*slope_trad
    #     elif (bodega_frio == 2) & (tdc>=limite_tdc_frio):
    #         tvn_inicial = 1.46184258e-05*limite_tdc_frio**5 -1.38737645e-03*limite_tdc_frio**4 + 4.92813314e-02*limite_tdc_frio**3 -8.12249406e-01*limite_tdc_frio**2 + 6.34283538e+00*limite_tdc_frio
    #         tdc_adicional = tdc - limite_tdc_frio
    #         return tvn_inicial + tdc_adicional*slope_frio
    #     else:
    #         return (1.46184258e-05* tdc**5 -1.38737645e-03*tdc**4 + 4.92813314e-02*tdc**3 -8.12249406e-01*tdc**2 + 6.34283538e+00*tdc)*bodega_frio+(1.46184258e-05* tdc**5 -1.38737645e-03*tdc**4 + 4.92813314e-02*tdc**3 -8.12249406e-01*tdc**2 + 6.34283538e+00*tdc)*(1-bodega_frio)

    def get_return_model_polynomial_value(tdc, bodega_frio):
        if (bodega_frio == 1):
            return estimate_tvn_con_frio(tdc)
        elif (bodega_frio == 0):
            return estimate_tvn_sin_frio(tdc)
        elif (bodega_frio == 2):
            return estimate_tvn_con_frio(tdc)
        else:
            return np.nan

    def fin_descarga_temp_value(ton,chata,linea,df_tiempo_descarga):
        if ton < 100: # antes id_chata
            vel_encontrada = df_tiempo_descarga[(df_tiempo_descarga['id_chata']==chata)&(df_tiempo_descarga['id_linea']==linea)]['velocidad_0_100_tons']
            if len(vel_encontrada) > 0:
                vel = float(vel_encontrada)
            else:
                vel = 90.0
        elif ton < 300: # antes id_chata
            vel_encontrada = df_tiempo_descarga[(df_tiempo_descarga['id_chata']==chata)&(df_tiempo_descarga['id_linea']==linea)]['velocidad_100_300_tons']
            if len(vel_encontrada) > 0:
                vel = float(vel_encontrada)
            else:
                vel = 100.0
        else: # antes id_chata
            vel_encontrada = df_tiempo_descarga[(df_tiempo_descarga['id_chata']==chata)&(df_tiempo_descarga['id_linea']==linea)]['velocidad_300_mas_tons']
            if len(vel_encontrada) > 0:
                vel = float(vel_encontrada)
            else:
                vel = 120.0
        horas = ton/vel
        return horas

    def limpiar_velocidades_max(df_sol_opt,df_pozas_estado_vel,timestamp,df_embarcaciones):
        """
        It takes a dataframe, and for each row, it calculates the distance between two points, and then
        compares that distance to the maximum speed of a boat. If the distance is greater than the
        maximum speed, it returns "MAX", otherwise it returns "OPT"
        
        :param df_sol_opt: a dataframe with the following columns:
        :param df_pozas_estado_vel: a dataframe with the following columns:
        :param timestamp: datetime.datetime(2020, 1, 1, 0, 0)
        :param df_embarcaciones: a dataframe with the following columns:
        :return: A dataframe with the following columns:
        """
        temp = df_sol_opt[df_sol_opt['velocidad_retorno'].notnull()]
        lista = []
        for index,row in temp.iterrows():
            lista.append(distancia_velocidad(row['marea_id'],row['planta_retorno']))
        temp['distancia'] = lista
        temp = temp.sort_values(['chata_descarga','distancia'])
        temp['count'] = temp.groupby(['chata_descarga']).cumcount()
        for index, row in temp[(temp['count']==0)|(temp['count']==1)].iterrows():
            marea = row['marea_id']
            planta = row['planta_retorno']
            chata = row['chata_descarga']
            hora_fin_chata = df_pozas_estado_vel[(df_pozas_estado_vel['id_planta']==planta)&(df_pozas_estado_vel['id_chata']==chata)]['tiempo_fin_descarga'].max() + datetime.timedelta(hours=0.5)
            dist = distancia_velocidad(marea,planta)
            velocidad_emb = float(df_embarcaciones[df_embarcaciones['marea_id']==marea]['SPEED_MAX_km'])
            tiempo_extra = dist / velocidad_emb
            tiempo_llegada = timestamp + datetime.timedelta(hours=tiempo_extra)
            if tiempo_llegada>hora_fin_chata:
                df_sol_opt.at[index, 'velocidad_retorno'] = 'MAX'
            else:
                df_sol_opt.at[index, 'velocidad_retorno'] = 'OPT'
        return df_sol_opt

        
    def distancia(lat_1, lng_1, lat_2, lng_2):
        lon2 = radians(lng_2)
        lat2 = radians(lat_2)
        lon1 = radians(lng_1)
        lat1 = radians(lat_1)
        R = 6371
        x = (lon2 - lon1) * cos( 0.5*(lat2+lat1))
        y = lat2 - lat1
        d = R * sqrt( x*x + y*y )
        return d

    def distancia_a_plantas(planta,mareaid,df_embarcaciones,df_pozas_estado):
        lat_emb=float(df_embarcaciones.loc[df_embarcaciones.marea_id==mareaid,'Latitud'].values[0])
        lon_emb=float(df_embarcaciones.loc[df_embarcaciones.marea_id==mareaid,'Longitud'].values[0])
        lat_pla=float(df_pozas_estado.loc[df_pozas_estado.id_planta==planta,'coordslatitude'].values[0])
        lon_pla=float(df_pozas_estado.loc[df_pozas_estado.id_planta==planta,'coordslongitude'].values[0])
        return distancia(lat_emb, lon_emb, lat_pla, lon_pla)

    def tiempo_a_plantas(planta,mareaid,velocidad,df_embarcaciones,df_pozas_estado,timestamp):
        dis=distancia_a_plantas(planta,mareaid,df_embarcaciones,df_pozas_estado)
        t=dis/velocidad
        tiempo=timestamp+datetime.timedelta(hours=t)
        return tiempo

    def calcular_tiempo_descarga(ton,id_chata,id_linea,df_tiempo_descarga):
        if ton<100:
            vel_encontrada=df_tiempo_descarga.loc[(df_tiempo_descarga.id_chata==id_chata)&(df_tiempo_descarga.id_linea==id_linea),'velocidad_0_100_tons']
            if len(vel_encontrada)>0:
                vel=float(vel_encontrada.values[0])
            else:
                vel=90.0
        elif ton<300:
            vel_encontrada=df_tiempo_descarga.loc[(df_tiempo_descarga.id_chata==id_chata)&(df_tiempo_descarga.id_linea==id_linea),'velocidad_100_300_tons']
            if len(vel_encontrada)>0:
                vel=float(vel_encontrada.values[0])
            else:
                vel=100.0
        else:
            vel_encontrada=df_tiempo_descarga.loc[(df_tiempo_descarga.id_chata==id_chata)&(df_tiempo_descarga.id_linea==id_linea),'velocidad_300_mas_tons']
            if len(vel_encontrada)>0:
                vel=float(vel_encontrada.values[0])
            else:
                vel=120.0
        
        tiempo=ton/vel
        return tiempo

    def cerca_tvn_limite(tvn,planta,porc,df_plantas_velocidad_limites):
        lim_sp=df_plantas_velocidad_limites.loc[df_plantas_velocidad_limites['id']==planta,'lim_tvn_cocina_super_prime'].values[0]
        lim_p=df_plantas_velocidad_limites.loc[df_plantas_velocidad_limites['id']==planta,'lim_tvn_cocina_prime'].values[0]
        cerca=0
        if (((tvn-lim_sp) < (lim_sp*porc)) & (tvn>lim_sp)) | (((tvn-lim_p) < (lim_p*porc)) & (tvn>lim_p)):
            cerca=1
        return cerca

    def cross_product(left,right):
        return (left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))
    
    def optimizacion_embarcaciones_retornando_terceros(df_embarcaciones_retornando,df_embarcaciones_terceros, df_pozas_estado,df_embarcaciones,timestamp,df_plantas_velocidad_limites,df_tiempo_descarga,df_requerimiento_plantas,df_lineas_reservada_terceros, df_chatas_lineas,df_planta_velocidad_anterior,df_embarcaciones_esperando_descarga,df_calidades_precio_venta, df_mareas_cerradas):
        #df_final_1=df_embarcaciones_retornando.merge(df_pozas_estado,how='cross')
        division_modelo = True
        
        if division_modelo:
            df_chata_linea = df_chatas_lineas.loc[df_chatas_lineas['habilitado_retorno'] == False, ['id_chata', 'id_linea', 'habilitado_retorno']]
            df_chata_linea['chata-linea'] = df_chata_linea['id_chata'] + '-' + df_chata_linea['id_linea']
            chata_deshabilitadas = df_chata_linea.loc[df_chata_linea['habilitado_retorno'] == False, 'chata-linea'].tolist()
        else:            
            df_chata_linea = df_chatas_lineas.loc[df_chatas_lineas['habilitado'] == False, ['id_chata', 'id_linea', 'habilitado']]
            df_chata_linea['chata-linea'] = df_chata_linea['id_chata'] + '-' + df_chata_linea['id_linea']        
            chata_deshabilitadas = df_chata_linea.loc[df_chata_linea['habilitado'] == False, 'chata-linea'].tolist()
        
        df_pozas_estado['chata-linea'] = df_pozas_estado['id_chata'] + '-' + df_pozas_estado['id_linea']
        # num_plantas_hab = df_pozas_estado['id_planta'].nunique()

        df_embarcaciones_retornando['fish_zone_departure_date']=pd.to_datetime(df_embarcaciones_retornando['fish_zone_departure_date'])
        mask = (df_embarcaciones_retornando['eta_plant'] != df_embarcaciones_retornando['planta_retorno'])
        df_embarcaciones_retornando.loc[mask, 'planta_retorno'] = df_embarcaciones_retornando.loc[mask, 'eta_plant'] 
        # Limpiar ETA Plant de embarcaciones Pescando
        mask_plant = df_embarcaciones_retornando['marea_status'] == 'PESCANDO'
        df_embarcaciones_retornando.loc[mask_plant,'eta_plant'] = np.nan 
        # df_final_1=cross_product(df_embarcaciones_retornando, df_pozas_estado[df_pozas_estado['Planta'].notna()].drop_duplicates(['Planta', 'id_chata']))
        df_final_1=cross_product(df_embarcaciones_retornando, df_pozas_estado[df_pozas_estado['id_planta'].notna()])
        
        mask_deshabilitadas = (df_pozas_estado['chata-linea'].isin(chata_deshabilitadas))
        df_lineas_reservada_terceros['chata-linea'] = df_lineas_reservada_terceros['id_chata'] + '-' + df_lineas_reservada_terceros['id_linea']
        chata_lineas_reservadas = list(df_lineas_reservada_terceros.loc[df_lineas_reservada_terceros['reserv_terc'] == True, 'chata-linea'].unique())
        mask_reservadas = (df_pozas_estado['chata-linea'].isin(chata_lineas_reservadas))
        num_plantas_hab = df_pozas_estado.loc[(~mask_deshabilitadas),'id_planta'].nunique()
        
        #TODO: Validar lineas reservadas para terceros
        df_prev_utilidad = cross_product(df_embarcaciones_retornando[['marea_id', 'boat_name','marea_status', 'Latitud', 'Longitud', 'declared_ton']], df_pozas_estado.loc[(~mask_deshabilitadas), ['id_planta', 'coordslatitude', 'coordslongitude']].drop_duplicates('id_planta').reset_index(drop=True))
        df_prev_utilidad['timestamp'] = timestamp
        distancia_vect = np.vectorize(distancia, otypes=[object])
        df_prev_utilidad['distancia'] = distancia_vect(df_prev_utilidad['Latitud'], df_prev_utilidad['Longitud'], df_prev_utilidad['coordslatitude'], df_prev_utilidad['coordslongitude'])
        df_prev_utilidad['distancia'] = df_prev_utilidad['distancia'].astype(float)
        df_prev_utilidad.sort_values(by=['marea_id', 'distancia'], inplace=True, ascending=[True, False])
        df_prev_utilidad.set_index('id_planta', inplace=True)
        df_prev_utilidad = df_prev_utilidad.groupby('marea_id')['distancia'].nsmallest(3).reset_index()
        
        if num_plantas_hab >= 3:
            df_prev_utilidad['Planta_label'] = np.tile(['PLANTA_1', 'PLANTA_2', 'PLANTA_3'], df_prev_utilidad['marea_id'].nunique()) 
            df_prev_utilidad['distancia_label'] = np.tile(['distancia_1', 'distancia_2', 'distancia_3'], df_prev_utilidad['marea_id'].nunique())
        elif num_plantas_hab == 2:
            df_prev_utilidad['Planta_label'] = np.tile(['PLANTA_1', 'PLANTA_2'], df_prev_utilidad['marea_id'].nunique())
            df_prev_utilidad['distancia_label'] = np.tile(['distancia_1', 'distancia_2'], df_prev_utilidad['marea_id'].nunique())
        else:
            df_prev_utilidad['Planta_label'] = np.tile(['PLANTA_1'], df_prev_utilidad['marea_id'].nunique())
            df_prev_utilidad['distancia_label'] = np.tile(['distancia_1'], df_prev_utilidad['marea_id'].nunique())
        
        df_plant = pd.pivot(df_prev_utilidad, index='marea_id', columns='Planta_label', values='id_planta')
        if num_plantas_hab == 2:
            df_plant['PLANTA_3'] = df_plant['PLANTA_1']
        elif num_plantas_hab == 1:
            df_plant['PLANTA_2'] = df_plant['PLANTA_1']
            df_plant['PLANTA_3'] = df_plant['PLANTA_1']
            
        df_dist = pd.pivot(df_prev_utilidad, index='marea_id', columns='distancia_label', values='distancia')
        if num_plantas_hab == 2:
            df_dist['distancia_3'] = df_dist['distancia_1']
        elif num_plantas_hab == 1:
            df_dist['distancia_2'] = df_dist['distancia_1']
            df_dist['distancia_3'] = df_dist['distancia_1']
        
        df_prev_utilidad = pd.merge(df_plant, df_dist, left_index=True, right_index=True).reset_index()
        df_prev_utilidad = pd.merge(df_prev_utilidad, df_embarcaciones_retornando[['marea_id', 'marea_status', 'gph_opt', 'gph_max', 'speed_opt', 'speed_max', 'SPEED_OPT_km', 'SPEED_MAX_km']], on='marea_id', how='left')
        df_prev_utilidad['combustible_opt_1'] = (df_prev_utilidad['distancia_1'] / (df_prev_utilidad['speed_opt'] * 1.852)) * df_prev_utilidad['gph_opt'] * costo_combustible
        df_prev_utilidad['combustible_opt_2'] = (df_prev_utilidad['distancia_2'] / (df_prev_utilidad['speed_opt'] * 1.852)) * df_prev_utilidad['gph_opt'] * costo_combustible
        df_prev_utilidad['combustible_opt_3'] = (df_prev_utilidad['distancia_3'] / (df_prev_utilidad['speed_opt'] * 1.852)) * df_prev_utilidad['gph_opt'] * costo_combustible

        df_prev_utilidad['combustible_max_1'] = (df_prev_utilidad['distancia_1'] / (df_prev_utilidad['speed_max'] * 1.852)) * df_prev_utilidad['gph_max'] * costo_combustible
        df_prev_utilidad['combustible_max_2'] = (df_prev_utilidad['distancia_2'] / (df_prev_utilidad['speed_max'] * 1.852)) * df_prev_utilidad['gph_max'] * costo_combustible
        df_prev_utilidad['combustible_max_3'] = (df_prev_utilidad['distancia_3'] / (df_prev_utilidad['speed_max'] * 1.852)) * df_prev_utilidad['gph_max'] * costo_combustible

        # Estos valores seran completados en la funcion optimizar ordenes
        df_prev_utilidad['stock_poza_planta_1'] = np.nan
        df_prev_utilidad['stock_poza_planta_2'] = np.nan
        df_prev_utilidad['stock_poza_planta_3'] = np.nan

        df_prev_utilidad['tvn_poza_planta_1'] = np.nan
        df_prev_utilidad['tvn_poza_planta_2'] = np.nan
        df_prev_utilidad['tvn_poza_planta_3'] = np.nan
        
        df_prev_utilidad['valor_marea_1'] = np.nan
        df_prev_utilidad['valor_marea_2'] = np.nan
        df_prev_utilidad['valor_marea_3'] = np.nan
        
        df_prev_utilidad['costo_consumo_comb_1'] = np.nan
        df_prev_utilidad['costo_consumo_comb_2'] = np.nan
        df_prev_utilidad['costo_consumo_comb_3'] = np.nan
        
        df_prev_utilidad['tradeoff_1'] = np.nan
        df_prev_utilidad['tradeoff_2'] = np.nan
        df_prev_utilidad['tradeoff_3'] = np.nan

        df_prev_utilidad['eta_plant'] = np.nan
        df_prev_utilidad['eta'] = np.nan

        df_prev_utilidad['exceso_capacidad'] = np.nan

        df_prev_utilidad['FEH_ARRIBO_ESTIMADA'] = np.nan
        df_prev_utilidad['FEH_DESCARGA_ESTIMADA'] = np.nan

        # mask_props = (df_embarcaciones['marea_status'] == 'ESPERANDO DESCARGA') & (df_embarcaciones['owner_group'] == 'P')
        # df_tons = df_embarcaciones.loc[mask_props, ['marea_status', 'declared_ton', 'marea_id', 'eta_plant']].groupby(['eta_plant'], as_index=False)['declared_ton'].sum()

        # df_prev_utilidad = pd.merge(df_prev_utilidad, df_tons, left_on='PLANTA_1', right_on='eta_plant', how='left')
        # Calcular columna para aumentar "x" tiempo para el consumo de stock de pozas
        stock_planta = df_pozas_estado.loc[:,['id_planta','poza_number','stock_actualizado','pozaCapacity']].drop_duplicates()
        stock_planta = stock_planta.groupby(['id_planta']).agg(stock_actualizado=('stock_actualizado','max'),
                                                           capacidad=('pozaCapacity','sum'))
        stock_planta = pd.DataFrame(stock_planta)
        stock_planta.reset_index(inplace=True)
        # Unir con las velocidad de cocina de cada planta
        stock_planta = stock_planta.merge(df_plantas_velocidad_limites[['id','velocidad']],left_on='id_planta',right_on='id')
        stock_planta = stock_planta[['id_planta','stock_actualizado','capacidad','velocidad']]
        # Validar si es que la velocidad de planta es 0
        stock_planta = stock_planta.merge(df_planta_velocidad_anterior[['id_planta','velocidad']],on='id_planta')
        stock_planta['velocidad'] = np.where(stock_planta['velocidad_x']==0,stock_planta['velocidad_y'],stock_planta['velocidad_x'])
        stock_planta = stock_planta[['id_planta','stock_actualizado','capacidad','velocidad']]    
        stock_planta['horas_consumo'] = stock_planta['stock_actualizado']/stock_planta['velocidad']
        try:
            stock_planta['ocupacion'] = stock_planta['stock_actualizado']*100/stock_planta['capacidad']
        except:
            stock_planta['ocupacion'] = 100
        
        # Si se excede un 60% de la capacidad de planta, aumentar el tiempo, sino dejar en 0
        tope_ocupacion = 60
        stock_planta['horas_consumo'] = np.where(stock_planta['ocupacion']>tope_ocupacion, stock_planta['horas_consumo'], 0)
        
        # Agregar las horas de consumo de pozas al tiempo_fin_descarga
        df_final_1 = df_final_1.merge(stock_planta[['id_planta','velocidad','horas_consumo']],on='id_planta')
        df_final_1['tiempo_fin_descarga'] = pd.to_datetime(df_final_1['tiempo_fin_descarga'])
        df_final_1['tiempo_fin_descarga'] += pd.to_timedelta(df_final_1.horas_consumo, unit='h')
 
         # Calcular cantidad de EPs esperando descarga por planta y tipo de embarcacion
        df_esp_desc = df_embarcaciones_esperando_descarga.groupby(['discharge_plant_name','owner_group']).marea_id.count()
        df_esp_desc = pd.DataFrame(df_esp_desc)
        df_esp_desc.reset_index(inplace=True)
        df_esp_desc_propias = df_esp_desc[df_esp_desc['owner_group']=='P']
        df_esp_desc_propias.columns = ['discharge_plant_name','Tipo_emb','Cantidad_propias']
        df_esp_desc_terc = df_esp_desc[df_esp_desc['owner_group']=='T']
        df_esp_desc_terc.columns = ['discharge_plant_name','Tipo_emb','Cantidad_terceras']
        
        df_final_1 = df_final_1.merge(df_esp_desc_propias[['discharge_plant_name','Cantidad_propias']], how='left', left_on='id_planta', right_on='discharge_plant_name')
        df_final_1['Cantidad_propias'] = df_final_1['Cantidad_propias'].fillna(0)
        
        df_final_1 = df_final_1.merge(df_esp_desc_terc[['discharge_plant_name','Cantidad_terceras']], how='left', left_on='id_planta', right_on='discharge_plant_name')
        df_final_1['Cantidad_terceras'] = df_final_1['Cantidad_terceras'].fillna(0)
        del df_final_1['discharge_plant_name']
        del df_final_1['discharge_plant_name_y']   
 
        print("previo optimizar ordenes")
        df_final_2=optimizar_ordenes(df_final_1,df_embarcaciones,df_pozas_estado,timestamp,df_plantas_velocidad_limites,df_tiempo_descarga,df_requerimiento_plantas,df_lineas_reservada_terceros, df_chatas_lineas,df_prev_utilidad,costo_combustible, df_calidades_precio_venta, df_mareas_cerradas)[1]
        df_embarcaciones_terceros['gph_opt'] = np.nan
        df_embarcaciones_terceros['gph_max'] = np.nan
        df_final_3=pd.concat([df_final_2,df_embarcaciones_terceros])[['orden','boat_name_trim','boat_name','owner_group','declared_ton','tipo_bodega','SPEED_OPT_km','SPEED_MAX_km','marea_id','first_cala_start_date','Latitud','Longitud','Hora_llegada','eta','eta_plant','fish_zone_departure_date','gph_opt','gph_max']].reset_index(drop=True)
        df_final_3.loc[df_final_3.owner_group!='P','Hora_llegada']=df_final_3[df_final_3.owner_group!='P'].eta
        df_final_3=df_final_3.sort_values(by='Hora_llegada').reset_index(drop=True)
        df_final_3['orden']=np.arange(1,len(df_final_3)+1)
        df_final_3.loc[(df_final_3.owner_group!='P'),['SPEED_OPT_km','SPEED_MAX_km']]=1
        # df_final_4=df_final_3.merge(df_pozas_estado,how='cross')
        df_final_4 = cross_product(df_final_3, df_pozas_estado)
        mask = ((df_final_4['owner_group'] == 'T') & (df_final_4['eta_plant'].notna()) & (df_final_4['id_planta'] == df_final_4['eta_plant']) | (df_final_4['owner_group'] == 'P'))
        df_final_4 = df_final_4[mask].reset_index(drop=True)
        # df_pozas_estado['tiempo_fin_descarga'] = df_final_1['tiempo_fin_descarga'].copy()
        # df_final_4=cross_product(df_embarcaciones_retornando, df_pozas_estado[df_pozas_estado['Planta'].notna()].drop_duplicates(['Planta', 'id_chata']))
        # df_final_4=cross_product(df_embarcaciones_retornando, df_pozas_estado[df_pozas_estado['id_planta'].notna()])
        df_final_4 = df_final_4.merge(stock_planta[['id_planta','velocidad','horas_consumo']],on='id_planta')
        df_final_4['tiempo_fin_descarga'] = pd.to_datetime(df_final_4['tiempo_fin_descarga'])
        df_final_4['tiempo_fin_descarga'] += pd.to_timedelta(df_final_4.horas_consumo, unit='h')
        # del df_final_4['tiempo_fin_descarga']
        # df_final_4 = pd.merge(df_final_4, df_final_1[['id_planta', 'lineaname_y', 'pozaname', 'tiempo_fin_descarga']], left_on=['id_planta', 'lineaname', 'pozaname'], right_on=['id_planta', 'lineaname_y', 'pozaname'], how='left')
        df_final_4['fish_zone_departure_date']=pd.to_datetime(df_final_4['fish_zone_departure_date'])
        col_adicionales = df_final_1[['id_planta', 'Cantidad_propias','Cantidad_terceras']].drop_duplicates()
        df_final_4 = df_final_4.merge(col_adicionales,on='id_planta')
        recomendaciones_final,dframe,df_pozas_estado_final,df_retorno_utilidad=optimizar_ordenes(df_final_4,df_embarcaciones,df_pozas_estado,timestamp,df_plantas_velocidad_limites,df_tiempo_descarga,df_requerimiento_plantas,df_lineas_reservada_terceros, df_chatas_lineas,df_prev_utilidad,costo_combustible, df_calidades_precio_venta, df_mareas_cerradas)
        df_recomendaciones=pd.DataFrame(recomendaciones_final,columns=['orden','boat_name','marea_id','owner_group','toneladas','id_planta','id_chata','id_linea','velocidad'])
        df_recomendaciones['tipo']=np.where(df_recomendaciones['owner_group']=='T','terceros','retornando')
        df_recomendaciones['prioridad']=list(np.arange(1,len(df_recomendaciones)+1))
        df_recomendaciones.drop(columns=['orden','owner_group'],inplace=True)
        df_retorno_utilidad = pd.merge(df_retorno_utilidad, df_recomendaciones[['marea_id', 'boat_name', 'toneladas']], how='left', on='marea_id')
        return df_pozas_estado_final, df_recomendaciones, df_retorno_utilidad

    def optimizar_ordenes(df_cross_join,df_embarcaciones,df_pozas_estado,timestamp,df_plantas_velocidad_limites,df_tiempo_descarga,df_requerimiento_plantas,df_lineas_reservada_terceros, df_chatas_lineas,df_prev_utilidad,costo_combustible,df_calidades_precio_venta, df_mareas_cerradas):
        #Se excluyen las chata-lineas reservadas para terceros
        
        
        ## Balanceo
        cerradas = df_mareas_cerradas.copy()
        # tope_historico = timestamp+datetime.timedelta(days=-18) #cambiar a una semana, solo es pruebas
        year_temporada = (pd.to_datetime(cerradas['production_date']).max() - datetime.timedelta(days=15)).year
        fecha_primera_temporada = f'{year_temporada}/03/10'
        fecha_segunda_temporada = f'{year_temporada}/10/10'

        dif_1 = (timestamp - pd.to_datetime(fecha_primera_temporada)).days
        dif_2 = (timestamp - pd.to_datetime(fecha_segunda_temporada)).days

        if dif_2 < 0:
            FECHA_INICIO = fecha_primera_temporada
        else:
            FECHA_INICIO = fecha_segunda_temporada

        if dif_1 < 0:
            FECHA_INICIO = '2019/01/01'
        cerradas['production_date'] = pd.to_datetime(cerradas['production_date'])
        filtro_historico = cerradas['production_date']>FECHA_INICIO

        cerradas = df_mareas_cerradas.loc[filtro_historico & cerradas['discharge_chata_name'].notnull(),
                                            ['marea_id','discharge_plant_name','discharge_chata_name','discharge_line_name','declared_ton']]
        cerradas.loc[cerradas['discharge_chata_name']=='TASA CALLAO','discharge_chata_name'] = 'CHILLON'
        # cerradas['chatalinea'] = cerradas['discharge_chata_name'] + '-' + cerradas['discharge_line_name']
        cerradas.reset_index(inplace=True)
        
        abiertas = df_embarcaciones.loc[(df_embarcaciones.marea_status=='DESCARGANDO')
                                        | (df_embarcaciones.marea_status=='EP ESPERANDO ZARPE'),
                                        ['marea_id','discharge_plant_name','discharge_chata_name','discharge_line_name','declared_ton']]
        # abiertas['chatalinea'] = abiertas['discharge_chata_name'] + '-' + abiertas['discharge_line_name']
        abiertas.reset_index(inplace=True)
        mareas_ult_semana = pd.concat([cerradas,abiertas],axis=0)
        
        df_descarga_planta = mareas_ult_semana.groupby(['discharge_plant_name','discharge_chata_name'],as_index=False).declared_ton.sum()
        df_descarga_planta = df_descarga_planta.rename(columns ={'declared_ton':'total_ton_declared'})                        
        df_descarga_planta['discharge_chata_name'] = str('CHATA ') + df_descarga_planta['discharge_chata_name']
        try:
            if (len(df_descarga_planta.loc[df_descarga_planta['discharge_plant_name']=='VEGUETA'])>0) & (len(df_descarga_planta.loc[df_descarga_planta['discharge_chata_name']=='CHATA TASA'])>0):
                df_descarga_planta.loc[(df_descarga_planta['discharge_plant_name']=='VEGUETA') & (df_descarga_planta['discharge_chata_name']=='CHATA TASA'),'total_ton_declared'] = 2*df_descarga_planta.loc[(df_descarga_planta['discharge_plant_name']=='VEGUETA') & (df_descarga_planta['discharge_chata_name']=='CHATA TASA'),'total_ton_declared']
        except:
            pass
        # print(df_descarga_planta)
        pozas_disponibles = df_pozas_estado.copy()
        # pozas_disponibles = pozas_disponibles.loc[(pozas_disponibles.habilitado==True)]
        pozas_disponibles['poza_number'] = pozas_disponibles['poza_number'].map('{0:g}'.format)
        pozas_disponibles['chatalineapoza'] = pozas_disponibles['id_chata'] + '-' + pozas_disponibles['id_linea'] + '-' + pozas_disponibles['poza_number']
        chatalineapoza_posible = pd.DataFrame(pozas_disponibles.chatalineapoza.value_counts().reset_index())
        chatalineapoza_posible.columns=['chatalineapoza','cuenta']
        # OJO: Si hay un cambio entre la conexion linea-poza, no se podra recuperar la relacion anterior
        # en las opciones posibles estas no apareceran
        chatalineapoza_posible = chatalineapoza_posible.merge(pozas_disponibles[['chatalineapoza','id_chata', 'id_linea']], how='left', on='chatalineapoza')
        consolidado_balanceo = chatalineapoza_posible.merge(df_descarga_planta,how='left', left_on='id_chata', right_on='discharge_chata_name')
        ##
        
        df_lineas_reservada_terceros=df_lineas_reservada_terceros[df_lineas_reservada_terceros.reserv_terc.notnull()].copy()
        df_lineas_reservada_terceros=df_lineas_reservada_terceros[df_lineas_reservada_terceros.reserv_terc].copy()
        df_lineas_reservada_terceros['chata-linea']=df_lineas_reservada_terceros['id_chata']+'-'+df_lineas_reservada_terceros['id_linea']
        chata_lineas_reservadas=list(df_lineas_reservada_terceros['chata-linea'].unique())

        df_cross_join['chata-linea']=df_cross_join['id_chata']+'-'+df_cross_join['id_linea']
        
        division_modelo = False
        
        if division_modelo:
            df_chata_linea = df_chatas_lineas.loc[df_chatas_lineas['habilitado_retorno'] == False, ['id_chata', 'id_linea', 'habilitado_retorno']]
            df_chata_linea['chata-linea'] = df_chata_linea['id_chata'] + '-' + df_chata_linea['id_linea']
            chata_deshabilitadas = df_chata_linea.loc[df_chata_linea['habilitado_retorno'] == False, 'chata-linea'].tolist()
        else:
            df_chata_linea = df_chatas_lineas.loc[df_chatas_lineas['habilitado'] == False, ['id_chata', 'id_linea', 'habilitado']]
            df_chata_linea['chata-linea'] = df_chata_linea['id_chata'] + '-' + df_chata_linea['id_linea']
            chata_deshabilitadas = df_chata_linea.loc[df_chata_linea['habilitado'] == False, 'chata-linea'].tolist()
        
        df_chatas_restantes = df_chatas_lineas[~df_chatas_lineas['chatalinea'].isin(chata_deshabilitadas)]
        
        # EPs con planta ETA asignada
        df_eta_ingresada = df_cross_join.groupby(['eta_plant']).agg(Cuenta=('marea_id','nunique'))
        df_eta_ingresada.reset_index(inplace=True)
        
        chatas_out = list()
        
        eta_registrados = df_eta_ingresada['eta_plant'].unique().tolist()
        df_terceros_restantes = df_lineas_reservada_terceros[~df_lineas_reservada_terceros['id_planta'].isin(eta_registrados)]
        
        for index, row in df_eta_ingresada.iterrows():
            planta_eta = row['eta_plant']
            df_filtro = df_chatas_restantes[df_chatas_restantes['id_planta']==planta_eta]
            df_deshabilitados = df_lineas_reservada_terceros.loc[df_lineas_reservada_terceros['id_planta']==planta_eta,'chata-linea'].tolist()
            df_filtro = df_filtro[~df_filtro['chatalinea'].isin(df_deshabilitados)]
            if len(df_filtro)>0:
                chatas_out.append(df_deshabilitados)
        
        # Agregar chatas reservadas de terceros de plantas no registradas en ETA
        chatas_out.append(df_terceros_restantes['chata-linea'].tolist())
        chatas_out = list(chain.from_iterable(chatas_out))
        plantas_terceras = df_lineas_reservada_terceros['id_planta'].unique().tolist()
        
        df_cross_join = df_cross_join[~df_cross_join['chata-linea'].isin(chata_deshabilitadas)]
        # df_cross_join=df_cross_join[(~(df_cross_join['chata-linea'].isin(chata_lineas_reservadas))&(df_cross_join['owner_group']=='P'))|(df_cross_join['owner_group']=='T')].copy()
        df_cross_join=df_cross_join.merge(df_requerimiento_plantas[['id','hora_inicio']],how='left',left_on='id_planta',right_on='id')

        df_cross_join['hora_inicio']=np.where((df_cross_join.fish_zone_departure_date.dt.hour>=12)& # TODO: Evaluar si fish_zone_departure_date se reemplaza por el timestamp
                                    (df_cross_join.fish_zone_departure_date.dt.date>df_cross_join.hora_inicio.dt.date),
                                    df_cross_join['hora_inicio']+datetime.timedelta(days=1),df_cross_join['hora_inicio'])
        df_cross_join['hora_inicio']=np.where((df_cross_join.fish_zone_departure_date.dt.hour<12)&
                                    (df_cross_join.fish_zone_departure_date.dt.date==df_cross_join.hora_inicio.dt.date),
                                    df_cross_join['hora_inicio']+datetime.timedelta(days=-1),df_cross_join['hora_inicio'])
        # del df_cross_join['eta_plant']
        # df_cross_join = pd.merge(df_cross_join, df_embarcaciones[['marea_id', 'eta_plant']], how='left', on='marea_id')
        # mask = (df_cross_join['eta_plant'] != df_cross_join['id_planta'])
        # df_cross_join.loc[mask, 'id_planta'] = df_cross_join.loc[mask, 'eta_plant']  
        # mask = ((df_cross_join['eta_plant'] == df_cross_join['id_planta']) | df_cross_join['eta_plant'].isna()) #| ((timestamp - df_cross_join['fish_zone_departure_date']).dt.total_seconds() / 60 < 30)
        # df_cross_join = df_cross_join[mask].reset_index(drop=True)
        try:
            df_cross_join.loc[df_cross_join.owner_group=='P','llegada_vel_opt']=df_cross_join.loc[df_cross_join.owner_group=='P'].apply(lambda x: tiempo_a_plantas(x.id_planta,x.marea_id,x.SPEED_OPT_km,df_embarcaciones,df_pozas_estado,timestamp),axis=1)
            df_cross_join.loc[df_cross_join.owner_group=='P','llegada_vel_max']=df_cross_join.loc[df_cross_join.owner_group=='P'].apply(lambda x: tiempo_a_plantas(x.id_planta,x.marea_id,x.SPEED_MAX_km,df_embarcaciones,df_pozas_estado,timestamp),axis=1)
            df_cross_join.loc[df_cross_join.owner_group=='P','distancia_planta']=df_cross_join.loc[df_cross_join.owner_group=='P'].apply(lambda x: distancia(x.coordslatitude,x.coordslongitude,x.Latitud,x.Longitud),axis=1)
        except:
            pass
        
        # Calcular la capacidad maxima de las pozas disponibles por planta
        df_capacidad_pozas = df_pozas_estado[['id_planta','poza_number','pozaCapacity']].drop_duplicates()
        df_capacidad_pozas = df_capacidad_pozas.groupby(['id_planta']).agg({'pozaCapacity':'sum'})
        df_capacidad_pozas.reset_index(inplace=True)
        df_capacidad_pozas.columns = ['id_planta','capacidad_poza']

        status_pozas_final=df_pozas_estado.copy()
        data_aux=df_cross_join.copy()
        data_aux['tvn_pozas'] = 0
        data_plantas_update = data_aux.copy(deep=True)
        data_plantas_update = data_plantas_update.groupby(['id_planta'], as_index=False)['tvn_pozas'].sum()
        data_plantas_update['num_recoms'] = 0
        df_mareas_cerradas_copy = pd.merge(df_mareas_cerradas, df_requerimiento_plantas[['id', 'hora_inicio']], how='left', left_on='discharge_plant_name', right_on='id') 
        mask = (df_mareas_cerradas_copy['marea_status'] == 'EP ESPERANDO ZARPE') & (pd.to_datetime(df_mareas_cerradas_copy['discharge_start_date']) > df_mareas_cerradas_copy['hora_inicio'])
        df_mareas_cerradas_copy = df_mareas_cerradas_copy[mask].groupby(['id']).agg(
            count_mareas=('marea_id', 'nunique')
        ).reset_index()
        if df_mareas_cerradas_copy['count_mareas'].sum() == 0:
            data_plantas_update['count_mareas'] = 0
        else:
            df_mareas_cerradas_copy['count_mareas'] = df_mareas_cerradas_copy['count_mareas'].fillna(0)
            data_plantas_update = pd.merge(data_plantas_update, df_mareas_cerradas_copy, left_on='id_planta', right_on='id')    
        lista_embarcaciones=[]
        recomendaciones=[]
        res_no_optimos = list()
        tope_tvn = 100
        orden=1
        while len(data_aux)>0:
            tope = 0.8
            try:
                data_aux=data_aux.sort_values(by='orden').reset_index(drop=True).copy()
            except: 
                data_aux=data_aux.sort_values(by='distancia_planta').reset_index(drop=True).copy()
            embarcacion_optimizar=data_aux.loc[0,'boat_name_trim']
            nombre_embarcacion=data_aux.loc[0,'boat_name']
            print(nombre_embarcacion)
            # Mantener full cross join original
            if orden==1:
                data_aux_original = data_aux.copy()
            
            tonelaje=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'declared_ton'].values[0]
            tipo_bodega=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'tipo_bodega'].values[0]
            speed_opt=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'SPEED_OPT_km'].values[0]
            speed_max=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'SPEED_MAX_km'].values[0]
            marea_id=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'marea_id'].values[0]
            first_cala=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'first_cala_start_date'].values[0]
            latitud=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'Latitud'].values[0]
            longitud=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'Longitud'].values[0]
            owner=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'owner_group'].values[0]
            gph_opt=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'gph_opt'].values[0]
            gph_max=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'gph_max'].values[0]

            eta_out = data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'eta'].values[0]
            eta_plant = data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'eta_plant'].values[0]
            fish_zone_departure_date = data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'fish_zone_departure_date'].values[0]
            
            # Eliminar lineas reservadas para terceras de acuerdo a si se tiene o no ingresada eta plant
            if eta_plant in plantas_terceras:
                data_aux = data_aux[(~(data_aux['chata-linea'].isin(chatas_out)) & (data_aux['owner_group']=='P') & (data_aux['marea_id']==marea_id)) | (data_aux['owner_group']=='T') | (data_aux['marea_id']!=marea_id)].copy()
            else:
                data_aux = data_aux[(~(data_aux['chata-linea'].isin(chata_lineas_reservadas)) & (data_aux['owner_group']=='P') & (data_aux['marea_id']==marea_id)) | (data_aux['owner_group']=='T') | (data_aux['marea_id']!=marea_id)].copy()
            
            if owner=='P':
                auxiliar=data_aux[data_aux.boat_name_trim==embarcacion_optimizar].copy()
                # tope = np.repeat(tope, len(auxiliar.index)).reshape(-1, )
                auxiliar['tiempo_consumo_cocina'] = 0
                auxiliar['combustible_opt'] = (auxiliar['distancia_planta'] / auxiliar['SPEED_OPT_km'] ) * auxiliar['gph_opt'] * costo_combustible
                auxiliar['combustible_max'] = (auxiliar['distancia_planta'] / auxiliar['SPEED_MAX_km']) * auxiliar['gph_max'] * costo_combustible
                auxiliar['descarga_vel_opt']=auxiliar[['llegada_vel_opt','tiempo_fin_descarga','hora_inicio']].max(axis=1)
                auxiliar['descarga_vel_max']=auxiliar[['llegada_vel_max','tiempo_fin_descarga','hora_inicio']].max(axis=1)
                auxiliar['tdc_max']=(auxiliar['descarga_vel_max']-auxiliar['first_cala_start_date'])/np.timedelta64(1,'h')
                auxiliar['tdc_opt']=(auxiliar['descarga_vel_opt']-auxiliar['first_cala_start_date'])/np.timedelta64(1,'h')
                auxiliar['bodega_frio']=np.where(auxiliar['tipo_bodega'].isin(['Frio-RC','Frio-GF']),1,0)
                auxiliar['tvn_max']=auxiliar.apply(lambda x: get_return_model_polynomial_value(x['tdc_max'],x['bodega_frio']),axis=1)
                auxiliar['tvn_opt']=auxiliar.apply(lambda x: get_return_model_polynomial_value(x['tdc_opt'],x['bodega_frio']),axis=1)
                auxiliar['limite_tvn_max']=auxiliar.apply(lambda x: cerca_tvn_limite(x.tvn_max,x.id_planta,0.1,df_plantas_velocidad_limites),axis=1)
                auxiliar['limite_tvn_opt']=auxiliar.apply(lambda x: cerca_tvn_limite(x.tvn_opt,x.id_planta,0.1,df_plantas_velocidad_limites),axis=1)
                # auxiliar['velocidad_recomendada']=np.where((auxiliar['descarga_vel_opt']>auxiliar['descarga_vel_max'])|((auxiliar['limite_tvn_max']==0)&(auxiliar['limite_tvn_opt']==1)),'MAX','OPT')
                # auxiliar['inicio_descarga_vel_recom']=np.where(auxiliar['velocidad_recomendada']=='MAX',auxiliar['descarga_vel_max'],auxiliar['descarga_vel_opt'])
                # auxiliar['hora_llegada_vel_recom']=np.where(auxiliar['velocidad_recomendada']=='MAX',auxiliar['llegada_vel_max'],auxiliar['llegada_vel_opt'])
                # auxiliar['tdc_llegada_vel_recom']=np.where(auxiliar['velocidad_recomendada'] == 'MAX',auxiliar['tdc_max'], auxiliar['tdc_opt'])
                # auxiliar['tvn_llegada_vel_recom']=np.where(auxiliar['velocidad_recomendada'] == 'MAX',auxiliar['tvn_max'], auxiliar['tvn_opt'])
                # auxiliar['combustible_recom'] = np.where(auxiliar['velocidad_recomendada'] == 'MAX',auxiliar['combustible_max'], auxiliar['combustible_opt'])
                # TODO: si timestamp < hora_inicio then opt, else tope = 1 (max) para el primero de cada planta
                # Calcular el saldo en pozas, recordar que todas inician en 0
                auxiliar['inicio_actual_opt'] = auxiliar['descarga_vel_opt']
                auxiliar['inicio_actual_max'] = auxiliar['descarga_vel_max']
                auxiliar_opt = auxiliar[['chata-linea','inicio_actual_opt']].sort_values('inicio_actual_opt').drop_duplicates(['chata-linea'], keep='first')
                auxiliar_max = auxiliar[['chata-linea','inicio_actual_max']].sort_values('inicio_actual_max').drop_duplicates(['chata-linea'], keep='first')
                
                # Agregar columna con el nuevo inicio calculado al dataframe con todas las combinaciones
                if 'inicio_actual' in data_aux_original.columns:
                    del data_aux_original['inicio_actual']
                
                if 'inicio_actual_opt' in data_aux_original.columns:
                    del data_aux_original['inicio_actual_opt']
                
                if 'inicio_actual_max' in data_aux_original.columns:
                    del data_aux_original['inicio_actual_max']
                
                data_aux_original = data_aux_original.merge(auxiliar_opt, how='left', on=['chata-linea'])
                data_aux_original = data_aux_original.merge(auxiliar_max, how='left', on=['chata-linea'])
                
                if orden==1:
                
                    # Inicializar campos de inicio, stock anterior, stock recalculo
                    data_aux_original['inicio_anterior'] = np.nan
                    data_aux_original['inicio_anterior'] = pd.to_datetime(data_aux_original['inicio_anterior'])
                    data_aux_original['stock_recalculo'] = np.where(data_aux_original['marea_id']==marea_id,tonelaje,0)
                    data_aux_original['stock_anterior'] = 0
                    data_aux_original['tvn_anterior'] = 0
                    auxiliar['stock_recalculo'] = tonelaje
                    auxiliar['stock_recalculo_opt'] = tonelaje
                    auxiliar['stock_recalculo_max'] = tonelaje
                    auxiliar['tvn_planta_opt'] = auxiliar['tvn_opt']
                    auxiliar['tvn_planta_max'] = auxiliar['tvn_max']
                    auxiliar['tvn_anterior'] = 0
                    data_aux_original = pd.merge(data_aux_original,auxiliar[['id_planta','chata-linea','pozaname','tvn_planta_opt','tvn_planta_max']],how='left',on=['id_planta','chata-linea','pozaname'])
                    
                else:
                    
                    # Actualizar el stock anterior con el ultimo stock de la iteracion anterior, actualizar stock actual
                    data_aux_original['stock_anterior'] = data_aux_original['stock_recalculo'].copy()
                    data_aux_original['stock_recalculo'] = np.where(data_aux_original['marea_id']==marea_id,data_aux_original['stock_recalculo'] + tonelaje, data_aux_original['stock_recalculo'])
                    
                    data_aux_orig_red = data_aux_original[['chata-linea','inicio_anterior']].drop_duplicates('chata-linea')
                    auxiliar = auxiliar.merge(data_aux_orig_red, how='left', on='chata-linea')
                    
                    # Calculo de tiempo transcurrido desde descarga anterior
                    # Para Caso OPT
                    auxiliar['tiempo_consumo_cocina_opt'] = (auxiliar['inicio_actual_opt'] - auxiliar['inicio_anterior'])/datetime.timedelta(hours=1)
                    auxiliar['tiempo_consumo_cocina_opt'] =np.where(auxiliar['tiempo_consumo_cocina_opt']<0,0,auxiliar['tiempo_consumo_cocina_opt'])
                    auxiliar['tiempo_consumo_cocina_opt'] = auxiliar['tiempo_consumo_cocina_opt'].fillna(0)
                    auxiliar['consumo_cocina_opt'] = auxiliar['tiempo_consumo_cocina_opt']*auxiliar['velocidad']
                    
                    # Para Caso MAX
                    auxiliar['tiempo_consumo_cocina_max'] = (auxiliar['inicio_actual_max'] - auxiliar['inicio_anterior'])/datetime.timedelta(hours=1)
                    auxiliar['tiempo_consumo_cocina_max'] =np.where(auxiliar['tiempo_consumo_cocina_max']<0,0,auxiliar['tiempo_consumo_cocina_max'])
                    auxiliar['tiempo_consumo_cocina_max'] = auxiliar['tiempo_consumo_cocina_max'].fillna(0)
                    auxiliar['consumo_cocina_max'] = auxiliar['tiempo_consumo_cocina_max']*auxiliar['velocidad']
                    
                    # Agregar stock y calcular primero consumo por alimentacion a cocina y despues stock hipotetico de descarga
                    auxiliar = auxiliar.merge(data_aux_original[['marea_id','id_planta','stock_recalculo','stock_anterior','tvn_anterior']].drop_duplicates(),how='left',on=['marea_id','id_planta'])
                    
                    # Para caso OPT
                    auxiliar['stock_cocina_opt'] = auxiliar['stock_anterior'] - auxiliar['consumo_cocina_opt']
                    auxiliar['stock_cocina_opt'] = np.where(auxiliar['stock_cocina_opt']<0,0,auxiliar['stock_cocina_opt'])
                    data_aux_original['stock_recalculo'] = np.where(data_aux_original['marea_id']==marea_id,data_aux_original['stock_recalculo'] - tonelaje,data_aux_original['stock_recalculo'])
                    auxiliar['stock_recalculo_opt'] = auxiliar['stock_cocina_opt'] + tonelaje
                    
                    # Para caso MAX
                    auxiliar['stock_cocina_max'] = auxiliar['stock_anterior'] - auxiliar['consumo_cocina_max']
                    auxiliar['stock_cocina_max'] = np.where(auxiliar['stock_cocina_max']<0,0,auxiliar['stock_cocina_max'])
                    data_aux_original['stock_recalculo'] = np.where(data_aux_original['marea_id']==marea_id,data_aux_original['stock_recalculo'] - tonelaje,data_aux_original['stock_recalculo'])
                    auxiliar['stock_recalculo_max'] = auxiliar['stock_cocina_max'] + tonelaje
                    
                    # Calculo tvn ponderado pozas por planta
                    # Aumento de TVN residencia pozas (residencia = tiempo_consumo_cocina)
                    # Para caso OPT
                    auxiliar['aumento_tvn_pozas_opt'] = auxiliar.apply(get_tvn_increase_in_poza_for_marea_opt, axis=1)
                    auxiliar['tvn_pozas_opt'] = auxiliar['tvn_anterior'] + auxiliar['aumento_tvn_pozas_opt']
                    auxiliar['tvn_planta_opt'] = (auxiliar['tvn_pozas_opt']*auxiliar['stock_cocina_opt'] + auxiliar['tvn_opt']*tonelaje)/auxiliar['stock_recalculo_opt']
                    
                    # Para caso MAX
                    auxiliar['aumento_tvn_pozas_max'] = auxiliar.apply(get_tvn_increase_in_poza_for_marea_max, axis=1)
                    auxiliar['tvn_pozas_max'] = auxiliar['tvn_anterior'] + auxiliar['aumento_tvn_pozas_max']
                    auxiliar['tvn_planta_max'] = (auxiliar['tvn_pozas_max']*auxiliar['stock_cocina_max'] + auxiliar['tvn_max']*tonelaje)/auxiliar['stock_recalculo_max']
                           
                # Calculo de Calidad
                auxiliar = pd.merge(auxiliar,df_plantas_velocidad_limites[['id','lim_tvn_cocina_super_prime','lim_tvn_cocina_prime']],how='left',left_on='id_planta',right_on='id')
                
                # Para Caso OPT
                auxiliar['calidad_planta_opt'] = np.where(auxiliar['tvn_planta_opt']<=auxiliar['lim_tvn_cocina_super_prime'],'SUPER_PRIME','PRIME')
                auxiliar['calidad_planta_opt'] = np.where(auxiliar['tvn_planta_opt']>auxiliar['lim_tvn_cocina_prime'],'OTHERS',auxiliar['calidad_planta_opt'])
                # Penalizacion al acercarse a los limites de tvn de la calidad
                # Para categoria OTHERS se coloca un limite alto (tope_tvn=150)
                auxiliar['proporcion_calidad_opt'] = 1 - (auxiliar['tvn_planta_opt']/tope_tvn)
                auxiliar = pd.merge(auxiliar,df_calidades_precio_venta[['QUALITY_LEVEL','PRICE_PER_TON']],how='left',left_on='calidad_planta_opt',right_on='QUALITY_LEVEL')
                
                # Para Caso MAX
                auxiliar['calidad_planta_max'] = np.where(auxiliar['tvn_planta_max']<=auxiliar['lim_tvn_cocina_super_prime'],'SUPER_PRIME','PRIME')
                auxiliar['calidad_planta_max'] = np.where(auxiliar['tvn_planta_max']>auxiliar['lim_tvn_cocina_prime'],'OTHERS',auxiliar['calidad_planta_max'])
                # Penalizacion al acercarse a los limites de tvn de la calidad
                # Para categoria OTHERS se coloca un limite alto (tope_tvn=150)
                auxiliar['proporcion_calidad_max'] = 1 - (auxiliar['tvn_planta_max']/tope_tvn)
                auxiliar = pd.merge(auxiliar,df_calidades_precio_venta[['QUALITY_LEVEL','PRICE_PER_TON']],how='left',left_on='calidad_planta_max',right_on='QUALITY_LEVEL')
                
                 # Calcular penalizacion por llenado de pozas
                # Para caso OPT
                auxiliar = auxiliar.merge(df_capacidad_pozas,on='id_planta')
                auxiliar['proporcion_pozas_opt'] = auxiliar['stock_recalculo_opt']/auxiliar['capacidad_poza']
                
                # Para caso MAX
                auxiliar['proporcion_pozas_max'] = auxiliar['stock_recalculo_max']/auxiliar['capacidad_poza']
                auxiliar = pd.merge(auxiliar, data_plantas_update[['id_planta', 'num_recoms', 'count_mareas']], how='left', on='id_planta')
                mask = (timestamp > auxiliar['hora_inicio']) & (auxiliar['num_recoms'] == 1) & (auxiliar['count_mareas'] == 0)
                # tope = np.where(mask, 1, tope)
                # tope = np.where(mask, 1, mask)
                tope = 1
                
                # Agregar limitacion de distancia plantas (3 veces de la distancia minima)
                # Solo se evaluan los que no estan demasiado lejos
                min_distancia = min(auxiliar['distancia_planta'])
                tope_distancia = min_distancia*3
                auxiliar['exceso_distancia'] = np.where(auxiliar['distancia_planta']>tope_distancia, 0, 1)
                auxiliar['exceso_capacidad_opt'] = np.where(auxiliar['exceso_distancia']>0,
                                                            np.where(auxiliar['proporcion_pozas_opt'] > tope, 0, 1), 0)
                
                # Si ninguno es apto, liberar la restriccion de tope
                if sum(auxiliar['exceso_capacidad_opt'])==0:
                    auxiliar['exceso_capacidad_opt'] = np.where(auxiliar['exceso_distancia']>0, 1, 0)
                
                # Replicar para MAX
                auxiliar['exceso_capacidad_max'] = np.where(auxiliar['exceso_distancia']>0,
                                                            np.where(auxiliar['proporcion_pozas_max'] > tope, 0, 1), 0)
                
                # Si ninguno es apto, liberar la restriccion de tope
                if sum(auxiliar['exceso_capacidad_max'])==0:
                    auxiliar['exceso_capacidad_max'] = np.where(auxiliar['exceso_distancia']>0, 1, 0)
                
                # Calculo del tradeoff (por marea entrante, no por TM de planta, incluyendo penalizacion)
                # La marea adopta la calidad del ponderado en poza
                # Para caso OPT
                # Se divide entre rendimiento
                auxiliar['valor_pozas_opt'] = auxiliar['PRICE_PER_TON_x']*tonelaje*auxiliar['proporcion_calidad_opt'] * auxiliar['exceso_capacidad_opt']/4
                auxiliar['tradeoff_opt'] = auxiliar['valor_pozas_opt'] - auxiliar['combustible_opt']
                
                # Para caso MAX
                auxiliar['valor_pozas_max'] = auxiliar['PRICE_PER_TON_y']*tonelaje*auxiliar['proporcion_calidad_max'] * auxiliar['exceso_capacidad_max']/4
                auxiliar['tradeoff_max'] = auxiliar['valor_pozas_max'] - auxiliar['combustible_max']
                
                # Mejor tradeoff entre OPT y MAX
                auxiliar['tradeoff'] = np.where(auxiliar['tradeoff_opt']<auxiliar['tradeoff_max'],auxiliar['tradeoff_max'],auxiliar['tradeoff_opt'])
                auxiliar['velocidad_recomendada'] = np.where(auxiliar['tradeoff']==auxiliar['tradeoff_opt'],'OPT','MAX')
                auxiliar['inicio_descarga_vel_recom'] = np.where(auxiliar['tradeoff']==auxiliar['tradeoff_opt'],auxiliar['inicio_actual_opt'],auxiliar['inicio_actual_max'])
                auxiliar['hora_llegada_vel_recom'] = np.where(auxiliar['tradeoff']==auxiliar['tradeoff_opt'],auxiliar['llegada_vel_opt'],auxiliar['llegada_vel_max'])
                auxiliar['combustible_recom'] = np.where(auxiliar['tradeoff']==auxiliar['tradeoff_opt'],auxiliar['combustible_opt'],auxiliar['combustible_max'])
                auxiliar['valor_pozas'] = np.where(auxiliar['tradeoff']==auxiliar['tradeoff_opt'],auxiliar['valor_pozas_opt'],auxiliar['valor_pozas_max'])
                auxiliar['stock_recalculo'] = np.where(auxiliar['tradeoff']==auxiliar['tradeoff_opt'],auxiliar['stock_recalculo_opt'],auxiliar['stock_recalculo_max'])
                auxiliar['tvn_planta'] = np.where(auxiliar['tradeoff']==auxiliar['tradeoff_opt'],auxiliar['tvn_planta_opt'],auxiliar['tvn_planta_max'])
                auxiliar['inicio_actual'] = np.where(auxiliar['tradeoff']==auxiliar['tradeoff_opt'],auxiliar['inicio_actual_opt'],auxiliar['inicio_actual_max'])
                data_aux_original = pd.merge(data_aux_original,auxiliar[['id_planta','chata-linea','pozaname','inicio_actual']],how='left',on=['id_planta','chata-linea','pozaname'])
                
                resumen_balanceo = consolidado_balanceo[['id_chata', 'id_linea', 'total_ton_declared']].drop_duplicates()
                auxiliar = pd.merge(auxiliar, resumen_balanceo, on=['id_chata', 'id_linea'])
                auxiliar = auxiliar.sort_values(by=['tradeoff', 'total_ton_declared'],ascending=[False,True]).reset_index(drop=True)

            else:
                planta=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'eta_plant'].values[0]
                eta=data_aux.loc[data_aux.boat_name_trim==embarcacion_optimizar,'eta'].values[0]
                auxiliar=data_aux[(data_aux.boat_name_trim==embarcacion_optimizar)&(data_aux.id_planta==planta)].copy()
                # tope = np.repeat(tope, len(auxiliar.index)).reshape(-1, )
                auxiliar['velocidad_recomendada']=np.nan
                auxiliar['hora_llegada_vel_recom']=eta
                auxiliar['inicio_descarga_vel_recom']=np.maximum(auxiliar['tiempo_fin_descarga'],eta)
                auxiliar['combustible_recom'] = 0
                auxiliar['bodega_frio'] = 0
                
                auxiliar['tdc'] = (auxiliar['eta'] - auxiliar['first_cala_start_date']).dt.total_seconds() / (3600)

                try:
                    planta_eta = auxiliar['eta_plant'].tolist()[0]
                    # tvn_terce = df_prev_utilidad.loc[df_prev_utilidad['PLANTA_1'] == planta_eta, 'tvn_poza_planta_1'].to_numpy()[-1]
                    # auxiliar['tvn_opt'] = tvn_terce
                    # auxiliar['tvn_max'] = tvn_terce
                    auxiliar['tvn_max'] = auxiliar.apply(lambda x: get_return_model_polynomial_value(x['tdc'], 0), axis=1)
                    auxiliar['tvn_opt'] = auxiliar.apply(lambda x: get_return_model_polynomial_value(x['tdc'], 0), axis=1)  
                except:
                    # auxiliar['tvn_max'] = auxiliar.apply(lambda x: get_return_model_polynomial_value(x['tdc'], 0), axis=1)
                    # auxiliar['tvn_opt'] = auxiliar.apply(lambda x: get_return_model_polynomial_value(x['tdc'], 0), axis=1)
                    auxiliar['tvn_opt'] = 22.0
                    auxiliar['tvn_max'] = 20.0 # TODO: Validar donde colocar el flag de 22, 20
                
                # Calcular el saldo en pozas, recordar que todas inician en 0
                auxiliar['inicio_actual_opt'] = auxiliar['inicio_descarga_vel_recom']
                auxiliar_opt = auxiliar[['chata-linea','inicio_actual_opt']].drop_duplicates()
                
                auxiliar['inicio_actual_max'] = auxiliar['inicio_descarga_vel_recom']
                auxiliar_max = auxiliar[['chata-linea','inicio_actual_max']].drop_duplicates()
                
                
                # Agregar columna con el nuevo inicio calculado al dataframe con todas las combinaciones
                if 'inicio_actual' in data_aux_original.columns:
                    del data_aux_original['inicio_actual']
                
                if 'inicio_actual_opt' in data_aux_original.columns:
                    del data_aux_original['inicio_actual_opt']
                    
                if 'inicio_actual_max' in data_aux_original.columns:
                    del data_aux_original['inicio_actual_max']
                    
                data_aux_original = data_aux_original.merge(auxiliar_opt, how='left', on=['chata-linea'])
                data_aux_original = data_aux_original.merge(auxiliar_max, how='left', on=['chata-linea'])
                                
                if orden==1:
                
                    # Inicializar campos de inicio, stock anterior, stock recalculo
                    data_aux_original['inicio_anterior'] = np.nan
                    data_aux_original['inicio_anterior'] = pd.to_datetime(data_aux_original['inicio_anterior'])
                    data_aux_original['stock_recalculo'] = 0 #np.where(data_aux_original['marea_id']==marea_id,tonelaje,0)
                    data_aux_original['stock_anterior'] = 0
                    data_aux_original['tvn_anterior'] = 0
                    auxiliar['stock_recalculo'] = tonelaje
                    auxiliar['stock_recalculo_opt'] = tonelaje
                    auxiliar['tvn_planta_opt'] = 0
                    auxiliar['tvn_planta_max'] = 0
                    auxiliar['tvn_anterior'] = 0
                    
                    data_aux_original = pd.merge(data_aux_original,auxiliar[['id_planta','chata-linea','pozaname','tvn_planta_opt','tvn_planta_max']],how='left',on=['id_planta','chata-linea','pozaname'])
                                        
                else:
                    
                    # Actualizar el stock anterior con el ultimo stock de la iteracion anterior, actualizar stock actual
                    data_aux_original['stock_anterior'] = data_aux_original['stock_recalculo'].copy()
                    data_aux_original['stock_recalculo'] = np.where(data_aux_original['marea_id']==marea_id,data_aux_original['stock_recalculo'] + tonelaje, data_aux_original['stock_recalculo'])
                    
                    data_aux_orig_red = data_aux_original[['chata-linea','inicio_anterior']].drop_duplicates('chata-linea')
                    auxiliar = auxiliar.merge(data_aux_orig_red, how='left', on='chata-linea')
                    
                    # Calculo de tiempo transcurrido desde descarga anterior
                    auxiliar['tiempo_consumo_cocina_opt'] = (auxiliar['inicio_actual_opt'] - auxiliar['inicio_anterior'])/datetime.timedelta(hours=1)
                    auxiliar['tiempo_consumo_cocina_opt'] =np.where(auxiliar['tiempo_consumo_cocina_opt']<0,0,auxiliar['tiempo_consumo_cocina_opt'])
                    auxiliar['tiempo_consumo_cocina_opt'] = auxiliar['tiempo_consumo_cocina_opt'].fillna(0)
                    auxiliar['consumo_cocina_opt'] = auxiliar['tiempo_consumo_cocina_opt']*auxiliar['velocidad']
                    
                    auxiliar['tiempo_consumo_cocina_max'] = auxiliar['tiempo_consumo_cocina_opt'].copy()
                    auxiliar['consumo_cocina_max'] = auxiliar['consumo_cocina_opt'].copy()
                    
                    # Agregar stock y calcular primero consumo por alimentacion a cocina y despues stock hipotetico de descarga
                    auxiliar = auxiliar.merge(data_aux_original[['marea_id','id_planta','stock_recalculo','stock_anterior','tvn_anterior']].drop_duplicates(),how='left',on=['marea_id','id_planta'])
                    auxiliar['stock_cocina_opt'] = auxiliar['stock_anterior'] - auxiliar['consumo_cocina_opt']
                    auxiliar['stock_cocina_opt'] = np.where(auxiliar['stock_cocina_opt']<0,0,auxiliar['stock_cocina_opt'])
                    data_aux_original['stock_recalculo'] = np.where(data_aux_original['marea_id']==marea_id,data_aux_original['stock_recalculo'] - tonelaje,data_aux_original['stock_recalculo'])
                    auxiliar['stock_recalculo_opt'] = auxiliar['stock_cocina_opt'] + tonelaje
                    
                    auxiliar['stock_cocina_max'] = auxiliar['stock_cocina_opt'].copy()
                    auxiliar['stock_recalculo_max'] = auxiliar['stock_recalculo_opt'].copy()
                    
                    # Calculo tvn ponderado pozas por planta
                    auxiliar['aumento_tvn_pozas_opt'] = auxiliar.apply(get_tvn_increase_in_poza_for_marea_opt, axis=1)
                    auxiliar['tvn_pozas_opt'] = auxiliar['tvn_anterior'] + auxiliar['aumento_tvn_pozas_opt']
                    auxiliar['tvn_planta_opt'] = (auxiliar['tvn_pozas_opt']*auxiliar['stock_cocina_opt'] + auxiliar['tvn_opt']*tonelaje)/auxiliar['stock_recalculo_opt']
                    
                    auxiliar['aumento_tvn_pozas_max'] = auxiliar['aumento_tvn_pozas_opt'].copy() 
                    auxiliar['tvn_pozas_max'] = auxiliar['tvn_pozas_opt'].copy()
                    auxiliar['tvn_planta_max'] = auxiliar['tvn_planta_opt'].copy()
                
                # Calculo del Calidad 
                auxiliar = pd.merge(auxiliar,df_plantas_velocidad_limites[['id','lim_tvn_cocina_super_prime','lim_tvn_cocina_prime']],how='left',left_on='id_planta',right_on='id')
                auxiliar['calidad_planta_opt'] = np.where(auxiliar['tvn_planta_opt']<=auxiliar['lim_tvn_cocina_super_prime'],'SUPER_PRIME','PRIME')
                auxiliar['calidad_planta_opt'] = np.where(auxiliar['tvn_planta_opt']>auxiliar['lim_tvn_cocina_prime'],'OTHERS',auxiliar['calidad_planta_opt'])
                # Penalizacion al acercarse a los limites de tvn de la calidad
                # Para categoria OTHERS se coloca un limite alto (tope_tvn=150)
                auxiliar['proporcion_calidad_opt'] = 1 - (auxiliar['tvn_planta_opt']/tope_tvn)
                auxiliar = pd.merge(auxiliar,df_calidades_precio_venta[['QUALITY_LEVEL','PRICE_PER_TON']],how='left',left_on='calidad_planta_opt',right_on='QUALITY_LEVEL')
                
                 # Calcular penalizacion por llenado de pozas
                # Para caso OPT
                auxiliar = auxiliar.merge(df_capacidad_pozas,on='id_planta')
                auxiliar['proporcion_pozas_opt'] = auxiliar['stock_recalculo_opt']/auxiliar['capacidad_poza']

                auxiliar['exceso_capacidad_opt'] = np.where(auxiliar['proporcion_pozas_opt'] > tope, 0, 1)
                
                # Calculo del tradeoff
                auxiliar['valor_pozas'] = auxiliar['PRICE_PER_TON']*tonelaje*auxiliar['proporcion_calidad_opt'] * auxiliar['exceso_capacidad_opt'] / 4  # Se divide entre el rendimiento
                auxiliar['tradeoff'] = auxiliar['valor_pozas'] - auxiliar['combustible_recom'] 
                auxiliar['stock_recalculo'] = auxiliar['stock_recalculo_opt'].copy()
                auxiliar['tvn_planta'] = auxiliar['tvn_planta_opt'].copy()
                auxiliar['inicio_actual'] = auxiliar['inicio_actual_opt'].copy()
                data_aux_original = pd.merge(data_aux_original,auxiliar[['id_planta','chata-linea','pozaname','inicio_actual']],how='left',on=['id_planta','chata-linea','pozaname'])
                
                resumen_balanceo = consolidado_balanceo[['id_chata', 'id_linea', 'total_ton_declared']].drop_duplicates()
                auxiliar = pd.merge(auxiliar, resumen_balanceo, on=['id_chata', 'id_linea'])
                auxiliar=auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True)
                    
            if len(auxiliar)==0:
                print('No se encontró poza operativa en la planta a la que la embarcación tercera se dirige')
                return recomendaciones,pd.DataFrame(lista_embarcaciones,columns=['orden','boat_name_trim','boat_name','owner_group','declared_ton','tipo_bodega','SPEED_OPT_km','SPEED_MAX_km','marea_id','first_cala_start_date','Latitud','Longitud','Hora_llegada', 'eta','eta_plant','fish_zone_departure_date']),status_pozas_final

            # Auxiliar: Se calcula df_pozas_estado_ Chapar la próxima que esté libre, y esté más cercana al 
            
            if (auxiliar.loc[0,'id_planta'] == auxiliar.loc[0,'eta_plant']) | (auxiliar.loc[0,'eta_plant'] is None) | (auxiliar.loc[0,'eta_plant'] is np.nan):
                if (auxiliar.loc[0, 'declared_ton'] < 100) and ((auxiliar.loc[0,'eta_plant'] is None) | (auxiliar.loc[0,'eta_plant'] is np.nan < 100)):
                    auxiliar = auxiliar.sort_values(['distancia_planta', 'tradeoff'], ascending=[True, False]).reset_index(drop=True)
                
                inicio_descarga=auxiliar.loc[0,'inicio_descarga_vel_recom']
                planta_recomendada=auxiliar.loc[0,'id_planta']
                chata_recomendada=auxiliar.loc[0,'id_chata']
                linea_recomendada=auxiliar.loc[0,'id_linea']
                velocidad_recomendada=auxiliar.loc[0,'velocidad_recomendada']
                hora_llegada=auxiliar.loc[0,'hora_llegada_vel_recom']
                stock_planta = auxiliar.loc[0,'stock_recalculo']
                # tvn_llegada_vel_recom = auxiliar.loc[0,'tvn_llegada_vel_recom']
                tvn_planta = auxiliar.loc[0,'tvn_planta']
                tiempo_descarga=calcular_tiempo_descarga(tonelaje,chata_recomendada,linea_recomendada,df_tiempo_descarga)   
            
            else:
                mask_aux = auxiliar['id_planta'] == auxiliar['eta_plant']
                try:
                    inicio_descarga=auxiliar[mask_aux].sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'inicio_descarga_vel_recom']
                    planta_recomendada=auxiliar[mask_aux].sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'id_planta']
                    chata_recomendada=auxiliar[mask_aux].sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'id_chata']
                    linea_recomendada=auxiliar[mask_aux].sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'id_linea']
                    velocidad_recomendada=auxiliar[mask_aux].sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'velocidad_recomendada']
                    hora_llegada=auxiliar[mask_aux].sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'hora_llegada_vel_recom']
                    stock_planta = auxiliar[mask_aux].sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'stock_recalculo']
                    tvn_planta = auxiliar[mask_aux].sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'tvn_planta']
                    tiempo_descarga=calcular_tiempo_descarga(tonelaje,chata_recomendada,linea_recomendada,df_tiempo_descarga)  
                except:
                    inicio_descarga=auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'inicio_descarga_vel_recom']
                    planta_recomendada=auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'id_planta']
                    chata_recomendada=auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'id_chata']
                    linea_recomendada=auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'id_linea']
                    velocidad_recomendada=auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'velocidad_recomendada']
                    hora_llegada=auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'hora_llegada_vel_recom']
                    stock_planta = auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'stock_recalculo']
                    tvn_planta = auxiliar.sort_values(by='inicio_descarga_vel_recom').reset_index(drop=True).loc[0,'tvn_planta']
                    tiempo_descarga=calcular_tiempo_descarga(tonelaje,chata_recomendada,linea_recomendada,df_tiempo_descarga)  

            # Guardar stocks de plantas mas cercanas segun nueva tabla
            # TODO: La mejor eleccion puede que no coincida con el TOP 3, debido a la restriccion de igualar planta a ETA plant
            if owner=='P':
                mask_1 = auxiliar['id_planta'].isin(df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'PLANTA_1'].tolist())
                mask_2 = auxiliar['id_planta'].isin(df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'PLANTA_2'].tolist())
                mask_3 = auxiliar['id_planta'].isin(df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'PLANTA_3'].tolist())
                try: 
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'stock_poza_planta_1'] = auxiliar[mask_1].reset_index(drop=True).loc[0,'stock_recalculo']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'stock_poza_planta_2'] = auxiliar[mask_2].reset_index(drop=True).loc[0,'stock_recalculo']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'stock_poza_planta_3'] = auxiliar[mask_3].reset_index(drop=True).loc[0,'stock_recalculo']

                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'eta_plant'] = auxiliar[mask_3].reset_index(drop=True).loc[0,'eta_plant']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'eta'] = auxiliar[mask_3].reset_index(drop=True).loc[0,'eta']

                    # df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'tvn_poza_planta_1'] = auxiliar[mask_3].reset_index(drop=True).loc[2, 'stock_recalculo']            
                    
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'valor_marea_1'] = auxiliar[mask_1].reset_index(drop=True).loc[0,'valor_pozas']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'valor_marea_2'] = auxiliar[mask_2].reset_index(drop=True).loc[0,'valor_pozas']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'valor_marea_3'] = auxiliar[mask_3].reset_index(drop=True).loc[0,'valor_pozas']
                    
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'costo_consumo_comb_1'] = auxiliar[mask_1].reset_index(drop=True).loc[0,'combustible_recom']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'costo_consumo_comb_2'] = auxiliar[mask_2].reset_index(drop=True).loc[0,'combustible_recom']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id),'costo_consumo_comb_3'] = auxiliar[mask_3].reset_index(drop=True).loc[0,'combustible_recom']
                                
                    # tdc_anterior_1 = auxiliar.loc[mask_1, 'tiempo_consumo_cocina'].max()
                    # tdc_anterior_2 = auxiliar.loc[mask_2, 'tiempo_consumo_cocina'].max()
                    # tdc_anterior_3 = auxiliar.loc[mask_3, 'tiempo_consumo_cocina'].max()

                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'tvn_poza_planta_1'] = auxiliar[mask_1].reset_index(drop=True).loc[0,'tvn_planta']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'tvn_poza_planta_2'] = auxiliar[mask_2].reset_index(drop=True).loc[0,'tvn_planta']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'tvn_poza_planta_3'] = auxiliar[mask_3].reset_index(drop=True).loc[0,'tvn_planta']
                    
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'tradeoff_1'] = df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'valor_marea_1'] - df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'costo_consumo_comb_1']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'tradeoff_2'] = df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'valor_marea_2'] - df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'costo_consumo_comb_2']
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'tradeoff_3'] = df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'valor_marea_3'] - df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'costo_consumo_comb_3']
                    
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'FEH_ARRIBO_ESTIMADA'] = hora_llegada
                    df_prev_utilidad.loc[(df_prev_utilidad['marea_id']==marea_id), 'FEH_DESCARGA_ESTIMADA'] = inicio_descarga

                except:
                    df_prev_utilidad[['stock_poza_planta_1', 'stock_poza_planta_2', 'stock_poza_planta_3', 'eta_plant', 'valor_marea_1', 'valor_marea_2', 'valor_marea_3']] = np.nan
                    df_prev_utilidad[['costo_consumo_comb_1', 'costo_consumo_comb_2', 'costo_consumo_comb_3', 'tvn_poza_planta_1', 'tvn_poza_planta_2', 'tvn_poza_planta_3']] = np.nan
                    df_prev_utilidad[['tradeoff_1', 'tradeoff_2', 'tradeoff_3', 'FEH_ARRIBO_ESTIMADA', 'FEH_DESCARGA_ESTIMADA']] = np.nan
                
            # Actualizar stock final de planta asignada
            data_aux_original['stock_recalculo'] = np.where((data_aux_original['id_planta']==planta_recomendada),stock_planta,data_aux_original['stock_recalculo'])
            data_aux_original['inicio_anterior'] = np.where((data_aux_original['id_planta']==planta_recomendada),data_aux_original['inicio_actual'],data_aux_original['inicio_anterior']) 
            data_aux_original['tvn_anterior'] = np.where((data_aux_original['id_planta']==planta_recomendada),tvn_planta,data_aux_original['tvn_anterior']) 
            
            # Guardar el resto de opciones para cada iteracion
            opciones_restantes = data_aux_original[(data_aux_original['marea_id']!=marea_id)]
            res_no_optimos.append(opciones_restantes)
            data_aux_original = data_aux_original[data_aux_original['marea_id']!=marea_id]

            status_pozas_final.loc[(status_pozas_final.id_planta==planta_recomendada)&(status_pozas_final.id_chata==chata_recomendada)&(status_pozas_final.id_linea==linea_recomendada),'tiempo_fin_descarga']=inicio_descarga+datetime.timedelta(hours=tiempo_descarga)#+datetime.timedelta(minutes=20)
            # TODO: Validar el tiempo extra adicionado
            data_aux.loc[(data_aux.id_planta==planta_recomendada)&(data_aux.id_chata==chata_recomendada)&(data_aux.id_linea==linea_recomendada),'tiempo_fin_descarga']=inicio_descarga+datetime.timedelta(hours=tiempo_descarga)#+datetime.timedelta(minutes=20)
            # data_aux.loc[(data_aux.id_planta==planta_recomendada), 'tvn_llegada_vel_recom']=inicio_descarga+datetime.timedelta(hours=tiempo_descarga)
            lista_embarcaciones.append([orden,embarcacion_optimizar,nombre_embarcacion,owner,tonelaje,tipo_bodega,speed_opt,speed_max,gph_opt,gph_max,marea_id,first_cala,latitud,longitud,hora_llegada, eta_out, eta_plant, fish_zone_departure_date])
            recomendaciones.append([orden,nombre_embarcacion,marea_id,owner,tonelaje,planta_recomendada,chata_recomendada,linea_recomendada,velocidad_recomendada])
            data_aux=data_aux[data_aux.boat_name_trim!=embarcacion_optimizar].copy()
            mask = (data_plantas_update['id_planta'] == planta_recomendada)
            data_plantas_update.loc[mask, 'num_recoms'] = data_plantas_update.loc[mask, 'num_recoms'] + 1
            orden+=1
        return recomendaciones,pd.DataFrame(lista_embarcaciones,columns=['orden','boat_name_trim','boat_name','owner_group','declared_ton','tipo_bodega','SPEED_OPT_km','SPEED_MAX_km','gph_opt','gph_max','marea_id','first_cala_start_date','Latitud','Longitud','Hora_llegada', 'eta','eta_plant','fish_zone_departure_date']),status_pozas_final,df_prev_utilidad

    def optimizacion_embarcaciones_esperando_descarga(df_embarcaciones_esperando_descarga,df_pozas_estado,timestamp,df_tiempo_descarga,df_plantas_velocidad_limites,df_restricciones,df_embarcaciones_descargando,df_priorizacion_linea,df_chatas_lineas,df_embarcaciones,df_mareas_acodere,df_mareas_cerradas,df_planta_velocidad_anterior, df_requerimiento_plantas, df_master_fajas_original, df_horas_produccion, df_prioridad_pozas, df_velocidad_descarga, df_pozas_ubicacion_capacidad, df_mareas_dia_prod):
        df_pozas_ubicacion_capacidad_copy = df_pozas_ubicacion_capacidad.copy()
        df_pozas_ubicacion_capacidad_copy['chata-linea'] = df_pozas_ubicacion_capacidad_copy['id_chata'] + '-' + df_pozas_ubicacion_capacidad_copy['id_linea']
        df_pozas_ubicacion_capacidad_copy_res = df_pozas_ubicacion_capacidad_copy.groupby(['chata-linea']).agg(POZAS_ASOCIADAS=('poza_number', list)).reset_index()
        mask = df_velocidad_descarga['NOM_CHATA_COMPLETO'] == "CHATA TASA CALLAO"
        df_velocidad_descarga.loc[mask, "NOM_CHATA_COMPLETO"] = "CHATA CHILLON"
        mask = df_velocidad_descarga['NOM_CHATA_COMPLETO'] == "CHATA EX-ABA"
        df_velocidad_descarga.loc[mask, "NOM_CHATA_COMPLETO"] = "CHATA EXABA"
        df_embarcaciones_descargando['discharge_plant_name']=df_embarcaciones_descargando['discharge_plant_name'].replace(to_replace =['PISCO',"PISCO SUR",'PISCO NORTE'],value ="PISCO")
        df_embarcaciones_descargando['discharge_chata_name']=df_embarcaciones_descargando['discharge_chata_name'].replace(to_replace =['CHATA TASA CALLAO'],value ="CHATA CHILLON")
        df_embarcaciones_descargando['discharge_chata_name']=df_embarcaciones_descargando['discharge_chata_name'].replace(to_replace =['CHATA EX-ABA'],value ="CHATA EXABA")
        df_embarcaciones_descargando=df_embarcaciones_descargando.merge(df_chatas_lineas.drop_duplicates(subset=['id_chata'])[['id_chata','name']],how='left',left_on='discharge_chata_name',right_on='id_chata')
        df_embarcaciones_descargando['chata-linea']=df_embarcaciones_descargando['id_chata']+'-'+df_embarcaciones_descargando['discharge_line_name']
        actualmente_descargando=list(df_embarcaciones_descargando['chata-linea'].unique())
        df_embarcaciones_descargando['id_poza'] = df_embarcaciones_descargando['discharge_plant_name'].map(str) + '-' + df_embarcaciones_descargando['discharge_poza_1'].map(str)
        df_actualmente_descargando = df_embarcaciones_descargando[['id_poza', 'chata-linea', 'id_chata', 'discharge_line_name', 'discharge_poza_1']].copy()
        
        df_master_fajas = df_master_fajas_original.copy()
        df_master_fajas['COM_CH_LN_PZ'] = 'CHATA ' + df_master_fajas['NOM_LINEA'] + '-'+ df_master_fajas['NUM_NUM_POZA'].astype(str)
        df_master_fajas['DESC_LINEA_AUTO_BLOQUEAD'] = df_master_fajas['DESC_LINEA_AUTO_BLOQUEAD'].apply(lambda x: x[1:-1].split(','))
        comb_ch_ln_pz = sorted(list(df_master_fajas['COM_CH_LN_PZ'].unique()))

        df_restricciones['id_chata_linea']=df_restricciones.id_chata+'-'+df_restricciones.id_linea
        mask = df_embarcaciones_esperando_descarga['discharge_plant_name'] != df_embarcaciones_esperando_descarga['eta_plant']
        df_embarcaciones_esperando_descarga.loc[mask, 'discharge_plant_name'] = df_embarcaciones_esperando_descarga.loc[mask, 'eta_plant']
        unicas=df_embarcaciones_esperando_descarga['discharge_plant_name'].unique()
        threshold_embarcaciones=5
        threshold_tiempo=1
        df_tiempo_descarga_original = df_tiempo_descarga.copy()

        set_emb_esp_desc=set(df_embarcaciones[df_embarcaciones.marea_status=='ESPERANDO DESCARGA'].marea_id)
        set_mareas_acodere=set(df_mareas_acodere[(df_mareas_acodere.linea_descarga_acodere.notnull()&(df_mareas_acodere.chata_descarga_acodere.notnull())) | (df_mareas_acodere.acodera_chata.notnull())].marea_id)
        interseccion=set_emb_esp_desc.intersection(set_mareas_acodere)
        #dict_bases es un diccionario que contendrá en los values, cómo quedará la base de datos si es que las embarcaciones optimizadas hasta el momento
        #hubieran ido a las chata-linea-pozas indicadas en el key.
        dict_bases=dict()
        orden_optimo=[[]]*len(unicas)
        recomendaciones_pozas=[[]]*len(unicas)
        minimos_tvn=[[]]*len(unicas)
        ordenes_minimos_tvn=[[]]*len(unicas)
        numero_planta=0

        df_fajas_estado = df_pozas_estado[['id_planta', 'poza_number']].drop_duplicates(subset=['id_planta', 'poza_number']) # TODO: Bajar después de habilitado
        # mask = df_fajas_estado['id_planta'] == 'CHIMBOTE'
        # df_fajas_estado = df_fajas_estado[mask]
        df_fajas_estado['id_poza'] = df_fajas_estado['id_planta'] + "-" + df_fajas_estado['poza_number'].astype(int).astype(str)
        df_fajas_estado = pd.merge(df_fajas_estado, df_actualmente_descargando, on='id_poza', how='left')
        df_fajas_estado = pd.merge(df_fajas_estado, df_master_fajas[['NOM_LINEA', 'NUM_NUM_POZA', 'NOM_FAJA', 'DESC_POZAS_AUTO_BLOQUEADAS', 'NOM_FAJA_AUTO_BLOQUEADA', 'DESC_LINEA_AUTO_BLOQUEAD']], how='left', left_on=['chata-linea', 'poza_number'], right_on=['NOM_LINEA', 'NUM_NUM_POZA'])
        # pozas_auto_bloq_all = df_fajas_estado.loc[df_fajas_estado['chata-linea'].notna(), 'pozas_auto_bloqueadas'].tolist()
        # pozas_auto_bloq_list = (chain.from_iterable([literal_eval(i) for i in pozas_auto_bloq_all]))
        df_fajas_estado = pd.merge(df_fajas_estado, df_pozas_ubicacion_capacidad_copy_res, how='left', left_on='chata-linea', right_on='chata-linea')
        df_fajas_estado_res = df_fajas_estado.groupby(['id_planta']).agg(DESC_POZAS_AUTO_BLOQUEADAS=('DESC_POZAS_AUTO_BLOQUEADAS', list), POZAS_ASOCIADAS=('POZAS_ASOCIADAS', list), FAJAS_BLOQUEADAS=('NOM_FAJA', list), LINEAS_BLOQUEDAS=('DESC_LINEA_AUTO_BLOQUEAD', list)).reset_index()

        df_pozas_estado=df_pozas_estado.merge(df_chatas_lineas[['id_planta','id_chata','id_linea','habilitado']],on=['id_planta','id_chata','id_linea'],how='left')
        df_pozas_estado=df_pozas_estado[df_pozas_estado.habilitado==True].reset_index(drop=True).copy()
        df_pozas_estado.loc[(df_pozas_estado.tipo_conservacion.notnull())&(df_pozas_estado.tipo_conservacion!='Frio'),'tipo_conservacion']='Otro'
        df_pozas_estado.loc[(df_pozas_estado.tipo_conservacion=='Frio')&(df_pozas_estado.frio_system_state=='GF'),'tipo_conservacion']='Frio-GF'
        df_pozas_estado.loc[(df_pozas_estado.tipo_conservacion=='Frio')&(df_pozas_estado.frio_system_state=='RC'),'tipo_conservacion']='Frio-RC'
        df_pozas_estado.loc[(df_pozas_estado.tipo_conservacion=='Frio')&(df_pozas_estado.frio_system_state=='IN'),'tipo_conservacion']='Otro'

        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega!='Frio'),'tipo_bodega']='Otro'
        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega=='Frio')&(df_embarcaciones_esperando_descarga.frio_system_state=='GF'),'tipo_bodega']='Frio-GF'
        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega=='Frio')&(df_embarcaciones_esperando_descarga.frio_system_state=='RC'),'tipo_bodega']='Frio-RC'
        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega=='Frio')&(df_embarcaciones_esperando_descarga.frio_system_state=='CH'),'tipo_bodega']='Frio-CH'
        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega=='Frio')&(df_embarcaciones_esperando_descarga.frio_system_state=='IN'),'tipo_bodega']='Otro'
        df_info_opciones_acumulado = None
        df_embarcaciones_pama = pd.DataFrame(columns=['marea_id', 'boat_name', 'toneladas', 'id_planta', 'tdc_pama', 'tvn_pama'])
        df_stock_menor_tres_horas = pd.DataFrame(columns=['planta', 'velocidad', 'orden_inicial', 'orden_final', 'marea_id', 'timestamp', 'tiempo_fin_descarga', 'declared_ton', 'diferencias_vol', 'flag_eps_cons_100'])
        
        # Calcular la capacidad maxima de las pozas disponibles por planta
        df_capacidad_pozas = df_pozas_estado[['id_planta','poza_number','pozaCapacity','nivel']].drop_duplicates()
        df_capacidad_pozas = df_capacidad_pozas.groupby(['id_planta']).agg({'pozaCapacity':'sum','nivel':'sum'})
        df_capacidad_pozas.reset_index(inplace=True)
        df_capacidad_pozas.columns = ['id_planta','capacidad_poza','stock_inicial']
        df_capacidad_pozas['stock_proyectado'] = df_capacidad_pozas['stock_inicial'].copy()
        df_capacidad_pozas['inicio_descarga'] = np.nan
        df_capacidad_pozas['inicio_descarga'] = pd.to_datetime(df_capacidad_pozas['inicio_descarga'])
        df_capacidad_pozas_original = df_capacidad_pozas.copy()
        
        # Completar plantas con velocidades no ingresadas
        df_velocidad = df_planta_velocidad_anterior[['id_planta','velocidad']].copy()
        df_velocidad.columns = ['id','velocidad_alt'] 
        df_plantas_velocidad_limites = df_plantas_velocidad_limites.merge(df_velocidad,on='id')
        df_plantas_velocidad_limites['velocidad'] = np.where(df_plantas_velocidad_limites['velocidad']==0,df_plantas_velocidad_limites['velocidad_alt'],df_plantas_velocidad_limites['velocidad'])
        
        # Esqueleto para info de mareas + flags
        df_embar_espe_desc = df_embarcaciones_esperando_descarga.copy()
        
        # Inicializacion de los flags de los condicionales
        df_embar_espe_desc['bodega_frio'] = 0
        # df_embar_espe_desc['chata_linea_desc'] = 0 # Se activa si hay chatas lineas que no esten ocupadas/deshabilitadas
        df_embar_espe_desc['capacidad_pozas'] = 0 # Se activa si hay pozas que puedan contener el TM de la EP
        df_embar_espe_desc['tipo_poza'] = 0 # Se activa si hay pozas con el mismo tipo de conservacion de la poza
        # df_embar_espe_desc['poza_libre'] = 0 # Se activa si hay pozas libres en el momento actual (esto puede generar conflicto porque se puede descargar en mas de una poza al mismo tiempo)
        df_embar_espe_desc['restriccion_ep'] = 0 # Se activa si es que hay chata-linea libre por restriccion fisica
        df_embar_espe_desc['flag_preservante'] = 1 # Se activa si se cumplen los limites de TDC y TVN (solo de ser necesario)
        df_embar_espe_desc['flag_desp_positivo'] = 0 # Se activa si hay chata-lineas que cumplan con el tipo de sistema absorbente
        df_embar_espe_desc['dif_tdc_ep_poza'] = 0 # Se activa si hay pozas que cumplan la diferencia de TDC entre la marea y poza
        # df_embar_espe_desc['rec_chata_anterior'] = 0 # Se activa si hay chata-lineas distintas a la recomendacion anterior
        df_embar_espe_desc['chata_linea_ingresada'] = 0 # Se activa si la marea ya tiene chata-linea asignada pero sigue esperando descarga
        df_embar_espe_desc['tvn_previo_desc_ep'] = 0 # Estimado TVN previo a la descarga
        df_embar_espe_desc['tvn_poza'] = 0 # Estimado TVN al momento de la captura de datos (no se recalcula luego de cada iteracion)
        df_embar_espe_desc['flag_limite_emb'] = 0 # Se deja de dar la rec luego de 5 EPs
        df_embar_espe_desc['flag_limite_tiempo'] = 0 # Se deja de ejecutar si es que se demora mas de minuto y medio
        df_embar_espe_desc['flag_planta_llena'] = False #False: planta con capacidad, True: planta va a superar la capacidad maxima
        
        listado_plantas_flags = list()
            
        for planta_seleccionada in unicas:
            # df_prioridades=df_priorizacion_linea[df_priorizacion_linea.planta==planta_seleccionada].reset_index(drop=True).copy()
            tvn_minimo=np.inf
            iteraciones_min_tres_horas=0
            iteraciones_min_cinco_horas=0
            pama_detectada=0
            #Creamos dos bases de datos auxiliares para la planta en análisis. Una es el status de la pozas en esta planta (temp_pozas)
            #y la otra es las embarcaciones que están esperando descarga en esa planta (temp_emb), ordenadas por orden de llegada.
            temp_pozas = df_pozas_estado[df_pozas_estado['id_planta']==planta_seleccionada].reset_index(drop=True).copy()
            if len(temp_pozas)==0:
                print('No hay pozas habilitadas en la planta {}, donde hay embarcaciones esperando.'.format(planta_seleccionada))
                continue
            temp_pozas['id_chata_linea'] = temp_pozas['id_chata'] + '-' + temp_pozas['id_linea']
            temp_pozas['id_chata_linea_poza'] = temp_pozas['id_chata'] + '-' + temp_pozas['id_linea']+'-'+temp_pozas['poza_number'].astype(int).astype(str)
            temp_pozas['capacidad_disponible'] = temp_pozas['pozaCapacity'] - temp_pozas['nivel']
            temp_pozas['tdc_minimo_embarcaciones']=np.nan
            dict_bases[planta_seleccionada]=dict()

            temp_emb = df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga['discharge_plant_name']==planta_seleccionada].reset_index(drop=True).copy()
            temp_emb = temp_emb.sort_values('discharge_plant_arrival_date').reset_index(drop=True).copy()

            temp_emb['bodega_frio']=np.where(temp_emb.tipo_bodega.isin(['Frio-RC','Frio-CH']),1,0)
            temp_emb['bodega_frio']=np.where(temp_emb.tipo_bodega.isin(['Frio-GF']),2,temp_emb['bodega_frio'])
            temp_emb['tdc_pama']=(timestamp-temp_emb.first_cala_start_date)/np.timedelta64(1, 'h')
            temp_emb['tvn_pama']=temp_emb.apply(lambda x: get_return_model_polynomial_value(x.tdc_pama,x.bodega_frio),axis=1)
            temp_emb_pama=temp_emb[((temp_emb.bodega_frio!=1)|(temp_emb.owner_group=='T'))&((temp_emb.tvn_pama>55)|(temp_emb.tdc_pama>36))].sort_values(by='discharge_plant_arrival_date').reset_index(drop=True).copy()[['marea_id','boat_name','declared_ton','discharge_plant_name', 'tdc_pama', 'tvn_pama']]
            temp_emb_pama.rename(columns={'declared_ton':'toneladas','discharge_plant_name':'id_planta'},inplace=True)
            
            # Validar si ha habido descargas en el dia de produccion actual
            fecha_analisis = timestamp.date()
            try:
                hora_inicio_hoy = list(df_horas_produccion.loc[(df_horas_produccion.date_production==str(fecha_analisis))&(df_horas_produccion.id_plant==planta_seleccionada),'hora'])[0]
            except:
                hora_inicio_hoy = str(fecha_analisis)
            primera_marea_dia_prod = temp_emb[(temp_emb.discharge_plant_arrival_date>=str(hora_inicio_hoy))&(temp_emb.discharge_plant_name==planta_seleccionada)]
            primera_marea_dia_prod = primera_marea_dia_prod.sort_values(['discharge_plant_arrival_date'])
            n_orden_modificados = 3
            plantas_name_piloto = ['CHIMBOTE', 'SUPE', 'CALLAO', 'MALABRIGO', 'SAMANCO', 'PISCO SUR', 'PISCO', 'ATICO', 'MATARANI', 'VEGUETA']
            if planta_seleccionada in plantas_name_piloto:
                # temp_emb_current = temp_emb.sort_values(['discharge_plant_arrival_date'])
                temp_emb_current = temp_emb.sort_values(['acodera_chata','discharge_chata_name','discharge_plant_arrival_date'])
                
                # Verificar si hay una diferencia mayor a 100 TM entre el top 3 de EPs
                temp_emb_current['tm_lag_orden_1'] = temp_emb_current['declared_ton'].shift(1)
                temp_emb_current['tm_lag_orden_1'] = temp_emb_current['tm_lag_orden_1'].fillna(0)
                temp_emb_current['dif_tm_orden'] = temp_emb_current['tm_lag_orden_1'] - temp_emb_current['declared_ton']
                mask_dif_tm_valida = (temp_emb_current['dif_tm_orden']>=100) & (temp_emb_current['bodega_frio']==0)
                list_mayor_100 = temp_emb_current[mask_dif_tm_valida].index.tolist()
                try:
                    orden_dif_mayor_100 = min(list_mayor_100)
                except:
                    orden_dif_mayor_100 = 100
                
                if ((len(primera_marea_dia_prod) > 0) & (planta_seleccionada in ["CHIMBOTE"]) & (orden_dif_mayor_100<n_orden_modificados)):
                    df_top = temp_emb_current.head(n_orden_modificados).sort_values('declared_ton', ascending=False)
                    df_tail = temp_emb_current[~temp_emb_current.index.isin(df_top.index)]
                    temp_emb_current = pd.concat([df_top, df_tail], axis=0, ignore_index=True)
                temp_emb_current['tiempo_espera'] = timestamp - temp_emb['discharge_plant_arrival_date']
                temp_df_chatas_lineas = df_chatas_lineas.copy()
                to_replace = {'PISCO SUR':'PISCO',
                'MOLLENDO':'MATARANI'}
                # temp_df_chatas_lineas.replace({'id_planta':to_replace}, inplace=True)
                temp_df_chatas_lineas = temp_df_chatas_lineas[temp_df_chatas_lineas['id_planta'] == planta_seleccionada]
                # mask = temp_df_chatas_lineas['habilitado'] == True
                n_lineas = temp_df_chatas_lineas.loc[:,'chatalinea'].nunique()
                # Se considera una linea adicional siempre para permitir hacer mas combinaciones en caso la ultima EP sea de frio
                n_lineas = n_lineas + 1
                temp_emb_current['initial_order'] = np.arange(0, len(temp_emb_current.index), 1)
                ordenes_cola = temp_emb_current.tail(len(temp_emb_current.index) - n_lineas).loc[:, 'initial_order'].tolist()
                temp_emb_current = temp_emb_current.head(n_lineas)
                temp_emb_current['flag_propio'] = 0
                temp_emb_current.loc[temp_emb_current['owner_group'] == 'P', 'flag_propio'] = 1
                temp_emb_current['flag_frio'] = temp_emb_current['bodega_frio'].copy()
                temp_emb_current['flag_frio_lag'] = temp_emb_current['bodega_frio'].shift(-1)
                temp_emb_current['initial_order_lag'] = temp_emb_current['initial_order'].shift(-1)
                temp_emb_current['flag_propio_lag'] = temp_emb_current['flag_propio'].shift(-1)
                
                temp_emb_current['flag_menor_4_horas'] = 0
                temp_emb_current.loc[temp_emb_current['tiempo_espera'].dt.total_seconds() / 3600 < 4, 'flag_menor_4_horas'] = 1

                temp_emb_current['flag_avance'] = np.nan

                temp_emb_current['flag_frio_up'] = temp_emb_current['flag_frio'].shift(1)
                temp_emb_current['flag_menor_4_horas_up'] = temp_emb_current['flag_menor_4_horas'].shift(1)

                mask_t = ((temp_emb_current['flag_propio'].isin([0, 1])) & (temp_emb_current['flag_frio'] == 0)) & (temp_emb_current['flag_menor_4_horas_up'] == 1) & (temp_emb_current['flag_frio_up'] == 1)
                mask_f = (temp_emb_current['flag_frio'] == 1) & (temp_emb_current['flag_menor_4_horas'] == 1) & ((temp_emb_current['flag_frio_lag'] == 0) & (temp_emb_current['flag_propio_lag'].isin([0, 1])))

                # mask_nan = ((temp_emb_current['flag_frio'] == 1) & (temp_emb_current['flag_menor_4_horas'] == 0)) | (((temp_emb_current['flag_frio'] == 0) | (tempemb_current['flag_propio'] == 0)) & ((temp_emb_current['flag_frio_lag'] == 0) | (temp_emb_current['flag_propio_lag'] == 0)))

                # Mareas con fecha acodere ingresada
                marea_asignada = temp_emb_current[(temp_emb_current['acodera_chata'].notna()) | (temp_emb_current['discharge_chata_name'].notna())]
                
                if len(marea_asignada)>0:
                    temp_reducido = temp_emb_current[~temp_emb_current.marea_id.isin(marea_asignada['marea_id'].tolist())]
                    temp_reducido = temp_reducido.reset_index(drop=True)
                    if (len(temp_reducido)>1):
                        mask_t_red = ((temp_reducido['flag_propio'] == 0) | (temp_reducido['flag_frio'] == 0)) & (temp_reducido['flag_menor_4_horas_up'] == 1) & (temp_reducido['flag_frio_up'] == 1)
                        mask_f_red = (temp_reducido['flag_frio'] == 1) & (temp_reducido['flag_menor_4_horas'] == 1) & ((temp_reducido['flag_frio_lag'] == 0) | (temp_reducido['flag_propio_lag'] == 0)) 
                        temp_reducido.loc[mask_t_red, 'flag_avance'] = True
                        temp_reducido.loc[mask_f_red, 'flag_avance'] = False
                        
                        for index, row in temp_emb_current.iterrows():
                            flag_value = temp_reducido.loc[temp_reducido['marea_id']==row['marea_id'],'flag_avance'].tolist()
                            if len(flag_value)>0:
                                temp_emb_current.loc[index,'flag_avance'] = flag_value
                    
                else:
                    temp_emb_current.loc[mask_t, 'flag_avance'] = True
                    temp_emb_current.loc[mask_f, 'flag_avance'] = False
                
                temp_emb_current['new_order'] = temp_emb_current['initial_order'].copy()
                temp_emb_current['new_order'] = np.where(temp_emb_current['flag_avance'] == True, temp_emb_current['new_order'] - 1, np.where(temp_emb_current['flag_avance'] == False, temp_emb_current['new_order'] + 1, temp_emb_current['new_order']))
                
                if len(temp_emb_current) < n_lineas:
                    ordenes_cola = []
                # ordenes_posibles = [temp_emb_current['new_order'].tolist() + ordenes_cola, temp_emb_current['initial_order'].tolist() + ordenes_cola]
                ordenes_posibles = [temp_emb_current['initial_order'].tolist() + ordenes_cola]
                ordenes_posibles = list(map(list, set(map(tuple, ordenes_posibles))))
            else:
                    
                # df_embar_espe_desc = df_embarcaciones_esperando_descarga.copy()
                if len(temp_emb_pama.index)>0:
                    embarcaciones_pama=temp_emb_pama.boat_name.unique()
                    # df_embarcaciones_pama = df_embarcaciones_pama.append(temp_emb_pama, ignore_index=True)
                    df_embarcaciones_pama = pd.concat([df_embarcaciones_pama, temp_emb_pama], ignore_index=True)
                    temp_emb=temp_emb[~temp_emb.boat_name.isin(embarcaciones_pama)].sort_values(by='discharge_plant_arrival_date').reset_index(drop=True).copy()
                    temp_emb_pama=temp_emb_pama[['marea_id','boat_name','toneladas','id_planta']].copy()
                    temp_emb_pama['orden_planta']=list(np.arange(len(temp_emb)+1,len(temp_emb)+len(temp_emb_pama)+1))
                    temp_emb_pama['id_chata']='TDC muy alto descargar en poza para pama'
                    temp_emb_pama['id_linea']=np.nan
                    temp_emb_pama['poza_number']=np.nan
                    temp_emb_pama['orden_chata_linea']=np.nan
                    pama_detectada=1

                ordenes_posibles=[list(range(len(temp_emb)))]

                if (len(temp_emb)==0) & (pama_detectada==1):
                    ordenes_minimos_tvn[numero_planta]=temp_emb_pama
                    minimos_tvn[numero_planta]=temp_pozas
                    continue


                if stock_menor_tres_horas(planta_seleccionada,temp_emb,temp_pozas,df_plantas_velocidad_limites,df_planta_velocidad_anterior, df_requerimiento_plantas)[0]:
                    orden_new = stock_menor_tres_horas(planta_seleccionada,temp_emb,temp_pozas,df_plantas_velocidad_limites,df_planta_velocidad_anterior, df_requerimiento_plantas)[1]
                    orden_initial = stock_menor_tres_horas(planta_seleccionada,temp_emb,temp_pozas,df_plantas_velocidad_limites,df_planta_velocidad_anterior, df_requerimiento_plantas)[2]
                    velocidad_planta = df_requerimiento_plantas.loc[df_requerimiento_plantas.id==planta_seleccionada, 'velocidad'].item()
                    
                    differences_vol =  np.c_[np.array([0]), np.diff(temp_emb['declared_ton']).reshape(1, -1)].flatten()
                    flag_eps_cons_100 = (differences_vol > 100).astype(int)
                    dict_temp = {'planta':planta_seleccionada, 'velocidad':velocidad_planta, 
                    'orden_inicial':orden_initial, 'orden_final':orden_new, 
                    'marea_id':temp_emb['marea_id'].tolist(), 
                    'timestamp':timestamp, 
                    'tiempo_fin_descarga':temp_pozas['tiempo_fin_descarga'].max(),
                    'declared_ton':temp_emb['declared_ton'].tolist(),
                    'diferencias_vol':differences_vol, 'flag_eps_cons_100':flag_eps_cons_100}
                    
                    # df_stock_menor_tres_horas = df_stock_menor_tres_horas.append(pd.DataFrame(dict_temp), ignore_index=True)
                    df_stock_menor_tres_horas = pd.concat([df_stock_menor_tres_horas, pd.DataFrame(dict_temp)], ignore_index=True)
                    ordenes_posibles.append(orden_new)


                if len(generar_cinco_ordenes_posibles(temp_emb,timestamp))>0:
                    ordenes_posibles.extend(generar_cinco_ordenes_posibles(temp_emb,timestamp))

            # Lista para almacenar flags por orden
            listado_flags = list()
            
            norden = 1
            #Para cada uno de los órdenes posibles, se encontrarán las combinaciones posibles de chata-linea-poza a las que pueden ser enviadas las embarcaciones
            for orden_posible in ordenes_posibles:
                
                df_capacidad_pozas = df_capacidad_pozas_original.copy()
                # df_embar_espe_desc['fin_descarga'] = np.nan 
                
                if len(set(orden_posible))!=len(orden_posible):
                    continue
                flag_encontrado1=0
                flag_encontrado2=0
                inicio1=time.time()
                dict_bases[planta_seleccionada][str(orden_posible)]=dict()
                try:
                    tvn_promedio=sum(temp_pozas.tvn*temp_pozas.nivel)/sum(temp_pozas.nivel)
                except:
                    tvn_promedio=0

                dict_bases[planta_seleccionada][str(orden_posible)]['[]']=(temp_pozas,tvn_promedio)

                #Se crea la variable orden actual que contendrá la prioridad de cada una de las embarcaciones, y se ordenarán por orden de prioridad.
                temp_emb = temp_emb.sort_values(['acodera_chata','discharge_chata_name','discharge_plant_arrival_date']).reset_index(drop=True).copy()
                if ((len(primera_marea_dia_prod) > 0) & (planta_seleccionada in ["CHIMBOTE"]) & (orden_dif_mayor_100<n_orden_modificados)):
                    df_top = temp_emb.head(n_orden_modificados).sort_values('declared_ton', ascending=False)
                    df_tail = temp_emb[~temp_emb.index.isin(df_top.index)]
                    temp_emb = pd.concat([df_top, df_tail], axis=0, ignore_index=True)
                temp_emb['orden_actual']=orden_posible
                temp_emb=temp_emb.sort_values(by='orden_actual').reset_index(drop=True)
                
                # Recalcular el orden para el df original
                primera_marea_dia_prod = primera_marea_dia_prod.merge(temp_emb[['marea_id','orden_actual']], how='left', on='marea_id')
                
                count_na = len(primera_marea_dia_prod[primera_marea_dia_prod['orden_actual'].isna()])
                
                if count_na==0:
                    primera_marea_dia_prod = primera_marea_dia_prod.sort_values(['orden_actual']).reset_index(drop=True)
                else:
                    primera_marea_dia_prod['orden_actual'] = primera_marea_dia_prod['orden_actual'].fillna(0)
                
                
                # Agregar esta columna al dataframe de df_embar_esp_desc
                df_embar_espe_desc = df_embar_espe_desc.merge(temp_emb[['marea_id','orden_actual']],how='left',on='marea_id')
                
                #orden_embarcacion será un contador que permitirá identificar el orden de la embarcación que estamos optimizando
                orden_embarcacion=0

                #Posibilidades es inicializada en una lista vacía, pero terminará teniendo todas las posibles combinaciones de órdenes para la planta en análisis
                posibilidades=[[]]

                while orden_embarcacion<len(temp_emb):

                    #La lista auxiliar, contendrá las posibles combinaciones de embarcaciones previas a la que se está buscando optimizar. Por ejemplo,
                    #si se está optimizando la embarcación 4, auxiliar contendrá todas las posibles combinaciones de las primeras 3 embarcaciones
                    auxiliar=posibilidades.copy()

                    #En cada iteración, luego de guardar las posibilidades en auxiliar, posibilidades se regresa a una lista vacía que será llenada
                    #considerando las posibles combinaciones entre auxiliar y las opciones de la embarcación en análisis.
                    posibilidades=[]

                    #Definimos las caracterísitcas (toneladas y tipo de bodega) de la embarcación para la cual evaluaremos las posibilidades 
                    emb_toneladas_actual=temp_emb.loc[orden_embarcacion,'declared_ton']
                    emb_tipo_bodega_actual = temp_emb.loc[orden_embarcacion,'tipo_bodega']
                    bodega_frio_actual=1 if ((emb_tipo_bodega_actual=='Frio-RC')|(emb_tipo_bodega_actual=='Frio-GF')|(emb_tipo_bodega_actual=='Frio-CH')) else 0
                    bodega_frio_actual=2 if (emb_tipo_bodega_actual=='Frio-GF') else bodega_frio_actual
                    emb_first_cala_start_date_actual = temp_emb.loc[orden_embarcacion,'first_cala_start_date']
                    emb_name = temp_emb.loc[orden_embarcacion,'boat_name']
                    emb_propietario_actual=temp_emb.loc[orden_embarcacion,'owner_group']
                    marea_actual=temp_emb.loc[orden_embarcacion,'marea_id']
                    
                    # Identificar cantidad EPs mayor a 18 TDC al arribo
                    # emb_excede_tdc = temp_emb[temp_emb['tdc_pama']>18].reset_index(drop=True)
                    # porc_emb_excede_tdc = len(emb_excede_tdc)/len(temp_emb)
                    
                    #Luego, se procederá a evaluar cuales son las opciones de chata-linea-poza que tiene la embarcación que queremos optimizar. Esto
                    #se hace considerando todas las combinaciones de las embarcaciones anteriores
                    for i in auxiliar:
                        #print(i)
                        fin1=time.time()
                        #temp_pozas_2 contendrá el status de las pozas si efectivamente sucediera la combinación 'i' de las embarcaciones anteriores
                        #Un diccionario no acepta de key una lista, es por eso que se convierten a string
                        temp_pozas2=dict_bases[planta_seleccionada][str(orden_posible)][str(i)][0]
                        temp_pozas2['flag_balanceo_temp'] = 0
                        # NUEVO CALCULO DE BALANCEO DE CHATAS
                        # Se calculo un aproximado de la descarga declarada en cada chata para el balanceo
                        # se toma info de una semana atras y sera la referencia para priorizar por balanceo
                        cerradas = df_mareas_cerradas.copy()
                        # tope_historico = timestamp+datetime.timedelta(days=-18) #cambiar a una semana, solo es pruebas
                        year_temporada = (pd.to_datetime(df_mareas_cerradas['production_date']).max() - datetime.timedelta(days=30)).year
                        fecha_primera_temporada = f'{year_temporada}/03/10'
                        fecha_segunda_temporada = f'{year_temporada}/10/10'

                        dif_1 = (timestamp - pd.to_datetime(fecha_primera_temporada)).days
                        dif_2 = (timestamp - pd.to_datetime(fecha_segunda_temporada)).days

                        if dif_2 < 0:
                            FECHA_INICIO = fecha_primera_temporada
                        else:
                            FECHA_INICIO = fecha_segunda_temporada

                        if dif_1 < 0:
                            FECHA_INICIO = '2019/01/01'

                        cerradas['production_date'] = pd.to_datetime(cerradas['production_date'])
                        filtro_historico = cerradas['production_date']>FECHA_INICIO
                        
                        cerradas = df_mareas_cerradas.loc[filtro_historico & cerradas['discharge_chata_name'].notnull(),
                                                          ['marea_id','discharge_plant_name','discharge_chata_name','discharge_line_name','declared_ton']]
                        cerradas.loc[cerradas['discharge_chata_name']=='TASA CALLAO','discharge_chata_name'] = 'CHILLON'
                        # cerradas['chatalinea'] = cerradas['discharge_chata_name'] + '-' + cerradas['discharge_line_name']
                        cerradas.reset_index(inplace=True)
                        
                        abiertas = df_embarcaciones.loc[(df_embarcaciones.marea_status=='DESCARGANDO')
                                                        | (df_embarcaciones.marea_status=='EP ESPERANDO ZARPE'),
                                                        ['marea_id','discharge_plant_name','discharge_chata_name','discharge_line_name','declared_ton']]
                        # abiertas['chatalinea'] = abiertas['discharge_chata_name'] + '-' + abiertas['discharge_line_name']
                        abiertas.reset_index(inplace=True)
                        mareas_ult_semana = pd.concat([cerradas,abiertas],axis=0)
                        
                        df_descarga_planta = mareas_ult_semana.groupby(['discharge_plant_name','discharge_chata_name'],as_index=False).declared_ton.sum()
                        df_descarga_planta = df_descarga_planta.rename(columns ={'declared_ton':'total_ton_declared'})                        
                        df_descarga_planta['discharge_chata_name'] = str('CHATA ') + df_descarga_planta['discharge_chata_name']
                        try:
                            if (len(df_descarga_planta.loc[df_descarga_planta['discharge_plant_name']=='VEGUETA'])>0) & (len(df_descarga_planta.loc[df_descarga_planta['discharge_chata_name']=='CHATA TASA'])>0):
                                df_descarga_planta.loc[(df_descarga_planta['discharge_plant_name']=='VEGUETA') & (df_descarga_planta['discharge_chata_name']=='CHATA TASA'),'total_ton_declared'] = 2*df_descarga_planta.loc[(df_descarga_planta['discharge_plant_name']=='VEGUETA') & (df_descarga_planta['discharge_chata_name']=='CHATA TASA'),'total_ton_declared']
                        except:
                            pass
                        # print(df_descarga_planta)
                        pozas_disponibles = df_pozas_estado.copy()
                        pozas_disponibles = pozas_disponibles.loc[(pozas_disponibles.id_planta==planta_seleccionada) & (pozas_disponibles.habilitado==True)]
                        pozas_disponibles['poza_number'] = pozas_disponibles['poza_number'].map('{0:g}'.format)
                        pozas_disponibles['chatalineapoza'] = pozas_disponibles['id_chata'] + '-' + pozas_disponibles['id_linea'] + '-' + pozas_disponibles['poza_number']
                        chatalineapoza_posible = pd.DataFrame(pozas_disponibles.chatalineapoza.value_counts().reset_index())
                        chatalineapoza_posible.columns=['chatalineapoza','cuenta']
                        # OJO: Si hay un cambio entre la conexion linea-poza, no se podra recuperar la relacion anterior
                        # en las opciones posibles estas no apareceran
                        chatalineapoza_posible = chatalineapoza_posible.merge(pozas_disponibles[['chatalineapoza','id_chata']],how='left',on='chatalineapoza')
                        consolidado_balanceo = chatalineapoza_posible.merge(df_descarga_planta,how='left',left_on='id_chata',right_on='discharge_chata_name')
                        chatas_posible = consolidado_balanceo['id_chata'].unique()
                        for chata in chatas_posible:
                            consolidado_balanceo['total_ton_declared'] = consolidado_balanceo['total_ton_declared'].fillna(0)
                            max_ton = consolidado_balanceo.loc[consolidado_balanceo['id_chata']==chata,'total_ton_declared'].max()
                            consolidado_balanceo.loc[consolidado_balanceo['id_chata']==chata,'total_ton_declared'] = max_ton
                        consolidado_balanceo['discharge_plant_name']=planta_seleccionada
                        consolidado_balanceo = consolidado_balanceo.sort_values(by='total_ton_declared')
                        consolidado_balanceo = consolidado_balanceo[['chatalineapoza','id_chata','discharge_plant_name','total_ton_declared']]
                        # El balanceo continuara mas abajo
                        
                        # Setear el valor de orden_chata con un valor alto (solo se considera balanceo en determinados casos)
                        orden_chata = 100
                        plantas_piloto_prioridad = ['CALLAO','CHIMBOTE','MALABRIGO','PISCO','PISCO SUR','SAMANCO','SUPE','VEGUETA']
                        # plantas_piloto_prioridad = []
                        
                        if marea_actual in interseccion:
                            chata_mareasacodere=df_mareas_acodere.loc[df_mareas_acodere.marea_id==marea_actual,'chata_descarga_acodere'].values[0]
                            mask = df_embar_espe_desc['marea_id']==marea_actual
                            df_embar_espe_desc.loc[mask,'bodega_frio'] = bodega_frio_actual
                            if (chata_mareasacodere is None) | (chata_mareasacodere is np.nan):
                                temp_pozas3 = temp_pozas2[temp_pozas2['id_planta'] == planta_seleccionada]
                                temp_pozas3 = temp_pozas3.sort_values(['tiempo_fin_descarga'], ascending=False).head(1)
                                chatanombre = temp_pozas3['id_chata'].item()
                                lineanombre = temp_pozas3['id_linea'].item()
                                pozanombre = temp_pozas3['poza_number'].item()
                            else:
                                chatanombre=df_chatas_lineas.loc[df_chatas_lineas.name==chata_mareasacodere,'id_chata'].values[0]
                                lineanombre=df_mareas_acodere.loc[df_mareas_acodere.marea_id==marea_actual,'linea_descarga_acodere'].values[0]
                                pozanombre=df_mareas_acodere.loc[df_mareas_acodere.marea_id==marea_actual,'poza_1'].values[0]  
                                # Marcar flag
                                df_embar_espe_desc.loc[mask,'chata_linea_ingresada'] = 1
                                
                                # opciones = temp_pozas2[(temp_pozas2.id_chata==chatanombre)&(temp_pozas2.id_linea==lineanombre)&(temp_pozas2.poza_number==pozanombre)].sort_values(by='capacidad_disponible', ascending=False).id_chata_linea_poza
                                # opciones = opciones.reset_index(drop=True)
                            
                            # INICIAL, SE MODIFICARA
                            # opciones=[chatanombre+'-'+lineanombre+'-'+str(int(pozanombre))]
                            opciones=[chatanombre+'-'+lineanombre]
                            # opciones_pozas = temp_pozas2[temp_pozas2['id_chata_linea_poza']==opciones[0]].copy()                                                
                            opciones_pozas = temp_pozas2[(temp_pozas2.id_chata==chatanombre)&(temp_pozas2.id_linea==lineanombre)].sort_values(by='capacidad_disponible', ascending=False)
                                    
                            # Puede ocurrir que se haya asignado una chata-linea o poza bloqueada
                            if len(opciones_pozas)>0:
                                # tdc_calculado = float((opciones_pozas['tiempo_fin_descarga'] - emb_first_cala_start_date_actual)/np.timedelta64(1, 'h'))
                                      
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'flag_balanceo'] = 0
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'chata_rec_orden'] = chatanombre
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'linea_rec_orden'] = lineanombre
                                
                                # Recalcular flags y condiciones de pozas
                                temp_pozas_aux = opciones_pozas.copy()
                                temp_pozas2 = temp_pozas_aux.copy()
                                
                                # Condicion 5
                                posibilidades_segun_restriccion=list(df_restricciones[(df_restricciones.boat_name==emb_name)&(df_restricciones.id_planta==planta_seleccionada)].id_chata_linea)
                                if len(posibilidades_segun_restriccion)>0:
                                    condicion5=(~temp_pozas2.id_chata_linea.isin(posibilidades_segun_restriccion))
                                else:
                                    # Si no se cumple, seleccionar todas las filas (CONDICION DUMMY)
                                    condicion5=(temp_pozas2.poza_number>0)
                                
                                temp_pozas_aux = temp_pozas2[condicion5].copy()
                                if len(temp_pozas_aux)==0:
                                    temp_pozas_aux = temp_pozas2.copy()
                                elif len(posibilidades_segun_restriccion)>0:
                                    df_embar_espe_desc.loc[mask,'restriccion_ep'] = 1
                                
                                # Condicion 3:
                                condicion3=((temp_pozas_aux.tipo_conservacion==emb_tipo_bodega_actual)|(temp_pozas_aux.tipo_conservacion.isnull())|(temp_pozas_aux['nivel'] == 0))
                                temp_pozas_bkp = temp_pozas_aux.copy()
                                temp_pozas_aux = temp_pozas_aux[condicion3]
                                if len(temp_pozas_aux)==0:
                                    temp_pozas_aux = temp_pozas_bkp.copy()
                                else:
                                    df_embar_espe_desc.loc[mask,'tipo_poza'] = 1

                                 # Condicion 8
                                temp_pozas_aux['tdc_hipotetico']=(temp_pozas_aux.tiempo_fin_descarga-emb_first_cala_start_date_actual)/np.timedelta64(1, 'h')
                                condicion8=(abs(temp_pozas_aux.tdc_minimo_embarcaciones-temp_pozas_aux.tdc_hipotetico)<=3)|(temp_pozas_aux.tdc_minimo_embarcaciones.isnull())
                                temp_pozas_bkp = temp_pozas_aux.copy()
                                temp_pozas_aux = temp_pozas_aux[condicion8]
                                if len(temp_pozas_aux)==0:
                                    temp_pozas_aux = temp_pozas_bkp.copy()
                                else:
                                    df_embar_espe_desc.loc[mask,'dif_tdc_ep_poza'] = 1
                                
                                
                                # Condicion 2:
                                # condicion2= (temp_pozas_aux.capacidad_disponible>=emb_toneladas_actual)
                                condicion2= (temp_pozas_aux.capacidad_disponible>0)
                                temp_pozas_bkp = temp_pozas_aux.copy()
                                temp_pozas_aux = temp_pozas_aux[condicion2]
                                if len(temp_pozas_aux)==0:
                                    temp_pozas_aux = temp_pozas_bkp.copy()
                                else:
                                    df_embar_espe_desc.loc[mask,'capacidad_pozas'] = 1
                                
                                 # Condicion 6:
                                # Solo se marcara la condicion si es que hay flag de preservante, si no se considera como que no se cumplio (valor 0)
                                condicion6=((temp_pozas_aux.con_hielo)&(((~temp_pozas_aux.tipo_conservacion.isin(['Frio-RC','Frio-GF','Frio-CH']))&(temp_pozas_aux.tdc_hipotetico<=14))|((temp_pozas_aux.tipo_conservacion.isin(['Frio-RC','Frio-GF','Frio-CH']))&(temp_pozas_aux.tdc_hipotetico<=30))))|(temp_pozas_aux.con_hielo==False)
                                temp_pozas_bkp = temp_pozas_aux.copy()
                                temp_pozas_aux = temp_pozas_aux[condicion6]
                                if len(temp_pozas_aux)==0:
                                    temp_pozas_aux = temp_pozas_bkp.copy()
                                    df_embar_espe_desc.loc[mask,'flag_preservante'] = 0
                                
                                # No recomendar la chata anterior
                                # TODO: Evaluar eliminacion
                                try:
                                    anterior=[x[0:x.find('-',x.find('-',0)+1)] for x in [i[-1]]]
                                except:
                                    anterior=[]
                                    
                                if i == []:
                                    anterior=[]
                                else:
                                    anterior=[x[0:x.find('-',x.find('-',0)+1)] for x in [i[-1]]]
                                
                                # condicion9 = ~temp_pozas_aux.id_chata_linea.isin(anterior)
                                # temp_pozas_bkp = temp_pozas_aux.copy()
                                # temp_pozas_aux = temp_pozas_aux[condicion9]
                                # if len(temp_pozas_aux)==0:
                                #     temp_pozas_aux = temp_pozas_bkp.copy()
                                
                                # Condicion 7:
                                # condicion7=~((('Frio' not in emb_tipo_bodega_actual)|(emb_propietario_actual=='T'))&(temp_pozas_aux.tdc_hipotetico>=18)&(temp_pozas_aux.sistema_absorbente!='DesplazamientoPositivo'))
                                sub_cond_1 = (('Frio' not in emb_tipo_bodega_actual) | (emb_propietario_actual == 'T'))
                                sub_cond_2 = (temp_pozas_aux['tdc_hipotetico'] >= 18)
                                condicion7 = np.where(sub_cond_1 & sub_cond_2, (temp_pozas_aux['sistema_absorbente'] == 'DesplazamientoPositivo'), (temp_pozas_aux['sistema_absorbente'].isin(['PresionVacio', 'DesplazamientoPositivo'])))
                                temp_pozas_bkp = temp_pozas_aux.copy()
                                
                                if ((len(temp_pozas_aux)==0) | (list(temp_pozas_aux.loc[condicion7, 'id_chata_linea'].unique()) == anterior)):
                                    temp_pozas_aux = temp_pozas_bkp.copy()
                                else:
                                    temp_pozas_aux = temp_pozas_aux[condicion7]
                                    if len(temp_pozas_aux.index) == 0:
                                        temp_pozas_aux = temp_pozas_bkp.copy()
                                    df_embar_espe_desc.loc[mask,'flag_desp_positivo'] = 1
                                
                                # TODO: Escoger la que se desocupe primero o la que tenga menor capacidad, comparar diferencias de tvn?
                                # Por el momento sera la de menor capacidad disponible
                                opciones_pozas = temp_pozas_aux[temp_pozas_aux['capacidad_disponible']>0].reset_index(drop=True)
                                if len(opciones_pozas) == 0:
                                    opciones_pozas = temp_pozas_aux.copy()
                                opciones_pozas = opciones_pozas.sort_values(by='capacidad_disponible')
                                opciones_pozas = opciones_pozas.head(1)
                                
                                # Flag balanceo
                                # TODO: En verdad deberia leerse el valor previo de este flag ya que la chata-linea ya fue elegida y la EP sigue Esp Desc
                                df_embar_espe_desc['flag_balanceo'] = 0
                        
                                tdc_calculado = float((opciones_pozas['tiempo_fin_descarga'] - emb_first_cala_start_date_actual)/np.timedelta64(1, 'h'))
                                opciones_pozas['tdc_ep'] = tdc_calculado
                                opciones = list(opciones_pozas['id_chata_linea_poza'])
                                pozanombre = int(opciones_pozas['poza_number'])
                                
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tvn_poza'] = float(opciones_pozas[opciones_pozas['id_chata_linea_poza'].isin(opciones)]['tvn'].unique())
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tdc_previo_desc_ep'] = tdc_calculado
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tvn_previo_desc_ep'] = get_return_model_polynomial_value(tdc_calculado,bodega_frio_actual)
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'poza1_rec_orden'] = pozanombre
                            
                            # Si es linea bloqueada, recomendar segun prioridad de descarga
                            # Escoger la combinacion correcta
                            df_chatas_lineas = df_chatas_lineas.sort_values(['id_planta','chatalinea'])
                            chatas_habilitadas = df_chatas_lineas.loc[(df_chatas_lineas['id_planta']==planta_seleccionada) & (df_chatas_lineas['habilitado']),'chatalinea'].tolist()
                            chata_linea_ingresada = chatanombre + '-' + lineanombre
                            
                            # Validar que la chata ingresada no este bloqueada
                            if chata_linea_ingresada not in chatas_habilitadas:
                                
                                # Se cumplira el orden de prioridad de descarga en pozas si es la primera marea del dia de produccion
                                df_prioridad_filtro = df_prioridad_pozas[df_prioridad_pozas['NOM_PLANTA']==planta_seleccionada].reset_index(drop=True)
                                chatas_bloqueadas = len(df_chatas_lineas[(df_chatas_lineas['id_planta']==planta_seleccionada) & (df_chatas_lineas['habilitado']==False)]['chatalinea'].tolist())
                                chatas_totales = df_prioridad_filtro.CTD_LINEAS_LIBRES.max()
                                chatas_disponibles = chatas_totales - chatas_bloqueadas
                                df_prioridad_filtro = df_prioridad_filtro[df_prioridad_filtro['CTD_LINEAS_LIBRES']==chatas_disponibles].reset_index(drop=True)
                                df_prioridad_filtro = df_prioridad_filtro.sort_values(['NUM_COMBINACION','COD_UNICO'])
                                
                                # Agregar linea bloqueada a lista de habilitadas
                                chatas_habilitadas.append(chata_linea_ingresada)
                                chatas_habilitadas = chatas_habilitadas.sort()
                                
                                df_prioridad_filtro['LINEAS_TOTALES'] = [df_prioridad_filtro.loc[df_prioridad_filtro['NUM_COMBINACION']==x,'COD_UNICO'].tolist() for x in df_prioridad_filtro['NUM_COMBINACION']]
                                df_combinacion = df_prioridad_filtro[df_prioridad_filtro['LINEAS_TOTALES'].isin([chatas_habilitadas])].reset_index(drop=True)
                                opciones_pozas = temp_pozas2.copy()
                                
                                # Comienzo iteracion por combinacion
                                for index,row in df_combinacion.iterrows():
                                    for j in range(1,10):
                                        columna_prioridad = 'NUM_PRIORIDAD_POZA_' + str(j)
                                        try:
                                            poza_prioridad = row['COD_UNICO'] + '-' + str(int(row[columna_prioridad]))
                                        except:
                                            poza_prioridad = row['COD_UNICO'] + '-0'
                                        opciones = opciones_pozas.loc[opciones_pozas['id_chata_linea_poza']==poza_prioridad,'id_chata_linea_poza'].reset_index(drop=True)
                                        poza_elegida = opciones_pozas.loc[opciones_pozas['id_chata_linea_poza']==poza_prioridad,'poza_number'].tolist()
                                        
                                        if len(poza_elegida)>0:
                                            break
                                    
                                    if len(poza_elegida)>0:
                                        poza_elegida = int(poza_elegida[0])
                                        break
                                
                                pozanombre = poza_elegida
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'flag_balanceo'] = 0
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'chata_rec_orden'] = chatanombre
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'linea_rec_orden'] = lineanombre
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'poza1_rec_orden'] = pozanombre
                                
                                # TODO: Dejar vacios los flags, crear un nuevo flag para este caso?
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tvn_poza'] = 0
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tdc_previo_desc_ep'] = 0
                                df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tvn_previo_desc_ep'] = 0
                            
                        else:
                            mask = df_embar_espe_desc['marea_id']==marea_actual
                            
                            # Guardar tipo bodega frio
                            df_embar_espe_desc.loc[mask,'bodega_frio'] = bodega_frio_actual
                            
                            # Condicion 5
                            posibilidades_segun_restriccion=list(df_restricciones[(df_restricciones.boat_name==emb_name)&(df_restricciones.id_planta==planta_seleccionada)].id_chata_linea)
                            if len(posibilidades_segun_restriccion)>0:
                                condicion5=(~temp_pozas2.id_chata_linea.isin(posibilidades_segun_restriccion))
                            else:
                                # Si no se cumple, seleccionar todas las filas (CONDICION DUMMY)
                                condicion5=(temp_pozas2.poza_number>0)
                            
                            # temp_pozas_aux = temp_pozas2.copy()
                            temp_pozas_aux = temp_pozas2[condicion5].copy()
                            if len(temp_pozas_aux)==0:
                                temp_pozas_aux = temp_pozas2.copy()
                            elif len(posibilidades_segun_restriccion)>0:
                                df_embar_espe_desc.loc[mask,'restriccion_ep'] = 1
                                
                            # Condicion 1: Deberia autoregularse, no impedir de que den como recs algo distinto a lo anterior recomendado
                            # temp_pozas_bkp = temp_pozas_aux.copy()
                            # temp_pozas_aux = temp_pozas_aux[condicion1]
                            # if len(temp_pozas_aux)==0:
                            #     temp_pozas_aux = temp_pozas_bkp.copy()
                            #     df_embar_espe_desc.loc[mask,'chata_linea_desc'] = 0
                            
                            # Condicion 8
                            temp_pozas_aux['tdc_hipotetico']=(temp_pozas_aux.tiempo_fin_descarga-emb_first_cala_start_date_actual)/np.timedelta64(1, 'h')
                            condicion8=(abs(temp_pozas_aux.tdc_minimo_embarcaciones-temp_pozas_aux.tdc_hipotetico)<=3)|(temp_pozas_aux.tdc_minimo_embarcaciones.isnull())
                            temp_pozas_bkp = temp_pozas_aux.copy()
                            temp_pozas_aux = temp_pozas_aux[condicion8]
                            if len(temp_pozas_aux)==0:
                                temp_pozas_aux = temp_pozas_bkp.copy()
                            else:
                                df_embar_espe_desc.loc[mask,'dif_tdc_ep_poza'] = 1
                                
                            # Condicion 3:
                            condicion3=((temp_pozas_aux.tipo_conservacion==emb_tipo_bodega_actual)|(temp_pozas_aux.tipo_conservacion.isnull())|(temp_pozas_aux['nivel'] == 0))
                            temp_pozas_bkp = temp_pozas_aux.copy()
                            temp_pozas_aux = temp_pozas_aux[condicion3]
                            if len(temp_pozas_aux)==0:
                                temp_pozas_aux = temp_pozas_bkp.copy()
                            else:
                                df_embar_espe_desc.loc[mask,'tipo_poza'] = 1
                            
                            # Condicion 2:
                            # condicion2= (temp_pozas_aux.capacidad_disponible>=emb_toneladas_actual)
                            condicion2= (temp_pozas_aux.capacidad_disponible>0)
                            temp_pozas_bkp = temp_pozas_aux.copy()
                            temp_pozas_aux = temp_pozas_aux[condicion2]
                            if len(temp_pozas_aux)==0:
                                temp_pozas_aux = temp_pozas_bkp.copy()
                            else:
                                df_embar_espe_desc.loc[mask,'capacidad_pozas'] = 1
                                
                            # Condicion 6:
                            # Solo se marcara la condicion si es que hay flag de preservante, si no se considera como que no se cumplio (valor 0)
                            # TODO: Solo deberia activarse si hay pozas con preservante --> Validar
                            condicion6=((temp_pozas_aux.con_hielo)&(((~temp_pozas_aux.tipo_conservacion.isin(['Frio-RC','Frio-GF','Frio-CH']))&(temp_pozas_aux.tdc_hipotetico<=14))|((temp_pozas_aux.tipo_conservacion.isin(['Frio-RC','Frio-GF','Frio-CH']))&(temp_pozas_aux.tdc_hipotetico<=30))))|(temp_pozas_aux.con_hielo==False)
                            temp_pozas_bkp = temp_pozas_aux.copy()
                            temp_pozas_aux = temp_pozas_aux[condicion6]
                            if len(temp_pozas_aux)==0:
                                temp_pozas_aux = temp_pozas_bkp.copy()
                                df_embar_espe_desc.loc[mask,'flag_preservante'] = 0
                            
                            # TODO: Temporal, prueba si hace sentido
                            # No recomendar la chata anterior
                            try:
                                anterior=[x[0:x.find('-',x.find('-',0)+1)] for x in [i[-1]]]
                            except:
                                anterior=[]
                                
                            if i == []:
                                anterior=[]
                            else:
                                anterior=[x[0:x.find('-',x.find('-',0)+1)] for x in [i[-1]]]                                

                            # Condicion 7:
                            # TODO: aplicar si es que mas de la mitad de las embarcaciones cumplen desplazamiento positivo
                            # condicion7=~((('Frio' not in emb_tipo_bodega_actual)|(emb_propietario_actual=='T'))&(temp_pozas_aux.tdc_hipotetico>=18)&(temp_pozas_aux.sistema_absorbente!='DesplazamientoPositivo'))
                            sub_cond_1 = (('Frio' not in emb_tipo_bodega_actual) | (emb_propietario_actual == 'T'))
                            sub_cond_2 = (temp_pozas_aux['tdc_hipotetico'] >= 18)
                            # sub_cond_3 = porc_emb_excede_tdc < 0.5
                            condicion7 = np.where(sub_cond_1 & sub_cond_2, (temp_pozas_aux['sistema_absorbente'] == 'DesplazamientoPositivo'), (temp_pozas_aux['sistema_absorbente'].isin(['PresionVacio', 'DesplazamientoPositivo'])))
                            temp_pozas_bkp = temp_pozas_aux.copy()
                            
                            if ((len(temp_pozas_aux)==0) | (list(temp_pozas_aux.loc[condicion7, 'id_chata_linea'].unique()) == anterior)):
                                temp_pozas_aux = temp_pozas_bkp.copy()
                            else:
                                temp_pozas_aux = temp_pozas_aux[condicion7]
                                if len(temp_pozas_aux.index) == 0:
                                    temp_pozas_aux = temp_pozas_bkp.copy()
                                df_embar_espe_desc.loc[mask,'flag_desp_positivo'] = 1
                            
                            # condicion9 = ~temp_pozas_aux.id_chata_linea.isin(anterior)
                            # temp_pozas_bkp = temp_pozas_aux.copy()
                            # temp_pozas_aux = temp_pozas_aux[condicion9]
                            # if len(temp_pozas_aux)==0:
                            #     temp_pozas_aux = temp_pozas_bkp.copy()
                            
                            
                            # Ya no se aplican las condiciones 1, 9 ni 4
                            # Agregar flag de balanceo, inicializa en 0
                            df_embar_espe_desc['flag_balanceo'] = 0
                            
                            # Aca continua el balanceo
                            # Solo lo aplicaremos para el primer orden
                            # El balanceo debe ser solo para plantas con mas de una chata
                            plantas_chatas = list(['CALLAO','VEGUETA','MALABRIGO','CHIMBOTE'])
    
                            # Listar las chatas ya asignadas a mareas descargando o esperando descarga
                            
                            if (orden_embarcacion==0) & (planta_seleccionada in plantas_chatas):
                                # Esperando Descarga
                                esp_desc_asignado = temp_emb.loc[temp_emb['discharge_plant_name']==planta_seleccionada,['discharge_chata_name','discharge_line_name']]
                                esp_desc_asignado['chata-linea'] = 'CHATA ' + esp_desc_asignado['discharge_chata_name'] + '-' + esp_desc_asignado['discharge_line_name']
                                esp_desc_asignado = esp_desc_asignado['chata-linea'].unique().tolist()
                                # Descargando
                                chatas_descargando = df_embarcaciones_descargando[(df_embarcaciones_descargando['discharge_plant_name']==planta_seleccionada)]['chata-linea'].unique().tolist()
                                chatas_ocupadas = chatas_descargando + esp_desc_asignado
                                chatas_ocupadas = [item for item in chatas_ocupadas if not(pd.isnull(item)) == True]
                                # chatas_libres = temp_pozas_aux[(temp_pozas_aux['id_planta']==planta_seleccionada) & (~temp_pozas_aux['id_chata_linea'].isin(chatas_ocupadas))]['id_chata_linea'].unique().tolist()
                                # Bloqueadas en interfaz y restriccion tabla estatica
                                chatas_bloqueadas = df_chatas_lineas[(df_chatas_lineas['id_planta']==planta_seleccionada) & (df_chatas_lineas['habilitado']==False)]['chatalinea'].tolist()
                                chatas_restricciones = df_restricciones[df_restricciones['boat_name']==emb_name]['id_chata_linea'].tolist()
                                
                                # TODO: Esto queda en stand by, hay casos en los que hay propios y todas las lineas estan bloqueadas para terceros
                                # Listar las chatas reservadas para terceros en planta solo para embarcaciones propias
                                # if emb_propietario_actual=='P':
                                #     lineas_terceros = df_lineas_reservada_terceros.copy()
                                #     lineas_terceros['chata-linea'] = lineas_terceros['id_chata'] + lineas_terceros['id_linea']
                                #     chatas_terceros = lineas_terceros[(lineas_terceros['id_planta']==planta_seleccionada) & (lineas_terceros['reserv_terc']==True)]['chata-linea'].tolist()
                                # else:
                                #     chatas_terceros = []
                                
                                # Listar las chatas lineas por plantas
                                temp_df_chatas_lineas = df_chatas_lineas.copy()
                                to_replace = {'PISCO SUR':'PISCO','MOLLENDO':'MATARANI'}
                                temp_df_chatas_lineas.replace({'id_planta':to_replace}, inplace=True)
                                temp_df_chatas_lineas = temp_df_chatas_lineas[temp_df_chatas_lineas['id_planta'] == planta_seleccionada]
                                n_lineas = temp_df_chatas_lineas['chatalinea'].nunique()
                                
                                chatas_nodisp = chatas_bloqueadas + chatas_restricciones + chatas_ocupadas
                                chatas_nodisp = list(dict.fromkeys(chatas_nodisp))
                                chatas_libres = temp_df_chatas_lineas[~temp_df_chatas_lineas['chatalinea'].isin(chatas_nodisp)]['chatalinea'].tolist()
                                chatas_disp = temp_df_chatas_lineas[temp_df_chatas_lineas['chatalinea'].isin(chatas_libres)]['id_chata'].unique().tolist()
                                
                                
                                # Solo aplicar balanceo si es que hay mas de una linea libre en las plantas con 2 chatas
                                if n_lineas - len(chatas_nodisp)>1:
                                    filtro = consolidado_balanceo[consolidado_balanceo['id_chata'].isin(chatas_disp)]
                                    temp_pozas_aux = temp_pozas_aux.sort_values(['tiempo_fin_descarga'])
                                    temp_pozas_aux['orden_libre'] = temp_pozas_aux.groupby('id')['tiempo_fin_descarga'].rank('dense')
                                    
                                    if len(filtro)>0:
                                        # Ojo es un valor, no una lista
                                        chata_balanceo = filtro.sort_values(['total_ton_declared'])['id_chata'].unique().tolist()[0]
                                        mask = temp_pozas_aux['id_chata'] == chata_balanceo
                                        try:
                                            orden_chata = int(np.min(temp_pozas_aux.loc[mask,'orden_libre']))
                                        except:
                                            pass
                                        if orden_chata==1:
                                            mask_marea = df_embar_espe_desc['marea_id'] == marea_actual
                                            df_embar_espe_desc.loc[mask_marea,'flag_balanceo'] = 1
                                            temp_pozas_aux.loc[mask, 'flag_balanceo_temp'] = 1                                       
                                            opciones_pozas = temp_pozas_aux[mask].copy()
                                        else:
                                            opciones_pozas = temp_pozas_aux.copy()
                                            
                                        # opciones_pozas = temp_pozas_aux[temp_pozas_aux['id_chata']==chata_balanceo]
                                        if (len(opciones_pozas)==0) | (orden_chata>1):
                                            opciones_pozas = temp_pozas_aux.copy()
                                        else:
                                            opciones_pozas = opciones_pozas.reset_index(drop=True)
                                        
                                    else:
                                        opciones_pozas = temp_pozas_aux.copy()
                                else:
                                    opciones_pozas = temp_pozas_aux.copy()
                                
                            else:
                                # Para el orden 2 en adelante, se escogera la chata-linea con pozas de TVN mas cercano al de la EP
                                opciones_pozas = temp_pozas_aux.copy()
                            
                            # Validar si es la primera EP del dia de produccion
                            
                            if len(primera_marea_dia_prod)>0:
                                primera_marea = primera_marea_dia_prod['marea_id'].head(1).tolist()
                            else:
                                primera_marea = 0
                            
                            poza_elegida = 0

                            if (planta_seleccionada in plantas_piloto_prioridad) & (primera_marea==marea_actual):
                                
                                # Se cumplira el orden de prioridad de descarga en pozas
                                df_prioridad_filtro = df_prioridad_pozas[df_prioridad_pozas['NOM_PLANTA']==planta_seleccionada].reset_index(drop=True)
                                chatas_bloqueadas = len(df_chatas_lineas[(df_chatas_lineas['id_planta']==planta_seleccionada) & (df_chatas_lineas['habilitado']==False)]['chatalinea'].tolist())
                                chatas_totales = df_prioridad_filtro.CTD_LINEAS_LIBRES.max()
                                chatas_disponibles = chatas_totales - chatas_bloqueadas
                                df_prioridad_filtro = df_prioridad_filtro[df_prioridad_filtro['CTD_LINEAS_LIBRES']==chatas_disponibles].reset_index(drop=True)
                                
                                # Escoger la combinacion correcta
                                df_chatas_lineas = df_chatas_lineas.sort_values(['id_planta','chatalinea'])
                                chatas_habilitadas = df_chatas_lineas.loc[(df_chatas_lineas['id_planta']==planta_seleccionada) & (df_chatas_lineas['habilitado']),'chatalinea'].tolist()
                                df_prioridad_filtro = df_prioridad_filtro.sort_values(['NUM_COMBINACION','COD_UNICO'])
                                
                                df_prioridad_filtro['LINEAS_TOTALES'] = [df_prioridad_filtro.loc[df_prioridad_filtro['NUM_COMBINACION']==x,'COD_UNICO'].tolist() for x in df_prioridad_filtro['NUM_COMBINACION']]
                                # df_combinacion = df_prioridad_filtro[df_prioridad_filtro['LINEAS_TOTALES'].isin([chatas_habilitadas])].reset_index(drop=True)
                                mask = df_prioridad_filtro['LINEAS_TOTALES'].apply(lambda s: len(set(s) & set(chatas_habilitadas)) > 0)
                                df_combinacion = df_prioridad_filtro[mask].reset_index(drop=True)
                                
                                for index,row in df_combinacion.iterrows():
                                    for j in range(1,10):
                                        columna_prioridad = 'NUM_PRIORIDAD_POZA_' + str(j)
                                        
                                        try:
                                            poza_prioridad = row['COD_UNICO'] + '-' + str(int(row[columna_prioridad]))
                                        except:
                                            poza_prioridad = row['COD_UNICO'] + '-0'
                                        
                                        opciones = opciones_pozas.loc[opciones_pozas['id_chata_linea_poza']==poza_prioridad,'id_chata_linea_poza'].reset_index(drop=True)
                                        poza_elegida = opciones_pozas.loc[opciones_pozas['id_chata_linea_poza']==poza_prioridad,'poza_number'].tolist()
                                        
                                        if len(poza_elegida)>0:
                                            break
                                    
                                    if len(poza_elegida)>0:
                                        poza_elegida = int(poza_elegida[0])
                                        break
                                    
                                if pd.isnull(poza_elegida):
                                    poza_elegida==0
                            
                            # TODO: ALERT: Validar
                            tdc_calculado = (opciones_pozas['tiempo_fin_descarga'] - emb_first_cala_start_date_actual)/np.timedelta64(1, 'h')
                            opciones_pozas['tdc_ep'] = tdc_calculado
                            opciones_pozas['tvn_ep'] = 0                            
                            if poza_elegida==0:
    
                                # Se continua con la logica normal
                                # Aca sigue el procedimiento estandar para los dos casos (primer orden balanceo y no balanceo)
                                # Se pasa a calculuar el nuevo TDC y TVN de la EP
                                tdc_calculado = (opciones_pozas['tiempo_fin_descarga'] - emb_first_cala_start_date_actual)/np.timedelta64(1, 'h')
                                opciones_pozas['tdc_ep'] = tdc_calculado
                                opciones_pozas['tvn_ep'] = 0
                                # TODO: Esto no toma en cuenta si la chata-linea se libera antes (puede decidir hacer esperar una EP hasta que se libere una chata)
                                # TODO: Validar si no se hace un recalculo del orden en ese caso
                                # print(bodega_frio_actual)
                                for index,row in opciones_pozas.iterrows():
                                    tdc_actual = float(row['tdc_ep'])
                                    opciones_pozas.loc[index,'tvn_ep'] = get_return_model_polynomial_value(tdc_actual,bodega_frio_actual)
                                
                                # Calcular diferencia entre TVNs
                                opciones_pozas['diff']=abs(opciones_pozas.tvn-opciones_pozas.tvn_ep)
                                # Ordenar por menor diferencia y mayor capacidad
                                # opciones_pozas = opciones_pozas.sort_values(['diff','nivel'], ascending=[True,False])
                                opciones_pozas = opciones_pozas.sort_values(['tiempo_fin_descarga','diff','nivel'], ascending=[True,True,False]).reset_index(drop=True)
                                
                                if planta_seleccionada == 'CHIMBOTE':
                                    comb_disponibles = set(opciones_pozas['id_chata_linea_poza'].tolist())

                                    mask = df_master_fajas['COM_CH_LN_PZ'].isin(comb_disponibles)
                                    set_auto = set(df_master_fajas.loc[mask, 'DESC_LINEA_AUTO_BLOQUEAD'].explode())

                                    opciones_disp = list(comb_disponibles - set_auto)
                                    mask = opciones_pozas['id_chata_linea_poza'].isin(opciones_disp)
                                    opciones_pozas_copia = opciones_pozas[mask].copy()
                                    if len(opciones_pozas_copia.index) > 0:
                                        opciones = opciones_pozas.loc[mask, 'id_chata_linea_poza'].head(1).tolist()
                                    else:
                                        opciones = opciones_pozas.loc[:, 'id_chata_linea_poza'].head(1).tolist()

                                else:
                                    opciones = opciones_pozas.loc[:, 'id_chata_linea_poza'].head(1).tolist()
                                # else:
                                #     opciones_pozas = opciones_pozas.sort_values(['tiempo_fin_descarga','nivel'], ascending=[True,False])
                                #     opciones = opciones_pozas['id_chata_linea_poza'].head(1).tolist()
                            # Eleccion de opcion final
                            # opciones = opciones_pozas['id_chata_linea_poza'].head(1).tolist()
                            
                            # print(marea_actual)
                            # print(opciones)
                            # Agregar TVN EP y Poza
                            if len(opciones) == 0:
                                opciones = opciones_pozas.loc[:, 'id_chata_linea_poza'].head(1).tolist()
                            
                            df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tdc_previo_desc_ep'] = float(opciones_pozas[opciones_pozas['id_chata_linea_poza'].isin(opciones)]['tdc_ep'].unique())
                            df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tvn_previo_desc_ep'] = float(opciones_pozas[opciones_pozas['id_chata_linea_poza'].isin(opciones)]['tvn_ep'].unique())
                            df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'tvn_poza'] = float(opciones_pozas[opciones_pozas['id_chata_linea_poza'].isin(opciones)]['tvn'].unique())
                            df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'chata_rec_orden'] = opciones_pozas[opciones_pozas['id_chata_linea_poza'].isin(opciones)]['id_chata'].unique().tolist()
                            df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'linea_rec_orden'] = opciones_pozas[opciones_pozas['id_chata_linea_poza'].isin(opciones)]['id_linea'].unique().tolist()
                            df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'poza1_rec_orden'] = opciones_pozas[opciones_pozas['id_chata_linea_poza'].isin(opciones)]['poza_number'].unique().tolist()
                            df_embar_espe_desc['tvn_poza'] = df_embar_espe_desc['tvn_poza'].clip(upper=1200.0, lower=0.0)
                            df_embar_espe_desc['tvn_previo_desc_ep'] = df_embar_espe_desc['tvn_previo_desc_ep'].clip(upper=1200.0, lower=0.0)
                            # TODO: Evaluar eliminacion
                            df_embar_espe_desc['flag_limite_emb'] = np.nan
                            df_embar_espe_desc['flag_limite_tiempo'] = np.nan
                            # Limite de EPs y tiempo de ejecucion
                            # if ((orden_embarcacion>=threshold_embarcaciones)|((fin1-inicio1)>threshold_tiempo)):
                            #     df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'flag_limite_emb'] = orden_embarcacion
                            #     df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'flag_limite_tiempo'] = fin1-inicio1
                            #     temp_pozas_aux['tdc_actual']=(temp_pozas_aux.tiempo_fin_descarga-emb_first_cala_start_date_actual)/np.timedelta64(1, 'h')
                            #     temp_pozas_aux['tvn_actual']=temp_pozas_aux.apply(lambda x: get_return_model_polynomial_value(x.tdc_actual,bodega_frio_actual),axis=1)                    
                            #     temp_pozas_aux['diff']=abs(temp_pozas_aux.tvn-temp_pozas_aux.tvn_actual)
                            #     opciones=temp_pozas_aux.sort_values(by='diff').reset_index(drop=True).loc[0:0].id_chata_linea_poza.unique()                        
                        try:
                            tdc_hipotetico = float(opciones_pozas[opciones_pozas['id_chata_linea_poza'].isin(opciones)]['tdc_ep'])
                        except:
                            tdc_hipotetico = 0
                        limite_tdc_descarga = df_velocidad_descarga['NUM_LIMITE_TDC'].unique().item()
                        #Una vez que se tengan ya todas las opciones de chata-linea-pozas a las que puede ir la embarcación, se calcula cómo quedaría el status de pozas
                        # en el caso que se opte por estas opciones.
                        #print('opciones',opciones)
                        df_info_opciones_acumulado = pd.DataFrame()
                        for z in opciones:
                            #temp_pozas2 sera la base de datos que contenga el status de pozas luego de la descarga de las embarcaciones anteriores.
                            temp_pozas2=dict_bases[planta_seleccionada][str(orden_posible)][str(i)][0].copy()
                            # VALIDAR: Tmb debe considerarse un tiempo extra para el acodere y desacodere de la chata (15 acodere + 10 desacodere)
                            try:
                                temp_pozas2.loc[temp_pozas2.tiempo_fin_descarga.notnull(),'tiempo_fin_descarga'] = temp_pozas2['tiempo_fin_descarga'] + np.timedelta64(minutes=25)
                            except:
                                pass
                            #A partir del string de chata-linea-poza, se obtiene la chata, la linea y la poza.
                            chata=z[0:z.find('-')]
                            linea=z[len(chata)+1:z.find('-',len(chata)+1)]
                            chata_linea=chata+'-'+linea
                            poza=int(z[len(chata)+len(linea)+2:])
                            
                            # Actualizar velocidad de descarga
                            if tdc_hipotetico<=limite_tdc_descarga:
                                vel_rango1 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea) & (df_velocidad_descarga['DES_RANGO_TM']=='<=100'),'NUM_VEL_LIMITE_INF'].item()
                                vel_rango2 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea) & (df_velocidad_descarga['DES_RANGO_TM']=='100-300'),'NUM_VEL_LIMITE_INF'].item()
                                vel_rango3 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea) & (df_velocidad_descarga['DES_RANGO_TM']=='>300'),'NUM_VEL_LIMITE_INF'].item()
                            else:
                                vel_rango1 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea) & (df_velocidad_descarga['DES_RANGO_TM']=='<=100'),'NUM_VEL_LIMITE_SUP'].item()
                                vel_rango2 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea) & (df_velocidad_descarga['DES_RANGO_TM']=='100-300'),'NUM_VEL_LIMITE_SUP'].item()
                                vel_rango3 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea) & (df_velocidad_descarga['DES_RANGO_TM']=='>300'),'NUM_VEL_LIMITE_SUP'].item()
                            
                            df_tiempo_descarga.loc[(df_tiempo_descarga['id_chata']==chata) & (df_tiempo_descarga['id_linea']==linea),'velocidad_0_100_tons'] = vel_rango1
                            df_tiempo_descarga.loc[(df_tiempo_descarga['id_chata']==chata) & (df_tiempo_descarga['id_linea']==linea),'velocidad_100_300_tons'] = vel_rango2
                            df_tiempo_descarga.loc[(df_tiempo_descarga['id_chata']==chata) & (df_tiempo_descarga['id_linea']==linea),'velocidad_300_mas_tons'] = vel_rango3
                            
                            #Se actualiza el tiempo en que la chata-linea se desocupará. Esto se hace sumándole el tiempo que demorará la descarga, al tiempo en que se desocupa la chata-linea.
                            tiempo_fin_descarga=temp_pozas2[temp_pozas2.id_chata_linea_poza==z].tiempo_fin_descarga.reset_index(drop=True)
                            
                            if orden_embarcacion==0:
                                df_capacidad_pozas.loc[df_capacidad_pozas['id_planta']==planta_seleccionada,'inicio_descarga'] = tiempo_fin_descarga
                                tiempo_hasta_descarga = tiempo_fin_descarga - timestamp
                            else:
                                descarga_anterior = df_capacidad_pozas.loc[df_capacidad_pozas['id_planta']==planta_seleccionada,'inicio_descarga'].reset_index(drop=True)
                                if descarga_anterior is None:
                                    descarga_anterior = tiempo_fin_descarga
                                df_capacidad_pozas.loc[df_capacidad_pozas['id_planta']==planta_seleccionada,'inicio_descarga'] = np.maximum(descarga_anterior, tiempo_fin_descarga)
                                tiempo_hasta_descarga = tiempo_fin_descarga - descarga_anterior
                            
                            # Calcular consumo de pozas hasta el maximo inicio de descarga en planta
                            velocidad_planta = df_plantas_velocidad_limites.loc[df_plantas_velocidad_limites['id']==planta_seleccionada,'velocidad'].unique()[0]#.item()
                            tiempo_hasta_descarga = (tiempo_hasta_descarga / np.timedelta64(1, 'h'))
                            consumo_pozas = np.max(tiempo_hasta_descarga*velocidad_planta, 0)
                            df_capacidad_pozas.loc[df_capacidad_pozas['id_planta']==planta_seleccionada,'stock_proyectado'] = max(df_capacidad_pozas.loc[df_capacidad_pozas['id_planta']==planta_seleccionada,'stock_proyectado'].item() - consumo_pozas,0)
                            
                            # Actualizar stock
                            df_capacidad_pozas.loc[df_capacidad_pozas['id_planta']==planta_seleccionada,'stock_proyectado'] = df_capacidad_pozas.loc[df_capacidad_pozas['id_planta']==planta_seleccionada,'stock_proyectado'].item() + emb_toneladas_actual
                            
                            # Marcar flag de planta llena
                            df_capacidad_pozas['planta_llena'] = np.where(df_capacidad_pozas['capacidad_poza']<df_capacidad_pozas['stock_proyectado'],True,False)
                            flag_planta_llena = df_capacidad_pozas.loc[df_capacidad_pozas['id_planta']==planta_seleccionada,'planta_llena'].item()
                            # TODO: Se esta descativando el flag de planta llena, validar por que tanta variabilidad
                            df_embar_espe_desc.loc[df_embar_espe_desc['marea_id']==marea_actual,'flag_planta_llena'] = False
                            
                            # tiempo_fin_descarga= tiempo_fin_descarga + np.timedelta(minutes=25)
                            tiempo_descarga=fin_descarga_temp_value(emb_toneladas_actual,chata,linea,df_tiempo_descarga)
                            #Luego, se calcula el tvn de la embarcación para luego calcular el tvn resultante de la poza.
                            # if len(tiempo_fin_descarga) == 0:
                            #     tiempo_fin_descarga = pd.Series(timestamp)
                            tiempo_fin_descarga = pd.Series(timestamp)
                            tdc=((tiempo_fin_descarga - emb_first_cala_start_date_actual) / np.timedelta64(1, 'h')).values[0]
                            tvn_embarcacion = get_return_model_polynomial_value(tdc,bodega_frio_actual)

                            #Se actualiza la capacidad disponible de la poza al restarle las toneladas de la embarcacion, y se actualiza el tiempo en que se desocupará la poza.
                            temp_pozas2.loc[temp_pozas2.poza_number==poza,'capacidad_disponible']=temp_pozas2.capacidad_disponible-emb_toneladas_actual
                            temp_pozas2['capacidad_disponible'] = np.where(temp_pozas2['capacidad_disponible']<0, 0, temp_pozas2['capacidad_disponible'])
                            temp_pozas2['nivel'] = np.where(temp_pozas2['nivel']>temp_pozas2['pozaCapacity'], temp_pozas2['pozaCapacity'], temp_pozas2['nivel'])
                            temp_pozas2.loc[temp_pozas2.id_chata_linea==chata_linea,'tiempo_fin_descarga']=temp_pozas2.tiempo_fin_descarga+datetime.timedelta(hours=tiempo_descarga)
                            
                            # display = temp_pozas2['tiempo_fin_descarga']
                            # display = pd.to_datetime(temp_pozas2.loc[temp_pozas2.id_chata_linea==chata_linea,'tiempo_fin_descarga'].unique()).item()
                            # print('DISPLAY TIEMPO FIN DE DESCARGA')
                            # print(display)
                            
                            if z in temp_pozas2['id_chata_linea_poza'].unique().tolist():

                            #Si el nivel de la poza es cero, su tipo de enfriamiento será el de la primera embarcación que descargue en ella.
                                if temp_pozas2.loc[temp_pozas2.poza_number==poza].nivel.values[0]==0:
                                    temp_pozas2.loc[temp_pozas2.poza_number==poza,'tipo_conservacion']=emb_tipo_bodega_actual

                            #Por último, se actualiza el nivel y el tvn de la poza                       
                            temp_pozas2.loc[temp_pozas2.poza_number==poza,'nivel']=temp_pozas2.nivel+emb_toneladas_actual
                            temp_pozas2.loc[temp_pozas2.poza_number==poza,'tvn']=(temp_pozas2.nivel*temp_pozas2.tvn+emb_toneladas_actual*tvn_embarcacion)/(temp_pozas2.nivel+emb_toneladas_actual)
                            temp_pozas2.loc[temp_pozas2.poza_number==poza,'tdc_minimo_embarcaciones']=temp_pozas2['tdc_minimo_embarcaciones'].apply(lambda x: np.nanmin([tdc,x]))
                            temp_pozas2['id_orden'] = norden
                            #La base de datos resultante se incluye
                            base_auxiliar_para_tvn=temp_pozas2.drop_duplicates(subset=['id_planta','id_chata_linea_poza']).copy()
                            # Nuevo dataframe para tabla de utilidad
                            mask = base_auxiliar_para_tvn['id_chata_linea_poza'] == z
                            df_info_opciones = base_auxiliar_para_tvn[mask]
                            df_info_opciones['marea_id'] = marea_actual
                            # df_info_opciones_acumulado = df_info_opciones_acumulado.append(df_info_opciones)
                            df_info_opciones_acumulado = pd.concat([df_info_opciones_acumulado, df_info_opciones], ignore_index=True)
                            
                            if sum(base_auxiliar_para_tvn.nivel)==0:
                                tvn_poza_actual=0
                            else:
                                tvn_poza_actual=sum(base_auxiliar_para_tvn.tvn*base_auxiliar_para_tvn.nivel)/sum(base_auxiliar_para_tvn.nivel)
                            dict_bases[planta_seleccionada][str(orden_posible)][str(i+[z])]=(temp_pozas2,tvn_poza_actual)

                            if (tvn_poza_actual<tvn_minimo) & (orden_embarcacion==(len(temp_emb)-1)):
                                orden_final_embarcaciones = temp_emb.sort_values(['acodera_chata','discharge_chata_name','discharge_plant_arrival_date']).reset_index(drop=True).copy()[['marea_id','boat_name','declared_ton']]
                                
                                if ((len(primera_marea_dia_prod) > 0) & (planta_seleccionada in ["CHIMBOTE"]) & (orden_dif_mayor_100<n_orden_modificados)):
                                    df_top = orden_final_embarcaciones.head(n_orden_modificados).sort_values('declared_ton', ascending=False)
                                    df_tail = orden_final_embarcaciones[~orden_final_embarcaciones.index.isin(df_top.index)]
                                    orden_final_embarcaciones = pd.concat([df_top, df_tail], axis=0, ignore_index=True)
                                orden_final_embarcaciones['orden_planta']=[t+1 for t in orden_posible]
                                orden_final_embarcaciones=orden_final_embarcaciones.sort_values(by='orden_planta').reset_index(drop=True).copy()
                                orden_final_embarcaciones['id_planta']=planta_seleccionada
                                orden_final_embarcaciones['chata_linea_poza']=i+[z]
                                orden_final_embarcaciones['id_chata']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[0:x.chata_linea_poza.find('-')],axis=1)
                                orden_final_embarcaciones['id_linea']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[len(x.id_chata)+1:x.chata_linea_poza.find('-',len(x.id_chata)+1)],axis=1)
                                orden_final_embarcaciones['poza_number']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[len(x.id_chata)+len(x.id_linea)+2:],axis=1)
                                orden_final_embarcaciones.rename(columns={'declared_ton':'toneladas'},inplace=True)
                                orden_final_embarcaciones['orden_chata_linea']=orden_final_embarcaciones.groupby(by=['id_chata','id_linea']).cumcount()+1
                                orden_final_embarcaciones.drop(columns=['chata_linea_poza'],inplace=True)
                                # orden_final_embarcaciones=pd.concat([orden_final_embarcaciones,temp_emb_pama]).reset_index(drop=True).copy()
                                ordenes_minimos_tvn[numero_planta]=orden_final_embarcaciones
                                orden_optimo[numero_planta]=orden_posible.copy()
                                recomendaciones_pozas[numero_planta]=i+[z]
                                tvn_minimo=tvn_poza_actual
                                minimos_tvn[numero_planta]=temp_pozas2
                        del dict_bases[planta_seleccionada][str(orden_posible)][str(i)]
                        
                        posibilidades.extend([p[0]+[p[1]] for p in itertools.product([i],opciones)])
                        opciones = list()
                        
                    orden_embarcacion+=1

                #Restricción numero 1
                if planta_seleccionada not in plantas_name_piloto:
                    if (orden_posible==ordenes_posibles[len(ordenes_posibles)-1]) & (flag_encontrado1==0):
                        if iteraciones_min_tres_horas==0:
                            indicador_demora,nuevo_orden=demora_mas_tres_horas(planta_seleccionada,orden_optimo[numero_planta],recomendaciones_pozas[numero_planta],timestamp,temp_emb,temp_pozas,df_tiempo_descarga,df_plantas_velocidad_limites)
                            if indicador_demora==False:
                                flag_encontrado1=1
                            if indicador_demora==True:
                                ordenes_posibles.append(nuevo_orden)
                                iteraciones_min_tres_horas+=1                            
                        else:
                            mejor_nueva_recomendacion=sorted(dict_bases[planta_seleccionada][str(orden_posible)].items(),key=lambda x:x[1][1])[0][0]
                            mejor_nueva_recomendacion=ast.literal_eval(mejor_nueva_recomendacion)
                            indicador_demora,nuevo_orden=demora_mas_tres_horas(planta_seleccionada,orden_posible,mejor_nueva_recomendacion,timestamp,temp_emb,temp_pozas,df_tiempo_descarga,df_plantas_velocidad_limites)
                            if (indicador_demora==False):
                                orden_final_embarcaciones = temp_emb.sort_values('discharge_plant_arrival_date').reset_index(drop=True).copy()[['marea_id','boat_name','declared_ton']]
                                orden_final_embarcaciones['orden_planta']=[t+1 for t in orden_posible]
                                orden_final_embarcaciones=orden_final_embarcaciones.sort_values(by='orden_planta').reset_index(drop=True).copy()
                                orden_final_embarcaciones['id_planta']=planta_seleccionada
                                orden_final_embarcaciones['chata_linea_poza']=mejor_nueva_recomendacion
                                orden_final_embarcaciones['id_chata']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[0:x.chata_linea_poza.find('-')],axis=1)
                                orden_final_embarcaciones['id_linea']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[len(x.id_chata)+1:x.chata_linea_poza.find('-',len(x.id_chata)+1)],axis=1)
                                orden_final_embarcaciones['poza_number']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[len(x.id_chata)+len(x.id_linea)+2:],axis=1)
                                orden_final_embarcaciones.rename(columns={'declared_ton':'toneladas'},inplace=True)
                                orden_final_embarcaciones['orden_chata_linea']=orden_final_embarcaciones.groupby(by=['id_chata','id_linea']).cumcount()+1
                                orden_final_embarcaciones.drop(columns=['chata_linea_poza'],inplace=True)
                                orden_final_embarcaciones=pd.concat([orden_final_embarcaciones,temp_emb_pama]).reset_index(drop=True).copy()
                                ordenes_minimos_tvn[numero_planta]=orden_final_embarcaciones
                                orden_optimo[numero_planta]=orden_posible.copy()
                                recomendaciones_pozas[numero_planta]=i+[z]
                                minimos_tvn[numero_planta]=dict_bases[planta_seleccionada][str(orden_posible)][str(mejor_nueva_recomendacion)][0]
                                flag_encontrado1=1
                            elif indicador_demora & (iteraciones_min_tres_horas<=3):
                                ordenes_posibles.append(nuevo_orden)
                                iteraciones_min_tres_horas+=1
                            else:
                                flag_encontrado1=1

                    #Restricción de retraso más de 5 horas
                    if (flag_encontrado1==1)&(flag_encontrado2==0):
                        pozas_originales=sorted(dict_bases[planta_seleccionada][str(list(range(len(temp_emb))))].items(),key=lambda x:x[1][1])[0][0]
                        pozas_originales=ast.literal_eval(pozas_originales)
                        indicador_retraso,embarcaciones_retraso=retrasadas_mas_de_cinco_horas(orden_optimo[numero_planta],pozas_originales,recomendaciones_pozas[numero_planta],timestamp,temp_emb,temp_pozas,df_tiempo_descarga,df_plantas_velocidad_limites)
                        if indicador_retraso==False:
                            mejor_nueva_recomendacion=sorted(dict_bases[planta_seleccionada][str(orden_posible)].items(),key=lambda x:x[1][1])[0][0]
                            mejor_nueva_recomendacion=ast.literal_eval(mejor_nueva_recomendacion)
                            orden_final_embarcaciones = temp_emb.sort_values('discharge_plant_arrival_date').reset_index(drop=True).copy()[['marea_id','boat_name','declared_ton']]
                            orden_final_embarcaciones['orden_planta']=[t+1 for t in orden_posible]
                            orden_final_embarcaciones=orden_final_embarcaciones.sort_values(by='orden_planta').reset_index(drop=True).copy()
                            orden_final_embarcaciones['id_planta']=planta_seleccionada
                            orden_final_embarcaciones['chata_linea_poza']=mejor_nueva_recomendacion
                            orden_final_embarcaciones['id_chata']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[0:x.chata_linea_poza.find('-')],axis=1)
                            orden_final_embarcaciones['id_linea']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[len(x.id_chata)+1:x.chata_linea_poza.find('-',len(x.id_chata)+1)],axis=1)
                            orden_final_embarcaciones['poza_number']=orden_final_embarcaciones.apply(lambda x: x.chata_linea_poza[len(x.id_chata)+len(x.id_linea)+2:],axis=1)
                            orden_final_embarcaciones.rename(columns={'declared_ton':'toneladas'},inplace=True)
                            orden_final_embarcaciones['orden_chata_linea']=orden_final_embarcaciones.groupby(by=['id_chata','id_linea']).cumcount()+1
                            orden_final_embarcaciones.drop(columns=['chata_linea_poza'],inplace=True)
                            orden_final_embarcaciones=pd.concat([orden_final_embarcaciones,temp_emb_pama]).reset_index(drop=True).copy()
                            ordenes_minimos_tvn[numero_planta]=orden_final_embarcaciones
                            orden_optimo[numero_planta]=orden_posible.copy()
                            recomendaciones_pozas[numero_planta]=i+[z]
                            minimos_tvn[numero_planta]=dict_bases[planta_seleccionada][str(orden_posible)][str(mejor_nueva_recomendacion)][0]
                            flag_encontrado2==1
                        else:
                            for i in embarcaciones_retraso:
                                data_embarcaciones=temp_emb.sort_values(by='discharge_plant_arrival_date').reset_index(drop=True).copy()
                                data_embarcaciones['orden']=orden_optimo[numero_planta]
                                data_embarcaciones=data_embarcaciones.sort_values(by='orden').reset_index(drop=True).copy()
                                orden_actual=data_embarcaciones.loc[data_embarcaciones.boat_name==i,'orden'].values[0]
                                orden_nuevo=data_embarcaciones.loc[(data_embarcaciones.orden<orden_actual)&(~data_embarcaciones.boat_name.isin(embarcaciones_retraso))&(data_embarcaciones.owner_group=='P')].orden.max()
                                embarcacion_switch=data_embarcaciones.loc[data_embarcaciones.orden==orden_nuevo,'boat_name'].values[0]
                                data_embarcaciones.loc[data_embarcaciones.boat_name==embarcacion_switch,'orden']=orden_actual
                                data_embarcaciones.loc[data_embarcaciones.boat_name==i,'orden']=orden_nuevo
                            data_embarcaciones=data_embarcaciones.sort_values(by='discharge_plant_arrival_date').reset_index(drop=True).copy()
                            nuevo_orden=list(data_embarcaciones.orden)
                            ordenes_posibles.append(nuevo_orden)
                    
                # Agregar resultados de orden a lista de dataframes
                filtro_flags = df_embar_espe_desc[df_embar_espe_desc['discharge_plant_name']==planta_seleccionada].copy()
                filtro_flags = filtro_flags.sort_values(['orden_actual'])
                
                # if len(filtro_flags)==len(orden_posible):
                #     filtro_flags['orden_emb'] = orden_posible
                # else:
                #     filtro_flags['orden_emb'] = 0
                
                filtro_flags['orden_emb'] = filtro_flags['orden_actual']
                del filtro_flags['orden_actual']
                del df_embar_espe_desc['orden_actual']
                del primera_marea_dia_prod['orden_actual']
                filtro_flags['id_orden'] = norden
                listado_flags.append(filtro_flags)
                norden = norden + 1
            
            # Agrupar resultados de ordenes por planta
            consolidado_flags = pd.concat(listado_flags)
            listado_plantas_flags.append(consolidado_flags)
            numero_planta+=1
            
        # Consolidar los resultados de los flags de todos los ordenes
        consolidado_flags_total = pd.concat(listado_plantas_flags)
        minimos_tvn=[p for p in minimos_tvn if type(p)!=list]
        
        # Generar lista de tuplas por orden (contiene pozas estado, planta, orden, TVN y recs)
        estado_final_pozas = dict_bases.copy()
        desglose_ordenes = []
        for level1 in estado_final_pozas.keys():
            for level2 in estado_final_pozas[level1].keys():
                for level3 in estado_final_pozas[level1][level2].keys():
                    estado_orden = estado_final_pozas[level1][level2][level3]
                    estado_orden += (level1, level2, level3,)
                    desglose_ordenes.append(estado_orden)
        
        # Acumular la lista de tuplas en un solo dataframe
        consolidado_ordenes = pd.DataFrame()
        for orden in range(0,len(desglose_ordenes)):
            desglose_ordenes[orden][0]['TVN_Ponderado'] = desglose_ordenes[orden][1]
            desglose_ordenes[orden][0]['Planta'] = desglose_ordenes[orden][2]
            desglose_ordenes[orden][0]['Orden'] = desglose_ordenes[orden][3]
            desglose_ordenes[orden][0]['Chata_linea_poza'] = desglose_ordenes[orden][4]
            desglose_ordenes[orden][0]['Opcion'] = orden
            consolidado_ordenes = pd.concat([consolidado_ordenes,desglose_ordenes[orden][0]], axis=0)
    
        # Agregar el TVN Ponderado de pozas al dataframe de salida
        intermedio = consolidado_ordenes[['id_planta','id_orden','TVN_Ponderado']].drop_duplicates()    
        
        df_embar_espe_desc_pama = pd.merge(df_embar_espe_desc, df_embarcaciones_pama, on='marea_id', how='left')
        df_embar_espe_desc_pama['es_pama'] = 0
        df_embar_espe_desc_pama.loc[df_embar_espe_desc_pama['tvn_pama'].notna(), 'es_pama'] = 1
        
        # Agregar al consolidado el flag de pama
        consolidado_flags_total = pd.merge(consolidado_flags_total,df_embar_espe_desc_pama[['marea_id','es_pama']],how='inner',on='marea_id')
        consolidado_flags_total = pd.merge(consolidado_flags_total,intermedio,how='left',left_on=['discharge_plant_name','id_orden'],right_on=['id_planta','id_orden'])
        
        plantas_distintas = list(chain.from_iterable([plant['id_planta'].unique().tolist() for plant in minimos_tvn]))
        ordenes_minimos_tvn=[p for p in ordenes_minimos_tvn if type(p)!=list]
        if len(minimos_tvn)>0:
            # concat_final1=df_pozas_estado[~df_pozas_estado.id_planta.isin(unicas)].copy()
            concat_final1 = df_pozas_estado[~df_pozas_estado['id_planta'].isin(plantas_distintas)].copy()
            concat_final2=pd.concat(minimos_tvn)
            concat_final=pd.concat([concat_final1,concat_final2])
            concat_final.drop(columns=['id_chata_linea','id_chata_linea_poza','capacidad_disponible'],inplace=True)
            if df_info_opciones_acumulado is None:
                df_info_opciones_acumulado = pd.DataFrame()
            return concat_final,pd.concat(ordenes_minimos_tvn).reset_index(drop=True),df_info_opciones_acumulado,consolidado_flags_total
        else:
            print('No hay recomendaciones para hacer')
            cols = ['id', 'id_planta', 'poza_number', 'pozaCapacity', 'nivel', 'tvn','tipo_conservacion', 'con_hielo', 'frio_system_state', 'fin_cocina', 'id_chata', 'id_linea', 'coordslatitude', 'coordslongitude', 'sistema_absorbente', 'lineaname', 'pozaname', 'tiempo_fin_descarga', 'discharge_poza_1', 'id_poza', 'habilitado', 'id_chata_linea', 'id_chata_linea_poza', 'capacidad_disponible', 'tdc_minimo_embarcaciones', 'marea_id']
            return df_pozas_estado[['id', 'id_planta', 'poza_number', 'pozaCapacity', 'nivel', 'tvn','tipo_conservacion', 'id_chata', 'id_linea', 'coordslatitude','coordslongitude', 'lineaname', 'pozaname', 'id_poza','discharge_poza_1', 'tiempo_fin_descarga']],pd.DataFrame(columns=['marea_id', 'boat_name', 'toneladas', 'orden_planta', 'id_planta','id_chata', 'id_linea', 'poza_number', 'orden_chata_linea']),pd.DataFrame(columns=cols),df_embar_espe_desc


    def demora_mas_tres_horas(planta,ordenes,pozas,timestamp,temp_emb,temp_pozas,df_tiempo_descarga,df_plantas_velocidad_limites):
        minimo_requerido=df_plantas_velocidad_limites.loc[df_plantas_velocidad_limites.id==planta,'minimo_arranque'].values[0]
        data_pozas=temp_pozas.copy()
        data_embarcaciones=temp_emb.copy() 
        data_embarcaciones['orden']=ordenes
        data_embarcaciones=data_embarcaciones.sort_values(by='orden').reset_index(drop=True).copy()
        data_embarcaciones['poza_descarga']=pozas
        data_embarcaciones['hora_fin_descarga']=np.nan
        for i in range(len(data_embarcaciones)):
            z=data_embarcaciones.loc[i,'poza_descarga']
            chata=z[0:z.find('-')]
            linea=z[len(chata)+1:z.find('-',len(chata)+1)]
            chata_linea=chata+'-'+linea
            poza=int(z[len(chata)+len(linea)+2:])
            toneladas=data_embarcaciones.loc[i,'declared_ton']
            if len(data_pozas[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea)])>0:
                hora_inicio_descarga=list(data_pozas.loc[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea),'tiempo_fin_descarga'])[0]
                tiempo_descargando=fin_descarga_temp_value(toneladas,chata,linea,df_tiempo_descarga)
                hora_fin=hora_inicio_descarga+datetime.timedelta(hours=tiempo_descargando)
                data_embarcaciones.loc[i,'hora_fin_descarga']=hora_fin
                data_pozas.loc[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea),'tiempo_fin_descarga']=hora_fin
            else:
                hora_inicio_descarga=timestamp
                cols = df_tiempo_descarga.columns
                df_tiempo_dummy = pd.DataFrame(columns=cols)
                tiempo_descargando = fin_descarga_temp_value(toneladas,chata,linea,df_tiempo_dummy)
                hora_fin=hora_inicio_descarga+datetime.timedelta(hours=tiempo_descargando)
                data_embarcaciones.loc[i,'hora_fin_descarga']=hora_fin
                
        hora_limite=timestamp+datetime.timedelta(hours=3)
        nivel_inicial_pozas=data_pozas.groupby('poza_number').first().nivel.sum()
        desembarco_menos_tres_horas=data_embarcaciones[data_embarcaciones.hora_fin_descarga<=hora_limite].declared_ton.sum()
        if ((data_embarcaciones.declared_ton.sum()+nivel_inicial_pozas)>=minimo_requerido) & ((nivel_inicial_pozas+desembarco_menos_tres_horas)<minimo_requerido):
            #ver qué cambio en orden considerar ahora para que se pueda llegar antes de las 3 horas.
            mayor_despues_tres_horas=data_embarcaciones[data_embarcaciones.hora_fin_descarga>hora_limite].declared_ton.max()
            orden_mayor_despues_tres_horas=data_embarcaciones.loc[(data_embarcaciones.declared_ton==mayor_despues_tres_horas)&(data_embarcaciones.hora_fin_descarga>hora_limite),'orden'].values[0]
            menor_antes_tres_horas=data_embarcaciones[data_embarcaciones.hora_fin_descarga<=hora_limite].declared_ton.min()
            if math.isnan(menor_antes_tres_horas):
                return False,None
            orden_menor_antes_tres_horas=data_embarcaciones.loc[(data_embarcaciones.declared_ton==menor_antes_tres_horas)&(data_embarcaciones.hora_fin_descarga<=hora_limite),'orden'].values[0]
            data_embarcaciones.loc[data_embarcaciones.orden==orden_mayor_despues_tres_horas,'orden']=orden_menor_antes_tres_horas
            data_embarcaciones.loc[data_embarcaciones.orden==orden_menor_antes_tres_horas,'orden']=orden_mayor_despues_tres_horas
            nuevo_orden=list(data_embarcaciones.sort_values(by='discharge_plant_arrival_date').orden)
            return True,nuevo_orden
        else:
            return False,None

    def stock_menor_tres_horas(planta,temp_emb,temp_pozas,df_plantas_velocidad_limites,df_planta_velocidad_anterior, df_requerimiento_plantas):
        data_pozas=temp_pozas.copy()
        data_embarcaciones=temp_emb.sort_values(by='discharge_plant_arrival_date').reset_index(drop=True).copy()
        orden_initial = data_embarcaciones['orden']=list(range(len(data_embarcaciones.index)))
        stock_inicial=data_pozas.groupby('poza_number').first().nivel.sum()
        # Inicialmente se escoge como velocidad de cocina la velocidad_por_ton_cocina y no la velocidad de planta
        # velocidad_planta=df_plantas_velocidad_limites.loc[df_plantas_velocidad_limites.id==planta,'velocidad_por_ton_cocina'].values[0]
        velocidad_planta = 0
        # print(velocidad_planta)
        if (velocidad_planta is None) | (velocidad_planta==0):
            velocidad_planta = df_requerimiento_plantas.loc[df_requerimiento_plantas.id==planta,'velocidad'].values[0]
        # print(velocidad_planta)
        if (velocidad_planta is None) | (velocidad_planta==0):
            velocidad_planta = df_planta_velocidad_anterior.loc[df_planta_velocidad_anterior['id_planta']==planta,'velocidad'].values[0]
        # print(velocidad_planta)
        tiempo_stock=stock_inicial/velocidad_planta
        maximo_ton_emb=data_embarcaciones.declared_ton.max()
        try: # TODO: Revisar para los siguientes barcos
            segundo_max_ton_emb=data_embarcaciones.declared_ton.sort_values().iloc[-2]
        except:
            segundo_max_ton_emb=data_embarcaciones.declared_ton.sort_values().iloc[-1]
        if (tiempo_stock<3) & (maximo_ton_emb-segundo_max_ton_emb>=100):
            orden_de_maximo=data_embarcaciones.loc[data_embarcaciones.declared_ton==maximo_ton_emb,'orden'].values[0]
            orden_cambio=max(orden_de_maximo-2,1)
            data_embarcaciones.loc[orden_de_maximo,'orden']=orden_cambio
            data_embarcaciones.loc[orden_cambio,'orden']=orden_de_maximo
            nuevo_orden=list(data_embarcaciones.orden)
            return True, nuevo_orden, orden_initial
        else:
            return False,None, None
            

    def retrasadas_mas_de_cinco_horas(ordenes,pozas_originales,pozas_nuevas,timestamp,temp_emb,temp_pozas,df_tiempo_descarga,df_plantas_velocidad_limites):
        #orden original
        data_pozas=temp_pozas.copy()
        data_embarcaciones=temp_emb.copy() 
        data_embarcaciones=data_embarcaciones.sort_values(by='discharge_plant_arrival_date').reset_index(drop=True).copy()
        data_embarcaciones['poza_descarga']=pozas_originales
        data_embarcaciones['hora_fin_descarga']=np.nan
        data_embarcaciones['hora_inicio_descarga_original']=np.nan
        for i in range(len(data_embarcaciones)):
            z=data_embarcaciones.loc[i,'poza_descarga']
            chata=z[0:z.find('-')]
            linea=z[len(chata)+1:z.find('-',len(chata)+1)]
            chata_linea=chata+'-'+linea
            poza=int(z[len(chata)+len(linea)+2:])
            toneladas=data_embarcaciones.loc[i,'declared_ton']
            if len(data_pozas[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea)])>0:
                hora_inicio_descarga=list(data_pozas.loc[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea),'tiempo_fin_descarga'])[0]
                data_embarcaciones.loc[i,'hora_inicio_descarga_original']=hora_inicio_descarga
                tiempo_descargando=fin_descarga_temp_value(toneladas,chata,linea,df_tiempo_descarga)
                hora_fin=hora_inicio_descarga+datetime.timedelta(hours=tiempo_descargando)
                data_embarcaciones.loc[i,'hora_fin_descarga']=hora_fin
                data_pozas.loc[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea),'tiempo_fin_descarga']=hora_fin
            else:
                hora_inicio_descarga = timestamp
                cols = df_tiempo_descarga.columns
                df_tiempo_dummy = pd.DataFrame(columns=cols)
                data_embarcaciones.loc[i,'hora_inicio_descarga_original']=hora_inicio_descarga
                tiempo_descargando=fin_descarga_temp_value(toneladas,chata,linea,df_tiempo_dummy)
                hora_fin=hora_inicio_descarga+datetime.timedelta(hours=tiempo_descargando)
                data_embarcaciones.loc[i,'hora_fin_descarga']=hora_fin            
        
        data_embarcaciones1=data_embarcaciones[['boat_name','hora_inicio_descarga_original','tipo_bodega','owner_group']].copy()

        #orden nuevo
        data_pozas=temp_pozas.copy()
        data_embarcaciones=temp_emb.copy() 
        data_embarcaciones['orden']=ordenes
        data_embarcaciones=data_embarcaciones.sort_values(by='orden').reset_index(drop=True).copy()
        data_embarcaciones['poza_descarga']=pozas_nuevas
        data_embarcaciones['hora_fin_descarga']=np.nan
        data_embarcaciones['hora_inicio_descarga_nueva']=np.nan
        for i in range(len(data_embarcaciones)):
            z=data_embarcaciones.loc[i,'poza_descarga']
            chata=z[0:z.find('-')]
            linea=z[len(chata)+1:z.find('-',len(chata)+1)]
            chata_linea=chata+'-'+linea
            poza=int(z[len(chata)+len(linea)+2:])
            toneladas=data_embarcaciones.loc[i,'declared_ton']
            if len(data_pozas[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea)])>0:
                hora_inicio_descarga=list(data_pozas.loc[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea),'tiempo_fin_descarga'])[0]
                data_embarcaciones.loc[i,'hora_inicio_descarga_nueva']=hora_inicio_descarga
                tiempo_descargando=fin_descarga_temp_value(toneladas,chata,linea,df_tiempo_descarga)
                hora_fin=hora_inicio_descarga+datetime.timedelta(hours=tiempo_descargando)
                data_embarcaciones.loc[i,'hora_fin_descarga']=hora_fin
                data_pozas.loc[(data_pozas.id_chata==chata)&(data_pozas.id_linea==linea),'tiempo_fin_descarga']=hora_fin
            else:
                hora_inicio_descarga = timestamp
                data_embarcaciones.loc[i,'hora_inicio_descarga_nueva']=hora_inicio_descarga
                cols = df_tiempo_descarga.columns
                df_tiempo_dummy = pd.DataFrame(columns=cols)
                tiempo_descargando=fin_descarga_temp_value(toneladas,chata,linea,df_tiempo_dummy)
                hora_fin=hora_inicio_descarga+datetime.timedelta(hours=tiempo_descargando)
                data_embarcaciones.loc[i,'hora_fin_descarga']=hora_fin
        
        data_embarcaciones2=data_embarcaciones[['boat_name','hora_inicio_descarga_nueva','tipo_bodega','owner_group']].copy()
        data_embarcaciones_final=data_embarcaciones1.merge(data_embarcaciones2,on=['boat_name','tipo_bodega','owner_group'])
        data_embarcaciones_final['diferencia']=(data_embarcaciones_final['hora_inicio_descarga_nueva']-data_embarcaciones_final['hora_inicio_descarga_original']).dt.total_seconds()/3600
        data_embarcaciones_final=data_embarcaciones_final.sort_values(by='diferencia',ascending=False).reset_index(drop=True).copy()
        lista_embarcaciones_cincoh=list(data_embarcaciones_final.loc[(data_embarcaciones_final.diferencia>5)&(data_embarcaciones_final.tipo_bodega.isin(['Frio-RC','Frio-GF','Frio-CH']))&(data_embarcaciones_final.owner_group=='P')].boat_name)
        if len(lista_embarcaciones_cincoh)>0:
            return True,lista_embarcaciones_cincoh
        else:
            return False,None

    def generar_cinco_ordenes_posibles(temp_emb,timestamp):
        ordenes_finales=[]
        data_embarcaciones=temp_emb.copy()
        data_embarcaciones=data_embarcaciones.sort_values(by='discharge_plant_arrival_date').reset_index(drop=True).copy()
        data_embarcaciones['bodega_frio']=np.where(data_embarcaciones.tipo_bodega.isin(['Frio-RC','Frio-GF','Frio-CH']),1,0)
        data_embarcaciones['tdc']=(timestamp-data_embarcaciones.first_cala_start_date)/np.timedelta64(1, 'h')
        data_embarcaciones['tvn']=data_embarcaciones.apply(lambda x: get_return_model_polynomial_value(x.tdc,x.bodega_frio),axis=1)
        data_embarcaciones['orden']=list(range(len(data_embarcaciones)))
        # data_embarcaciones=data_embarcaciones.sort_values(by='tvn').reset_index(drop=True).copy()

        data_embarcaciones2=data_embarcaciones.loc[~((data_embarcaciones.owner_group=='T')|((data_embarcaciones.owner_group=='P')&(data_embarcaciones.frio_system_state=='GF')))].reset_index(drop=True).copy()
        ordenes_ascendentes_tvn=list(data_embarcaciones2.orden)

        if len(ordenes_ascendentes_tvn)>1:
            parte_inferior=ordenes_ascendentes_tvn[0:int(len(ordenes_ascendentes_tvn)/2)]
            parte_superior=ordenes_ascendentes_tvn[int(len(ordenes_ascendentes_tvn)/2):]   
            for m in parte_inferior:
                for n in parte_superior:
                    if (m<n)&(abs(m-n)<=2):
                        orden_candidato=list(range(len(data_embarcaciones))) # TODO: Definir campos que se usan, implementar una nueva lógica de adelanto de barcos.
                        orden_candidato[n]=m
                        orden_candidato[m]=n
                        ordenes_finales.append(orden_candidato)
        if len(ordenes_finales)>0:
            ordenes_finales=random.sample(ordenes_finales,min(len(ordenes_finales),5)) # TODO: Repensar
        return ordenes_finales
    
    def recomendacion_final_velocidades(df_sol_opt,df_embarcaciones,df_embarcaciones_retornando,df_embarcaciones_terceros,df_pozas_estado_esp_desc,df_requerimiento_plantas,df_tiempo_descarga,timestamp):
        a1=df_sol_opt[df_sol_opt.orden_descarga_global.isnull()].copy()
        a2=a1.merge(df_embarcaciones_retornando[['marea_id','SPEED_OPT_km','SPEED_MAX_km','owner_group','Latitud','Longitud','fish_zone_departure_date','declared_ton', 'eta_plant']],on='marea_id',how='left')
        a3=a2.merge(df_embarcaciones_terceros[['marea_id','eta','declared_ton']],on='marea_id',how='left')
        a3['declared_ton']=np.where(a3['declared_ton_x'].isnull(),a3['declared_ton_y'],a3['declared_ton_x'])
        a4=a3.merge(df_pozas_estado_esp_desc[['id_planta','id_chata','id_linea','coordslatitude','coordslongitude','tiempo_fin_descarga']].drop_duplicates(keep='first'),how='left',left_on=['planta_retorno','chata_descarga','linea_descarga'],right_on=['id_planta','id_chata','id_linea'])
        a5=a4.merge(df_requerimiento_plantas[['id','hora_inicio']],how='left',left_on='id_planta',right_on='id')
        a5['hora_inicio']=np.where((a5.fish_zone_departure_date.dt.hour>=12)&
                                    (a5.fish_zone_departure_date.dt.date>a5.hora_inicio.dt.date),
                                    a5['hora_inicio']+datetime.timedelta(days=1),a5['hora_inicio'])
        a5['hora_inicio']=np.where((a5.fish_zone_departure_date.dt.hour<12)&
                                    (a5.fish_zone_departure_date.dt.date==a5.hora_inicio.dt.date),
                                    a5['hora_inicio']+datetime.timedelta(days=-1),a5['hora_inicio'])

        mask = (a5['eta_plant'] != a5['planta_retorno'])
        a5.loc[mask, 'planta_retorno'] = a5.loc[mask, 'eta_plant']
        del a5['eta_plant']
        # a5['coordslatitude'].fillna(a5['Latitud'], inplace=True)
        # a5['coordslongitude'].fillna(a5['Longitud'], inplace=True)  
        try:
            a5.loc[a5.owner_group=='P','llegada_vel_opt']=a5.loc[a5.owner_group=='P'].apply(lambda x: tiempo_a_plantas(x.planta_retorno,x.marea_id,x.SPEED_OPT_km,df_embarcaciones,df_pozas_estado_esp_desc,timestamp),axis=1)
            a5.loc[a5.owner_group=='P','llegada_vel_max']=a5.loc[a5.owner_group=='P'].apply(lambda x: tiempo_a_plantas(x.planta_retorno,x.marea_id,x.SPEED_MAX_km,df_embarcaciones,df_pozas_estado_esp_desc,timestamp),axis=1)
            a5.loc[a5.owner_group=='P','distancia_planta']=a5.loc[a5.owner_group=='P'].apply(lambda x: distancia(x.coordslatitude,x.coordslongitude,x.Latitud,x.Longitud),axis=1)
        except:
            pass
        a5.loc[a5.owner_group.isnull(),'hora_llegada']=a5.loc[a5.owner_group.isnull(),'eta']
        a5.loc[(a5.owner_group=='P')&(a5.velocidad_retorno=='MAX'),'hora_llegada']=a5.loc[(a5.owner_group=='P')&(a5.velocidad_retorno=='MAX'),'llegada_vel_max']
        a5.loc[(a5.owner_group=='P')&(a5.velocidad_retorno=='OPT'),'hora_llegada']=a5.loc[(a5.owner_group=='P')&(a5.velocidad_retorno=='OPT'),'llegada_vel_opt']
        recomendaciones_velocidad_final=[]
        while len(a5)>0:
            a5=a5.sort_values(by='hora_llegada').reset_index(drop=True)
            marea_optimizar=a5.loc[0,'marea_id']
            owner=a5.loc[0,'owner_group']
            eta=a5.loc[0,'eta']
            planta=a5.loc[0,'planta_retorno']
            chata=a5.loc[0,'chata_descarga']
            linea=a5.loc[0,'linea_descarga']
            ton=a5.loc[0,'declared_ton']
            llegada_vel_opt=a5.loc[0,'llegada_vel_opt']
            llegada_vel_max=a5.loc[0,'llegada_vel_max']
            hora_inicio=a5.loc[0,'hora_inicio']
            tfd=a5.loc[0,'tiempo_fin_descarga']
            descarga_vel_opt=max(llegada_vel_opt,hora_inicio,tfd)
            descarga_vel_max=max(llegada_vel_max,hora_inicio,tfd)
            if descarga_vel_max<descarga_vel_opt:
                if owner=='P': 
                    velocidad_recomendada='MAX'
                    inicio_descarga=descarga_vel_max 
                else: 
                    velocidad_recomendada=np.nan
                    inicio_descarga=eta
            else:
                if owner=='P':
                    velocidad_recomendada='OPT'
                    inicio_descarga=descarga_vel_max 
                else: 
                    velocidad_recomendada=np.nan
                    inicio_descarga=eta
            recomendaciones_velocidad_final.append([marea_optimizar,velocidad_recomendada])
            tiempo_descarga=calcular_tiempo_descarga(ton,chata,linea,df_tiempo_descarga)
            a5.loc[((a5.linea_descarga)==linea)&((a5.chata_descarga)==chata),'tiempo_fin_descarga']=inicio_descarga+datetime.timedelta(hours=tiempo_descarga)+datetime.timedelta(minutes=20)
            a5=a5[a5.marea_id!=marea_optimizar].copy()
        bd_final=pd.DataFrame(recomendaciones_velocidad_final,columns=['marea','velocidad'])
        return bd_final[bd_final.velocidad.notnull()].reset_index(drop=True)

    def recomendar_mas_de_una_poza(df_pozas_estado_original,df_pozas_estado_esp_desc,df_soluciones_optimizacion,df_embarcaciones_esperando_descarga,df_tiempo_descarga):

        #Las variables discharge_poza_1, sistema_absorbente se pueden obtener de df_pozas_estado
        df_pozas_estado_original.loc[(df_pozas_estado_original.tipo_conservacion.notnull())&(df_pozas_estado_original.tipo_conservacion!='Frio'),'tipo_conservacion']='Otro'
        df_pozas_estado_original.loc[(df_pozas_estado_original.tipo_conservacion=='Frio')&(df_pozas_estado_original.frio_system_state=='GF'),'tipo_conservacion']='Frio-GF'
        df_pozas_estado_original.loc[(df_pozas_estado_original.tipo_conservacion=='Frio')&(df_pozas_estado_original.frio_system_state=='RC'),'tipo_conservacion']='Frio-RC'
        df_pozas_estado_original.loc[(df_pozas_estado_original.tipo_conservacion=='Frio')&(df_pozas_estado_original.frio_system_state=='IN'),'tipo_conservacion']='Otro'
        df_pozas_estado_original['capacidad_disponible']=df_pozas_estado_original['pozaCapacity']-df_pozas_estado_original['nivel']
        df_pozas_estado_original['id_chata_linea_poza']=df_pozas_estado_original['id_chata'] + '-' + df_pozas_estado_original['id_linea']+'-'+df_pozas_estado_original['poza_number'].astype(int).astype(str)
        df_pozas_estado_original['id_chata_linea']=df_pozas_estado_original['id_chata'] + '-' + df_pozas_estado_original['id_linea']
        df_pozas_estado_original['tdc_minimo_embarcaciones']=np.nan

        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega=='Frio')&(df_embarcaciones_esperando_descarga.frio_system_state=='GF'),'tipo_bodega']='Frio-GF'
        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega=='Frio')&(df_embarcaciones_esperando_descarga.frio_system_state=='RC'),'tipo_bodega']='Frio-RC'
        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega=='Frio')&(df_embarcaciones_esperando_descarga.frio_system_state=='CH'),'tipo_bodega']='Frio-CH'
        df_embarcaciones_esperando_descarga.loc[(df_embarcaciones_esperando_descarga.tipo_bodega=='Frio')&(df_embarcaciones_esperando_descarga.frio_system_state=='IN'),'tipo_bodega']='Otro'
        df_embarcaciones_esperando_descarga['bodega_frio']=np.where(df_embarcaciones_esperando_descarga.tipo_bodega.isin(['Frio-GF','Frio-RC','Frio-CH']),1,0)

        # df_pozas_estado_original es el estado de las pozas luego de la optimizacion de emabarcaciones descargando
        lista_plantas=df_soluciones_optimizacion[df_soluciones_optimizacion.poza_number.notnull()].id_planta.unique()
        nuevas_filas=pd.DataFrame()
        for i in lista_plantas:
            embarcaciones_planta=df_soluciones_optimizacion[(df_soluciones_optimizacion.poza_number.notnull())&
                                                            (df_soluciones_optimizacion.id_planta==i)].sort_values(by='orden_planta').copy()
            embarcaciones_planta.reset_index(inplace=True, drop=True)                                         
            for index,row in embarcaciones_planta.iterrows():
                try:
                    int(row.poza_number)
                except:
                    continue
                toneladas_embarcacion=row.toneladas
                marea=row.marea_id
                poza=int(row.poza_number)
                poza_1=poza
                try:
                    capacidad_poza=df_pozas_estado_original[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i)].capacidad_disponible.values[0]
                    capacidad_total_poza=df_pozas_estado_original[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i)].pozaCapacity.values[0]
                except:
                    capacidad_poza=df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)].capacidad_disponible.values[0]
                    capacidad_total_poza=df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)].pozaCapacity.values[0]                    
                tipo_bodega=df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga.marea_id==marea].tipo_bodega.values[0]
                # bodega_frio=1 if ((tipo_bodega=='Frio-RC')|(tipo_bodega=='Frio-GF')|(tipo_bodega=='Frio-CH')) else 0
                bodega_frio=1 if ((tipo_bodega=='Frio-RC')|(tipo_bodega=='Frio-CH')) else 0
                bodega_frio=2 if (tipo_bodega=='Frio-GF') else bodega_frio
                #Se obtiene el nombre de la chata y linea que se le recomendó
                chata=row.id_chata
                linea=row.id_linea
                chata_linea=chata+'-'+linea
                chata_linea_poza=chata+'-'+linea+'-'+str(poza)
                if toneladas_embarcacion<=capacidad_total_poza:
                    #Si la recomendación inicial, que siempre es de una poza, es correcta en temas de capacidad (es decir que la poza si
                    #tenía capacidad para recibir toda la carga de la embarcación), se procede a actualizar df_pozas_estado.

                    #Luego, se calcula la hora en la que empezará a descargar y cuanto demorará en hacerlo
                    tiempo_fin_descarga=df_pozas_estado_original[df_pozas_estado_original.id_chata_linea==chata_linea].tiempo_fin_descarga
                    tiempo_descarga=fin_descarga_temp_value(toneladas_embarcacion,chata,linea,df_tiempo_descarga)
                    
                    #Luego se calcula el TVN  de la embarcación, esto a través de calcular primero el tdc.
                    first_cala_start_date=df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga.marea_id==marea].first_cala_start_date.values[0]
                    tdc=((tiempo_fin_descarga-first_cala_start_date)/np.timedelta64(1,'h')).values[0]
                    tvn_embarcacion = get_return_model_polynomial_value(tdc,bodega_frio)
                    
                    #Luego se actualiza la capacidad disponible de las pozas
                    df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'capacidad_disponible']=capacidad_poza-toneladas_embarcacion
                    
                    #Se actualiza el tiempo en que se desocupará la chata linea
                    df_pozas_estado_original.loc[df_pozas_estado_original.id_chata_linea==chata_linea,'tiempo_fin_descarga']=tiempo_fin_descarga+datetime.timedelta(hours=tiempo_descarga)
                    
                    
                    #Si el nivel de la poza es cero, su tipo de enfriamiento será el de la primera embarcación que descargue en ella.
                    try:
                        if df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i)].nivel.values[0]==0:
                            df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'tipo_conservacion']=tipo_bodega
                    except:
                        pass
                    #Se actualiza el nivel y el tvn de la poza
                    df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'nivel']=df_pozas_estado_original.nivel+toneladas_embarcacion

                    df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'tvn']=(df_pozas_estado_original.nivel*df_pozas_estado_original.tvn+toneladas_embarcacion*tvn_embarcacion)/(df_pozas_estado_original.nivel+toneladas_embarcacion)
                    df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'tdc_minimo_embarcaciones']=df_pozas_estado_original['tdc_minimo_embarcaciones'].apply(lambda x: np.nanmin([tdc,x]))
                else:
                    #En caso no haya capacidad suficiente en la poza para recibir la descarga entera de la embarcación, se descarga
                    #en la poza recomendada todo lo que se pueda y luego se eligen otras pozas.
                    toneladas_depositadas=0
                    iteraciones=0

                    while (toneladas_depositadas<toneladas_embarcacion)&(iteraciones<3):
                        toneladas_a_depositar=np.minimum(capacidad_total_poza,toneladas_embarcacion-toneladas_depositadas)
                        tiempo_fin_descarga=df_pozas_estado_original[df_pozas_estado_original.id_chata_linea==chata_linea].tiempo_fin_descarga
                        tiempo_descarga=fin_descarga_temp_value(toneladas_a_depositar,chata,linea,df_tiempo_descarga)
                        first_cala_start_date=df_embarcaciones_esperando_descarga[df_embarcaciones_esperando_descarga.marea_id==marea].first_cala_start_date.values[0]
                        tdc=((tiempo_fin_descarga-first_cala_start_date)/np.timedelta64(1,'h')).values[0]
                        tvn_embarcacion = get_return_model_polynomial_value(tdc,bodega_frio)
                        df_pozas_estado_original.loc[df_pozas_estado_original.id_chata_linea==chata_linea,'tiempo_fin_descarga']=tiempo_fin_descarga+datetime.timedelta(hours=tiempo_descarga)
                        if df_pozas_estado_original.loc[df_pozas_estado_original.poza_number==poza].nivel.values[0]==0:
                            df_pozas_estado_original.loc[df_pozas_estado_original.poza_number==poza,'tipo_conservacion']=tipo_bodega
                        df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'nivel']=df_pozas_estado_original.nivel+toneladas_a_depositar
                        df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'capacidad_disponible']=df_pozas_estado_original.capacidad_disponible-toneladas_a_depositar
                        df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'tvn']=(df_pozas_estado_original.nivel*df_pozas_estado_original.tvn+toneladas_a_depositar*tvn_embarcacion)/(df_pozas_estado_original.nivel+toneladas_a_depositar)
                        df_pozas_estado_original.loc[(df_pozas_estado_original.poza_number==poza)&(df_pozas_estado_original.id_planta==i),'tdc_minimo_embarcaciones']=df_pozas_estado_original['tdc_minimo_embarcaciones'].apply(lambda x: np.nanmin([tdc,x]))
                        if iteraciones==0:
                            df_soluciones_optimizacion.loc[df_soluciones_optimizacion.marea_id==marea,'toneladas']=toneladas_a_depositar
                        toneladas_depositadas=toneladas_depositadas+toneladas_a_depositar
                        if (toneladas_depositadas<toneladas_embarcacion)&(iteraciones<2):
            
                            #Para elegir pozas posibles primero se ven aquellas que no tengan enfriamiento nulo, para no modificar la decisión
                            #de las siguientes embarcaciones, o también se coloca en una poza con el mismo frío que colocará una de las siguientes
                            #que tengan capacidad, que tengan tvn cercano y tdc minimo con diferencia menor a 3 horas.
                            
                            #Pozas que tengan mismo tipo de frío
                            pozas_posibles_1=set(df_pozas_estado_esp_desc[(df_pozas_estado_esp_desc.tipo_conservacion==tipo_bodega)&(df_pozas_estado_esp_desc.id_planta==i)].poza_number)
                            
                            #Pozas no usadas por las otras embarcaciones
                            pozas_no_usadas=set(df_pozas_estado_original[df_pozas_estado_original.id_planta==i].poza_number)-set(df_soluciones_optimizacion[df_soluciones_optimizacion.id_planta==i].poza_number)
                            pozas_enf_o_nulo=set(df_pozas_estado_esp_desc[((df_pozas_estado_esp_desc.tipo_conservacion==tipo_bodega)|(df_pozas_estado_esp_desc.tipo_conservacion.isnull()))&(df_pozas_estado_esp_desc.id_planta==i)].poza_number)
                            pozas_posibles_2=pozas_no_usadas.intersection(pozas_enf_o_nulo)
                            
                            #Pozas con tdc diferencial menor a 3
                            pozas_posibles_3=set(df_pozas_estado_original[((abs(df_pozas_estado_original.tdc_minimo_embarcaciones-tdc)<=3)|
                                                                    (df_pozas_estado_original.tdc_minimo_embarcaciones.isnull()))&
                                                                    (df_pozas_estado_original.id_planta==i)].poza_number)
                            
                            #Pozas con algo de capacidad.
                            pozas_posibles_4=set(df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)&(df_pozas_estado_original.capacidad_disponible>0)].poza_number)

                            #Pozas depositables desde la chata-linea
                            pozas_posibles_5=set(df_pozas_estado_original[df_pozas_estado_original.id_chata_linea==chata_linea].poza_number)
                            
                            pozas_posibles_final=(pozas_posibles_1.union(pozas_posibles_2)).intersection(pozas_posibles_3).intersection(pozas_posibles_4).intersection(pozas_posibles_5)
                            if len(pozas_posibles_2.intersection(pozas_posibles_3).intersection(pozas_posibles_4).intersection(pozas_posibles_5))>0:
                                posibilidades=pozas_posibles_2.intersection(pozas_posibles_3).intersection(pozas_posibles_4).intersection(pozas_posibles_5)
                                elegida=df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)&
                                                            (df_pozas_estado_original.poza_number.isin(posibilidades))].sort_values(by='capacidad_disponible',ascending=False).reset_index(drop=True).iloc[0]
                                poza=elegida.poza_number
                                capacidad_total_poza=elegida.pozaCapacity
                            elif len(pozas_posibles_final):
                                df_aux=df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)&(df_pozas_estado_original.poza_number.isin(pozas_posibles_final))].copy()
                                df_aux['diff_tvn']=abs(df_aux['tvn']-tvn_embarcacion)
                                elegida=df_aux.sort_values(by='diff_tvn').reset_index(drop=True).iloc[0]
                                poza=elegida.poza_number
                                capacidad_total_poza=elegida.pozaCapacity                    
                            elif len(pozas_posibles_4.intersection(pozas_posibles_5))>0:
                                posibilidades=pozas_posibles_4.intersection(pozas_posibles_5)
                                elegida=df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)&
                                                            (df_pozas_estado_original.poza_number.isin(posibilidades))].sort_values(by='capacidad_disponible',ascending=False).reset_index(drop=True).iloc[0]
                                poza=elegida.poza_number
                                capacidad_total_poza=elegida.pozaCapacity
                            else:
                                try:
                                    ya_usadas=set(df_soluciones_optimizacion[df_soluciones_optimizacion.marea_id==marea].poza_number.astype(int)).union(set(nuevas_filas[nuevas_filas.marea_id==marea].poza_number))    
                                except:
                                    ya_usadas=set(df_soluciones_optimizacion[df_soluciones_optimizacion.marea_id==marea].poza_number.astype(int))
                                por_usar=set(df_pozas_estado_original[df_pozas_estado_original.id_planta==i].poza_number)-ya_usadas
                                posibilidades_por_usar=list(por_usar.intersection(pozas_posibles_5))
                                if len(posibilidades_por_usar)>0:
                                    poza=posibilidades_por_usar[0]
                                    capacidad_total_poza=df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)&(df_pozas_estado_original.poza_number==poza)].pozaCapacity.values[0]
                                else:
                                    iteraciones=iteraciones+3
                                    continue                              
                                poza=posibilidades_por_usar[0]
                                capacidad_total_poza=df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)&(df_pozas_estado_original.poza_number==poza)].pozaCapacity.values[0]
                            
                            # Agregar segunda validacion de elegir poza conectada a chata-linea
                            # Se forzara a escoger entre las pozas conectadas a la chata-linea esocgida si es que no se esta cumpliendo esa condicion
                            # TODO: No es lo ideal, aun hay que validar la logica de toda esta funcion
                            if poza not in pozas_posibles_5:
                                poza = pozas_posibles_5[0]
                                if poza==poza_1:
                                    poza = pozas_posibles_5[1]
                                else:
                                    capacidad_total_poza = df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)&(df_pozas_estado_original.poza_number==poza)].pozaCapacity.values[0]
                            
                            toneladas_a_depositar=np.minimum(capacidad_total_poza,toneladas_embarcacion-toneladas_depositadas)
                            # nueva_fila=row.copy()
                            nueva_fila=pd.DataFrame(row).transpose()
                            nueva_fila.poza_number=poza
                            nueva_fila.toneladas=toneladas_a_depositar
                            capacidad_total_poza=df_pozas_estado_original[(df_pozas_estado_original.id_planta==i)&(df_pozas_estado_original.poza_number==poza)].pozaCapacity.values[0]
                            # nuevas_filas=nuevas_filas.append(nueva_fila)
                            nuevas_filas = pd.concat([nuevas_filas, nueva_fila], ignore_index=True)

                        iteraciones+=1
        # return df_soluciones_optimizacion.append(nuevas_filas).reset_index(drop=True)
        return pd.concat([df_soluciones_optimizacion, nuevas_filas], ignore_index=True)

    # Funcion agregada para calcular diferencia entre fechas para cocinas
    def get_hours_passed_from_date(comparison_date, timestamp):
        if comparison_date is not None:
            env_argument_param = sys.argv[1]
            if env_argument_param == 'dumped':
                date_now = get_dumped_data_date()
            else:
                date_now = datetime.datetime.utcnow()
            # date_now = timestamp #solo en dev
    
            # return (date_now - comparison_date).total_seconds() / 3600
            if os.getenv("ENVIRONMENT") == 'dev':
                date_now = datetime.datetime.fromisoformat(get_data.get_dump_data()['dump_date'].item())
            return ((date_now - comparison_date)/np.timedelta64(1, 's')) / 3600
        else:
            return None

    ### LIMPIAR DATA
    #timestamp = pd.to_datetime(df_embarcaciones['last_modification'].max()) ##TODO: ESTO DEBERIA DE SER TIME NOW???
    def get_current_tdc_from_first_cala():
        env_argument_param = sys.argv[1]
        if env_argument_param == 'dumped':
            date_now = get_dumped_data_date()
        else:
            date_now = datetime.datetime.utcnow()
        if os.getenv("ENVIRONMENT") == 'dev':
            date_now = datetime.datetime.fromisoformat(get_data.get_dump_data()['dump_date'].item())
        return date_now
    timestamp = get_current_tdc_from_first_cala()
    print(timestamp)
    
    # Quitar embarcaciones con mas de 3 dias de zarpe (logica SP)
    df_embarcaciones['departure_port_date'] = pd.to_datetime(df_embarcaciones['departure_port_date'])
    # mask_errados = ((timestamp - df_embarcaciones['departure_port_date'])/np.timedelta64(1,'D') < 3) | (df_embarcaciones['boat_name_trim'] == 'TASA55')
    # df_embarcaciones= df_embarcaciones[mask_errados]
    
    # Filtrar conexion lineas plantas dehabilitadas en la historia
    df_pozas_ubicacion_capacidad = df_pozas_ubicacion_capacidad[(df_pozas_ubicacion_capacidad['fec_deshabilitado']>timestamp) | (df_pozas_ubicacion_capacidad['fec_deshabilitado'].isnull())]
    
    try:
        # Ajustes inicial al df pozas estado
        df_pozas_estado['tipo_bodega'] = np.where((df_pozas_estado['tipo_bodega']=='Frio') & ((df_pozas_estado['frio_system_state']=='RC') | (df_pozas_estado['frio_system_state']=='GF')),'Frio','Tradicional')
        df_pozas_estado = data_pozas(df_pozas_ubicacion_capacidad, df_pozas_estado)
        df_pozas_estado['tvn'].fillna(0,inplace=True)
        df_embarcaciones = limpieza_df_embarcaciones(df_embarcaciones,df_ultimas_recomendaciones_orig)
        # mask = df_embarcaciones['planta_retorno'].notna()
        # df_embarcaciones.loc[mask, 'eta_plant'] = df_embarcaciones.loc[mask, 'planta_retorno']
        df_tiempo_descarga = limpieza_df_tiempo_descarga(df_tiempo_descarga)
        df_pozas_estado = limpieza_df_pozas_estado(df_pozas_estado,df_plantas_habilitadas,df_requerimiento_plantas,df_tiempo_descarga)
        df_embarcaciones_descargando, df_embarcaciones_esperando_descarga, df_embarcaciones_retornando, df_embarcaciones_terceros, df_emb_errores = dividir_embarcaciones(df_embarcaciones, timestamp, df_min_perc_bodego_recom_retorno, df_pozas_estado)
        #print(df_embarcaciones_esperando_descarga)
        df_pozas_estado, df_lineas_estado, df_plantas_estado = dividir_lineas_plantas(df_pozas_estado)
        
        print('Exito en la limpieza de datos')
    except Exception as ex:
        print("Limpieza de datos type error: " + ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        
    
    # Agregar la columna de fin cocina al data pozas estado(se elimino previamente, en la original si se encontraba)
    # Se colocara el fin max de cocina de cad a poza
    df_pozas_estado_cocina['inicio_cocina'] = pd.to_datetime(df_pozas_estado_cocina['inicio_cocina'])
    df_pozas_estado_cocina['fin_cocina'] = pd.to_datetime(df_pozas_estado_cocina['fin_cocina'])
    poza_fin_cocina = df_pozas_estado_cocina.groupby(by=['id_planta','pozaNumber']).agg({'inicio_cocina':'max','fin_cocina':'max','stock_actual_update_date':'max','perc_poza_cocina':'first'})
    poza_fin_cocina = poza_fin_cocina.reset_index()
    poza_fin_cocina['id_planta_poza'] = poza_fin_cocina['id_planta'] + '-' + poza_fin_cocina['pozaNumber'].map('{0:g}'.format).astype(str)
    poza_fin_cocina = poza_fin_cocina.rename(columns={'fin_cocina':'fin_cocina_max','inicio_cocina':'inicio_cocina_max'})
    df_pozas_estado = pd.merge(df_pozas_estado,poza_fin_cocina[['id_planta_poza','inicio_cocina_max','fin_cocina_max','stock_actual_update_date','perc_poza_cocina']],how='left',left_on='id',right_on='id_planta_poza') 
    df_pozas_estado['fin_cocina'] = df_pozas_estado['fin_cocina_max']
    del df_pozas_estado['fin_cocina_max']
    del df_pozas_estado['id_planta_poza']
    
    # Calcular fin de cocina para pozas cocinandose
    df_pozas_estado['stock_actual_hours_passed'] = get_hours_passed_from_date(pd.to_datetime(df_pozas_estado.inicio_cocina_max),timestamp)
    # df_pozas_estado_cocina = df_pozas_estado_cocina[(df_pozas_estado_cocina['cocinandose']==True) & (df_pozas_estado_cocina['inicio_cocina'].notnull()) & (df_pozas_estado_cocina['fin_cocina'].isnull())].reset_index(drop=True)
    
    df_pozas_estado_cocina['id_planta_poza'] = df_pozas_estado_cocina['id_planta'] + '-' + df_pozas_estado_cocina['pozaNumber'].map('{0:g}'.format).astype(str)
    lista_cocina = df_pozas_estado_cocina.loc[(df_pozas_estado_cocina['cocinandose']==True) & (df_pozas_estado_cocina['inicio_cocina'].notnull()) & (df_pozas_estado_cocina['fin_cocina'].isnull()),'id_planta_poza'].reset_index(drop=True)

    # TODO: Chequear los casos que ya haya pasado mucho tiempo desde que inicio cocina VALIDAR
    for index_original,poza in df_pozas_estado.iterrows():
        vel_cocina = df_requerimiento_plantas.loc[df_requerimiento_plantas['id']==poza['id_planta'],'velocidad'].reset_index(drop=True)
        if (np.array(vel_cocina)>0) & (poza['id'] in (lista_cocina.unique())):
            current_estimated_stock = poza['nivel'] - vel_cocina * poza['stock_actual_hours_passed'] * (poza['perc_poza_cocina'] / 100)
            hours_to_finish_poza = current_estimated_stock / (vel_cocina * (poza['perc_poza_cocina'] / 100))
            if np.array(hours_to_finish_poza)>0:
                df_pozas_estado.loc[index_original,'fin_cocina'] = (timestamp + (hours_to_finish_poza)*np.timedelta64(1, 'h')).values
    
    # Eliminar las columnas agregadas por precaucion (ya no se usaran)
    del df_pozas_estado['stock_actual_hours_passed']
    del df_pozas_estado['perc_poza_cocina']
    del df_pozas_estado['stock_actual_update_date']
    del df_pozas_estado['inicio_cocina_max']
    
    ################################################################################################
    ### OPTIMIZAR

    ## Calcular descarga de embarcaciones y actualizar data de pozas
    df_soluciones_optimizacion = pd.DataFrame(columns=['boat_name','id_planta','id_chata','id_linea','poza_number','prioridad','orden_planta','orden_chata_linea','toneladas','tipo','velocidad','marea_id'])
    try:
        if len(df_embarcaciones_descargando.index) == 0:
            print('No hay embarcaciones descargando')
            df_pozas_estado['tiempo_fin_descarga'] = np.nan
            df_pozas_estado['discharge_poza_1'] = np.nan
            df_pozas_estado['id_poza'] = np.nan
        else:
            df_pozas_estado = opt_emb_descargando(df_embarcaciones_descargando, df_pozas_estado,df_tiempo_descarga,timestamp,df_pozas_estado_leido)
            df_pozas_estado['fin_cocina'].fillna(np.nan, inplace=True)
            df_pozas_estado['tiempo_fin_descarga']=np.where(df_pozas_estado['fin_cocina'].notnull(),np.maximum(pd.to_datetime(df_pozas_estado['fin_cocina'].explode()),pd.to_datetime(df_pozas_estado['tiempo_fin_descarga'])),df_pozas_estado['tiempo_fin_descarga'])
        df_pozas_estado['tiempo_fin_descarga'] = df_pozas_estado['tiempo_fin_descarga'].fillna(timestamp)
        df_pozas_estado_original=df_pozas_estado.copy()
        print('Exito Optimizacion Embarcaciones Descargando')
    except Exception as e:
        print("Optimizacion Embarcaciones Descargando type error: " + str(e))        
    
    # Definir primera marea del dia de produccion actual
    cols_dia_prod = ['marea_id','boat_name','discharge_plant_name','discharge_plant_arrival_date','marea_status']
    fecha_actual = timestamp.date()
    inicio_dia_prod = df_horas_produccion[(df_horas_produccion.date_production==str(fecha_actual))]
    inicio_dia_prod = inicio_dia_prod[['id_plant','hora']].reset_index(drop=True)
    df_emb_cerradas = df_mareas_cerradas[cols_dia_prod]
    df_emb_des_res = df_embarcaciones_descargando.copy()
    df_emb_des_res = df_emb_des_res.merge(df_embarcaciones[['marea_id','discharge_plant_arrival_date']], how='left', on='marea_id')
    df_emb_des_res = df_emb_des_res[cols_dia_prod]
    df_emb_esp_desc_res = df_embarcaciones_esperando_descarga[cols_dia_prod]
    df_emb_total = pd.concat([df_emb_cerradas,df_emb_des_res,df_emb_esp_desc_res], axis=0)
    df_emb_total = df_emb_total.sort_values(['discharge_plant_name','discharge_plant_arrival_date'])
    df_mareas_dia_prod = df_emb_total.merge(inicio_dia_prod, how='left', left_on='discharge_plant_name', right_on='id_plant')
    mask_inicio = df_mareas_dia_prod['discharge_plant_arrival_date']>=df_mareas_dia_prod['hora']
    df_mareas_dia_prod = df_mareas_dia_prod[mask_inicio].reset_index(drop=True)
    df_mareas_dia_prod['ranking'] = df_mareas_dia_prod.groupby(['discharge_plant_name'])['discharge_plant_arrival_date'].rank(method='first')
    df_mareas_dia_prod = df_mareas_dia_prod[df_mareas_dia_prod['ranking']==1].reset_index(drop=True)
    
    ## Optimizacion de embarcaciones esperando descarga y actualizacion de data de pozas
    try:
        if len(df_embarcaciones_esperando_descarga) == 0:
            print('No Hay Embarcaciones Esperando Descarga')
            df_pozas_estado_esp_desc=df_pozas_estado.copy()
            # Inicializar dataframes vacios de flags
            cols_flags = ['timestamp','marea_id','discharge_plant_name','discharge_chata_name','discharge_line_name','discharge_start_date','tdc_arrival','tvn_discharge','discharge_poza_1','discharge_poza_2','discharge_poza_3','discharge_poza_4','tipo_bodega','frio_system_state','bodega_frio','restriccion_ep','chata_linea_ingresada']
            df_flags_cabecera = pd.DataFrame(columns=cols_flags)
            
            cols_ordenes = ['id_orden','timestamp','marea_id','tdc_ponderado','capacidad_pozas','tipo_poza','flag_preservante','flag_presion_vacio','dif_tdc_ep_poza','tvn_previo_desc_ep','tvn_poza','flag_limite_emb','flag_limite_tiempo','flag_balanceo','tdc_previo_desc_ep','orden_emb','es_pama','flag_planta_llena','chata_rec_orden','linea_rec_orden','poza1_rec_orden','poza2_rec_orden','poza3_rec_orden','poza4_rec_orden']
            df_flags_ordenes = pd.DataFrame(columns=cols_ordenes)
            
            cols_utilidad = ['discharge_plant_name','marea_id','boat_name','discharge_plant_arrival_date','id_orden','tvn_pozas_ponderado','orden_emb','chata_rec_orden','linea_rec_orden','poza1_rec_orden','poza2_rec_orden','poza3_rec_orden','poza4_rec_orden','tdc_previo_desc_ep','tvn_previo_desc_ep','tvn_poza','tipo_poza','dif_tdc_ep_poza','flag_preservante','flag_presion_vacio','flag_balanceo','msg_tipo_poza','msg_dif_tdc','msg_preservante','msg_presion_vacio','msg_balanceo']
            df_tabla_utilidad = pd.DataFrame(columns=cols_utilidad)
            # df_soluciones_optimizacion_esperando_descarga = pd.DataFrame(columns=['id_planta', 'toneladas'])
        else:
            df_pozas_estado, df_soluciones_optimizacion_esperando_descarga,df_utilidad,df_flags = optimizacion_embarcaciones_esperando_descarga(df_embarcaciones_esperando_descarga,df_pozas_estado,timestamp,df_tiempo_descarga,df_plantas_velocidad_limites,df_restricciones,df_embarcaciones_descargando,df_priorizacion_linea,df_chatas_lineas,df_embarcaciones,df_mareas_acodere,df_mareas_cerradas,df_planta_velocidad_anterior, df_requerimiento_plantas, df_master_fajas, df_horas_produccion, df_prioridad_pozas, df_velocidad_descarga, df_pozas_ubicacion_capacidad, df_mareas_dia_prod)
            
            # Separar las tablas para guardado de flags
            # Primero generar la cabecera
            if len(df_flags)>0:
                
                df_flags['timestamp'] = timestamp
                df_flags_cabecera = df_flags[['timestamp','marea_id','discharge_plant_name','discharge_chata_name','discharge_line_name','discharge_start_date','tdc_arrival','tvn_discharge','discharge_poza_1','discharge_poza_2','discharge_poza_3','discharge_poza_4','tipo_bodega','frio_system_state','bodega_frio','restriccion_ep','chata_linea_ingresada']].copy()
                df_flags_cabecera = df_flags_cabecera.drop_duplicates()
            
                # Luego generar los ordenes analizados
                df_flags_ordenes = df_flags[['id_orden','timestamp','marea_id','tdc_ponderado','capacidad_pozas','tipo_poza','flag_preservante','flag_desp_positivo','dif_tdc_ep_poza','tvn_previo_desc_ep','tvn_poza','flag_limite_emb','flag_limite_tiempo','flag_balanceo','tdc_previo_desc_ep','orden_emb','es_pama','flag_planta_llena','chata_rec_orden','linea_rec_orden','poza1_rec_orden']]
                df_flags_ordenes['poza2_rec_orden'] = np.nan
                df_flags_ordenes['poza3_rec_orden'] = np.nan
                df_flags_ordenes['poza4_rec_orden'] = np.nan
                df_flags_ordenes = df_flags_ordenes.rename(columns={'flag_desp_positivo':'flag_presion_vacio'})
                
                # Generar tabla utilidades actual para los resultados de esperando descarga
                parte1 = df_flags_cabecera[['discharge_plant_name','marea_id']]
                parte2 = df_flags_ordenes[['marea_id','id_orden','orden_emb','chata_rec_orden','linea_rec_orden','poza1_rec_orden','poza2_rec_orden','poza3_rec_orden','poza4_rec_orden','tdc_previo_desc_ep','tvn_previo_desc_ep','tvn_poza','tipo_poza','dif_tdc_ep_poza','flag_preservante','flag_presion_vacio','flag_balanceo','flag_planta_llena']]
                
                df_tabla_utilidad = pd.merge(parte1,parte2, on='marea_id',how='left') 
                
                # # Agregar la fecha de arribo y nombre EP y TVN ponderado
                df_tabla_utilidad = pd.merge(df_tabla_utilidad,df_embarcaciones[['marea_id','boat_name','discharge_plant_arrival_date']])
                df_tabla_utilidad = pd.merge(df_tabla_utilidad,df_flags[['discharge_plant_name','id_orden','TVN_Ponderado']].drop_duplicates(),how='left',on=['discharge_plant_name','id_orden'])
                df_tabla_utilidad = df_tabla_utilidad[['discharge_plant_name','marea_id','boat_name','discharge_plant_arrival_date','id_orden','TVN_Ponderado','orden_emb','chata_rec_orden','linea_rec_orden','poza1_rec_orden','poza2_rec_orden','poza3_rec_orden','poza4_rec_orden','tdc_previo_desc_ep','tvn_previo_desc_ep','tvn_poza','tipo_poza','dif_tdc_ep_poza','flag_preservante','flag_presion_vacio','flag_balanceo','flag_planta_llena']]
                df_tabla_utilidad = df_tabla_utilidad.rename(columns={'TVN_Ponderado':'tvn_pozas_ponderado'})
                
                # Agregar comentarios
                # df_tabla_utilidad['msg_tipo_poza'] = df_comentarios.loc['']
                df_tabla_utilidad['msg_tipo_poza'] = np.nan
                df_tabla_utilidad['msg_dif_tdc'] = np.nan
                df_tabla_utilidad['msg_preservante'] = np.nan
                df_tabla_utilidad['msg_presion_vacio'] = np.nan
                df_tabla_utilidad['msg_balanceo'] = np.nan
                df_tabla_utilidad['timestamp'] = np.nan
                    
                for index,row in df_tabla_utilidad.iterrows():
                    
                    if row['tipo_poza']==1:
                        df_tabla_utilidad.loc[index,'msg_tipo_poza'] = df_comentarios.loc[0,'mensaje_cumple']
                    else:
                        df_tabla_utilidad.loc[index,'msg_tipo_poza'] = df_comentarios.loc[0,'mensaje_no_cumple']
                    
                    if row['dif_tdc_ep_poza']==1:
                        df_tabla_utilidad.loc[index,'msg_dif_tdc'] = df_comentarios.loc[3,'mensaje_cumple']
                    else:
                        df_tabla_utilidad.loc[index,'msg_dif_tdc'] = df_comentarios.loc[3,'mensaje_no_cumple']
                    
                    if row['flag_preservante']==1:
                        df_tabla_utilidad.loc[index,'msg_preservante'] = df_comentarios.loc[1,'mensaje_cumple']
                    else:
                        df_tabla_utilidad.loc[index,'msg_preservante'] = df_comentarios.loc[1,'mensaje_no_cumple']
                    
                    if row['flag_presion_vacio']==1:
                        df_tabla_utilidad.loc[index,'msg_presion_vacio'] = df_comentarios.loc[2,'mensaje_cumple']
                    else:
                        df_tabla_utilidad.loc[index,'msg_presion_vacio'] = df_comentarios.loc[2,'mensaje_no_cumple']
    
                    if row['flag_balanceo']==1:
                        df_tabla_utilidad.loc[index,'msg_balanceo'] = df_comentarios.loc[4,'mensaje_cumple']
                    else:
                        df_tabla_utilidad.loc[index,'msg_balanceo'] = df_comentarios.loc[4,'mensaje_no_cumple']                
                    
            else:
                cols_flags = ['timestamp','marea_id','discharge_plant_name','discharge_chata_name','discharge_line_name','discharge_start_date','tdc_arrival','tvn_discharge','discharge_poza_1','discharge_poza_2','discharge_poza_3','discharge_poza_4','tipo_bodega','frio_system_state','bodega_frio','restriccion_ep','chata_linea_ingresada']
                df_flags_cabecera = pd.DataFrame(columns=cols_flags)
                
                cols_ordenes = ['id_orden','timestamp','marea_id','tdc_ponderado','capacidad_pozas','tipo_poza','flag_preservante','flag_presion_vacio','dif_tdc_ep_poza','tvn_previo_desc_ep','tvn_poza','flag_limite_emb','flag_limite_tiempo','flag_balanceo','tdc_previo_desc_ep','orden_emb','es_pama','flag_planta_llena','chata_rec_orden','linea_rec_orden','poza1_rec_orden','poza2_rec_orden','poza3_rec_orden','poza4_rec_orden']
                df_flags_ordenes = pd.DataFrame(columns=cols_ordenes)
                
                cols_utilidad = ['discharge_plant_name','marea_id','boat_name','discharge_plant_arrival_date','id_orden','tvn_pozas_ponderado','orden_emb','chata_rec_orden','linea_rec_orden','poza1_rec_orden','poza2_rec_orden','poza3_rec_orden','poza4_rec_orden','tdc_previo_desc_ep','tvn_previo_desc_ep','tvn_poza','tipo_poza','dif_tdc_ep_poza','flag_preservante','flag_presion_vacio','flag_balanceo','flag_planta_llena','msg_tipo_poza','msg_dif_tdc','msg_preservante','msg_presion_vacio','msg_balanceo']
                df_tabla_utilidad = pd.DataFrame(columns=cols_utilidad)
            
            # df_utilidad = pd.merge(df_utilidad, df_embarcaciones_esperando_descarga[['boat_name', 'discharge_plant_arrival_date', 'marea_id']], on='marea_id', how='left')
            df_pozas_estado_esp_desc=df_pozas_estado.copy()
            # print(df_soluciones_optimizacion_esperando_descarga)
            print('Exito Optimizacion Embarcaciones Esperando Descarga')
            df_soluciones_optimizacion_esperando_descarga['tipo'] = 'esperando_descarga'
            df_soluciones_optimizacion_esperando_descarga['last_modification'] = timestamp
            # df_soluciones_optimizacion = df_soluciones_optimizacion.append(df_soluciones_optimizacion_esperando_descarga)
            df_soluciones_optimizacion = pd.concat([df_soluciones_optimizacion, df_soluciones_optimizacion_esperando_descarga], ignore_index=True)
    except Exception as e:
        df_pozas_estado_esp_desc=df_pozas_estado.copy()
        print("Optimizacion Embarcaciones Esperando Descarga type error: " + str(e))
        traceback.print_exc()
    
    
    # Agregar el saldo en pozas manual
    # Stock inicial de pozas por planta
    df_pozas_estado_aux = df_pozas_estado_leido[['id_planta','pozaNumber','stock_actual']].copy()
    df_pozas_estado_aux = df_pozas_estado_aux.drop_duplicates()
    # df_pozas_estado_aux['id_planta_poza'] = df_pozas_estado_aux['id_planta'] + '-' + df_pozas_estado_aux['pozaNumber'].map('{0:g}'.format).astype(str)
    df_pozas_estado_aux = df_pozas_estado_aux.groupby(['id_planta']).stock_actual.sum()
    df_pozas_estado_aux = pd.DataFrame(df_pozas_estado_aux)
    df_pozas_estado_aux.reset_index(inplace=True)
    
    # Agregar toneladas asignadas a las pozas
    if len(df_embarcaciones_esperando_descarga)!=0:
        df_tm_asignadas = df_soluciones_optimizacion_esperando_descarga[['id_planta','toneladas']].groupby('id_planta').toneladas.sum()
        df_tm_asignadas = pd.DataFrame(df_tm_asignadas)
        df_tm_asignadas.reset_index(inplace=True)
        
        df_pozas_estado_aux = df_pozas_estado_aux.merge(df_tm_asignadas,how='left',on='id_planta')
        df_pozas_estado_aux['toneladas'] = df_pozas_estado_aux['toneladas'].fillna(0)
        df_pozas_estado_aux['stock_actualizado'] = df_pozas_estado_aux['stock_actual'] + df_pozas_estado_aux['toneladas']
        
        df_pozas_estado = df_pozas_estado.merge(df_pozas_estado_aux[['id_planta','stock_actualizado']],how='left',on='id_planta')
    else:
        df_pozas_estado['stock_actualizado'] = 0
    
    df_back_up = df_pozas_estado.copy()
    df_back_up['tdc_hipotetico'] = np.nan
    df_back_up['capacidad_disponible'] = df_back_up['pozaCapacity'] - df_back_up['nivel']
    df_back_up['capacidad_disponible'] = np.where(df_back_up['capacidad_disponible']<0, 0, df_back_up['capacidad_disponible'])
    df_back_up['id_chata_linea'] = df_back_up['id_chata'] + '-' + df_back_up['id_linea']
    df_pozas_estado_retorno = df_pozas_estado.copy()
        
    ## Optimizar embarcaciones terceras y retornando
    try:
        if (len(df_embarcaciones_retornando) == 0):
            df_retorno_utilidad = pd.DataFrame(columns=['marea_id', 'prioridad'])
            print('No Hay Embarcaciones Retornando ni de terceros')
        else:
            df_pozas_estado, df_soluciones_optimizacion_retornando_terceros, df_retorno_utilidad = optimizacion_embarcaciones_retornando_terceros(df_embarcaciones_retornando,df_embarcaciones_terceros, df_pozas_estado,df_embarcaciones,timestamp,df_plantas_velocidad_limites,df_tiempo_descarga,df_requerimiento_plantas,df_lineas_reservada_terceros, df_chatas_lineas,df_planta_velocidad_anterior,df_embarcaciones_esperando_descarga,df_calidades_precio_venta, df_mareas_cerradas)
            print('Exito Optimizacion Embarcaciones Terceros y retornando')
            df_soluciones_optimizacion_retornando_terceros['last_modification'] = timestamp
            # df_soluciones_optimizacion = df_soluciones_optimizacion.append(df_soluciones_optimizacion_retornando_terceros)
            df_soluciones_optimizacion = pd.concat([df_soluciones_optimizacion, df_soluciones_optimizacion_retornando_terceros], ignore_index=True)
            df_retorno_utilidad = pd.merge(df_retorno_utilidad, df_soluciones_optimizacion_retornando_terceros[['marea_id', 'prioridad']], how='left', on='marea_id')
            # df_soluciones_optimizacion_retornando_terceros.to_csv('outputs/df_soluciones_optimizacion_retornando_terceros_NEW.csv')
            
            # Nueva logica de asignacion de chata-linea en retornando
            activar = True
            mask = df_velocidad_descarga['NOM_CHATA_COMPLETO'] == "CHATA TASA CALLAO"
            df_velocidad_descarga.loc[mask, "NOM_CHATA_COMPLETO"] = "CHATA CHILLON"
            mask = df_velocidad_descarga['NOM_CHATA_COMPLETO'] == "CHATA EX-ABA"
            df_velocidad_descarga.loc[mask, "NOM_CHATA_COMPLETO"] = "CHATA EXABA"
            df_restricciones['id_chata_linea'] = df_restricciones['id_chata'] + '-' +  df_restricciones['id_linea']
            
            if activar:
                df_test = df_soluciones_optimizacion_retornando_terceros.copy()
                # Llegada para propias
                df_test = pd.merge(df_test, df_retorno_utilidad[['marea_id','FEH_ARRIBO_ESTIMADA']], how='left', on='marea_id')
                # Llegada para terceras
                extra_columns = ['marea_id','eta','first_cala_start_date','tipo_bodega','frio_system_state','owner_group']
                df_test = pd.merge(df_test, df_embarcaciones_retornando[extra_columns], how='left', on='marea_id')
                df_test['fecha_arribo'] = np.where(df_test['FEH_ARRIBO_ESTIMADA'].isnull(), df_test['eta'], pd.to_datetime(df_test['FEH_ARRIBO_ESTIMADA']))
                df_test = df_test.sort_values(['id_planta','prioridad'])
                df_test['tipo_bodega_actual'] = np.where((df_test['tipo_bodega']=='Frio') & ((df_test['frio_system_state']=='RC') | (df_test['frio_system_state']=='GF')), 'Frio-RC', 'Otro')
                df_test['id_chata_linea'] = df_test['id_chata'] + '-' + df_test['id_linea']
                df_test = df_test.reset_index(drop=True)
                limite_tdc_descarga = df_velocidad_descarga['NUM_LIMITE_TDC'].unique().item()
                df_test_original = df_test.copy()
                
                # Aplicacion de reglas de chata linea poza para la eleccion de chata-linea en retorno
                for index_marea,row_marea in df_test.iterrows():
                    # print(row_marea['marea_id'])
                    # Primero se aplican condicionales
                    df_back_up_aux = df_back_up[df_back_up['id_planta']==row_marea['id_planta']].reset_index(drop=True)
                    df_back_up_bkp = df_back_up_aux.copy()
                    # Condicion 5
                    posibilidades_segun_restriccion=list(df_restricciones[(df_restricciones.boat_name==row_marea['boat_name'])&(df_restricciones.id_planta==row_marea['id_planta'])].id_chata_linea)
                    if len(posibilidades_segun_restriccion)>0:
                        condicion5=(~df_back_up_aux.id_chata_linea.isin(posibilidades_segun_restriccion))
                    else:
                        # Si no se cumple, seleccionar todas las filas (CONDICION DUMMY)
                        condicion5=(df_back_up.poza_number>0)
                    
                    df_back_up_aux = df_back_up_aux[condicion5].copy()
                    if len(df_back_up_aux)==0:
                        df_back_up_aux = df_back_up_bkp.copy()
                    # elif len(posibilidades_segun_restriccion)>0:
                    #     df_embar_espe_desc.loc[mask,'restriccion_ep'] = 1
                    
                    # Condicion 8
                    df_back_up_aux['tdc_hipotetico']=(df_back_up_aux.tiempo_fin_descarga-row_marea['first_cala_start_date'])/np.timedelta64(1, 'h')
                    condicion8=(abs(df_back_up_aux.tvn-df_back_up_aux.tdc_hipotetico)<=3)|(df_back_up_aux.tvn.isnull())
                    df_back_up_bkp = df_back_up_aux.copy()
                    df_back_up_aux = df_back_up_aux[condicion8]
                    if len(df_back_up_aux)==0:
                        df_back_up_aux = df_back_up_bkp.copy()
                    # else:
                    #     df_embar_espe_desc.loc[mask,'dif_tdc_ep_poza'] = 1
                    
                    # Condicion 3:
                    condicion3=((df_back_up_aux.tipo_conservacion==row_marea['tipo_bodega_actual'])|(df_back_up_aux.tipo_conservacion.isnull())|(df_back_up_aux['nivel'] == 0))
                    df_back_up_bkp = df_back_up_aux.copy()
                    df_back_up_aux = df_back_up_aux[condicion3]
                    if len(df_back_up_aux)==0:
                        df_back_up_aux = df_back_up_bkp.copy()
                    # else:
                    #     df_embar_espe_desc.loc[mask,'tipo_poza'] = 1
                    
                    # Condicion 2:
                    condicion2= (df_back_up_aux.capacidad_disponible>0)
                    df_back_up_bkp = df_back_up_aux.copy()
                    df_back_up_aux = df_back_up_aux[condicion2]
                    if len(df_back_up_aux)==0:
                        df_back_up_aux = df_back_up_bkp.copy()
                    # else:
                    #     df_embar_espe_desc.loc[mask,'capacidad_pozas'] = 1
                    
                    # Condicion 6:
                    # Solo se marcara la condicion si es que hay flag de preservante, si no se considera como que no se cumplio (valor 0)
                    condicion6=((df_back_up_aux.con_hielo)&(((~df_back_up_aux.tipo_conservacion.isin(['Frio-RC','Frio-GF','Frio-CH']))&(df_back_up_aux.tdc_hipotetico<=14))|((df_back_up_aux.tipo_conservacion.isin(['Frio-RC','Frio-GF','Frio-CH']))&(df_back_up_aux.tdc_hipotetico<=30))))|(df_back_up_aux.con_hielo==False)
                    df_back_up_bkp = df_back_up_aux.copy()
                    df_back_up_aux = df_back_up_aux[condicion6]
                    if len(df_back_up_aux)==0:
                        df_back_up_aux = df_back_up_bkp.copy()
                        # df_embar_espe_desc.loc[mask,'flag_preservante'] = 0
                    # df_test[(index_marea-1):index_marea]
                    
                    try:
                        anterior=[x[0:x.find('-',x.find('-',0)+1)] for x in [df_test['id_chata_linea'][(index_marea-1):index_marea]]]
                    except:
                        anterior=[x[0:x.find('-',x.find('-',0)+1)] for x in [df_test['id_chata_linea'][0]]]
                        
                    # if i == []:
                    #     anterior=[]
                    # else:
                    #     anterior=[x[0:x.find('-',x.find('-',0)+1)] for x in [df_test['id_chata_linea'][(index_marea-1):index_marea]]]
                        
                    sub_cond_1 = (('Frio-RC' not in row_marea['tipo_bodega_actual']) | (row_marea['owner_group'] == 'T'))
                    sub_cond_2 = (df_back_up_aux['tdc_hipotetico'] >= 18)
                    condicion7 = np.where(sub_cond_1 & sub_cond_2, (df_back_up_aux['sistema_absorbente'] == 'DesplazamientoPositivo'), (df_back_up_aux['sistema_absorbente'].isin(['PresionVacio', 'DesplazamientoPositivo'])))
                    df_back_up_bkp = df_back_up_aux.copy()
                    
                    if ((len(df_back_up_aux)==0) | (list(df_back_up_aux.loc[condicion7, 'id_chata_linea'].unique()) == anterior)):
                        df_back_up_aux = df_back_up_bkp.copy()
                    else:
                        df_back_up_aux = df_back_up_aux[condicion7]
                        if len(df_back_up_aux.index) == 0:
                            df_back_up_aux = df_back_up_bkp.copy()
                        # df_embar_espe_desc.loc[mask,'flag_desp_positivo'] = 1
                    
                    # Escoger la primera con menor tvn
                    # Calcular diferencia entre TVNs
                    # Ordenar por menor diferencia y mayor capacidad
                    # opciones_pozas = opciones_pozas.sort_values(['diff','nivel'], ascending=[True,False])
                    opciones_pozas = df_back_up_aux.sort_values(['tiempo_fin_descarga','nivel'], ascending=[True,False]).reset_index(drop=True)
                    df_val = df_back_up_aux.sort_values(['tiempo_fin_descarga','nivel'], ascending=[True,False]).reset_index(drop=True)
                    opciones_pozas = opciones_pozas.head(1)
                    chata_optima = opciones_pozas['id_chata'].item()
                    linea_optima = opciones_pozas['id_linea'].item()
                    
                    # Sobreescribir en el data frame de resultados
                    df_test.loc[df_test['marea_id']==row_marea['marea_id'], 'id_chata'] = chata_optima
                    df_test.loc[df_test['marea_id']==row_marea['marea_id'], 'id_linea'] = linea_optima
                    df_test.loc[df_test['marea_id']==row_marea['marea_id'], 'id_chata_linea'] = chata_optima + '-' + linea_optima
                    
                    # Calcular tdc de arribo
                    df_test['tdc_arribo'] = df_test['fecha_arribo'] - df_test['first_cala_start_date']
                    df_test['tdc_arribo'] = df_test['tdc_arribo'] / datetime.timedelta(hours=1)
                    tdc_hipotetico = df_test.loc[df_test['marea_id']==row_marea['marea_id'],'tdc_arribo'].item()
                    
                    # Actualizar velocidad de descarga
                    if tdc_hipotetico<=limite_tdc_descarga:
                        vel_rango1 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata_optima) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea_optima) & (df_velocidad_descarga['DES_RANGO_TM']=='<=100'),'NUM_VEL_LIMITE_INF'].item()
                        vel_rango2 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata_optima) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea_optima) & (df_velocidad_descarga['DES_RANGO_TM']=='100-300'),'NUM_VEL_LIMITE_INF'].item()
                        vel_rango3 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata_optima) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea_optima) & (df_velocidad_descarga['DES_RANGO_TM']=='>300'),'NUM_VEL_LIMITE_INF'].item()
                    else:
                        vel_rango1 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata_optima) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea_optima) & (df_velocidad_descarga['DES_RANGO_TM']=='<=100'),'NUM_VEL_LIMITE_SUP'].item()
                        vel_rango2 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata_optima) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea_optima) & (df_velocidad_descarga['DES_RANGO_TM']=='100-300'),'NUM_VEL_LIMITE_SUP'].item()
                        vel_rango3 = df_velocidad_descarga.loc[(df_velocidad_descarga['NOM_CHATA_COMPLETO']==chata_optima) & (df_velocidad_descarga['NOM_LINEA_DESCARGA']==linea_optima) & (df_velocidad_descarga['DES_RANGO_TM']=='>300'),'NUM_VEL_LIMITE_SUP'].item()
                    
                    df_tiempo_descarga.loc[(df_tiempo_descarga['id_chata']==chata_optima) & (df_tiempo_descarga['id_linea']==linea_optima),'velocidad_0_100_tons'] = vel_rango1
                    df_tiempo_descarga.loc[(df_tiempo_descarga['id_chata']==chata_optima) & (df_tiempo_descarga['id_linea']==linea_optima),'velocidad_100_300_tons'] = vel_rango2
                    df_tiempo_descarga.loc[(df_tiempo_descarga['id_chata']==chata_optima) & (df_tiempo_descarga['id_linea']==linea_optima),'velocidad_300_mas_tons'] = vel_rango3
                    
                    # Actualizar hora fin descarga
                    tiempo_descarga = fin_descarga_temp_value(row_marea['toneladas'], chata_optima, linea_optima, df_tiempo_descarga)
                    df_back_up.loc[(df_back_up['id_chata']==chata_optima) & (df_back_up['id_linea']==linea_optima),'tiempo_fin_descarga'] = df_back_up.loc[(df_back_up['id_chata']==chata_optima) & (df_back_up['id_linea']==linea_optima),'tiempo_fin_descarga'] + datetime.timedelta(hours=tiempo_descarga)
                    
                    # Actualizar df soluciones
                    df_soluciones_optimizacion.loc[df_soluciones_optimizacion['marea_id']==row_marea['marea_id'],'id_chata'] = chata_optima
                    df_soluciones_optimizacion.loc[df_soluciones_optimizacion['marea_id']==row_marea['marea_id'],'id_linea'] = linea_optima
                    
                    df_soluciones_optimizacion_retornando_terceros.loc[df_soluciones_optimizacion_retornando_terceros['marea_id']==row_marea['marea_id'],'id_chata'] = chata_optima
                    df_soluciones_optimizacion_retornando_terceros.loc[df_soluciones_optimizacion_retornando_terceros['marea_id']==row_marea['marea_id'],'id_linea'] = linea_optima
            
    except Exception as e:
        print("Optimizacion Embarcaciones Terceros y Retornando type error: " + str(e))
    df_soluciones_optimizacion.reset_index(inplace=True,drop=True)
    
        
    try:
        df_soluciones_optimizacion=recomendar_mas_de_una_poza(df_pozas_estado_original,df_pozas_estado_esp_desc,df_soluciones_optimizacion,df_embarcaciones_esperando_descarga,df_tiempo_descarga)
    except:
        print("Error en algoritmo de recomendación de más de una poza.")
    ################################################################################################
    try:
        # df_soluciones_optimizacion = df_soluciones_optimizacion.drop_duplicates('marea_id', keep='first')
        df_sol_opt = df_soluciones_optimizacion.sort_values(['toneladas'], ascending=False).groupby(['boat_name'],as_index=False).agg({'marea_id':'first','id_planta':'first','id_chata':'first','id_linea':'first',
                                                                        'prioridad':'first','orden_planta':'first','orden_chata_linea':'first','tipo':'first','velocidad':'first','toneladas':list,'poza_number':list}).reset_index(drop=True)
        if len(df_sol_opt) <= 0:
            print('No hay embarcaciones')
            # df_sol_opt[['tons_poza_descarga_1','tons_poza_descarga_2','poza_descarga_1','poza_descarga_2','tons_poza_descarga_3','poza_descarga_3', 'tons_poza_descarga_4','poza_descarga_4']]=None
            new_cols = ['tons_poza_descarga_1','tons_poza_descarga_2','poza_descarga_1','poza_descarga_2','tons_poza_descarga_3','poza_descarga_3', 'tons_poza_descarga_4','poza_descarga_4']
            for col in new_cols:
                df_sol_opt[col] = None
        else:
            df_sol_opt['toneladas'] = df_sol_opt.apply(lambda row : row['toneladas'][0:4], axis = 1) 
            df_sol_opt['poza_number'] = df_sol_opt.apply(lambda row : row['poza_number'][0:4], axis = 1) 
            largo = df_sol_opt['toneladas'].str.len().max()
            if largo == 4:
                df_sol_opt[['tons_poza_descarga_1','tons_poza_descarga_2','tons_poza_descarga_3', 'tons_poza_descarga_4']] = pd.DataFrame(df_sol_opt['toneladas'].values.tolist(), index= df_sol_opt.index)
                df_sol_opt[['poza_descarga_1','poza_descarga_2','poza_descarga_3', 'poza_descarga_4']] = pd.DataFrame(df_sol_opt['poza_number'].values.tolist(), index= df_sol_opt.index)
            elif largo == 3:
                df_sol_opt[['tons_poza_descarga_1','tons_poza_descarga_2','tons_poza_descarga_3']] = pd.DataFrame(df_sol_opt['toneladas'].values.tolist(), index= df_sol_opt.index)
                df_sol_opt[['poza_descarga_1','poza_descarga_2','poza_descarga_3']] = pd.DataFrame(df_sol_opt['poza_number'].values.tolist(), index= df_sol_opt.index)
                df_sol_opt['tons_poza_descarga_4'] = np.nan
                df_sol_opt['poza_descarga_4'] = np.nan
            elif largo == 2:
                df_sol_opt[['tons_poza_descarga_1','tons_poza_descarga_2']] = pd.DataFrame(df_sol_opt['toneladas'].values.tolist(), index= df_sol_opt.index)
                df_sol_opt[['poza_descarga_1','poza_descarga_2']] = pd.DataFrame(df_sol_opt['poza_number'].values.tolist(), index= df_sol_opt.index)
                df_sol_opt['tons_poza_descarga_3'] = np.nan
                df_sol_opt['poza_descarga_3'] = np.nan
                df_sol_opt['tons_poza_descarga_4'] = np.nan
                df_sol_opt['poza_descarga_4'] = np.nan
            else:
                df_sol_opt[['tons_poza_descarga_1']] = pd.DataFrame(df_sol_opt['toneladas'].values.tolist(), index= df_sol_opt.index)
                df_sol_opt[['poza_descarga_1']] = pd.DataFrame(df_sol_opt['poza_number'].values.tolist(), index= df_sol_opt.index)
                df_sol_opt['tons_poza_descarga_2'] = np.nan
                df_sol_opt['poza_descarga_2'] = np.nan
                df_sol_opt['tons_poza_descarga_3'] = np.nan
                df_sol_opt['poza_descarga_3'] = np.nan
                df_sol_opt['tons_poza_descarga_4'] = np.nan
                df_sol_opt['poza_descarga_4'] = np.nan
            del df_sol_opt['toneladas']
            del df_sol_opt['poza_number']
            print('Exito en consolidacion de resultados')
    except Exception as e:
        print("Error en consolidacion de resultados: " + str(e))
        
    # Completar campos de pozas
    if len(df_tabla_utilidad)>0:
        for index_util,row_util in df_tabla_utilidad.iterrows():
            marea_util = int(row_util['marea_id'])
            try: # ALERT: Modificar y ver por qué está pasando
                poza_2 = df_sol_opt.loc[df_sol_opt['marea_id']==marea_util,'poza_descarga_2'].item()
                poza_3 = df_sol_opt.loc[df_sol_opt['marea_id']==marea_util,'poza_descarga_3'].item()
                poza_4 = df_sol_opt.loc[df_sol_opt['marea_id']==marea_util,'poza_descarga_4'].item()
                df_tabla_utilidad.loc[index_util,'poza2_rec_orden'] = poza_2
                df_tabla_utilidad.loc[index_util,'poza3_rec_orden'] = poza_3
                df_tabla_utilidad.loc[index_util,'poza4_rec_orden'] = poza_4
            except:
                pass
    # print(df_tabla_utilidad)
        
    ## Limpiar recomendaciones
    ## Limpiar recomendacion planta
    try:
        df_sol_opt = pd.merge(df_sol_opt, df_embarcaciones[['marea_id','Latitud','Longitud']].groupby('marea_id').head(1).reset_index(drop=True), how = 'left', on = 'marea_id')
        # df_sol_opt = limpiar_recomendacion_planta(df_sol_opt, df_pozas_estado)
        print('Exito en limpieza de recomendaciones planta')
    except Exception as e:
        print("Error en limpieza de recomendaciones planta: " + str(e)) 
    

    df_sol_opt = df_sol_opt.rename(columns={'id_planta':'planta_retorno','id_chata':'chata_descarga','id_linea':'linea_descarga','orden_chata_linea':'orden_descarga','velocidad':'velocidad_retorno','orden_planta':'orden_descarga_global'}).copy()
    solution_opt_cols = ['marea_id','planta_retorno','chata_descarga','linea_descarga','orden_descarga','orden_descarga_global','velocidad_retorno','tons_poza_descarga_1','tons_poza_descarga_2','poza_descarga_1','poza_descarga_2','tons_poza_descarga_3','poza_descarga_3', 'tons_poza_descarga_4', 'poza_descarga_4']
    df_sol_opt = df_sol_opt[solution_opt_cols].copy()
    
    if len(df_sol_opt) > 0:
        ## limpieza recomendaciones eta
        try:
            df_pozas_estado_esp_desc['cap_usar'] = df_pozas_estado_esp_desc['pozaCapacity'] - df_pozas_estado_esp_desc['nivel']
            df_pozas_estado_esp_desc['cap_usar'] = df_pozas_estado_esp_desc['cap_usar'].map(float)
            df_sol_opt = limpiar_eta(df_sol_opt,df_embarcaciones,df_pozas_estado_esp_desc,df_chatas_lineas)
            print('Exito en limpieza de recomendaciones eta')
        except Exception as e:
            print("Error en limpieza de recomendaciones eta: " + str(e)) 
            # df_sol_opt = limpiar_eta(df_sol_opt,df_embarcaciones,df_pozas_estado)
            # print('Exito en limpieza de recomendaciones eta')

        ## limpieza recomendacion velocidad maxima
        # try:
        #     # df_sol_opt = limpiar_velocidades_max(df_sol_opt,df_pozas_estado,timestamp, df_embarcaciones)
        #     print('Exito en limpieza de recomendaciones velocidad')
        # except Exception as e:
        #     print("Error en limpieza de recomendaciones velocidad : " + str(e)) 

    ## Limpiar errrores

    dict_tipo_errores = {'Latitud null':1,'Longitud null':1,'toneladas 0 o menor a 0':2,'discharge_plant_name null':3,'discharge_start_date null':4,
                        'discharge_chata_name null':5,'discharge_line_name null':6,'discharge_poza_1 null':7,'horas desde primera cala mayor a 72':8,
                        'tiempo negativo':9,'eta_plant null':10,'mas de 0.5 horas':11,'first_cala_start_date null':12, 'eta plant no habilitada':13}
    df_emb_errores = pd.merge(df_emb_errores,df_embarcaciones[['marea_id','owner_group']],how='left',on='marea_id')
    # Eliminar mareas con igual boat_name, solo se conservara la ultima
    df_emb_errores_aux = df_emb_errores.groupby(['embarcaciones']).agg({'marea_id':'max'})
    df_emb_errores_aux.reset_index(inplace=True)
    df_emb_errores = df_emb_errores[df_emb_errores['marea_id'].isin(df_emb_errores_aux['marea_id'])]
    df_emb_errores['error_is_global'] = False
    df_emb_errores['global_error_description'] = np.nan
    df_emb_errores['error_category_id'] = df_emb_errores['tipo_error'].map(dict_tipo_errores).fillna(df_emb_errores['tipo_error'])
    df_emb_errores = df_emb_errores.rename(columns={'timestamp':'last_modification','tipo_emb':'recom_state','embarcaciones':'boat_name'})
    df_emb_errores = df_emb_errores[['error_is_global', 'global_error_description', 'marea_id', 'boat_name', 'recom_state','owner_group','error_category_id']]
    
    print('Fin return optimization')
    # print(df_sol_opt.head())
    # print(df_emb_errores.head())

    #df_sol_opt['orden_descarga_acumulado']=df_sol_opt['orden_descarga_global'].copy()
    df_sol_opt = df_sol_opt.sort_values(by=['planta_retorno','orden_descarga_global']).reset_index(drop=True)
    
    #Se reorganizan los órdenes para que inicien en uno en caso haya embarcaciones antes y despues de la fecha de corte, las de después de a fecha de corte tengan recomendaciones empezando en 1
    # fecha_analisis=timestamp.date()
    if os.getenv("ENVIRONMENT") == 'dev':
        fecha_analisis = datetime.datetime.fromisoformat(get_data.get_dump_data()['dump_date'].item()).date()
    else:
        fecha_analisis=timestamp.date()
    
    to_replace = {'PISCO': 'PISCO SUR'}
    df_mareas_cerradas.replace({'discharge_plant_name':to_replace}, inplace=True)
    df_embarcaciones_descargando.replace({'discharge_plant_name':to_replace}, inplace=True)
    
    # Eliminar columnas que causan conflicto de df_emb_descargando
    try:
        del df_embarcaciones_descargando['PORC_CAPACIDAD']
    except:
        pass
    try:
        del df_embarcaciones_descargando['tiempo_fin_descarga']
    except:
        pass
    try:
        del df_embarcaciones_descargando['tvn_emb_descargando']
    except:
        pass
    try:
        del df_embarcaciones_descargando['id_poza']
    except:
        pass
    try:
        del df_embarcaciones_descargando['id_temp']
    except:
        pass
    try:
        del df_embarcaciones_descargando['tipo_conservacion']
    except:
        pass
    
    df_desc_no_validas = df_desc_no_validas[df_embarcaciones_descargando.columns]
    df_desc_no_validas['discharge_plant_name'].replace('CHICAMA', 'MALABRIGO',inplace=True)
    df_desc_no_validas['discharge_plant_name'].replace('MOLLENDO', 'MATARANI',inplace=True)
    df_desc_no_validas = df_desc_no_validas[~df_desc_no_validas['marea_id'].isin(df_embarcaciones_descargando['marea_id'])]
    df_embarcaciones_descargando = pd.concat([df_embarcaciones_descargando, df_desc_no_validas], axis=0)

    # Eliminar mareas cerradas que esten siendo optimizadas (error del sp)
    mareas_optimizadas = df_sol_opt.marea_id.tolist()
    df_mareas_cerradas = df_mareas_cerradas[~df_mareas_cerradas['marea_id'].isin(mareas_optimizadas)]
    
    print('retorno utilidad')
    print(len(df_retorno_utilidad))
    # for i in df_embarcaciones_esperando_descarga.discharge_plant_name.unique():
    #     if i == 'PACIFICO CENTRO':
    #         continue
    #     # TODO: Monitorear si es que hay problemas en el cambio de dia (sobretodo si es que hay EPs de mas de un dia desde su arribo)
    #     # Se va a evaluar EPs hasta dos dias hacia atras
    #     hora_inicio_hoy=list(df_horas_produccion.loc[(df_horas_produccion.date_production==str(fecha_analisis))&(df_horas_produccion.id_plant==i),'hora'])[0]
    #     hora_inicio_prod_ayer=list(df_horas_produccion.loc[(df_horas_produccion.date_production==str((fecha_analisis+datetime.timedelta(days=-1))))&(df_horas_produccion.id_plant==i),'hora'])[0]
    #     # hora_inicio_prod_manana=list(df_horas_produccion.loc[(df_horas_produccion.date_production==str((fecha_analisis+datetime.timedelta(days=1))))&(df_horas_produccion.id_plant==i),'hora'])[0]
    #     hora_inicio_prod_anteayer=list(df_horas_produccion.loc[(df_horas_produccion.date_production==str((fecha_analisis+datetime.timedelta(days=-2))))&(df_horas_produccion.id_plant==i),'hora'])[0]
        
    #     # Se segmenta de acuerdo a si es que hay embarcaciones en (t-2) dias desde su arribo
    #     antes_corte_2d = len(df_embarcaciones_esperando_descarga[(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date<str(hora_inicio_prod_ayer))&(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date>=str(hora_inicio_prod_anteayer))&(df_embarcaciones_esperando_descarga.discharge_plant_name==i)])
    #     antes_de_corte = len(df_embarcaciones_esperando_descarga[(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date<str(hora_inicio_hoy))&(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date>=str(hora_inicio_prod_ayer))&(df_embarcaciones_esperando_descarga.discharge_plant_name==i)])
    #     despues_corte = len(df_embarcaciones_esperando_descarga[(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date>=str(hora_inicio_hoy))&(df_embarcaciones_esperando_descarga.discharge_plant_name==i)])
        
    #     # Se mapean las actualmente descargando con fecha de arribo a planta y orden acodere
    #     emb_desc_actual = pd.merge(df_embarcaciones_descargando,df_embarcaciones[['marea_id','discharge_plant_arrival_date','acodere_orden_descarga']],how='left',on='marea_id')
    #     # emb_desc_actual.rename(columns={'discharge_plant_arrival_date_y':'discharge_plant_arrival_date'}, inplace=True)
        
    #     primer_corte = 0
    #     corte_anterior = 0
        
    #     if antes_corte_2d>0:
    #         # Se busca el ultimo orden asignado para el dia de analisis
    #         emb_2_dias = df_mareas_cerradas[(df_mareas_cerradas.discharge_plant_arrival_date<str(hora_inicio_prod_ayer)) & (df_mareas_cerradas.discharge_plant_arrival_date>=str(hora_inicio_prod_anteayer)) & (df_mareas_cerradas.discharge_plant_name==i)]
    #         # emb_2_dias = emb_2_dias[emb_2_dias['acodere_orden_descarga'].notnull()]
    #         mask = (emb_2_dias['declared_ton'] > 0) & (emb_2_dias['acodere_orden_descarga'].isna())
    #         mask_1 = (emb_2_dias['declared_ton'] > 0)
            
    #         # Cantidad de mareas cerradas sin orden
    #         # n_nans_cerradas = len(emb_2_dias[mask].index)
            
    #         # Mareas cerradas
    #         emb_2_validas = emb_2_dias[mask_1]
            
    #         # Mareas cerradas con orden
    #         # mask = emb_2_dias['acodere_orden_descarga'].notna()
    #         # emb_2_dias = emb_2_dias[mask].reset_index(drop=True)            
            
    #         # Se busca si es que hay EPs descargando provenientes del dia de analisis
    #         emb_desc_2_dias = emb_desc_actual[(emb_desc_actual.discharge_plant_arrival_date<str(hora_inicio_prod_ayer))&(emb_desc_actual.discharge_plant_arrival_date>=str(hora_inicio_prod_anteayer))&(emb_desc_actual.discharge_plant_name==i)]

    #         # Mareas descargando sin orden
    #         # mask = (emb_desc_2_dias['declared_ton'] > 0) & (emb_desc_2_dias['acodere_orden_descarga'].isna())
    #         # n_nans_descar = len(emb_desc_2_dias[mask].index)
    #         # emb_desc_2_dias['acodere_orden_descarga'] = emb_desc_2_dias['acodere_orden_descarga'].fillna(0)
            
    #         # if len(emb_2_dias)>0:
    #         #     max_orden = int(emb_2_dias.acodere_orden_descarga.max())
    #         # else:
    #         #     max_orden = 0
            
    #         # Totalidad mareas cerradas
    #         if (len(emb_2_validas)>0):
    #             max_orden = len(emb_2_validas)
    #         else:
    #             max_orden = 0
            
    #         # max_orden = max_orden + n_nans_cerradas
                
    #         if len(emb_desc_2_dias)>0:
    #             # max_orden_desc = int(emb_desc_2_dias.acodere_orden_descarga.max())
    #             max_orden_desc = len(emb_desc_2_dias)
    #         else:
    #             max_orden_desc = 0
                
    #         # max_orden_desc = max_orden_desc + n_nans_descar
            
    #         # Ordenes anteriores para EP (t-2), se escoge el maximo entre los que estan descargando y las mareas cerradas
    #         # orden_2_dias = np.maximum(max_orden,max_orden_desc) + n_nans_cerradas + n_nans_descar
    #         orden_2_dias = max_orden + max_orden_desc
            
    #         # Adicionar ordenes anteriores directo al dataframe de soluciones
    #         total_emb_2d = df_embarcaciones_esperando_descarga[(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date<str(hora_inicio_prod_ayer))&(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date>=str(hora_inicio_prod_anteayer))&(df_embarcaciones_esperando_descarga.discharge_plant_name==i)]
    #         emb_antes_corte_2d = df_sol_opt[df_sol_opt.marea_id.isin(total_emb_2d)].sort_values(by='orden_descarga_global').reset_index(drop=True).copy()
            
    #         for index,row in emb_antes_corte_2d.iterrows():
    #             df_sol_opt.loc[df_sol_opt.marea_id==row.marea_id,'orden_descarga_global'] = index + orden_2_dias + 1
    #             orden_original = df_tabla_utilidad.loc[df_tabla_utilidad.marea_id==row.marea_id,'orden_emb']
    #             df_tabla_utilidad.loc[df_tabla_utilidad.marea_id==row.marea_id,'orden_emb'] = orden_original + orden_2_dias + 1
            
    #         corte_anterior = len(emb_antes_corte_2d)
        
    #     # Se segmenta para embarcaciones del dia anterior
    #     if antes_de_corte>0:
    #         # Se busca el ultimo orden asignado para el dia de analisis
    #         emb_1_dia = df_mareas_cerradas[(df_mareas_cerradas.discharge_plant_arrival_date<str(hora_inicio_hoy)) & (df_mareas_cerradas.discharge_plant_arrival_date>=str(hora_inicio_prod_ayer)) & (df_mareas_cerradas.discharge_plant_name==i)]
    #         # emb_1_dia = emb_1_dia[emb_1_dia['acodere_orden_descarga'].notnull()].reset_index(drop=True)
    #         mask = (emb_1_dia['declared_ton'] > 0) & (emb_1_dia['acodere_orden_descarga'].isna())
    #         mask_1 = (emb_1_dia['declared_ton'] > 0)
            
    #         # Cantidad de mareas cerradas sin orden
    #         n_nans_cerradas = len(emb_1_dia[mask].index)
    #         # Cantidad de mareas cerradas
    #         emb_1_validas = emb_1_dia[mask_1]
            
    #         # Mareas cerradas con orden
    #         mask = emb_1_dia['acodere_orden_descarga'].notna()
    #         emb_1_dia = emb_1_dia[mask].reset_index(drop=True)
            
    #         # Se busca si es que hay EPs descargando provenientes del dia de analisis
    #         emb_desc_1_dia = emb_desc_actual[(emb_desc_actual.discharge_plant_arrival_date<str(hora_inicio_hoy))&(emb_desc_actual.discharge_plant_arrival_date>=str(hora_inicio_prod_ayer))&(emb_desc_actual.discharge_plant_name==i)].reset_index(drop=True)
            
    #         # Mareas descargando sin orden
    #         mask = (emb_desc_1_dia['declared_ton'] > 0) & (emb_desc_1_dia['acodere_orden_descarga'].isna())
    #         n_nans_descar = len(emb_desc_1_dia[mask].index)
    #         emb_desc_1_dia['acodere_orden_descarga'] = emb_desc_1_dia['acodere_orden_descarga'].fillna(0)
            
    #         # Mareas con fecha acodere ingresada (excluidas de optimizacion)
    #         # acodere_antes_corte = df_adic[(df_adic.discharge_plant_arrival_date<str(hora_inicio_hoy))&(df_adic.discharge_plant_arrival_date>=str(hora_inicio_prod_ayer))&(df_adic.discharge_plant_name==i)]
            
    #         # if len(emb_1_dia)>0:
    #         #     max_orden = int(emb_1_dia.acodere_orden_descarga.max())
    #         # else:
    #         #     max_orden = 0
            
    #         # Conteo de mareas cerradas
    #         if (len(emb_1_validas)>0):
    #             max_orden = len(emb_1_validas)
    #         else:
    #             max_orden = 0
                
    #         if len(emb_desc_1_dia)>0:
    #             # max_orden_desc = int(emb_desc_1_dia.acodere_orden_descarga.max())
    #             max_orden_desc = len(emb_desc_1_dia)
    #         else:
    #             max_orden_desc = 0
            
    #         # if len(acodere_antes_corte)>0:
    #         #     n_acodere = len(acodere_antes_corte)
    #         # else:
    #         #     n_acodere = 0
                
    #         # max_orden_desc = max_orden_desc + n_nans_descar
            
    #         # Ordenes anteriores para EP (t-1), se escoge el maximo entre los que estan descargando y las mareas cerradas
    #         # orden_1_dia = np.maximum(max_orden,max_orden_desc) + n_nans_cerradas + n_nans_descar
    #         orden_1_dia = max_orden + max_orden_desc
            
    #         # Adicionar ordenes anteriores directo al dataframe de soluciones
    #         total_emb_1d = df_embarcaciones_esperando_descarga[(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date<str(hora_inicio_hoy))&(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date>=str(hora_inicio_prod_ayer))&(df_embarcaciones_esperando_descarga.discharge_plant_name==i)].marea_id.unique()
    #         emb_antes_corte_1d = df_sol_opt[df_sol_opt.marea_id.isin(total_emb_1d)].sort_values(by='orden_descarga_global').reset_index(drop=True).copy()
            
    #         for index,row in emb_antes_corte_1d.iterrows():
    #             df_sol_opt.loc[df_sol_opt.marea_id==row.marea_id,'orden_descarga_global'] = index + orden_1_dia + 1
    #             orden_original = df_tabla_utilidad.loc[df_tabla_utilidad.marea_id==row.marea_id,'orden_emb']
    #             df_tabla_utilidad.loc[df_tabla_utilidad.marea_id==row.marea_id,'orden_emb'] = orden_original + orden_1_dia + 1 - corte_anterior
            
    #         primer_corte = len(emb_antes_corte_1d)
                
    #     if despues_corte>0:
    #         # Se busca el ultimo orden asignado para el dia de analisis
    #         emb_hoy = df_mareas_cerradas[(df_mareas_cerradas.discharge_plant_arrival_date>=str(hora_inicio_hoy)) & (df_mareas_cerradas.discharge_plant_name==i)]
    #         # emb_hoy = emb_hoy[emb_hoy['acodere_orden_descarga'].notnull()].reset_index(drop=True)
    #         mask = (emb_hoy['declared_ton'] > 0) & (emb_hoy['acodere_orden_descarga'].isna())
    #         mask_1 = (emb_hoy['declared_ton'] > 0)
            
    #         # Cantidad de mareas sin orden
    #         n_nans_cerradas = len(emb_hoy[mask].index)
    #         # Mareas cerradas con orden
    #         mask = emb_hoy['acodere_orden_descarga'].notna()
    #         df_last_emb = emb_hoy[mask].reset_index(drop=True)
            
    #         # Nuevo
    #         emb_hoy_validas = emb_hoy[mask_1]
            
    #         # Se busca si es que hay EPs descargando provenientes del dia de analisis
    #         emb_desc_hoy = emb_desc_actual[(emb_desc_actual.discharge_plant_arrival_date>=str(hora_inicio_hoy))&(emb_desc_actual.discharge_plant_name==i)].reset_index(drop=True)
            
    #         # Mareas descargando sin orden
    #         mask = (emb_desc_hoy['declared_ton'] > 0) & (emb_desc_hoy['acodere_orden_descarga'].isna())
    #         n_nans_descar = len(emb_desc_hoy[mask].index)
    #         emb_desc_hoy['acodere_orden_descarga'] = emb_desc_hoy['acodere_orden_descarga'].fillna(0)
            
            
    #         # if (len(df_last_emb)>0):
    #         #     max_orden = int(df_last_emb.acodere_orden_descarga.max())
    #         # else:
    #         #     max_orden = 0
            
    #         if (len(emb_hoy_validas)>0):
    #             max_orden = len(emb_hoy_validas)
    #         else:
    #             max_orden = 0

    #         if len(emb_desc_hoy)>0:
    #             # max_orden_desc = int(emb_desc_hoy.acodere_orden_descarga.max())
    #             max_orden_desc = len(emb_desc_hoy)
    #         else:
    #             max_orden_desc = 0
            
    #         # max_orden_desc = max_orden_desc + n_nans_descar
            
    #         # Ordenes anteriores para EP (t), se escoge el maximo entre los que estan descargando y las mareas cerradas
    #         # orden_hoy = np.maximum(max_orden,max_orden_desc) + n_nans_cerradas + n_nans_descar
    #         orden_hoy = max_orden + max_orden_desc
            
    #         # Adicionar ordenes anteriores directo al dataframe de soluciones
    #         total_emb_hoy = df_embarcaciones_esperando_descarga[(df_embarcaciones_esperando_descarga.discharge_plant_arrival_date>=str(hora_inicio_hoy))&(df_embarcaciones_esperando_descarga.discharge_plant_name==i)].marea_id.unique()
    #         emb_despues_corte = df_sol_opt[df_sol_opt.marea_id.isin(total_emb_hoy)].sort_values(by='orden_descarga_global').reset_index(drop=True).copy()
            
    #         for index,row in emb_despues_corte.iterrows():
    #             df_sol_opt.loc[df_sol_opt.marea_id==row.marea_id,'orden_descarga_global'] = index + orden_hoy + 1
    #             orden_original = df_tabla_utilidad.loc[df_tabla_utilidad.marea_id==row.marea_id,'orden_emb']
    #             df_tabla_utilidad.loc[df_tabla_utilidad.marea_id==row.marea_id,'orden_emb'] = orden_original + orden_hoy + 1 - corte_anterior - primer_corte
    
    # print('tabla utilidad')
    # print(df_tabla_utilidad)
    # if len(df_sol_opt[df_sol_opt.velocidad_retorno.notnull()])>0:
    #     nuevas_velocidades=recomendacion_final_velocidades(df_sol_opt,df_embarcaciones,df_embarcaciones_retornando,df_embarcaciones_terceros,df_pozas_estado_esp_desc,df_requerimiento_plantas,df_tiempo_descarga,timestamp)
    #     for z in nuevas_velocidades.marea:
    #         df_sol_opt.loc[df_sol_opt.marea_id==z,'velocidad_retorno']=nuevas_velocidades[nuevas_velocidades.marea==z].velocidad.values[0]
    df_sol_opt=df_sol_opt.merge(df_chatas_lineas.drop_duplicates(subset=['id_chata'])[['id_chata','name']],how='left',left_on='chata_descarga',right_on='id_chata')
    df_sol_opt.drop(columns=['chata_descarga','id_chata'],inplace=True)
    df_sol_opt.rename(columns={'name':'chata_descarga'},inplace=True)

    df_sol_opt.loc[(df_sol_opt.orden_descarga_global.isnull())|(df_sol_opt.poza_descarga_1.isnull()),['tons_poza_descarga_1','tons_poza_descarga_2','tons_poza_descarga_3','tons_poza_descarga_4',
                                                     'poza_descarga_1','poza_descarga_2','poza_descarga_3', 'poza_descarga_4']]=np.nan

    # Hacer join para agregar el flag de exceso de capacidad
    # habilitar_stop_rec = True
    
    # Agregar flag de planta llena si solo hay EPs retornando
    # if ('flag_planta_llena' not in df_sol_opt) & ('flag_planta_llena' not in df_tabla_utilidad):
    #     print('agregar sol opt')
    #     df_sol_opt['flag_planta_llena'] = False
    
    if ('flag_planta_llena' not in df_sol_opt):
        df_sol_opt['flag_planta_llena'] = False
        
    # print(df_sol_opt.columns)
    # print(df_tabla_utilidad.columns)
    
    # if ('flag_planta_llena' not in df_sol_opt):
    #     print('ok')
    #     df_utilidad_opt = df_tabla_utilidad.loc[df_tabla_utilidad['id_orden']==1,['marea_id','flag_planta_llena']]
    #     df_sol_opt = df_sol_opt.merge(df_utilidad_opt,on='marea_id', how='left')
        
    #     # Limpiar recomendaciones cuando la capacidad se sobrepasa
    #     for index,row in df_sol_opt.iterrows():
    #         if row['flag_planta_llena']==True:
    #             df_sol_opt.loc[index,'chata_descarga'] = np.nan
    #             df_sol_opt.loc[index,'linea_descarga'] = np.nan
    #             df_sol_opt.loc[index,'orden_descarga'] = np.nan
    #             df_sol_opt.loc[index,'orden_descarga_global'] = np.nan
    #             df_sol_opt.loc[index,'tons_poza_descarga_1'] = np.nan
    #             df_sol_opt.loc[index,'poza_descarga_1'] = np.nan
    #             df_sol_opt.loc[index,'tons_poza_descarga_2'] = np.nan
    #             df_sol_opt.loc[index,'poza_descarga_2'] = np.nan
    #             df_sol_opt.loc[index,'tons_poza_descarga_3'] = np.nan
    #             df_sol_opt.loc[index,'poza_descarga_3'] = np.nan
    #             df_sol_opt.loc[index,'tons_poza_descarga_4'] = np.nan
    #             df_sol_opt.loc[index,'poza_descarga_4'] = np.nan
        
    # Eliminar columna de flag de la tabal de utilidad
    try:
        del df_tabla_utilidad['flag_planta_llena']
    except:
        pass
    # df_tabla_utilidad.to_csv('outputs/df_tabla_utilidad_chata_V2.csv')
    return df_sol_opt, df_emb_errores, df_flags_cabecera, df_flags_ordenes, df_tabla_utilidad, df_retorno_utilidad

