import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import os
import sys
import warnings
from functools import partial
from data.get_dumped_data import get_dumped_data_date
from models.tvn_prediction import get_mareas_return_tvn_prediction, add_tvn_poza_slope_increase, get_tvn_poza_model_slope_increase, get_tvn_increase
warnings.filterwarnings("ignore")

# FUNCIONES DE TRATAMIENTO DE DATOS

def get_pozas_hermanadas_per_plant(df_pozas_hermanadas):
    dict_hermanas = {}
    for planta in list(df_pozas_hermanadas['id_planta'].unique()):
        df_temp = df_pozas_hermanadas[(df_pozas_hermanadas['id_planta'] == planta) & (df_pozas_hermanadas['es_hermanada'] == 1)]
        df_temp['poza_hermanda'] = df_temp['poza_hermanda'].map(int)
        temp_lista = df_temp[['pozaNumber', 'poza_hermanda']].values.tolist()
        lista_hermanas = [list(sorted(x)) for x in {frozenset(x) for x in temp_lista}]
        dict_hermanas[planta] = lista_hermanas
    return dict_hermanas


def get_current_tdc_from_first_cala(cala):
    env_argument_param = os.getenv("ENVIRONMENT")
    if env_argument_param == 'dumped':
        date_now = get_dumped_data_date()
    else:
        date_now = datetime.utcnow()

    # Esto es requerido porque algunas calas se ingresan con año erróneo. Se exceptúa en las fechas cerca a fin de año.
    if not ((date_now.month == 12 and date_now.day >= 29) or (date_now.month == 1 and date_now.day <= 3)):
        cala = cala.replace(year=date_now.year)

    return (date_now - cala).total_seconds() / 3600


def get_hours_passed_from_date(comparison_date):
    if comparison_date is not None:
        env_argument_param = os.getenv("ENVIRONMENT")
        if env_argument_param == 'dumped':
            date_now = get_dumped_data_date()
        else:
            date_now = datetime.utcnow()

        return (date_now - comparison_date).total_seconds() / 3600
    else:
        return None


def replace_discharge_ton(pozas_estado_row):
    if pozas_estado_row['divided_actual_ton_discharge'] is not None and pozas_estado_row['divided_actual_ton_discharge'] > 0:
        pozas_estado_row['declared_ton'] = pozas_estado_row['divided_actual_ton_discharge']
        return pozas_estado_row
    else:
        pozas_estado_row['declared_ton'] = pozas_estado_row['divided_declared_ton']
        return pozas_estado_row


def limpiar_data_pozas(df_pozas_estado, df_plants_info, df_plantas_habilitadas):
    df_pozas_estado['id'] = df_pozas_estado['id_planta'] + '-' + df_pozas_estado['pozaNumber'].map(str)

    df_pozas_estado = df_pozas_estado.reset_index(drop=True)

    # Filtering to get only relevant discharges
    # TODO: Check for cases with stock and no marea_id was found
    df_pozas_estado = df_pozas_estado[(df_pozas_estado['marea_id'].notna()) & (df_pozas_estado['stock_actual'] > 0) & (df_pozas_estado['declared_ton'] > 0) & (df_pozas_estado['fin_cocina'].isnull())]

    # Poza is in plantas habilitadas list
    df_pozas_estado = df_pozas_estado[df_pozas_estado['id_planta'].isin(list(df_plants_info['id']))]

    df_pozas_estado['first_cala_start_date'] = pd.to_datetime(df_pozas_estado['first_cala_start_date'])
    df_pozas_estado['discharge_start_date'] = pd.to_datetime(df_pozas_estado['discharge_start_date'])
    df_pozas_estado['discharge_end_date'] = pd.to_datetime(df_pozas_estado['discharge_end_date'])
    df_pozas_estado['tdc_actual'] = df_pozas_estado['first_cala_start_date'].apply(get_current_tdc_from_first_cala)
    df_pozas_estado['stock_actual_update_date'] = pd.to_datetime(df_pozas_estado['stock_actual_update_date'])
    df_pozas_estado['stock_actual_hours_passed'] = df_pozas_estado['stock_actual_update_date'].apply(get_hours_passed_from_date)

    df_pozas_estado['inicio_cocina'] = pd.to_datetime(df_pozas_estado['inicio_cocina'])
    df_pozas_estado['cook_start_hours_passed'] = df_pozas_estado['inicio_cocina'].apply(get_hours_passed_from_date)

    df_pozas_estado['% Llenado'] = df_pozas_estado['pp_llenado_bodega']
    df_pozas_estado['Descarg.'] = df_pozas_estado['divided_declared_ton']
    df_pozas_estado['# Calas'] = df_pozas_estado['calas_count']

    df_pozas_estado['working_frio'] = (df_pozas_estado['tipo_bodega'] == 'Frio') & (
                (df_pozas_estado['frio_system_state'] == 'RC') | (df_pozas_estado['frio_system_state'] == 'CH') | (df_pozas_estado['frio_system_state'] == 'GF'))

    df_pozas_estado = get_mareas_return_tvn_prediction(df_pozas_estado)

    df_pozas_estado = add_tvn_poza_slope_increase(df_pozas_estado)

    DISCHARGE_CONTINGENCIA_TDC_LIMIT = 36
    df_pozas_estado['traditional_discharge_contingencia'] = 0
    df_pozas_estado['traditional_discharge_contingencia'][(df_pozas_estado['working_frio'] != 1) & (df_pozas_estado['tdc_discharge'] >= DISCHARGE_CONTINGENCIA_TDC_LIMIT)] = 1

    # Any of this limits for contingencia_in_poza is considered contingencia in poza and won't be recommended
    df_pozas_estado['contingencia_in_poza'] = 0
    
    # Join to plants with contingency logic activated
    df_pozas_estado = df_pozas_estado.merge(df_plantas_habilitadas[['planta','habilitada_contingencia']], how='left', left_on='id_planta', right_on='planta')
    del df_pozas_estado['planta']
    
    # TODO: For 2022-T2, the new rule is to delete contingency filters, the limits for TDC and TVN will be high, so if they decide to return to this solution the change will be easier
    TRADITIONAL_TDC_IN_POZA_CONTINGENCIA_DOWN = 100 #36
    TRADITIONAL_TDC_IN_POZA_CONTINGENCIA_UP = 36
    df_pozas_estado['TRADITIONAL_TDC_IN_POZA_CONTINGENCIA'] = np.where(df_pozas_estado['habilitada_contingencia']==True, TRADITIONAL_TDC_IN_POZA_CONTINGENCIA_UP, TRADITIONAL_TDC_IN_POZA_CONTINGENCIA_DOWN)
    df_pozas_estado['contingencia_in_poza'][(df_pozas_estado['working_frio'] != 1) & (df_pozas_estado['tdc_actual'] >= df_pozas_estado['TRADITIONAL_TDC_IN_POZA_CONTINGENCIA'])] = 1

    TVN_IN_POZA_CONTINGENCIA_DOWN = 200 #60
    TVN_IN_POZA_CONTINGENCIA_UP = 60
    df_pozas_estado['TVN_IN_POZA_CONTINGENCIA'] = np.where(df_pozas_estado['habilitada_contingencia']==True, TVN_IN_POZA_CONTINGENCIA_UP, TVN_IN_POZA_CONTINGENCIA_DOWN)
    df_pozas_estado['contingencia_in_poza'][(df_pozas_estado['tvn'] >= df_pozas_estado['TVN_IN_POZA_CONTINGENCIA']) & (df_pozas_estado['contingencia_in_poza']==0)] = 1

    FRIO_TDC_CONTINGENCIA = 100 #36
    FRIO_CONTINGENCIA_HOURS_IN_POZA = 3
    # df_pozas_estado['contingencia_in_poza'][(df_pozas_estado['working_frio'] == 1) & (df_pozas_estado['tdc_actual'] >= FRIO_TDC_CONTINGENCIA)
    #                                         & ((df_pozas_estado['tdc_actual'] - df_pozas_estado['tdc_discharge']) >= FRIO_CONTINGENCIA_HOURS_IN_POZA)] = 1
    
    # Delete new columns on df_pozas_estado
    del df_pozas_estado['habilitada_contingencia']
    del df_pozas_estado['TRADITIONAL_TDC_IN_POZA_CONTINGENCIA']
    del df_pozas_estado['TVN_IN_POZA_CONTINGENCIA']
    
    # Limits where recommendation is automatically given
    TRADITIONAL_TDC_IN_POZA_LIMIT = 30
    df_pozas_estado['tdc_in_poza_tradicional_warning_limit'] = 0
    # df_pozas_estado['tdc_in_poza_tradicional_warning_limit'][(df_pozas_estado['working_frio'] != 1) & (df_pozas_estado['tdc_actual'] >= TRADITIONAL_TDC_IN_POZA_LIMIT)] = 1

    TVN_IN_POZA_LIMIT = 50
    df_pozas_estado['tvn_in_poza_warning_limit'] = 0
    # df_pozas_estado['tvn_in_poza_warning_limit'][df_pozas_estado['tvn'] >= TVN_IN_POZA_LIMIT] = 1

    # Si una EP frio descarga con > de 34 horas y

    FRIO_LINE_TDC_LIMIT = 36
    FRIO_LINE_HOURS_IN_POZA_LIMIT = 1
    df_pozas_estado['tdc_linea_frio'] = 0
    # df_pozas_estado['tdc_linea_frio'][(df_pozas_estado['working_frio'] == 1) & (df_pozas_estado['tdc_actual'] >= FRIO_LINE_TDC_LIMIT)
    #                                   & ((df_pozas_estado['tdc_actual'] - df_pozas_estado['tdc_discharge']) >= FRIO_LINE_HOURS_IN_POZA_LIMIT)] = 1

    # Si una descarga fria con > 36TDC y supera las 2.5 horas en poza es contingencia y ya no se recomienda

    APROVECHAMIENTO_FRIO_HOURS_IN_POZA = 5
    df_pozas_estado['aprovechamiento_frio'] = 0
    # df_pozas_estado['aprovechamiento_frio'][(df_pozas_estado['working_frio'] == 1)
    #                                         & ((df_pozas_estado['tdc_actual'] - df_pozas_estado['tdc_discharge']) >= APROVECHAMIENTO_FRIO_HOURS_IN_POZA)] = 1

    df_pozas_estado['divided_declared_ton'] = df_pozas_estado['divided_declared_ton'].fillna(0)

    # df_pozas_estado = df_pozas_estado.apply(replace_discharge_ton, axis=1)
    # df_pozas_estado['declared_ton'] = df_pozas_estado['stock_actual']

    # df_pozas_estado['declared_ton'] = df_pozas_estado['divided_declared_ton']

    # agregar descargas en pozas y quedarme con columnas relevantes
    df_pozas_estado = df_pozas_estado[
        ['id', 'declared_ton', 'tvn', 'id_planta', 'pozaNumber', 'pozaCapacity', 'cocinandose', 'cook_start_hours_passed', 'stock_poza_inicial', 'perc_poza_cocina', 'tdc_discharge', 'tdc_actual',
         'tdc_in_poza_tradicional_warning_limit', 'tvn_in_poza_warning_limit', 'tdc_linea_frio', 'aprovechamiento_frio', 'traditional_discharge_contingencia', 'stock_actual',
         'stock_actual_hours_passed', 'contingencia_in_poza', 'con_hielo', 'working_frio']
    ]

    df_pozas_estado['declared_ton'] = df_pozas_estado['declared_ton'].fillna(0)
    df_pozas_estado['tvn'][df_pozas_estado['declared_ton'] == 0] = 0
    df_pozas_estado['tvn_ton'] = df_pozas_estado['declared_ton'] * df_pozas_estado['tvn']

    df_pozas_estado = df_pozas_estado.groupby('id').agg(
        {'declared_ton': 'sum', 'tvn_ton': 'sum', 'id_planta': 'first', 'pozaNumber': 'first', 'pozaCapacity': 'first', 'cocinandose': 'max', 'cook_start_hours_passed': 'max',
         'stock_poza_inicial': 'max', 'perc_poza_cocina': 'max', 'tdc_discharge': 'max', 'tdc_actual': 'max', 'tdc_in_poza_tradicional_warning_limit': 'max', 'tvn_in_poza_warning_limit': 'max',
         'tdc_linea_frio': 'max',
         'aprovechamiento_frio': 'max', 'traditional_discharge_contingencia': 'max', 'stock_actual': 'max', 'stock_actual_hours_passed': 'max', 'contingencia_in_poza': 'max',
         'con_hielo': 'max', 'working_frio': 'max'}).reset_index()

    df_pozas_estado['tvn'] = df_pozas_estado['tvn_ton'] / df_pozas_estado['declared_ton']
    del df_pozas_estado['tvn_ton']
    df_pozas_estado['tvn'] = df_pozas_estado['tvn'].fillna(0)
    return df_pozas_estado


#################################################################################################


# FUNCIONES DE OPTIMIZACION

# Funciones para crear el dataframe de las potenciales combinaciones de pozas

def get_pozas_possible_combinations(pozas_numbers_list):
    return pozas_numbers_list + [list(x) for x in combinations(pozas_numbers_list, 2)]


# Está función te retorna las posibles combinaciones de pozas excluyendo una.
def get_pozas_combinations_excluding_one(pozas_combinations_list, poza_combination_to_exclude):
    if type(poza_combination_to_exclude) is list:
        if len(get_pozas_possible_combinations([x for x in pozas_combinations_list if x not in poza_combination_to_exclude])) == 0:
            return [0]
        else:
            return get_pozas_possible_combinations([x for x in pozas_combinations_list if x not in poza_combination_to_exclude])
    else:
        lista_upd = pozas_combinations_list.copy()
        lista_upd.remove(poza_combination_to_exclude)
        if len(get_pozas_possible_combinations(lista_upd)) == 0:
            return [np.nan]
        else:
            return get_pozas_possible_combinations(lista_upd)


def convert_to_list(x):
    if type(x) is list:
        return x
    else:
        return [x]


CURRENTLY_COOKING_STATES = {
    'NO_COOKING_POZAS': 'NO_COOKING_POZAS',
    'ONE_COOKING_POZA': 'ONE_COOKING_POZA',
    'MULTIPLE_COOKING_POZAS': 'MULTIPLE_COOKING_POZAS',
    'ERROR': 'ERROR'
}


def get_currently_cooking_pozas_state(df_pozas_currently_cooking):
    global CURRENTLY_COOKING_STATES

    if len(df_pozas_currently_cooking) == 0:
        return CURRENTLY_COOKING_STATES['NO_COOKING_POZAS']
    elif len(df_pozas_currently_cooking) == 1:
        return CURRENTLY_COOKING_STATES['ONE_COOKING_POZA']
    elif len(df_pozas_currently_cooking) >= 2:
        return CURRENTLY_COOKING_STATES['MULTIPLE_COOKING_POZAS']
    else:
        return CURRENTLY_COOKING_STATES['ERROR']


# This will find the hours to finish cooking except the last one (poza_fixed_cooking) for recom type Mezcla Actual
def get_currently_cooking_pozas_hours_to_finish_all(df_pozas_currently_cooking, vel_cocina, poza_fixed_cooking):
    hours_to_finish_all = 0

    for index, poza in df_pozas_currently_cooking.iterrows():
        if poza['pozaNumber'] == poza_fixed_cooking:
            continue

        # remaining_stock = poza['stock_poza_inicial'] - vel_cocina * poza['cook_start_hours_passed'] * (poza['perc_poza_cocina'] / 100)
        remaining_stock = poza['stock_actual'] - vel_cocina * poza['stock_actual_hours_passed'] * (poza['perc_poza_cocina'] / 100)
        hours_to_finish_poza = remaining_stock / (vel_cocina * (poza['perc_poza_cocina'] / 100))
        if hours_to_finish_poza > hours_to_finish_all:
            hours_to_finish_all = hours_to_finish_poza

    return hours_to_finish_all


# funcion para calcular utilidad de cocinar una poza o una combinacion de poza
# incr --> Es un factor de incremento de TVN de las pozas que estas dejando de lado tvn = tvn * (1 + (incr * marg)) que al final refleja en la utilidad
# marg --> Indica el tiempo que estas dejando la poza de lado; es relevante si la poza a alimentar es tiempo 1 o mayor ya que en tiempo 0 este valor es 0.vel_cocina,calidad_precio
def calcular_utilidad(df_plant_pozas, poza, calidad_precio, hours_to_wait_before_cook, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist):
    TVN_MARGINAL_RAISE_PER_HOUR = get_tvn_poza_model_slope_increase()

    lim_sp = df_plant_info['lim_tvn_cocina_super_prime']
    lim_p = df_plant_info['lim_tvn_cocina_prime']
    vel_cocina = df_plant_info['velocidad']

    # if vel_cocina is None or vel_cocina <= 0:
    #     vel_cocina = df_plant_info['velocidad_por_ton_cocina'].values[0]
    
    if vel_cocina is None or vel_cocina <= 0:
        vel_cocina = df_filtro_hist['velocidad'].values[0]

    hours_to_finish_cooking_current_pozas = get_currently_cooking_pozas_hours_to_finish_all(df_pozas_currently_cooking, vel_cocina, poza_fixed_cooking)

    # df_plant_pozas['tvn_ton'] = df_plant_pozas['stock_actual'] * df_plant_pozas['tvn']

    poza = eval(poza)
    if type(poza) is list:
        pozas_cook_tons = 0
        total_tvn_times_stock = 0
        for index, row in df_plant_pozas[df_plant_pozas['pozaNumber'].isin(poza)].iterrows():
            if row['pozaNumber'] == poza_fixed_cooking:
                remaining_stock = row['stock_actual'] - vel_cocina * row['stock_actual_hours_passed'] * (row['perc_poza_cocina'] / 100)
            else:
                remaining_stock = row['stock_actual']

            cook_duration = remaining_stock / (vel_cocina * 0.5) # TODO: Agregar condicional y revisar/eliminar el 0.5. Chekear si se cumple varias pozas
            tvn_with_cook_duration = row['tvn'] + get_tvn_increase(cook_duration + (hours_to_wait_before_cook + hours_to_finish_cooking_current_pozas), row['working_frio'], row['con_hielo'], row['id_planta'])
            pozas_cook_tons = pozas_cook_tons + remaining_stock
            total_tvn_times_stock = total_tvn_times_stock + (tvn_with_cook_duration * remaining_stock)

        pozas_cook_weighted_tvn = total_tvn_times_stock / pozas_cook_tons
        #df_plant_pozas['tvn_ton'] = df_plant_pozas['stock_actual'] * df_plant_pozas['tvn_with_cook_duration']
        #pozas_cook_tons = df_plant_pozas[df_plant_pozas['pozaNumber'].isin(poza)]['stock_actual'].sum()
        #pozas_cook_weighted_tvn = df_plant_pozas[df_plant_pozas['pozaNumber'].isin(poza)]['tvn_ton'].sum() / pozas_cook_tons
    else:
        # Poza equal to 0 indicates there are no more pozas left for this time so returns no utility and no added cook time
        if poza == 0:
            return 0, 0, 0, 0, 0
        poza_candidate = df_plant_pozas[df_plant_pozas['pozaNumber'] == poza].reset_index(drop=True).iloc[0]
        cook_duration = poza_candidate['stock_actual'] / vel_cocina
        pozas_cook_tons = poza_candidate['stock_actual']
        pozas_cook_weighted_tvn = poza_candidate['tvn'] + get_tvn_increase(cook_duration + (hours_to_wait_before_cook + hours_to_finish_cooking_current_pozas), poza_candidate['working_frio'], poza_candidate['con_hielo'], poza_candidate['id_planta'])

        # pozas_cook_tons = df_plant_pozas[df_plant_pozas['pozaNumber'] == poza]['stock_actual'].reset_index(drop=True)[0]
        # pozas_cook_weighted_tvn = df_plant_pozas[df_plant_pozas['pozaNumber'] == poza]['tvn'].reset_index(drop=True)[0]

    # pozas_cook_weighted_tvn = pozas_cook_weighted_tvn + (hours_to_wait_before_cook + hours_to_finish_cooking_current_pozas)

    if pozas_cook_weighted_tvn <= lim_sp:
        price_per_ton = calidad_precio[0]
        mix_quality = 'SP'
    elif pozas_cook_weighted_tvn <= lim_p:
        price_per_ton = calidad_precio[1]
        mix_quality = 'P'
    else:
        price_per_ton = calidad_precio[2]
        mix_quality = 'Otros'

    utilidad = pozas_cook_tons * price_per_ton

    hours_to_wait_for_next_cook = pozas_cook_tons / vel_cocina

    # Retornar: pozas_cook_weighted_tvn, the quality (SP, P or others), pozas_cook_tons
    return utilidad, hours_to_wait_for_next_cook, pozas_cook_weighted_tvn, mix_quality, pozas_cook_tons


def filter_for_fixed_recom_poza(poza_number_to_fix, pozas_combination):
    if type(pozas_combination) is list:
        return poza_number_to_fix in pozas_combination
    else:
        return pozas_combination == poza_number_to_fix


# When getting fixed cooking poza combination you have to mix the cooking poza with something (you cannot recommend only the cooking poza)
def filter_for_fixed_cooking_poza(cooking_poza_number_to_fix, pozas_combination):
    if type(pozas_combination) is list:
        return cooking_poza_number_to_fix in pozas_combination
    else:
        return False


def create_comb_up_to_time1(df_planta_pozas_to_optimize, calidad_precio, poza_fixed_for_recom, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist):
    pozas_numbers_list = df_planta_pozas_to_optimize['pozaNumber'].tolist()
    pozas_numbers_possible_combinations = get_pozas_possible_combinations(pozas_numbers_list)

    if poza_fixed_for_recom is not None and poza_fixed_cooking is not None:
        pozas_numbers_possible_combinations = [[poza_fixed_for_recom, poza_fixed_cooking]]
    elif poza_fixed_for_recom is not None and poza_fixed_cooking is None:
        pozas_numbers_possible_combinations = list(filter(partial(filter_for_fixed_recom_poza, poza_fixed_for_recom), pozas_numbers_possible_combinations))
    elif poza_fixed_for_recom is None and poza_fixed_cooking is not None:
        pozas_numbers_possible_combinations = list(filter(partial(filter_for_fixed_cooking_poza, poza_fixed_cooking), pozas_numbers_possible_combinations))

    df_combinaciones_1 = pd.DataFrame(columns=['tiempo_0', 'tiempo_1', 'utilidad_0', 'utilidad_1', 'utilidad', 'hours_to_wait_for_next_cook_0', 'hours_to_wait_for_next_cook_1',
                                               'pozas_cook_weighted_tvn_0', 'pozas_cook_weighted_tvn_1', 'mix_quality_0', 'mix_quality_1', 'pozas_cook_tons_0', 'pozas_cook_tons_1'])

    # Esta sección genera un df de todas las posibles combinaciones a alimentar entre el tiempo 0 y el tiempo 1.
    for poza_combination in pozas_numbers_possible_combinations:
        df_temp = pd.DataFrame()
        df_temp['tiempo_1'] = get_pozas_combinations_excluding_one(pozas_numbers_list, poza_combination)
        df_temp['tiempo_0'] = str(poza_combination)
        df_combinaciones_1 = df_combinaciones_1.append(df_temp)

    df_combinaciones_1 = df_combinaciones_1.reset_index(drop=True)
    df_combinaciones_1['tiempo_1'] = df_combinaciones_1['tiempo_1'].map(str)

    # Esta sección le añade al dataframe las columnas de utilidad. Se calcula la utilidad para cada tiempo y al final la suma de las utilidades.
    # Se toma en cuenta los rangos y precios por calidad, la velocidad de la planta actual y el tiempo de espera para la siguiente cocina aumentandole por
    # un factor en base al modelo TVN en pozas.
    for index, row in df_combinaciones_1.iterrows():
        utilidad_0, hours_to_wait_for_next_cook_0, pozas_cook_weighted_tvn_0, mix_quality_0, pozas_cook_tons_0 = calcular_utilidad(df_planta_pozas_to_optimize, row['tiempo_0'], calidad_precio, 0,
                                                                                                                                   df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        utilidad_1, hours_to_wait_for_next_cook_1, pozas_cook_weighted_tvn_1, mix_quality_1, pozas_cook_tons_1 = calcular_utilidad(df_planta_pozas_to_optimize, row['tiempo_1'], calidad_precio,
                                                                                                                                   hours_to_wait_for_next_cook_0, df_plant_info,
                                                                                                                                   df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)

        df_combinaciones_1['utilidad_0'][index] = utilidad_0
        df_combinaciones_1['utilidad_1'][index] = utilidad_1
        df_combinaciones_1['utilidad'][index] = utilidad_0 + utilidad_1
        df_combinaciones_1['hours_to_wait_for_next_cook_0'][index] = hours_to_wait_for_next_cook_0
        df_combinaciones_1['hours_to_wait_for_next_cook_1'][index] = hours_to_wait_for_next_cook_1
        df_combinaciones_1['pozas_cook_weighted_tvn_0'][index] = pozas_cook_weighted_tvn_0
        df_combinaciones_1['pozas_cook_weighted_tvn_1'][index] = pozas_cook_weighted_tvn_1
        df_combinaciones_1['mix_quality_0'][index] = mix_quality_0
        df_combinaciones_1['mix_quality_1'][index] = mix_quality_1
        df_combinaciones_1['pozas_cook_tons_0'][index] = pozas_cook_tons_0
        df_combinaciones_1['pozas_cook_tons_1'][index] = pozas_cook_tons_1

    # Se ordena en función a la utilidad para posteriormente escoger solo las primeras 2 opciones.
    df_combinaciones_1 = df_combinaciones_1.sort_values('utilidad', ascending=False)

    return df_combinaciones_1


def create_comb_up_to_time2(df, df_combinaciones_1, calidad_precio, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist):
    pozas_numbers_list = df['pozaNumber'].tolist()
    df_combinaciones_2 = pd.DataFrame(columns=['tiempo_0', 'tiempo_1', 'tiempo_2', 'utilidad_0', 'utilidad_1', 'utilidad_2', 'utilidad',
                                               'hours_to_wait_for_next_cook_0', 'hours_to_wait_for_next_cook_1', 'hours_to_wait_for_next_cook_2',
                                               'pozas_cook_weighted_tvn_0', 'pozas_cook_weighted_tvn_1', 'pozas_cook_weighted_tvn_2',
                                               'mix_quality_0', 'mix_quality_1', 'mix_quality_2',
                                               'pozas_cook_tons_0', 'pozas_cook_tons_1', 'pozas_cook_tons_2'])

    for index, row in df_combinaciones_1.iterrows():
        lista_elem = convert_to_list(eval(row['tiempo_0'])) + convert_to_list(eval(row['tiempo_1']))
        temp = pd.DataFrame()
        temp['tiempo_2'] = get_pozas_combinations_excluding_one(pozas_numbers_list, lista_elem)
        temp['tiempo_1'] = row['tiempo_1']
        temp['tiempo_0'] = row['tiempo_0']
        df_combinaciones_2 = df_combinaciones_2.append(temp)
    df_combinaciones_2['tiempo_0'] = df_combinaciones_2['tiempo_0'].map(str)
    df_combinaciones_2['tiempo_1'] = df_combinaciones_2['tiempo_1'].map(str)
    df_combinaciones_2['tiempo_2'] = df_combinaciones_2['tiempo_2'].map(str)
    df_combinaciones_2 = df_combinaciones_2.reset_index(drop=True)

    for index, row in df_combinaciones_2.iterrows():
        utilidad_0, hours_to_wait_for_next_cook_0, pozas_cook_weighted_tvn_0, mix_quality_0, pozas_cook_tons_0 = calcular_utilidad(df, row['tiempo_0'], calidad_precio, 0, df_plant_info,
                                                                                                                                   df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        utilidad_1, hours_to_wait_for_next_cook_1, pozas_cook_weighted_tvn_1, mix_quality_1, pozas_cook_tons_1 = calcular_utilidad(df, row['tiempo_1'], calidad_precio, hours_to_wait_for_next_cook_0,
                                                                                                                                   df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        utilidad_2, hours_to_wait_for_next_cook_2, pozas_cook_weighted_tvn_2, mix_quality_2, pozas_cook_tons_2 = calcular_utilidad(df, row['tiempo_2'], calidad_precio,
                                                                                                                                   hours_to_wait_for_next_cook_0 + hours_to_wait_for_next_cook_1,
                                                                                                                                   df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)

        df_combinaciones_2['utilidad_0'][index] = utilidad_0
        df_combinaciones_2['utilidad_1'][index] = utilidad_1
        df_combinaciones_2['utilidad_2'][index] = utilidad_2
        df_combinaciones_2['utilidad'][index] = utilidad_0 + utilidad_1 + utilidad_2
        df_combinaciones_2['hours_to_wait_for_next_cook_0'][index] = hours_to_wait_for_next_cook_0
        df_combinaciones_2['hours_to_wait_for_next_cook_1'][index] = hours_to_wait_for_next_cook_1
        df_combinaciones_2['hours_to_wait_for_next_cook_2'][index] = hours_to_wait_for_next_cook_2
        df_combinaciones_2['pozas_cook_weighted_tvn_0'][index] = pozas_cook_weighted_tvn_0
        df_combinaciones_2['pozas_cook_weighted_tvn_1'][index] = pozas_cook_weighted_tvn_1
        df_combinaciones_2['pozas_cook_weighted_tvn_2'][index] = pozas_cook_weighted_tvn_2
        df_combinaciones_2['mix_quality_0'][index] = mix_quality_0
        df_combinaciones_2['mix_quality_1'][index] = mix_quality_1
        df_combinaciones_2['mix_quality_2'][index] = mix_quality_2
        df_combinaciones_2['pozas_cook_tons_0'][index] = pozas_cook_tons_0
        df_combinaciones_2['pozas_cook_tons_1'][index] = pozas_cook_tons_1
        df_combinaciones_2['pozas_cook_tons_2'][index] = pozas_cook_tons_2

    df_combinaciones_2 = df_combinaciones_2.sort_values('utilidad', ascending=False)
    return df_combinaciones_2


def create_comb_up_to_time3(df, df_combinaciones_2, calidad_precio, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist):
    pozas_numbers_list = df['pozaNumber'].tolist()
    df_combinaciones_3 = pd.DataFrame(columns=['tiempo_0', 'tiempo_1', 'tiempo_2', 'tiempo_3',
                                               'utilidad_0', 'utilidad_1', 'utilidad_2', 'utilidad_3', 'utilidad',
                                               'hours_to_wait_for_next_cook_0', 'hours_to_wait_for_next_cook_1', 'hours_to_wait_for_next_cook_2', 'hours_to_wait_for_next_cook_3',
                                               'pozas_cook_weighted_tvn_0', 'pozas_cook_weighted_tvn_1', 'pozas_cook_weighted_tvn_2', 'pozas_cook_weighted_tvn_3',
                                               'mix_quality_0', 'mix_quality_1', 'mix_quality_2', 'mix_quality_3',
                                               'pozas_cook_tons_0', 'pozas_cook_tons_1', 'pozas_cook_tons_2', 'pozas_cook_tons_3'])

    for index, row in df_combinaciones_2.iterrows():
        lista_elem = convert_to_list(eval(row['tiempo_0'])) + convert_to_list(eval(row['tiempo_1'])) + convert_to_list(eval(row['tiempo_2']))
        temp = pd.DataFrame()
        temp['tiempo_3'] = get_pozas_combinations_excluding_one(pozas_numbers_list, lista_elem)
        temp['tiempo_2'] = row['tiempo_2']
        temp['tiempo_1'] = row['tiempo_1']
        temp['tiempo_0'] = row['tiempo_0']
        df_combinaciones_3 = df_combinaciones_3.append(temp)
    df_combinaciones_3['tiempo_0'] = df_combinaciones_3['tiempo_0'].map(str)
    df_combinaciones_3['tiempo_1'] = df_combinaciones_3['tiempo_1'].map(str)
    df_combinaciones_3['tiempo_2'] = df_combinaciones_3['tiempo_2'].map(str)
    df_combinaciones_3['tiempo_3'] = df_combinaciones_3['tiempo_3'].map(str)
    df_combinaciones_3 = df_combinaciones_3.reset_index(drop=True)

    for index, row in df_combinaciones_3.iterrows():
        utilidad_0, hours_to_wait_for_next_cook_0, pozas_cook_weighted_tvn_0, mix_quality_0, pozas_cook_tons_0 = calcular_utilidad(df, row['tiempo_0'], calidad_precio, 0, df_plant_info,
                                                                                                                                   df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        utilidad_1, hours_to_wait_for_next_cook_1, pozas_cook_weighted_tvn_1, mix_quality_1, pozas_cook_tons_1 = calcular_utilidad(df, row['tiempo_1'], calidad_precio, hours_to_wait_for_next_cook_0,
                                                                                                                                   df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        utilidad_2, hours_to_wait_for_next_cook_2, pozas_cook_weighted_tvn_2, mix_quality_2, pozas_cook_tons_2 = calcular_utilidad(df, row['tiempo_2'], calidad_precio,
                                                                                                                                   hours_to_wait_for_next_cook_0 + hours_to_wait_for_next_cook_1,
                                                                                                                                   df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        utilidad_3, hours_to_wait_for_next_cook_3, pozas_cook_weighted_tvn_3, mix_quality_3, pozas_cook_tons_3 = calcular_utilidad(df, row['tiempo_3'], calidad_precio,
                                                                                                                                   hours_to_wait_for_next_cook_0 + hours_to_wait_for_next_cook_1 + hours_to_wait_for_next_cook_2,
                                                                                                                                   df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)

        df_combinaciones_3['utilidad_0'][index] = utilidad_0
        df_combinaciones_3['utilidad_1'][index] = utilidad_1
        df_combinaciones_3['utilidad_2'][index] = utilidad_2
        df_combinaciones_3['utilidad_3'][index] = utilidad_3
        df_combinaciones_3['utilidad'][index] = utilidad_0 + utilidad_1 + utilidad_2 + utilidad_3
        df_combinaciones_3['hours_to_wait_for_next_cook_0'][index] = hours_to_wait_for_next_cook_0
        df_combinaciones_3['hours_to_wait_for_next_cook_1'][index] = hours_to_wait_for_next_cook_1
        df_combinaciones_3['hours_to_wait_for_next_cook_2'][index] = hours_to_wait_for_next_cook_2
        df_combinaciones_3['hours_to_wait_for_next_cook_3'][index] = hours_to_wait_for_next_cook_3
        df_combinaciones_3['pozas_cook_weighted_tvn_0'][index] = pozas_cook_weighted_tvn_0
        df_combinaciones_3['pozas_cook_weighted_tvn_1'][index] = pozas_cook_weighted_tvn_1
        df_combinaciones_3['pozas_cook_weighted_tvn_2'][index] = pozas_cook_weighted_tvn_2
        df_combinaciones_3['pozas_cook_weighted_tvn_3'][index] = pozas_cook_weighted_tvn_3
        df_combinaciones_3['mix_quality_0'][index] = mix_quality_0
        df_combinaciones_3['mix_quality_1'][index] = mix_quality_1
        df_combinaciones_3['mix_quality_2'][index] = mix_quality_2
        df_combinaciones_3['mix_quality_3'][index] = mix_quality_3
        df_combinaciones_3['pozas_cook_tons_0'][index] = pozas_cook_tons_0
        df_combinaciones_3['pozas_cook_tons_1'][index] = pozas_cook_tons_1
        df_combinaciones_3['pozas_cook_tons_2'][index] = pozas_cook_tons_2
        df_combinaciones_3['pozas_cook_tons_3'][index] = pozas_cook_tons_3

    df_combinaciones_3 = df_combinaciones_3.sort_values('utilidad', ascending=False)
    return df_combinaciones_3


def eliminar_pozas_hermanadas(df_combinaciones, plant_pozas_hermanadas, columna):
    df_combinaciones = df_combinaciones.reset_index(drop=True)
    lista_index = []
    for index, row in df_combinaciones.iterrows():
        pozas = eval(row[columna])
        if type(pozas) is list:
            pozas = sorted(pozas)
            if pozas in plant_pozas_hermanadas:
                lista_index.append(index)
    df_combinaciones = df_combinaciones.drop(df_combinaciones.index[lista_index])
    df_combinaciones = df_combinaciones.reset_index(drop=True)
    return df_combinaciones


def get_formato_recomendacion_row_values(df_planta_pozas_to_optimize, poza, utilidad, recom_type, cooking_poza_number):
    planta = df_planta_pozas_to_optimize.iloc[0]['id_planta']

    selected_poza_data = df_planta_pozas_to_optimize[(df_planta_pozas_to_optimize['pozaNumber'] == poza)]
    stock_poza = float(selected_poza_data['declared_ton'])
    tdc_poza = float(selected_poza_data['tdc_actual'])
    tvn_poza = float(selected_poza_data['tvn'])

    return [planta, poza, stock_poza, tdc_poza, tvn_poza, None, utilidad, recom_type, cooking_poza_number]


def get_mix_quality_weight(row):
    quality = row['mix_quality_0']
    if quality == 'SP':
        return 3
    elif quality == 'P':
        return 2
    elif quality == 'Otros':
        return 1
    else:
        return 0


def get_poza_combination_weighted_tvn(row, df_planta_pozas_to_optimize):
    pozas = eval(row['tiempo_0'])
    if type(pozas) is list:
        pozas_cook_tons = df_planta_pozas_to_optimize[df_planta_pozas_to_optimize['pozaNumber'].isin(pozas)]['declared_ton'].sum()
        row['weighted_tvn'] = df_planta_pozas_to_optimize[df_planta_pozas_to_optimize['pozaNumber'].isin(pozas)]['tvn_ton'].sum() / pozas_cook_tons
        row['sin_hielo'] = int(df_planta_pozas_to_optimize[df_planta_pozas_to_optimize['pozaNumber'].isin(pozas)]['con_hielo'].sum() == False)
    else:
        row['weighted_tvn'] = df_planta_pozas_to_optimize[df_planta_pozas_to_optimize['pozaNumber'] == pozas]['tvn'].reset_index(drop=True)[0]
        row['sin_hielo'] = int(df_planta_pozas_to_optimize[df_planta_pozas_to_optimize['pozaNumber'] == pozas]['con_hielo'].reset_index(drop=True)[0] == False)
    return row


# Select only single pozas is only used for mezcla actual where only single poza should be recommended
# Para mostrar la interpretabilidad va a ser necesario sortear todas las filas de df_utility_table y retornanrlo completo
# La selección se haría en formato_recomendacion_cocina()
def optimization_utility_sort_and_select(df_utility_table, df_planta_pozas_to_optimize, select_only_single_pozas=False):
    if select_only_single_pozas:
        df_utility_table = df_utility_table.loc[df_utility_table.apply(lambda row: type(eval(row['tiempo_0'])) is not list, axis=1)]

    df_planta_pozas_to_optimize['tvn_ton'] = df_planta_pozas_to_optimize['declared_ton'] * df_planta_pozas_to_optimize['tvn']
    df_optimizacion_with_weighted_tvn = df_utility_table.apply(get_poza_combination_weighted_tvn, axis=1, df_planta_pozas_to_optimize=df_planta_pozas_to_optimize)
    df_optimizacion_with_weighted_tvn['multiple_poza'] = df_optimizacion_with_weighted_tvn.apply(lambda row: type(eval(row['tiempo_0'])) is list, axis=1)

    df_optimizacion_with_weighted_tvn['mix_quality_0_weight'] = df_optimizacion_with_weighted_tvn.apply(get_mix_quality_weight, axis=1)
    df_optimizacion_with_weighted_tvn.sort_values(by=['utilidad', 'sin_hielo', 'mix_quality_0_weight', 'multiple_poza', 'weighted_tvn'], ascending=False, inplace=True)
    df_optimizacion_with_weighted_tvn.reset_index(drop=True, inplace=True)
    df_optimizacion_with_weighted_tvn['sort_order'] = df_optimizacion_with_weighted_tvn.index
    #df_optimizacion_with_weighted_tvn.drop('sin_hielo', axis=1, inplace=True)
    df_optimizacion_with_weighted_tvn.drop('mix_quality_0_weight', axis=1, inplace=True)
    return df_optimizacion_with_weighted_tvn.head(20)


def formato_recomendacion_cocina(df_utility_table, df_planta_pozas_to_optimize, recom_type, cooking_poza_number, select_only_single_pozas=False):
    if df_utility_table is None or len(df_utility_table) == 0:
        return None, None

    df_optimizacion_agg = pd.DataFrame(
        columns=['id_planta', 'numero_poza', 'stock_inicial', 'tdc_mas_antiguo_en_poza', 'tvn_estimado', 'priorizacion', 'utilidad', 'recom_type', 'mix_with_cooking_poza'])

    sorted_utility_poza_combinations = optimization_utility_sort_and_select(df_utility_table, df_planta_pozas_to_optimize, select_only_single_pozas)
    utilidad_max=sorted_utility_poza_combinations.iloc[0].utilidad
    candidatos=sorted_utility_poza_combinations.loc[(sorted_utility_poza_combinations.utilidad==utilidad_max)&(sorted_utility_poza_combinations.sin_hielo==1)]
    if len(candidatos)>0:
        sorted_optimal_poza_combination=candidatos.iloc[0]
    else:
        sorted_optimal_poza_combination = sorted_utility_poza_combinations.iloc[0]

    cont = 0
    pozas = sorted_optimal_poza_combination['tiempo_0']
    utilidad = float(sorted_optimal_poza_combination['utilidad'])
    pozas = eval(pozas)
    if type(pozas) is list:
        for poza in pozas:
            if poza != cooking_poza_number:
                df_optimizacion_agg.loc[cont] = get_formato_recomendacion_row_values(df_planta_pozas_to_optimize, poza, utilidad, recom_type, cooking_poza_number)
                cont = cont + 1
            else:
                print('Cooking poza is not passed as a recommended row')

    else:
        df_optimizacion_agg.loc[cont] = get_formato_recomendacion_row_values(df_planta_pozas_to_optimize, pozas, utilidad, recom_type, None)

    # Acá se tendría que agregar la columna id_cocina_recom que sea el mismo para df_optimizacion_agg y para sorted_utility_poza_combinations
    id_cocina_recom = df_optimizacion_agg['id_planta'][0] + ' - ' + df_optimizacion_agg['recom_type'][0] + ' - ' + datetime.utcnow().strftime("%m/%d/%Y_%H:%M:%S") + ' - utable'
    df_optimizacion_agg['id_cocina_recom'] = id_cocina_recom
    df_optimizacion_agg['has_utility_table'] = True
    sorted_utility_poza_combinations['id_cocina_recom'] = id_cocina_recom

    # Acá retorno este parámetro más el utility table
    return df_optimizacion_agg, sorted_utility_poza_combinations


def optimizacion_multiples_con_estados(df_planta_pozas_to_optimize, calidad_precio, df_plant_info, plant_pozas_hermanadas, poza_fixed_for_recom, df_pozas_currently_cooking, df_filtro_hist):
    global CURRENTLY_COOKING_STATES
    cooking_pozas_state = get_currently_cooking_pozas_state(df_pozas_currently_cooking)

    vel_cocina = df_plant_info['velocidad']
    if vel_cocina is None or vel_cocina<=0:
        vel_cocina = df_filtro_hist['velocidad'].values[0]

    if cooking_pozas_state == CURRENTLY_COOKING_STATES['NO_COOKING_POZAS']:
        print('NO_COOKING_POZAS')

        optimization_utility_df = optimizacion_multiples_pozas(df_planta_pozas_to_optimize, calidad_precio, df_plant_info, plant_pozas_hermanadas, poza_fixed_for_recom, df_pozas_currently_cooking,
                                                               None, df_filtro_hist)
        df_current_mix_formatted, current_mix_combinations_utility = formato_recomendacion_cocina(optimization_utility_df, df_planta_pozas_to_optimize, 'MEZCLA_ACTUAL', None, True)

        df_posterior_formatted, posterior_combinations_utility = formato_recomendacion_cocina(optimization_utility_df, df_planta_pozas_to_optimize, 'POSTERIOR', None)

        # Se va a tener que retornar esto más el utility table
        return pd.concat([df_current_mix_formatted, df_posterior_formatted]), pd.concat([current_mix_combinations_utility, posterior_combinations_utility])

    elif cooking_pozas_state == CURRENTLY_COOKING_STATES['ONE_COOKING_POZA']:
        print('ONE_COOKING_POZA')

        # 1. Doing recom_tye: MEZCLA_ACTUAL
        # !!! Al pasarlo al selected se tiene que ponerle el stock ya reducido en funcion a cuanto tiempo ya ha estado cocinando esa poza de tal manera que la utilidad se calcule correctamente
        # En el estado ONE_COOKING_POZA solo debería tener una fila df_pozas_currently_cooking
        df_pozas_cooking_selected = df_pozas_currently_cooking
        cooking_poza_number = df_pozas_cooking_selected.iloc[0]['pozaNumber']
        pozas_to_optimize_with_cooking = pd.concat([df_planta_pozas_to_optimize, df_pozas_cooking_selected])

        df_current_mix_utility = optimizacion_multiples_pozas(pozas_to_optimize_with_cooking, calidad_precio, df_plant_info, plant_pozas_hermanadas, poza_fixed_for_recom, df_pozas_currently_cooking,
                                                              cooking_poza_number, df_filtro_hist)
        df_current_mix_formatted, current_mix_combinations_utility = formato_recomendacion_cocina(df_current_mix_utility, df_planta_pozas_to_optimize, 'MEZCLA_ACTUAL', cooking_poza_number)

        # 2. Doing recom_tye: POSTERIOR
        df_posterior_utility = optimizacion_multiples_pozas(df_planta_pozas_to_optimize, calidad_precio, df_plant_info, plant_pozas_hermanadas, poza_fixed_for_recom, df_pozas_currently_cooking, None, df_filtro_hist)
        df_posterior_formatted, posterior_combinations_utility = formato_recomendacion_cocina(df_posterior_utility, df_planta_pozas_to_optimize, 'POSTERIOR', None)

        return pd.concat([df_current_mix_formatted, df_posterior_formatted]), pd.concat([current_mix_combinations_utility, posterior_combinations_utility])

    elif cooking_pozas_state == CURRENTLY_COOKING_STATES['MULTIPLE_COOKING_POZAS']:
        print('MULTIPLE_COOKING_POZAS')

        # 1. Doing recom_tye: MEZCLA_ACTUAL
        last_ending_cooking_poza, last_ending_poza_remaining_stock = get_currently_cooking_pozas_last_ending(df_pozas_currently_cooking, vel_cocina)
        last_ending_cooking_poza['declared_ton'] = last_ending_poza_remaining_stock
        cooking_poza_number = last_ending_cooking_poza['pozaNumber']
        pozas_to_optimize_with_cooking = df_planta_pozas_to_optimize.append(last_ending_cooking_poza)

        df_current_mix_utility = optimizacion_multiples_pozas(pozas_to_optimize_with_cooking, calidad_precio, df_plant_info, plant_pozas_hermanadas, poza_fixed_for_recom, df_pozas_currently_cooking,
                                                              cooking_poza_number, df_filtro_hist)
        df_current_mix_formatted, current_mix_combinations_utility = formato_recomendacion_cocina(df_current_mix_utility, df_planta_pozas_to_optimize, 'MEZCLA_ACTUAL', cooking_poza_number)

        # 2. Doing recom_tye: POSTERIOR
        df_posterior_utility = optimizacion_multiples_pozas(df_planta_pozas_to_optimize, calidad_precio, df_plant_info, plant_pozas_hermanadas, poza_fixed_for_recom, df_pozas_currently_cooking, None, df_filtro_hist)
        df_posterior_formatted, posterior_combinations_utility = formato_recomendacion_cocina(df_posterior_utility, df_planta_pozas_to_optimize, 'POSTERIOR', None)

        # Este concat ya tiene que ser con formato
        return pd.concat([df_current_mix_formatted, df_posterior_formatted]), pd.concat([current_mix_combinations_utility, posterior_combinations_utility])

    else:
        print('Error encontrando cook state. Nunca debería entrar acá!!!')
        return pd.DataFrame()


def optimizacion_multiples_pozas(df_planta_pozas_to_optimize, calidad_precio, df_plant_info, plant_pozas_hermanadas, poza_fixed_for_recom, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist):
    cant_pozas = len(df_planta_pozas_to_optimize)
    if cant_pozas <= 1:
        print("ERROR!!! Nunca debería entrar acá!")
        return pd.DataFrame()
    elif cant_pozas == 2:
        df_combinaciones_1 = create_comb_up_to_time1(df_planta_pozas_to_optimize, calidad_precio, poza_fixed_for_recom, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        df_combinaciones_1 = eliminar_pozas_hermanadas(df_combinaciones_1, plant_pozas_hermanadas, 'tiempo_0')
        df_combinaciones_1 = eliminar_pozas_hermanadas(df_combinaciones_1, plant_pozas_hermanadas, 'tiempo_1')
        print("Optimizacion exitosa")
        return df_combinaciones_1
    elif cant_pozas == 3:
        df_combinaciones_1 = create_comb_up_to_time1(df_planta_pozas_to_optimize, calidad_precio, poza_fixed_for_recom, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        df_combinaciones_2 = create_comb_up_to_time2(df_planta_pozas_to_optimize, df_combinaciones_1, calidad_precio, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        df_combinaciones_2 = eliminar_pozas_hermanadas(df_combinaciones_2, plant_pozas_hermanadas, 'tiempo_0')
        df_combinaciones_2 = eliminar_pozas_hermanadas(df_combinaciones_2, plant_pozas_hermanadas, 'tiempo_1')
        df_combinaciones_2 = eliminar_pozas_hermanadas(df_combinaciones_2, plant_pozas_hermanadas, 'tiempo_2')
        print("Optimizacion exitosa")
        return df_combinaciones_2
    else:
        if cant_pozas >= 5:
            df_planta_pozas_to_optimize = df_planta_pozas_to_optimize.sort_values(['cocinandose', 'tvn'], ascending=[False, False]).head(5)
        df_combinaciones_1 = create_comb_up_to_time1(df_planta_pozas_to_optimize, calidad_precio, poza_fixed_for_recom, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        df_combinaciones_2 = create_comb_up_to_time2(df_planta_pozas_to_optimize, df_combinaciones_1, calidad_precio, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        df_combinaciones_3 = create_comb_up_to_time3(df_planta_pozas_to_optimize, df_combinaciones_2, calidad_precio, df_plant_info, df_pozas_currently_cooking, poza_fixed_cooking, df_filtro_hist)
        df_combinaciones_3 = eliminar_pozas_hermanadas(df_combinaciones_3, plant_pozas_hermanadas, 'tiempo_0')
        df_combinaciones_3 = eliminar_pozas_hermanadas(df_combinaciones_3, plant_pozas_hermanadas, 'tiempo_1')
        df_combinaciones_3 = eliminar_pozas_hermanadas(df_combinaciones_3, plant_pozas_hermanadas, 'tiempo_2')
        df_combinaciones_3 = eliminar_pozas_hermanadas(df_combinaciones_3, plant_pozas_hermanadas, 'tiempo_3')
        print("Optimizacion exitosa")
        return df_combinaciones_3


def get_remaining_stock_on_last_ending_cooking_poza(df_pozas_currently_cooking, vel_cocina, last_ending_poza_number):
    last_ending_poza = df_pozas_currently_cooking[df_pozas_currently_cooking['pozaNumber'] == last_ending_poza_number].iloc[0]
    # last_ending_current_estimated_stock = last_ending_poza['stock_poza_inicial'] - vel_cocina * last_ending_poza['cook_start_hours_passed'] * (last_ending_poza['perc_poza_cocina'] / 100)
    last_ending_current_estimated_stock = last_ending_poza['stock_actual'] - vel_cocina * last_ending_poza['stock_actual_hours_passed'] * (last_ending_poza['perc_poza_cocina'] / 100)

    hours_to_finish_cooking_other_pozas = 0

    for index, poza in df_pozas_currently_cooking.reset_index(drop=True).iterrows():
        if poza['pozaNumber'] == last_ending_poza_number:
            continue
        else:
            # current_estimated_stock = poza['stock_poza_inicial'] - vel_cocina * poza['cook_start_hours_passed'] * (poza['perc_poza_cocina'] / 100)
            current_estimated_stock = poza['stock_actual'] - vel_cocina * poza['stock_actual_hours_passed'] * (poza['perc_poza_cocina'] / 100)
            hours_to_finish_poza = current_estimated_stock / (vel_cocina * (poza['perc_poza_cocina'] / 100))
            if hours_to_finish_poza > hours_to_finish_cooking_other_pozas:
                hours_to_finish_cooking_other_pozas = hours_to_finish_poza

    last_ending_poza_remaining_stock = last_ending_current_estimated_stock - vel_cocina * hours_to_finish_cooking_other_pozas * (last_ending_poza['perc_poza_cocina'] / 100)

    if last_ending_poza_remaining_stock < 0:
        last_ending_poza_remaining_stock = 0

    return last_ending_poza_remaining_stock


def get_currently_cooking_pozas_last_ending(df_pozas_currently_cooking, vel_cocina):
    last_ending_poza = None

    for index, poza in df_pozas_currently_cooking.reset_index(drop=True).iterrows():
        # current_estimated_stock = poza['stock_poza_inicial'] - vel_cocina * poza['cook_start_hours_passed'] * (poza['perc_poza_cocina'] / 100)
        current_estimated_stock = poza['stock_actual'] - vel_cocina * poza['stock_actual_hours_passed'] * (poza['perc_poza_cocina'] / 100)
        hours_to_finish_poza = current_estimated_stock / (vel_cocina * (poza['perc_poza_cocina'] / 100))
        if index == 0:
            hours_to_finish_last_ending = hours_to_finish_poza
            last_ending_poza = poza
        # Buscas encontrar la poza que va a terminar último por lo tanto la que su hours_to_finish_poza sea mayor
        elif hours_to_finish_poza > hours_to_finish_last_ending:
            hours_to_finish_last_ending = hours_to_finish_poza
            last_ending_poza = poza

    return last_ending_poza, get_remaining_stock_on_last_ending_cooking_poza(df_pozas_currently_cooking, vel_cocina, last_ending_poza['pozaNumber'])


def get_poza_hermanada_pair(poza_number, plant_pozas_hermanadas):
    poza_hermanada_number = []
    for hermanada_pair in plant_pozas_hermanadas:
        if poza_number in hermanada_pair:
            # This second "for" looks for the hermanada poza. If this reaches here poza_hermanada_number should get a value.
            for poza_n in hermanada_pair:
                if poza_number != poza_n:
                    poza_hermanada_number.append(poza_n)
    return poza_hermanada_number


def single_or_contingencia_select_avoiding_hermanadas(plant_pozas_to_select, cooking_poza_number, plant_pozas_hermanadas, number_of_pozas_to_select):
    if number_of_pozas_to_select == 1:
        if cooking_poza_number is None:
            return plant_pozas_to_select.head(1)
        else:
            # Select top poza that is not hermanada with cooking_poza_number
            cooking_poza_hermanada_number = get_poza_hermanada_pair(cooking_poza_number, plant_pozas_hermanadas)

            if cooking_poza_hermanada_number is not None and len(cooking_poza_hermanada_number) > 0:
                for index, row in plant_pozas_to_select.iterrows():
                    # if cooking_poza_hermanada_number != row['pozaNumber']:
                    if row['pozaNumber'] not in cooking_poza_hermanada_number:
                        return plant_pozas_to_select[index:index + 1]
                return plant_pozas_to_select.head(0)
            else:
                return plant_pozas_to_select.head(1)
    # If number to select is 2, it means that there wont be mixing with a cooking poza
    elif number_of_pozas_to_select == 2:
        top_poza_row = plant_pozas_to_select.head(1)
        top_poza_hermanada_number = get_poza_hermanada_pair(top_poza_row.iloc[0]['pozaNumber'], plant_pozas_hermanadas)
        if top_poza_hermanada_number is not None and len(top_poza_hermanada_number) > 0:
            for index, row in plant_pozas_to_select.iterrows():
                if top_poza_row.iloc[0]['pozaNumber'] == row['pozaNumber']:
                    continue
                else:
                    # if top_poza_hermanada_number != row['pozaNumber']:
                    if row['pozaNumber'] not in top_poza_hermanada_number:
                        # Joining the top poza and the non hermanada poza
                        pozas_selected_to_return = top_poza_row.append(row)
                        return pozas_selected_to_return

            return top_poza_row
        else:
            # This means that the top poza doesn't have a hermanada pair so just returns the top 2
            return plant_pozas_to_select.head(2)
    else:
        print('Error single_or_contingencia_select_avoiding_hermanadas! Nunca debió entrar acá')
        return plant_pozas_to_select.head(number_of_pozas_to_select)


def optimization_for_contingencia_or_single_poza(parameter_to_sort_by, parameter_to_filter_by, plant_pozas_df, df_plant_info, df_pozas_currently_cooking, plant_pozas_hermanadas, df_filtro_hist):
    plant_pozas_to_edit = plant_pozas_df
    plant_pozas_to_edit = plant_pozas_to_edit.reset_index(drop=True)

    if parameter_to_filter_by is not None:
        plant_pozas_to_edit = plant_pozas_df[plant_pozas_df[parameter_to_filter_by] == 1]

    if parameter_to_sort_by is not None:
        plant_pozas_to_edit = plant_pozas_to_edit.reset_index(drop=True)
        plant_pozas_to_edit = plant_pozas_to_edit.sort_values(parameter_to_sort_by, ascending=False)

    global CURRENTLY_COOKING_STATES

    cooking_pozas_state = get_currently_cooking_pozas_state(df_pozas_currently_cooking)

    if cooking_pozas_state == CURRENTLY_COOKING_STATES['NO_COOKING_POZAS']:
        print('NO_COOKING_POZAS')

        # Selecting max top 1 poza for MEZCLA_ACTUAL
        plant_pozas_selected_current_mix = plant_pozas_to_edit.head(1)

        df_to_append_current_mix = plant_pozas_selected_current_mix[['id_planta', 'pozaNumber', 'declared_ton', 'tdc_actual', 'tvn']]
        df_to_append_current_mix.rename(columns={"pozaNumber": "numero_poza", "declared_ton": "stock_inicial", "tdc_actual": "tdc_mas_antiguo_en_poza", "tvn": "tvn_estimado"}, inplace=True)

        df_to_append_current_mix['priorizacion'] = None
        df_to_append_current_mix['utilidad'] = None
        df_to_append_current_mix['recom_type'] = 'MEZCLA_ACTUAL'
        df_to_append_current_mix['mix_with_cooking_poza'] = None

        # Selecting max top 2 pozas for Posterior
        plant_pozas_selected_posterior = single_or_contingencia_select_avoiding_hermanadas(plant_pozas_to_edit, None, plant_pozas_hermanadas, 2)

        df_to_append_posterior = plant_pozas_selected_posterior[['id_planta', 'pozaNumber', 'declared_ton', 'tdc_actual', 'tvn']]
        df_to_append_posterior.rename(columns={"pozaNumber": "numero_poza", "declared_ton": "stock_inicial", "tdc_actual": "tdc_mas_antiguo_en_poza", "tvn": "tvn_estimado"}, inplace=True)

        df_to_append_posterior['priorizacion'] = None
        df_to_append_posterior['utilidad'] = None
        df_to_append_posterior['recom_type'] = 'POSTERIOR'
        df_to_append_posterior['mix_with_cooking_poza'] = None
        return pd.concat([df_to_append_current_mix, df_to_append_posterior])

    elif cooking_pozas_state == CURRENTLY_COOKING_STATES['ONE_COOKING_POZA']:
        print('ONE_COOKING_POZA')

        # 1. Doing recom_tye: MEZCLA_ACTUAL
        plant_pozas_selected_current_mix = plant_pozas_to_edit.head(1)
        cooking_poza_selected = df_pozas_currently_cooking
        cooking_poza_number = cooking_poza_selected.iloc[0]['pozaNumber']

        plant_pozas_selected_current_mix = single_or_contingencia_select_avoiding_hermanadas(plant_pozas_to_edit, cooking_poza_number, plant_pozas_hermanadas, 1)

        df_to_append_current_mix = plant_pozas_selected_current_mix[['id_planta', 'pozaNumber', 'declared_ton', 'tdc_actual', 'tvn']]
        df_to_append_current_mix.rename(columns={"pozaNumber": "numero_poza", "declared_ton": "stock_inicial", "tdc_actual": "tdc_mas_antiguo_en_poza", "tvn": "tvn_estimado"}, inplace=True)
        df_to_append_current_mix['priorizacion'] = None
        df_to_append_current_mix['utilidad'] = None
        df_to_append_current_mix['recom_type'] = 'MEZCLA_ACTUAL'
        df_to_append_current_mix['mix_with_cooking_poza'] = cooking_poza_number

        # 2. Doing recom_tye: POSTERIOR
        # plant_pozas_selected_posterior = plant_pozas_to_edit.head(2)
        plant_pozas_selected_posterior = single_or_contingencia_select_avoiding_hermanadas(plant_pozas_to_edit, None, plant_pozas_hermanadas, 2)

        df_to_append_posterior = plant_pozas_selected_posterior[['id_planta', 'pozaNumber', 'declared_ton', 'tdc_actual', 'tvn']]
        df_to_append_posterior.rename(columns={"pozaNumber": "numero_poza", "declared_ton": "stock_inicial", "tdc_actual": "tdc_mas_antiguo_en_poza", "tvn": "tvn_estimado"}, inplace=True)
        df_to_append_posterior['priorizacion'] = None
        df_to_append_posterior['utilidad'] = None
        df_to_append_posterior['recom_type'] = 'POSTERIOR'
        df_to_append_posterior['mix_with_cooking_poza'] = None

        return pd.concat([df_to_append_current_mix, df_to_append_posterior])

    elif cooking_pozas_state == CURRENTLY_COOKING_STATES['MULTIPLE_COOKING_POZAS']:
        print('MULTIPLE_COOKING_POZAS')

        vel_cocina = df_plant_info['velocidad']
        if vel_cocina is None or vel_cocina<=0:
            vel_cocina = df_filtro_hist['velocidad'].values[0]

        # 1. Doing recom_tye: MEZCLA_ACTUAL
        plant_pozas_selected_current_mix = plant_pozas_to_edit.head(1)
        last_ending_poza, last_ending_poza_remaining_stock = get_currently_cooking_pozas_last_ending(df_pozas_currently_cooking, vel_cocina)
        cooking_poza_number = last_ending_poza['pozaNumber']

        plant_pozas_selected_current_mix = single_or_contingencia_select_avoiding_hermanadas(plant_pozas_to_edit, cooking_poza_number, plant_pozas_hermanadas, 1)

        df_to_append_current_mix = plant_pozas_selected_current_mix[['id_planta', 'pozaNumber', 'declared_ton', 'tdc_actual', 'tvn']]
        df_to_append_current_mix.rename(columns={"pozaNumber": "numero_poza", "declared_ton": "stock_inicial", "tdc_actual": "tdc_mas_antiguo_en_poza", "tvn": "tvn_estimado"}, inplace=True)
        df_to_append_current_mix['priorizacion'] = None
        df_to_append_current_mix['utilidad'] = None
        df_to_append_current_mix['recom_type'] = 'MEZCLA_ACTUAL'
        df_to_append_current_mix['mix_with_cooking_poza'] = cooking_poza_number

        # 2. Doing recom_tye: POSTERIOR
        plant_pozas_selected_posterior = plant_pozas_to_edit.head(2)
        plant_pozas_selected_posterior = single_or_contingencia_select_avoiding_hermanadas(plant_pozas_to_edit, None, plant_pozas_hermanadas, 2)

        df_to_append_posterior = plant_pozas_selected_posterior[['id_planta', 'pozaNumber', 'declared_ton', 'tdc_actual', 'tvn']]
        df_to_append_posterior.rename(columns={"pozaNumber": "numero_poza", "declared_ton": "stock_inicial", "tdc_actual": "tdc_mas_antiguo_en_poza", "tvn": "tvn_estimado"}, inplace=True)
        df_to_append_posterior['priorizacion'] = None
        df_to_append_posterior['utilidad'] = None
        df_to_append_posterior['recom_type'] = 'POSTERIOR'
        df_to_append_posterior['mix_with_cooking_poza'] = None

        return pd.concat([df_to_append_current_mix, df_to_append_posterior])

    else:
        print('Error nunca debería entrar acá')
        return pd.DataFrame()


# Esto se agrega al df de retorno las pozas que se están alimentando, las que tienen contingencia de descarga o las que no fueron recomendadas
# El motivo de agregarlas es poder visualizar su TVN
def add_non_recommended_pozas(df_recommended_pozas, df_all_pozas_estado):
    for index, row in df_all_pozas_estado.iterrows():
        if not ((df_recommended_pozas['numero_poza'] == row['pozaNumber']) & (df_recommended_pozas['id_planta'] == row['id_planta'])).any():
            row_to_append = {'id_planta': row['id_planta'], 'numero_poza': row['pozaNumber'], 'stock_inicial': row['declared_ton'],
                             'tdc_mas_antiguo_en_poza': row['tdc_actual'], 'tvn_estimado': row['tvn'], 'priorizacion': None,
                             'utilidad': None, 'contingencia': None, 'linea_frio': None, 'aprovechamiento_frio': None, 'recommended': False}
            df_recommended_pozas = df_recommended_pozas.append(row_to_append, ignore_index=True)

    return df_recommended_pozas


def verify_if_at_least_one_pozas_has_content(df_pozas_estado):
    df_pozas_estado_filtered = df_pozas_estado[
        (df_pozas_estado['marea_id'].notna()) & (df_pozas_estado['stock_actual'] > 0) & (df_pozas_estado['declared_ton'] > 0)
        & (df_pozas_estado['fin_cocina'].isnull())]
    return len(df_pozas_estado_filtered) > 0


def verify_if_at_least_one_poza_doesnt_have_contingencia(df_planta_pozas):
    df_planta_pozas_filtered = df_planta_pozas[(df_planta_pozas['traditional_discharge_contingencia'] == 0) & (df_planta_pozas['contingencia_in_poza'] == 0)]
    return len(df_planta_pozas_filtered) > 0


def run_cocina_optimization(get_data):
    print('Started to run cocina optimization: ' + os.getenv("ENVIRONMENT"))

    ## CARGAR DATOS

    ## datos de planta
    df_plants_info = get_data.get_requerimiento_planta()
    
    ## datos historico pozas
    df_stock_historico = get_data.get_stock_pozas_recientes()
    try:
        del df_stock_historico['Unnamed: 0']
    except:
        pass
    df_stock_historico_original = df_stock_historico.copy()

    # Obtener segunda lectura
    df_lectura2 = df_stock_historico.loc[df_stock_historico['Lecturas']==2,['id_planta','numero_poza','stock']].reset_index(drop=True)
    df_stock_historico = df_stock_historico.merge(df_lectura2[['id_planta','numero_poza','stock']], how='left', on=['id_planta','numero_poza'])
    
    # Obtener tercera lectura
    df_lectura3 = df_stock_historico_original.loc[df_stock_historico_original['Lecturas']==3,['id_planta','numero_poza','stock']].reset_index(drop=True)
    df_stock_historico = df_stock_historico.merge(df_lectura3[['id_planta','numero_poza','stock']], how='left', on=['id_planta','numero_poza'])
    
    df_stock_historico = df_stock_historico[df_stock_historico['Lecturas']==1].reset_index(drop=True)
    df_stock_historico.columns = ['id_planta', 'numero_poza', 'stock_actual', 'update_date', 'report_date',
           'report_hour', 'not_report_label', 'Lecturas', 'stock_previo', 'stock_anterior']
    
    df_stock_historico['stock_previo'] = df_stock_historico['stock_previo'].fillna(0)
    df_stock_historico['stock_anterior'] = df_stock_historico['stock_anterior'].fillna(0)
    
    ## Velocidades plantas
    # df_data_plantas = get_data.get_plantas_velocidades()
    
    # Leer embarcaciones
    df_embarcaciones = get_data.get_active_mareas_with_location_and_static_data()
    
    ## Historico velocidades planta
    df_velocidad_historico = get_data.get_plantas_velocidades_historico()
    
    # Leer plantas habilitadas
    df_plantas_habilitadas = get_data.get_plantas_habilitadas()
     
    ## datos de pozas
    df_pozas_estado = get_data.get_pozas_estado()
    df_pozas_estado['pozaNumber'] = df_pozas_estado['pozaNumber'].astype(int)
    if not ('con_hielo' in df_pozas_estado):
        df_pozas_estado['con_hielo']=np.where(df_pozas_estado.frio_system_state=='CH',True,False)
    # print('df_pozas_estado \n', df_pozas_estado[df_pozas_estado['id_planta'] == 'CHIMBOTE'])
    hora_exceso_40=df_pozas_estado[df_pozas_estado.con_hielo==True][['id_planta','pozaNumber']].drop_duplicates()
    # try:
    #     importar_horas_exceso=pd.read_csv('hora_exceso_40.csv')
    # except:
    importar_horas_exceso=pd.DataFrame(columns=['id_planta','pozaNumber','hora_excedio_40'])
    hora_exceso_40=hora_exceso_40.merge(importar_horas_exceso,how='left',on=['id_planta','pozaNumber'])

    if len(hora_exceso_40)>0:
        aux_cocinandose=df_pozas_estado[df_pozas_estado.cocinandose==True].drop_duplicates(subset=['id_planta','pozaNumber']).copy()
        for index,row in aux_cocinandose.iterrows():
            hora_exceso_40.loc[(hora_exceso_40.id_planta==row.id_planta)&(hora_exceso_40.pozaNumber==row.pozaNumber),'hora_excedio_40']=0

    env_argument_param = os.getenv("ENVIRONMENT")
    if env_argument_param == 'dumped':
        timestamp = get_dumped_data_date()
    else:
        timestamp = datetime.utcnow()


    ## datos pozas hermanadas
    df_pozas_hermanadas = get_data.get_pozas_hermanadas()
    df_pozas_hermanadas['pozaNumber'] = df_pozas_hermanadas['pozaNumber'].astype(int)
    dict_hermanadas = get_pozas_hermanadas_per_plant(df_pozas_hermanadas)

    hermanadas_callao = dict_hermanadas['CALLAO']
    hermanadas_callao.extend([[10, 11], [10, 12], [10, 13], [11, 12], [11, 13], [12, 13]])
    dict_hermanadas['CALLAO'] = hermanadas_callao

    ## precio de venta
    df_calidades_precio_venta = get_data.get_calidades_precio_venta()
    calidad_precio = list(df_calidades_precio_venta['PRICE_PER_TON'])

    #################################################################################################

    # Optimization final dataframe
    df_optimizacion_agg = pd.DataFrame(
        columns=['id_cocina_recom', 'id_planta', 'numero_poza', 'stock_inicial', 'tdc_mas_antiguo_en_poza', 'tvn_estimado', 'priorizacion', 'utilidad', 'contingencia', 'linea_frio',
                 'aprovechamiento_frio'
            , 'recommended', 'has_utility_table']
    )

    # Este campo es el que va a guardar el agregado de las tablas de utilidad y el que se va a enviar a la base de datos como interpretabilidad
    df_utility_table_agg = pd.DataFrame(columns=['tiempo_0', 'tiempo_1', 'tiempo_2', 'tiempo_3',
                                                 'utilidad_0', 'utilidad_1', 'utilidad_2', 'utilidad_3', 'utilidad',
                                                 'hours_to_wait_for_next_cook_0', 'hours_to_wait_for_next_cook_1', 'hours_to_wait_for_next_cook_2', 'hours_to_wait_for_next_cook_3',
                                                 'pozas_cook_weighted_tvn_0', 'pozas_cook_weighted_tvn_1', 'pozas_cook_weighted_tvn_2', 'pozas_cook_weighted_tvn_3',
                                                 'mix_quality_0', 'mix_quality_1', 'mix_quality_2', 'mix_quality_3',
                                                 'pozas_cook_tons_0', 'pozas_cook_tons_1', 'pozas_cook_tons_2', 'pozas_cook_tons_3',
                                                 'id_cocina_recom',
                                                 'weighted_tvn', 'multiple_poza','planta'])  # This columns will eventually be dropped when inserting to the DB

    # Calcular tdc descarga para los que no lo tengan calculado
    mask_tdc_na = (df_pozas_estado['marea_id'].notnull()) & (df_pozas_estado['tvn_discharge'].isnull()) & (df_pozas_estado['stock_actual']>0)
    df_pozas_estado['first_cala_start_date'] = pd.to_datetime(df_pozas_estado['first_cala_start_date'])
    df_pozas_estado.loc[mask_tdc_na,'tdc_discharge'] = timestamp - df_pozas_estado.loc[mask_tdc_na,'first_cala_start_date']
    df_pozas_estado['tdc_discharge'] = pd.to_timedelta(df_pozas_estado['tdc_discharge'])
    df_pozas_estado['tdc_discharge'] = df_pozas_estado['tdc_discharge'].dt.total_seconds() / (60 * 60)
    # df_pozas_estado.loc[mask_tdc_na,'tdc_discharge'] = df_pozas_estado.loc[mask_tdc_na,'tdc_discharge']/np.timedelta64(1, 'h')
    df_pozas_estado['tdc_discharge'] = df_pozas_estado['tdc_discharge'].astype(float)

    
    # Completar con inicio de succion las mareas que no tengan inicio de descarga
    cols_emb = ['marea_id','inicio_succion']
    df_pozas_estado["marea_id"] = df_pozas_estado["marea_id"].fillna(np.nan)
    df_pozas_estado = df_pozas_estado.merge(df_embarcaciones[cols_emb], how='left', on='marea_id')
    df_pozas_estado['discharge_start_date'] = df_pozas_estado['discharge_start_date'].fillna(df_pozas_estado['inicio_succion'])
    del df_pozas_estado['inicio_succion']
    
    # Verification if at least one value to optimize

    #print('df_pozas_estado', df_pozas_estado)
    if verify_if_at_least_one_pozas_has_content(df_pozas_estado):
        # Cleaning data

        df_pozas_estado_limpio_with_currently_cooking = limpiar_data_pozas(df_pozas_estado, df_plants_info, df_plantas_habilitadas)

        df_pozas_estado_limpio = df_pozas_estado_limpio_with_currently_cooking[df_pozas_estado_limpio_with_currently_cooking['cocinandose'] == 0]
        df_pozas_estado_currently_cooking = df_pozas_estado_limpio_with_currently_cooking[df_pozas_estado_limpio_with_currently_cooking['cocinandose'] == 1]

        # print('df_pozas_estado_limpio \n', df_pozas_estado_limpio)
        # print('df_pozas_estado_currently_cooking \n', df_pozas_estado_currently_cooking)

        #Se eliminan de la optimizacion aquellas pozas con preservante para las que aun no han pasado 4 horas desde que llenaron el 40% de la poza.
        for index,row in df_pozas_estado_limpio.iterrows():
            porcentaje_llenado=row.stock_actual/row.pozaCapacity
            if (porcentaje_llenado>=0.4) & (len(hora_exceso_40)>0):
                try:
                    if (hora_exceso_40.loc[(hora_exceso_40.id_planta==row.id_planta)&(hora_exceso_40.pozaNumber==row.pozaNumber),'hora_excedio_40']==0):
                        hora_exceso_40.loc[(hora_exceso_40.id_planta==row.id_planta)&(hora_exceso_40.pozaNumber==row.pozaNumber),'hora_excedio_40']!=timestamp
                except:
                    pass

        eliminar_pozas=[]
        for index,row in hora_exceso_40.iterrows():
            horas_transcurridas_dese_40=0
            if row.hora_excedio_40!=0:
                try:
                    horas_transcurridas_dese_40=get_hours_passed_from_date(row.hora_excedio_40)
                except:
                    pass
            if ((row.hora_excedio_40==0)|(horas_transcurridas_dese_40<4)):
                id_concat=row.id_planta+'-'+str(row.pozaNumber)
                eliminar_pozas.append(id_concat)

        df_pozas_estado_limpio=df_pozas_estado_limpio[~df_pozas_estado_limpio.id.isin(eliminar_pozas)].reset_index(drop=True).copy()

        
        # Contar pozas con stock por planta
        df_pozas_plantas = df_pozas_estado_limpio.groupby(['id_planta']).agg(num_pozas=('pozaNumber','count'))
        df_pozas_plantas.reset_index(inplace=True)
        df_pozas_estado_limpio = df_pozas_estado_limpio.merge(df_pozas_plantas, how='left', on='id_planta')
    
        
        # Se eliminan de la optimizacion las pozas con stock < 50, si es que es su primera lectura (logica de acopio)
        eliminar_por_acopio = []
        for index_acopio,row_acopio in df_pozas_estado_limpio.iterrows():
            row_acopio.reset_index(drop=True)
            if (row_acopio['num_pozas']>1) & (row_acopio['stock_actual']<50):
                for index_precio,row_previo in df_stock_historico.iterrows():
                    row_previo.reset_index(drop=True)
                    if (row_acopio['id_planta']==row_previo['id_planta']) & (row_acopio['pozaNumber']==row_previo['numero_poza']):
                        stock_anterior = np.array(row_previo['stock_anterior'])
                        stock_previo = np.array(row_previo['stock_previo'])
                        # stock_actual = np.array(row_acopio['stock_actual'])
                        if ((stock_previo<50) & (stock_previo>0)) & ((stock_anterior<50) & (stock_anterior>0)):
                            id_eliminar = []
                        else:
                            id_eliminar = row_acopio.id_planta+'-'+str(row_acopio.pozaNumber)
                            eliminar_por_acopio.append(id_eliminar)
            
        df_pozas_estado_limpio=df_pozas_estado_limpio[~df_pozas_estado_limpio.id.isin(eliminar_por_acopio)].reset_index(drop=True).copy()
        del df_pozas_estado_limpio['num_pozas']

        # optimizar por planta
        for planta in list(df_pozas_estado_limpio['id_planta'].unique()):
            df_planta_pozas = df_pozas_estado_limpio[df_pozas_estado_limpio['id_planta'] == planta]
            df_pozas_currently_cooking = df_pozas_estado_currently_cooking[df_pozas_estado_currently_cooking['id_planta'] == planta]

            # Si alguno de estos valores es mayor a 1 indica que en la planta por lo menos una poza paso o esta cerca de pasar alguno de los límites definidos
            discharge_contingencia = df_planta_pozas['traditional_discharge_contingencia'].sum()
            contingencia_in_poza = df_planta_pozas['contingencia_in_poza'].sum()

            df_plant_info = df_plants_info[df_plants_info['id'] == planta].reset_index(drop=True).iloc[0]
            df_filtro_hist = df_velocidad_historico[df_velocidad_historico['id_planta']==planta].reset_index(drop=True)
            # print(df_filtro_hist)
            plant_pozas_hermanadas = dict_hermanadas[planta]

            df_temp = None
            df_plant_utility_table = None

            # Pozas con contingencia sólo se recomiendan si son la única poza y no se está alimentando ninguna al momento. En caso contrario no se incluyen en la optimización.
            if (discharge_contingencia >= 1 or contingencia_in_poza >= 1) and len(df_planta_pozas) == 1 and len(df_pozas_currently_cooking) == 0:
                df_temp = optimization_for_contingencia_or_single_poza(None, None, df_planta_pozas, df_plant_info, df_pozas_currently_cooking, plant_pozas_hermanadas, df_filtro_hist)

            elif verify_if_at_least_one_poza_doesnt_have_contingencia(df_planta_pozas):
                # Eliminando pozas con contingencia ya que se definió por el negocio no incluirlas en la recomendación
                df_planta_pozas = df_planta_pozas[df_planta_pozas['traditional_discharge_contingencia'] == 0]
                df_planta_pozas = df_planta_pozas[df_planta_pozas['contingencia_in_poza'] == 0]
                
                # Estos indicadores han quedado en desuso desde antes del traspaso de BREIN
                tdc_in_poza_tradicional_warning_limit = df_planta_pozas['tdc_in_poza_tradicional_warning_limit'].sum()  # This is tdc in poza
                tvn_in_poza_warning_limit = df_planta_pozas['tvn_in_poza_warning_limit'].sum()  # This is tvn in poza
                tdc_linea_frio = df_planta_pozas['tdc_linea_frio'].sum()
                aprovechamiento_frio = df_planta_pozas['aprovechamiento_frio'].sum()

                if len(df_planta_pozas) == 1:
                    df_temp = optimization_for_contingencia_or_single_poza(None, None, df_planta_pozas, df_plant_info, df_pozas_currently_cooking, plant_pozas_hermanadas, df_filtro_hist)

                else:
                    df_temp, df_plant_utility_table = optimizacion_multiples_con_estados(df_planta_pozas, calidad_precio, df_plant_info, plant_pozas_hermanadas, None, df_pozas_currently_cooking, df_filtro_hist)
                    df_plant_utility_table['planta'] = planta

            if df_temp is not None:
                df_optimizacion_agg = df_optimizacion_agg.append(df_temp)

            if df_plant_utility_table is not None:
                df_utility_table_agg = df_utility_table_agg.append(df_plant_utility_table)

            df_optimizacion_agg = df_optimizacion_agg.reset_index(drop=True)
            df_utility_table_agg = df_utility_table_agg.reset_index(drop=True)
        
        # Eliminar la columna "sin frio"
        # print(df_utility_table_agg)
        if 'sin_hielo' in df_utility_table_agg:
            del df_utility_table_agg['sin_hielo']
        
        df_optimizacion_agg[['contingencia', 'linea_frio', 'aprovechamiento_frio']] = df_optimizacion_agg[['contingencia', 'linea_frio', 'aprovechamiento_frio']].fillna(value=False)

        df_optimizacion_agg['recommended'] = True

        df_optimizacion_final = add_non_recommended_pozas(df_optimizacion_agg, df_pozas_estado_limpio_with_currently_cooking)
    else:
        print('No se encontró ninguna poza para optimizar.')
        return df_optimizacion_agg, df_utility_table_agg

    return df_optimizacion_final, df_utility_table_agg
