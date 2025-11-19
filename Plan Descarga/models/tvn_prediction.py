import pandas as pd
import sys
from datetime import datetime
import os

# from data.get_dumped_data import get_dumped_data_date

def get_return_model_polynomial(temp):
    temp = temp.reset_index(drop=True)
    limite_tdc_frio = 22
    limite_tdc_trad = 20
    slope_frio = 0.2
    slope_trad = 0.9
    lista_tvn = []
    for index, row in temp.iterrows():
        if (row['Bodega_Frio'] == 1) & (row['TDC-Desc']>=limite_tdc_frio):
            tvn_inicial = 1.46184258e-05*limite_tdc_frio**5 -1.38737645e-03*limite_tdc_frio**4 + 4.92813314e-02*limite_tdc_frio**3 -8.12249406e-01*limite_tdc_frio**2 + 6.34283538e+00*limite_tdc_frio
            tdc_adicional = row['TDC-Desc'] - limite_tdc_frio
            lista_tvn.append((tvn_inicial + tdc_adicional*slope_frio))
        elif (row['Bodega_Frio'] == 0) & (row['TDC-Desc']>=limite_tdc_trad):
            tvn_inicial = 1.46184258e-05* limite_tdc_trad**5 -1.38737645e-03*limite_tdc_trad**4 + 4.92813314e-02*limite_tdc_trad**3 -8.12249406e-01*limite_tdc_trad**2 + 6.34283538e+00*limite_tdc_trad
            tdc_adicional = row['TDC-Desc'] - limite_tdc_trad
            lista_tvn.append((tvn_inicial + tdc_adicional*slope_trad))
        else:
            lista_tvn.append((1.46184258e-05* row['TDC-Desc']**5 -1.38737645e-03*row['TDC-Desc']**4 + 4.92813314e-02*row['TDC-Desc']**3 -8.12249406e-01*row['TDC-Desc']**2 + 6.34283538e+00*row['TDC-Desc'])*row['Bodega_Frio']+(1.46184258e-05* row['TDC-Desc']**5 -1.38737645e-03*row['TDC-Desc']**4 + 4.92813314e-02*row['TDC-Desc']**3 -8.12249406e-01*row['TDC-Desc']**2 + 6.34283538e+00*row['TDC-Desc'])*(row['Bodega_Estanca']+row['Bodega_Tradicional']))
    return pd.Series(lista_tvn)

def get_tvn_poza_model_slope_increase():
    return 1.194


def replace_percentage_of_bodega_for_thirds(marea_row):
    if marea_row['owner_group'] == 'T':
        marea_row['% Llenado'] = 75
        return marea_row
    else:
        return marea_row


# Según lo último acordado para el modelo del retorno se calculara su TVN hasta el fin de descarga. En caso que fin de descarga sea nulo (sigue descargando) el TDC
# para el cálulo es el tdc actual
def get_return_model_tdc_reference(marea_row):
    return marea_row['tdc_discharge']
    # if marea_row['discharge_end_date'] is not None and type(marea_row['discharge_end_date']) is pd.Timestamp:
    #     return (marea_row['discharge_end_date'] - marea_row['first_cala_start_date']).total_seconds() / 3600
    # else:
    #     return marea_row['tdc_actual']

def get_return_model_polynomial_apply(marea_tdc, equation, TDC_MODEL_THRESHOLD, over_threshold_slope):
    if marea_tdc < TDC_MODEL_THRESHOLD:
        return equation(marea_tdc)
    else:
        return equation(TDC_MODEL_THRESHOLD) + (marea_tdc - TDC_MODEL_THRESHOLD) * over_threshold_slope

# The coefficients are gotten when running the NewMultipleVarModel Notebook and copied here
def get_return_model_polynomial_equation_frio(tdc):
    return 1.46184258e-05*tdc**5 -1.38737645e-03*tdc**4 + 4.92813314e-02*tdc**3 -8.12249406e-01*tdc**2 + 6.34283538e+00*tdc

def get_return_model_polynomial_frio(marea_frios):
    # This frio slope increase is defined as the difference between the return modelo tdc = 22 - tdc
    OVER_MODEL_THRESHOLD_SLOPE_FRIO = 0.2
    return marea_frios['TDC-Desc'].apply(get_return_model_polynomial_apply, args=(get_return_model_polynomial_equation_frio, 22, OVER_MODEL_THRESHOLD_SLOPE_FRIO))

# The coefficients are gotten when running the NewMultipleVarModel Notebook and copied here
def get_return_model_polynomial_equation_trad(tdc):
    return 3.70670930e-05 * tdc ** 5 - 3.06803767e-03 * tdc ** 4 + 9.28971935e-02 * tdc ** 3 - 1.25298480e+00 * tdc ** 2 + 8.00813747e+00 * tdc

def get_return_model_polynomial_trad(marea_trad):
    # This trad slope increase is defined as the difference between the return modelo tdc = 20 - tdc
    OVER_MODEL_THRESHOLD_SLOPE_TRAD = 0.9
    return marea_trad['TDC-Desc'].apply(get_return_model_polynomial_apply, args=(get_return_model_polynomial_equation_trad, 20, OVER_MODEL_THRESHOLD_SLOPE_TRAD))


def get_mareas_return_tvn_prediction(mareas_to_predict):

    mareas_to_predict['TDC-Desc'] = mareas_to_predict.apply(get_return_model_tdc_reference, axis=1)

    mareas_to_predict = mareas_to_predict.apply(replace_percentage_of_bodega_for_thirds, axis=1)

    mareas_frio = mareas_to_predict[(mareas_to_predict['working_frio'] == 1)].copy()

    mareas_tradicional = mareas_to_predict[(mareas_to_predict['working_frio'] != 1)].copy()

    if len(mareas_frio) > 0:
        mareas_frio_with_real_discharge_tvn = mareas_frio[mareas_frio['tvn_discharge'].notnull()].copy()
        if len(mareas_frio_with_real_discharge_tvn) >= 1:
            # In this case, where there is a real tvn from the discharge start, you need to add the additional TVN up to the discharge end date. For this the following formula is done:
            # TVN return model = real_tvn_discharge + (estimated_tvn_discharge_end_date - estimated_tvn_discharge_tdc)
            mareas_frio_with_real_discharge_tvn['tvn'] = mareas_frio_with_real_discharge_tvn['tvn_discharge']
            # mareas_frio_with_real_discharge_tvn['tvn'] = mareas_frio_with_real_discharge_tvn['tvn'] + get_return_model_polynomial_frio(mareas_frio_with_real_discharge_tvn[['TDC-Desc', '% Llenado', '# Calas']])
            # mareas_frio_with_real_discharge_tvn['TDC-Desc'] = mareas_frio_with_real_discharge_tvn['tdc_discharge']
            # mareas_frio_with_real_discharge_tvn['tvn'] = mareas_frio_with_real_discharge_tvn['tvn'] - get_return_model_polynomial_frio(mareas_frio_with_real_discharge_tvn[['TDC-Desc', '% Llenado', '# Calas']])

        mareas_frio_without_real_discharge_tvn = mareas_frio[mareas_frio['tvn_discharge'].isnull()].copy()
        if len(mareas_frio_without_real_discharge_tvn) >= 1:
            mareas_frio_without_real_discharge_tvn['tvn'] = get_return_model_polynomial_frio(mareas_frio_without_real_discharge_tvn[['TDC-Desc', '% Llenado', '# Calas']])

        mareas_frio = pd.concat([mareas_frio_with_real_discharge_tvn, mareas_frio_without_real_discharge_tvn])
    else:
        mareas_frio['tvn'] = None

    if len(mareas_tradicional) > 0:
        mareas_tradicional_with_real_discharge_tvn = mareas_tradicional[mareas_tradicional['tvn_discharge'].notnull()].copy()
        if len(mareas_tradicional_with_real_discharge_tvn) >= 1:
            # In this case, where there is a real tvn from the discharge start, you need to add the additional TVN up to the discharge end date. For this the following formula is done:
            # TVN return model = real_tvn_discharge + (estimated_tvn_discharge_end_date - estimated_tvn_discharge_tdc)
            mareas_tradicional_with_real_discharge_tvn['tvn'] = mareas_tradicional_with_real_discharge_tvn['tvn_discharge']
            # mareas_tradicional_with_real_discharge_tvn['tvn'] = mareas_tradicional_with_real_discharge_tvn['tvn'] + get_return_model_polynomial_trad(mareas_tradicional_with_real_discharge_tvn[['TDC-Desc', '% Llenado', '# Calas']])
            # mareas_tradicional_with_real_discharge_tvn['TDC-Desc'] = mareas_tradicional_with_real_discharge_tvn['tdc_discharge']
            # mareas_tradicional_with_real_discharge_tvn['tvn'] = mareas_tradicional_with_real_discharge_tvn['tvn'] - get_return_model_polynomial_trad(mareas_tradicional_with_real_discharge_tvn[['TDC-Desc', '% Llenado', '# Calas']])

        mareas_tradicional_without_real_discharge_tvn = mareas_tradicional[mareas_tradicional['tvn_discharge'].isnull()].copy()
        if len(mareas_tradicional_without_real_discharge_tvn) >= 1:
            mareas_tradicional_without_real_discharge_tvn['tvn'] = get_return_model_polynomial_trad(mareas_tradicional_without_real_discharge_tvn[['TDC-Desc', '% Llenado', '# Calas']])

        mareas_tradicional = pd.concat([mareas_tradicional_with_real_discharge_tvn, mareas_tradicional_without_real_discharge_tvn])
    else:
        mareas_tradicional['tvn'] = None

    mareas_with_tvn = pd.concat([mareas_frio, mareas_tradicional])
    return mareas_with_tvn


# El aumento del TVN de esta fase se considera el posterior al fin de descarga. Por lo tanto si el fin de descarga todavía es nulo este aumento es 0 (todavía no ha terminado de descargar)
# Caso contrario es el aumento de tdc posterior al fin de descarga
def get_marea_residence_hours(marea_row):
    env_argument_param = os.getenv("ENVIRONMENT")
    if env_argument_param == 'dumped':
        date_now = get_dumped_data_date()
    else:
        date_now = datetime.utcnow()
    return (date_now - marea_row['discharge_start_date']).total_seconds() / 3600
    # if marea_row['discharge_end_date'] is not None and type(marea_row['discharge_end_date']) is pd.Timestamp:
    #     return (date_now - marea_row['discharge_end_date']).total_seconds() / 3600
    # else:
    #     return 0


def negate_category(field):
    if field == 1:
        return 0
    elif field == 0:
        return 1
    else:
        return field


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


def get_tvn_increase_in_poza_for_marea(marea_row):
    return get_tvn_increase(marea_row['residence_hours'], marea_row['working_frio'], marea_row['con_hielo'], marea_row['id_planta'])
    # if not marea_row['working_frio'] and not marea_row['con_hielo']:
    #     if marea_row['residence_hours'] < 7.1:
    #         return 1.72 * marea_row['residence_hours']
    #     else:
    #         return 1.72 * 7.1 + 2.02 * (marea_row['residence_hours'] - 7.1)
    #
    # elif not marea_row['working_frio'] and marea_row['con_hielo']:
    #     if marea_row['residence_hours'] < 12:
    #         return 1.04 * marea_row['residence_hours']
    #     else:
    #         return 1.04 * 12 + 1.15 * (marea_row['residence_hours'] - 12)
    #
    # elif marea_row['working_frio'] and not marea_row['con_hielo']:
    #     if marea_row['residence_hours'] < 7.6:
    #         return 1.09 * marea_row['residence_hours']
    #     else:
    #         return 1.09 * 7.6 + 1.23 * (marea_row['residence_hours'] - 7.6)
    #
    # elif marea_row['working_frio'] and marea_row['con_hielo']:
    #     if marea_row['residence_hours'] < 7.7:
    #         return 0.54 * marea_row['residence_hours']
    #     else:
    #         return 0.54 * 7.7 + 0.57 * (marea_row['residence_hours'] - 7.7)

def add_tvn_poza_slope_increase(mareas_to_add_tvn):
    SLOPE_INCREASE = get_tvn_poza_model_slope_increase()

    mareas_to_add_tvn['residence_hours'] = mareas_to_add_tvn.apply(get_marea_residence_hours, axis=1)

    mareas_to_add_tvn['tvn_desc'] = mareas_to_add_tvn['tvn']

    # mareas_to_add_tvn['tvn'] = mareas_to_add_tvn['tvn'] + (mareas_to_add_tvn['residence_hours'] * SLOPE_INCREASE)

    # mareas_to_add_tvn['tvn'] = 15.4756501992 + 0.82041*mareas_to_add_tvn['tvn'] - 3.3602454738*mareas_to_add_tvn['tdc_discharge'] + 0.3064113283*(mareas_to_add_tvn['tdc_discharge']**2) \
    #                              - 0.0104527259*(mareas_to_add_tvn['tdc_discharge'] ** 3) + 0.0001219146*(mareas_to_add_tvn['tdc_discharge'] ** 4) \
    #                              + 0.3796920054*mareas_to_add_tvn['residence_hours'] \
    #                              + 0.52351*mareas_to_add_tvn['working_frio'].apply(negate_category)*mareas_to_add_tvn['con_hielo']*mareas_to_add_tvn['residence_hours'] \
    #                              + 1.5046545953*mareas_to_add_tvn['working_frio'].apply(negate_category)*mareas_to_add_tvn['con_hielo'].apply(negate_category)*mareas_to_add_tvn['residence_hours'] \
    #                              + 0.3432826495*mareas_to_add_tvn['working_frio']*mareas_to_add_tvn['con_hielo'].apply(negate_category)*mareas_to_add_tvn['residence_hours'] \
    #                              + 0*mareas_to_add_tvn['working_frio']*mareas_to_add_tvn['con_hielo']*mareas_to_add_tvn['residence_hours']

    mareas_to_add_tvn['tvn'] = mareas_to_add_tvn['tvn_desc'] + mareas_to_add_tvn.apply(get_tvn_increase_in_poza_for_marea, axis=1)

    return mareas_to_add_tvn
