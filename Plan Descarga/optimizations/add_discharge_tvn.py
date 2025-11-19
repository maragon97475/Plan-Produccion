import os
import pandas as pd
from models.tvn_prediction import replace_percentage_of_bodega_for_thirds, get_return_model_polynomial_frio, get_return_model_polynomial_trad


def run_add_discharge_tvn_to_pozas_estados(get_data):
    print('Started to run_add_discharge_tvn_to_pozas_estados: ' + os.getenv("ENVIRONMENT"))

    ## datos de pozas
    df_pozas_estado = get_data.get_pozas_estado()

    df_pozas_estado = df_pozas_estado.drop_duplicates(subset=['marea_id'])

    df_pozas_estado['TDC-Desc'] = df_pozas_estado['tdc_discharge']
    df_pozas_estado['% Llenado'] = df_pozas_estado['pp_llenado_bodega']
    df_pozas_estado['# Calas'] = df_pozas_estado['calas_count']

    mareas_to_predict = df_pozas_estado.apply(replace_percentage_of_bodega_for_thirds, axis=1)

    mareas_frio = mareas_to_predict[mareas_to_predict['tipo_bodega'] == 'Frio'].copy()

    mareas_tradicional = mareas_to_predict[(mareas_to_predict['tipo_bodega'] == 'Tradicional') | (mareas_to_predict['tipo_bodega'] == 'Estanca') | (mareas_to_predict['owner_group'] == 'T')].copy()

    if len(mareas_frio) > 0:
        mareas_frio['estimated_discharge_tvn'] = get_return_model_polynomial_frio(mareas_frio[['TDC-Desc', '% Llenado', '# Calas']])
    else:
        mareas_frio['estimated_discharge_tvn'] = None

    if len(mareas_tradicional) > 0:
        mareas_tradicional['estimated_discharge_tvn'] = get_return_model_polynomial_trad(mareas_tradicional[['TDC-Desc', '% Llenado', '# Calas']])
    else:
        mareas_tradicional['estimated_discharge_tvn'] = None

    df_pozas_estado_with_tvn = pd.concat([mareas_frio, mareas_tradicional])

    return df_pozas_estado_with_tvn
