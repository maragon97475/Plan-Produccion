import sys
sys.path.append('/home/manuel/Documents/TASA/MIO/tamio2-optimization-data')
import pandas as pd

# from models.tvn_prediction import get_return_model_polynomial_frio, get_return_model_polynomial_trad
from tvn_prediction import get_return_model_polynomial_frio, get_return_model_polynomial_trad

# import opt_utils
# from data.get_data_from_db import get_season_tdc_tvn_data

if __name__ == '__main__':
    print('Reached get_season_tvn_return_model_mae.py')
    # env_file = opt_utils.get_env_file_to_load(os.getenv("ENVIRONMENT"))
    # season_marea_data = get_season_tdc_tvn_data('2020-11-18 00:00:00.000', '2021-02-05 00:00:00.000')
    # season_marea_data = get_season_tdc_tvn_data('2020-11-18 00:00:00.000', '2021-02-05 00:00:00.000')
    season_marea_data = pd.read_csv('analisis_220324.csv')
    
    print(season_marea_data)
    season_marea_data['TDC-Desc'] = season_marea_data['tdc_arrival']
    # season_marea_data['TDC-Desc'] = season_marea_data['TDC-Desc'].astype(float)
    season_marea_data['% Llenado'] = None
    season_marea_data['# Calas'] = None
    # mareas_frio = season_marea_data[(season_marea_data['tipo_bodega'] == 'Frio') & (season_marea_data['frio_system_state'] == 'RC')]
    mareas_frio = season_marea_data[(season_marea_data['tipo_bodega'] == 'Frio')]
    print('len(mareas_frio)', len(mareas_frio))

    # mareas_tradicional = season_marea_data[(season_marea_data['tipo_bodega'] == 'Tradicional') | (season_marea_data['tipo_bodega'] == 'Estanca') | (season_marea_data['owner_group'] == 'T')
    #                                        | ((season_marea_data['tipo_bodega'] == 'Frio') & (season_marea_data['frio_system_state'] != 'RC'))]
    mareas_tradicional = season_marea_data[season_marea_data['tipo_bodega']=='Tradicional']
    print('len(mareas_tradicional)', len(mareas_tradicional))

    mareas_frio['tvn_pred'] = get_return_model_polynomial_frio(mareas_frio[['TDC-Desc', '% Llenado', '# Calas']])
    mareas_frio['tvn_error'] = abs(mareas_frio['tvn_pred'] - mareas_frio['tvn_discharge'])

    mareas_tradicional['tvn_pred'] = get_return_model_polynomial_trad(mareas_tradicional[['TDC-Desc', '% Llenado', '# Calas']])
    mareas_tradicional['tvn_error'] = abs(mareas_tradicional['tvn_pred'] - mareas_tradicional['tvn_discharge'])

    mareas_with_tvn_pred = pd.concat([mareas_frio, mareas_tradicional])
    print(mareas_with_tvn_pred)
    print('Frio describe', mareas_frio.describe())
    print('Trad describe', mareas_tradicional.describe())
    print('Frio mae', mareas_frio['tvn_error'].mean())
    print('Trad mae', mareas_tradicional['tvn_error'].mean())
    print('Total mae', mareas_with_tvn_pred['tvn_error'].mean())

    mareas_with_tvn_pred.to_csv('mareas_with_tvn_pred_202203.csv')


