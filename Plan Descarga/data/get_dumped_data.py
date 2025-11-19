# Standard library imports
import os

# Third party imports
from dotenv import load_dotenv
import pyodbc
import pandas as pd
from datetime import datetime

def get_plantas_habilitadas():
    return pd.read_csv('./dumped_data/df_plantas_habilitadas_130520.csv')

def get_active_mareas_with_location_and_static_data():
    return pd.read_csv('./dumped_data/df_embarcaciones_130520.csv')

def get_pozas_estado():
    return pd.read_csv('./dumped_data/df_pozas_estado_130520.csv')

def get_ubicacion_capacidad_pozas():
    return pd.read_csv('./dumped_data/df_pozas_ubicacion_capacidad_130520.csv')

def get_lineas_velocidad_descarga():
    return pd.read_csv('./dumped_data/df_tiempo_descarga_130520.csv')

def get_requerimiento_planta():
    return pd.read_csv('./dumped_data/df_requerimiento_plantas_130520.csv')

def get_lineas_reservadas_terceros():
    return pd.read_csv('./dumped_data/df_lineas_reservada_terceros_130520.csv')

def get_plantas_velocidades():
    return pd.read_csv('./dumped_data/df_plantas_velocidades_130520.csv')

def get_pozas_hermanadas():
    return pd.read_csv('./dumped_data/df_pozas_hermanadas.csv')

def get_minimo_perc_bodega_recom_retorno():
    return pd.read_csv('./dumped_data/get_minimo_perc_bodega_recom_retorno_130520.csv')

def get_calidades_precio_venta():
    return pd.read_csv('./dumped_data/df_calidades_precio_venta.csv')

def get_costo_combustible():
    return pd.read_csv('./dumped_data/get_costo_combustible_130520.csv')

def get_data_restricciones():
    return pd.read_csv('./dumped_data/df_restricciones.csv')

def get_minimos_planta():
    return pd.read_csv('./dumped_data/df_minimo_plantas.csv')

def get_contingencia_max_tdc_limit_discharge_tradicionales():
    return 24

def get_contingencia_max_tdc_limit_cocina_tradicionales():
    return 36

def get_contingencia_max_tvn_limit_cocina_all():
    return 60

def get_dumped_data_date():
    return datetime.strptime(pd.read_csv('./dumped_data/dump_date_df.csv', parse_dates=True).loc[0, 'dump_date'], '%Y-%m-%d %H:%M:%S.%f')



def main():
    print('get dumped data hello world')


if __name__ == "data.get_saved_dummy_data":
    main()
