import pandas as pd


def change_bodega_names(bodega):
    if bodega == 'Flota S/Frio' or bodega == 'Tercero':
        return 'Tradicional'
    elif bodega == 'Flota P/Frio':
        return 'Frio'


def change_bodega_names_for_2019_1(bodega):
    if bodega == 'PROPIO S/FRIO' or bodega == 'TERCERO':
        return 'Tradicional'
    elif bodega == 'PROPIO C/FRIO':
        return 'Frio'


def change_plant_name_for_retorno(planta):
    if planta == 'Callao Norte':
        return 'CALLAO'
    elif planta == 'Malabrigo Sur':
        return 'MALABRIGO'
    return planta


# This includes from 2013-II to 2018-I but only 2018-I is selected
mareas_retorno_past_data = pd.read_csv('./models/raw_data/DB_MAREAS_analisis_temporada.csv')
mareas_retorno_past_data = mareas_retorno_past_data[['Marea', 'Planta', 'Embarcacion', 'Bodega', '% Llenado', '# Calas', 'TDC-Desc', 'TVN', 'año', 'temporada_año']]

mareas_retorno_past_data_2018_1 = mareas_retorno_past_data[(mareas_retorno_past_data['temporada_año'] == '2018-I')]
mareas_retorno_past_data_2018_1.to_csv('./models/inputs/mareas_retorno_past_data_2018_1.csv', index=False)
mareas_retorno_past_data_2018_2 = mareas_retorno_past_data[(mareas_retorno_past_data['temporada_año'] == '2018-II')]
mareas_retorno_past_data_2018_2.to_csv('./models/inputs/mareas_retorno_past_data_2018_2.csv', index=False)


mareas_retorno_2019_1 = pd.read_excel('./models/raw_data/Mareas_2019 I.xls')
mareas_retorno_2019_1 = mareas_retorno_2019_1[['MAREA', 'PLANTA', 'E/P', 'FLOTA', '% Uso bodega', 'TDC_DESCAR', 'TVN_DESCAR', 'AÑO']]
mareas_retorno_2019_1['temporada_año'] = '2019-I'
mareas_retorno_2019_1['PLANTA'] = mareas_retorno_2019_1['PLANTA'].apply(change_plant_name_for_retorno)
mareas_retorno_2019_1['PLANTA'] = mareas_retorno_2019_1['PLANTA'].str.upper()
mareas_retorno_2019_1 = mareas_retorno_2019_1[(mareas_retorno_2019_1['TDC_DESCAR'].notnull()) & mareas_retorno_2019_1['TVN_DESCAR'].notnull()]
mareas_retorno_2019_1['FLOTA'] = mareas_retorno_2019_1['FLOTA'].apply(change_bodega_names_for_2019_1)
mareas_retorno_2019_1['% Uso bodega'] = mareas_retorno_2019_1['% Uso bodega'] * 100

calas_2019 = pd.read_csv('./models/raw_data/calas_2019_1.csv')
calas_2019 = calas_2019[(calas_2019['total_catch'] > 0) & (calas_2019['number_of_calas'] >= 1)]

mareas_retorno_2019_1 = pd.merge(mareas_retorno_2019_1, calas_2019, how='inner', left_on='MAREA', right_on='marea_id')

mareas_retorno_2019_1 = mareas_retorno_2019_1[['MAREA', 'PLANTA', 'E/P', 'FLOTA', '% Uso bodega', 'TDC_DESCAR', 'TVN_DESCAR', 'AÑO', 'temporada_año', 'number_of_calas']]

mareas_retorno_2019_1.rename(columns={'MAREA': 'Marea', 'PLANTA': 'Planta', 'E/P': 'Embarcacion', 'FLOTA': 'Bodega', '% Uso bodega': '% Llenado',
                                      'number_of_calas': '# Calas', 'TDC_DESCAR': 'TDC-Desc', 'TVN_DESCAR': 'TVN', 'AÑO': 'año'}, inplace=True)

mareas_retorno_2019_1.to_csv('./models/inputs/mareas_retorno_2019_1.csv', index=False)


mareas_retorno_2019_2 = pd.read_excel('./models/raw_data/Mareas_2019 II.xlsx')
mareas_retorno_2019_2 = mareas_retorno_2019_2[['MAREA', 'PLANTA', 'E_P', 'FLOTA', 'USO_BODEGA', 'NROCALAS', 'TDC_DESCAR', 'TVN_DESCAR', 'AÑO']]
mareas_retorno_2019_2['temporada_año'] = '2019-II'
mareas_retorno_2019_2['PLANTA'] = mareas_retorno_2019_2['PLANTA'].apply(change_plant_name_for_retorno)
mareas_retorno_2019_2['PLANTA'] = mareas_retorno_2019_2['PLANTA'].str.upper()
mareas_retorno_2019_2 = mareas_retorno_2019_2[(mareas_retorno_2019_2['TDC_DESCAR'].notnull()) & mareas_retorno_2019_2['TVN_DESCAR'].notnull()]
mareas_retorno_2019_2['FLOTA'] = mareas_retorno_2019_2['FLOTA'].apply(change_bodega_names)
mareas_retorno_2019_2['% Uso bodega'] = mareas_retorno_2019_2['USO_BODEGA'] * 100

mareas_retorno_2019_2 = mareas_retorno_2019_2[['MAREA', 'PLANTA', 'E_P', 'FLOTA', '% Uso bodega', 'NROCALAS', 'TDC_DESCAR', 'TVN_DESCAR', 'AÑO', 'temporada_año']]

mareas_retorno_2019_2.rename(columns={'MAREA': 'Marea', 'PLANTA': 'Planta', 'E_P': 'Embarcacion', 'FLOTA': 'Bodega', '% Uso bodega': '% Llenado',
                                      'NROCALAS': '# Calas', 'TDC_DESCAR': 'TDC-Desc', 'TVN_DESCAR': 'TVN', 'AÑO': 'año'}, inplace=True)

mareas_retorno_2019_2.to_csv('./models/inputs/mareas_retorno_2019_2.csv', index=False)


mareas_retorno_2020_1 = pd.read_excel('./models/raw_data/Mareas_2020 I.XLSX')
mareas_retorno_2020_1 = mareas_retorno_2020_1[['MAREA', 'PLANTA', 'FLOTA', 'NROCALAS', 'TDC_DESCAR', 'TVN_DESCAR', 'AÑO']]
mareas_retorno_2020_1['temporada_año'] = '2020-I'
mareas_retorno_2020_1['PLANTA'] = mareas_retorno_2020_1['PLANTA'].apply(change_plant_name_for_retorno)
mareas_retorno_2020_1['PLANTA'] = mareas_retorno_2020_1['PLANTA'].str.upper()
mareas_retorno_2020_1 = mareas_retorno_2020_1[(mareas_retorno_2020_1['TDC_DESCAR'].notnull()) & mareas_retorno_2020_1['TVN_DESCAR'].notnull()]
mareas_retorno_2020_1['FLOTA'] = mareas_retorno_2020_1['FLOTA'].apply(change_bodega_names)

mareas_retorno_2020_1.rename(columns={'MAREA': 'Marea', 'PLANTA': 'Planta', 'FLOTA': 'Bodega', 'NROCALAS': '# Calas',
                                      'TDC_DESCAR': 'TDC-Desc', 'TVN_DESCAR': 'TVN', 'AÑO': 'año'}, inplace=True)

mareas_retorno_2020_1.to_csv('./models/inputs/mareas_retorno_2020_1.csv', index=False)

print(mareas_retorno_2020_1)


mareas_retorno_full_data_all_seasons = pd.concat([mareas_retorno_past_data_2018_1, mareas_retorno_past_data_2018_2, mareas_retorno_2019_1, mareas_retorno_2019_2, mareas_retorno_2020_1])
mareas_retorno_full_data_all_seasons.to_csv('./models/inputs/mareas_retorno_full_data_all_seasons.csv', index=False)

mareas_retorno_full_data_second_seasons = pd.concat([mareas_retorno_past_data_2018_2, mareas_retorno_2019_2])
mareas_retorno_full_data_second_seasons.to_csv('./models/inputs/mareas_retorno_full_data_second_seasons.csv', index=False)

# Data del 2019-II hasta los primeros días de la temporada 2020-I
tvn_a_cocina_historico = pd.read_excel('./models/raw_data/TVN_A_COCINA_HISTORICO.xlsx')
tvn_a_cocina_historico = tvn_a_cocina_historico[['PLANTA', 'FECHA P.', 'HORA', 'TDC REAL', 'TVN COCINA']]
tvn_a_cocina_historico = tvn_a_cocina_historico[(tvn_a_cocina_historico['TDC REAL'].notnull()) & tvn_a_cocina_historico['TVN COCINA'].notnull()]
tvn_a_cocina_historico['PLANTA'] = tvn_a_cocina_historico['PLANTA'].apply(lambda x: 'PISCO SUR' if x == 'P. Sur' else x)
tvn_a_cocina_historico['PLANTA'] = tvn_a_cocina_historico['PLANTA'].str.upper()

tvn_a_cocina_historico.to_csv('./models/inputs/tvn_a_cocina_historico.csv', index=False)
print(tvn_a_cocina_historico)
