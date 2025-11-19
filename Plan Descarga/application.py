import sys
import os
from time import time, sleep
from datetime import datetime
from dotenv import load_dotenv
from optimizations import return_optimization as return_opt
from optimizations import cocina_optimization as cocina_opt
from optimizations import add_discharge_tvn
from index_fp import main_fp, main_disponibilidad_eps
from data import insert_data_to_db, get_data_from_db, db_connection, get_dumped_data, insert_data_to_db_nmd, get_data_from_db_nmd
import opt_utils
import traceback
import logging
import requests

import pandas as pd

from flask import Flask
from flask import request
from flask_cors import CORS
import routes as api_routes
from datetime import timedelta
app = Flask(__name__)
CORS(app)

pd.set_option('display.max_columns', None)

typeSaveData = ['save', 'only-print', 'save-all']
typeExecute = ['all', 'retorno', 'cocina', 'forecastplanning', 'panelbarcos']
time_min = 5

@app.route(api_routes.hello_world)
def hello_world():
    try:
        print('reached hello!')
        return 'Hello! This is the MIO OPT API - ' + os.getenv("ENVIRONMENT")
    except Exception as e:
        logging.error("hello_world error: " + str(e))
        return {"error": str(e)}, 500

# @app.route(api_routes.execute_interpolation)
# def execute_interpolation():
#     print('Ejecutando interpolacion')
#     try:
#         import interpolation
#         time_execution = interpolation.main_execute()
#         print('Termino la ejecucion de interpolacion en ', str(time_execution))
#         return {'res':'Se ejecuto la interpolacion'}, 200
#     except Exception as e:
#         logging.error("Interpolacion error: " + str(e))
#         return {"error": str(e)}, 500


def send_backend_mio():
    try:
        backendurl = os.getenv("BACKEND_URL") + '/marea/update-mareas'
        print(backendurl)
        requests.get(backendurl)
        # ,headers = {"Authorization": 'Bearer ' + token})
    except Exception as e:
        print("send backend mio error: " + str(e))

def generateToken():
    try:
        details = {
            'client_id': os.getenv("AD_CLIENT_ID"),
            'grant_type': 'client_credentials',
            'scope': os.getenv("AD_CLIENT_ID") + '/.default',
            'client_secret': os.getenv("AD_SECRET")
        }

        result = requests.post('https://login.microsoftonline.com/' + os.getenv("AD_TENANT") + '/oauth2/v2.0/token',
            data=details
        )
        res = result.json();

        return res
    except Exception as e:
        print("testToken error: " + str(e))

def run_main_optimization():
    tic_initial = time();
    print('Started to run main optimization')
    env_argument_param = os.getenv("ENVIRONMENT")
    print('env_argument_param =>' + env_argument_param)
    opt_result_argument_param = os.getenv("RESULT_OPT")
    print('opt_result_argument_param =>' + opt_result_argument_param)
    algorithms_to_run = os.getenv("ALGORITHM_RUN")
    print('algorithms_to_run =>' + algorithms_to_run)

    # get_data_from_db and get_saved_dummy_data needs to have the same function names so it is indifferent for the optimizations.
    if env_argument_param == 'prod':
        get_data = get_data_from_db
    elif env_argument_param == 'dev':
        from data import get_data_from_csv
        get_data = get_data_from_csv
    elif env_argument_param == 'dumped':
        get_data = get_dumped_data
    else:
        raise Exception("Missing or invalid first argument")

    tic_retorno = time();

    db_connection.initiate_connection()
    get_data_from_db.import_connection()
    insert_data_to_db.import_connection()
    get_data_from_db_nmd.import_connection()
    insert_data_to_db_nmd.import_connection()

    if (algorithms_to_run in ['all', 'retorno']):
        try:
            if os.getenv("MACHINE") == 'azure':
                print('Save optimization start date')
                df_execution_model = get_data_from_db.get_ejecucion_modelo()
                if len(df_execution_model)>0:
                    last_execution_model = df_execution_model[-1:]
                
                    last_end_execution = last_execution_model['FEH_FIN'].item()
                    id_execution = last_execution_model['ID_EJECUCION'].item()

                    start_execution = datetime.utcnow()
                    df_execution_time = pd.DataFrame(columns=['FEH_INICIO','FEH_FIN'])
                    thrs = 10
                    diferencia_minutos = (start_execution - last_execution_model['FEH_INICIO']).dt.total_seconds() / 60
                    if ((not pd.notnull(last_end_execution)) & (diferencia_minutos < thrs)).item():
                        print('EJECUCIÓN EN PROGRESO ... IGNORANDO EJECUCIÓN ACTUAL', 'FECHA:', start_execution)
                        sys.exit()
                    else:
                        pass
                        # Guardar id ejecucion
                        df_execution_time.loc[0,'FEH_INICIO'] = start_execution
                        insert_data_to_db.insert_execution_model(df_execution_time)
                        df_execution_model = get_data_from_db.get_ejecucion_modelo()
                        last_execution_model = df_execution_model[-1:]
                        id_execution = last_execution_model['ID_EJECUCION'].item()
            
            print('***Started retorno optimization')
            
            
            return_optimization_date = datetime.utcnow()
            # return_opt_result_df, return_opt_errors_df, return_flags_head, return_flags_ordenes = return_opt.run_return_optimization(get_data)
            return_opt_result_df, return_opt_errors_df, return_flags_head, return_flags_ordenes, return_utility_table, df_retorno_utilidad = return_opt.run_return_optimization(get_data)
            df_retorno_utilidad = pd.merge(df_retorno_utilidad, return_opt_result_df[['marea_id', 'planta_retorno', 'velocidad_retorno']], how='left', on='marea_id')
            print('End return optimization - printing results')
            return_optimization_end = datetime.utcnow()
            
            df_current_execution = get_data_from_db.get_ejecucion_modelo()
            current_execution = int(max(df_current_execution['ID_EJECUCION']))
            insert_data_to_db.update_execution_model(return_optimization_end, current_execution)
            
            
            # print('return_opt_result_df: \n', return_opt_result_df)
            # print('return_opt_errors_df: \n', return_opt_errors_df)
            if os.getenv("MACHINE") == 'local':
                return_opt_result_df.to_csv(f'outputs/return_opt_result_df_{str(return_optimization_date.date())}_{return_optimization_date.hour}_{return_optimization_date.minute}_{return_optimization_date.second}.csv')
            if env_argument_param == 'dev':
                return_opt_result_df.to_csv(f'outputs/return_opt_result_df_{str(return_optimization_date.date())}_{return_optimization_date.hour}_{return_optimization_date.minute}_{return_optimization_date.second}.csv')
                return_opt_errors_df.to_csv(f'outputs/return_opt_errors_df_{str(return_optimization_date.date())}_{return_optimization_date.hour}_{return_optimization_date.minute}_{return_optimization_date.second}.csv')
                df_retorno_utilidad.sort_values('prioridad').to_csv(f'outputs/df_retorno_utilidad{str(return_optimization_date.date())}_{return_optimization_date.hour}_{return_optimization_date.minute}_{return_optimization_date.second}.csv')

            if opt_result_argument_param == 'save':
                insert_data_to_db.insert_return_optimization_result(return_opt_result_df, return_optimization_date)
                insert_data_to_db.insert_return_optimization_errors(return_opt_errors_df, return_optimization_date)
                insert_data_to_db.insert_return_plant_utility(df_retorno_utilidad, return_optimization_date)
                insert_data_to_db.insert_return_flags_and_ordenes(return_flags_head, return_flags_ordenes, return_optimization_date, return_utility_table)
                
            if opt_result_argument_param == 'save-all':
                insert_data_to_db.insert_return_optimization_result(return_opt_result_df, return_optimization_date)
                insert_data_to_db.insert_return_optimization_errors(return_opt_errors_df, return_optimization_date)
                insert_data_to_db.insert_return_plant_utility(df_retorno_utilidad, return_optimization_date)
                insert_data_to_db.insert_return_flags_and_ordenes(return_flags_head, return_flags_ordenes, return_optimization_date, return_utility_table)

                # Nuevo modelo de Datos
                insert_data_to_db_nmd.insert_recom_return(return_opt_result_df, return_optimization_date)
                insert_data_to_db_nmd.insert_return_optimization_errors(return_opt_errors_df, return_optimization_date)
                insert_data_to_db_nmd.insert_recom_discharge(return_opt_result_df, return_optimization_date)
                # insert_data_to_db_nmd.insert_discharge_utility(return_flags_head, return_flags_ordenes, return_optimization_date, return_utility_table)

        except Exception as ex:
            print(
                "Error return optimization: " + ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
    toc_retorno = time();
    print('***Finished algorithm retorno in ' + str(toc_retorno - tic_retorno))
    
    if (algorithms_to_run in ['all', 'cocina']):
        try:
            print('***Started cocina optimization')

            cocina_optimization_date = datetime.utcnow()
            cocina_opt_result_df, df_utility_table_agg = cocina_opt.run_cocina_optimization(get_data)
            # print('cocina_opt_result_df: \n', cocina_opt_result_df)
            # print('df_utility_table_agg: \n', df_utility_table_agg)

            if opt_result_argument_param in ['save', 'save-all']:
                insert_data_to_db.insert_cocina_optimization_result(cocina_opt_result_df, df_utility_table_agg, cocina_optimization_date)

            print('Finished cocina optimization')
        except Exception as ex:
            print("Error cocina optimization: " + ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))

    # print('Finished running main optimization.')
    toc_cocina = time();
    print('***Finished algorithm cocina in ' + str(toc_cocina - toc_retorno))

    if (algorithms_to_run in ['all', 'cocina']):
        try:
            print('***Started running add discharge tvn to database')

            add_discharge_tvn_date = datetime.utcnow()
            df_pozas_estado_with_tvn = add_discharge_tvn.run_add_discharge_tvn_to_pozas_estados(get_data)

            if opt_result_argument_param in ['save']:
                insert_data_to_db.add_discharge_tvn_to_db(df_pozas_estado_with_tvn, add_discharge_tvn_date)

            if opt_result_argument_param in ['save-all']:
                insert_data_to_db.add_discharge_tvn_to_db(df_pozas_estado_with_tvn, add_discharge_tvn_date)
                # insert_data_to_db_nmd.add_discharge_tvn_to_db(df_pozas_estado_with_tvn, add_discharge_tvn_date)
            toc_cocina2 = time();
            print('***Finished running add discharge tvn to database in '+ str(toc_cocina2 - toc_cocina))
            # print('df_pozas_estado_with_tvn: \n', df_pozas_estado_with_tvn)
        except Exception as ex:
            print("Error add discharge tvn: " + ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))

    try:
        df_execution_time.loc[0, 'FEH_FIN'] = return_optimization_end
        insert_data_to_db.update_execution_model(return_optimization_end, id_execution)
    except:
        print('No se registra log de finalizacion del modelo')
    
    toc_finished = time();
    print('******Finished all in ' + str(toc_finished - tic_initial))
    
    db_connection.close_connections()
    
@app.route(api_routes.execute_opt_api)
def execute_opt_api():
    try:
        print('Inicia algoritmo')
        print("RESULT_OPT => " + os.getenv("RESULT_OPT"))
        print("ALGORITHM_RUN => " + os.getenv("ALGORITHM_RUN"))

        if (os.getenv("RESULT_OPT") in typeSaveData) and (os.getenv("ALGORITHM_RUN") in typeExecute):
            print("Starting the execution")
            run_main_optimization()

            print("Sending update to Backend MIO")
            # dataToken = generateToken()
            # token = dataToken['access_token']
            send_backend_mio()

        else:
            print('Missing valid env arguments')
        return {'res':'Se ejecuto el algoritmo'}, 200
    except Exception as e:
        try:
            db_connection.close_connections()
        except Exception as e:
            print("algoritmo error: " + str(e))
            return {"error": str(e)}, 500
        print("algoritmo error: " + str(e))
        return {"error": str(e)}, 500

@app.route(api_routes.execute_fp)
def execute_fp():
    try:
        print('Inicia Forecast Planning')
        print("RESULT_OPT => " + os.getenv("RESULT_OPT"))
        print("ALGORITHM_RUN => " + os.getenv("ALGORITHM_RUN"))
        opt_result_argument_param = os.getenv("RESULT_OPT")
        if (os.getenv("RESULT_OPT") in typeSaveData) and (os.getenv("ALGORITHM_RUN") in typeExecute):
            print("Starting the execution")
            tic_initial = time();
            db_connection.initiate_connection()
            get_data_from_db.import_connection()
            insert_data_to_db.import_connection()
            get_data_from_db_nmd.import_connection()
            insert_data_to_db_nmd.import_connection()
       
            tabla_simulation_descarga, df_dz, df_horas, df_densidad_arribos = main_fp()

            if opt_result_argument_param in ['save-all']:
                insert_data_to_db.insert_fp_pd(tabla_simulation_descarga)
                insert_data_to_db.insert_fp_densidad_arribo(df_densidad_arribos)
                insert_data_to_db.insert_fp_horas(df_horas)
                insert_data_to_db.insert_fp_dz(df_dz)
                insert_data_to_db.insert_fp_dz_backup(df_dz)

                # Llamar al PP
            toc_finished = time();
            print('******Finished Forecast Planning in ' + str(toc_finished - tic_initial))
        return {'res':'Se ejecuto el algoritmo Forecast Planning'}, 200
    except Exception as e:
        try:
            db_connection.close_connections()
        except Exception as e:
            print("algoritmo error Forecast Planning: " + str(e))
            return {"error": str(e)}, 500
        print("algoritmo error Forecast Planning: " + str(e))
        return {"error": str(e)}, 500

@app.route(api_routes.execute_panel_pesca)
def execute_panel_pesca():
    try:
        print('Inicia Panel Pesca')
        print("RESULT_OPT => " + os.getenv("RESULT_OPT"))
        print("ALGORITHM_RUN => " + os.getenv("ALGORITHM_RUN"))
        opt_result_argument_param = os.getenv("RESULT_OPT")
        if (os.getenv("RESULT_OPT") in typeSaveData) and (os.getenv("ALGORITHM_RUN") in typeExecute):
            print("Starting the execution")
            tic_initial = time();
            db_connection.initiate_connection()
            get_data_from_db.import_connection()
            insert_data_to_db.import_connection()
            get_data_from_db_nmd.import_connection()
            insert_data_to_db_nmd.import_connection()
       
            panel_disponibilidad_eps = main_disponibilidad_eps().reset_index(drop=True)

            if opt_result_argument_param in ['save-all']:
                # pass
                insert_data_to_db.insert_panel_disponibilidad(panel_disponibilidad_eps)
                # insert_data_to_db.insert_fp_densidad_arribo(df_densidad_arribos)
                # insert_data_to_db.insert_fp_horas(df_horas)
                # insert_data_to_db.insert_fp_dz(df_dz)
                # insert_data_to_db.insert_fp_dz_backup(df_dz)
            toc_finished = time();
            print('******Finished Panel Pesca' + str(toc_finished - tic_initial))
        return {'res':'Se ejecuto el algoritmo Forecast Planning'}, 200
    except Exception as e:
        try:
            db_connection.close_connections()
        except Exception as e:
            print("algoritmo error Forecast Planning: " + str(e))
            return {"error": str(e)}, 500
        print("algoritmo error Forecast Planning: " + str(e))
        return {"error": str(e)}, 500


@app.route(api_routes.execute_dz)
def execute_dz():
    try:
        print('Inicia Descarga Zona')
        print("RESULT_OPT => " + os.getenv("RESULT_OPT"))
        print("ALGORITHM_RUN => " + os.getenv("ALGORITHM_RUN"))
        opt_result_argument_param = os.getenv("RESULT_OPT")
        if (os.getenv("RESULT_OPT") in typeSaveData) and (os.getenv("ALGORITHM_RUN") in typeExecute):
            print("Starting the execution")
            tic_initial = time();
            db_connection.initiate_connection()
            get_data_from_db.import_connection()
            insert_data_to_db.import_connection()
            get_data_from_db_nmd.import_connection()
            insert_data_to_db_nmd.import_connection()
       
            panel_disponibilidad_eps = main_disponibilidad_eps().reset_index(drop=True)

            if opt_result_argument_param in ['save-all']:
                # pass
                insert_data_to_db.insert_panel_disponibilidad(panel_disponibilidad_eps)
                # insert_data_to_db.insert_fp_densidad_arribo(df_densidad_arribos)
                # insert_data_to_db.insert_fp_horas(df_horas)
                # insert_data_to_db.insert_fp_dz(df_dz)
                # insert_data_to_db.insert_fp_dz_backup(df_dz)
            toc_finished = time();
            print('******Finished Descarga Zona' + str(toc_finished - tic_initial))
        return {'res':'Se ejecuto el algoritmo Descarga Zona'}, 200
    except Exception as e:
        try:
            db_connection.close_connections()
        except Exception as e:
            print("algoritmo error Descarga Zona: " + str(e))
            return {"error": str(e)}, 500
        print("algoritmo error Descarga Zona: " + str(e))
        return {"error": str(e)}, 500
    
if __name__ == '__main__':
    # The environment file should include fields to alternate between runnning cplex locally (installed) or calling a CPLEX Watson API
    print('Reached main optimization __main__')
    print(sys.argv[1])
    env_file = opt_utils.get_env_file_to_load(sys.argv[1])
    load_dotenv(env_file)
    
    app.debug = False
    app.run(host='0.0.0.0', port=os.getenv("PORT"))

    # time_as_second = time_min * 60
    # if env_file is not None and 
    # starttime = time()
    # while True: 
    #     print("RESULT_OPT => " + os.getenv("RESULT_OPT"))
    #     print("ALGORITHM_RUN => " + os.getenv("ALGORITHM_RUN"))

    #     if (os.getenv("RESULT_OPT") in typeSaveData) and (os.getenv("ALGORITHM_RUN") in typeExecute):
    #         print("Starting the execution")
    #         run_main_optimization()
    #     else:
    #         print('Missing valid env arguments')

    #     sleep(time_as_second - ((time() - starttime) % time_as_second))

