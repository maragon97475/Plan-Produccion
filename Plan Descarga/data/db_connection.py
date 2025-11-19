# System imports
import os
import sys

# Third party imports
from dotenv import load_dotenv
import opt_utils
#? import pyodbc
import sqlalchemy
from sqlalchemy import create_engine


#? Initial connection to the database
def initial_query():
    print('Initiating hello world connection')

    #? pyodbc requieres the creation of a cursor for execute
    #? as this is done inside the connection, a manual commit is requiered
    # cursor = cnxn.cursor()
    # cursor.execute("SELECT 1+1 as result")
    # cnxn.commit()
    
    #? sqlalchemy permits execute from 
    #? an engine (in which case a connection will be created, commited and closed internally)
    #? or a connection (commits are automatic)
    connection.execute("SELECT 1+1 as result")

    print('Finishing hello world connection')
    return


#? pyodbc and sqlalchemy can be used for mostly the same purpose
#? sqlalchemy is recommended by pandas for read_sql and requiered for to_sql
def main():
    #? pyodbc.connect creates a connection object
    # global cnxn
    # cnxn = pyodbc.connect('DRIVER={' + os.getenv("SQL_SERVER_DRIVER") + '};'
    #                       + 'SERVER=' + os.getenv("DATABASE_SERVER") + ';'
    #                       + 'DATABASE=' + os.getenv("DATABASE_NAME") + ';'
    #                       + 'UID=' + os.getenv("DATABASE_USERNAME") + ';'
    #                       + 'PWD=' + os.getenv("DATABASE_PASSWORD") + ';'
    #                       )

    #? fast_executemany has to be called for each cursor created from the cursor itself
    # global crsr
    # crsr = cnxn.cursor()
    # crsr.fast_executemany = True

    #? sqlalchemy's create_engine creates a sort of connection generator that has a connection pool
    #? fast_executemany=True can be passed as a second parameter
    global engine
    engine = create_engine('mssql+pyodbc://' 
                            + os.getenv("DATABASE_USERNAME") + ':' 
                            + os.getenv("DATABASE_PASSWORD") + '@' 
                            + os.getenv("DATABASE_SERVER") + ':1433/' 
                            + os.getenv("DATABASE_NAME") + '?DRIVER=' 
                            + os.getenv("SQL_SERVER_DRIVER"))
    
    #? connections can be created from the engine
    global connection
    connection = engine.connect()

    initial_query()
    print('Driver ', os.getenv("SQL_SERVER_DRIVER"), ' is being used')
    print('Finished creating connection')


#? connections can be closed when done, engines can clear their connection pools
def close_connections():
    if connection:
        connection.close()
        print('connection closed')
    if engine:
        engine.dispose()
        print('engine disposed')


#? This function gets the environment variables and then initiates the connection
def initiate_connection():
    print('Starting connection')
    env_file = opt_utils.get_env_file_to_load(sys.argv[1])

    if env_file is not None:
        load_dotenv(env_file)
        main()
    else:
        print('Missing valid env argument')
