import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pyodbc

plantas = ["MALABRIGO", "CHIMBOTE", "SUPE", "VEGUETA", "CALLAO", "PISCO SUR"]
threshold_degradacion = 3
pozas = 10

cnxn_mio = pyodbc.connect('DRIVER={SQL Server};SERVER=srv-db-east-us003.database.windows.net;DATABASE=db_cfa_prd01;UID=userdbowner;PWD=$P4ssdbowner01#;')
cnxn_juv = pyodbc.connect('DRIVER={SQL Server};SERVER=srv-db-east-us-tasa-his-02.database.windows.net;DATABASE=db_bi_production_prd;UID=userpowerbi;PWD=#p4ssw0rdp0w3rb1#;')