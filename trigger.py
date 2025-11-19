
import time
import subprocess

while True:
    # subprocess.run(["python3", "automate_pp.py"])
    print("Running automate_pp.py...")
    # try: 
    subprocess.run([
    r"C:\Users\rnina\Desktop\DA\Proyeccion-plan-de-produccion-diaria\proy_plan_prod\Scripts\python.exe",
    r"C:\Users\rnina\Desktop\DA\Proyeccion-plan-de-produccion-diaria\automate_pp.py"
    ])
    # except:
    #     "ERROR"
    print("automate_pp.py finished,  waiting for 10 minutes...")
    time.sleep(300) # 300 segundos = 5 minutos