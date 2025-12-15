import os,sys
paths=[r"output\_stale\amicaouttmp_2\W",
       r"output\_stale\amicaouttmp_2\out.txt",
       r"output\_stale\amicaouttmp_28\W",
       r"output\_stale\amicaouttmp_28\out.txt",
       r"output\_stale\amicaouttmp_19\W",
       r"output\_stale\amicaouttmp_19\out.txt",
       r"output\_stale\amicaouttmp_33\W",
       r"output\_stale\amicaouttmp_33\out.txt"]
from datetime import datetime
for p in paths:
    if os.path.exists(p):
        m=os.path.getmtime(p)
        print(p, datetime.fromtimestamp(m).isoformat())
    else:
        print(p, 'MISSING')
