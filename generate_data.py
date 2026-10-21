"""Regenerate the demo's synthetic operations dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "operations_data.csv"
rng = np.random.default_rng(42)
n = 1600
steps = ["Material Prep", "Assembly", "Quality Check", "Packaging", "Dispatch"]
machines = ["M-01", "M-02", "M-03", "M-04", "M-05", "M-06"]
products = ["Aster", "Boreal", "Cobalt", "Delta"]
shifts = ["Morning", "Evening", "Night"]
base = {"Material Prep":24,"Assembly":48,"Quality Check":18,"Packaging":22,"Dispatch":15}
defect = {"Material Prep":.025,"Assembly":.055,"Quality Check":.018,"Packaging":.03,"Dispatch":.012}
rows=[]
start=pd.Timestamp("2025-01-01")
for i in range(n):
    ts=start+pd.Timedelta(hours=int(rng.integers(0,24*270)))
    step=rng.choice(steps,p=[.18,.30,.18,.20,.14]); machine=rng.choice(machines)
    product=rng.choice(products); shift=rng.choice(shifts,p=[.4,.35,.25]); units=int(rng.integers(20,201))
    planned=base[step]*(.8+units/200*.5)*rng.uniform(.88,1.12)
    mf={"M-01":1.0,"M-02":.94,"M-03":1.08,"M-04":1.16,"M-05":.98,"M-06":1.04}[machine]
    sf={"Morning":.96,"Evening":1.0,"Night":1.12}[shift]
    downtime=max(0,rng.gamma(1.5,3)+(4 if machine=="M-04" else 0)+(3 if shift=="Night" else 0))
    actual=max(3,planned*mf*sf+downtime+rng.normal(0,3))
    dp=defect[step]+(.025 if machine=="M-04" else 0)+(.012 if shift=="Night" else 0)
    defects=int(rng.binomial(units,min(dp,.25)))
    energy=max(1,units*rng.uniform(.18,.42)+actual*rng.uniform(.12,.28)+downtime*1.5)
    rows.append({"record_id":f"OPS-{i+1:05d}","timestamp":ts.strftime("%Y-%m-%d %H:%M:%S"),"product":product,
    "process_step":step,"machine":machine,"shift":shift,"units_processed":units,"planned_minutes":round(planned,2),
    "actual_minutes":round(actual,2),"downtime_minutes":round(downtime,2),"defect_count":defects,"energy_kwh":round(energy,2)})
OUT.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(OUT,index=False)
print(f"Wrote {n} synthetic records to {OUT}")
