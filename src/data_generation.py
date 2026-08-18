from __future__ import annotations
import numpy as np
import pandas as pd

CLASSES=['Healthy','Anthracnose','Neoscytalidium','Bacterial soft rot','Rust-like stress']
WAVELENGTHS=np.arange(400,1001,10)

def _base_curve(w):
    red_abs=np.exp(-((w-675)/38)**2)
    green_bump=np.exp(-((w-550)/48)**2)
    red_edge=1/(1+np.exp(-(w-720)/18))
    return 0.08+0.07*green_bump-0.06*red_abs+0.42*red_edge

def generate_demo(seed:int=7, plants_per_class:int=24, reps:int=3):
    """Synthetic hyperspectral signatures for disease-scanning portfolio demo."""
    rng=np.random.default_rng(seed); rows=[]
    for cls in CLASSES:
        for p in range(plants_per_class):
            plant_id=f"{cls[:3].upper()}-{p+1:03d}"; severity=0 if cls=='Healthy' else rng.uniform(.25,.95)
            for r in range(reps):
                curve=_base_curve(WAVELENGTHS).copy()
                if cls=='Anthracnose': curve += severity*(0.045*np.exp(-((WAVELENGTHS-540)/55)**2)-0.08*np.exp(-((WAVELENGTHS-730)/60)**2))
                elif cls=='Neoscytalidium': curve += severity*(0.055*np.exp(-((WAVELENGTHS-610)/65)**2)-0.06*np.exp(-((WAVELENGTHS-800)/90)**2))
                elif cls=='Bacterial soft rot': curve += severity*(0.07*np.exp(-((WAVELENGTHS-500)/70)**2)-0.10*np.exp(-((WAVELENGTHS-900)/80)**2))
                elif cls=='Rust-like stress': curve += severity*(0.085*np.exp(-((WAVELENGTHS-585)/45)**2)-0.05*np.exp(-((WAVELENGTHS-760)/70)**2))
                curve=np.clip(curve+rng.normal(0,.008,len(curve)),.01,.85)
                rec={'plant_id':plant_id,'replicate':r+1,'class':cls,'severity_demo':round(float(severity),3)}
                rec.update({f'r_{w}':round(float(v),5) for w,v in zip(WAVELENGTHS,curve)}); rows.append(rec)
    return pd.DataFrame(rows)
