from __future__ import annotations
import pandas as pd
from .data_generation import WAVELENGTHS

def reflectance_columns(df): return [f'r_{w}' for w in WAVELENGTHS if f'r_{w}' in df.columns]

def add_summary_features(df:pd.DataFrame)->pd.DataFrame:
    x=df.copy(); r=lambda w:x[f'r_{w}']
    x['ndvi_proxy']=(r(800)-r(670))/(r(800)+r(670)+1e-12)
    x['red_edge_nd']=(r(800)-r(720))/(r(800)+r(720)+1e-12)
    x['green_red_ratio']=r(550)/(r(670)+1e-12)
    x['nir_mean']=x[[f'r_{w}' for w in range(760,901,10)]].mean(axis=1)
    x['vis_mean']=x[[f'r_{w}' for w in range(450,701,10)]].mean(axis=1)
    x['red_edge_slope']=(r(750)-r(700))/50
    return x
