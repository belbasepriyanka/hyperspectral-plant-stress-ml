from pathlib import Path
import sys, pandas as pd, numpy as np, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_generation import generate_demo,WAVELENGTHS
from src.modeling import evaluate_models,pca_projection,healthy_anomaly
for d in [ROOT/'data',ROOT/'results',ROOT/'figures']: d.mkdir(exist_ok=True)
df=generate_demo(); df.to_csv(ROOT/'data'/'sample_hyperspectral_disease_demo.csv',index=False)
metrics,imp,pred,cm,labels,cols=evaluate_models(df); pca,var=pca_projection(df); risk=healthy_anomaly(df)
metrics.to_csv(ROOT/'results'/'model_comparison.csv',index=False); imp.to_csv(ROOT/'results'/'rf_feature_importance.csv',index=False); pred.to_csv(ROOT/'results'/'classification_predictions.csv',index=False); risk.to_csv(ROOT/'results'/'scouting_risk_scores.csv',index=False); pd.DataFrame({'component':['PC1','PC2'],'explained_variance':var}).to_csv(ROOT/'results'/'pca_variance.csv',index=False)
fig,ax=plt.subplots(figsize=(9,5))
for cls,g in df.groupby('class'): ax.plot(WAVELENGTHS,g[[f'r_{w}' for w in WAVELENGTHS]].mean().values,label=cls)
ax.set(xlabel='Wavelength (nm)',ylabel='Reflectance',title='Synthetic mean spectral signatures'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/'figures'/'spectral_signatures.svg'); plt.close(fig)
print(metrics.to_string(index=False))
