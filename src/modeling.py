from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .spectral_features import reflectance_columns, add_summary_features

def evaluate_models(df:pd.DataFrame):
    x=add_summary_features(df); cols=reflectance_columns(x)+['ndvi_proxy','red_edge_nd','green_red_ratio','nir_mean','vis_mean','red_edge_slope']
    X=x[cols]; y=x['class']; groups=x['plant_id']; cv=GroupKFold(5)
    rf=RandomForestClassifier(n_estimators=450,max_depth=14,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)
    svm=Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=.98,svd_solver='full')),('svm',SVC(C=4,kernel='rbf',gamma='scale'))])
    results=[]; preds={}
    for name,model in [('Random Forest',rf),('PCA + SVM',svm)]:
        pred=cross_val_predict(model,X,y,groups=groups,cv=cv); results.append({'model':name,'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro')}); preds[name]=pred
    rf.fit(X,y); imp=pd.DataFrame({'feature':cols,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    best=max(results,key=lambda d:d['macro_f1'])['model']; pred=preds[best]; labels=sorted(y.unique()); cm=confusion_matrix(y,pred,labels=labels)
    out=x[['plant_id','replicate','class','severity_demo']].copy(); out['predicted_class']=pred
    return pd.DataFrame(results),imp,out,cm,labels,cols

def pca_projection(df:pd.DataFrame):
    cols=reflectance_columns(df); X=StandardScaler().fit_transform(df[cols]); p=PCA(n_components=2,random_state=42); z=p.fit_transform(X)
    return pd.DataFrame({'PC1':z[:,0],'PC2':z[:,1],'class':df['class'].values,'plant_id':df['plant_id'].values}),p.explained_variance_ratio_

def healthy_anomaly(df:pd.DataFrame):
    cols=reflectance_columns(df); healthy=df['class'].eq('Healthy'); iso=IsolationForest(contamination=.10,random_state=42).fit(df.loc[healthy,cols]); score=-iso.score_samples(df[cols])
    out=df[['plant_id','replicate','class']].copy(); out['spectral_anomaly_score']=score; lo,hi=np.percentile(score,[5,95]); out['risk_score']=np.clip(100*(score-lo)/(hi-lo+1e-12),0,100).round(1); out['scouting_priority']=pd.cut(out.risk_score,[-1,35,60,100],labels=['Normal','Monitor','Inspect']).astype(str)
    return out.sort_values('risk_score',ascending=False)
