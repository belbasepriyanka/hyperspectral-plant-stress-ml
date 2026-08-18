from src.data_generation import generate_demo
from src.modeling import evaluate_models,pca_projection,healthy_anomaly

def test_demo_shape():
    df=generate_demo(); assert len(df)==360; assert df['plant_id'].nunique()==120

def test_models():
    df=generate_demo(); m,_,_,_,_,_=evaluate_models(df); assert m.accuracy.between(0,1).all(); p,var=pca_projection(df); assert len(p)==len(df); r=healthy_anomaly(df); assert r.risk_score.between(0,100).all()
