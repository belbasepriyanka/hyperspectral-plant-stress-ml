from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='AI Disease Scouting',layout='wide')
st.title('AI Dragon Fruit Disease Scouting & Risk Dashboard')
st.caption('Synthetic hyperspectral demonstration only — not a field diagnosis.')
risk=pd.read_csv(ROOT/'results'/'scouting_risk_scores.csv'); pred=pd.read_csv(ROOT/'results'/'classification_predictions.csv'); df=risk.merge(pred[['plant_id','replicate','predicted_class']],on=['plant_id','replicate'])
priority=st.selectbox('Priority',['All','Inspect','Monitor','Normal']); view=df if priority=='All' else df[df.scouting_priority==priority]
c1,c2,c3=st.columns(3); c1.metric('Scans',len(view)); c2.metric('Mean risk',f"{view.risk_score.mean():.1f}"); c3.metric('Inspect',int((view.scouting_priority=='Inspect').sum()))
st.dataframe(view.sort_values('risk_score',ascending=False),use_container_width=True); st.bar_chart(view.groupby('predicted_class').size())
