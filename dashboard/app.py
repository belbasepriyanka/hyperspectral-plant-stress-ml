from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_demo
from src.modeling import evaluate_models, healthy_anomaly

st.set_page_config(page_title='AI Disease Scouting', layout='wide')
st.title('AI Dragon Fruit Disease Scouting & Risk Dashboard')
st.caption('Synthetic hyperspectral demonstration only — not a field diagnosis.')

@st.cache_data
def build_demo():
    spectra = generate_demo()
    metrics, _, predictions, _, _, _ = evaluate_models(spectra)
    risk = healthy_anomaly(spectra)
    view = risk.merge(
        predictions[['plant_id', 'replicate', 'predicted_class']],
        on=['plant_id', 'replicate'],
    )
    return metrics, view

metrics, df = build_demo()
priority = st.selectbox('Scouting priority', ['All', 'Inspect', 'Monitor', 'Normal'])
view = df if priority == 'All' else df[df.scouting_priority == priority]

c1, c2, c3 = st.columns(3)
c1.metric('Scans', len(view))
c2.metric('Mean risk', f"{view.risk_score.mean():.1f}" if len(view) else '—')
c3.metric('Inspect', int((view.scouting_priority == 'Inspect').sum()))

st.subheader('Model comparison')
st.dataframe(metrics, use_container_width=True)
st.subheader('Scouting queue')
st.dataframe(view.sort_values('risk_score', ascending=False), use_container_width=True)
if len(view):
    st.bar_chart(view.groupby('predicted_class').size())
