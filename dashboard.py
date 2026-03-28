import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from core import TimeSeriesModel, prepare_data, create_sequences

st.set_page_config(page_title="Solar AI Forecast", page_icon="☀️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.metric-container { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(15px); border-radius: 20px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.05); text-align: center; transition: all 0.4s ease; min-height: 160px; display: flex; flex-direction: column; justify-content: center; align-items: center; }
.metric-container:hover { transform: translateY(-8px); border: 1px solid rgba(0, 210, 255, 0.2); }
.metric-value { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%); -webkit-background-clip: text; -webkit-fill-color: transparent; }
.metric-title { font-size: 1.1rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }
.header-title { font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #FFB75E 0%, #ED8F03 100%); -webkit-background-clip: text; -webkit-fill-color: transparent; }
.sub-header { color: #CBD5E1; font-size: 1.2rem; margin-bottom: 40px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">☀️ Solar Power Forecast Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced multivariate LSTM network predictions.</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_all():
    df, data, target = prepare_data()
    sz = int(0.8 * len(data))
    sx, sy = MinMaxScaler(), MinMaxScaler()
    sx.fit(data[:sz])
    sy.fit(target[:sz])
    xs, ys = create_sequences(sx.transform(data), sy.transform(target), 24)
    model = TimeSeriesModel(xs.shape[2])
    model.load_weights("solar_lstm_best.pth")
    model.eval()
    with torch.no_grad():
        preds = sy.inverse_transform(model(torch.tensor(xs, dtype=torch.float32)).numpy())
        actual = sy.inverse_transform(ys.reshape(-1, 1))
    
    stps = 24
    last_t = df["time"].iloc[-1]
    ft_t = pd.date_range(start=pd.Timestamp(f"{last_t.date()} 08:00:00"), periods=stps, freq="h")
    ft_df = pd.DataFrame({"time": ft_t, "hour": ft_t.hour, "day_of_year": ft_t.dayofyear})
    ft_df["temp"], ft_df["cell"], ft_df["pres"] = df["temperature_2m (°C)"].iloc[-1], df["Cell_Temp (°C)"].iloc[-1], df["pressure_msl"].iloc[-1]
    ft_df["rad"] = ft_df["hour"].apply(lambda h: 800 * np.sin(np.pi * (h - 6) / 13) if 6 <= h <= 18 else 0.0)
    
    ft_ps, curr = [], xs[-1].copy()
    for i in range(stps):
        with torch.no_grad():
            p = model(torch.tensor(curr).unsqueeze(0).float()).numpy()[0][0]
        ft_ps.append(p)
        raw_p = sy.inverse_transform([[p]])[0][0]
        row = [ft_df["temp"].iloc[i], ft_df["rad"].iloc[i], ft_df["cell"].iloc[i], ft_df["pres"].iloc[i], ft_df["hour"].iloc[i], ft_df["day_of_year"].iloc[i], raw_p]
        curr = np.vstack((curr[1:], sx.transform([row])[0]))
    return actual.flatten(), preds.flatten(), ft_t, sy.inverse_transform(np.array(ft_ps).reshape(-1, 1)).flatten()

with st.spinner("Initializing..."):
    act, pre, ft_t, ft_ps = load_all()

m, r = mean_absolute_error(act, pre), np.sqrt(mean_squared_error(act, pre))
r2 = r2_score(act, pre)
mask = act > 0.01
mape = mean_absolute_percentage_error(act[mask], pre[mask]) if mask.sum() > 0 else 0.0

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.markdown(f'<div class="metric-container"><div class="metric-title">MAE</div><div class="metric-value">{m:.3f}</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-container"><div class="metric-title">MAPE</div><div class="metric-value" style="color:#10b981;">{mape:.1%}</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-container"><div class="metric-title">R²</div><div class="metric-value">{r2:.3f}</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="metric-container"><div class="metric-title">24h Gen</div><div class="metric-value" style="color:#00D2FF;">{np.sum(ft_ps):.1f} kWh</div></div>', unsafe_allow_html=True)
col5.markdown(f'<div class="metric-container"><div class="metric-title">Peak Time</div><div class="metric-value" style="font-size:1.2rem;">{ft_t[np.argmax(ft_ps)].strftime("%H:%M")}</div></div>', unsafe_allow_html=True)
col6.markdown(f'<div class="metric-container"><div class="metric-title">Peak Power</div><div class="metric-value" style="color:#FFB75E;">{np.max(ft_ps):.2f} kW</div></div>', unsafe_allow_html=True)

pts = st.slider("Timeline window", 100, 2000, 750, 50)
fig = go.Figure()
fig.add_trace(go.Scatter(y=act[:pts], name="Actual", line=dict(color="#00D2FF", width=2), fill='tozeroy', fillcolor='rgba(0, 210, 255, 0.1)'))
fig.add_trace(go.Scatter(y=pre[:pts], name="AI Prediction", line=dict(color="#FFB75E", width=2, dash='dot')))
fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=50, b=0))
st.plotly_chart(fig, width='stretch')

st.markdown(f"<div class='sub-header'>Next 24h Forecast</div>", unsafe_allow_html=True)
fig2 = go.Figure(go.Scatter(x=ft_t, y=ft_ps, name="AI Forecast", line=dict(color="#ED8F03", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(237, 143, 3, 0.2)'))
fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig2, width='stretch')
