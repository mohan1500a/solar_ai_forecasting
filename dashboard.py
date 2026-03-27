import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_mulivariate_lstm import LSTMModel, create_sequences
from preprocess import load_and_preprocess

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Solar AI Forecast",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# PREMIUM CSS STYLING
# ================================
st.markdown("""
<style>
/* Import sleek modern font */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphism containers */
.metric-container {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
    text-align: center;
    transition: all 0.4s ease;
}

.metric-container:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 40px 0 rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(0, 210, 255, 0.2);
}

.metric-value {
    font-size: 3rem;
    font-weight: 800;
    /* Vibrant dynamic gradient */
    background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
}

.metric-title {
    font-size: 1.1rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

.header-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFB75E 0%, #ED8F03 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 5px;
}

.sub-header {
    color: #CBD5E1;
    font-size: 1.2rem;
    margin-bottom: 40px;
}
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown('<div class="header-title">☀️ Solar Power Forecast Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced multivariate Long Short-Term Memory (LSTM) network predictions.</div>', unsafe_allow_html=True)

# ================================
# CACHED MODEL LOADING 
# ================================
from sklearn.preprocessing import MinMaxScaler

@st.cache_resource(show_spinner=False)
def load_data_and_model():
    # Load and preprocess the data exactly as in training
    df = pd.read_csv("solar_data.csv")
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"] + pd.Timedelta(hours=5)
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear
    
    feature_columns = [
        "temperature_2m (°C)",
        "shortwave_radiation (W/m²)",
        "Cell_Temp (°C)",
        "hour",
        "day_of_year",
        "Solar_Power (kW)"
    ]
    
    data = df[feature_columns].values
    target = df["Solar_Power (kW)"].values.reshape(-1, 1)
    
    # Scale exactly like training (only fitting on train_size avoids test leakage, though for dashboard we just fit to 80%) 
    train_size = int(0.8 * len(data))
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(data[:train_size])
    scaler_y.fit(target[:train_size])
    
    X_scaled = scaler_X.transform(data)
    y_scaled = scaler_y.transform(target)
    
    seq_length = 24
    X, y = create_sequences(X_scaled, y_scaled, seq_length)
    
    # Convert sequence to PyTorch Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Load optimized neural network
    input_size = X_tensor.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=128)
    model.load_state_dict(torch.load("solar_lstm_v3.pth", weights_only=True))
    model.eval()
    
    # Run historical prediction block
    with torch.no_grad():
        preds = model(X_tensor).numpy()
        
    predictions = scaler_y.inverse_transform(preds)
    actual = scaler_y.inverse_transform(y.reshape(-1, 1))
    
    # --------------------------------
    # AUTOREGRESSIVE 24H FORECAST
    # --------------------------------
    steps = 24
    last_time = df["time"].iloc[-1]
    future_times = pd.date_range(start=last_time, periods=steps+1, freq="h")[1:]
    
    future_df = pd.DataFrame({"time": future_times})
    future_df["hour"] = future_df["time"].dt.hour
    future_df["day_of_year"] = future_df["time"].dt.dayofyear
    future_df["temperature_2m (°C)"] = df["temperature_2m (°C)"].iloc[-1]
    future_df["Cell_Temp (°C)"] = df["Cell_Temp (°C)"].iloc[-1]
    
    # Day/Night curve mapping
    future_df["shortwave_radiation (W/m²)"] = future_df["hour"].apply(
        lambda h: 800 * np.sin(np.pi * (h - 6) / 13) if 6 <= h <= 18 else 0.0
    )
    future_df["Solar_Power (kW)"] = 0.0 
    
    future_predictions = []
    current_seq = X[-1].copy() 
    
    for i in range(steps):
        seq_tensor = torch.tensor(current_seq).unsqueeze(0).float()
        with torch.no_grad():
            power_pred = model(seq_tensor).numpy()[0][0]
            
        # We let the model predict raw values directly without constraints
            
        future_predictions.append(power_pred)
        
        # Inverse transform the power specifically before doing a master Row Scaler transformation
        raw_power_pred = scaler_y.inverse_transform([[power_pred]])[0][0]
        
        next_raw_row = [
            future_df["temperature_2m (°C)"].iloc[i],
            future_df["shortwave_radiation (W/m²)"].iloc[i],
            future_df["Cell_Temp (°C)"].iloc[i],
            future_df["hour"].iloc[i],
            future_df["day_of_year"].iloc[i],
            raw_power_pred
        ]
        
        next_scaled_row = scaler_X.transform([next_raw_row])[0]
        current_seq = np.vstack((current_seq[1:], next_scaled_row))
        
    future_predictions = np.array(future_predictions).reshape(-1,1)
    future_predictions = scaler_y.inverse_transform(future_predictions).flatten()
    
    return actual.flatten(), predictions.flatten(), future_times, future_predictions

# Apply loading spinner 
with st.spinner("Initializing Deep Learning Engine..."):
    actual, predictions, future_times, future_predictions = load_data_and_model()

# Evaluate Metrics globally 
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
total_future_energy = np.sum(future_predictions)

# ================================
# PERFORMANCE METRICS ROW
# ================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-container"><div class="metric-title">Mean Absolute Error</div><div class="metric-value">{mae:.3f} kW</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-container"><div class="metric-title">Root Mean Squared</div><div class="metric-value">{rmse:.3f} kW</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-container"><div class="metric-title">Pred. Energy (24h)</div><div class="metric-value" style="color:#00D2FF;">{total_future_energy:.2f} kWh</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-container"><div class="metric-title">Data Lookback</div><div class="metric-value">24 Hours</div></div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ================================
# INTERACTIVE DATA VISUALIZATION
# ================================
# Create slider to select how much data to show to improve browser performance
points_to_show = st.slider("Select Timeline Window (Time Steps)", min_value=100, max_value=2000, value=750, step=50)

fig = go.Figure()

# Plot Ground Truth (Actual Power)
fig.add_trace(go.Scatter(
    y=actual[:points_to_show],
    name="Ground Truth (Actual)",
    line=dict(color="#00D2FF", width=2),
    fill='tozeroy',
    fillcolor='rgba(0, 210, 255, 0.1)',
    mode='lines'
))

# Plot Predictions
fig.add_trace(go.Scatter(
    y=predictions[:points_to_show],
    name="AI Prediction",
    line=dict(color="#FFB75E", width=2, dash='dot'),
    mode='lines'
))

# Polish Plotly aesthetics
fig.update_layout(
    title="Real-Time Generation vs AI Forecast Pipeline",
    title_font=dict(size=20, family="Outfit"),
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=50, b=0),
    xaxis=dict(showgrid=False, title="Time Step Timeline", title_font=dict(size=14)),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Power Output (kW)"),
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="right", 
        x=1,
        font=dict(size=14)
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><br><div class='sub-header'>Next 24 Hours Generation Prediction</div>", unsafe_allow_html=True)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=future_times,
    y=future_predictions,
    name="Future Autoregressive AI Forecast",
    line=dict(color="#ED8F03", width=3, shape='spline'),
    fill='tozeroy',
    fillcolor='rgba(237, 143, 3, 0.2)',
    mode='lines+markers'
))

fig2.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(showgrid=False, title="Time (Next 24 Hours)", tickformat="%H:%M"),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Power Output (kW)"),
)

st.plotly_chart(fig2, use_container_width=True)
