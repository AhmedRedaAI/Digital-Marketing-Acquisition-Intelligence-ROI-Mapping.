import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Property Valuation AI", layout="wide")

# Glassmorphism & Modern Prop-tech styling
st.markdown("""
    <style>
    .main { background-color: #0a0e17; }
    .sidebar .sidebar-content { background-color: #111827; }
    .valuation-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏠 Real Estate Hub & Automated Valuation Model (AVM)")

# Load Data
df = pd.read_csv('real_estate_data.csv')

# --- Sidebar Filters ---
st.sidebar.header("🎛️ Filter Properties")
max_area = int(df['Area'].max())
min_area = int(df['Area'].min())

selected_area = st.sidebar.slider("Minimum Area (sqm)", min_area, max_area, min_area)
selected_rooms = st.sidebar.multiselect("Number of Bedrooms", options=sorted(df['Rooms'].unique()), default=sorted(df['Rooms'].unique()))

# Filter Dataset
filtered_df = df[(df['Area'] >= selected_area) & (df['Rooms'].isin(selected_rooms))]

# Main Layout Split
col_chart, col_model = st.columns([3, 2])

with col_chart:
    st.subheader("🔍 Market Inventory Mapping")
    fig_scatter = px.scatter(filtered_df, x='Area', y='Price', size='Rooms', color='Price',
                             color_continuous_scale='Viridis', hover_data=['Rooms'], template='plotly_dark')
    fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_model:
    st.subheader("🔮 Instant AI Valuation")
    
    # Train Model
    X = df[['Area', 'Rooms']]
    y = df['Price']
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X, y)
    
    # Inputs
    input_area = st.number_input("Property Area (sqm):", min_value=30, max_value=500, value=120)
    input_rooms = st.slider("Total Bedrooms:", 1, 6, 3)
    
    # Predict
    pred_input = [[input_area, input_rooms]]
    predicted_price = model_rf.predict(pred_input)[0]
    
    # Visual Output
    st.markdown(f"""
    <div class="valuation-card">
        <span style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1.5px;">Estimated Market Value</span>
        <h1 style="color: #00f2fe; margin: 15px 0; font-size: 3.2rem; font-weight: 800;">${predicted_price:,.2f}</h1>
        <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Powered by Random Forest Regressor Model</p>
    </div>
    """, unsafe_allow_html=True)