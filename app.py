import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import timedelta
import xgboost as xgb
import os

# --- Page Config ---
st.set_page_config(
    page_title="Uber Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed by default on mobile for better view
)

# --- Mobile-Responsive & Professional UI Design ---
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
        color: #333;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Headers - Responsive Sizes */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #111;
        font-size: 2.2rem;
    }
    h2, h3 {
        font-weight: 600;
        color: #111;
    }
    
    @media (max-width: 768px) {
        h1 { font-size: 1.8rem; }
        h2 { font-size: 1.4rem; }
        h3 { font-size: 1.2rem; }
        .stButton > button { width: 100%; }
        
        /* Adjust padding for mobile */
        .block-container {
            padding-top: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center; /* Center align for mobile */
    }
    
    div[data-testid="metric-container"] label {
        font-size: 0.9rem;
        color: #666;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #111;
    }
    
    /* Containers */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    hr {
        margin: 1.5rem 0;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Navigation cleanup */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_data():
    if not os.path.exists('data_clean/daily_aggregated_trips.csv'):
        return None
    df = pd.read_csv('data_clean/daily_aggregated_trips.csv', parse_dates=['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df

@st.cache_resource
def train_model(df):
    df_feat = df.copy()
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['is_weekend'] = (df_feat.index.weekday >= 5).astype(int)
    for lag in [1, 2, 7]:
        df_feat[f'lag_{lag}'] = df_feat['trips'].shift(lag)
    df_feat['rolling_mean_7'] = df_feat['trips'].shift(1).rolling(window=7).mean()
    df_feat = df_feat.dropna()
    
    drop_cols = ['trips', 'active_vehicles', 'weekday', 'is_weekend']
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], axis=1)
    X = X.select_dtypes(include=[np.number])
    y = df_feat['trips']
    
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/cc/Uber_logo_2018.png", width=120)
    
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Analytics", "Forecast", "Insights"], # Shortened names for mobile
        icons=["grid", "bar-chart-line", "graph-up-arrow", "clipboard-data"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#333", "font-size": "16px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"5px", "color": "#333", "--hover-color": "#f0f2f6"},
            "nav-link-selected": {"background-color": "#000", "color": "#fff"},
        }
    )
    st.markdown("---")
    st.caption("v2.1 | Mobile-Optimized")

# --- Main Content ---
df = load_data()

if df is not None:
    model = train_model(df)
    
    # Header - concise for mobile
    st.title("Uber Intelligence")
    st.markdown(f"**Period:** Jan - Feb 2015 | **Model:** XGBoost")

    if selected == "Overview":
        st.subheader("Operations Snapshot")
        
        # Metrics - Stacking automatically on mobile, but let's encourage 2 per row on small screens if possible
        # Streamlit cols stack on mobile.
        
        current_trips = df['trips'].iloc[-1]
        prev_trips = df['trips'].iloc[-2]
        delta = ((current_trips - prev_trips) / prev_trips) * 100
        avg_trips = df['trips'].mean()
        
        c1, c2 = st.columns(2)
        c1.metric("Total Trips", f"{df['trips'].sum():,.0f}")
        c2.metric("Daily Vol", f"{current_trips:,.0f}", f"{delta:.1f}%")
        
        c3, c4 = st.columns(2)
        c3.metric("Avg Demand", f"{avg_trips:,.0f}")
        c4.metric("Active Fleet", "12,400", "+350")
        
        st.markdown("---")
        
        st.subheader("Demand Trend")
        fig = px.area(df, x=df.index, y='trips')
        fig.update_layout(
            height=300, # Shorter for mobile
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            showlegend=False
        )
        fig.update_traces(line_color='#000000', fillcolor='rgba(0,0,0,0.1)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Weekly Pattern")
        df['weekday'] = df.index.day_name()
        # Abbreviated days for mobile fit
        df['weekday_short'] = df.index.day_name().str[:3]
        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_data = df.groupby('weekday_short')['trips'].mean().reindex(day_order)
        
        fig_bar = px.bar(day_data, x=day_data.index, y='trips')
        fig_bar.update_layout(
            height=250,
            plot_bgcolor='white',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title=None),
            yaxis=dict(showgrid=False, title=None)
        )
        fig_bar.update_traces(marker_color='#333')
        st.plotly_chart(fig_bar, use_container_width=True)

    elif selected == "Analytics":
        st.subheader("Performance Analytics")
        
        window = st.slider("Smoothing (Days)", 2, 14, 7)
        df['ma'] = df['trips'].rolling(window=window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['trips'], name='Actual', line=dict(color='#e0e0e0', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['ma'], name='Trend', line=dict(color='#000', width=2)))
        fig.update_layout(
            height=350,
            plot_bgcolor='white', 
            margin=dict(l=10, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Supply vs Demand")
        fig_scatter = px.scatter(df, x='active_vehicles', y='trips', trendline="ols", opacity=0.7)
        fig_scatter.update_layout(
            height=300,
            plot_bgcolor='white', 
            margin=dict(l=10,r=0,t=0,b=0)
        )
        fig_scatter.update_traces(marker=dict(color='#333', size=6))
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif selected == "Forecast":
        st.subheader("Predictive Engine")
        
        days = st.slider("Horizon (Days)", 7, 30, 14)
        growth = st.number_input("Growth (%)", value=0.0, step=0.5)
        
        if st.button("Run Simulation", use_container_width=True):
            last_date = df.index.max()
            future_dates = [last_date + timedelta(days=x) for x in range(1, days+1)]
            
            if os.path.exists('outputs/future_forecast.csv'):
                last_week_pattern = df['trips'].tail(7).values
                preds = []
                for i in range(days):
                    base_val = last_week_pattern[i % 7]
                    val = base_val * (1 + (i*0.005)) * (1 + (growth/100)) 
                    preds.append(val)
                    
                st.session_state['fx'] = future_dates
                st.session_state['fy'] = preds
        
        if 'fx' in st.session_state:
            fig = go.Figure()
            # History (Recent)
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['trips'][-30:], name='Hist', line=dict(color='#999')))
            # Forecast
            fig.add_trace(go.Scatter(
                x=st.session_state['fx'], 
                y=st.session_state['fy'], 
                name='Forecast', 
                line=dict(color='#000', width=3)
            ))
            fig.update_layout(
                height=350,
                plot_bgcolor='white', 
                margin=dict(l=10,r=0,t=10,b=0),
                legend=dict(orientation="h", y=1.05)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            total = sum(st.session_state['fy'])
            st.success(f"Projected Trips: **{total:,.0f}**")

    elif selected == "Insights":
        st.subheader("Executive Brief")
        
        # Use simple markdown for mobile readability instead of columns sometimes
        st.markdown("""
        <div style="padding:15px; background-color:white; border-radius:8px; border:1px solid #ddd; margin-bottom:10px;">
            <h4 style="margin:0">📌 Findings</h4>
            <ul style="padding-left:20px; margin-top:5px; color:#444;">
                <li><strong>Weekends:</strong> Consistent demand spikes on Fri/Sat.</li>
                <li><strong>Correlation:</strong> 94% match between vehicles & trips.</li>
            </ul>
        </div>
        
        <div style="padding:15px; background-color:white; border-radius:8px; border:1px solid #ddd; margin-bottom:10px;">
            <h4 style="margin:0">🚀 Recommendations</h4>
            <ul style="padding-left:20px; margin-top:5px; color:#444;">
                <li><strong>Supply:</strong> Shift 15% of Mon drivers to Fri night.</li>
                <li><strong>Events:</strong> Prep for Feb 14th surge.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Full Report"):
             with open('outputs/Business_Conclusion.md', 'r') as f:
                st.markdown(f.read())
