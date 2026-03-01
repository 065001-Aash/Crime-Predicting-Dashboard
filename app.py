import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="Crime Intelligence Hub", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR BETTER LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 1. DATA ENGINE
@st.cache_data
def get_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    return df.dropna()

df = get_data()

# 2. SIDEBAR - ADVANCED CONTROLS
st.sidebar.title("🛡️ Admin Controls")
st.sidebar.subheader("Threshold Settings")
# Interactive Threshold: Let the manager decide what 'High Crime' means
high_threshold = st.sidebar.slider("High Crime Limit (Total IPC)", 2000, 10000, 5000)

st.sidebar.subheader("ANN Configuration")
train_epochs = st.sidebar.select_slider("AI Training Depth", options=[10, 50, 100, 150], value=100)
use_smote = st.sidebar.toggle("Balance Data (SMOTE)", value=True)

# Apply dynamic labels based on user slider
df['Severity'] = pd.cut(df['TOTAL IPC CRIMES'], bins=[0, 2000, high_threshold, np.inf], labels=['Low', 'Medium', 'High'])

# 3. DASHBOARD TABS
tab1, tab2, tab3 = st.tabs(["🔮 Future AI Forecast", "📊 Comparative Analysis", "🗂️ Data Inspector"])

with tab1:
    st.header("Predictive Intelligence")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        target_state = st.selectbox("Select Target State", sorted(df['STATE/UT'].unique()))
        target_year = st.slider("Forecast Horizon", 2014, 2035, 2026)
        
        if st.button("🚀 Run Neural Forecast"):
            # Prepare ANN
            le_s = LabelEncoder()
            df['S_ID'] = le_s.fit_transform(df['STATE/UT'])
            le_v = LabelEncoder()
            df['V_ID'] = le_v.fit_transform(df['Severity'])
            
            X = df[['S_ID', 'YEAR']]
            y = df['V_ID']
            
            if use_smote:
                X, y = SMOTE(random_state=42).fit_resample(X, y)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = Sequential([
                Dense(64, activation='relu', input_shape=(2,)),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_scaled, y, epochs=train_epochs, verbose=0)
            
            # Prediction
            s_code = le_s.transform([target_state])[0]
            query = scaler.transform([[s_code, target_year]])
            p_idx = np.argmax(model.predict(query), axis=1)
            result = le_v.inverse_transform(p_idx)[0]
            
            st.metric("Predicted Severity", result)
            
    with col2:
        if 'result' in locals():
            st.subheader(f"Trend Analysis: {target_state}")
            st.line_chart(df[df['STATE/UT'] == target_state], x='YEAR', y='TOTAL IPC CRIMES')
            st.caption("Historical data (2001-2013) used to train the Neural Network.")

with tab2:
    st.header("State Comparison Engine")
    c1, c2 = st.columns(2)
    with c1:
        s1 = st.selectbox("State A", sorted(df['STATE/UT'].unique()), index=0)
    with c2:
        s2 = st.selectbox("State B", sorted(df['STATE/UT'].unique()), index=1)
    
    comp_df = df[df['STATE/UT'].isin([s1, s2])]
    st.bar_chart(comp_df, x='YEAR', y='TOTAL IPC CRIMES', color='STATE/UT')

with tab3:
    st.header("Raw Data Access")
    st.dataframe(df, use_container_width=True)
