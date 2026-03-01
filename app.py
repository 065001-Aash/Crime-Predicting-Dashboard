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

# Page Setup
st.set_page_config(page_title="Crime AI Intelligence", layout="wide")
st.title("🛡️ Indian Crime Forecasting & ANN Analysis")
st.markdown("---")

# 1. LOAD DATA
@st.cache_data
def get_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    # Clean column names
    df.columns = df.columns.str.strip()
    # Create Severity Levels
    df['Severity'] = pd.cut(df['TOTAL IPC CRIMES'], bins=[0, 2000, 5000, np.inf], labels=['Low', 'Medium', 'High'])
    return df.dropna()

try:
    df = get_data()

    # 2. INTERACTIVE SIDEBAR
    st.sidebar.header("🛠️ Model Configuration")
    forecast_horizon = st.sidebar.slider("Forecast Year", 2014, 2035, 2026)
    training_rounds = st.sidebar.select_slider("AI Training Intensity", options=[20, 50, 100], value=50)

    # 3. TABS FOR INTERACTIVITY
    tab1, tab2 = st.tabs(["🔮 Future Prediction", "📊 Comparison Engine"])

    with tab1:
        st.subheader("Deep Learning Forecast")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            state_select = st.selectbox("Select State/UT", sorted(df['STATE/UT'].unique()))
            if st.button("🚀 Generate AI Prediction"):
                # Data Prep
                le_state = LabelEncoder()
                df['S_ID'] = le_state.fit_transform(df['STATE/UT'])
                le_sev = LabelEncoder()
                df['V_ID'] = le_sev.fit_transform(df['Severity'])
                
                X = df[['S_ID', 'YEAR']]
                y = df['V_ID']
                
                # SAFE SMOTE: Only runs if enough data exists to avoid the ValueError
                if len(np.unique(y)) > 1:
                    sm = SMOTE(random_state=42, k_neighbors=1) # Reduced neighbors to prevent crash
                    X_res, y_res = sm.fit_resample(X, y)
                else:
                    X_res, y_res = X, y

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_res)
                
                # ANN Model
                model = Sequential([
                    Dense(32, activation='relu', input_shape=(2,)),
                    Dense(16, activation='relu'),
                    Dense(3, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                with st.spinner('Neural Network Training...'):
                    model.fit(X_scaled, y_res, epochs=training_rounds, verbose=0)
                
                # Predict
                s_idx = le_state.transform([state_select])[0]
                query = scaler.transform([[s_idx, forecast_horizon]])
                p_idx = np.argmax(model.predict(query), axis=1)
                final_res = le_sev.inverse_transform(p_idx)[0]
                
                st.metric(f"Risk for {forecast_horizon}", final_res)
                if final_res == 'High':
                    st.error("Action Recommended: Strategic Patrol Increase.")
                else:
                    st.success("Stable Pattern Detected.")

        with col2:
            st.subheader("Historical Context")
            state_data = df[df['STATE/UT'] == state_select]
            st.line_chart(state_data, x='YEAR', y='TOTAL IPC CRIMES')

    with tab2:
        st.subheader("State-by-State Comparison")
        comp_states = st.multiselect("Select States to Compare", sorted(df['STATE/UT'].unique()), default=sorted(df['STATE/UT'].unique())[:2])
        if comp_states:
            comp_df = df[df['STATE/UT'].isin(comp_states)]
            st.bar_chart(comp_df, x='YEAR', y='TOTAL IPC CRIMES', color='STATE/UT')

except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.info("Check if 'Crimes_in_india_2001-2013.csv' is correctly uploaded to GitHub.")
