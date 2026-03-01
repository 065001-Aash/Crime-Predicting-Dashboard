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

# Page Config for a Professional Look
st.set_page_config(page_title="Crime Intelligence AI", layout="wide")
st.title("🛡️ Police Resource Intelligence Dashboard")
st.markdown("---")

# 1. LOAD & CLEAN
@st.cache_data
def get_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    df['Severity'] = pd.cut(df['TOTAL IPC CRIMES'], bins=[0, 2000, 5000, np.inf], labels=['Low', 'Medium', 'High'])
    return df.dropna()

df = get_data()

# 2. PREPROCESSING
le_state = LabelEncoder()
df['STATE_ID'] = le_state.fit_transform(df['STATE/UT'])
le_sev = LabelEncoder()
df['SEV_ID'] = le_sev.fit_transform(df['Severity'])

# 3. SIDEBAR CONTROLS
st.sidebar.header("🕹️ AI Control Panel")
train_rounds = st.sidebar.slider("Training Intensity (Epochs)", 10, 100, 50)
balance_data = st.sidebar.toggle("Enable SMOTE Balancing", value=True)

# 4. THE PREDICTOR TOOL (Managerial Requirement)
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Predict Risk Level")
    input_state = st.selectbox("Select State/UT", sorted(df['STATE/UT'].unique()))
    input_year = st.number_input("Enter Year", 2014, 2025, 2024)
    
    if st.button("Generate AI Forecast"):
        # Setup ANN
        X = df[['STATE_ID', 'YEAR']]
        y = df['SEV_ID']
        
        if balance_data:
            X, y = SMOTE().fit_resample(X, y)
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = Sequential([
            Dense(32, activation='relu', input_shape=(2,)),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_scaled, y, epochs=train_rounds, verbose=0)
        
        # Make Prediction
        query = scaler.transform([[le_state.transform([input_state])[0], input_year]])
        res = np.argmax(model.predict(query), axis=1)
        final_label = le_sev.inverse_transform(res)[0]
        
        st.metric(label="Predicted Crime Risk", value=final_label)
        if final_label == 'High':
            st.error("⚠️ ACTION REQUIRED: Deploy additional units to this sector.")
        else:
            st.success("✅ STABLE: Current patrol levels are sufficient.")

with col2:
    st.subheader("📈 Historical Trends")
    state_df = df[df['STATE/UT'] == input_state]
    st.line_chart(state_df, x='YEAR', y='TOTAL IPC CRIMES')
