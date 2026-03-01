import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("🛡️ Crime Severity Predictor (ANN)")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    # Severity bins: Low, Medium, High
    df['Severity'] = pd.cut(df['TOTAL IPC CRIMES'], bins=[0, 2000, 5000, np.inf], labels=[0, 1, 2])
    return df.dropna()

df = load_data()

# 2. Process Data
le = LabelEncoder()
df['STATE_NUM'] = le.fit_transform(df['STATE/UT'])
X = df[['STATE_NUM', 'YEAR']]
y = df['Severity'].astype(int)

# 3. Handle Unbalanced Data (Requirement E)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 4. ANN Model (Requirement C)
st.sidebar.header("Model Parameters")
epochs = st.sidebar.slider("Training Rounds", 5, 50, 10)

if st.button("🚀 Start Deep Learning Analysis"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # Building the ANN architecture
    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    with st.spinner('ANN is learning crime patterns...'):
        model.fit(X_scaled, y_res, epochs=epochs, verbose=0)
    
    # 5. Output for Managers (Requirement H)
    st.subheader("📊 Managerial Accuracy Matrix")
    y_pred = np.argmax(model.predict(X_scaled), axis=1)
    cm = confusion_matrix(y_res, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    st.pyplot(fig)
    st.success("Analysis Complete: Focus resources on predicted 'High' risk zones.")
