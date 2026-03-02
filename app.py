import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Page Configuration
st.set_page_config(page_title="NCRB Crime Intelligence Hub", layout="wide")

# 1. DATA ENGINE
@st.cache_data
def load_ncrb_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_ncrb_data()
crime_columns = ['MURDER', 'ATTEMPT TO MURDER', 'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER', 
                 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE', 'KIDNAPPING & ABDUCTION', 
                 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS', 'KIDNAPPING AND ABDUCTION OF OTHERS', 
                 'DACOITY', 'PREPARATION AND ASSEMBLY FOR DACOITY', 'ROBBERY', 'BURGLARY', 'THEFT']

# --- SIDEBAR FILTERS ---
st.sidebar.title("🔍 Dashboard Controls")
selected_year = st.sidebar.slider("Historical Data Range", 2001, 2013, (2001, 2013))
selected_states = st.sidebar.multiselect("Filter by State/UT", sorted(df['STATE/UT'].unique()))

filtered_df = df[(df['YEAR'] >= selected_year[0]) & (df['YEAR'] <= selected_year[1])]
if selected_states:
    filtered_df = filtered_df[filtered_df['STATE/UT'].isin(selected_states)]

# --- I. KPI CARDS ---
st.title("🛡️ Advanced Crime Analytics & AI Forecast")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total IPC Cases", f"{filtered_df['TOTAL IPC CRIMES'].sum():,}")
k2.metric("Avg Crime Rate", f"{int(filtered_df['TOTAL IPC CRIMES'].mean()):,}")
k3.metric("Top Crime State", filtered_df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().idxmax())
k4.metric("States Analyzed", len(filtered_df['STATE/UT'].unique()))

st.markdown("---")

# --- II. NEW CHARTS: CORRELATION & DISTRIBUTION ---
tab1, tab2, tab3 = st.tabs(["📊 Crime Patterns", "🔮 AI Forecasting", "⚖️ State Comparison"])

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🔗 Crime Correlation Heatmap")
        st.write("Shows how different crimes move together (e.g., Burglary vs Theft).")
        corr = filtered_df[crime_columns].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_b:
        st.subheader("📦 Crime Volume Distribution (Box Plot)")
        st.write("Identifies statistical outliers and median crime counts across states.")
        # Selecting a few key crimes for the boxplot to keep it clean
        box_data = filtered_df[['MURDER', 'ROBBERY', 'BURGLARY', 'THEFT']]
        fig_box = px.box(box_data, points="all", title="Spread of Major Crimes")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader("📈 Yearly Growth Rate (%)")
        growth = df.groupby('YEAR')['TOTAL IPC CRIMES'].sum().pct_change() * 100
        fig_growth = px.bar(growth, x=growth.index, y='TOTAL IPC CRIMES', 
                            title="Year-over-Year % Change", labels={'TOTAL IPC CRIMES':'% Change'})
        st.plotly_chart(fig_growth, use_container_width=True)

    with col_d:
        st.subheader("🍩 IPC vs Other Crimes Ratio")
        total_vals = filtered_df[crime_columns].sum().sort_values(ascending=False).head(5)
        fig_pie = px.pie(values=total_vals.values, names=total_vals.index, hole=0.4, title="Top 5 Crime Categories")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("🔮 Neural Network Future Projection (2027 - 2035)")
    target_year = st.slider("Forecast Target Year", 2027, 2035, 2030)
    
    if st.button("🚀 Execute ANN Forecast"):
        X = df[['YEAR']].values
        y = df['TOTAL IPC CRIMES'].values
        
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_s, y_s = scaler_X.fit_transform(X), scaler_y.fit_transform(y.reshape(-1, 1))
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(1,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        with st.spinner('AI is simulating historical crime cycles...'):
            model.fit(X_s, y_s, epochs=150, verbose=0)
            
        pred = scaler_y.inverse_transform(model.predict(scaler_X.transform([[target_year]])))[0][0]
        st.metric(f"AI Predicted Total IPC for {target_year}", f"{int(pred):,}")
        
        # Trend Visualization
        future_yrs = np.array(range(2001, 2036)).reshape(-1, 1)
        future_preds = scaler_y.inverse_transform(model.predict(scaler_X.transform(future_yrs)))
        forecast_chart = pd.DataFrame({'Year': future_yrs.flatten(), 'Cases': future_preds.flatten()})
        fig_f = px.line(forecast_chart, x='Year', y='Cases', title="AI Predicted Trendline to 2035")
        fig_f.add_vrect(x0=2013, x1=2035, fillcolor="red", opacity=0.1, annotation_text="AI Forecast Zone")
        st.plotly_chart(fig_f, use_container_width=True)

with tab3:
    st.subheader("⚖️ State Comparison Engine")
    states = st.multiselect("Pick States", sorted(df['STATE/UT'].unique()), default=sorted(df['STATE/UT'].unique())[:3])
    if states:
        fig_comp = px.area(df[df['STATE/UT'].isin(states)], x="YEAR", y="TOTAL IPC CRIMES", color="STATE/UT",
                           title="Cumulative Crime Comparison")
        st.plotly_chart(fig_comp, use_container_width=True)
