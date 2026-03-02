import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Page Configuration
st.set_page_config(page_title="National Crime Intelligence Hub", layout="wide")

# 1. DATA ENGINE
@st.cache_data
def load_ncrb_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_ncrb_data()
crime_types = ['MURDER', 'ATTEMPT TO MURDER', 'ROBBERY', 'BURGLARY', 'THEFT', 'RAPE', 'KIDNAPPING & ABDUCTION']

# --- SIDEBAR: MULTILEVEL FILTERS ---
st.sidebar.title("🔍 Strategic Filters")
target_crime = st.sidebar.selectbox("🎯 Select Primary Crime Category", ['TOTAL IPC CRIMES'] + crime_types)
selected_year = st.sidebar.slider("Historical Range", 2001, 2013, (2001, 2013))
selected_states = st.sidebar.multiselect("Focus States", sorted(df['STATE/UT'].unique()))

filtered_df = df[(df['YEAR'] >= selected_year[0]) & (df['YEAR'] <= selected_year[1])]
if selected_states:
    filtered_df = filtered_df[filtered_df['STATE/UT'].isin(selected_states)]

# --- MAIN DASHBOARD ---
st.title("🛡️ Indian Crime Intelligence & ANN Forecast")
st.markdown(f"### Analysis Target: **{target_crime}**")

# 1-4. KPI CARDS (4 Visual Elements)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(f"Total {target_crime}", f"{filtered_df[target_crime].sum():,}")
kpi2.metric("Annual Average", f"{int(filtered_df[target_crime].mean()):,}")
kpi3.metric("Highest Intensity State", filtered_df.groupby('STATE/UT')[target_crime].sum().idxmax())
kpi4.metric("Growth Momentum", f"{((df.groupby('YEAR')[target_crime].sum().pct_change().iloc[-1])*100):.2f}%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Crime Patterns", "🔮 AI Forecasting", "⚖️ State Comparison"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # 5. HEATMAP BAR CHART
        st.subheader("📍 Regional Intensity")
        fig_bar = px.bar(filtered_df.groupby('STATE/UT')[target_crime].sum().reset_index(), 
                         x='STATE/UT', y=target_crime, color=target_crime, 
                         color_continuous_scale='Reds', title="Crime Distribution by State")
        st.plotly_chart(fig_bar, use_container_width=True)

        # 6. BOX PLOT (Distribution)
        st.subheader("📦 Statistical Variance")
        fig_box = px.box(filtered_df, y=target_crime, points="all", title="Crime Volume Spread")
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        # 7. DONUT CHART (Contribution)
        st.subheader("🍩 Top 5 State Contribution")
        top_5 = filtered_df.groupby('STATE/UT')[target_crime].sum().nlargest(5).reset_index()
        fig_donut = px.pie(top_5, values=target_crime, names='STATE/UT', hole=0.5, title="Volume Share")
        st.plotly_chart(fig_donut, use_container_width=True)

        # 8. SCATTER PLOT (Correlation)
        st.subheader("🎯 Correlation: Murder vs Theft")
        fig_scatter = px.scatter(filtered_df, x="MURDER", y="THEFT", size="TOTAL IPC CRIMES", 
                                 hover_name="STATE/UT", log_x=True, size_max=60, title="Category Correlation")
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("🔮 Neural Network Future Projection (2026-2035)")
    target_year = st.slider("Select Forecast Horizon", 2026, 2035, 2030)
    
    if st.button("🚀 Execute ANN Forecast"):
        X = df[['YEAR']].values
        y = df[target_crime].values
        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_s, y_s = sc_X.fit_transform(X), sc_y.fit_transform(y.reshape(-1, 1))
        
        model = Sequential([Dense(64, activation='relu', input_shape=(1,)), Dense(32, activation='relu'), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_s, y_s, epochs=100, verbose=0)
        
        pred = sc_y.inverse_transform(model.predict(sc_X.transform([[target_year]])))[0][0]
        st.metric(f"AI Prediction for {target_year}", f"{int(pred):,}")

        # 9. FORECAST LINE CHART
        future_yrs = np.array(range(2001, 2036)).reshape(-1, 1)
        future_preds = sc_y.inverse_transform(model.predict(sc_X.transform(future_yrs)))
        forecast_df = pd.DataFrame({'Year': future_yrs.flatten(), 'Predicted Cases': future_preds.flatten()})
        fig_forecast = px.line(forecast_df, x='Year', y='Predicted Cases', title="Trendline Projection to 2035")
        fig_forecast.add_vrect(x0=2013, x1=2035, fillcolor="red", opacity=0.1, annotation_text="AI Predicted Zone")
        st.plotly_chart(fig_forecast, use_container_width=True)

with tab3:
    # 10. MULTI-STATE AREA CHART
    st.subheader("⚖️ Side-by-Side Comparison Engine")
    comp_states = st.multiselect("Choose States to Compare", sorted(df['STATE/UT'].unique()), default=sorted(df['STATE/UT'].unique())[:3])
    if len(comp_states) >= 2:
        fig_comp = px.area(df[df['STATE/UT'].isin(comp_states)], x='YEAR', y=target_crime, 
                           color='STATE/UT', title="Cumulative Volume Comparison")
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # 11. MOMENTUM BAR CHART (Growth Rates)
    st.subheader("📈 Yearly Growth Momentum (%)")
    growth_data = df.groupby('YEAR')[target_crime].sum().pct_change() * 100
    fig_growth = px.bar(growth_data, x=growth_data.index, y=target_crime, title="Historical Growth Velocity")
    st.plotly_chart(fig_growth, use_container_width=True)
