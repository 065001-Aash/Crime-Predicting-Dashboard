import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Page Config
st.set_page_config(page_title="NCRB Crime Intelligence Hub", layout="wide")
st.title("🛡️ Indian Crime Analytics & ANN Prediction")

# 1. DATA ENGINE
@st.cache_data
def load_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()
crime_types = ['MURDER', 'ATTEMPT TO MURDER', 'ROBBERY', 'BURGLARY', 'THEFT', 'KIDNAPPING & ABDUCTION', 'RAPE']

# 2. SIDEBAR FILTERS
st.sidebar.title("🔍 Global Filters")
target_crime = st.sidebar.selectbox("🎯 Select Primary Crime Category", crime_types)
year_range = st.sidebar.slider("Timeline Range", 2001, 2013, (2001, 2013))
focus_states = st.sidebar.multiselect("Select States", sorted(df['STATE/UT'].unique()))

# Filter Logic
f_df = df[(df['YEAR'] >= year_range[0]) & (df['YEAR'] <= year_range[1])]
if focus_states:
    f_df = f_df[f_df['STATE/UT'].isin(focus_states)]

# --- DASHBOARD TABS (Grouping 15+ Visuals) ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 National Overview", "📈 Trends & Momentum", "📉 Statistical Deep-Dive", "🔮 AI ANN Forecast"])

with tab1:
    st.header("National Crime Metrics")
    c1, c2, c3, c4 = st.columns(4)
    # 1. KPI Metric Cards
    c1.metric("Total IPC Cases", f"{f_df['TOTAL IPC CRIMES'].sum():,}")
    c2.metric("Target Crime Total", f"{f_df[target_crime].sum():,}")
    c3.metric("Peak Year", f_df.groupby('YEAR')['TOTAL IPC CRIMES'].sum().idxmax())
    c4.metric("States Active", len(f_df['STATE/UT'].unique()))

    col1, col2 = st.columns(2)
    with col1:
        # 2. Bar Chart: Total IPC by State
        st.plotly_chart(px.bar(f_df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index(), 
                        x='STATE/UT', y='TOTAL IPC CRIMES', color='TOTAL IPC CRIMES', title="1. Total IPC Volume per State"), use_container_width=True)
        # 3. Funnel Chart: Severity Ranking
        top_10 = f_df.groupby('STATE/UT')[target_crime].sum().nlargest(10).reset_index()
        st.plotly_chart(px.funnel(top_10, x=target_crime, y='STATE/UT', title="2. Top 10 States Funnel (Target Crime)"), use_container_width=True)
    with col2:
        # 4. Donut Chart: Crime Category Split
        crime_sums = f_df[crime_types].sum().reset_index()
        st.plotly_chart(px.pie(crime_sums, values=0, names='index', hole=0.5, title="3. Crime Type Distribution Ratio"), use_container_width=True)
        # 5. Horizontal Bar: Target Crime Comparison
        st.plotly_chart(px.bar(top_10, y='STATE/UT', x=target_crime, orientation='h', title="4. Categorical Ranking by State"), use_container_width=True)

with tab2:
    st.header("Time-Series & Growth Trends")
    col3, col4 = st.columns(2)
    with col3:
        # 6. Line Chart: Yearly IPC Trend
        st.plotly_chart(px.line(f_df.groupby('YEAR')['TOTAL IPC CRIMES'].sum().reset_index(), 
                        x='YEAR', y='TOTAL IPC CRIMES', markers=True, title="5. Yearly IPC Growth Trend"), use_container_width=True)
        # 7. Area Chart: Target Crime over time
        st.plotly_chart(px.area(f_df.groupby('YEAR')[target_crime].sum().reset_index(), 
                         x='YEAR', y=target_crime, title="6. Volumetric Area Trend (Target Crime)"), use_container_width=True)
    with col4:
        # 8. Histogram: Distribution of occurrences
        st.plotly_chart(px.histogram(f_df, x=target_crime, nbins=20, title="7. Frequency Distribution Histogram"), use_container_width=True)
        # 9. Stepped Line: Year-over-Year Progression
        st.plotly_chart(px.line(f_df.groupby('YEAR')[target_crime].sum().reset_index(), 
                         x='YEAR', y=target_crime, line_shape='hv', title="8. Stepped Progression Trend"), use_container_width=True)

with tab3:
    st.header("Statistical Analysis")
    col5, col6 = st.columns(2)
    with col5:
        # 10. Box Plot: Yearly Variance
        st.plotly_chart(px.box(f_df, x='YEAR', y=target_crime, title="9. Yearly Statistical Variance (Outliers)"), use_container_width=True)
        # 11. Violin Plot: Density Estimation
        st.plotly_chart(px.violin(f_df, y=target_crime, box=True, title="10. Crime Density Violin Plot"), use_container_width=True)
    with col6:
        # 12. Scatter Plot: Murder vs Attempt (Relationship)
        st.plotly_chart(px.scatter(f_df, x='MURDER', y='ATTEMPT TO MURDER', trendline="ols", title="11. Correlation: Murder vs Attempted"), use_container_width=True)
        # 13. Bar Chart: Growth Percentage
        growth = df.groupby('YEAR')[target_crime].sum().pct_change() * 100
        st.plotly_chart(px.bar(growth, title="12. Year-over-Year Growth Velocity (%)"), use_container_width=True)

with tab4:
    st.header("Deep Learning ANN Prediction")
    target_yr = st.slider("Select Forecast Target", 2026, 2035, 2030)
    if st.button("🚀 Run ANN Deep Analysis"):
        # ANN Prep
        X, y = df[['YEAR']].values, df[target_crime].values
        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_s, y_s = sc_X.fit_transform(X), sc_y.fit_transform(y.reshape(-1, 1))
        
        model = Sequential([Dense(64, activation='relu', input_shape=(1,)), Dense(32, activation='relu'), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_s, y_s, epochs=80, verbose=0)
        
        # 14. Forecast Prediction
        pred = sc_y.inverse_transform(model.predict(sc_X.transform([[target_yr]])))[0][0]
        st.metric(f"Predicted {target_crime} for {target_yr}", f"{int(pred):,}")

        # 15. Forecast Visualization Line
        future_yrs = np.array(range(2001, 2036)).reshape(-1, 1)
        future_preds = sc_y.inverse_transform(model.predict(sc_X.transform(future_yrs)))
        forecast_df = pd.DataFrame({'Year': future_yrs.flatten(), 'Prediction': future_preds.flatten()})
        st.plotly_chart(px.line(forecast_df, x='Year', y='Prediction', title="13. Complete AI Trend Projection (2001-2035)"), use_container_width=True)
        
        # 16. Comparison Bar (Final Known vs Future)
        comp_data = pd.DataFrame({'Label': ['2013 Actual', f'{target_yr} Predicted'], 'Count': [df[df['YEAR']==2013][target_crime].sum(), pred]})
        st.plotly_chart(px.bar(comp_data, x='Label', y='Count', color='Label', title="14. Gap Analysis: Actual vs Forecast"), use_container_width=True)
        
        # 17. Growth Gauge (Managerial)
        st.plotly_chart(go.Figure(go.Indicator(mode = "gauge+number", value = pred, title = {'text': "Predicted Volume Gauge"}, gauge = {'axis': {'range': [None, pred*1.5]}})), use_container_width=True)
