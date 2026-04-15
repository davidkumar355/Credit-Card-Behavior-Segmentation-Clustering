import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Segmentation Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Minimalistic Dark Aesthetic
st.markdown("""
<style>
    .stApp header {background-color: transparent;}
    .css-18e3th9 {padding-top: 2rem;}
    .metric-box {
        background-color: #1E293B; 
        border-radius: 12px; 
        padding: 20px; 
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {font-size: 28px; font-weight: bold; color: #3B82F6;}
    .metric-title {font-size: 14px; color: #94A3B8; text-transform: uppercase;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. MACHINE LEARNING PIPELINE LOAD (Joblib)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """Loads pre-trained models via Joblib and the base dataset for charting."""
    
    # 1. Load exported models and scaling configurations
    scaler = joblib.load("model/scaler.pkl")
    pca = joblib.load("model/pca.pkl")
    kmeans = joblib.load("model/kmeans_model.pkl")
    reflection_max_vals = joblib.load("model/reflection_max_vals.pkl")
    
    # 2. Feature columns order
    features = [
        'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 
        'INSTALLMENTS_PURCHASES', 'ONEOFF_PURCHASES_FREQUENCY', 
        'PRC_FULL_PAYMENT', 'TENURE', 'SPEND_INTENSITY', 
        'PAYMENT_DISCIPLINE', 'REVOLVING_BEHAVIOR', 'CASH_DEPENDENCY'
    ]
    
    # 3. Load UI summary data (the dataset with clusters joined via kmeans prediction)
    # This powers the Dashboard tabs without retraining from scratch
    df_raw = pd.read_csv("credit_card_data.csv")
    
    df = df_raw.copy()
    df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
    df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(), inplace=True)
    
    df['SPEND_INTENSITY'] = df['PURCHASES'] / (df['CREDIT_LIMIT'] + 1)
    df['PAYMENT_DISCIPLINE'] = df['PAYMENTS'] / (df['MINIMUM_PAYMENTS'] + 1)
    df['REVOLVING_BEHAVIOR'] = df['BALANCE'] / (df['CREDIT_LIMIT'] + 1)
    df['CASH_DEPENDENCY'] = df['CASH_ADVANCE'] / (df['PURCHASES'] + 1)
    
    # For UI visualization, apply the models locally 
    # (Since we didn't export a finished CSV, we apply the pickle models here onto the raw data)
    # This also acts as a nice validation that models loaded correctly
    df_transformed = pd.DataFrame(index=df.index)
    for col in features:
        # Instead of winsorizing locally, we use global limits (approximation here or raw since it's just for building df_final display)
        # Using raw to feed to the pipeline
        raw_val = df[col]
        
        if col in ['BALANCE_FREQUENCY', 'TENURE']:
            max_val = reflection_max_vals[col]
            df_transformed[col] = np.log1p(max_val + 1 - raw_val)
        else:
            df_transformed[col] = np.log1p(raw_val)
            
    X_scaled = scaler.transform(df_transformed)
    X_pca = pca.transform(X_scaled)
    clusters = kmeans.predict(X_pca)
    
    df['cluster'] = clusters
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    df['PC3'] = X_pca[:, 2]
    
    return df, df_raw, scaler, pca, kmeans, features, reflection_max_vals

with st.spinner("Loading production models..."):
    df_final, df_raw, scaler, pca, kmeans, feature_cols, max_v = load_models_and_data()


# Cluster Meta-data mapping
CLUSTER_META = {
    0: {'name': 'Silent Revolver', 'icon': '🟡', 'strategy': 'Payoff Incentives', 'desc': 'Moderate spenders but chronically carry a balance. Never pay in full.'},
    1: {'name': 'Cash Dependent', 'icon': '🔴', 'strategy': 'Restructure Debt', 'desc': 'Heavy use of cash advances. Financial survival mode.'},
    2: {'name': 'Power Spender', 'icon': '🟢', 'strategy': 'Retain & Upsell Premium', 'desc': 'High usage, clears balance regularly. Highly profitable, low risk.'},
    3: {'name': 'Disciplined Payer', 'icon': '🔵', 'strategy': 'Limit Increase', 'desc': 'Uses card for convenience. Pays perfectly. Reliable.'}
}


# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.title("💳 Segments Dashboard")
st.sidebar.markdown("Navigate between analytic views and the real-time tagging engine.")
menu = st.sidebar.radio("Select View:", [
    "👥 Persona Profiles", 
    "🌍 Interactive Segment Explorer", 
    "⚡ Real-Time Customer Tagging"
])


# ------------------------------------------------------------------------------
# 4. TAB 1: PERSONA DASHBOARD
# ------------------------------------------------------------------------------
if menu == "👥 Persona Profiles":
    st.title("Customer Persona Dashboard")
    st.markdown("Overview of the 4 behavioral archetypes identified within the customer base.")
    
    tabs = st.tabs([f"{c['icon']} {c['name']}" for c in CLUSTER_META.values()])
    
    radar_cols = [
        'SPEND_INTENSITY', 'PAYMENT_DISCIPLINE', 'REVOLVING_BEHAVIOR', 
        'CASH_DEPENDENCY', 'PRC_FULL_PAYMENT', 'BALANCE_FREQUENCY'
    ]
    profile_means = df_final.groupby('cluster')[radar_cols].mean()
    profile_norm = (profile_means - profile_means.min()) / (profile_means.max() - profile_means.min() + 1e-9)

    for i, tab in enumerate(tabs):
        with tab:
            st.header(f"Cluster {i}: {CLUSTER_META[i]['name']}")
            st.subheader(f"💡 Strategy: {CLUSTER_META[i]['strategy']}")
            st.write(CLUSTER_META[i]['desc'])
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("<br>", unsafe_allow_html=True)
                size = len(df_final[df_final['cluster'] == i])
                st.markdown(f"<div class='metric-box'><div class='metric-title'>Segment Size</div><div class='metric-value'>{size:,} Users</div></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                avg_bal = df_final[df_final['cluster'] == i]['BALANCE'].mean()
                st.markdown(f"<div class='metric-box'><div class='metric-title'>Avg. Balance</div><div class='metric-value'>${avg_bal:,.0f}</div></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                avg_pur = df_final[df_final['cluster'] == i]['PURCHASES'].mean()
                st.markdown(f"<div class='metric-box'><div class='metric-title'>Avg. Purchases</div><div class='metric-value'>${avg_pur:,.0f}</div></div>", unsafe_allow_html=True)
                
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=profile_norm.loc[i].values.tolist() + [profile_norm.loc[i].values[0]],
                    theta=radar_cols + [radar_cols[0]],
                    fill='toself',
                    name=CLUSTER_META[i]['name'],
                    line_color="#3B82F6"
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#f8fafc")
                )
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# 5. TAB 2: INTERACTIVE EXPLORER
# ------------------------------------------------------------------------------
elif menu == "🌍 Interactive Segment Explorer":
    st.title("Segment Exploration Space")
    st.markdown("Examine the distribution of customers in the 3-Dimensional PCA space.")
    
    plot_df = df_final.copy()
    plot_df['Segment Name'] = plot_df['cluster'].map(lambda c: CLUSTER_META[c]['name'])
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("### Filters")
        selected_clusters = st.multiselect(
            "Select Segments to View:",
            options=plot_df['Segment Name'].unique(),
            default=plot_df['Segment Name'].unique()
        )
        
    with col2:
        filt_df = plot_df[plot_df['Segment Name'].isin(selected_clusters)]
        if len(filt_df) > 5000:
            filt_df = filt_df.sample(5000, random_state=42)
            
        fig3d = px.scatter_3d(
            filt_df, x='PC1', y='PC2', z='PC3',
            color='Segment Name',
            color_discrete_sequence=["#F59E0B", "#EF4444", "#10B981", "#3B82F6"],
            hover_data=['PURCHASES', 'CASH_ADVANCE', 'BALANCE', 'CREDIT_LIMIT'],
            opacity=0.7,
            title="3D PCA Customer Landscape"
        )
        fig3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True, height=700)


# ------------------------------------------------------------------------------
# 6. TAB 3: REAL-TIME TAGGING
# ------------------------------------------------------------------------------
elif menu == "⚡ Real-Time Customer Tagging":
    st.title("Real-Time Customer Archetype Prediction")
    st.markdown("Input a new customer's raw data to instantly predict their segment via the pickled models.")
    
    st.markdown("---")
    
    with st.form("customer_input_form"):
        st.subheader("Customer Behavior Inputs")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_limit = st.slider("Credit Limit ($)", 50.0, 30000.0, float(df_raw['CREDIT_LIMIT'].median()), 100.0)
            inp_balance = st.slider("Revolving Balance ($)", 0.0, 20000.0, float(df_raw['BALANCE'].median()), 50.0)
            inp_purchases = st.slider("Total Purchases ($)", 0.0, 50000.0, float(df_raw['PURCHASES'].median()), 50.0)
            inp_oneoff = st.slider("One-off Purchases ($)", 0.0, 40000.0, float(df_raw['ONEOFF_PURCHASES'].median()), 50.0)
            
        with c2:
            inp_cash = st.slider("Cash Advance ($)", 0.0, 50000.0, float(df_raw['CASH_ADVANCE'].median()), 50.0)
            inp_payments = st.slider("Total Payments ($)", 0.0, 50000.0, float(df_raw['PAYMENTS'].median()), 50.0)
            inp_min_payments = st.slider("Minimum Payments ($)", 0.0, 10000.0, float(df_raw['MINIMUM_PAYMENTS'].median()), 10.0)
            inp_installments = st.slider("Installment Purchases ($)", 0.0, 25000.0, float(df_raw['INSTALLMENTS_PURCHASES'].median()), 50.0)
            
        with c3:
            inp_bal_freq = st.slider("Balance Frequency (0 to 1)", 0.0, 1.0, float(df_raw['BALANCE_FREQUENCY'].median()), 0.05)
            inp_pur_freq_oneoff = st.slider("One-off Purchase Freq (0 to 1)", 0.0, 1.0, float(df_raw['ONEOFF_PURCHASES_FREQUENCY'].median()), 0.05)
            inp_full_pay = st.slider("Percent Full Payment (0 to 1)", 0.0, 1.0, float(df_raw['PRC_FULL_PAYMENT'].median()), 0.05)
            inp_tenure = st.slider("Tenure (Months)", 1.0, 12.0, float(df_raw['TENURE'].median()), 1.0)

        submitted = st.form_submit_button("Predict Archetype & Pitch")
        
    if submitted:
        # 1. Compute Ratios
        spend_intensity = inp_purchases / (inp_limit + 1)
        payment_discipline = inp_payments / (inp_min_payments + 1)
        revolving_behavior = inp_balance / (inp_limit + 1)
        cash_dependency = inp_cash / (inp_purchases + 1)
        
        # 2. Extract into proper df shape
        user_data = pd.DataFrame([{
            'BALANCE_FREQUENCY': inp_bal_freq,
            'PURCHASES': inp_purchases,
            'ONEOFF_PURCHASES': inp_oneoff,
            'INSTALLMENTS_PURCHASES': inp_installments,
            'ONEOFF_PURCHASES_FREQUENCY': inp_pur_freq_oneoff,
            'PRC_FULL_PAYMENT': inp_full_pay,
            'TENURE': inp_tenure,
            'SPEND_INTENSITY': spend_intensity,
            'PAYMENT_DISCIPLINE': payment_discipline,
            'REVOLVING_BEHAVIOR': revolving_behavior,
            'CASH_DEPENDENCY': cash_dependency
        }])
        
        # 3. Apply skewness mapping with saved reflection upper limits
        user_transformed = pd.DataFrame(index=[0])
        for col in feature_cols:
            val_raw = user_data[col].iloc[0]
            
            if col in ['BALANCE_FREQUENCY', 'TENURE']:
                # Utilize the strict mapping saved statically from training
                user_transformed[col] = [np.log1p(max_v[col] + 1 - val_raw)]
            else:
                user_transformed[col] = [np.log1p(val_raw)]
                
        # 4. Standard Scale via Pickled Model
        user_scaled = scaler.transform(user_transformed)
        
        # 5. PCA via Pickled Model
        user_pca = pca.transform(user_scaled)
        
        # 6. Predict via Pickled Model
        predicted_cluster = kmeans.predict(user_pca)[0]
        meta = CLUSTER_META[predicted_cluster]
        
        st.markdown("---")
        st.success(f"### Prediction Result: {meta['icon']} {meta['name']}")
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.info(f"**Customer Insight:** {meta['desc']}")
        with sc2:
            st.warning(f"**Immediate Pitch Strategy:** {meta['strategy']}")
            
        st.markdown(f"*(Model Coordinates Pipeline execution complete - Pickled System: `True`)*")
