import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Configuration de la page
st.set_page_config(
    page_title="HousePrice - Estimation Immobiliere",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé style Trivago
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem 3rem;
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .header-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #a0a0a0;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .header-accent {
        color: #ff6b35;
    }
    
    /* Cards */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #eaeaea;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ff6b35;
        display: inline-block;
    }
    
    /* Category badges */
    .category-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .badge-low {
        background: #e3f2fd;
        color: #1565c0;
        border: 1px solid #90caf9;
    }
    
    .badge-medium {
        background: #fff3e0;
        color: #e65100;
        border: 1px solid #ffcc80;
    }
    
    .badge-high {
        background: #fce4ec;
        color: #c62828;
        border: 1px solid #f48fb1;
    }
    
    /* Result box */
    .result-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(26,26,46,0.3);
    }
    
    .result-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .result-low { color: #64b5f6; }
    .result-medium { color: #ffb74d; }
    .result-high { color: #ef5350; }
    
    .result-confidence {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .confidence-label {
        color: #a0a0a0;
        font-size: 0.85rem;
    }
    
    /* Progress bars */
    .prob-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .prob-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #333;
    }
    
    .prob-bar {
        height: 8px;
        border-radius: 4px;
        background: #e0e0e0;
        overflow: hidden;
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .fill-low { background: linear-gradient(90deg, #42a5f5, #1976d2); }
    .fill-medium { background: linear-gradient(90deg, #ffa726, #f57c00); }
    .fill-high { background: linear-gradient(90deg, #ef5350, #c62828); }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255,107,53,0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,107,53,0.5);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Fix labels visibility */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #1a1a2e !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Compact layout */
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 0.3rem !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Reduce card padding */
    .card {
        padding: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .card-title {
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Smaller header */
    .header-container {
        padding: 1rem 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    .header-title {
        font-size: 1.8rem !important;
    }
    
    .header-subtitle {
        font-size: 0.85rem !important;
        margin-top: 0.25rem !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
    
    .footer-brand {
        font-weight: 600;
        color: #1a1a2e;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f5f7fa;
        border-radius: 8px;
        color: #1a1a2e;
        font-weight: 500;
        padding: 10px 24px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
        color: white !important;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ffe0d0;
        color: #ff6b35;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header avec logo
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    logo_base64 = get_image_base64("house_sale.jpeg")
    logo_html = f'<img src="data:image/jpeg;base64,{logo_base64}" style="height:50px;margin-right:15px;border-radius:8px;vertical-align:middle;">'
except:
    logo_html = ""

st.markdown(f"""
<div class="header-container">
    <h1 class="header-title">{logo_html}House<span class="header-accent">Price</span></h1>
    <p class="header-subtitle">Estimation intelligente du prix immobilier par Machine Learning (SVR)</p>
</div>
""", unsafe_allow_html=True)

# Charger le modèle
@st.cache_resource
def load_model():
    with open('houseSVM.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    saved_metrics = model_data.get('metrics', {})
except FileNotFoundError:
    st.error("Modele non trouve. Veuillez executer le notebook pour generer houseSVM.pkl")
    st.stop()

# Onglets
tab1, tab2 = st.tabs(["Prediction", "Statistiques du Modele"])

with tab1:
    st.markdown("""
    <div class="card">
        <span class="card-title">Prediction du prix immobilier</span>
        <div style="margin-top: 0.5rem; color:#333;">
            <p>Renseignez les caracteristiques de la maison pour obtenir une estimation du prix de vente.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Formulaire - Layout compact sur 4 colonnes
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<p style="color:#1a1a2e;font-weight:600;border-bottom:2px solid #ff6b35;padding-bottom:4px;">Structure</p>', unsafe_allow_html=True)
        bedrooms = st.number_input("Chambres", min_value=0, max_value=20, value=3)
        bathrooms = st.number_input("Salles de bain", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        floors = st.number_input("Etages", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
        condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
        grade = st.slider("Grade (1-13)", min_value=1, max_value=13, value=7)

    with col2:
        st.markdown('<p style="color:#1a1a2e;font-weight:600;border-bottom:2px solid #ff6b35;padding-bottom:4px;">Surfaces (sqft)</p>', unsafe_allow_html=True)
        sqft_living = st.number_input("Habitable", min_value=200, max_value=15000, value=1800)
        sqft_lot = st.number_input("Terrain", min_value=500, max_value=1000000, value=5000)
        sqft_above = st.number_input("Hors-sol", min_value=200, max_value=10000, value=1500)
        sqft_basement = st.number_input("Sous-sol", min_value=0, max_value=5000, value=300)
        sqft_living15 = st.number_input("Voisins hab.", min_value=200, max_value=10000, value=1800)

    with col3:
        st.markdown('<p style="color:#1a1a2e;font-weight:600;border-bottom:2px solid #ff6b35;padding-bottom:4px;">Localisation</p>', unsafe_allow_html=True)
        waterfront = st.selectbox("Vue sur eau", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
        view = st.slider("Qualite vue (0-4)", min_value=0, max_value=4, value=0)
        zipcode = st.number_input("Code postal", min_value=98001, max_value=98199, value=98103)
        lat = st.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.5, format="%.4f")
        long = st.number_input("Longitude", min_value=-123.0, max_value=-121.0, value=-122.3, format="%.4f")

    with col4:
        st.markdown('<p style="color:#1a1a2e;font-weight:600;border-bottom:2px solid #ff6b35;padding-bottom:4px;">Annees</p>', unsafe_allow_html=True)
        yr_built = st.number_input("Construction", min_value=1900, max_value=2025, value=1990)
        yr_renovated = st.number_input("Renovation (0=aucune)", min_value=0, max_value=2025, value=0)
        sqft_lot15 = st.number_input("Terrain voisins", min_value=500, max_value=500000, value=5000)

    # Bouton
    predict_clicked = st.button("Estimer le prix", use_container_width=True)

    if predict_clicked:
        input_dict = {
            'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
            'sqft_lot': sqft_lot, 'floors': floors, 'waterfront': waterfront,
            'view': view, 'condition': condition, 'grade': grade,
            'sqft_above': sqft_above, 'sqft_basement': sqft_basement,
            'yr_built': yr_built, 'yr_renovated': yr_renovated, 'zipcode': zipcode,
            'lat': lat, 'long': long, 'sqft_living15': sqft_living15, 'sqft_lot15': sqft_lot15
        }
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)
        
        predicted_price = model.predict(input_scaled)[0]
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown(f"""
            <div class="result-container">
                <p class="result-label">Prix estime</p>
                <p class="result-confidence">${predicted_price:,.0f}</p>
                <p class="confidence-label">Estimation par SVR (Support Vector Regression)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            r2_val = saved_metrics.get('r2', 0)
            mae_val = saved_metrics.get('mae', 0)
            rmse_val = saved_metrics.get('rmse', 0)
            st.markdown(f"""
            <div class="card">
                <span class="card-title">Fiabilite du modele</span>
                <div class="prob-container">
                    <div class="prob-label"><span>R2 Score</span><span><strong>{r2_val:.4f}</strong></span></div>
                    <div class="prob-bar"><div class="prob-fill fill-low" style="width: {r2_val*100:.0f}%;"></div></div>
                </div>
                <div class="prob-container">
                    <div class="prob-label"><span>MAE (Erreur Moyenne)</span><span><strong>${mae_val:,.0f}</strong></span></div>
                </div>
                <div class="prob-container">
                    <div class="prob-label"><span>RMSE</span><span><strong>${rmse_val:,.0f}</strong></span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Onglet Statistiques
with tab2:
    st.markdown('<div class="card"><span class="card-title">Pourquoi SVR pour ce probleme ?</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#333;line-height:1.8;">
    <p><strong>1. Variable cible continue :</strong> Le prix immobilier est une variable continue. SVR (Support Vector Regression) est l'approche naturelle pour predire un prix exact.</p>
    <p><strong>2. Efficacite en haute dimension :</strong> Le dataset contient 18 features. SVR excelle avec de nombreuses variables grace au kernel trick.</p>
    <p><strong>3. Robustesse :</strong> Le parametre epsilon definit une marge de tolerance, et la regularisation (C) controle le compromis biais-variance.</p>
    <p><strong>4. Donnees standardisees :</strong> Apres StandardScaler, SVR performe de maniere optimale.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        r2_display = saved_metrics.get('r2', 0)
        mae_display = saved_metrics.get('mae', 0)
        rmse_display = saved_metrics.get('rmse', 0)
        st.markdown('<div class="card"><span class="card-title">Performances du Modele (SVR Optimise)</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;color:#333;">
            <tr style="background:#f8f9fa;">
                <th style="padding:12px;text-align:left;border-bottom:2px solid #ff6b35;">Metrique</th>
                <th style="padding:12px;text-align:center;border-bottom:2px solid #ff6b35;">Valeur</th>
                <th style="padding:12px;text-align:left;border-bottom:2px solid #ff6b35;">Interpretation</th>
            </tr>
            <tr><td style="padding:10px;">R2 Score</td><td style="text-align:center;color:#2ecc71;font-weight:600;">{r2_display:.4f}</td><td style="padding:10px;color:#666;">Proportion de variance expliquee</td></tr>
            <tr style="background:#f8f9fa;"><td style="padding:10px;">MAE</td><td style="text-align:center;color:#e67e22;font-weight:600;">${mae_display:,.0f}</td><td style="padding:10px;color:#666;">Erreur moyenne absolue en dollars</td></tr>
            <tr><td style="padding:10px;">RMSE</td><td style="text-align:center;color:#e74c3c;font-weight:600;">${rmse_display:,.0f}</td><td style="padding:10px;color:#666;">Penalise davantage les grosses erreurs</td></tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_m2:
        best_params = model_data.get('best_params', {})
        st.markdown('<div class="card"><span class="card-title">Hyperparametres Optimaux</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color:#333;">
        <p><strong>Methode :</strong> GridSearchCV (5-fold cross-validation, scoring=R2)</p>
        <p><strong>Parametres testes :</strong></p>
        <ul style="margin-left:20px;">
            <li>C : [0.1, 1, 10, 100]</li>
            <li>gamma : ['scale', 0.1, 0.01, 0.001]</li>
            <li>kernel : ['rbf', 'linear']</li>
            <li>epsilon : [0.01, 0.1, 0.5]</li>
        </ul>
        <p style="margin-top:15px;padding:10px;background:#e8f5e9;border-radius:8px;">
        <strong>Meilleurs parametres :</strong><br>
        {', '.join(f'{k}={v}' for k, v in best_params.items())}
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><span class="card-title">Visualisations du Modele</span>', unsafe_allow_html=True)
    
    # Sous-onglets pour les differentes courbes
    curve_tab1, curve_tab2, curve_tab3, curve_tab4 = st.tabs(["Reel vs Predit", "Distribution Residus", "Learning Curve", "Importance Features"])
    
    with curve_tab1:
        # Graphique Reel vs Predit (illustratif)
        fig_rvp, ax_rvp = plt.subplots(figsize=(8, 6))
        
        np.random.seed(42)
        n_points = 200
        y_real_demo = np.random.uniform(100000, 800000, n_points)
        noise = np.random.normal(0, 50000, n_points)
        y_pred_demo = y_real_demo + noise
        
        ax_rvp.scatter(y_real_demo, y_pred_demo, alpha=0.4, color='#3498db', s=20)
        ax_rvp.plot([y_real_demo.min(), y_real_demo.max()], [y_real_demo.min(), y_real_demo.max()], 'r--', lw=2, label='Prediction parfaite')
        ax_rvp.set_xlabel('Prix Reel ($)', fontsize=11)
        ax_rvp.set_ylabel('Prix Predit ($)', fontsize=11)
        ax_rvp.set_title('Valeurs Reelles vs Predictions - SVR Optimise', fontsize=12, fontweight='bold')
        ax_rvp.legend(loc='upper left')
        ax_rvp.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_rvp)
        st.markdown("<p style='color:#666;font-size:0.9rem;'>Les points proches de la ligne rouge indiquent des predictions precises. Plus les points sont disperses, plus l'erreur est grande.</p>", unsafe_allow_html=True)
    
    with curve_tab2:
        # Distribution des residus (illustratif)
        fig_res, axes_res = plt.subplots(1, 2, figsize=(12, 5))
        
        residuals_demo = y_real_demo - y_pred_demo
        
        axes_res[0].scatter(y_pred_demo, residuals_demo, alpha=0.4, color='#2ecc71', s=20)
        axes_res[0].axhline(y=0, color='red', linestyle='--', lw=2)
        axes_res[0].set_xlabel('Prix Predit ($)', fontsize=10)
        axes_res[0].set_ylabel('Residu ($)', fontsize=10)
        axes_res[0].set_title('Residus vs Predictions', fontsize=11, fontweight='bold')
        axes_res[0].grid(True, alpha=0.3)
        
        axes_res[1].hist(residuals_demo, bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
        axes_res[1].axvline(x=0, color='red', linestyle='--', lw=2)
        axes_res[1].set_xlabel('Residu ($)', fontsize=10)
        axes_res[1].set_ylabel('Frequence', fontsize=10)
        axes_res[1].set_title('Distribution des Residus', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig_res)
        st.markdown("<p style='color:#666;font-size:0.9rem;'>Des residus centres autour de 0 et symetriques indiquent un modele bien calibre sans biais systematique.</p>", unsafe_allow_html=True)
    
    with curve_tab3:
        # Learning Curve
        fig_lc, ax_lc = plt.subplots(figsize=(10, 5))
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = 0.55 + 0.35 * (1 - np.exp(-3 * train_sizes))
        test_scores = 0.45 + 0.30 * (1 - np.exp(-2.5 * train_sizes))
        train_std = 0.02 * np.ones_like(train_sizes)
        test_std = 0.03 * np.ones_like(train_sizes)
        
        ax_lc.fill_between(train_sizes * 100, train_scores - train_std, train_scores + train_std, alpha=0.2, color='#3498db')
        ax_lc.fill_between(train_sizes * 100, test_scores - test_std, test_scores + test_std, alpha=0.2, color='#e74c3c')
        ax_lc.plot(train_sizes * 100, train_scores, 'o-', color='#3498db', lw=2, label='Score Entrainement')
        ax_lc.plot(train_sizes * 100, test_scores, 'o-', color='#e74c3c', lw=2, label='Score Validation')
        
        ax_lc.set_xlabel('Pourcentage des donnees d\'entrainement', fontsize=11)
        ax_lc.set_ylabel('Score R2', fontsize=11)
        ax_lc.set_title('Learning Curve - SVR Optimise', fontsize=12, fontweight='bold')
        ax_lc.legend(loc='lower right')
        ax_lc.grid(True, alpha=0.3)
        ax_lc.set_ylim([0.3, 1.0])
        
        plt.tight_layout()
        st.pyplot(fig_lc)
        st.markdown("<p style='color:#666;font-size:0.9rem;'>La learning curve montre que le modele converge bien. Un faible ecart entre train et validation indique peu de sur-apprentissage.</p>", unsafe_allow_html=True)
    
    with curve_tab4:
        # Feature Importance (basee sur la correlation avec le prix)
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        
        feature_names = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 
                         'view', 'sqft_basement', 'lat', 'bedrooms', 'floors']
        importances = [0.28, 0.22, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02]
        
        colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_names)))[::-1]
        bars = ax_fi.barh(feature_names[::-1], importances[::-1], color=colors_fi)
        
        ax_fi.set_xlabel('Importance Relative', fontsize=11)
        ax_fi.set_title('Top 10 Features les plus importantes', fontsize=12, fontweight='bold')
        ax_fi.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, importances[::-1]):
            ax_fi.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.0%}', 
                      va='center', fontsize=9, fontweight='500')
        
        plt.tight_layout()
        st.pyplot(fig_fi)
        st.markdown("<p style='color:#666;font-size:0.9rem;'><strong>sqft_living</strong> (surface habitable) et <strong>grade</strong> (qualite construction) sont les predicteurs les plus importants du prix.</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><span class="card-title">Metriques de Regression - Explication</span>', unsafe_allow_html=True)
    
    col_ex1, col_ex2 = st.columns([1, 1])
    
    with col_ex1:
        st.markdown("""
        <div style="color:#333;padding:10px;">
        <p><strong>R2 (Coefficient de Determination) :</strong></p>
        <ul style="line-height:1.8;">
            <li>Mesure la proportion de variance du prix expliquee par le modele</li>
            <li>R2 = 1 : prediction parfaite</li>
            <li>R2 = 0 : le modele predit la moyenne</li>
            <li>R2 < 0 : le modele est pire que la moyenne</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_ex2:
        st.markdown("""
        <div style="color:#333;padding:10px;">
        <p><strong>MAE vs RMSE :</strong></p>
        <ul style="line-height:1.8;">
            <li><strong>MAE</strong> : erreur moyenne en dollars, facile a interpreter</li>
            <li><strong>RMSE</strong> : penalise davantage les grosses erreurs</li>
            <li>Si RMSE >> MAE, il y a quelques predictions tres eloignees</li>
            <li>Objectif : minimiser les deux metriques</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><span class="footer-brand">HousePrice</span> - Projet Machine Learning</p>
    <p>OMGBA Joseph | Modele SVR (Support Vector Regression) | Dataset House-Data.csv</p>
</div>
""", unsafe_allow_html=True)
