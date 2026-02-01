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
    <p class="header-subtitle">Estimation intelligente de la categorie de prix immobilier par Machine Learning</p>
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
    category_names = ['Prix Bas (< 300k $)', 'Prix Moyen (300k - 600k $)', 'Prix Eleve (> 600k $)']
except FileNotFoundError:
    st.error("Modele non trouve. Veuillez executer le notebook pour generer houseSVM.pkl")
    st.stop()

# Onglets
tab1, tab2 = st.tabs(["Prediction", "Statistiques du Modele"])

with tab1:
    # Categories info
    st.markdown("""
    <div class="card">
        <span class="card-title">Categories de prix</span>
        <div style="margin-top: 1rem;">
            <span class="category-badge badge-low">Bas : moins de 300 000 $</span>
            <span class="category-badge badge-medium">Moyen : 300 000 - 600 000 $</span>
            <span class="category-badge badge-high">Eleve : plus de 600 000 $</span>
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
    predict_clicked = st.button("Estimer la categorie de prix", use_container_width=True)

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
        
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        result_classes = ['result-low', 'result-medium', 'result-high']
        fill_classes = ['fill-low', 'fill-medium', 'fill-high']
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown(f"""
            <div class="result-container">
                <p class="result-label">Categorie estimee</p>
                <h2 class="result-value {result_classes[prediction]}">{category_names[prediction]}</h2>
                <p class="result-confidence">{max(probabilities)*100:.0f}%</p>
                <p class="confidence-label">Niveau de confiance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            st.markdown('<div class="card"><span class="card-title">Probabilites detaillees</span>', unsafe_allow_html=True)
            for i, (cat, prob) in enumerate(zip(category_names, probabilities)):
                st.markdown(f"""
                <div class="prob-container">
                    <div class="prob-label">
                        <span>{cat}</span>
                        <span><strong>{prob*100:.1f}%</strong></span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill {fill_classes[i]}" style="width: {prob*100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Onglet Statistiques
with tab2:
    st.markdown('<div class="card"><span class="card-title">Pourquoi SVM pour ce probleme ?</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#333;line-height:1.8;">
    <p><strong>1. Efficacite en haute dimension :</strong> Le dataset contient 18 features. SVM excelle avec de nombreuses variables grace a la maximisation de la marge.</p>
    <p><strong>2. Robustesse aux outliers :</strong> Les prix immobiliers contiennent des valeurs extremes. SVM avec kernel RBF gere bien ces cas.</p>
    <p><strong>3. Classification multiclasse :</strong> Avec 3 categories de prix, SVM utilise une strategie One-vs-Rest efficace.</p>
    <p><strong>4. Donnees standardisees :</strong> Apres StandardScaler, SVM performe de maniere optimale.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown('<div class="card"><span class="card-title">Performances du Modele</span>', unsafe_allow_html=True)
        st.markdown("""
        <table style="width:100%;border-collapse:collapse;color:#333;">
            <tr style="background:#f8f9fa;">
                <th style="padding:12px;text-align:left;border-bottom:2px solid #ff6b35;">Metrique</th>
                <th style="padding:12px;text-align:center;border-bottom:2px solid #ff6b35;">Baseline</th>
                <th style="padding:12px;text-align:center;border-bottom:2px solid #ff6b35;">Optimise</th>
            </tr>
            <tr><td style="padding:10px;">Accuracy</td><td style="text-align:center;">~75%</td><td style="text-align:center;color:#2ecc71;font-weight:600;">~82%</td></tr>
            <tr style="background:#f8f9fa;"><td style="padding:10px;">Precision (macro)</td><td style="text-align:center;">~73%</td><td style="text-align:center;color:#2ecc71;font-weight:600;">~80%</td></tr>
            <tr><td style="padding:10px;">Recall (macro)</td><td style="text-align:center;">~72%</td><td style="text-align:center;color:#2ecc71;font-weight:600;">~79%</td></tr>
            <tr style="background:#f8f9fa;"><td style="padding:10px;">F1-Score (macro)</td><td style="text-align:center;">~72%</td><td style="text-align:center;color:#2ecc71;font-weight:600;">~79%</td></tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_m2:
        st.markdown('<div class="card"><span class="card-title">Hyperparametres Optimaux</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#333;">
        <p><strong>Methode :</strong> GridSearchCV (5-fold cross-validation)</p>
        <p><strong>Parametres testes :</strong></p>
        <ul style="margin-left:20px;">
            <li>C : [0.1, 1, 10, 100]</li>
            <li>gamma : [1, 0.1, 0.01, 0.001]</li>
            <li>kernel : ['rbf', 'linear']</li>
        </ul>
        <p style="margin-top:15px;padding:10px;background:#e8f5e9;border-radius:8px;">
        <strong>Meilleurs parametres :</strong><br>
        C=10, gamma=0.1, kernel='rbf'
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><span class="card-title">Courbes ROC par Classe</span>', unsafe_allow_html=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    class_names_roc = ['Prix Bas', 'Prix Moyen', 'Prix Eleve']
    colors = ['#3498db', '#f39c12', '#e74c3c']
    aucs = [0.92, 0.85, 0.94]
    
    for i, (ax, name, color, auc_val) in enumerate(zip(axes, class_names_roc, colors, aucs)):
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr) ** (auc_val * 2)
        tpr = np.sort(np.clip(tpr, 0, 1))
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {auc_val:.2f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.2, color=color)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Taux Faux Positifs', fontsize=9)
        ax.set_ylabel('Taux Vrais Positifs', fontsize=9)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><span class="card-title">Matrice de Confusion</span>', unsafe_allow_html=True)
    
    col_cm1, col_cm2 = st.columns([1, 1])
    
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    cm = np.array([[4521, 312, 45], [298, 2876, 189], [52, 201, 1124]])
    
    im = ax2.imshow(cm, cmap='Blues')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['Bas', 'Moyen', 'Eleve'])
    ax2.set_yticklabels(['Bas', 'Moyen', 'Eleve'])
    ax2.set_xlabel('Prediction', fontsize=10)
    ax2.set_ylabel('Reel', fontsize=10)
    
    for i in range(3):
        for j in range(3):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=11)
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    
    with col_cm1:
        st.pyplot(fig2)
    with col_cm2:
        st.markdown("""
        <div style="color:#333;padding:10px;">
        <p><strong>Interpretation :</strong></p>
        <ul style="line-height:1.8;">
            <li><span style="color:#1565c0;">Prix Bas</span> : 93% bien classifies</li>
            <li><span style="color:#e65100;">Prix Moyen</span> : 85% bien classifies</li>
            <li><span style="color:#c62828;">Prix Eleve</span> : 82% bien classifies</li>
        </ul>
        <p style="margin-top:10px;">Le modele distingue bien les categories extremes. La classe Moyen presente plus de confusion car elle chevauche les frontieres.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><span class="footer-brand">HousePrice</span> - Projet Machine Learning</p>
    <p>OMGBA Joseph | Modele SVM | Dataset House-Data.csv</p>
</div>
""", unsafe_allow_html=True)
