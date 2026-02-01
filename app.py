import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="HousePrice - Estimation Immobiliere",
    page_icon="house",
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
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">House<span class="header-accent">Price</span></h1>
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

# Formulaire
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><span class="card-title">Structure</span>', unsafe_allow_html=True)
    bedrooms = st.number_input("Chambres", min_value=0, max_value=20, value=3)
    bathrooms = st.number_input("Salles de bain", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
    floors = st.number_input("Etages", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    condition = st.slider("Condition", min_value=1, max_value=5, value=3)
    grade = st.slider("Grade qualite", min_value=1, max_value=13, value=7)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><span class="card-title">Surfaces</span>', unsafe_allow_html=True)
    sqft_living = st.number_input("Surface habitable (sqft)", min_value=200, max_value=15000, value=1800)
    sqft_lot = st.number_input("Surface terrain (sqft)", min_value=500, max_value=1000000, value=5000)
    sqft_above = st.number_input("Surface hors-sol (sqft)", min_value=200, max_value=10000, value=1500)
    sqft_basement = st.number_input("Surface sous-sol (sqft)", min_value=0, max_value=5000, value=300)
    sqft_living15 = st.number_input("Surface voisins (sqft)", min_value=200, max_value=10000, value=1800)
    sqft_lot15 = st.number_input("Terrain voisins (sqft)", min_value=500, max_value=500000, value=5000)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><span class="card-title">Localisation</span>', unsafe_allow_html=True)
    waterfront = st.selectbox("Vue sur eau", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    view = st.slider("Qualite vue", min_value=0, max_value=4, value=0)
    yr_built = st.number_input("Annee construction", min_value=1900, max_value=2025, value=1990)
    yr_renovated = st.number_input("Annee renovation", min_value=0, max_value=2025, value=0)
    zipcode = st.number_input("Code postal", min_value=98001, max_value=98199, value=98103)
    lat = st.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.5, format="%.4f")
    long = st.number_input("Longitude", min_value=-123.0, max_value=-121.0, value=-122.3, format="%.4f")
    st.markdown('</div>', unsafe_allow_html=True)

# Bouton
st.markdown("<br>", unsafe_allow_html=True)
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
    
    st.markdown("<br>", unsafe_allow_html=True)
    
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

# Footer
st.markdown("""
<div class="footer">
    <p><span class="footer-brand">HousePrice</span> - Projet Machine Learning</p>
    <p>OMGBA Joseph | Modele SVM | Dataset House-Data.csv</p>
</div>
""", unsafe_allow_html=True)
