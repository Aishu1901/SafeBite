%%writefile streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import PyPDF2

# Configure page
st.set_page_config(page_title="SafeBite", layout="wide")
st.title("SafeBite - Smarter menus, safer meals")

# Load data
def load_data():
    return pd.read_excel("SafeBite_dataset.xlsx")

df = load_data()

# Simulated kitchen workflow defaults
workflow_col = {
    "shared_fryer": 1,
    "shared_prep_surface": 1,
    "separate_allergen_area": 0
}
for col, default in workflow_col.items():
    if col not in df.columns:
        df[col] = default

dish_risk = {"uses_shared_equipment": 1}
for col, default in dish_risk.items():
    if col not in df.columns:
        df[col] = default

# Rename columns
if 'Restaurnt name' in df.columns:
    df.rename(columns={'Restaurnt name': 'restaurant_name'}, inplace=True)
if 'Food Name' in df.columns:
    df.rename(columns={'Food Name': 'food_name'}, inplace=True)

# Ensure 'full_ingredient_list' column exists
if 'full_ingredient_list' not in df.columns:
    found_ingredient_col = False
    for col in df.columns:
        if 'ingredient' in col.lower() and 'list' in col.lower():
            df.rename(columns={col: 'full_ingredient_list'}, inplace=True)
            found_ingredient_col = True
            break
    if not found_ingredient_col:
        df['full_ingredient_list'] = ""

# Ensure 'allergy_safety_status' column exists
if 'allergy_safety_status' not in df.columns:
    df['allergy_safety_status'] = 'caution'

# Allergies
allergy = [
    "milk", "eggs", "wheat_gluten", "soy", "peanuts", 
    "tree_nuts", "fish", "shellfish", "sesame"
]

allergen_keywords = {
    "milk": ["milk", "cheese", "butter", "cream", "yogurt", "ghee", "whey", "casein", "lactose"],
    "eggs": ["egg", "eggs", "albumin"],
    "wheat_gluten": ["wheat", "gluten", "flour", "bread", "pasta"],
    "soy": ["soy", "soya", "tofu", "edamame"],
    "peanuts": ["peanut", "groundnut"],
    "tree_nuts": ["almond", "cashew", "walnut", "pistachio", "hazelnut"],
    "fish": ["fish", "salmon", "tuna", "cod"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster"],
    "sesame": ["sesame", "tahini"]
}

hidden_risk_terms = {
    "cross_contamination": [
        "may contain", "shared", "facility", "same kitchen", 
        "same equipment", "traces", "processed in", "manufactured in"
    ],
    "ambiguous_ingredients": ["natural flavors", "spices", "seasoning", "sauce", "flavoring"]
}

# Ensure allergy columns exist and are numeric
for alg in allergy:
    if alg not in df.columns:
        df[alg] = 0
    else:
        df[alg] = pd.to_numeric(df[alg], errors='coerce').fillna(0).astype(int)

# User allergies
st.sidebar.header("Your Allergies")
st.sidebar.markdown("---")
allergy_severity = st.sidebar.selectbox(
    "Allergy Severity", ["Mild", "Moderate", "Severe (Anaphylactic)"]
)

user_allergies = {}
for allergen in allergy:
    user_allergies[allergen] = st.sidebar.checkbox(allergen.replace("_", " ").title())

custom_input = st.sidebar.text_input(
    "Other allergies (comma separated)", placeholder="mustard, kiwi"
)
custom_allergies = [a.strip().lower() for a in custom_input.split(",") if a.strip()]
use_ml = st.sidebar.checkbox("Use AI model for risk review", value=True)

# OCR functions
def extract_text_from_image(uploaded_file):
    image = Image.open(io.BytesIO(uploaded_file.read()))
    text = pytesseract.image_to_string(image)
    return text.lower()

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()

def parse_menu_text(text, restaurant_name):
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 3]
    rows = []
    for line in lines:
        rows.append({
            "restaurant_name": restaurant_name,
            "restaurant_type": "uploaded",
            "menu_selection": "Unknown",
            "food_name": line.title(),
            "full_ingredient_list": line,
            "allergy_safety_status": "caution",
            **{a: 0 for a in allergy}
        })
    return pd.DataFrame(rows)

# Restaurant menu upload
st.sidebar.markdown("---")
st.sidebar.header("Restaurant upload")
uploaded_file = st.sidebar.file_uploader("Upload restaurant menu", type=None)

if uploaded_file:
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".csv"):
            upload_df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            upload_df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".pdf"):
            st.sidebar.info("Extracting menu from PDF")
            text = extract_text_from_pdf(uploaded_file)
            upload_df = parse_menu_text(text, uploaded_file.name.split('.')[0])
        elif file_name.endswith((".png", ".jpg", ".jpeg")):
            st.sidebar.info("Extracting menu from image (OCR)")
            text = extract_text_from_image(uploaded_file)
            upload_df = parse_menu_text(text, "Uploaded Image Menu")
        else:
            st.sidebar.error("Unsupported file format.")
            upload_df = None

        if upload_df is not None:
            for alg in allergy:
                if alg not in upload_df.columns:
                    upload_df[alg] = 0
                else:
                    upload_df[alg] = pd.to_numeric(upload_df[alg], errors='coerce').fillna(0).astype(int)

            required_cols = ["restaurant_name", "food_name", "full_ingredient_list", "allergy_safety_status"] + allergy
            missing = [c for c in required_cols if c not in upload_df.columns]
            if missing:
                st.sidebar.error(f"Missing columns: {missing}")
            else:
                if 'Restaurnt name' in upload_df.columns:
                    upload_df.rename(columns={'Restaurnt name': 'restaurant_name'}, inplace=True)
                df = pd.concat([df, upload_df], ignore_index=True)
                df.to_excel("SafeBite_dataset.xlsx", index=False)
                st.sidebar.success("Menu uploaded successfully and saved to SafeBite_dataset.xlsx")
    except Exception as e:
        st.sidebar.error(f"Failed to process uploaded file: {e}")

# Rule-based safety logic
def rule_based_safety(row):
    ingredients = str(row.get("full_ingredient_list", "")).lower().strip()
    dish_text = f"{row['food_name']} {ingredients}"
    uses_fryer = any(x in dish_text for x in ["fried", "fries", "tempura", "crispy"])
    raw_prep = any(x in dish_text for x in ["salad", "raw", "sashimi"])

    if not any(user_allergies.values()) and not custom_allergies:
        return "safe", ["No allergies selected"]

    reasons = []
    if ingredients == "":
        return "caution", ["Ingredient information not available"]

    for allergen, selected in user_allergies.items():
        if not selected:
            continue
        if row.get(allergen, 0) == 1:
            reasons.append(f"Contains {allergen.replace('_', ' ')}")
        for keyword in allergen_keywords.get(allergen, []):
            if keyword in ingredients:
                reasons.append(f"Contains {keyword}")

    for allergy_item in custom_allergies:
        if allergy_item in ingredients:
            reasons.append(f"Contains {allergy_item}")

    if reasons:
        return "unsafe", list(set(reasons))

    hidden_reasons = []
    for phrase in hidden_risk_terms["cross_contamination"]:
        if phrase in ingredients:
            hidden_reasons.append(f"Possible cross-contamination: '{phrase}'")
    for phrase in hidden_risk_terms["ambiguous_ingredients"]:
        if phrase in ingredients:
            hidden_reasons.append(f"Ambiguous ingredient: '{phrase}'")
    if hidden_reasons:
        return "caution", list(set(hidden_reasons))

    if any(user_allergies.values()) or custom_allergies:
        if uses_fryer and row.get("shared_fryer", 1) == 1:
            reasons.append("High risk: Shared fryer")
        if not raw_prep and row.get("shared_prep_surface", 1) == 1:
            reasons.append("Medium risk: Shared prep surface")
        if row.get("separate_allergen_area", 0) == 0:
            reasons.append("No separate allergen prep area")
        if row.get("uses_shared_equipment", 1) == 1:
            reasons.append("Dish prepared using shared equipment")
        if reasons and "unsafe" not in [r.lower() for r in reasons]:
            return "caution", list(set(reasons))
        if allergy_severity == "Severe (Anaphylactic)" and reasons:
            return "unsafe", list(set(reasons))

    return "safe", ["No selected allergens detected"]

def compute_safety_score(final_status, reasons, allergy_severity):
    base_scores = {"safe": 90, "caution": 55, "unsafe": 15}
    score = base_scores.get(final_status.lower(), 50)
    score -= 10 * len(reasons)
    if allergy_severity == "Moderate":
        score -= 10
    elif allergy_severity == "Severe (Anaphylactic)":
        score -= 20
    return max(0, min(100, score))

def compute_restaurant_safety(df_restaurant, allergy_severity):
    scores = []
    for _, row in df_restaurant.iterrows():
        rule_status, reasons = rule_based_safety(row)
        score = compute_safety_score(rule_status, reasons, allergy_severity)
        scores.append(score)
    return round(np.mean(scores), 2) if scores else 50

def restaurant_risk_breakdown(df_restaurant):
    breakdown = {"safe": 0, "caution": 0, "unsafe": 0}
    for _, row in df_restaurant.iterrows():
        rule_status, _ = rule_based_safety(row)
        breakdown[rule_status] += 1
    return breakdown

# ML model
@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer(max_features=500)
    X_text = tfidf.fit_transform(df["full_ingredient_list"])
    X_allergens = df[allergy].values
    X = np.hstack([X_text.toarray(), X_allergens])
    y = df["allergy_safety_status"].map({"safe": 0, "caution": 1, "unsafe": 2})

    if len(y.unique()) < 2:
        #st.warning("Warning: Not enough diverse data to train the ML model. Defaulting to rule-based predictions.")

        class DummyModel:
            def predict(self, X):
                return np.full(X.shape[0], y.iloc[0] if not y.empty else 1)
        return DummyModel(), tfidf, 0.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, tfidf, acc

model, vectorizer, model_acc = train_model(df)

# Search UI
st.markdown("Search menu")
col1, col2 = st.columns(2)
with col1:
    restaurant = st.selectbox("Select Restaurant", sorted(df["restaurant_name"].unique()))
with col2:
    query = st.text_input("Search dish (optional)")

menu_df = df[df["restaurant_name"] == restaurant]
if query:
    menu_df = menu_df[menu_df["food_name"].str.contains(query, case=False)]

restaurant_score = compute_restaurant_safety(menu_df, allergy_severity)
st.markdown(f"Overall safety score for {restaurant}: {restaurant_score}")
breakdown = restaurant_risk_breakdown(menu_df)
st.markdown("Breakdown of risk:")
st.write(breakdown)

# Display results
st.markdown(f"{restaurant} Menu")
for _, row in menu_df.iterrows():
    rule_status, explanation = rule_based_safety(row)
    final_status = rule_status
    if use_ml:
        text_vec = vectorizer.transform([row["full_ingredient_list"]])
        allergen_vec = row[allergy].values.reshape(1, -1)
        X_input = np.hstack([text_vec.toarray(), allergen_vec])
        ml_pred = model.predict(X_input)[0]
        ml_label = {0: "safe", 1: "caution", 2: "unsafe"}[ml_pred]
        if rule_status == "caution" and ml_label == "safe":
            final_status = "safe"
        elif rule_status == "safe" and ml_label == "unsafe":
            final_status = "caution"
        else:
            final_status = rule_status

    safety_score = compute_safety_score(final_status, explanation, allergy_severity)
    if final_status == "unsafe":
        st.error(f"{row['food_name']} - Safety score: {safety_score}/100")
    elif final_status == "caution":
        st.warning(f"{row['food_name']} - Safety Score: {safety_score}/100")
    else:
        st.success(f"{row['food_name']} - Safety Score: {safety_score}/100")

    with st.expander("Why this result?"):
        for reason in explanation:
            st.write(f"- {reason}")

# Footer
st.markdown("---")


from pyngrok import ngrok
import subprocess
import os

#kill any running ngrok tunnels to free up port 8501
os.system("kill $(lsof -t -i:8501)")

#authenticate ngrok
from google.colab import userdata
NGROK_AUTH_TOKEN = userdata.get('NGROK_AUTH_TOKEN') #get token form https://dashboard.ngrok.com/
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

#start streamlit in the background
!streamlit run streamlit_app.py &>/dev/null &

import time
time.sleep(5)

#set up tunnel to port 8501 (streamlit default)
public_url = ngrok.connect(addr='8501')
print("Streamlit app URL:", public_url)














