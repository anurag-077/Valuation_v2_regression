# cluster_regression_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import os

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
EXCEL_FILE = "All_Project_data_WITH_Amenity_Scores.xlsx"
AMENITY_DIR = "amenities"
POI_SEARCH_RADIUS_M = 1000

DEFAULT_WEIGHTS = {
    "Metro": 0.25, "Bus": 0.15, "Mall": 0.225,
    "School": 0.225, "Hospital": 0.075, "Garden": 0.075
}

AMENITY_TO_CATEGORY = {
    "subway_entrance": "Metro", "metro_station": "Metro", "railway=station": "Metro",
    "bus_stop": "Bus", "bus_station": "Bus", "public_transport=stop_position": "Bus",
    "public_transport=platform": "Bus",
    "mall": "Mall", "department_store": "Mall", "supermarket": "Mall",
    "convenience": "Mall", "marketplace": "Mall", "malls": "Mall",
    "school": "School", "schools": "School", "college": "School", "university": "School",
    "hospital": "Hospital", "hospitals": "Hospital", "clinic": "Hospital",
    "doctors": "Hospital", "pharmacy": "Hospital",
    "park": "Garden", "gardens": "Garden", "playground": "Garden",
    "sports_centre": "Garden", "pitch": "Garden"
}

# --------------------------------------------------------------
# UTILS
# --------------------------------------------------------------
def haversine_vectorized(lat1, lon1, lats2, lons2):
    import numpy as np
    lat1, lon1, lats2, lons2 = map(np.radians, [lat1, lon1, lats2, lons2])
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return 6371000 * c

def decay(d):
    return 1.0 / (1.0 + min(d, POI_SEARCH_RADIUS_M) / 200.0)

# --------------------------------------------------------------
# Load Amenities
# --------------------------------------------------------------
@st.cache_data
def load_amenities():
    if not os.path.exists(AMENITY_DIR):
        return pd.DataFrame()
    frames = []
    for file in os.listdir(AMENITY_DIR):
        if not file.lower().endswith(".xlsx"):
            continue
        key = file[:-5].lower()
        category = AMENITY_TO_CATEGORY.get(key)
        if not category:
            continue
        try:
            df = pd.read_excel(os.path.join(AMENITY_DIR, file))
            df = df.dropna(subset=["lat", "lng"])
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
            df = df.dropna(subset=["lat", "lng"])
            if df.empty:
                continue
            df["category"] = category
            frames.append(df[["lat", "lng", "category"]])
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# --------------------------------------------------------------
# Amenity Score
# --------------------------------------------------------------
def compute_amenity_score(lat, lng, amenities, weights):
    if amenities.empty:
        return 0.0
    total = 0.0
    for cat, w in weights.items():
        cat_df = amenities[amenities["category"] == cat]
        if cat_df.empty:
            continue
        dists = haversine_vectorized(lat, lng, cat_df["lat"].values, cat_df["lng"].values)
        mask = dists <= POI_SEARCH_RADIUS_M
        if not mask.any():
            continue
        decays = [decay(d) for d in dists[mask]]
        decays.sort(reverse=True)
        top3 = sum(decays[:3])
        total += w * min(1.0, top3)
    return round(total, 3)

# --------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------
st.set_page_config(page_title="Cluster Editor & Regression", layout="wide")
st.title("Cluster Editor & Regression Trainer")
st.caption("3-decimal scores | DB default | Recalculates on weight change | **No NaN crash**")

if not os.path.exists(EXCEL_FILE):
    st.error(f"`{EXCEL_FILE}` not found!")
    st.stop()

with st.spinner("Loading data..."):
    project_df = pd.read_excel(EXCEL_FILE)

# Ensure Project_ID
if 'Project_ID' not in project_df.columns:
    project_df['Project_ID'] = [f"P{i:04d}" for i in range(1, len(project_df)+1)]

required = ['Project_Name', 'Mid_Rate', 'project_lat', 'project_lng',
            'Cluster_LatLong', 'Cluster_LatLongCategory']
for c in required:
    if c not in project_df.columns:
        st.error(f"Missing: `{c}`")
        st.stop()

# Default columns
defaults = [('Road_Category','B'), ('total fsi (sqmtr)',1000.0),
            ('Age_Of_The_Building_Till_11thOct2025',5)]
for c, d in defaults:
    if c not in project_df.columns:
        project_df[c] = d

# Use DB amenity score
project_df['amenity_score'] = project_df.get('amenity_score',
                                            project_df.get('Amenity_Raw_R_0_1', 0.0))

# --------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------
with st.sidebar:
    st.header("Amenity Weights")
    weights = {}
    total_w = 0
    for cat, val in DEFAULT_WEIGHTS.items():
        w = st.number_input(cat, 0.0, 1.0, val, step=0.01, key=f"wt_{cat}")
        weights[cat] = w
        total_w += w
    st.metric("Total Weight", f"{total_w:.2f}")
    weights_changed = any(abs(w - DEFAULT_WEIGHTS[cat]) > 0.001 for cat, w in weights.items())

    st.header("Regression Features")
    feature_options = {
        "Amenity Score": "amenity_score",
        "Road Category": "road_numeric",
        "Total FSI": "total fsi (sqmtr)",
        "Age": "Age_Of_The_Building_Till_11thOct2025"
    }
    selected_features = st.multiselect(
        "Select features", list(feature_options.keys()),
        default=["Amenity Score", "Road Category"]
    )

# --------------------------------------------------------------
# CLUSTER SELECTION
# --------------------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    cluster_type = st.selectbox("Cluster Type", ['Cluster_LatLong', 'Cluster_LatLongCategory'])
with col2:
    clusters = sorted(project_df[cluster_type].dropna().unique())
    selected_cluster = st.selectbox("Select Cluster", clusters)

cluster_df = project_df[project_df[cluster_type] == selected_cluster].copy().reset_index(drop=True)

# --------------------------------------------------------------
# RECALCULATE SCORES
# --------------------------------------------------------------
all_amenities = load_amenities()
if weights_changed and not all_amenities.empty:
    st.info("Recalculating scores...")
    scores = [
        compute_amenity_score(row['project_lat'], row['project_lng'], all_amenities, weights)
        for _, row in cluster_df.iterrows()
    ]
    cluster_df['amenity_score'] = scores
    st.success("Scores updated!")
else:
    st.info("Using **DB scores**")

# Road numeric
road_map = {'A':1, 'B':2, 'C':3, 'D':4}
cluster_df['road_numeric'] = cluster_df['Road_Category'].map(road_map).fillna(2)

# --------------------------------------------------------------
# EDITABLE TABLE
# --------------------------------------------------------------
st.subheader(f"Projects in `{selected_cluster}`")

cluster_df['amenity_numeric'] = pd.to_numeric(cluster_df['amenity_score'], errors='coerce').fillna(0.0)
cluster_df['amenity_display'] = cluster_df['amenity_numeric'].apply(lambda x: f"{x:.3f}")

display_df = cluster_df[[
    'Project_ID', 'Project_Name', 'Mid_Rate', 'Road_Category',
    'amenity_display', 'total fsi (sqmtr)', 'Age_Of_The_Building_Till_11thOct2025'
]].copy()

display_df.columns = [
    'Project ID', 'Project Name', 'Rate (₹/sqft)', 'Road Type',
    'Amenity Score', 'Total FSI', 'Age (Years)'
]

edited_df = st.data_editor(
    display_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Project ID": st.column_config.TextColumn(disabled=True),
        "Amenity Score": st.column_config.TextColumn(disabled=True),
        "Road Type": st.column_config.SelectboxColumn(options=['A','B','C','D']),
        "Rate (₹/sqft)": st.column_config.NumberColumn(format="₹%.0f"),
        "Total FSI": st.column_config.NumberColumn(format="%.0f"),
        "Age (Years)": st.column_config.NumberColumn(format="%.1f")
    },
    hide_index=False
)

# SYNC EDITS + CLEAN NaN
cluster_df['Mid_Rate'] = pd.to_numeric(edited_df['Rate (₹/sqft)'], errors='coerce')
cluster_df['Road_Category'] = edited_df['Road Type']
cluster_df['total fsi (sqmtr)'] = pd.to_numeric(edited_df['Total FSI'], errors='coerce')
cluster_df['Age_Of_The_Building_Till_11thOct2025'] = pd.to_numeric(edited_df['Age (Years)'], errors='coerce')
cluster_df['road_numeric'] = cluster_df['Road_Category'].map(road_map).fillna(2)
cluster_df['amenity_score'] = cluster_df['amenity_numeric']

# --------------------------------------------------------------
# TRAIN MODEL – NO NaN ALLOWED
# --------------------------------------------------------------
if st.button("Train Model", type="primary", use_container_width=True):
    if not selected_features:
        st.error("Select at least one feature.")
    else:
        # Build feature matrix
        X_cols = [feature_options[f] for f in selected_features]
        X = cluster_df[X_cols].copy()
        y = cluster_df['Mid_Rate'].copy()

        # CRITICAL: Drop rows with ANY NaN
        before = len(X)
        combined = pd.concat([X, y], axis=1)
        combined = combined.dropna()
        after = len(combined)
        X_clean = combined[X_cols]
        y_clean = combined['Mid_Rate']

        if after == 0:
            st.error("No valid data after removing NaN. Check your inputs.")
            st.stop()

        if after < before:
            st.warning(f"Dropped {before - after} rows with missing values.")

        # Train
        model = LinearRegression()
        model.fit(X_clean.values, y_clean.values)
        y_pred = model.predict(X_clean.values)
        r2 = r2_score(y_clean, y_pred)

        # Equation
        terms = [f"{c:.2f}×{f}" for c, f in zip(model.coef_, selected_features)]
        eq = "Rate = " + " + ".join(terms)
        eq += f" + {model.intercept_:.0f}" if model.intercept_ >= 0 else f" – {abs(model.intercept_):.0f}"

        st.success(f"Model trained! R² = {r2:.4f}")
        col1, col2 = st.columns(2)
        col1.metric("R²", f"{r2:.4f}")
        col2.code(eq)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_clean, y=y_pred, mode='markers',
                                 text=combined['Project_ID'],
                                 hovertemplate='<b>%{text}</b><br>Actual: ₹%{x:,.0f}<br>Pred: ₹%{y:,.0f}'))
        fig.add_trace(go.Scatter(x=[y_clean.min(), y_clean.max()], y=[y_clean.min(), y_clean.max()],
                                 mode='lines', line=dict(dash='dash', color='red')))
        fig.update_layout(title, xaxis_title="Actual", yaxis_title="Predicted", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.session_state['model'] = {'model': model, 'features': X_cols, 'eq': eq, 'r2': r2}

# --------------------------------------------------------------
# PREDICT NEW
# --------------------------------------------------------------
with st.expander("Predict New Project"):
    if 'model' not in st.session_state:
        st.info("Train model first.")
    else:
        inputs = {}
        for f in st.session_state['model']['features']:
            label = [k for k, v in feature_options.items() if v == f][0]
            if f == 'amenity_score':
                inputs[f] = st.slider(label, 0.0, 1.0, 0.6, 0.01)
            elif f == 'road_numeric':
                rd = st.selectbox(label, ['A','B','C','D'])
                inputs[f] = {'A':1,'B':2,'C':3,'D':4}[rd]
            elif 'fsi' in f.lower():
                inputs[f] = st.number_input(label, 0.0, 20000.0, 2500.0)
            else:
                inputs[f] = st.number_input(label, 0.0, 50.0, 5.0)

        if st.button("Predict"):
            X_new = np.array([[inputs[c] for c in st.session_state['model']['features']]])
            pred = st.session_state['model']['model'].predict(X_new)[0]
            st.success(f"**₹{pred:,.0f}/sqft**")
            st.caption(st.session_state['model']['eq'])