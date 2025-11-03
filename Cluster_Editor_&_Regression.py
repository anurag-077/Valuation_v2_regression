# cluster_regression_app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
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
    "Metro": 0.25, "Bus": 0.15, "Mall": 0.23,
    "School": 0.23, "Hospital": 0.07, "Garden": 0.07
}

DEFAULT_RATE_RANGES = {
    "Affordable": (0, 7000),
    "Mid-Segment": (7000, 13000),
    "Luxury": (13000, float('inf'))
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
    lat1, lon1, lats2, lons2 = map(np.radians, [lat1, lon1, lats2, lons2])
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return 6371000 * c

def decay(d):
    return 1.0 / (1.0 + min(d, POI_SEARCH_RADIUS_M) / 200.0)

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
        S_c = sum(decays)
        s_c = 1 - math.exp(-0.8 * S_c)
        total += w * s_c
    return round(total, 3)

def categorize_rate(rate, ranges):
    for cat, (low, high) in ranges.items():
        if low <= rate < high:
            return cat
    return "Luxury"

# --------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------
st.set_page_config(page_title="Cluster Editor & Regression", layout="wide")
st.title("Cluster Editor & Regression Trainer")
st.caption("**Instant edits** | **Train on selected rows only** | **Rate on Salable Area**")

# === LOAD DATA ONCE ===
if 'project_df' not in st.session_state:
    if not os.path.exists(EXCEL_FILE):
        st.error(f"`{EXCEL_FILE}` not found!")
        st.stop()
    with st.spinner("Loading data..."):
        df = pd.read_excel(EXCEL_FILE)
        if 'Project_ID' not in df.columns:
            df['Project_ID'] = [f"P{i:04d}" for i in range(1, len(df)+1)]
        required = ['Project_Name', 'Mid_Rate', 'project_lat', 'project_lng',
                    'Cluster_LatLong', 'Cluster_LatLongCategory']
        for c in required:
            if c not in df.columns:
                st.error(f"Missing: `{c}`")
                st.stop()
        defaults = [('Road_Category','B'), ('total fsi (sqmtr)',1000.0),
                    ('Age_Of_The_Building_Till_11thOct2025',5)]
        for c, d in defaults:
            if c not in df.columns:
                df[c] = d
        df['amenity_score'] = df.get('amenity_score', df.get('Amenity_Raw_R_0_1', 0.0))
        st.session_state.project_df = df

project_df = st.session_state.project_df

# --------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------
with st.sidebar:
    st.header("Amenity Weights")
    if 'custom_weights' not in st.session_state:
        st.session_state.custom_weights = DEFAULT_WEIGHTS.copy()
    if 'weights_applied' not in st.session_state:
        st.session_state.weights_applied = False

    weights = {}
    for cat, default_val in DEFAULT_WEIGHTS.items():
        current_val = st.session_state.custom_weights.get(cat, default_val)
        w = st.number_input(cat, 0.0, 1.0, current_val, 0.01, key=f"wt_{cat}")
        weights[cat] = w
    st.metric("Total Weight", f"{sum(weights.values()):.2f}")

    st.markdown("---")
    st.subheader("Rate Category Ranges (₹/sqft on Salable Area)")

    if 'rate_ranges' not in st.session_state:
        st.session_state.rate_ranges = DEFAULT_RATE_RANGES.copy()

    r1 = st.number_input("Affordable: Up to", value=st.session_state.rate_ranges["Affordable"][1], step=500, key="r1")
    r2 = st.number_input("Mid-Segment: From", value=st.session_state.rate_ranges["Mid-Segment"][0], step=500, key="r2")
    r3 = st.number_input("Mid-Segment: Up to", value=st.session_state.rate_ranges["Mid-Segment"][1], step=500, key="r3")
    r4 = st.number_input("Luxury: Above", value=st.session_state.rate_ranges["Luxury"][0], step=500, key="r4")

    updated_ranges = {
        "Affordable": (0, r1),
        "Mid-Segment": (r2, r3),
        "Luxury": (r4, float('inf'))
    }
    st.session_state.rate_ranges = updated_ranges

    st.caption(f"**Current:**\n• Affordable: < ₹{r1:,}\n• Mid-Segment: ₹{r2:,} – ₹{r3:,}\n• Luxury: > ₹{r4:,}")

    col1, col2 = st.columns(2)
    with col1:
        apply_btn = st.button("Apply Weights", type="primary")
    with col2:
        reset_btn = st.button("Reset", type="secondary")

    if reset_btn:
        st.session_state.custom_weights = DEFAULT_WEIGHTS.copy()
        st.session_state.rate_ranges = DEFAULT_RATE_RANGES.copy()
        st.session_state.weights_applied = False
        # Fully clear recalc & cluster state
        for key in list(st.session_state.keys()):
            if key.startswith("recalc_") or key.startswith("cluster_"):
                if key in st.session_state:
                    del st.session_state[key]
        st.success("Reset! Reverting to DB scores.")
        st.rerun()

    if apply_btn:
        st.session_state.custom_weights = weights.copy()
        st.session_state.weights_applied = True
        st.success("Weights applied! Recalculating scores...")
        # FORCE RECALC EVERY TIME
        for key in list(st.session_state.keys()):
            if key.startswith("recalc_"):
                if key in st.session_state:
                    del st.session_state[key]
        st.rerun()

    active_weights = st.session_state.custom_weights
    rate_ranges = st.session_state.rate_ranges

    st.header("Regression Features")
    feature_options = {
        "Amenity Score": "amenity_score",
        "Road Category": "road_numeric",
        "Total FSI": "total fsi (sqmtr)",
        "Age": "Age_Of_The_Building_Till_11thOct2025",
        "Project Category": "category_encoded"
    }
    selected_features = st.multiselect(
        "Select features", list(feature_options.keys()),
        default=["Amenity Score", "Road Category"],
        key="feat_sel"
    )

# --------------------------------------------------------------
# CLUSTER SELECTION (PERSISTENT)
# --------------------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    cluster_type = st.selectbox("Cluster Type", ['Cluster_LatLong', 'Cluster_LatLongCategory'], key="ctype")
with col2:
    clusters = sorted(project_df[cluster_type].dropna().unique())
    # Persist selection
    last_key = f"last_cluster_{cluster_type}"
    if last_key not in st.session_state:
        st.session_state[last_key] = clusters[0]
    selected_cluster = st.selectbox("Select Cluster", clusters, 
                                    index=clusters.index(st.session_state[last_key]) if st.session_state[last_key] in clusters else 0,
                                    key="cluster_sel")
st.session_state[last_key] = selected_cluster

# === ISOLATE CLUSTER ===
cluster_key = f"cluster_{cluster_type}_{selected_cluster}"
if cluster_key not in st.session_state:
    cluster_df = project_df[project_df[cluster_type] == selected_cluster].copy().reset_index(drop=True)
    cluster_df['_original_amenity'] = cluster_df['amenity_score']
    st.session_state[cluster_key] = cluster_df
else:
    cluster_df = st.session_state[cluster_key]

# --------------------------------------------------------------
# RECALCULATE AMENITY SCORES
# --------------------------------------------------------------
all_amenities = load_amenities()
recalc_key = f"recalc_{cluster_key}"

if st.session_state.weights_applied and not all_amenities.empty:
    if recalc_key not in st.session_state:
        with st.spinner("Recalculating amenity scores..."):
            scores = [
                compute_amenity_score(row['project_lat'], row['project_lng'], all_amenities, active_weights)
                for _, row in cluster_df.iterrows()
            ]
            cluster_df['amenity_score'] = scores
            st.session_state[recalc_key] = True
        st.success("Scores updated!")
else:
    # Revert to original if not applied
    if not st.session_state.weights_applied and recalc_key in st.session_state:
        cluster_df['amenity_score'] = cluster_df['_original_amenity']
        del st.session_state[recalc_key]
    st.info("Using DB scores")

road_map = {'A':1, 'B':2, 'C':3, 'D':4}
cluster_df['road_numeric'] = cluster_df['Road_Category'].map(road_map).fillna(2)

# --------------------------------------------------------------
# PREPARE DISPLAY + EDIT TABLE
# --------------------------------------------------------------
cluster_df['Rate_on_Salable'] = pd.to_numeric(cluster_df['Mid_Rate'], errors='coerce').fillna(0)
cluster_df['Category'] = cluster_df['Rate_on_Salable'].apply(lambda x: categorize_rate(x, rate_ranges))
cluster_df['Category_Affordable'] = (cluster_df['Category'] == 'Affordable').astype(int)
cluster_df['Category_MidSegment'] = (cluster_df['Category'] == 'Mid-Segment').astype(int)

cluster_df['amenity_numeric'] = pd.to_numeric(cluster_df['amenity_score'], errors='coerce').fillna(0.0)
cluster_df['amenity_display'] = cluster_df['amenity_numeric'].apply(lambda x: f"{x:.3f}")

display_df = cluster_df[[
    'Project_ID', 'Project_Name', 'Rate_on_Salable', 'Road_Category',
    'amenity_display', 'total fsi (sqmtr)', 'Age_Of_The_Building_Till_11thOct2025', 'Category'
]].copy()

display_df.columns = [
    'Project ID', 'Project Name', 'Rate (₹/sqft on Salable Area)', 'Road Type',
    'Amenity Score', 'Total FSI', 'Age (Years)', 'Category'
]

# ADD SELECT COLUMN
display_df['Select'] = display_df.index.isin(st.session_state.get(f"selected_{cluster_key}", []))

st.subheader(f"Projects in `{selected_cluster}` | **Check to train**")

edited_df = st.data_editor(
    display_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Project ID": st.column_config.TextColumn(disabled=True),
        "Amenity Score": st.column_config.TextColumn(disabled=True),
        "Category": st.column_config.TextColumn(disabled=True),
        "Road Type": st.column_config.SelectboxColumn(options=['A','B','C','D']),
        "Rate (₹/sqft on Salable Area)": st.column_config.NumberColumn(format="₹%.0f"),
        "Total FSI": st.column_config.NumberColumn(format="%.0f"),
        "Age (Years)": st.column_config.NumberColumn(format="%.1f"),
        "Select": st.column_config.CheckboxColumn("Select", default=False)
    },
    hide_index=False,
    key=f"editor_{cluster_key}"
)

# === INSTANT SYNC ===
cluster_df.loc[edited_df.index, 'Rate_on_Salable'] = pd.to_numeric(edited_df['Rate (₹/sqft on Salable Area)'], errors='coerce')
cluster_df.loc[edited_df.index, 'Road_Category'] = edited_df['Road Type']
cluster_df.loc[edited_df.index, 'total fsi (sqmtr)'] = pd.to_numeric(edited_df['Total FSI'], errors='coerce')
cluster_df.loc[edited_df.index, 'Age_Of_The_Building_Till_11thOct2025'] = pd.to_numeric(edited_df['Age (Years)'], errors='coerce')
cluster_df['road_numeric'] = cluster_df['Road_Category'].map(road_map).fillna(2)

# Recompute Category
cluster_df['Category'] = cluster_df['Rate_on_Salable'].apply(lambda x: categorize_rate(x, rate_ranges))
cluster_df['Category_Affordable'] = (cluster_df['Category'] == 'Affordable').astype(int)
cluster_df['Category_MidSegment'] = (cluster_df['Category'] == 'Mid-Segment').astype(int)
cluster_df['Mid_Rate'] = cluster_df['Rate_on_Salable']

# Save selected
selected_mask = edited_df['Select']
st.session_state[f"selected_{cluster_key}"] = edited_df[selected_mask].index.tolist()
train_df = cluster_df.loc[st.session_state[f"selected_{cluster_key}"]].copy()

if len(train_df) == 0:
    st.warning("**No rows selected. Check 'Select' to train.**")
else:
    st.success(f"**{len(train_df)} projects selected**")

# === TRAIN ===
if st.button("Train Model on Selected Rows", type="primary", use_container_width=True):
    if not selected_features:
        st.error("Select at least one feature.")
    elif len(train_df) < 2:
        st.error("Need at least 2 selected projects.")
    else:
        X_cols = []
        display_names = []
        for f in selected_features:
            if f == "Project Category":
                X_cols.extend(['Category_Affordable', 'Category_MidSegment'])
                display_names.extend(['Affordable', 'Mid-Segment'])
            else:
                X_cols.append(feature_options[f])
                display_names.append(f)

        X = train_df[X_cols].copy()
        y = train_df['Mid_Rate'].copy()

        combined = pd.concat([X, y], axis=1).dropna()
        if len(combined) < 2:
            st.error("Not enough valid data.")
            st.stop()

        X_clean, y_clean = combined[X_cols], combined['Mid_Rate']

        model = LinearRegression()
        model.fit(X_clean.values, y_clean.values)
        y_pred = model.predict(X_clean.values)
        r2 = r2_score(y_clean, y_pred)

        terms = []
        for i, name in enumerate(display_names):
            coef = model.coef_[i]
            if name in ['Affordable', 'Mid-Segment']:
                terms.append(f"{coef:+.0f} if {name}")
            else:
                terms.append(f"{coef:.2f}×{name}")
        eq = "Rate = " + " + ".join(terms)
        eq += f" + {model.intercept_:.0f}" if model.intercept_ >= 0 else f" – {abs(model.intercept_):.0f}"

        st.success(f"Trained on **{len(X_clean)} projects**! R² = {r2:.4f}")
        col1, col2 = st.columns(2)
        col1.metric("R²", f"{r2:.4f}")
        col2.code(eq)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_clean, y=y_pred, mode='markers',
                                 text=train_df.loc[y_clean.index, 'Project_ID'],
                                 hovertemplate='<b>%{text}</b><br>Actual: ₹%{x:,.0f}<br>Pred: ₹%{y:,.0f}'))
        fig.add_trace(go.Scatter(x=[y_clean.min(), y_clean.max()], y=[y_clean.min(), y_clean.max()],
                                 mode='lines', line=dict(dash='dash', color='red')))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.session_state['model'] = {
            'model': model,
            'features': X_cols,
            'display_features': selected_features,
            'eq': eq,
            'r2': r2
        }

# --------------------------------------------------------------
# PREDICT NEW
# --------------------------------------------------------------
with st.expander("Predict New Project"):
    if 'model' not in st.session_state:
        st.info("Train model first.")
    else:
        inputs = {}
        for f in st.session_state['model']['display_features']:
            if f == "Project Category":
                cat = st.selectbox("Project Category", ["Affordable", "Mid-Segment", "Luxury"], key="pred_cat")
                inputs['Category_Affordable'] = 1 if cat == "Affordable" else 0
                inputs['Category_MidSegment'] = 1 if cat == "Mid-Segment" else 0
            else:
                col = feature_options[f]
                if col == 'amenity_score':
                    inputs[col] = st.slider(f, 0.0, 1.0, 0.6, 0.01, key=f"pred_{col}")
                elif col == 'road_numeric':
                    rd = st.selectbox(f, ['A','B','C','D'], key=f"pred_{col}")
                    inputs[col] = {'A':1,'B':2,'C':3,'D':4}[rd]
                elif 'fsi' in col.lower():
                    inputs[col] = st.number_input(f, 0.0, 20000.0, 2500.0, key=f"pred_{col}")
                else:
                    inputs[col] = st.number_input(f, 0.0, 50.0, 5.0, key=f"pred_{col}")

        if st.button("Predict", type="primary"):
            X_new = np.array([[inputs.get(c, 0) for c in st.session_state['model']['features']]])
            pred = st.session_state['model']['model'].predict(X_new)[0]
            pred_cat = categorize_rate(pred, rate_ranges)
            st.success(f"**₹{pred:,.0f}/sqft on Salable Area**")
            st.caption(f"**Category:** {pred_cat}")
            st.caption(st.session_state['model']['eq'])