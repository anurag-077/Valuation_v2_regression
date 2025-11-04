# cluster_regression_app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
import os
from typing import List, Dict

# ==============================================================
# CONFIG
# ==============================================================
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
ORDINAL_CATEGORY_MAP = {"Affordable": 1, "Mid-Segment": 2, "Luxury": 3}
ROAD_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

# ==============================================================
# CBD MASTER LIST (same as cbd_score_nearest_routes.py)
# ==============================================================
CBD_MASTER = [
    {"name": "Shivajinagar", "lat": 18.5305, "lng": 73.8472, "area": "Pune"},
    {"name": "Camp", "lat": 18.5127, "lng": 73.8795, "area": "Pune"},
    {"name": "Koregaon Park", "lat": 18.5377, "lng": 73.8855, "area": "Pune"},
    {"name": "Pimpri", "lat": 18.6275, "lng": 73.8060, "area": "PCMC"},
    {"name": "Akurdi", "lat": 18.6427, "lng": 73.7585, "area": "PCMC"},
    {"name": "Chinchwad", "lat": 18.6297, "lng": 73.7850, "area": "PCMC"},
]

OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

# ==============================================================
# CBD SCORE HELPERS (exact copy of the logic in the reference file)
# ==============================================================
def get_fastest_route(start_lng, start_lat, end_lng, end_lat) -> Dict:
    """Return the fastest OSRM route (or empty dict on error)."""
    url = f"{OSRM_URL}{start_lng},{start_lat};{end_lng},{end_lat}"
    params = {"overview": "false", "alternatives": "false", "steps": "false"}
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code == 200:
            data = r.json()
            routes = data.get("routes", [])
            return routes[0] if routes else {}
    except Exception:
        pass
    return {}

def calculate_cbd_score(dist_km: float, time_min: float) -> float:
    """Same formula as in cbd_score_nearest_routes.py."""
    score_dist = max(0.6, 1 / (1 + dist_km / 10))
    score_time = max(0.6, 1 / (1 + time_min / 30))
    return round(0.6 * score_dist + 0.4 * score_time, 3)

@st.cache_data(show_spinner=False)
def cbd_score_for_project(lat: float, lng: float) -> float:
    """Fastest route to the best CBD → final score."""
    best = 0.0
    for cbd in CBD_MASTER:
        route = get_fastest_route(lng, lat, cbd["lng"], cbd["lat"])
        if not route:
            continue
        dist_km = route["distance"] / 1000
        time_min = route["duration"] / 60
        score = calculate_cbd_score(dist_km, time_min)
        if score > best:
            best = score
    return best

# ==============================================================
# UTILS (amenity, haversine, etc.)
# ==============================================================
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
        cat = AMENITY_TO_CATEGORY.get(key)
        if not cat:
            continue
        try:
            df = pd.read_excel(os.path.join(AMENITY_DIR, file))
            df = df.dropna(subset=["lat", "lng"])
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
            df = df.dropna(subset=["lat", "lng"])
            if df.empty:
                continue
            df["category"] = cat
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

# ==============================================================
# MAP (selected cluster only)
# ==============================================================
def plot_selected_cluster_map(df, cluster_col, cluster_val):
    filtered = df[df[cluster_col] == cluster_val].copy()
    if filtered.empty:
        st.warning(f"No projects in {cluster_col} = {cluster_val}")
        return None

    if 'Village' in filtered.columns:
        filtered['hover_text'] = filtered.apply(
            lambda r: f"<b>{r['Project_Name']}</b><br>₹{r['Mid_Rate']:.0f}/sqft<br>{r['Village']}", axis=1
        )
    else:
        filtered['hover_text'] = filtered.apply(
            lambda r: f"<b>{r['Project_Name']}</b><br>₹{r['Mid_Rate']:.0f}/sqft", axis=1
        )

    fig = px.scatter_mapbox(
        filtered,
        lat='project_lat', lon='project_lng',
        hover_name='hover_text',
        color='Mid_Rate',
        color_continuous_scale='viridis',
        zoom=13,
        height=550,
        title=f"Cluster: {cluster_val} ({len(filtered)} projects)",
        labels={'Mid_Rate': 'Rate (₹/sqft)'}
    )
    fig.update_traces(marker=dict(size=16, opacity=0.9))

    points = filtered[['project_lng', 'project_lat']].values
    if len(points) >= 3:
        try:
            hull = ConvexHull(points, qhull_options="QJ")
            verts = hull.vertices
            lons = points[verts, 0].tolist() + [points[verts[0], 0]]
            lats = points[verts, 1].tolist() + [points[verts[0], 1]]
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats,
                mode='lines',
                line=dict(width=3, color='red'),
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                name='Cluster Boundary',
                hoverinfo='skip'
            ))
        except Exception:
            st.warning("Could not draw cluster boundary (insufficient variation).")

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(t=80, l=10, r=10, b=10),
        legend=dict(title="Rate (₹/sqft)", orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.9)"),
        title=dict(x=0.5, xanchor="center", font=dict(size=18)),
        dragmode='zoom',
        uirevision='map'
    )
    fig.add_annotation(
        text="Red polygon = cluster boundary | Points coloured by rate",
        xref="paper", yref="paper", x=0.01, y=0.01,
        showarrow=False, font=dict(size=11, color="gray"),
        bgcolor="white", bordercolor="gray", borderwidth=1
    )
    return fig

# ==============================================================
# APP LAYOUT
# ==============================================================
st.set_page_config(page_title="Cluster Editor & Regression", layout="wide", initial_sidebar_state="expanded")
st.title("Cluster Editor & Regression Trainer")
st.caption("Instant edits | Train on selected rows only | Rate on Salable Area")

# --------------------- LOAD DATA ---------------------
if 'project_df' not in st.session_state:
    if not os.path.exists(EXCEL_FILE):
        st.error(f"`{EXCEL_FILE}` not found!")
        st.stop()
    with st.spinner("Loading data…"):
        df = pd.read_excel(EXCEL_FILE)
        if 'Project_ID' not in df.columns:
            df['Project_ID'] = [f"P{i:04d}" for i in range(1, len(df)+1)]
        required = ['Project_Name', 'Mid_Rate', 'project_lat', 'project_lng',
                    'Cluster_LatLong', 'Cluster_LatLongCategory']
        for c in required:
            if c not in df.columns:
                st.error(f"Missing column: `{c}`")
                st.stop()
        defaults = [('Road_Category','B'), ('total fsi (sqmtr)',1000.0),
                    ('Age_Of_The_Building_Till_11thOct2025',5)]
        for c, d in defaults:
            if c not in df.columns:
                df[c] = d
        df['amenity_score'] = df.get('amenity_score', df.get('Amenity_Raw_R_0_1', 0.0))
        st.session_state.project_df = df

project_df = st.session_state.project_df

# --------------------- SIDEBAR ---------------------
with st.sidebar:
    st.header("Amenity Weights")
    if 'custom_weights' not in st.session_state:
        st.session_state.custom_weights = DEFAULT_WEIGHTS.copy()
    if 'weights_applied' not in st.session_state:
        st.session_state.weights_applied = False

    weights = {}
    for cat, default_val in DEFAULT_WEIGHTS.items():
        cur = st.session_state.custom_weights.get(cat, default_val)
        w = st.number_input(cat, 0.0, 1.0, cur, 0.01, key=f"wt_{cat}")
        weights[cat] = w
    st.metric("Total Weight", f"{sum(weights.values()):.2f}")

    st.markdown("---")
    st.subheader("Rate Category Ranges (₹/sqft on Salable Area)")
    if 'rate_ranges' not in st.session_state:
        st.session_state.rate_ranges = DEFAULT_RATE_RANGES.copy()

    r1 = st.number_input("Affordable: Up to", value=st.session_state.rate_ranges["Affordable"][1], step=500, key="r1")
    r3 = st.number_input("Mid-Segment: Up to", value=st.session_state.rate_ranges["Mid-Segment"][1], step=500, key="r3")
    updated_ranges = {"Affordable": (0, r1), "Mid-Segment": (r1, r3), "Luxury": (r3, float('inf'))}
    st.session_state.rate_ranges = updated_ranges
    st.caption(f"• Affordable: < ₹{r1:,}\n• Mid-Segment: ₹{r1:,} – ₹{r3:,}\n• Luxury: > ₹{r3:,}")

    c1, c2 = st.columns(2)
    with c1:
        apply_btn = st.button("Apply Weights", type="primary", use_container_width=True)
    with c2:
        reset_btn = st.button("Reset", type="secondary", use_container_width=True)

    if reset_btn:
        st.session_state.custom_weights = DEFAULT_WEIGHTS.copy()
        st.session_state.rate_ranges = DEFAULT_RATE_RANGES.copy()
        st.session_state.weights_applied = False
        for k in list(st.session_state.keys()):
            if k.startswith("recalc_") or k.startswith("cluster_"):
                st.session_state.pop(k, None)
        st.success("Reset! Reverting to DB scores.")
        st.rerun()

    if apply_btn:
        st.session_state.custom_weights = weights.copy()
        st.session_state.weights_applied = True
        st.success("Weights applied! Recalculating scores…")
        for k in list(st.session_state.keys()):
            if k.startswith("recalc_"):
                st.session_state.pop(k, None)
        st.rerun()

    active_weights = st.session_state.custom_weights
    rate_ranges = st.session_state.rate_ranges

    st.markdown("---")
    st.header("Regression Features")
    feature_options = {
        "Amenity Score": "amenity_score",
        "Road Category": "road_numeric",
        "Total FSI": "total fsi (sqmtr)",
        "Age": "Age_Of_The_Building_Till_11thOct2025",
        "Project Category (Ordinal)": "Category_Ordinal",
        "CBD Score": "cbd_score"                     # NEW FEATURE
    }
    selected_features = st.multiselect(
        "Select features", list(feature_options.keys()),
        default=["Amenity Score", "Road Category"], key="feat_sel"
    )
    with st.expander("Ordinal Encoding Reference"):
        st.write("1 = Affordable | 2 = Mid-Segment | 3 = Luxury")

# --------------------- CLUSTER SELECTION ---------------------
st.subheader("Cluster Selection")
c1, c2 = st.columns([1, 3])
with c1:
    cluster_type = st.selectbox("Cluster Type", ['Cluster_LatLong', 'Cluster_LatLongCategory'], key="ctype")
with c2:
    clusters = sorted(project_df[cluster_type].dropna().unique())
    last_key = f"last_cluster_{cluster_type}"
    if last_key not in st.session_state:
        st.session_state[last_key] = clusters[0]
    selected_cluster = st.selectbox(
        "Select Cluster", clusters,
        index=clusters.index(st.session_state[last_key]) if st.session_state[last_key] in clusters else 0,
        key="cluster_sel"
    )
st.session_state[last_key] = selected_cluster

# --------------------- ISOLATE CLUSTER ---------------------
cluster_key = f"cluster_{cluster_type}_{selected_cluster}"
if cluster_key not in st.session_state:
    cluster_df = project_df[project_df[cluster_type] == selected_cluster].copy().reset_index(drop=True)
    cluster_df['_original_amenity'] = cluster_df['amenity_score']
    st.session_state[cluster_key] = cluster_df
else:
    cluster_df = st.session_state[cluster_key]

# --------------------- RECALCULATE AMENITY SCORES ---------------------
all_amenities = load_amenities()
recalc_key = f"recalc_{cluster_key}"
if st.session_state.weights_applied and not all_amenities.empty:
    if recalc_key not in st.session_state:
        with st.spinner("Recalculating amenity scores…"):
            scores = [
                compute_amenity_score(row['project_lat'], row['project_lng'], all_amenities, active_weights)
                for _, row in cluster_df.iterrows()
            ]
            cluster_df['amenity_score'] = scores
            st.session_state[recalc_key] = True
        st.success("Scores updated!")
else:
    if not st.session_state.weights_applied and recalc_key in st.session_state:
        cluster_df['amenity_score'] = cluster_df['_original_amenity']
        del st.session_state[recalc_key]
    st.info("Using DB scores")

cluster_df['road_numeric'] = cluster_df['Road_Category'].map(ROAD_MAP).fillna(2)

# --------------------- CBD SCORE CALCULATION ---------------------
cbd_cache_key = f"cbd_{cluster_key}"
if cbd_cache_key not in st.session_state:
    with st.spinner("Calculating CBD scores for every project…"):
        cbd_scores = [
            cbd_score_for_project(row['project_lat'], row['project_lng'])
            for _, row in cluster_df.iterrows()
        ]
        cluster_df['cbd_score'] = cbd_scores
        st.session_state[cbd_cache_key] = True
else:
    # ensure column exists even if cached
    if 'cbd_score' not in cluster_df.columns:
        cluster_df['cbd_score'] = 0.0

# --------------------- MAP (FIRST) ---------------------
st.markdown("---")
st.subheader(f"Map – Cluster **{selected_cluster}**")
map_fig = plot_selected_cluster_map(project_df, cluster_type, selected_cluster)
if map_fig:
    st.plotly_chart(map_fig, use_container_width=True)
else:
    st.info("Map not available for this cluster.")

# --------------------- DISPLAY / EDIT TABLE (now with CBD Score) ---------------------
def build_display_df(df, ranges):
    df = df.copy()
    df['Rate_on_Salable'] = pd.to_numeric(df['Mid_Rate'], errors='coerce').fillna(0)
    df['Category'] = df['Rate_on_Salable'].apply(lambda x: categorize_rate(x, ranges))
    df['Category_Ordinal'] = df['Category'].map(ORDINAL_CATEGORY_MAP).fillna(3)

    df['amenity_display'] = pd.to_numeric(df['amenity_score'], errors='coerce').fillna(0).apply(lambda x: f"{x:.3f}")
    df['cbd_display'] = df['cbd_score'].apply(lambda x: f"{x:.3f}")

    disp = df[[
        'Project_ID', 'Project_Name', 'Rate_on_Salable', 'Road_Category',
        'amenity_display', 'cbd_display', 'total fsi (sqmtr)',
        'Age_Of_The_Building_Till_11thOct2025', 'Category'
    ]].copy()
    disp.columns = [
        'Project ID', 'Project Name', 'Rate (₹/sqft)', 'Road Type',
        'Amenity Score', 'CBD Score', 'Total FSI', 'Age (Years)', 'Category'
    ]
    disp.insert(0, 'Select', disp.index.isin(st.session_state.get(f"selected_{cluster_key}", [])))
    return disp

display_df = build_display_df(cluster_df, rate_ranges)

st.subheader(f"Projects in **{selected_cluster}** ({len(cluster_df)} total)")

# Bulk buttons
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("Select All", use_container_width=True):
        st.session_state[f"selected_{cluster_key}"] = list(range(len(cluster_df)))
        st.rerun()
with b2:
    if st.button("Deselect All", use_container_width=True):
        st.session_state[f"selected_{cluster_key}"] = []
        st.rerun()
with b3:
    if st.button("Refresh View", use_container_width=True):
        st.rerun()

# Editable table (CBD Score is read-only)
edited_df = st.data_editor(
    display_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Select": st.column_config.CheckboxColumn("Select", default=False,
            help="Check to include in training"),
        "Project ID": st.column_config.TextColumn(disabled=True, width="small"),
        "Project Name": st.column_config.TextColumn(disabled=True, width="medium"),
        "Rate (₹/sqft)": st.column_config.NumberColumn(format="₹%.0f", width="small"),
        "Road Type": st.column_config.SelectboxColumn(options=['A','B','C','D'], width="small"),
        "Amenity Score": st.column_config.TextColumn(disabled=True, width="small"),
        "CBD Score": st.column_config.TextColumn(disabled=True, width="small"),
        "Total FSI": st.column_config.NumberColumn(format="%.0f", width="small"),
        "Age (Years)": st.column_config.NumberColumn(format="%.1f", width="small"),
        "Category": st.column_config.TextColumn(disabled=True, width="small")
    },
    hide_index=True,
    key=f"editor_{cluster_key}"
)

# Sync edits instantly
cluster_df.loc[edited_df.index, 'Rate_on_Salable'] = pd.to_numeric(edited_df['Rate (₹/sqft)'], errors='coerce')
cluster_df.loc[edited_df.index, 'Road_Category'] = edited_df['Road Type']
cluster_df.loc[edited_df.index, 'total fsi (sqmtr)'] = pd.to_numeric(edited_df['Total FSI'], errors='coerce')
cluster_df.loc[edited_df.index, 'Age_Of_The_Building_Till_11thOct2025'] = pd.to_numeric(edited_df['Age (Years)'], errors='coerce')
cluster_df['road_numeric'] = cluster_df['Road_Category'].map(ROAD_MAP).fillna(2)

cluster_df['Category'] = cluster_df['Rate_on_Salable'].apply(lambda x: categorize_rate(x, rate_ranges))
cluster_df['Category_Ordinal'] = cluster_df['Category'].map(ORDINAL_CATEGORY_MAP).fillna(3)
cluster_df['Mid_Rate'] = cluster_df['Rate_on_Salable']

st.session_state[cluster_key] = cluster_df

selected_mask = edited_df['Select']
st.session_state[f"selected_{cluster_key}"] = edited_df[selected_mask].index.tolist()
train_df = cluster_df.loc[st.session_state[f"selected_{cluster_key}"]].copy()

st.caption("Edits sync instantly. Use bulk buttons for faster selection.")

# --------------------- SELECTED PROJECTS SUMMARY (now with CBD) ---------------------
if train_df.empty:
    st.warning("No rows selected – check the **Select** column to include projects.")
else:
    st.success(f"{len(train_df)} projects selected for training")
    with st.expander(f"View selected projects ({len(train_df)})", expanded=False):
        summary = train_df[['Project_ID', 'Project_Name', 'Rate_on_Salable']].copy()
        summary.columns = ['Project ID', 'Project Name', 'Rate (₹/sqft)']
        for f in selected_features:
            if f == "Amenity Score":
                summary["Amenity Score"] = train_df["amenity_score"].round(3)
            elif f == "Road Category":
                summary["Road Type"] = train_df["Road_Category"]
            elif f == "Total FSI":
                summary["Total FSI"] = train_df["total fsi (sqmtr)"]
            elif f == "Age":
                summary["Age (Years)"] = train_df["Age_Of_The_Building_Till_11thOct2025"]
            elif f == "Project Category (Ordinal)":
                summary["Category (Ordinal)"] = train_df["Category_Ordinal"]
            elif f == "CBD Score":
                summary["CBD Score"] = train_df["cbd_score"].round(3)
        summary["Category"] = summary["Rate (₹/sqft)"].apply(lambda x: categorize_rate(x, rate_ranges))
        base = ['Project ID', 'Project Name', 'Rate (₹/sqft)']
        feat_cols = [c for c in summary.columns if c not in base + ["Category"]]
        summary = summary[base + feat_cols + ["Category"]]
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(f"Features used: {', '.join(selected_features)}")

# --------------------- MODEL TRAINING ---------------------
st.markdown("---")
st.subheader("Model Training")
if st.button("Train Model on Selected Rows", type="primary", use_container_width=True):
    if not selected_features:
        st.error("Select at least one feature.")
    elif len(train_df) < 2:
        st.error("Need at least 2 selected projects.")
    else:
        X_cols = []
        disp_names = []
        for f in selected_features:
            if f == "Project Category (Ordinal)":
                X_cols.append("Category_Ordinal")
                disp_names.append("Category (1=Aff,2=Mid,3=Lux)")
            elif f == "CBD Score":
                X_cols.append("cbd_score")
                disp_names.append("CBD Score")
            else:
                X_cols.append(feature_options[f])
                disp_names.append(f)

        X = train_df[X_cols].copy()
        y = train_df['Mid_Rate'].copy()

        combined = pd.concat([X, y], axis=1).dropna()
        if len(combined) < 2:
            st.error("Not enough valid data after removing NaN.")
            st.stop()

        X_clean, y_clean = combined[X_cols], combined['Mid_Rate']

        model = LinearRegression()
        model.fit(X_clean.values, y_clean.values)
        y_pred = model.predict(X_clean.values)
        r2 = r2_score(y_clean, y_pred)

        terms = [f"{coef:.2f}×{name}" for coef, name in zip(model.coef_, disp_names)]
        eq = "Rate = " + " + ".join(terms)
        eq += f" + {model.intercept_:.0f}" if model.intercept_ >= 0 else f" – {abs(model.intercept_):.0f}"

        st.success(f"Trained on {len(X_clean)} projects – R² = {r2:.4f}")
        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.code(eq, language="latex")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_clean, y=y_pred, mode='markers',
                                 text=train_df.loc[y_clean.index, 'Project_ID'],
                                 hovertemplate='<b>%{text}</b><br>Actual: ₹%{x:,.0f}<br>Pred: ₹%{y:,.0f}'))
        fig.add_trace(go.Scatter(x=[y_clean.min(), y_clean.max()], y=[y_clean.min(), y_clean.max()],
                                 mode='lines', line=dict(dash='dash', color='red')))
        fig.update_layout(title="Actual vs Predicted Rates",
                          xaxis_title="Actual Rate (₹/sqft)",
                          yaxis_title="Predicted Rate (₹/sqft)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.session_state['model'] = {
            'model': model,
            'features': X_cols,
            'display_features': selected_features,
            'eq': eq,
            'r2': r2
        }

# --------------------- PREDICT NEW PROJECT ---------------------
st.markdown("---")
st.subheader("Predict New Project")
if 'model' not in st.session_state:
    st.info("Train a model first.")
else:
    st.info(f"Using model – R² = {st.session_state['model']['r2']:.4f}")
    inputs = {}
    for f in st.session_state['model']['display_features']:
        if f == "Project Category (Ordinal)":
            cat = st.selectbox("Project Category", ["Affordable", "Mid-Segment", "Luxury"], key="pred_cat_ord")
            inputs["Category_Ordinal"] = ORDINAL_CATEGORY_MAP[cat]
        elif f == "CBD Score":
            # User can slide a realistic range (0.0 – 1.0)
            inputs["cbd_score"] = st.slider("CBD Score", 0.0, 1.0, 0.7, 0.01, key="pred_cbd")
        else:
            col = feature_options[f]
            if col == 'amenity_score':
                inputs[col] = st.slider(f, 0.0, 1.0, 0.6, 0.01, key=f"pred_{col}")
            elif col == 'road_numeric':
                rd = st.selectbox("Road Type", ['A','B','C','D'], key=f"pred_{col}")
                inputs[col] = ROAD_MAP[rd]
            elif 'fsi' in col.lower():
                inputs[col] = st.number_input("Total FSI", 0.0, 20000.0, 2500.0, key=f"pred_{col}")
            else:
                inputs[col] = st.number_input("Age (Years)", 0.0, 50.0, 5.0, key=f"pred_{col}")

    if st.button("Predict Rate", type="primary", use_container_width=True):
        X_new = np.array([[inputs.get(c, 0) for c in st.session_state['model']['features']]])
        pred = st.session_state['model']['model'].predict(X_new)[0]
        pred_cat = categorize_rate(pred, rate_ranges)
        st.success(f"Predicted: **₹{pred:,.0f}/sqft** on Salable Area")
        st.caption(f"Category: **{pred_cat}**")
        st.caption(f"Equation: `{st.session_state['model']['eq']}`")