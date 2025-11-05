# cluster_regression_app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import os
from typing import List, Dict
import logging
from road_identifier import identify_road
from sklearn.preprocessing import StandardScaler

# ==============================================================
# PAGE CONFIG & PROFESSIONAL STYLING
# ==============================================================
st.set_page_config(
    page_title="Valuation Engine Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, modern, professional CSS
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stApp {background: #f9fafb;}
    .section-title {
        font-size: 1.6rem; font-weight: 600; color: #1f2937;
        margin: 2.5rem 0 0.75rem; border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    .subsection {
        font-size: 1.2rem; font-weight: 500; color: #374151;
        margin: 1.8rem 0 0.5rem;
    }
    .caption {
        font-size: 0.9rem; color: #6b7280; line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    .card {
        background: white; padding: 1.25rem; border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem;
    }
    .stDataFrame, .stDataEditor {
        border-radius: 0.75rem; overflow: hidden;
    }
    .stPlotlyChart {
        border-radius: 0.75rem; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 0.5rem; font-weight: 500; height: 2.5rem;
    }
    .metric-card {
        background: #f3f4f6; padding: 0.75rem 1rem;
        border-radius: 0.5rem; font-weight: 600; text-align: center;
    }
    .stAlert {border-radius: 0.75rem;}
    .stExpander {border-radius: 0.75rem; background: #f9fafb;}
</style>
""", unsafe_allow_html=True)

# ==============================================================
# CONFIG
# ==============================================================
EXCEL_FILE = "All_Project_data_WITH_Amenity_Scores.xlsx"
AMENITY_DIR = "amenities"
POI_SEARCH_RADIUS_M = 1000

DEFAULT_WEIGHTS = {"Metro": 0.25, "Bus": 0.15, "Mall": 0.23, "School": 0.23, "Hospital": 0.07, "Garden": 0.07}
DEFAULT_RATE_RANGES = {"Affordable": (0, 7000), "Mid-Segment": (7000, 13000), "Luxury": (13000, float('inf'))}
AMENITY_TO_CATEGORY = {
    "subway_entrance": "Metro", "metro_station": "Metro", "railway=station": "Metro",
    "bus_stop": "Bus", "bus_station": "Bus", "public_transport=stop_position": "Bus",
    "public_transport=platform": "Bus", "mall": "Mall", "department_store": "Mall", "supermarket": "Mall",
    "convenience": "Mall", "marketplace": "Mall", "malls": "Mall", "school": "School", "schools": "School",
    "college": "School", "university": "School", "hospital": "Hospital", "hospitals": "Hospital",
    "clinic": "Hospital", "doctors": "Hospital", "pharmacy": "Hospital", "park": "Garden",
    "gardens": "Garden", "playground": "Garden", "sports_centre": "Garden", "pitch": "Garden"
}
ROAD_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

# CBD Master
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
# HELPERS
# ==============================================================
def haversine_vectorized(lat1, lon1, lats2, lons2):
    lat1, lon1, lats2, lons2 = map(np.radians, [lat1, lon1, lats2, lons2])
    dlat = lats2 - lat1; dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
    return 6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def decay(d): return 1.0 / (1.0 + min(d, POI_SEARCH_RADIUS_M) / 200.0)

@st.cache_data(show_spinner=False)
def load_amenities():
    if not os.path.exists(AMENITY_DIR):
        st.warning(f"Amenity directory `{AMENITY_DIR}` not found.")
        return pd.DataFrame(columns=["lat", "lng", "category", "amenity_name"])
    frames = []
    for file in os.listdir(AMENITY_DIR):
        if not file.lower().endswith(".xlsx"): continue
        key = file[:-5].lower()
        cat = AMENITY_TO_CATEGORY.get(key)
        if not cat: continue
        try:
            df = pd.read_excel(os.path.join(AMENITY_DIR, file))
            if 'lat' not in df.columns or 'lng' not in df.columns: continue
            df = df.dropna(subset=["lat", "lng"])
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
            df = df.dropna(subset=["lat", "lng"])
            if df.empty: continue
            df["category"] = cat
            df["amenity_name"] = df.get("name", df.index.astype(str)).fillna("Unnamed")
            frames.append(df[["lat", "lng", "category", "amenity_name"]])
        except Exception as e:
            st.warning(f"Error reading `{file}`: {e}")
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["lat", "lng", "category", "amenity_name"])

def compute_amenity_score(lat, lng, amenities, weights):
    if amenities.empty or "category" not in amenities.columns: return 0.0
    total = 0.0
    for cat, w in weights.items():
        if cat not in amenities["category"].values: continue
        cat_df = amenities[amenities["category"] == cat]
        if cat_df.empty: continue
        dists = haversine_vectorized(lat, lng, cat_df["lat"].values, cat_df["lng"].values)
        mask = dists <= POI_SEARCH_RADIUS_M
        if not mask.any(): continue
        S_c = sum(decay(d) for d in dists[mask])
        s_c = 1 - math.exp(-0.8 * S_c)
        total += w * s_c
    return round(total, 3)

def categorize_rate(rate, ranges):
    for cat, (low, high) in ranges.items():
        if low <= rate < high: return cat
    return "Luxury"

# CBD
def get_fastest_route(start_lng, start_lat, end_lng, end_lat) -> Dict:
    url = f"{OSRM_URL}{start_lng},{start_lat};{end_lng},{end_lat}"
    params = {"overview": "false", "alternatives": "false", "steps": "false"}
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code == 200: return r.json().get("routes", [{}])[0]
    except: pass
    return {}

def calculate_cbd_score(dist_km: float, time_min: float) -> float:
    score_dist = 1 / (1 + dist_km / 10)
    score_time = 1 / (1 + time_min / 30)
    return round(0.6 * score_dist + 0.4 * score_time, 3)

@st.cache_data(show_spinner=False)
def cbd_score_for_project(lat: float, lng: float) -> float:
    best = 0.0
    for cbd in CBD_MASTER:
        route = get_fastest_route(lng, lat, cbd["lng"], cbd["lat"])
        if not route: continue
        dist_km = route["distance"] / 1000
        time_min = route["duration"] / 60
        score = calculate_cbd_score(dist_km, time_min)
        best = max(best, score)
    return best

def calculate_cbd_for_selected(df: pd.DataFrame, selected_idx: List[int]) -> pd.DataFrame:
    if not selected_idx: return df
    df = df.copy()
    with st.spinner(f"Calculating CBD for {len(selected_idx)} project(s)…"):
        scores = [cbd_score_for_project(df.loc[i, 'project_lat'], df.loc[i, 'project_lng']) for i in selected_idx]
        df.loc[selected_idx, 'cbd_score'] = scores
    return df

# ==============================================================
# MAP
# ==============================================================
def plot_selected_cluster_map(df, cluster_col, cluster_val, subject_lat=None, subject_lng=None):
    filtered = df[df[cluster_col] == cluster_val].copy()
    fig = go.Figure()

    if not filtered.empty:
        hover = filtered.apply(lambda r: f"<b>{r['Project_Name']}</b><br>Rate: ₹{r['Mid_Rate']:,.0f}/sqft<br>{r.get('Village', '—')}", axis=1)
        fig.add_trace(go.Scattermapbox(
            lat=filtered['project_lat'], lon=filtered['project_lng'],
            mode='markers', marker=dict(size=14, color=filtered['Mid_Rate'], colorscale='Viridis',
                                        showscale=True, colorbar=dict(title="Rate (₹/sqft)", x=1.02)),
            text=hover, hovertemplate='%{text}<extra></extra>', name='Projects'
        ))

        pts = filtered[['project_lng', 'project_lat']].values
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts, qhull_options='QJ')
                v = hull.vertices
                lons = pts[v, 0].tolist() + [pts[v[0], 0]]
                lats = pts[v, 1].tolist() + [pts[v[0], 1]]
                fig.add_trace(go.Scattermapbox(lon=lons, lat=lats, mode='lines',
                    line=dict(width=2, color='#dc2626'), fill='toself', fillcolor='rgba(239,68,68,0.1)',
                    name='Cluster Boundary', hoverinfo='skip'))
            except: pass

    if subject_lat and subject_lng:
        fig.add_trace(go.Scattermapbox(lat=[subject_lat], lon=[subject_lng],
            mode='markers', marker=dict(size=36, color='white'), showlegend=False))
        fig.add_trace(go.Scattermapbox(lat=[subject_lat], lon=[subject_lng],
            mode='markers', marker=dict(size=28, color='#dc2626'), name='Subject',
            hovertemplate=f"<b>Subject</b><br>Lat: {subject_lat:.6f}<br>Lng: {subject_lng:.6f}<extra></extra>"))

    center_lat = filtered['project_lat'].mean() if not filtered.empty else (subject_lat or 18.52)
    center_lon = filtered['project_lng'].mean() if not filtered.empty else (subject_lng or 73.85)

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=13.5),
        margin=dict(t=60, l=0, r=0, b=0), height=560,
        title=dict(text=f"Cluster: {cluster_val}", x=0.5, font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="center", x=0.5)
    )
    return fig

# ==============================================================
# LOAD DATA
# ==============================================================
if 'project_df' not in st.session_state:
    if not os.path.exists(EXCEL_FILE):
        st.error(f"Database file `{EXCEL_FILE}` not found.")
        st.stop()
    with st.spinner("Loading project database..."):
        df = pd.read_excel(EXCEL_FILE)
        required = ['Project_Name', 'Mid_Rate', 'project_lat', 'project_lng', 'Cluster_LatLong', 'Cluster_LatLongCategory']
        for c in required:
            if c not in df.columns: st.error(f"Missing column: `{c}`"); st.stop()
        if 'Project_ID' not in df.columns: df['Project_ID'] = [f"P{i:04d}" for i in range(1, len(df)+1)]
        defaults = [('Road_Category','B'), ('total fsi (sqmtr)',1000.0), ('Age_Of_The_Building_Till_11thOct2025',5)]
        for c, d in defaults: df[c] = df.get(c, d)
        df['amenity_score'] = df.get('amenity_score', df.get('Amenity_Raw_R_0_1', 0.0))
        st.session_state.project_df = df

project_df = st.session_state.project_df
all_amenities = load_amenities()

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## Configuration")

    with st.expander("Amenity Weights", expanded=True):
        st.caption("Adjust relative impact of proximity to key amenities.")
        if 'custom_weights' not in st.session_state: st.session_state.custom_weights = DEFAULT_WEIGHTS.copy()
        weights = {}
        for cat, val in DEFAULT_WEIGHTS.items():
            w = st.number_input(cat, 0.0, 1.0, st.session_state.custom_weights.get(cat, val), 0.01, key=f"wt_{cat}")
            weights[cat] = w
        st.markdown(f"<div class='metric-card'>Total: {sum(weights.values()):.2f}</div>", unsafe_allow_html=True)

    with st.expander("Rate Segmentation", expanded=False):
        st.caption("Define rate bands for market categorization.")
        if 'rate_ranges' not in st.session_state: st.session_state.rate_ranges = DEFAULT_RATE_RANGES.copy()
        r1 = st.number_input("Affordable up to (₹/sqft)", value=st.session_state.rate_ranges["Affordable"][1], step=500)
        r3 = st.number_input("Mid-Segment up to (₹/sqft)", value=st.session_state.rate_ranges["Mid-Segment"][1], step=500)
        updated = {"Affordable": (0, r1), "Mid-Segment": (r1, r3), "Luxury": (r3, float('inf'))}
        st.session_state.rate_ranges = updated
        st.caption(f"Affordable: < ₹{r1:,}\nMid-Segment: ₹{r1:,} – ₹{r3:,}\nLuxury: > ₹{r3:,}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apply", type="primary", use_container_width=True):
            st.session_state.custom_weights = weights.copy()
            st.session_state.weights_applied = True
            for k in list(st.session_state.keys()):
                if k.startswith("recalc_"): st.session_state.pop(k, None)
            st.success("Applied.")
            st.rerun()
    with c2:
        if st.button("Reset", use_container_width=True):
            st.session_state.custom_weights = DEFAULT_WEIGHTS.copy()
            st.session_state.rate_ranges = DEFAULT_RATE_RANGES.copy()
            st.session_state.weights_applied = False
            for k in list(st.session_state.keys()):
                if k.startswith("recalc_") or k.startswith("cluster_"): st.session_state.pop(k, None)
            st.success("Reset.")
            st.rerun()

    active_weights = st.session_state.custom_weights
    rate_ranges = st.session_state.rate_ranges

    st.markdown("---")
    st.markdown("### Model Features")
    feature_options = {
    "Amenity Score": "amenity_score",
    "Road Category": "road_numeric",
    "Total FSI": "total fsi (sqmtr)",
    "Age": "Age_Of_The_Building_Till_11thOct2025",
    "CBD Score": "cbd_score"
}
    selected_features = st.multiselect("Predictors", list(feature_options.keys()), default=["Amenity Score", "Road Category"], key="feat_sel")


# ==============================================================
# MAIN CONTENT
# ==============================================================
st.markdown("<div class='section-title'>Valuation Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='caption'>Micro-market rate prediction using location, infrastructure, and amenities.</div>", unsafe_allow_html=True)

# --- 1. Cluster & Subject ---
st.markdown("<div class='subsection'>1. Cluster Type & Subject Location</div>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    cluster_options = {
        "Location Only (LatLong)": ("Cluster_LatLong", "Grouped by location only.\nUse when: Projects are in the same micro-market."),
        "Location + Road Type": ("Cluster_LatLongCategory", "Grouped by location + road type (A, B, C, D).\nUse when: A project on a main road ≠ one on a narrow lane.")
    }
    selected_name = st.selectbox("Cluster By", options=list(cluster_options.keys()), index=0, key="cluster_type_selector")
    cluster_type, tooltip = cluster_options[selected_name]
    st.caption(tooltip)

with col2:
    st.markdown("<div class='caption'>Enter subject coordinates to locate nearest cluster.</div>", unsafe_allow_html=True)
    lat_lng_input = st.text_input("Subject: `lat, lng`", placeholder="18.5204, 73.8567", key="lat_lng_paste")

subject_lat = subject_lng = None
if lat_lng_input.strip():
    parts = [p.strip() for p in lat_lng_input.replace(" ", "").split(",")]
    if len(parts) == 2:
        try:
            subject_lat, subject_lng = float(parts[0]), float(parts[1])
            st.success(f"Lat: {subject_lat:.6f} | Lng: {subject_lng:.6f}")
        except: st.error("Invalid format.")
    else: st.error("Use: `lat, lng`")

if st.button("Locate Nearest Cluster", type="primary", use_container_width=True):
    if not subject_lat: st.error("Enter coordinates.")
    else:
        def find_nearest_cluster(lat, lng, df, col):
            clusters = df[[col, 'project_lat', 'project_lng']].dropna()
            cents = clusters.groupby(col).agg(lat=('project_lat','mean'), lng=('project_lng','mean')).reset_index()
            dists = haversine_vectorized(lat, lng, cents['lat'].values, cents['lng'].values)
            idx = np.argmin(dists)
            best = cents.iloc[idx]
            return best[col], float(dists[idx]), float(best['lat']), float(best['lng'])
        cluster, dist, _, _ = find_nearest_cluster(subject_lat, subject_lng, project_df, cluster_type)
        st.session_state.update({"subject_cluster": cluster, "subject_cluster_type": cluster_type, "subject_lat": subject_lat, "subject_lng": subject_lng, "subject_dist": dist})
        st.success(f"Nearest: **{cluster}** ({dist:,.0f} m)")

# --- Load Cluster ---
cluster_type = st.session_state.get("subject_cluster_type", "Cluster_LatLong")
selected_cluster = st.session_state.get("subject_cluster", project_df[cluster_type].dropna().iloc[0])
cluster_key = f"cluster_{cluster_type}_{selected_cluster}"

if cluster_key not in st.session_state:
    cluster_df = project_df[project_df[cluster_type] == selected_cluster].copy().reset_index(drop=True)
    cluster_df['_original_amenity'] = cluster_df['amenity_score']
    cluster_df['cbd_score'] = 0.0
    st.session_state[cluster_key] = cluster_df
else:
    cluster_df = st.session_state[cluster_key]

# Recalculate amenity scores
recalc_key = f"recalc_{cluster_key}"
if st.session_state.get("weights_applied") and not all_amenities.empty and recalc_key not in st.session_state:
    with st.spinner("Updating amenity scores..."):
        scores = [compute_amenity_score(r['project_lat'], r['project_lng'], all_amenities, active_weights) for _, r in cluster_df.iterrows()]
        cluster_df['amenity_score'] = scores
        st.session_state[recalc_key] = True
    st.success("Amenity scores updated.")
elif not st.session_state.get("weights_applied") and recalc_key in st.session_state:
    cluster_df['amenity_score'] = cluster_df['_original_amenity']
    del st.session_state[recalc_key]

cluster_df['road_numeric'] = cluster_df['Road_Category'].map(ROAD_MAP).fillna(2)

# --- 2. Map ---
st.markdown(f"<div class='subsection'>2. Cluster Map – {selected_name.split('(')[0].strip()}: {selected_cluster}</div>", unsafe_allow_html=True)
map_fig = plot_selected_cluster_map(project_df, cluster_type, selected_cluster, st.session_state.get("subject_lat"), st.session_state.get("subject_lng"))
st.plotly_chart(map_fig, use_container_width=True, config={'displayModeBar': False})

# --- 3. CBD Score ---
st.markdown("<div class='subsection'>3. Calculate CBD Score</div>", unsafe_allow_html=True)
st.caption("Distance + time to nearest CBD → higher = better")
if st.button("Calculate CBD Score (selected only)", type="secondary", use_container_width=True):
    sel_idx = st.session_state.get(f"selected_{cluster_key}", [])
    if not sel_idx: st.warning("No projects selected.")
    else:
        cluster_df = calculate_cbd_for_selected(cluster_df, sel_idx)
        st.session_state[cluster_key] = cluster_df
        st.success(f"CBD calculated for {len(sel_idx)} project(s)!")
        st.rerun()

# --- 4. Edit Table ---
st.markdown("<div class='subsection'>4. Edit & Select Projects</div>", unsafe_allow_html=True)
def build_display_df(df, ranges):
    df = df.copy()
    df['Rate_on_Salable'] = pd.to_numeric(df['Mid_Rate'], errors='coerce').fillna(0)
    df['Category'] = df['Rate_on_Salable'].apply(lambda x: categorize_rate(x, ranges))
    df['amenity_display'] = df['amenity_score'].apply(lambda x: f"{x:.3f}")
    df['cbd_display'] = df['cbd_score'].apply(lambda x: f"{x:.3f}")
    disp = df[['Project_ID', 'Project_Name', 'Rate_on_Salable', 'Road_Category', 'amenity_display', 'cbd_display', 'total fsi (sqmtr)', 'Age_Of_The_Building_Till_11thOct2025', 'Category']].copy()
    disp.columns = ['ID', 'Project', 'Rate (₹/sqft)', 'Road', 'Amenity', 'CBD', 'FSI', 'Age', 'Segment']
    disp.insert(0, 'Select', disp.index.isin(st.session_state.get(f"selected_{cluster_key}", [])))
    return disp

display_df = build_display_df(cluster_df, rate_ranges)
b1, b2, b3 = st.columns(3)
with b1: st.button("Select All", use_container_width=True, on_click=lambda: st.session_state.update({f"selected_{cluster_key}": list(range(len(cluster_df)))}), key="sel_all")
with b2: st.button("Clear", use_container_width=True, on_click=lambda: st.session_state.update({f"selected_{cluster_key}": []}), key="sel_clear")
with b3: st.button("Refresh", use_container_width=True, on_click=st.rerun, key="refresh")

edited_df = st.data_editor(display_df, num_rows="dynamic", use_container_width=True,
    column_config={
        "Select": st.column_config.CheckboxColumn("Select"),
        "ID": st.column_config.TextColumn(disabled=True),
        "Project": st.column_config.TextColumn(disabled=True),
        "Rate (₹/sqft)": st.column_config.NumberColumn(format="₹%.0f"),
        "Road": st.column_config.SelectboxColumn(options=['A','B','C','D']),
        "Amenity": st.column_config.TextColumn(disabled=True),
        "CBD": st.column_config.TextColumn(disabled=True),
        "FSI": st.column_config.NumberColumn(format="%.0f"),
        "Age": st.column_config.NumberColumn(format="%.1f"),
        "Segment": st.column_config.TextColumn(disabled=True)
    }, hide_index=True, key=f"editor_{cluster_key}")

# Save edits
cluster_df.loc[edited_df.index, ['Rate_on_Salable', 'Road_Category', 'total fsi (sqmtr)', 'Age_Of_The_Building_Till_11thOct2025']] = edited_df[['Rate (₹/sqft)', 'Road', 'FSI', 'Age']].values
cluster_df['road_numeric'] = cluster_df['Road_Category'].map(ROAD_MAP).fillna(2)
cluster_df['Mid_Rate'] = cluster_df['Rate_on_Salable']
st.session_state[cluster_key] = cluster_df
st.session_state[f"selected_{cluster_key}"] = edited_df[edited_df['Select']].index.tolist()
train_df = cluster_df.loc[st.session_state[f"selected_{cluster_key}"]].copy()

# --- 5. Train Regression Model ---
st.markdown("<div class='subsection'>5. Train Regression Model</div>", unsafe_allow_html=True)

if st.button("Train Model", type="primary", use_container_width=True):
    if not selected_features:
        st.error("Select at least one predictor.")
    elif len(train_df) < 2:
        st.error("Need ≥2 projects to train.")
    else:
        # ------------------------------------------------------------------
        # 1. Build X / y – keep **only rows that have ALL selected features**
        # ------------------------------------------------------------------
        X_cols = [feature_options[f] for f in selected_features]   # e.g. ['amenity_score','road_numeric',...]
        X_raw  = train_df[X_cols].copy()
        y      = train_df['Mid_Rate'].copy()

        combined = pd.concat([X_raw, y], axis=1).dropna()
        if len(combined) < 2:
            st.error("Not enough complete rows after dropping NaNs.")
            st.stop()

        X = combined[X_cols]
        y = combined['Mid_Rate']

        # ------------------------------------------------------------------
        # 2. **NO NORMALISATION** – feed raw data directly to the model
        # ------------------------------------------------------------------
        model = LinearRegression()
        model.fit(X, y)

        pred = model.predict(X)
        r2   = r2_score(y, pred)

        # ------------------------------------------------------------------
        # 3. Build readable equation (coefficients are already on original scale)
        # ------------------------------------------------------------------
        eq_parts = [f"{c:.2f}×{n}" for c, n in zip(model.coef_, selected_features)]
        eq = "Rate = " + " + ".join(eq_parts)
        eq += f" + {model.intercept_:.0f}" if model.intercept_ >= 0 else f" – {abs(model.intercept_):.0f}"

        # ------------------------------------------------------------------
        # 4. Store model + indices of rows that were actually used
        # ------------------------------------------------------------------
        st.session_state['model'] = {
            'model'            : model,
            'features'         : X_cols,
            'display_features' : selected_features,
            'eq'               : eq,
            'r2'               : r2,
            'train_idx'        : X.index.tolist()          # <-- only these rows
        }

        # ------------------------------------------------------------------
        # 5. UI – success + metrics + plot
        # ------------------------------------------------------------------
        st.success(f"Trained on **{len(combined)}** projects | R² = {r2:.4f}")
        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.code(eq)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y, y=pred, mode='markers',
            text=train_df.loc[X.index, 'Project_ID'],
            hovertemplate='<b>%{text}</b><br>Actual: ₹%{x:,.0f}<br>Pred: ₹%{y:,.0f}'
        ))
        fig.add_trace(go.Scatter(
            x=[y.min(), y.max()], y=[y.min(), y.max()],
            mode='lines', line=dict(dash='dash', color='red')
        ))
        fig.update_layout(
            title="Actual vs Predicted (training data)",
            xaxis_title="Actual Rate", yaxis_title="Predicted Rate",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------------
        # 6. **Toggle Table** – *only* the projects that were used for training
        # ------------------------------------------------------------------
        with st.expander("Training Projects (strictly used for model)", expanded=False):
            train_used = train_df.loc[X.index].copy()

            # Show only the columns that were actually fed to the model
            cols_to_show = ['Project_ID', 'Project_Name', 'Mid_Rate'] + X_cols
            train_used = train_used[cols_to_show]

            # Human-readable column names
            rename_map = {
                'Project_ID' : 'ID',
                'Project_Name': 'Project',
                'Mid_Rate'   : 'Rate (₹/sqft)'
            }
            display_to_col = {v: k for k, v in feature_options.items()}
            for col in X_cols:
                rename_map[col] = display_to_col.get(col, col.replace('_', ' ').title())
            train_used.rename(columns=rename_map, inplace=True)

            # Formatting
            fmt_dict = {c: "{:,.0f}" for c in train_used.columns if "Rate" in c}
            fmt_dict.update({c: "{:.3f}" for c in train_used.columns if c in ["Amenity Score", "CBD Score"]})
            fmt_dict.update({c: "{:.1f}" for c in train_used.columns if "Age" in c})

            st.dataframe(
                train_used.style.format(fmt_dict),
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"**{len(train_used)}** projects were *actually* used (rows with any missing feature were dropped).")
            
# --- 6. Predict Subject Rate ---
st.markdown("<div class='subsection'>6. Predict Subject Rate</div>", unsafe_allow_html=True)
if "model" not in st.session_state:
    st.info("Train a model to enable prediction.")
else:
    model_info = st.session_state["model"]
    if "run_id" not in st.session_state: 
        st.session_state.run_id = 0

    cols = st.columns(min(len(model_info["features"]), 4))
    inputs = {}
    phs = {}
    name_map = {
        "amenity_score": "Amenity Score",
        "cbd_score": "CBD Score",
        "road_numeric": "Road Type",
        "total fsi (sqmtr)": "Total FSI (sqm)",
        "Age_Of_The_Building_Till_11thOct2025": "Age (years)"
    }

    for i, col in enumerate(model_info["features"]):
        with cols[i % len(cols)]:
            name = name_map.get(col, col.replace("_", " ").title())

            # COMPUTED FIELDS
            if col in ["amenity_score", "cbd_score", "road_numeric"]:
                ph = st.empty()
                phs[col] = ph
                ph.text_input(
                    f"**{name}** (computed)",
                    value="—",
                    disabled=True,
                    key=f"ph_{col}_{st.session_state.run_id}"
                )
            else:
                # MANUAL INPUT FIELDS
                default_val = st.session_state.get(f"inp_{col}", 2500.0 if "fsi" in col else 5.0)
                step_val = 0.1 if "Age" in name else 100.0
                format_str = "%.1f" if "Age" in name else "%.0f"

                val = st.number_input(
                    f"**{name}**",
                    min_value=0.0,
                    value=float(default_val),
                    step=float(step_val),
                    format=format_str,
                    key=f"inp_{col}"
                )
                inputs[col] = val

    # COMPUTE & PREDICT BUTTON
    if st.button("Compute & Predict", type="primary", use_container_width=True):
        st.session_state.run_id += 1
        run_id = st.session_state.run_id

        if not subject_lat or not subject_lng:
            st.error("Enter subject location first!")
            st.stop()

        with st.spinner("Analyzing location..."):
            # 1. Amenity Score
            amenity = 0.0
            amenity_df = pd.DataFrame(columns=["Amenity Name", "Type", "Category", "Distance (m)", "Influence Factor f(d)"])
            if not all_amenities.empty:
                amenity = compute_amenity_score(subject_lat, subject_lng, all_amenities, active_weights)

                amenity_rows = []
                for cat, w in active_weights.items():
                    if cat not in all_amenities["category"].values: 
                        continue
                    cat_df = all_amenities[all_amenities["category"] == cat]
                    if cat_df.empty: 
                        continue
                    dists = haversine_vectorized(subject_lat, subject_lng, cat_df["lat"].values, cat_df["lng"].values)
                    mask = dists <= POI_SEARCH_RADIUS_M
                    if not mask.any(): 
                        continue
                    for d, row in zip(dists[mask], cat_df[mask].itertuples()):
                        influence = decay(d)
                        name = getattr(row, 'amenity_name', "Unnamed")
                        amenity_rows.append({
                            "Amenity Name": name,
                            "Type": cat.lower(),
                            "Category": cat,
                            "Distance (m)": round(d, 1),
                            "Influence Factor f(d)": round(influence, 3)
                        })
                amenity_df = pd.DataFrame(amenity_rows).sort_values("Distance (m)").reset_index(drop=True) if amenity_rows else amenity_df

            # 2. CBD Score
            cbd = cbd_score_for_project(subject_lat, subject_lng)

            # 3. Road Detection
            roads_all, nearest_road = identify_road(subject_lat, subject_lng)
            road_cat = nearest_road.get("category", "B")
            road_num = ROAD_MAP.get(road_cat, 2)

            # 4. Build feature vector
            vec = {
                "amenity_score": amenity,
                "cbd_score": cbd,
                "road_numeric": road_num,
                "total fsi (sqmtr)": inputs.get("total fsi (sqmtr)", 2500.0),
                "Age_Of_The_Building_Till_11thOct2025": inputs.get("Age_Of_The_Building_Till_11thOct2025", 5.0)
            }
            X_sub = [vec.get(c, 0.0) for c in model_info["features"]]
            pred_rate = model_info["model"].predict([X_sub])[0]
            pred_cat = categorize_rate(pred_rate, rate_ranges)

        # UPDATE COMPUTED FIELDS
        computed_vals = {
            "amenity_score": round(amenity, 3),
            "cbd_score": round(cbd, 3),
            "road_numeric": road_cat  # Display category, not numeric
        }
        for col, val in computed_vals.items():
            if col in phs:
                phs[col].text_input(
                    f"**{name_map[col]}** (computed)",
                    value=str(val),
                    disabled=True,
                    key=f"done_{col}_{run_id}"
                )

        # DISPLAY RESULTS
        st.success(f"**Predicted Rate: ₹{pred_rate:,.0f}/sqft on Salable Area**")
        st.caption(f"**Market Segment:** {pred_cat}")
        st.code(model_info["eq"], language="latex")

        # SUBJECT ATTRIBUTES TABLE
        st.markdown("#### Subject Location Attributes")
        attr_rows = []
        display_map = {
            "amenity_score": ("Amenity Score", f"{amenity:.3f}"),
            "cbd_score": ("CBD Score", f"{cbd:.3f}"),
            "road_numeric": ("Road Type", road_cat),
            "total fsi (sqmtr)": ("Total FSI (sqm)", f"{vec['total fsi (sqmtr)']:.0f}"),
            "Age_Of_The_Building_Till_11thOct2025": ("Age (years)", f"{vec['Age_Of_The_Building_Till_11thOct2025']:.1f}")
        }
        for col in model_info["features"]:
            if col in display_map:
                attr_rows.append({"Attribute": display_map[col][0], "Value": display_map[col][1]})
        if attr_rows:
            st.dataframe(
                pd.DataFrame(attr_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Attribute": st.column_config.TextColumn(width="medium"),
                    "Value": st.column_config.TextColumn(width="small")
                }
            )
        else:
            st.info("No model features to display.")

        # AMENITY TABLE
        st.markdown("#### Amenities within 1 km")
        if not amenity_df.empty:
            st.dataframe(
                amenity_df.style.format({
                    "Distance (m)": "{:.1f}",
                    "Influence Factor f(d)": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"**Total amenities:** {len(amenity_df)} | **Amenity Score:** {amenity:.3f}")
        else:
            st.info("No amenities found within 1 km.")

        # ROAD TABLE
        st.markdown("#### Nearby Highways (≤ 200 m)")
        road_records = [
            {"Road Name": r["name"], "Highway Type": r["highway"], "Category": r["category"], "Distance (m)": round(r["distance_m"], 1)}
            for r in roads_all if r["distance_m"] <= 200
        ]
        road_df = pd.DataFrame(road_records).sort_values("Distance (m)").reset_index(drop=True) if road_records else pd.DataFrame(
            columns=["Road Name", "Highway Type", "Category", "Distance (m)"]
        )
        if not road_df.empty:
            st.dataframe(
                road_df.style.format({"Distance (m)": "{:.1f}"}),
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"**Nearest road:** {road_cat} | **{len(road_df)} roads found**")
        else:
            st.info("No roads within 200 m.")