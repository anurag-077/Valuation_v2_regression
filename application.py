import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import requests
from shapely.geometry import Point, LineString
from shapely.ops import transform
from pyproj import CRS, Transformer
import json
import logging
import time
from scipy.spatial import ConvexHull

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Default static weights for amenities
DEFAULT_WEIGHTS = {
    'Metro': 0.25,
    'Bus': 0.15,
    'Mall': 0.225,
    'School': 0.225,
    'Hospital': 0.075,
    'Garden': 0.075
}

# Amenity types mapping
AMENITY_TYPES = {
    'bus_stop': 'Bus',
    'bus_station': 'Bus',
    'railway=station': 'Bus',
    'subway_entrance': 'Metro',
    'tram_stop': 'Bus',
    'public_transport=stop_position': 'Bus',
    'public_transport=platform': 'Bus',
    'public_transport=station': 'Bus',
    'metro_station': 'Metro',
    'school': 'School',
    'schools': 'School',
    'college': 'School',
    'university': 'School',
    'hospital': 'Hospital',
    'hospitals': 'Hospital',
    'clinic': 'Hospital',
    'doctors': 'Hospital',
    'pharmacy': 'Hospital',
    'park': 'Garden',
    'gardens': 'Garden',
    'playground': 'Garden',
    'sports_centre': 'Garden',
    'pitch': 'Garden',
    'supermarket': 'Mall',
    'convenience': 'Mall',
    'department_store': 'Mall',
    'mall': 'Mall',
    'malls': 'Mall',
    'marketplace': 'Mall'
}

POI_SEARCH_RADIUS_M = 1000
SEARCH_RADIUS_M = 200  # Road search radius (meters)
THRESHOLD_M = 200.0  # Adjacency logic (meters)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]
HIGHWAY_CLASSES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "residential", "unclassified", "service",
    "track", "path", "living_street", "pedestrian", "road"
]
CATEGORY_BY_OSM = {
    "tertiary":       ("A", "Category A – Minor Road (<12 m)", 2.5),
    "residential":    ("A", "Category A – Minor Road (<12 m)", 2.5),
    "unclassified":   ("A", "Category A – Minor Road (<12 m)", 2.5),
    "service":        ("A", "Category A – Minor Road (<12 m)", 2.5),
    "track":          ("A", "Category A – Minor Road (<12 m)", 2.5),
    "path":           ("A", "Category A – Minor Road (<12 m)", 2.5),
    "living_street":  ("A", "Category A – Minor Road (<12 m)", 2.5),
    "pedestrian":     ("A", "Category A – Minor Road (<12 m)", 2.5),
    "road":           ("A", "Category A – Minor Road (<12 m)", 2.5),
    "secondary":      ("B", "Category B – Local Main Road (12–18 m)", 5.0),
    "secondary_link": ("B", "Category B – Local Main Road (12–18 m)", 5.0),
    "primary":        ("C", "Category C – Major / Sub-Arterial (18–30 m)", 7.5),
    "primary_link":   ("C", "Category C – Major / Sub-Arterial (18–30 m)", 7.5),
    "trunk_link":     ("C", "Category C – Major / Sub-Arterial (18–30 m)", 7.5),
    "trunk":          ("D", "Category D – Arterial / Highway (30–75 m)", 10.0),
    "motorway":       ("D", "Category D – Arterial / Highway (30–75 m)", 10.0),
    "motorway_link":  ("D", "Category D – Arterial / Highway (30–75 m)", 10.0),
}

@st.cache_data
def load_project_data():
    """Load project data from All_Project_data_WITH_Amenity_Scores.xlsx."""
    file = 'All_Project_data_WITH_Amenity_Scores.xlsx'
    if not os.path.exists(file):
        st.error("All_Project_data_WITH_Amenity_Scores.xlsx not found!")
        st.stop()
    df = pd.read_excel(file)
    
    # Print column names for debugging
    print("Columns in project_df:", df.columns.tolist())
    
    # Attempt to rename common latitude/longitude columns
    df = df.rename(columns={
        'Latitude': 'project_lat', 'longitude': 'project_lng',
        'lat': 'project_lat', 'lng': 'project_lng',
        'Project_Latitude': 'project_lat', 'Project_Longitude': 'project_lng'
    })
    
    # Validate required columns
    required_columns = ['project_lat', 'project_lng', 'Project_Name', 'Mid_Rate', 'Village', 'Cluster_LatLong']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns in project data: {', '.join(missing_columns)}")
        st.stop()
    
    print(f"Loaded Projects: {len(df)} rows")
    return df

@st.cache_data
def load_regression_data():
    """Load regression results from regression_results.xlsx."""
    file = 'regression_results.xlsx'
    if not os.path.exists(file):
        st.warning("regression_results.xlsx not found! Run regression first.")
        return {}
    
    data = {}
    sheet_names = [
        'LatLong_Amenity_vs_Rate', 'LatLong_RoadCat_vs_Rate', 'LatLong_Both_vs_Rate',
        'LatLongRate_Amenity_vs_Rate', 'LatLongRate_RoadCat_vs_Rate', 'LatLongRate_Both_vs_Rate',
        'LatLongCategory_Amenity_vs_Rate'
    ]
    
    for sheet in sheet_names:
        try:
            data[sheet] = pd.read_excel(file, sheet_name=sheet)
            print(f"Loaded: {sheet} ({len(data[sheet])} rows)")
        except:
            data[sheet] = pd.DataFrame()
    return data

@st.cache_data
def load_amenities(amenity_dir: str = "amenities"):
    """Load amenities from the 'amenities' folder."""
    if not os.path.exists(amenity_dir):
        st.warning("'amenities' folder not found. Using sample data.")
        return create_sample_data()
    
    found_files = [f for f in os.listdir(amenity_dir) if f.lower().endswith('.xlsx')]
    if not found_files:
        st.warning("No .xlsx files in 'amenities' folder. Using sample data.")
        return create_sample_data()
    
    data = []
    for file in found_files:
        type_name = file[:-5].lower()
        if type_name not in AMENITY_TYPES:
            continue
        
        group_name = AMENITY_TYPES[type_name]
        file_path = os.path.join(amenity_dir, file)
        
        try:
            df = pd.read_excel(file_path)
            if 'lat' not in df.columns or 'lng' not in df.columns:
                continue
                
            df = df.dropna(subset=['lat', 'lng'])
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
            df = df.dropna(subset=['lat', 'lng'])
            
            if df.empty:
                continue
            
            if 'name' in df.columns:
                df['name'] = df['name'].fillna('Unnamed')
            else:
                df['name'] = f"{type_name.capitalize()}-{pd.Series(range(1, len(df)+1))}"
            
            df['category'] = group_name
            df['type_name'] = type_name
            data.append(df[['lat', 'lng', 'category', 'type_name', 'name']])
            print(f"Loaded {len(df)} {type_name}")
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if data:
        result = pd.concat(data, ignore_index=True)
        print(f"Total Amenities: {len(result)}")
        return result
    return create_sample_data()

def create_sample_data():
    """Generate sample amenity data for testing."""
    sample_data = [
        {'lat': 18.5530, 'lng': 73.7589, 'name': 'Metro-1', 'type_name': 'metro_station', 'category': 'Metro'},
        {'lat': 18.5536, 'lng': 73.7595, 'name': 'Metro-2', 'type_name': 'metro_station', 'category': 'Metro'},
    ]
    return pd.DataFrame(sample_data)

def haversine_vectorized(lat1: float, lon1: float, lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lats2_rad = np.radians(lats2)
    lons2_rad = np.radians(lons2)
    
    dlat = lats2_rad - lat1_rad
    dlon = lons2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lats2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371000
    return R * c

def calculate_amenity_scores(lat: float, lon: float, all_amenities: pd.DataFrame, weights: dict) -> pd.DataFrame:
    if all_amenities.empty:
        return pd.DataFrame()
    
    lats = all_amenities['lat'].values
    lons = all_amenities['lng'].values
    dists = haversine_vectorized(lat, lon, lats, lons)
    
    mask = dists <= POI_SEARCH_RADIUS_M
    if not np.any(mask):
        return pd.DataFrame()
    
    filtered_df = all_amenities[mask].copy()
    filtered_df['distance_m'] = dists[mask]
    filtered_df['f_d'] = 1 / (1 + filtered_df['distance_m'] / 200)
    
    category_scores = filtered_df.groupby('category')['f_d'].sum().reset_index()
    category_scores.columns = ['category', 'S_c']
    category_scores['s_c'] = 1 - np.exp(-0.8 * category_scores['S_c'])
    category_scores['weight'] = category_scores['category'].map(weights).fillna(0)
    category_scores['Weight × s_c'] = category_scores['weight'] * category_scores['s_c']
    category_scores['count'] = filtered_df.groupby('category').size().reindex(category_scores['category']).fillna(0)
    
    total_score = category_scores['Weight × s_c'].sum()
    category_scores['total_score'] = total_score
    
    for cat in weights:
        if cat not in category_scores['category'].values:
            category_scores = pd.concat([category_scores, pd.DataFrame({
                'category': [cat], 'S_c': [0], 's_c': [0], 'weight': [weights[cat]],
                'Weight × s_c': [0], 'count': [0], 'total_score': [total_score]
            })], ignore_index=True)
    
    return category_scores.sort_values('Weight × s_c', ascending=False)

def get_detailed_amenities(lat: float, lon: float, all_amenities: pd.DataFrame) -> pd.DataFrame:
    if all_amenities.empty:
        return pd.DataFrame()
    
    lats = all_amenities['lat'].values
    lons = all_amenities['lng'].values
    dists = haversine_vectorized(lat, lon, lats, lons)
    
    mask = dists <= POI_SEARCH_RADIUS_M
    filtered_df = all_amenities[mask].copy()
    filtered_df['distance_m'] = dists[mask]
    filtered_df['f_d'] = 1 / (1 + filtered_df['distance_m'] / 200)
    
    return filtered_df[['name', 'type_name', 'category', 'distance_m', 'f_d', 'lat', 'lng']].sort_values('distance_m')

def haversine_distance(lat1, lon1, lats2, lons2):
    """Vectorized haversine distance for cluster finding."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lats2_rad, lons2_rad = np.radians(lats2), np.radians(lons2)
    
    dlat = lats2_rad - lat1_rad
    dlon = lons2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lats2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return 6371000 * c

def find_nearest_cluster(df, lat, lon, cluster_cols=['Cluster_LatLong', 'Cluster_LatLongCategory']):
    distances = haversine_distance(lat, lon, df['project_lat'].values, df['project_lng'].values)
    min_idx = np.argmin(distances)
    min_dist_km = distances[min_idx] / 1000
    
    cluster_info = {}
    for col in cluster_cols:
        if col in df.columns:
            cluster_info[col] = df.iloc[min_idx][col]
    
    return cluster_info, min_dist_km, df.iloc[min_idx]

def overpass_query_any(lat: float, lon: float, radius_m: int = SEARCH_RADIUS_M, timeout_s: int = 300, max_retries: int = 5) -> dict:
    highway_filter = "|".join(HIGHWAY_CLASSES)
    query = f"""
    [out:json][timeout:{timeout_s}];
    way(around:{radius_m},{lat},{lon})["highway"~"^{highway_filter}$"];
    out tags geom;
    """
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(max_retries):
            try:
                logging.info(f"Querying Overpass API at {endpoint} for lat={lat}, lon={lon}, radius={radius_m}m, attempt {attempt + 1}")
                r = requests.post(endpoint, data={"data": query}, timeout=timeout_s + 10)
                r.raise_for_status()
                result = r.json()
                logging.info(f"Found {len(result.get('elements', []))} ways for lat={lat}, lon={lon}")
                return result
            except requests.RequestException as e:
                logging.warning(f"Request failed: {e}. Retrying after {1.5 ** attempt}s...")
                time.sleep(1.5 ** attempt)
    raise RuntimeError("Overpass query failed after retries.")

def local_metric_transformer(lat: float, lon: float):
    zone = int((lon + 180) // 6) + 1
    is_northern = lat >= 0
    utm_epsg = 32600 + zone if is_northern else 32700 + zone
    try:
        crs_src = CRS.from_epsg(4326)
        crs_dst = CRS.from_epsg(utm_epsg)
    except Exception:
        logging.warning("UTM projection failed, falling back to EPSG:3395")
        crs_src = CRS.from_epsg(4326)
        crs_dst = CRS.from_epsg(3395)
    return Transformer.from_crs(crs_src, crs_dst, always_xy=True), Transformer.from_crs(crs_dst, crs_src, always_xy=True)

def linestring_from_overpass_geom(way: dict) -> LineString:
    geom = way.get("geometry")
    if not geom or len(geom) < 2:
        logging.warning(f"Invalid geometry for way ID {way.get('id')}")
        return None
    coords = [(pt["lon"], pt["lat"]) for pt in geom]
    return LineString(coords)

def category_from_osm_highway(highway: str) -> tuple:
    return CATEGORY_BY_OSM.get(highway, (None, None, 0.0))

def get_highways_within_radius(lat: float, lon: float) -> tuple:
    try:
        data = overpass_query_any(lat, lon, radius_m=SEARCH_RADIUS_M)
        elements = [el for el in data.get("elements", []) if el.get("type") == "way"]
        if not elements:
            logging.info(f"No roads found for lat={lat}, lon={lon}")
            return [], {"category": None, "category_label": None, "distance_m": None, "name": None}

        to_m, _ = local_metric_transformer(lat, lon)
        project = lambda x, y: to_m.transform(x, y)
        pt_ll = Point((lon, lat))
        pt_m = transform(project, pt_ll)

        road_distances = []
        for w in elements:
            geom_ll = linestring_from_overpass_geom(w)
            if geom_ll is None:
                continue
            geom_m = transform(project, geom_ll)
            dist_m = pt_m.distance(geom_m)
            if dist_m > SEARCH_RADIUS_M:
                continue
            tags = w.get("tags", {})
            highway = tags.get("highway")
            code, label, pct = category_from_osm_highway(highway)
            if code is None:
                continue
            road_distances.append({
                "highway": highway,
                "category": code,
                "category_label": label,
                "distance_m": float(dist_m),
                "increase_pct": pct,
                "name": tags.get("name", "Unnamed"),
                "geometry": list(geom_ll.coords) if geom_ll else None
            })

        if not road_distances:
            return [], {"category": None, "category_label": None, "distance_m": None, "name": None}

        # Sort by increase_pct (descending) and then distance_m (ascending)
        road_distances.sort(key=lambda r: (-r["increase_pct"], r["distance_m"]))
        nearest_biggest_highway = road_distances[0]
        # For the all_highways table, sort by distance
        all_highways_sorted = sorted(road_distances, key=lambda r: r["distance_m"])
        return all_highways_sorted, nearest_biggest_highway

    except Exception as e:
        logging.error(f"Error querying roads for lat={lat}, lon={lon}: {e}")
        return [], {"category": None, "category_label": None, "distance_m": None, "name": None}

def plot_cluster_map(df, cluster_col, cluster_num, title="Cluster Map"):
    filtered = df.copy()
    
    if cluster_num is not None:
        filtered = df[df[cluster_col] == cluster_num].copy()
    
    if filtered.empty:
        st.warning(f"No projects in {cluster_col} = {cluster_num if cluster_num is not None else 'All'}")
        return None
    
    filtered['hover_text'] = filtered.apply(
        lambda row: f"<b>{row['Project_Name']}</b><br>₹{row['Mid_Rate']:.1f} per sqft<br>{row['Village']}", axis=1
    )
    
    fig = px.scatter_mapbox(
        filtered,
        lat='project_lat', lon='project_lng',
        hover_name='hover_text',
        color='Mid_Rate',  # Color by rate for visual appeal
        color_continuous_scale='viridis',
        zoom=11,
        height=500,
        title=f"{title} - Projects in Cluster {cluster_num if cluster_num is not None else 'All'}",
        labels={'Mid_Rate': 'Rate (₹ per sqft)'}
    )
    
    fig.update_traces(
        marker=dict(size=14, opacity=0.8)
    )
    
    # Add borders if single cluster
    if cluster_num is not None:
        points = filtered[['project_lng', 'project_lat']].values
        if len(points) >= 3:
            try:
                hull = ConvexHull(points, qhull_options="QJ")  # Use QJ to joggle points
                vertices = hull.vertices
                lons = points[vertices, 0]
                lats = points[vertices, 1]
                lons = np.append(lons, lons[0])
                lats = np.append(lats, lats[0])
                fig.add_trace(
                    go.Scattermapbox(
                        lon=lons,
                        lat=lats,
                        mode='lines',
                        line=dict(width=2, color='red'),
                        fill='none',  # No fill to keep projects visible
                        name="Cluster Boundary",
                        hoverinfo='skip'
                    )
                )
            except Exception as e:
                logging.warning(f"Failed to compute convex hull for cluster {cluster_num}: {e}")
                st.warning(f"Could not draw boundary for cluster {cluster_num} due to insufficient point variation.")
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(t=60),
        hovermode='closest',
        legend=dict(
            title="Rate (₹ per sqft)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.add_annotation(
        text="Note: Points represent projects colored by their mid rate.",
        xref="paper", yref="paper", x=0.01, y=0.01,
        showarrow=False, font=dict(size=12, color="gray"),
        bgcolor="white", bordercolor="gray", borderwidth=1
    )
    return fig

def show_regression_visuals(regression_data, cluster_num, category):
    available_sheets = []
    if category == 'LatLong':
        available_sheets = ['LatLong_Amenity_vs_Rate', 'LatLong_RoadCat_vs_Rate', 'LatLong_Both_vs_Rate']
    elif category == 'LatLongRate':
        available_sheets = ['LatLongRate_Amenity_vs_Rate', 'LatLongRate_RoadCat_vs_Rate', 'LatLongRate_Both_vs_Rate']
    else:
        available_sheets = ['LatLongCategory_Amenity_vs_Rate']
    
    cluster_data = {}
    for sheet in available_sheets:
        if sheet in regression_data and not regression_data[sheet].empty:
            match = regression_data[sheet][regression_data[sheet]['Cluster'] == cluster_num]
            if not match.empty:
                cluster_data[sheet] = match.iloc[0]
    
    if not cluster_data:
        st.info("No regression data available for this cluster.")
        return
    
    cols = st.columns(min(3, len(cluster_data)))
    for idx, (sheet, row) in enumerate(cluster_data.items()):
        with cols[idx]:
            if 'Both' in sheet:
                title = "Combined Amenity & Road vs Rate"
            elif 'Amenity' in sheet:
                title = "Amenity Score vs Rate"
                x_range, x_label = 10, "Amenity Score (1-10)"
                slope = row['Slope_Amenity']
            elif 'RoadCat' in sheet:
                title = "Road Category vs Rate"
                x_range, x_label = 4, "Road Category (1-4)"
                slope = row['Slope_RoadCat']
            else:
                title = "Amenity Score vs Rate"
                x_range, x_label = 10, "Amenity Score (1-10)"
                slope = row['Slope_Amenity']
            
            fig = create_regression_plot(
                row['Equation'], slope, row['Intercept'],
                x_label, "Rate (₹ per sqft)", row['Num_Projects'], x_range, title
            )
            st.plotly_chart(fig, use_container_width=True)
    
    eqs = []
    for sheet, row in cluster_data.items():
        eqs.append({
            'Model': sheet.replace('_vs_Rate', ''),
            'Equation': row['Equation'],
            'Sample Size': row['Num_Projects']
        })
    st.dataframe(pd.DataFrame(eqs).style.set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', '#f0f2f6')]}]), use_container_width=True)

def create_regression_plot(equation, slope, intercept, x_label, y_label, n, x_range, title):
    x = np.linspace(1, x_range, 100)
    y = slope * x + intercept
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', 
                           line=dict(color='#1f77b4', width=4, dash='solid')))
    
    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text=f"<b>{equation}</b><br>Sample Size: {n}",
        showarrow=False, font=dict(size=12), 
        bgcolor="white", bordercolor="#1f77b4", borderwidth=1
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=280,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    fig.add_annotation(
        text="Note: Line represents the regression model fit.",
        xref="paper", yref="paper", x=0.01, y=0.01,
        showarrow=False, font=dict(size=12, color="gray"),
        bgcolor="white", bordercolor="gray", borderwidth=1
    )
    return fig

def main():
    st.set_page_config(page_title="Valuation Analyzer Pro", layout="wide")
    
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 20px; color: white; margin-bottom: 30px;'>
        <h1 style='margin: 0; font-size: 2.5em;'>Valuation Analyzer Pro</h1>
        <p style='margin: 10px 0 0 0; font-size: 1.1em;'>Interactive tool for analyzing property valuations based on location, amenities, highways, and regression models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading data..."):
        project_df = load_project_data()
        regression_data = load_regression_data()
        all_amenities = load_amenities("amenities")
    
    tab1, tab2 = st.tabs(["Cluster Explorer", "Location Analyzer"])
    
    with tab1:
        st.header("Cluster Explorer")
        st.caption("Explore project clusters on a map based on geographic or categorical groupings. Selecting a village shows all projects in clusters containing at least one project from that village.")
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_type = st.selectbox("Cluster Type", 
                                      ['Cluster_LatLong', 'Cluster_LatLongCategory'],
                                      help="Select the type of clustering: LatLong (geographic) or LatLongCategory (geographic and categorical).")
        with col2:
            villages = sorted(project_df['Village'].dropna().unique())
            selected_village = st.selectbox("Select Village", villages,
                                          help="Choose a village to filter clusters. Shows all projects in clusters that include this village.")
            # Get clusters that contain at least one project from the selected village
            filtered_df = project_df[project_df['Village'] == selected_village]
            clusters = sorted(filtered_df[cluster_type].dropna().unique())
            clusters = ['All'] + clusters  # Add "All" as the first option
            selected_cluster = st.selectbox("Cluster Number", clusters,
                                          help="Choose a specific cluster number to visualize all projects in that cluster, or select 'All' to see all relevant clusters.")
        
        # Visualize all clusters containing the selected village
        if selected_cluster == 'All':
            # Get all projects in clusters that have at least one project from the selected village
            relevant_clusters = project_df[project_df[cluster_type].isin(clusters)]
            relevant_clusters['hover_text'] = relevant_clusters.apply(
                lambda row: f"<b>{row['Project_Name']}</b><br>₹{row['Mid_Rate']:.1f} per sqft<br>{row['Village']}", axis=1
            )
            num_villages = len(relevant_clusters['Village'].unique())
            fig_map = px.scatter_mapbox(
                relevant_clusters,
                lat='project_lat', lon='project_lng',
                hover_name='hover_text',
                color=cluster_type,  # Color by cluster type
                color_discrete_sequence=px.colors.qualitative.Plotly,  # Distinct colors for each cluster
                zoom=11,
                height=500,
                title=f"Cluster Map - All Projects in Clusters with {selected_village} (Spans {num_villages} Villages)",
                labels={cluster_type: 'Cluster', 'Mid_Rate': 'Rate (₹ per sqft)'}
            )
            fig_map.update_traces(
                marker=dict(size=14, opacity=0.8)
            )
            # Add borders for each cluster
            colors = px.colors.qualitative.Plotly
            unique_clusters = relevant_clusters[cluster_type].unique()
            for idx, cluster in enumerate(unique_clusters):
                cluster_data = relevant_clusters[relevant_clusters[cluster_type] == cluster]
                points = cluster_data[['project_lng', 'project_lat']].values
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points, qhull_options="QJ")  # Use QJ to joggle points
                        vertices = hull.vertices
                        lons = points[vertices, 0]
                        lats = points[vertices, 1]
                        lons = np.append(lons, lons[0])
                        lats = np.append(lats, lats[0])
                        fig_map.add_trace(
                            go.Scattermapbox(
                                lon=lons,
                                lat=lats,
                                mode='lines',
                                line=dict(width=2, color=colors[idx % len(colors)]),
                                fill='none',  # No fill to keep projects visible
                                name=f"Cluster {cluster} Boundary",
                                hoverinfo='skip'
                            )
                        )
                    except Exception as e:
                        logging.warning(f"Failed to compute convex hull for cluster {cluster}: {e}")
                        st.warning(f"Could not draw boundary for cluster {cluster} due to insufficient point variation.")
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(t=60),
                hovermode='closest',
                legend=dict(
                    title="Clusters",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig_map.add_annotation(
                text="Note: Points represent projects colored by their cluster. Clusters include projects from selected village and others.",
                xref="paper", yref="paper", x=0.01, y=0.01,
                showarrow=False, font=dict(size=12, color="gray"),
                bgcolor="white", bordercolor="gray", borderwidth=1
            )
            st.plotly_chart(fig_map, use_container_width=True)
            if num_villages > 1:
                st.info(f"These clusters span {num_villages} villages, including {selected_village}.")
        else:
            # For specific cluster, use full project_df to show all projects in the cluster
            full_filtered = project_df[project_df[cluster_type] == selected_cluster]
            num_villages = len(full_filtered['Village'].unique())
            title = f"Cluster Map - Cluster {selected_cluster}"
            if num_villages > 1:
                title += f" (Spans {num_villages} Villages)"
            fig_map = plot_cluster_map(full_filtered, cluster_type, selected_cluster, title)
            if fig_map:
                st.plotly_chart(fig_map, use_container_width=True)
            if num_villages > 1:
                st.info(f"This cluster spans {num_villages} villages, including {selected_village}.")
            else:
                st.caption(f"Showing projects in cluster {selected_cluster} from {selected_village}.")
        
        category_map = {'Cluster_LatLong': 'LatLong', 
                       'Cluster_LatLongCategory': 'LatLongCategory'}
        category = category_map.get(cluster_type, 'LatLong')
        st.subheader("Regression Analysis")
        st.caption("Regression models illustrating the relationship between variables (amenity score, road category) and property rate.")
        if selected_cluster != 'All':
            show_regression_visuals(regression_data, selected_cluster, category)
        else:
            st.info("Regression analysis is not available for 'All' clusters. Please select a specific cluster.")
    
    with tab2:
        st.header("Location Analyzer")
        st.caption("Analyze a specific location to evaluate nearby highways, amenities, and predicted property rates.")
        
        with st.sidebar:
            st.markdown("### Amenity Weights")
            st.caption("Adjust the weights for different amenity categories to influence the amenity score calculation.")
            categories = list(DEFAULT_WEIGHTS.keys())
            custom_weights = {}
            total_weight = 0
            
            for cat in categories:
                weight = st.number_input(
                    cat, min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHTS[cat],
                    step=0.01, format="%.2f", key=f"wt_{cat}",
                    help=f"Weight for {cat} amenities (0.0 to 1.0). Higher weights increase their impact on the amenity score."
                )
                custom_weights[cat] = weight
                total_weight += weight
            
            st.metric("Total Weight", f"{total_weight:.2f}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Coordinates**")
        with col2:
            coord_input = st.text_input("", value="18.5530, 73.7589",
                                      help="Enter coordinates in the format: latitude, longitude (e.g., 18.5530, 73.7589)")
        
        try:
            lat, lon = map(float, coord_input.split(','))
        except:
            st.error("Invalid format! Use: latitude,longitude")
            st.stop()
        
        if st.button("Analyze Location", type="primary", use_container_width=True):
            current_weights = {cat: st.session_state[f"wt_{cat}"] for cat in categories}
            
            # 1. Find Cluster
            cluster_info, dist_km, nearest_project = find_nearest_cluster(project_df, lat, lon)
            
            # 2. Find Highways
            with st.spinner("Querying nearby highways..."):
                all_highways, nearest_biggest_highway = get_highways_within_radius(lat, lon)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if dist_km < 0.5:
                    st.success("Within Cluster")
                else:
                    st.warning(f"Nearest Cluster ({dist_km:.1f} km)")
            with col2:
                st.metric("LatLong Cluster", cluster_info.get('Cluster_LatLong', 'N/A'))
            with col3:
                st.metric("Category Cluster", cluster_info.get('Cluster_LatLongCategory', 'N/A'))
            with col4:
                if nearest_biggest_highway['category']:
                    st.metric("Nearest Major Highway", f"{nearest_biggest_highway['category_label']} ({nearest_biggest_highway['distance_m']:.0f}m)")
                else:
                    st.metric("Nearest Major Highway", "None found")
            
            # 3. Display All Highways
            st.markdown("### Nearby Highways (200m Radius)")
            st.caption("List of highways within 200 meters, sourced from OpenStreetMap, sorted by proximity.")
            if all_highways:
                highway_df = pd.DataFrame(all_highways)
                highway_df = highway_df[['name', 'highway', 'category', 'category_label', 'distance_m']].rename(
                    columns={
                        'name': 'Road Name',
                        'highway': 'Highway Type',
                        'category': 'Category',
                        'category_label': 'Category Description',
                        'distance_m': 'Distance (m)'
                    }
                )
                highway_df['Distance (m)'] = highway_df['Distance (m)'].round(1)
                styled_df = highway_df.style.set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#667eea'), ('color', 'white'), ('font-weight', 'bold')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#f0f2f6')]}
                ]).format({'Distance (m)': '{:.1f}'})
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("No highways found within 200m.")
            
            # 4. Amenity Analysis
            st.markdown("---")
            st.markdown("### Amenity Score Analysis (1km Radius)")
            st.caption("Calculated amenity score based on the proximity and type of amenities within a 1km radius.")
            
            with st.spinner("Calculating amenities..."):
                category_df = calculate_amenity_scores(lat, lon, all_amenities, current_weights)
                detailed_df = get_detailed_amenities(lat, lon, all_amenities)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Amenity Score", f"{category_df['total_score'].iloc[0]:.3f}" if not category_df.empty else "0.000")
            with col_m2:
                st.metric("Amenities Found", len(detailed_df))
            
            st.subheader("Amenity Category Breakdown")
            st.caption("Breakdown of amenity scores by category, including count and weighted contributions.")
            if not category_df.empty:
                styled_category = category_df[['category', 'count', 'S_c', 's_c', 'weight', 'Weight × s_c']].round(3).style.set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#667eea'), ('color', 'white'), ('font-weight', 'bold')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#f0f2f6')]}
                ]).format(precision=3)
                st.dataframe(styled_category, use_container_width=True, hide_index=True)
            else:
                st.info("No amenity data available.")
            
            st.subheader("Amenities and Highways Map")
            st.caption("Interactive map displaying amenities (colored by category) and highways (colored lines by type) within the search radius. Hover over points or lines for details.")
            if not detailed_df.empty:
                detailed_df['hover_text'] = detailed_df.apply(
                    lambda row: f"{row['name']}<br>{row['category']}<br>{row['distance_m']:.0f}m", axis=1
                )
                
                fig_amenity = px.scatter_mapbox(
                    detailed_df, lat='lat', lon='lng', hover_name='hover_text',
                    color='category', size='f_d', size_max=15,
                    color_continuous_scale='viridis',  # Attractive color scale
                    zoom=14, height=500, center={"lat": lat, "lon": lon}
                )
                fig_amenity.update_traces(
                    marker=dict(opacity=0.8, size=10)
                )
                fig_amenity.add_trace(go.Scattermapbox(
                    lat=[lat], lon=[lon], mode='markers',
                    marker=dict(size=20, color='red', symbol='star'),
                    name="Selected Location", hovertemplate="<b>Selected Location</b>"
                ))
                # Add highways to the map with colored lines based on category
                category_colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red'}
                for highway in all_highways:
                    if highway['geometry']:
                        lons, lats = zip(*highway['geometry'])
                        fig_amenity.add_trace(go.Scattermapbox(
                            lon=lons,
                            lat=lats,
                            mode='lines',
                            line=dict(width=3, color=category_colors.get(highway['category'], 'gray')),
                            name=f"{highway['name']} ({highway['category']})",
                            hovertemplate=f"Road: {highway['name']}<br>Category: {highway['category_label']}<extra></extra>"
                        ))
                fig_amenity.update_layout(
                    mapbox_style="open-street-map",
                    legend=dict(
                        title="Legend",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    ),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                fig_amenity.add_annotation(
                    text="Note: Amenities are colored by category; highways by type (A: green, B: blue, C: orange, D: red).",
                    xref="paper", yref="paper", x=0.01, y=0.01,
                    showarrow=False, font=dict(size=12, color="gray"),
                    bgcolor="white", bordercolor="gray", borderwidth=1
                )
                st.plotly_chart(fig_amenity, use_container_width=True)
            else:
                st.info("No amenities found within 1km.")
            
            # 5. Cluster Map
            if dist_km >= 0.5:
                st.markdown("### Nearest Cluster Projects")
                st.caption("Map of projects in the nearest cluster, showing their locations and rates.")
                for col, clus in cluster_info.items():
                    if pd.notna(clus):
                        fig = plot_cluster_map(project_df, col, clus, f"Nearest {col}: {clus}")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        break
            
            # 6. Regression Analysis
            selected_cluster = cluster_info.get('Cluster_LatLong', cluster_info.get('Cluster_LatLongCategory'))
            if pd.notna(selected_cluster):
                st.markdown("### Regression Analysis")
                st.caption("Regression models illustrating the relationship between variables (amenity score, road category) and property rate.")
                category = 'LatLong' if 'Cluster_LatLong' in cluster_info else 'LatLongCategory'
                show_regression_visuals(regression_data, selected_cluster, category)
            
            # 7. Valuation Prediction
            st.markdown("### Valuation Prediction")
            st.caption("Estimated property rate per square foot based on regression models using amenity score and road category.")
            
            highway_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            road = highway_map.get(nearest_biggest_highway['category'], 2)
            predicted_amenity = float(category_df['total_score'].iloc[0]) if not category_df.empty else 0.0
            
            # LatLong Cluster Prediction using Both (amenity + road)
            latlong_pred = 'N/A'
            latlong_eq = 'N/A'
            sheet_latlong = 'LatLong_Both_vs_Rate'
            if sheet_latlong in regression_data and not regression_data[sheet_latlong].empty:
                row = regression_data[sheet_latlong][regression_data[sheet_latlong]['Cluster'] == cluster_info.get('Cluster_LatLong')]
                if not row.empty:
                    row = row.iloc[0]
                    latlong_pred = row['Slope_Amenity'] * predicted_amenity + row['Slope_RoadCat'] * road + row['Intercept']
                    latlong_eq = row['Equation']
            
            # LatLongCategory Cluster Prediction using Amenity only
            category_pred = 'N/A'
            category_eq = 'N/A'
            sheet_category = 'LatLongCategory_Amenity_vs_Rate'
            if sheet_category in regression_data and not regression_data[sheet_category].empty:
                row = regression_data[sheet_category][regression_data[sheet_category]['Cluster'] == cluster_info.get('Cluster_LatLongCategory')]
                if not row.empty:
                    row = row.iloc[0]
                    category_pred = row['Slope_Amenity'] * predicted_amenity + row['Intercept']
                    category_eq = row['Equation']
            
            st.subheader("Prediction Results")
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.markdown("**LatLong Cluster Prediction**")
                st.info(f"Predicted Rate: ₹{latlong_pred:.0f} per sqft" if latlong_pred != 'N/A' else "No data available")
                st.caption(f"Model: {latlong_eq} (uses amenity score and road category: A=1, B=2, C=3, D=4)")
            with col_pred2:
                st.markdown("**LatLongCategory Cluster Prediction**")
                st.info(f"Predicted Rate: ₹{category_pred:.0f} per sqft" if category_pred != 'N/A' else "No data available")
                st.caption(f"Model: {category_eq} (uses amenity score)")

if __name__ == "__main__":
    main()
