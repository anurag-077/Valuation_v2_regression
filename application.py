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
    
    print("Columns in project_df:", df.columns.tolist())
    
    df = df.rename(columns={
        'Latitude': 'project_lat', 'longitude': 'project_lng',
        'lat': 'project_lat', 'lng': 'project_lng',
        'Project_Latitude': 'project_lat', 'Project_Longitude': 'project_lng'
    })
    
    required_columns = ['project_lat', 'project_lng', 'Project_Name', 'Mid_Rate', 'Village', 'Cluster_LatLong']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns in project data: {', '.join(missing_columns)}")
        st.stop()
    
    # Clean data: Remove rows with NaN in critical columns and ensure numeric coordinates
    df = df.dropna(subset=['project_lat', 'project_lng', 'Village', 'Mid_Rate', 'Project_Name'])
    df['project_lat'] = pd.to_numeric(df['project_lat'], errors='coerce')
    df['project_lng'] = pd.to_numeric(df['project_lng'], errors='coerce')
    df = df.dropna(subset=['project_lat', 'project_lng'])
    
    if df.empty:
        st.error("No valid project data after cleaning. Please check the data file.")
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
            logging.warning(f"Skipping {file}: type_name {type_name} not in AMENITY_TYPES")
            continue
        
        group_name = AMENITY_TYPES[type_name]
        file_path = os.path.join(amenity_dir, file)
        
        try:
            df = pd.read_excel(file_path)
            if 'lat' not in df.columns or 'lng' not in df.columns:
                logging.warning(f"Skipping {file}: missing 'lat' or 'lng' columns")
                continue
                
            df = df.dropna(subset=['lat', 'lng'])
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
            df = df.dropna(subset=['lat', 'lng'])
            
            if df.empty:
                logging.warning(f"Skipping {file}: no valid data after cleaning")
                continue
            
            if 'name' in df.columns:
                df['name'] = df['name'].fillna('Unnamed')
            else:
                df['name'] = f"{type_name.capitalize()}-{pd.Series(range(1, len(df)+1))}"
            
            df['category'] = group_name
            df['type_name'] = type_name
            data.append(df[['lat', 'lng', 'category', 'type_name', 'name']])
            logging.info(f"Loaded {len(df)} {type_name} amenities from {file}")
            
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            continue
    
    if not data:
        st.warning("No valid amenity data loaded. Using sample data.")
        return create_sample_data()
    
    result = pd.concat(data, ignore_index=True)
    logging.info(f"Total Amenities Loaded: {len(result)}")
    return result

def create_sample_data():
    """Generate sample amenity data for testing."""
    sample_data = [
        {'lat': 18.5530, 'lng': 73.7589, 'name': 'Metro Station 1', 'type_name': 'metro_station', 'category': 'Metro'},
        {'lat': 18.5540, 'lng': 73.7590, 'name': 'Metro Station 2', 'type_name': 'metro_station', 'category': 'Metro'},
        {'lat': 18.5525, 'lng': 73.7595, 'name': 'Bus Stop 1', 'type_name': 'bus_stop', 'category': 'Bus'},
        {'lat': 18.5535, 'lng': 73.7585, 'name': 'Mall 1', 'type_name': 'mall', 'category': 'Mall'},
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
        logging.warning("all_amenities is empty")
        return pd.DataFrame()
    
    lats = all_amenities['lat'].values
    lons = all_amenities['lng'].values
    if len(lats) == 0 or len(lons) == 0:
        logging.warning("No valid coordinates in all_amenities")
        return pd.DataFrame()
    
    dists = haversine_vectorized(lat, lon, lats, lons)
    logging.info(f"Calculated distances for {len(dists)} amenities, max distance: {dists.max():.1f}m")
    
    mask = dists <= POI_SEARCH_RADIUS_M
    if not np.any(mask):
        logging.warning(f"No amenities within {POI_SEARCH_RADIUS_M}m of lat={lat}, lon={lon}")
        return pd.DataFrame()
    
    filtered_df = all_amenities[mask].copy()
    filtered_df['distance_m'] = dists[mask]
    filtered_df['f_d'] = 1 / (1 + filtered_df['distance_m'] / 200)
    logging.info(f"Filtered {len(filtered_df)} amenities within {POI_SEARCH_RADIUS_M}m")
    
    # Validate and clean category column
    if 'category' not in filtered_df.columns or filtered_df['category'].isna().all():
        logging.warning("No valid 'category' column in filtered_df")
        return pd.DataFrame()
    filtered_df = filtered_df.dropna(subset=['category'])
    if filtered_df.empty:
        logging.warning("No amenities with valid categories after filtering")
        return pd.DataFrame()
    
    # Calculate category counts
    category_counts = filtered_df.groupby('category').size()
    logging.info(f"Category counts: {category_counts.to_dict()}")
    
    category_scores = filtered_df.groupby('category')['f_d'].sum().reset_index()
    category_scores.columns = ['category', 'S_c']
    category_scores['s_c'] = 1 - np.exp(-0.8 * category_scores['S_c'])
    category_scores['weight'] = category_scores['category'].map(weights).fillna(0)
    category_scores['Weight × s_c'] = category_scores['weight'] * category_scores['s_c']
    category_scores['count'] = category_scores['category'].map(category_counts).fillna(0).astype(int)
    
    total_score = category_scores['Weight × s_c'].sum()
    category_scores['total_score'] = total_score
    
    for cat in weights:
        if cat not in category_scores['category'].values:
            category_scores = pd.concat([category_scores, pd.DataFrame({
                'category': [cat], 'S_c': [0], 's_c': [0], 'weight': [weights[cat]],
                'Weight × s_c': [0], 'count': [0], 'total_score': [total_score]
            })], ignore_index=True)
    
    logging.info(f"Category scores: {category_scores.to_dict()}")
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

def find_nearest_cluster(df, lat, lon, cluster_cols=['Cluster_LatLong', 'Cluster_LatLongCategory'], max_dist_km=5.0):
    if df.empty:
        return {}, float('inf'), None

    distances = haversine_distance(lat, lon, df['project_lat'].values, df['project_lng'].values)
    min_idx = np.argmin(distances)
    min_dist_km = distances[min_idx] / 1000

    if min_dist_km > max_dist_km:
        logging.warning(f"Nearest project is {min_dist_km:.1f} km away (> {max_dist_km} km). No cluster assigned.")
        return {}, min_dist_km, None  # No cluster assigned

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

        road_distances.sort(key=lambda r: (-r["increase_pct"], r["distance_m"]))
        nearest_biggest_highway = road_distances[0]
        all_highways_sorted = sorted(road_distances, key=lambda r: r["distance_m"])
        return all_highways_sorted, nearest_biggest_highway

    except Exception as e:
        logging.error(f"Error querying roads for lat={lat}, lon={lon}: {e}")
        return [], {"category": None, "category_label": None, "distance_m": None, "name": None}

def cluster_summary(df: pd.DataFrame, cluster_col: str, cluster_id) -> dict:
    """Return a dict with min / max / percentiles for a given cluster."""
    if pd.isna(cluster_id):
        return {}
    sub = df[df[cluster_col] == cluster_id]['Mid_Rate']
    if sub.empty:
        return {}
    return {
        "Min": sub.min(),
        "Max": sub.max(),
        "Average": sub.mean(),
        "50th (Median)": sub.quantile(0.50),
        "75th": sub.quantile(0.75),
        "90th": sub.quantile(0.90),
        "95th": sub.quantile(0.95),
    }


def plot_cluster_map(df, cluster_col, cluster_num, title="Cluster Map", subject_lat=None, subject_lon=None):
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
        color='Mid_Rate',
        color_continuous_scale='viridis',
        zoom=12,  # Reduced zoom for more context
        height=500,
        title=f"{title} - Projects in Cluster {cluster_num if cluster_num is not None else 'All'}",
        labels={'Mid_Rate': 'Rate (₹ per sqft)'}
    )
    
    fig.update_traces(
        marker=dict(size=14, opacity=0.8)
    )
    
    if cluster_num is not None:
        points = filtered[['project_lng', 'project_lat']].values
        if len(points) >= 3:
            try:
                hull = ConvexHull(points, qhull_options="QJ")
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
                        fill='none',
                        name="Cluster Boundary",
                        hoverinfo='skip'
                    )
                )
            except Exception as e:
                logging.warning(f"Failed to compute convex hull for cluster {cluster_num}: {e}")
                st.warning(f"Could not draw boundary for cluster {cluster_num} due to insufficient point variation.")
    
    # Add subject location as the last trace to ensure it's on top
    if subject_lat is not None and subject_lon is not None:
        logging.info(f"Adding subject location marker at lat={subject_lat}, lon={subject_lon}")
        fig.add_trace(go.Scattermapbox(
            lat=[subject_lat],
            lon=[subject_lon],
            mode='markers',
            marker=dict(
                size=30,
                color='red',
                symbol='circle',
                opacity=1.0,
                sizemode='area'
            ),
            name="Subject Location",
            hovertemplate="<b>Subject Location</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
            showlegend=True
        ))
        # Debug: Log the trace details
        logging.info(f"Subject location trace added: {fig.data[-1]}")
    
    # Center the map on the subject location if provided
    if subject_lat is not None and subject_lon is not None:
        fig.update_layout(
            mapbox=dict(
                center=dict(lat=subject_lat, lon=subject_lon),
                zoom=12  # Match initial zoom
            )
        )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(t=60, b=60),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title="Rate (₹ per sqft)",
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=18)
        )
    )
    
    fig.add_annotation(
        text="Note: Points represent projects colored by their mid rate. Cluster boundaries shown where applicable. Subject location marked with a red circle. Use mouse scroll or zoom buttons to adjust view.",
        xref="paper", yref="paper", x=0.01, y=0.01,
        showarrow=False, font=dict(size=12, color="gray"),
        bgcolor="white", bordercolor="gray", borderwidth=1
    )
    
    # Enable interactive controls
    fig.update_layout(
        mapbox=dict(
            pitch=0,
            bearing=0,
            zoom=12  # Ensure consistent zoom
        ),
        dragmode='zoom',  # Enable zoom/pan interactivity
        uirevision='map'  # Preserve map state on updates
    )
    
    return fig

def show_regression_visuals(regression_data, cluster_num, category):
    available_sheets = []
    if category == 'LatLong':
        available_sheets = ['LatLong_Amenity_vs_Rate', 'LatLong_RoadCat_vs_Rate', 'LatLong_Both_vs_Rate']
    elif category == 'LatLongCategory':
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
    
    for sheet, row in cluster_data.items():
        st.subheader(sheet.replace('_vs_Rate', ''))
        slope2 = None
        x_label2 = None
        x_range2 = None
        if 'Both' in sheet:
            title = "Combined Amenity & Road vs Rate"
            slope1 = row['Slope_Amenity']
            slope2 = row['Slope_RoadCat']
            x_label1 = "Amenity Score (0-1)"
            x_label2 = "Road Category (1-4)"
            x_range1 = 1
            x_range2 = 4
            fig = create_regression_plot(
                row['Equation'], slope1, row['Intercept'],
                x_label1, "Rate (₹ per sqft)", row['Num_Projects'], x_range1, title,
                slope2=slope2, x_label2=x_label2, x_range2=x_range2
            )
        elif 'Amenity' in sheet:
            title = "Amenity Score vs Rate"
            x_range1 = 1
            x_label1 = "Amenity Score (0-1)"
            slope1 = row['Slope_Amenity']
            fig = create_regression_plot(
                row['Equation'], slope1, row['Intercept'],
                x_label1, "Rate (₹ per sqft)", row['Num_Projects'], x_range1, title
            )
        elif 'RoadCat' in sheet:
            title = "Road Category vs Rate"
            x_range1 = 4
            x_label1 = "Road Category (1-4)"
            slope1 = row['Slope_RoadCat']
            fig = create_regression_plot(
                row['Equation'], slope1, row['Intercept'],
                x_label1, "Rate (₹ per sqft)", row['Num_Projects'], x_range1, title
            )
        else:
            title = "Amenity Score vs Rate"
            x_range1 = 1
            x_label1 = "Amenity Score (0-1)"
            slope1 = row['Slope_Amenity']
            fig = create_regression_plot(
                row['Equation'], slope1, row['Intercept'],
                x_label1, "Rate (₹ per sqft)", row['Num_Projects'], x_range1, title
            )
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_{sheet}_{cluster_num}")
    
    eqs = []
    for sheet, row in cluster_data.items():
        eqs.append({
            'Model': sheet.replace('_vs_Rate', ''),
            'Equation': row['Equation'],
            'Sample Size': row['Num_Projects']
        })
    st.dataframe(pd.DataFrame(eqs).style.set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', '#f0f2f6')]}]), 
                 use_container_width=True, 
                 key=f"df_regression_{cluster_num}")

def create_regression_plot(equation, slope, intercept, x_label, y_label, n, x_range, title, slope2=None, x_label2=None, x_range2=None):
    if slope2 is not None:
        # 3D plot for multiple regression
        x = np.linspace(0, x_range, 20)
        y = np.linspace(1, x_range2, 20)
        x, y = np.meshgrid(x, y)
        z = slope * x + slope2 * y + intercept
        
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='viridis')])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=x_label2,
                zaxis_title=y_label,
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                zaxis=dict(showgrid=True, gridcolor='lightgray')
            ),
            height=400,
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False,
            plot_bgcolor='white'
        )
        
        fig.add_annotation(
            x=0.05, y=0.05, xref="paper", yref="paper",
            text=f"<b>{equation}</b><br>Sample Size: {n}",
            showarrow=False, font=dict(size=12), 
            bgcolor="white", bordercolor="#1f77b4", borderwidth=1
        )
        
        fig.add_annotation(
            text="Note: Surface represents the regression model fit.",
            xref="paper", yref="paper", x=0.01, y=0.01,
            showarrow=False, font=dict(size=12, color="gray"),
            bgcolor="white", bordercolor="gray", borderwidth=1
        )
    else:
        # 2D plot for single variable
        x = np.linspace(1, x_range, 100) if 'Road Category' in x_label else np.linspace(0, x_range, 100)
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
            height=400,
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
    st.set_page_config(page_title="Valuation Analyzer Pro", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: white;
        color: black;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    .stMarkdown, .stCaption {
        color: black;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
    }
    .stSelectbox div, .stTextInput div {
        background-color: white;
        color: black;
    }
    .dataframe {
        background-color: white;
    }
    body {
        color: black;
        background-color: white;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
        st.caption("Explore project clusters on a map based on geographic or categorical groupings. Selecting a village shows all projects in clusters containing at least one project from that village. Clusters are predefined based on project locations and attributes.")
        
        if project_df.empty:
            st.error("No project data available. Please ensure 'All_Project_data_WITH_Amenity_Scores.xlsx' contains valid data.")
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_type = st.selectbox("Cluster Type", 
                                      ['Cluster_LatLong', 'Cluster_LatLongCategory'],
                                      help="Select the type of clustering: LatLong (geographic) or LatLongCategory (geographic and categorical).")
        
        with col2:
            villages = sorted(project_df['Village'].dropna().unique())
            if not villages:
                st.error("No valid villages found in the data. Please check the 'Village' column in the project data.")
                st.stop()
            selected_village = st.selectbox("Select Village", villages,
                                          help="Choose a village to filter clusters. Shows all projects in clusters that include this village.")
            filtered_df = project_df[project_df['Village'] == selected_village]
            if filtered_df.empty:
                st.warning(f"No projects found for village: {selected_village}")
                st.stop()
            clusters = sorted(filtered_df[cluster_type].dropna().unique())
            if not clusters:
                st.warning(f"No valid clusters found for {cluster_type} in village: {selected_village}")
                st.stop()
            clusters = ['All'] + clusters
            selected_cluster = st.selectbox("Cluster Number", clusters,
                                          help="Choose a specific cluster number to visualize all projects in that cluster, or select 'All' to see all relevant clusters.")
        
        if selected_cluster == 'All':
            relevant_clusters = project_df[project_df[cluster_type].isin(clusters[1:])]
            if relevant_clusters.empty:
                st.warning(f"No projects found in clusters associated with {selected_village}. Try another village or cluster type.")
                st.stop()
            relevant_clusters['hover_text'] = relevant_clusters.apply(
                lambda row: f"<b>{row['Project_Name']}</b><br>₹{row['Mid_Rate']:.1f} per sqft<br>{row['Village']}", axis=1
            )
            num_villages = len(relevant_clusters['Village'].unique())
            fig_map = px.scatter_mapbox(
                relevant_clusters,
                lat='project_lat', lon='project_lng',
                hover_name='hover_text',
                color=cluster_type,
                color_discrete_sequence=px.colors.qualitative.Plotly,
                zoom=11,
                height=500,
                title=f"Cluster Map - All Projects in Clusters with {selected_village} (Spans {num_villages} Villages)",
                labels={cluster_type: 'Cluster', 'Mid_Rate': 'Rate (₹ per sqft)'}
            )
            fig_map.update_traces(
                marker=dict(size=14, opacity=0.8)
            )
            colors = px.colors.qualitative.Plotly
            unique_clusters = relevant_clusters[cluster_type].unique()
            for idx, cluster in enumerate(unique_clusters):
                cluster_data = relevant_clusters[relevant_clusters[cluster_type] == cluster]
                points = cluster_data[['project_lng', 'project_lat']].values
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points, qhull_options="QJ")
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
                                fill='none',
                                name=f"Cluster {cluster} Boundary",
                                hoverinfo='skip'
                            )
                        )
                    except Exception as e:
                        logging.warning(f"Failed to compute convex hull for cluster {cluster}: {e}")
                        st.warning(f"Could not draw boundary for cluster {cluster} due to insufficient point variation.")
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(t=60, b=60),
                hovermode='closest',
                legend=dict(
                    title="Clusters",
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray",
                    borderwidth=1
                )
            )
            fig_map.add_annotation(
                text="Note: Points represent projects colored by their cluster. Clusters include projects from selected village and others.",
                xref="paper", yref="paper", x=0.01, y=0.01,
                showarrow=False, font=dict(size=12, color="gray"),
                bgcolor="white", bordercolor="gray", borderwidth=1
            )
            st.plotly_chart(fig_map, use_container_width=True, key="cluster_map_all")
            if num_villages > 1:
                st.info(f"These clusters span {num_villages} villages, including {selected_village}.")
        else:
            full_filtered = project_df[project_df[cluster_type] == selected_cluster]
            if full_filtered.empty:
                st.warning(f"No projects found in {cluster_type} = {selected_cluster}. Try another cluster.")
                st.stop()
            num_villages = len(full_filtered['Village'].unique())
            title = f"Cluster Map - Cluster {selected_cluster}"
            if num_villages > 1:
                title += f" (Spans {num_villages} Villages)"
            fig_map = plot_cluster_map(full_filtered, cluster_type, selected_cluster, title)
            if fig_map:
                st.plotly_chart(fig_map, use_container_width=True, key=f"cluster_map_{selected_cluster}")
            if num_villages > 1:
                st.info(f"This cluster spans {num_villages} villages, including {selected_village}.")
            else:
                st.caption(f"Showing projects in cluster {selected_cluster} from {selected_village}.")
        
        category_map = {'Cluster_LatLong': 'LatLong', 
                       'Cluster_LatLongCategory': 'LatLongCategory'}
        category = category_map.get(cluster_type, 'LatLong')
        st.subheader("Regression Analysis")
        st.caption("Regression models help us understand how property prices change based on factors such as nearby amenities and road quality in a given area or cluster.")
        if selected_cluster != 'All':
            show_regression_visuals(regression_data, selected_cluster, category)
        else:
            st.info("Regression analysis is not available for 'All' clusters. Please select a specific cluster.")
    
    with tab2:
        st.header("Location Analyzer")
        st.caption("Analyze a specific location to evaluate nearby highways, amenities, and predicted property rates. Select a cluster type to view the corresponding cluster map with the subject location highlighted.")
        
        with st.sidebar:
            st.markdown("### Amenity Weights")
            st.caption("Adjust the weights for different amenity categories to influence the amenity score calculation. Standard weights are assigned as follows: Metro (0.25), Bus (0.15), Mall (0.225), School (0.225), Hospital (0.075), Garden (0.075). Customize them to your preference.")
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
            coord_input = st.text_input("", value=st.session_state.get('coord_input', "18.5530, 73.7589"),
                                       key="coord_input",
                                       help="Enter coordinates in the format: latitude, longitude (e.g., 18.5530, 73.7589)")
        
        try:
            lat, lon = map(float, coord_input.split(','))
        except:
            st.error("Invalid format! Use: latitude,longitude")
            st.stop()
        
        # Initialize session state
        if 'analysis_done' not in st.session_state:
            st.session_state['analysis_done'] = False
        if 'selected_cluster_type' not in st.session_state:
            st.session_state['selected_cluster_type'] = 'Cluster_LatLong'
        
        if st.button("Analyze Location", type="primary", use_container_width=True):
            st.session_state['analysis_done'] = True
            st.session_state['lat'] = lat
            st.session_state['lon'] = lon
            current_weights = {cat: st.session_state[f"wt_{cat}"] for cat in categories}
            st.session_state['current_weights'] = current_weights
        
        if st.session_state['analysis_done']:
            lat = st.session_state['lat']
            lon = st.session_state['lon']
            current_weights = st.session_state['current_weights']
            
            cluster_info, dist_km, nearest_project = find_nearest_cluster(project_df, lat, lon)
            
            with st.spinner("Querying nearby highways..."):
                all_highways, nearest_biggest_highway = get_highways_within_radius(lat, lon)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if dist_km < 0.5:
                    st.success("Within Cluster")
                else:
                    st.warning(f"Nearest Cluster ({dist_km:.1f} km)")
            with col2:
                latlong_cluster = cluster_info.get('Cluster_LatLong', 'N/A')
                if st.button("Show LatLong Cluster Map", key="latlong_cluster_button"):
                    st.session_state['selected_cluster_type'] = 'Cluster_LatLong'
                    st.rerun()
                st.metric("LatLong Cluster", latlong_cluster)
            with col3:
                category_cluster = cluster_info.get('Cluster_LatLongCategory', 'N/A')
                if st.button("Show Category Cluster Map", key="category_cluster_button"):
                    st.session_state['selected_cluster_type'] = 'Cluster_LatLongCategory'
                    st.rerun()
                st.metric("Category Cluster", category_cluster)
            with col4:
                if nearest_biggest_highway['category']:
                    st.metric("Nearest Major Highway", f"{nearest_biggest_highway['category_label']} ({nearest_biggest_highway['distance_m']:.0f}m)")
                else:
                    st.metric("Nearest Major Highway", "None found")
            
            st.markdown("### Nearest Cluster Projects")
            st.caption("Map of projects in the selected cluster, with the subject location highlighted as a red circle. Use buttons above to switch cluster type.")
            cluster_type = st.session_state['selected_cluster_type']
            selected_cluster = cluster_info.get(cluster_type)
            if pd.notna(selected_cluster):
                fig = plot_cluster_map(project_df, cluster_type, selected_cluster, title=f"Nearest {cluster_type}: {selected_cluster}", subject_lat=lat, subject_lon=lon)
                if fig:
                    # Debug: Print trace details
                    logging.info(f"Cluster map traces: {fig.data}")
                    st.plotly_chart(fig, use_container_width=True, key=f"cluster_map_{cluster_type}_{selected_cluster}")
            else:
                st.warning(f"No valid {cluster_type} cluster found for this location.")
            
            st.markdown("### Nearby Highways (200m Radius)")
            st.caption("List of highways within a 200-meter radius of the subject location, sourced from OpenStreetMap, sorted by proximity. The widest highway is considered for valuation impact.")
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
            
            st.markdown("---")
            st.markdown("### Amenity Score Analysis (1km Radius)")
            st.caption("Calculated amenity score based on all amenities within a 1km radius of the subject location, using standard weights adjustable in the sidebar.")
            
            with st.spinner("Calculating amenities..."):
                category_df = calculate_amenity_scores(lat, lon, all_amenities, current_weights)
                detailed_df = get_detailed_amenities(lat, lon, all_amenities)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Amenity Score", f"{category_df['total_score'].iloc[0]:.3f}" if not category_df.empty else "0.000")
            with col_m2:
                st.metric("Amenities Found", len(detailed_df))
            
            st.subheader("Amenity Category Breakdown")
            st.caption("Breakdown of amenity scores by category, including count and weighted contributions based on amenities within 1km.")
            if not category_df.empty:
                if category_df['count'].sum() == 0:
                    st.warning("Unexpected zero count for amenities. This may indicate a data or filtering issue. Check logs for details.")
                styled_category = category_df[['category', 'count', 'S_c', 's_c', 'weight', 'Weight × s_c']].round(3).style.set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#667eea'), ('color', 'white'), ('font-weight', 'bold')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#f0f2f6')]}
                ]).format(precision=3)
                st.dataframe(styled_category, use_container_width=True, hide_index=True)
            else:
                st.info("No amenity data available.")
            
            with st.expander("View Detailed Amenities (1km Radius)", expanded=False):
                st.caption("Detailed list of all amenities within a 1km radius, including their type, category, distance, and influence factor (f_d).")
                if not detailed_df.empty:
                    amenity_df = detailed_df[['name', 'type_name', 'category', 'distance_m', 'f_d']].rename(
                        columns={
                            'name': 'Amenity Name',
                            'type_name': 'Type',
                            'category': 'Category',
                            'distance_m': 'Distance (m)',
                            'f_d': 'Influence Factor (f_d)'
                        }
                    )
                    amenity_df['Distance (m)'] = amenity_df['Distance (m)'].round(1)
                    amenity_df['Influence Factor (f_d)'] = amenity_df['Influence Factor (f_d)'].round(3)
                    styled_amenity_df = amenity_df.style.set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#667eea'), ('color', 'white'), ('font-weight', 'bold')]},
                        {'selector': 'tr:hover', 'props': [('background-color', '#f0f2f6')]}
                    ]).format({
                        'Distance (m)': '{:.1f}',
                        'Influence Factor (f_d)': '{:.3f}'
                    })
                    st.dataframe(styled_amenity_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No amenities found within 1km.")
            
            st.markdown("### Amenities and Highways Map")
            st.caption("Interactive map displaying all amenities within a 1km radius (colored by category), highways within 200m (colored lines by type, with the widest highway considered for valuation), and the subject location (red circle). Hover for details.")
            if not detailed_df.empty:
                detailed_df['hover_text'] = detailed_df.apply(
                    lambda row: f"{row['name']}<br>{row['category']}<br>{row['distance_m']:.0f}m", axis=1
                )
                
                fig_amenity = px.scatter_mapbox(
                    detailed_df, 
                    lat='lat', 
                    lon='lng', 
                    hover_name='hover_text',
                    color='category',
                    size='f_d', 
                    size_max=12,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    zoom=14, 
                    height=600,
                    center={"lat": lat, "lon": lon},
                    title="Nearby Amenities and Highways"
                )
                
                fig_amenity.update_traces(
                    marker=dict(
                        opacity=0.85,
                        sizemin=6
                    ),
                    hovertemplate="%{hovertext}<extra></extra>"
                )
                
                category_colors = {
                    'A': '#28a745',  # Green
                    'B': '#007bff',  # Blue
                    'C': '#fd7e14',  # Orange
                    'D': '#dc3545'   # Red
                }
                for highway in all_highways:
                    if highway['geometry']:
                        lons, lats = zip(*highway['geometry'])
                        fig_amenity.add_trace(go.Scattermapbox(
                            lon=lons,
                            lat=lats,
                            mode='lines',
                            line=dict(
                                width=4,
                                color=category_colors.get(highway['category'], '#6c757d')
                            ),
                            name=f"{highway['name']} ({highway['category']})",
                            hovertemplate=f"<b>Road: {highway['name']}</b><br>Category: {highway['category_label']}<br>Distance: {highway['distance_m']:.0f}m<extra></extra>",
                            below=''  # Ensure highways are below markers
                        ))
                
                fig_amenity.add_trace(go.Scattermapbox(
                    lat=[lat], 
                    lon=[lon], 
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='circle',
                        opacity=1.0,
                        sizemode='diameter'
                    ),
                    name="Subject Location",
                    hovertemplate="<b>Subject Location</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
                ))
                
                traces = list(fig_amenity.data)
                subject_trace = traces[-1]  # Subject location is the last added
                other_traces = traces[:-1]
                fig_amenity.data = tuple(other_traces + [subject_trace])
                
                fig_amenity.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(l=20, r=20, t=60, b=80),
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(
                        title="Amenities & Highways",
                        yanchor="top",
                        y=1.0,
                        xanchor="right",
                        x=0.98,
                        orientation="v",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=11),
                        itemsizing='constant'
                    ),
                    title=dict(
                        text="Nearby Amenities and Highways",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=18, color="black")
                    )
                )
                
                fig_amenity.add_annotation(
                    text="Note: Amenities (within 1km) are colored by category; highways (within 200m) by type (A: green, B: blue, C: orange, D: red). Subject location marked with a red circle.",
                    xref="paper", 
                    yref="paper", 
                    x=0.01, 
                    y=0.01,
                    showarrow=False, 
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.9)", 
                    bordercolor="gray", 
                    borderwidth=1,
                    borderpad=4
                )
                
                show_legend = st.checkbox("Show Legend", value=True, key="legend_toggle")
                if not show_legend:
                    fig_amenity.update_layout(showlegend=False)
                
                st.plotly_chart(fig_amenity, use_container_width=True, key="amenity_highway_map")
            else:
                fig_amenity = go.Figure()
                fig_amenity.add_trace(go.Scattermapbox(
                    lat=[lat], 
                    lon=[lon], 
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='circle',
                        opacity=1.0,
                        sizemode='diameter'
                    ),
                    name="Subject Location",
                    hovertemplate="<b>Subject Location</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
                ))
                
                category_colors = {
                    'A': '#28a745',
                    'B': '#007bff',
                    'C': '#fd7e14',
                    'D': '#dc3545'
                }
                for highway in all_highways:
                    if highway['geometry']:
                        lons, lats = zip(*highway['geometry'])
                        fig_amenity.add_trace(go.Scattermapbox(
                            lon=lons,
                            lat=lats,
                            mode='lines',
                            line=dict(
                                width=4,
                                color=category_colors.get(highway['category'], '#6c757d')
                            ),
                            name=f"{highway['name']} ({highway['category']})",
                            hovertemplate=f"<b>Road: {highway['name']}</b><br>Category: {highway['category_label']}<br>Distance: {highway['distance_m']:.0f}m<extra></extra>",
                            below=''
                        ))
                
                fig_amenity.update_layout(
                    mapbox_style="open-street-map",
                    mapbox=dict(
                        center=dict(lat=lat, lon=lon),
                        zoom=14
                    ),
                    margin=dict(l=20, r=20, t=60, b=80),
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(
                        title="Highways",
                        yanchor="top",
                        y=1.0,
                        xanchor="right",
                        x=0.98,
                        orientation="v",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=11),
                        itemsizing='constant'
                    ),
                    title=dict(
                        text="Subject Location and Nearby Highways",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=18, color="black")
                    )
                )
                
                fig_amenity.add_annotation(
                    text="Note: No amenities found within 1km. Highways within 200m shown by type (A: green, B: blue, C: orange, D: red). Subject location marked with a red circle.",
                    xref="paper", 
                    yref="paper", 
                    x=0.01, 
                    y=0.01,
                    showarrow=False, 
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.9)", 
                    bordercolor="gray", 
                    borderwidth=1,
                    borderpad=4
                )
                
                show_legend = st.checkbox("Show Legend", value=True, key="legend_toggle_no_amenities")
                if not show_legend:
                    fig_amenity.update_layout(showlegend=False)
                
                st.plotly_chart(fig_amenity, use_container_width=True, key="highway_map_no_amenities")
                st.info("No amenities found within 1km, showing subject location and highways.")
            
            st.markdown("### Regression Analysis")
            st.caption("Regression models help us understand how property prices change based on factors such as nearby amenities and road quality in a given area or cluster.")
            selected_cluster_latlong = cluster_info.get('Cluster_LatLong')
            selected_cluster_category = cluster_info.get('Cluster_LatLongCategory')
            
            if pd.notna(selected_cluster_latlong) or pd.notna(selected_cluster_category):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("LatLong Cluster Regression")
                    if pd.notna(selected_cluster_latlong):
                        show_regression_visuals(regression_data, selected_cluster_latlong, 'LatLong')
                    else:
                        st.info("No LatLong cluster data available.")
                
                with col2:
                    st.subheader("LatLongCategory Cluster Regression")
                    if pd.notna(selected_cluster_category):
                        show_regression_visuals(regression_data, selected_cluster_category, 'LatLongCategory')
                    else:
                        st.info("No LatLongCategory cluster data available.")
            
            st.markdown("### Valuation Prediction")
            st.caption("Estimated property rate per square foot based on regression models using amenity score and the widest highway within 200m.")
            highway_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            road = highway_map.get(nearest_biggest_highway['category'], 2)
            predicted_amenity = float(category_df['total_score'].iloc[0]) if not category_df.empty else 0.0
            
            latlong_pred = 'N/A'
            latlong_eq = 'N/A'
            sheet_latlong = 'LatLong_Both_vs_Rate'
            if pd.notna(selected_cluster_latlong) and sheet_latlong in regression_data and not regression_data[sheet_latlong].empty:
                row = regression_data[sheet_latlong][regression_data[sheet_latlong]['Cluster'] == selected_cluster_latlong]
                if not row.empty:
                    row = row.iloc[0]
                    latlong_pred = row['Slope_Amenity'] * predicted_amenity + row['Slope_RoadCat'] * road + row['Intercept']
                    latlong_eq = row['Equation']
            
            category_pred = 'N/A'
            category_eq = 'N/A'
            sheet_category = 'LatLongCategory_Amenity_vs_Rate'
            if pd.notna(selected_cluster_category) and sheet_category in regression_data and not regression_data[sheet_category].empty:
                row = regression_data[sheet_category][regression_data[sheet_category]['Cluster'] == selected_cluster_category]
                if not row.empty:
                    row = row.iloc[0]
                    category_pred = row['Slope_Amenity'] * predicted_amenity + row['Intercept']
                    category_eq = row['Equation']
            
            st.subheader("Prediction Results")
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.markdown("**LatLong Cluster Prediction**")
                if latlong_pred != 'N/A':
                    cluster_avg = project_df[
                        project_df['Cluster_LatLong'] == selected_cluster_latlong
                    ]['Mid_Rate'].mean()
                    st.info(
                        f"**Predicted Rate (Salable Area):** ₹{latlong_pred:,.0f} / sqft"
                    )
                else:
                    st.info("No data available")
                st.caption(
                    f"Model: {latlong_eq}\n"
                    "(uses amenity score + road category: A=1, B=2, C=3, D=4)"
                )

            with col_pred2:
                st.markdown("**LatLongCategory Cluster Prediction**")
                if category_pred != 'N/A':
                    cluster_avg = project_df[
                        project_df['Cluster_LatLongCategory'] == selected_cluster_category
                    ]['Mid_Rate'].mean()
                    st.info(
                        f"**Predicted Rate (Salable Area):** ₹{category_pred:,.0f} / sqft"
                    )
                else:
                    st.info("No data available")
                st.caption(f"Model: {category_eq}\n(uses amenity score only)")
                
            # ----- NEW COLLAPSIBLE CLUSTER SUMMARY -----
            with st.expander("Show Cluster Summary", expanded=False):
                st.caption("Statistical distribution of **Mid_Rate** for the two nearest clusters.")
                
                # LatLong cluster
                if pd.notna(selected_cluster_latlong):
                    latlong_stats = cluster_summary(project_df, 'Cluster_LatLong', selected_cluster_latlong)
                    if latlong_stats:
                        st.markdown("**LatLong Cluster**")
                        latlong_df = pd.DataFrame.from_dict(
                            latlong_stats, orient='index', columns=['Rate (₹/sqft)']
                        ).round(0).astype(int)
                        st.dataframe(
                            latlong_df.style.format("{:,}"),
                            use_container_width=True
                        )
                    else:
                        st.info("No rate data for the LatLong cluster.")
                else:
                    st.info("LatLong cluster not available.")
                
                # LatLongCategory cluster
                if pd.notna(selected_cluster_category):
                    cat_stats = cluster_summary(project_df, 'Cluster_LatLongCategory', selected_cluster_category)
                    if cat_stats:
                        st.markdown("**LatLongCategory Cluster**")
                        cat_df = pd.DataFrame.from_dict(
                            cat_stats, orient='index', columns=['Rate (₹/sqft)']
                        ).round(0).astype(int)
                        st.dataframe(
                            cat_df.style.format("{:,}"),
                            use_container_width=True
                        )
                    else:
                        st.info("No rate data for the LatLongCategory cluster.")
                else:
                    st.info("LatLongCategory cluster not available.")

if __name__ == "__main__":
    main()