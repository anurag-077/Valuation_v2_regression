# cbd_score_nearest_routes.py
import requests
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple

# ==============================================================
# 1. OSRM PUBLIC SERVER
# ==============================================================
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

# ==============================================================
# 2. MASTER CBD LIST
# ==============================================================
CBD_MASTER = [
    {"name": "Shivajinagar", "lat": 18.5305, "lng": 73.8472, "area": "Pune"},
    {"name": "Camp", "lat": 18.5127, "lng": 73.8795, "area": "Pune"},
    {"name": "Koregaon Park", "lat": 18.5377, "lng": 73.8855, "area": "Pune"},
    {"name": "Pimpri", "lat": 18.6275, "lng": 73.8060, "area": "PCMC"},
    {"name": "Akurdi", "lat": 18.6427, "lng": 73.7585, "area": "PCMC"},
    {"name": "Chinchwad", "lat": 18.6297, "lng": 73.7850, "area": "PCMC"},
]

# ==============================================================
# 3. ROAD TYPE MAPPING
# ==============================================================
OSM_TO_CATEGORY = {
    "motorway": "D", "motorway_link": "D",
    "trunk": "D", "trunk_link": "C",
    "primary": "C", "primary_link": "C",
    "secondary": "B", "secondary_link": "B",
    "tertiary": "A",
    "residential": "A", "unclassified": "A", "service": "A",
    "living_street": "A", "road": "A"
}

# ==============================================================
# 4. GET MULTIPLE ROUTES TO ONE CBD
# ==============================================================
def get_routes_to_cbd(start_lng, start_lat, end_lng, end_lat) -> List[Dict]:
    url = f"{OSRM_URL}{start_lng},{start_lat};{end_lng},{end_lat}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "alternatives": "true",
        "steps": "true"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("routes", [])
        else:
            st.warning(f"OSRM Error: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Request failed: {e}")
        return []

# ==============================================================
# 5. EXTRACT ROAD TYPE
# ==============================================================
def get_dominant_road_type(legs: List) -> str:
    if not legs or not legs[0].get("steps"):
        return "A"
    total_dist = 0
    weighted = {"A": 0, "B": 0, "C": 0, "D": 0}
    for step in legs[0]["steps"]:
        dist = step.get("distance", 0)
        total_dist += dist
        name = step.get("name", "").lower()
        if "nh" in name or "express" in name:
            cat = "D"
        elif "road" in name or "marg" in name:
            cat = "C"
        elif "lane" in name or "galli" in name:
            cat = "A"
        else:
            cat = "B"
        weighted[cat] += dist
    return max(weighted, key=weighted.get) if total_dist > 0 else "A"

# ==============================================================
# 6. EXTRACT COORDS
# ==============================================================
def extract_coords(geometry: Dict) -> List[Tuple[float, float]]:
    if not geometry or geometry.get("type") != "LineString":
        return []
    return [(lat, lng) for lng, lat in geometry.get("coordinates", [])]

# ==============================================================
# 7. SCORE FUNCTION
# ==============================================================
def calculate_score(dist_km: float, time_min: float) -> float:
    score_dist = max(0.6, 1 / (1 + dist_km / 10))
    score_time = max(0.6, 1 / (1 + time_min / 30))
    return round(0.6 * score_dist + 0.4 * score_time, 3)

# ==============================================================
# 8. MAIN: FIND NEAREST CBD + ALL ROUTES TO IT
# ==============================================================
@st.cache_data(show_spinner=False)
def analyze_nearest_cbd(project_lat: float, project_lng: float) -> Dict:
    best_cbd = None
    best_score = 0
    best_route_data = None

    for cbd in CBD_MASTER:
        routes = get_routes_to_cbd(project_lng, project_lat, cbd["lng"], cbd["lat"])
        if not routes:
            continue
        fastest = routes[0]
        dist_km = fastest["distance"] / 1000
        time_min = fastest["duration"] / 60
        score = calculate_score(dist_km, time_min)
        if score > best_score:
            best_score = score
            best_cbd = cbd
            best_route_data = fastest

    if not best_cbd:
        st.error("No route found to any CBD.")
        return {}

    # Get ALL routes to nearest CBD
    all_routes = get_routes_to_cbd(project_lng, project_lat, best_cbd["lng"], best_cbd["lat"])
    route_details = []
    selected_geometry = None

    for i, r in enumerate(all_routes):
        dist_km = r["distance"] / 1000
        time_min = r["duration"] / 60
        road_type = get_dominant_road_type(r["legs"])
        score = calculate_score(dist_km, time_min)
        is_selected = (i == 0)
        if is_selected:
            selected_geometry = r["geometry"]
        route_details.append({
            "Route": f"Route {i+1}",
            "Distance_km": round(dist_km, 2),
            "Time_min": round(time_min, 1),
            "Road_Type": road_type,
            "Score": score,
            "Is_Selected": is_selected,
            "Geometry": r["geometry"]
        })

    return {
        "Nearest_CBD": best_cbd["name"],
        "CBD_Area": best_cbd["area"],
        "CBD_Score": best_score,
        "Selected_Distance_km": round(best_route_data["distance"] / 1000, 2),
        "Selected_Time_min": round(best_route_data["duration"] / 60, 1),
        "Selected_Road_Type": get_dominant_road_type(best_route_data["legs"]),
        "All_Routes": route_details,
        "Selected_Geometry": selected_geometry,
        "Project_Lat": project_lat,
        "Project_Lng": project_lng,
        "CBD_Lat": best_cbd["lat"],
        "CBD_Lng": best_cbd["lng"]
    }

# ==============================================================
# 9. PLOT MAP
# ==============================================================
def plot_routes_to_cbd(data: Dict):
    fig = go.Figure()

    # Project
    fig.add_trace(go.Scattermapbox(
        lat=[data["Project_Lat"]], lon=[data["Project_Lng"]],
        mode="markers", marker=dict(size=16, color="blue"),
        name="Project", text="Your Project", hoverinfo="text"
    ))

    # Nearest CBD
    fig.add_trace(go.Scattermapbox(
        lat=[data["CBD_Lat"]], lon=[data["CBD_Lng"]],
        mode="markers", marker=dict(size=16, color="orange"),
        name="Nearest CBD", text=f"{data['Nearest_CBD']} ({data['CBD_Area']})", hoverinfo="text"
    ))

    # All routes
    for route in data["All_Routes"]:
        coords = extract_coords(route["Geometry"])
        if not coords:
            continue
        lats, lons = zip(*coords)
        color = "green" if route["Is_Selected"] else "red"
        width = 6 if route["Is_Selected"] else 2
        opacity = 1.0 if route["Is_Selected"] else 0.5
        name = f"{route['Distance_km']} km, {route['Time_min']} min"
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="lines", line=dict(width=width, color=color),
            opacity=opacity, name=name,
            hoverinfo="text",
            text=f"{name} | Type: {route['Road_Type']} | Score: {route['Score']}"
        ))

    center_lat = (data["Project_Lat"] + data["CBD_Lat"]) / 2
    center_lng = (data["Project_Lng"] + data["CBD_Lng"]) / 2
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=center_lat, lon=center_lng), zoom=12),
        margin=dict(l=0, r=0, t=50, b=0),
        height=650,
        title=f"All Routes to {data['Nearest_CBD']} | Green = Used for CBD Score"
    )
    return fig

# ==============================================================
# 10. STREAMLIT UI – SINGLE INPUT BOX
# ==============================================================
st.set_page_config(page_title="CBD Score - Nearest CBD Routes", layout="wide")
st.title("CBD Score – All Routes to Nearest CBD")
st.caption("Paste lat,lon (e.g., 18.592624745040947, 73.80011752521669)")

# SINGLE INPUT BOX
coord_input = st.text_input(
    "Enter Latitude, Longitude",
    placeholder="18.592624745040947, 73.80011752521669",
    help="Paste coordinates in format: latitude, longitude"
)

if st.button("Analyze Nearest CBD", type="primary", use_container_width=True):
    if not coord_input.strip():
        st.error("Please enter coordinates.")
        st.stop()

    # Parse input
    try:
        parts = [p.strip() for p in coord_input.split(",")]
        if len(parts) != 2:
            raise ValueError
        lat = float(parts[0])
        lng = float(parts[1])
        if not (10 <= lat <= 35 and 60 <= lng <= 95):  # India bounds
            raise ValueError
    except:
        st.error("Invalid format. Use: latitude, longitude (e.g., 18.52, 73.85)")
        st.stop()

    with st.spinner("Finding nearest CBD and all routes..."):
        result = analyze_nearest_cbd(lat, lng)

    if not result:
        st.stop()

    # Final Score
    st.success(f"**CBD SCORE = {result['CBD_Score']:.3f}**")
    st.markdown(f"**Nearest CBD:** {result['Nearest_CBD']} ({result['CBD_Area']})")
    st.markdown(f"**Selected Route:** {result['Selected_Distance_km']} km, {result['Selected_Time_min']} min, Type **{result['Selected_Road_Type']}**")

    # Map
    st.subheader("All Routes to Nearest CBD")
    fig = plot_routes_to_cbd(result)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("Route Options to Nearest CBD")
    df = pd.DataFrame(result["All_Routes"])
    df = df.drop(columns=["Geometry", "Is_Selected"])
    df["Selected"] = df["Score"].apply(lambda x: "Yes" if x == result['CBD_Score'] else "")
    df = df.sort_values("Score", ascending=False)
    st.dataframe(
        df.style.apply(lambda row: ['background: #d4edda' if row['Selected'] == 'Yes' else ''] * len(row), axis=1),
        use_container_width=True
    )

    st.caption(
        "Green route = used for CBD Score | "
        "Only routes to nearest CBD shown | "
        "Score: 10 km or 30 min = midpoint"
    )