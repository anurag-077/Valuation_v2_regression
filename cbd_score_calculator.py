# cbd_score_ors.py
import requests
import numpy as np
import streamlit as st
from typing import Dict, Optional

# ==============================================================
# 1. CONFIG — GET YOUR FREE API KEY FROM:
# https://openrouteservice.org/dev/#/signup
# ==============================================================
ORS_API_KEY = "your-api-key&start=8.681495,49.41461&end=8.687872,49.420318"  # ← REPLACE THIS

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
# 3. ORS DIRECTIONS API CALL
# ==============================================================
def get_ors_route(start_lng, start_lat, end_lng, end_lat) -> Optional[Dict]:
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {"Authorization": ORS_API_KEY}
    body = {
        "coordinates": [[start_lng, start_lat], [end_lng, end_lat]],
        "instructions": False,
        "preference": "fastest"
    }
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"ORS Error {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

# ==============================================================
# 4. EXTRACT ROAD TYPE FROM ORS SEGMENTS
# ==============================================================
ORS_ROAD_CLASS_TO_TYPE = {
    "motorway": "A", "motorway_link": "A",
    "trunk": "A", "trunk_link": "A",
    "primary": "B", "primary_link": "B",
    "secondary": "C", "secondary_link": "C",
    "tertiary": "C", "tertiary_link": "C",
    "unclassified": "D", "residential": "D", "service": "D", "living_street": "D"
}

def extract_dominant_road_type(segments):
    if not segments:
        return "D"
    road_classes = []
    total_dist = sum(s.get("distance", 0) for s in segments)
    for seg in segments:
        dist = seg.get("distance", 0)
        way = seg.get("way", {})
        road_class = way.get("road_class", "unclassified")
        weight = dist / total_dist if total_dist > 0 else 1
        road_classes.append((road_class, weight))
    # Weighted vote
    scores = {"A": 0, "B": 0, "C": 0, "D": 0}
    for rc, w in road_classes:
        mapped = ORS_ROAD_CLASS_TO_TYPE.get(rc, "D")
        scores[mapped] += w
    return max(scores, key=scores.get)

# ==============================================================
# 5. MAIN CBD SCORE FUNCTION (AUTO ROAD DISTANCE + TYPE)
# ==============================================================
def calculate_cbd_score_ors(
    project_lat: float,
    project_lng: float
) -> Dict:
    results = []
    
    with st.spinner("Fetching real road routes to all CBDs..."):
        for cbd in CBD_MASTER:
            route = get_ors_route(
                project_lng, project_lat,
                cbd["lng"], cbd["lat"]
            )
            if not route or "routes" not in route or not route["routes"]:
                # Fallback to Haversine if ORS fails
                from math import radians, sin, cos, sqrt, atan2
                R = 6371
                dlat = radians(cbd["lat"] - project_lat)
                dlon = radians(cbd["lng"] - project_lng)
                a = sin(dlat/2)**2 + cos(radians(project_lat)) * cos(radians(cbd["lat"])) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                dist_km = R * c
                time_min = (dist_km / 30) * 60  # assume 30 km/h
                road_type = "C"
            else:
                r = route["routes"][0]
                summary = r["summary"]
                dist_km = summary["distance"] / 1000
                time_min = summary["duration"] / 60
                road_type = extract_dominant_road_type(r.get("segments", []))

            # Score calculations
            score_dist = max(0.6, 1 / (1 + dist_km / 10))
            score_time = max(0.6, 1 / (1 + time_min / 30))
            final_score = round(0.6 * score_dist + 0.4 * score_time, 3)
            final_score = max(0.6, min(1.0, final_score))

            results.append({
                "CBD": cbd["name"],
                "Area": cbd["area"],
                "Road_Distance_km": round(dist_km, 2),
                "Travel_Time_min": round(time_min, 1),
                "Road_Type": road_type,
                "Score_Dist": round(score_dist, 3),
                "Score_Time": round(score_time, 3),
                "CBD_Score": final_score
            })

    # Find best CBD
    best = max(results, key=lambda x: x["CBD_Score"])
    
    return {
        "CBD_Score": best["CBD_Score"],
        "Nearest_CBD": best["CBD"],
        "CBD_Area": best["Area"],
        "Road_Distance_km": best["Road_Distance_km"],
        "Travel_Time_min": best["Travel_Time_min"],
        "Road_Type": best["Road_Type"],
        "All_CBDs": results,
        "Input_Lat": project_lat,
        "Input_Lng": project_lng
    }

# ==============================================================
# STREAMLIT DEMO
# ==============================================================
def run_demo():
    st.title("CBD Score — Real Road Distance (Auto)")
    st.caption("Uses OpenRouteService API | No manual road type | Actual drivable path")

    if ORS_API_KEY == "YOUR_ORS_API_KEY_HERE":
        st.error("Please set your OpenRouteService API key in the code!")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=18.5204, format="%.6f")
    with col2:
        lng = st.number_input("Longitude", value=73.8566, format="%.6f")

    if st.button("Calculate CBD Score (Road Network)", type="primary"):
        result = calculate_cbd_score_ors(lat, lng)
        
        st.success(f"**CBD Score = {result['CBD_Score']:.3f}**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Nearest CBD", f"{result['Nearest_CBD']} ({result['CBD_Area']})")
            st.metric("Road Distance", f"{result['Road_Distance_km']} km")
        with c2:
            st.metric("Travel Time", f"{result['Travel_Time_min']} min")
            st.metric("Road Type", result['Road_Type'])

        with st.expander("All CBDs (Detailed)", expanded=False):
            df = pd.DataFrame(result["All_CBDs"])
            st.dataframe(df, use_container_width=True)

        st.caption("Based on **actual road network** • 10 km or 30 min = decay midpoint")

# ==============================================================
# RUN
# ==============================================================
if __name__ == "__main__":
    run_demo()