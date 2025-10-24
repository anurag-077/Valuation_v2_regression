import streamlit as st
import pandas as pd
import os
from typing import List, Dict, Any
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Default static weights
DEFAULT_WEIGHTS = {
    'Metro': 0.25,
    'Bus': 0.15,
    'Mall': 0.225,
    'School': 0.225,
    'Hospital': 0.075,
    'Garden': 0.075
}

# SIMPLIFIED - Only categories needed
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

@st.cache_data
def load_amenities(amenity_dir: str):
    if not os.path.exists(amenity_dir):
        return create_sample_data()
    
    found_files = [f for f in os.listdir(amenity_dir) if f.lower().endswith('.xlsx')]
    if not found_files:
        return create_sample_data()
    
    data = []
    for file in found_files:
        type_name = file[:-5].lower()
        if type_name not in AMENITY_TYPES:
            continue
        
        group_name = AMENITY_TYPES[type_name]  # ✅ SIMPLIFIED - Only category
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
            
        except:
            continue
    
    if data:
        return pd.concat(data, ignore_index=True)
    return create_sample_data()

def create_sample_data():
    sample_data = [
        {'lat': 18.5530, 'lng': 73.7589, 'name': 'Metro-1', 'type_name': 'metro_station', 'category': 'Metro'},
        {'lat': 18.5536, 'lng': 73.7595, 'name': 'Metro-2', 'type_name': 'metro_station', 'category': 'Metro'},
        {'lat': 18.5531, 'lng': 73.7590, 'name': 'Bus-1', 'type_name': 'bus_station', 'category': 'Bus'},
        {'lat': 18.5533, 'lng': 73.7592, 'name': 'Bus-2', 'type_name': 'bus_station', 'category': 'Bus'},
        {'lat': 18.5535, 'lng': 73.7594, 'name': 'Bus-3', 'type_name': 'bus_station', 'category': 'Bus'},
        {'lat': 18.5546, 'lng': 73.7600, 'name': 'Bus-4', 'type_name': 'bus_station', 'category': 'Bus'},
        {'lat': 18.5532, 'lng': 73.7591, 'name': 'Mall-1', 'type_name': 'malls', 'category': 'Mall'},
        {'lat': 18.5534, 'lng': 73.7593, 'name': 'Mall-2', 'type_name': 'malls', 'category': 'Mall'},
        {'lat': 18.5529, 'lng': 73.7588, 'name': 'School-1', 'type_name': 'schools', 'category': 'School'},
        {'lat': 18.5537, 'lng': 73.7596, 'name': 'School-2', 'type_name': 'schools', 'category': 'School'},
        {'lat': 18.5539, 'lng': 73.7598, 'name': 'School-3', 'type_name': 'schools', 'category': 'School'},
        {'lat': 18.5528, 'lng': 73.7587, 'name': 'Hospital-1', 'type_name': 'hospitals', 'category': 'Hospital'},
        {'lat': 18.5538, 'lng': 73.7597, 'name': 'Hospital-2', 'type_name': 'hospitals', 'category': 'Hospital'},
        {'lat': 18.5540, 'lng': 73.7601, 'name': 'Garden-1', 'type_name': 'gardens', 'category': 'Garden'},
        {'lat': 18.5523, 'lng': 73.7583, 'name': 'Garden-2', 'type_name': 'gardens', 'category': 'Garden'}
    ]
    return pd.DataFrame(sample_data)

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

def parse_lat_long(input_text: str) -> tuple:
    try:
        parts = input_text.strip().split(',')
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return lat, lon
    except:
        return None, None

# MAIN APP - REST OF CODE SAME (unchanged)
def main():
    st.set_page_config(page_title="Amenity Score Calculator", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
        .main { background-color: #f8f9fc; padding: 20px; border-radius: 10px; }
        h1, h2, h3 { color: #2c3e50; font-family: 'Arial', sans-serif; }
        .stButton > button { background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 10px 20px; font-size: 16px; }
        .stButton > button:hover { background-color: #45a049; }
        .stNumberInput input { border-radius: 5px; border: 1px solid #ddd; padding: 8px; }
        .stTextInput input { border-radius: 5px; border: 1px solid #ddd; padding: 8px; }
        .stMetric { background-color: white; border-radius: 5px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stAlert { border-radius: 5px; }
        .sidebar .sidebar-content { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='margin: 0;'>🏠 Amenity Score Calculator</h1>
        <p style='color: #555; margin: 5px 0 0;'>Customize weights • 1km radius analysis • Interactive visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    amenity_dir = "amenities"
    if 'all_amenities' not in st.session_state:
        with st.spinner("🔄 Loading amenities data..."):
            st.session_state.all_amenities = load_amenities(amenity_dir)
    
    st.markdown("<h3 style='text-align: center; color: #34495e;'>📍 Enter Location Coordinates</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("**Format Example**: 18.588505544243457, 73.81598473693089")
        coord_input = st.text_input("", value="18.5530, 73.7589", label_visibility="collapsed")
    
    lat, lon = parse_lat_long(coord_input)
    if lat is None or lon is None:
        st.error("❌ Invalid format! Please use 'lat, long'.")
        st.stop()
    
    st.markdown(f"<p style='text-align: center; color: #27ae60; font-weight: bold;'>✅ Parsed: Latitude = {lat:.6f}, Longitude = {lon:.6f}</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<h3 style='color: #2980b9;'>⚖️ Custom Weights</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #777; font-size: 14px;'>Adjust priorities (total should sum to 1.0)</p>", unsafe_allow_html=True)
        
        categories = list(DEFAULT_WEIGHTS.keys())
        custom_weights = {}
        total_weight = 0
        
        for cat in categories:
            weight = st.number_input(
                cat,
                min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHTS[cat],
                step=0.01, format="%.2f",
                key=f"wt_{cat}"
            )
            custom_weights[cat] = weight
            total_weight += weight
        
        st.markdown("---")
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.metric("Total Weight", f"{total_weight:.2f}")
        with col_sum2:
            if abs(total_weight - 1.0) < 0.01:
                st.metric("Status", "✅ Good")
            else:
                st.metric("Status", "⚠️ Adjust", delta_color="inverse")
        
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            for cat in categories:
                st.session_state[f"wt_{cat}"] = DEFAULT_WEIGHTS[cat]
            st.rerun()
        
        st.markdown("---")
        st.markdown("<h4 style='color: #2980b9;'>📘 Formula Quick View</h4>", unsafe_allow_html=True)
        st.latex(r"f(d) = \frac{1}{1 + \frac{d}{200}}")
        st.latex(r"S_c = \sum f(d)")
        st.latex(r"s_c = 1 - e^{-0.8 \cdot S_c}")

    st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
    if st.button("🚀 Calculate Scores", type="primary", use_container_width=False):
        current_weights = {cat: st.session_state[f"wt_{cat}"] for cat in categories}
        
        with st.spinner("⚡ Processing your request..."):
            category_df = calculate_amenity_scores(lat, lon, st.session_state.all_amenities, current_weights)
            detailed_df = get_detailed_amenities(lat, lon, st.session_state.all_amenities)
        
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #2c3e50;'>📊 Calculation Results</h2>", unsafe_allow_html=True)
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("🌟 Total Amenity Score", f"{category_df['total_score'].iloc[0]:.3f}", help="Overall weighted score based on your preferences")
        with col_metric2:
            st.metric("🏘️ Amenities Found", len(detailed_df), help="Number of amenities within 1km radius")
        
        st.subheader("Category Breakdown", anchor=False)
        st.dataframe(
            category_df[['category', 'count', 'S_c', 's_c', 'weight', 'Weight × s_c']].round(3),
            use_container_width=True,
            hide_index=True,
            column_config={
                "category": "Category",
                "count": "Count",
                "S_c": "Raw Sum (S_c)",
                "s_c": "Saturation (s_c)",
                "weight": "Your Weight",
                "Weight × s_c": "Contribution"
            }
        )
        
        st.subheader("Contribution Visualization", anchor=False)
        fig = px.bar(
            category_df, x='category', y='Weight × s_c',
            color='Weight × s_c', color_continuous_scale='Viridis',
            labels={'Weight × s_c': 'Contribution'}
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Contribution: %{y:.3f}<br>Count: %{customdata[0]}<br>S_c: %{customdata[1]:.3f}<br>s_c: %{customdata[2]:.3f}<br>Weight: %{customdata[3]:.3f}",
            customdata=category_df[['count', 'S_c', 's_c', 'weight']].values
        )
        fig.update_layout(
            title_text="Amenity Contributions by Category",
            title_x=0.5,
            xaxis_title="Category",
            yaxis_title="Contribution Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View Detailed Amenities List", expanded=True):
            if not detailed_df.empty:
                st.dataframe(
                    detailed_df.rename(columns={
                        'name': 'Name', 'type_name': 'Type', 'category': 'Category',
                        'distance_m': 'Distance (m)', 'f_d': 'f(d)'
                    }).round(2),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Name": st.column_config.TextColumn(width="medium"),
                        "Type": st.column_config.TextColumn(),
                        "Category": st.column_config.TextColumn(),
                        "Distance (m)": st.column_config.NumberColumn(format="%.1f m"),
                        "f(d)": st.column_config.NumberColumn(format="%.3f")
                    }
                )
            else:
                st.info("No amenities found within the 1km radius.")
        
        st.subheader("🗺️ Interactive Amenities Map", anchor=False)
        if not detailed_df.empty:
            detailed_df['hover_text'] = detailed_df.apply(
                lambda row: f"{row['name']}<br>Category: {row['category']}<br>Type: {row['type_name']}<br>Distance: {row['distance_m']:.1f}m<br>f(d): {row['f_d']:.3f}",
                axis=1
            )
            
            fig_map = px.scatter_mapbox(
                detailed_df,
                lat="lat",
                lon="lng",
                hover_name="hover_text",
                color="category",
                size="f_d",
                size_max=15,
                zoom=14,
                height=500,
                center={"lat": lat, "lon": lon},
                title="Nearby Amenities (Sized by Contribution)"
            )
            
            fig_map.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers+text',
                marker=dict(size=20, color='red', symbol='star'),
                text=["Your Location"],
                textposition="top center",
                name="Your Location",
                hovertemplate="<b>Your Location</b><extra></extra>"
            ))
            
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0, "t":40, "l":0, "b":0},
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
                title_text="Interactive Map of Amenities (1km Radius)",
                title_x=0.5,
                hovermode="closest"
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("🔍 **Zoom & Pan** to explore")
            with col2:
                st.info("🎨 **Hover** for full details")
        else:
            st.info("No amenities to display on the map.")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888; font-size: 12px;'>Powered by Streamlit • Data from Local Amenities Folder</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()