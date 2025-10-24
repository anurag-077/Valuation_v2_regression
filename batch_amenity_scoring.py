import streamlit as st
import pandas as pd
import os
from typing import List, Dict, Any
import numpy as np
import plotly.express as px
from tqdm import tqdm

# Static weights (EXACT from your table)
STATIC_WEIGHTS = {
    'Metro': 0.25,
    'Bus': 0.15,
    'Mall': 0.225,
    'School': 0.225,
    'Hospital': 0.075,
    'Garden': 0.075
}

# AMENITY_TYPES mapping
AMENITY_TYPES = {
    'bus_stop': ('Bus', 0.80),
    'bus_station': ('Bus', 0.80),
    'railway=station': ('Bus', 0.80),
    'subway_entrance': ('Metro', 0.80),
    'tram_stop': ('Bus', 0.80),
    'public_transport=stop_position': ('Bus', 0.80),
    'public_transport=platform': ('Bus', 0.80),
    'public_transport=station': ('Bus', 0.80),
    'metro_station': ('Metro', 0.80),
    'school': ('School', 1.00),
    'schools': ('School', 1.00),
    'college': ('School', 1.00),
    'university': ('School', 1.00),
    'hospital': ('Hospital', 1.00),
    'hospitals': ('Hospital', 1.00),
    'clinic': ('Hospital', 1.00),
    'doctors': ('Hospital', 1.00),
    'pharmacy': ('Hospital', 1.00),
    'park': ('Garden', 0.50),
    'gardens': ('Garden', 0.50),
    'playground': ('Garden', 0.50),
    'sports_centre': ('Garden', 0.50),
    'pitch': ('Garden', 0.50),
    'supermarket': ('Mall', 0.60),
    'convenience': ('Mall', 0.60),
    'department_store': ('Mall', 0.60),
    'mall': ('Mall', 0.60),
    'malls': ('Mall', 0.60),
    'marketplace': ('Mall', 0.60)
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
        st.error(f"❌ **Folder not found**: `{amenity_dir}`")
        return create_sample_data()
    
    found_files = [f for f in os.listdir(amenity_dir) if f.lower().endswith('.xlsx')]
    if not found_files:
        st.warning("⚠️ **No Excel files found!** Using sample data.")
        return create_sample_data()
    
    data = []
    total_records = 0
    
    for file in found_files:
        type_name = file[:-5].lower()
        if type_name not in AMENITY_TYPES:
            continue
        
        group_name, _ = AMENITY_TYPES[type_name]
        file_path = os.path.join(amenity_dir, file)
        
        try:
            df = pd.read_excel(file_path)
            required_cols = ['lat', 'lng']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
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
                df['name'] = f"{type_name.capitalize()}-{range(1, len(df)+1)}"
            
            df['category'] = group_name
            df['type_name'] = type_name
            data.append(df[['lat', 'lng', 'category', 'type_name', 'name']])
            total_records += len(df)
            
        except Exception as e:
            continue
    
    if data:
        final_df = pd.concat(data, ignore_index=True)
        st.success(f"✅ **TOTAL: {total_records} amenities** loaded")
        return final_df
    else:
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

def calculate_single_project_score(lat: float, lon: float, all_amenities: pd.DataFrame) -> Dict[str, float]:
    """Calculate amenity score for ONE project"""
    if all_amenities.empty:
        return {'total_score': 0.0}
    
    lats = all_amenities['lat'].values
    lons = all_amenities['lng'].values
    dists = haversine_vectorized(lat, lon, lats, lons)
    
    mask = dists <= POI_SEARCH_RADIUS_M
    if not np.any(mask):
        return {'total_score': 0.0}
    
    filtered_df = all_amenities[mask].copy()
    filtered_df['distance_m'] = dists[mask]
    filtered_df['f_d'] = 1 / (1 + filtered_df['distance_m'] / 200)
    
    category_scores = filtered_df.groupby('category')['f_d'].sum().reset_index()
    category_scores.columns = ['category', 'S_c']
    category_scores['s_c'] = 1 - np.exp(-0.8 * category_scores['S_c'])
    category_scores['weight'] = category_scores['category'].map(STATIC_WEIGHTS).fillna(0)
    category_scores['Weight × s_c'] = category_scores['weight'] * category_scores['s_c']
    
    total_score = category_scores['Weight × s_c'].sum()
    return {'total_score': total_score}

# 🚀 MAIN BATCH PROCESSING FUNCTION
def process_all_projects():
    st.title("🏠 **BATCH AMENITY SCORING**")
    st.markdown("**Process ALL projects from `All_Project_data.xlsx`**")
    
    # Load amenities
    amenity_dir = "amenities"
    with st.spinner("Loading amenities..."):
        all_amenities = load_amenities(amenity_dir)
    
    # Load projects
    input_file = "All_Project_data.xlsx"
    if not os.path.exists(input_file):
        st.error(f"❌ **File not found**: `{input_file}`")
        st.info("💡 **Place `All_Project_data.xlsx` in same folder**")
        st.stop()
    
    with st.spinner("Loading projects..."):
        projects_df = pd.read_excel(input_file)
    
    required_cols = ['project_lat', 'project_lng']
    missing_cols = [col for col in required_cols if col not in projects_df.columns]
    if missing_cols:
        st.error(f"❌ **Missing columns**: {missing_cols}")
        st.info("💡 **Need**: `project_lat`, `project_lng`")
        st.stop()
    
    # Clean lat/lng
    projects_df = projects_df.dropna(subset=required_cols)
    projects_df['project_lat'] = pd.to_numeric(projects_df['project_lat'], errors='coerce')
    projects_df['project_lng'] = pd.to_numeric(projects_df['project_lng'], errors='coerce')
    projects_df = projects_df.dropna(subset=required_cols)
    
    st.success(f"✅ **Loaded {len(projects_df)} projects**")
    st.write("**Sample:**")
    st.dataframe(projects_df[['project_lat', 'project_lng']].head())
    
    # BATCH PROCESS
    if st.button("🚀 **CALCULATE SCORES FOR ALL PROJECTS**", type="primary"):
        with st.spinner(f"Processing {len(projects_df)} projects..."):
            total_scores = []
            
            progress_bar = st.progress(0)
            for idx, row in tqdm(projects_df.iterrows(), total=len(projects_df)):
                lat = float(row['project_lat'])
                lon = float(row['project_lng'])
                score_data = calculate_single_project_score(lat, lon, all_amenities)
                total_scores.append(score_data['total_score'])
                progress_bar.progress((idx + 1) / len(projects_df))
            
            # Add scores to dataframe
            projects_df['amenity_score'] = total_scores
            output_file = "All_Project_data_WITH_Amenity_Scores.xlsx"
            projects_df.to_excel(output_file, index=False)
            
            st.success(f"✅ **COMPLETED!** Scores saved to: `{output_file}`")
            
            # Show results
            st.markdown("---")
            st.subheader("📊 **SUMMARY**")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("**Projects**", len(projects_df))
            with col2: st.metric("**Avg Score**", f"{projects_df['amenity_score'].mean():.3f}")
            with col3: st.metric("**Max Score**", f"{projects_df['amenity_score'].max():.3f}")
            with col4: st.metric("**Min Score**", f"{projects_df['amenity_score'].min():.3f}")
            
            # Distribution chart
            fig = px.histogram(projects_df, x='amenity_score', 
                             title="Amenity Score Distribution",
                             nbins=20, color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 10 projects
            st.markdown("---")
            st.subheader("🏆 **TOP 10 PROJECTS**")
            top_projects = projects_df.nlargest(10, 'amenity_score')[['project_lat', 'project_lng', 'amenity_score']]
            st.dataframe(top_projects.round(3), use_container_width=True)
            
            # Download button
            with open(output_file, 'rb') as f:
                st.download_button(
                    label="📥 **DOWNLOAD UPDATED FILE**",
                    data=f.read(),
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

def main():
    process_all_projects()

if __name__ == "__main__":
    main()