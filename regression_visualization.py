import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load regression results - YOUR EXACT 7 SHEETS
@st.cache_data
def load_regression_data():
    excel_file = 'regression_results.xlsx'
    if not os.path.exists(excel_file):
        st.error("❌ regression_results.xlsx not found! Run regression script first.")
        st.stop()
    
    data = {}
    sheet_names = [
        'LatLong_Amenity_vs_Rate',
        'LatLong_RoadCat_vs_Rate', 
        'LatLong_Both_vs_Rate',
        'LatLongRate_Amenity_vs_Rate',
        'LatLongRate_RoadCat_vs_Rate', 
        'LatLongRate_Both_vs_Rate',
        'LatLongCategory_Amenity_vs_Rate'
    ]
    
    for sheet in sheet_names:
        try:
            data[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
            print(f"✅ Loaded: {sheet} ({len(data[sheet])} rows)")
        except Exception as e:
            data[sheet] = pd.DataFrame()
            print(f"❌ Error loading {sheet}: {e}")
    
    return data

# Main App
def main():
    st.set_page_config(page_title="Regression Line Visualizer", layout="wide")
    
    st.markdown("""
    <div style='text-align: center; background-color: #e8f4fd; padding: 20px; border-radius: 10px;'>
        <h1 style='margin: 0;'>📈 Regression Line Visualizer</h1>
        <p style='color: #555;'>✅ YOUR NUMERIC CLUSTERS (0,1,2...) • CORRECT SHEETS!</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔄 Loading YOUR 7 sheets..."):
        regression_data = load_regression_data()
    
    # 🚀 Get ALL clusters from ALL sheets
    all_clusters = set()
    for df in regression_data.values():
        if not df.empty:
            clusters = df['Cluster'].dropna().unique()
            all_clusters.update(clusters)
    all_clusters = sorted(list(all_clusters))
    
    if not all_clusters:
        st.error("❌ No clusters found!")
        st.stop()
    
    # 📍 FIXED: Category → CORRECT SHEET MAPPING
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        categories = ['LatLong', 'LatLongRate', 'LatLongCategory']
        selected_category = st.selectbox(
            "🎯 **Select Category**:", 
            options=categories,
            index=0
        )
    
    with col2:
        # ✅ FIXED: Map Category to CORRECT SHEETS & CLUSTERS
        if selected_category == 'LatLong':
            available_sheets = ['LatLong_Amenity_vs_Rate', 'LatLong_RoadCat_vs_Rate', 'LatLong_Both_vs_Rate']
        elif selected_category == 'LatLongRate':
            available_sheets = ['LatLongRate_Amenity_vs_Rate', 'LatLongRate_RoadCat_vs_Rate', 'LatLongRate_Both_vs_Rate']
        else:  # LatLongCategory
            available_sheets = ['LatLongCategory_Amenity_vs_Rate']
        
        # Get clusters FROM THESE SPECIFIC SHEETS ONLY
        category_clusters = set()
        for sheet in available_sheets:
            if sheet in regression_data and not regression_data[sheet].empty:
                clusters = regression_data[sheet]['Cluster'].dropna().unique()
                category_clusters.update(clusters)
        
        category_clusters = sorted(list(category_clusters))
        
        if category_clusters:
            selected_num = st.selectbox(
                f"**{selected_category} Cluster No** ({len(category_clusters)} found):", 
                options=category_clusters,
                index=0,
                format_func=lambda x: f"Cluster {x}"
            )
        else:
            st.error(f"❌ No clusters in {selected_category} sheets!")
            st.stop()
    
    # ✅ Filter data for selected cluster FROM CORRECT SHEETS ONLY
    cluster_data = {}
    for sheet_name in available_sheets:
        if sheet_name in regression_data and not regression_data[sheet_name].empty:
            df = regression_data[sheet_name]
            if selected_num in df['Cluster'].values:
                cluster_data[sheet_name] = df[df['Cluster'] == selected_num].iloc[0]
    
    if not cluster_data:
        st.warning(f"❌ No data for Cluster {selected_num} in {selected_category}")
        st.stop()
    
    # 📊 DISPLAY GRAPHS
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>📊 {selected_category} - Cluster {selected_num}</h2>", unsafe_allow_html=True)
    
    num_graphs = len(cluster_data)
    cols = st.columns(min(3, num_graphs))
    
    graph_idx = 0
    for sheet_name, row in cluster_data.items():
        if pd.notna(row.get('Equation')) and row['Num_Projects'] >= 2:
            with cols[graph_idx % 3]:
                if 'Amenity' in sheet_name:
                    icon, title = "🏠", "Amenity Score"
                    slope = row['Slope_Amenity']
                    x_label = "Amenity Score"
                    x_range = 10  # 1 to 10
                elif 'RoadCat' in sheet_name:
                    icon, title = "🛣️", "Road Type"
                    slope = row['Slope_RoadCat']
                    x_label = "Road Category"
                    x_range = 4   # 1 to 4
                else:  # Both
                    icon, title = "🔄", "Both Factors"
                    slope = row['Slope_Amenity']
                    x_label = "Amenity Score"
                    x_range = 10
                
                st.markdown(f"### {icon} **{title}**")
                fig = create_regression_plot(
                    row['Equation'], slope, row['Intercept'],
                    x_label, "Mid Rate", row['Num_Projects'], x_range
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"n={row['Num_Projects']}")
            
            graph_idx += 1
    
    # ✅ EQUATIONS TABLE
    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center;'>📋 {selected_category} Equations</h3>", unsafe_allow_html=True)
    
    equations_list = []
    for sheet, row in cluster_data.items():
        if pd.notna(row.get('Equation')) and row['Num_Projects'] >= 2:
            slope_val = row.get('Slope_Amenity', row.get('Slope_RoadCat', 'N/A'))
            equations_list.append({
                'Sheet': sheet.replace('_vs_Rate', ''),
                'Equation': row['Equation'],
                'n': row['Num_Projects'],
                'Slope': slope_val
            })
    
    if equations_list:
        equations_df = pd.DataFrame(equations_list)
        st.dataframe(
            equations_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sheet": st.column_config.TextColumn("SHEET"),
                "Equation": st.column_config.TextColumn(width="medium"),
                "n": st.column_config.NumberColumn("n"),
                "Slope": st.column_config.NumberColumn("Slope")
            }
        )
    
    # 📈 PREDICTION TOOL
    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center;'>🔮 Predict - Cluster {selected_num}</h3>", unsafe_allow_html=True)
    
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        amenity_score = st.number_input("🏠 Amenity (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    with col_pred2:
        if 'RoadCat' in str(cluster_data.keys()):
            road_category = st.number_input("🛣️ Road (1-4)", min_value=1, max_value=4, value=2)
        else:
            st.info("**Only Amenity**")
            road_category = 0
    
    if st.button("🚀 Predict Mid Rate", type="primary"):
        amenity_sheet = [s for s in cluster_data.keys() if 'Amenity' in s]
        if amenity_sheet:
            row = cluster_data[amenity_sheet[0]]
            predicted_rate = row['Slope_Amenity'] * amenity_score + row['Intercept']
            st.success(f"**Predicted: ₹{predicted_rate:.2f} Cr**")
            st.caption(f"Using: {row['Equation']}")

def create_regression_plot(equation, slope, intercept, x_label, y_label, n, x_range):
    x = np.linspace(1, x_range, 100)
    y = slope * x + intercept
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Regression Line',
                            line=dict(color='blue', width=3)))
    
    fig.add_annotation(
        x=0.95, y=0.95, xref="paper", yref="paper",
        text=f"<b>{equation}</b><br>n={n}",
        showarrow=False, font=dict(size=12), bgcolor="white",
        bordercolor="blue", borderwidth=1
    )
    
    fig.update_layout(
        title=f"{x_label} vs {y_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=300,
        showlegend=False,
        plot_bgcolor='white'
    )
    return fig

if __name__ == "__main__":
    main()