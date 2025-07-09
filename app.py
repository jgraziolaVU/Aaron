# app.py - Environmental Impact Explorer
# A beginner-friendly Streamlit app for visualizing environmental impacts

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Dict, Any

# -------------- CONFIGURATION --------------
# Set up the page configuration (this should be the first Streamlit command)
st.set_page_config(
    page_title="Environmental Impact Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------- HELPER FUNCTIONS --------------
def convert_power_to_kwh_per_year(value: float, unit: str) -> float:
    """
    Convert different power units to kWh/year for calculations.
    
    Args:
        value: The numerical value
        unit: The unit of measurement
    
    Returns:
        float: Value converted to kWh/year
    """
    if unit == "kWh/yr":
        return value
    elif unit == "kWh/mo":
        return value * 12  # 12 months per year
    elif unit == "kW":
        return value * 8760  # 8760 hours per year
    elif unit == "MW":
        return value * 1000 * 8760  # Convert MW to kW, then to kWh/year
    else:
        return 0

def convert_water_to_liters_per_year(value: float, unit: str) -> float:
    """
    Convert different water units to liters/year for calculations.
    
    Args:
        value: The numerical value
        unit: The unit of measurement
    
    Returns:
        float: Value converted to liters/year
    """
    if unit == "L/yr":
        return value
    elif unit == "L/mo":
        return value * 12  # 12 months per year
    elif unit == "L/s":
        return value * 31536000  # 365.25 * 24 * 3600 seconds per year
    elif unit == "gpm":  # gallons per minute
        return value * 525600 * 3.78541  # minutes per year * liters per gallon
    elif unit == "gal/mo":  # gallons per month
        return value * 12 * 3.78541  # months per year * liters per gallon
    else:
        return 0

def validate_numeric_input(value: str, field_name: str) -> tuple[bool, float]:
    """
    Validate that a text input contains a valid positive number.
    
    Args:
        value: The input string to validate
        field_name: Name of the field for error messages
    
    Returns:
        tuple: (is_valid, numeric_value)
    """
    if not value.strip():
        return False, 0.0
    
    try:
        numeric_value = float(value)
        if numeric_value < 0:
            st.error(f"{field_name} must be a positive number")
            return False, 0.0
        return True, numeric_value
    except ValueError:
        st.error(f"{field_name} must be a valid number")
        return False, 0.0

# -------------- DATA LOADING --------------
@st.cache_data
def load_data() -> Dict[str, Any]:
    """
    Load the environmental data from the .mat file.
    The @st.cache_data decorator ensures this only runs once and caches the result.
    
    Returns:
        Dict containing the loaded data arrays
    """
    try:
        # Load .mat data file
        metrics = scipy.io.loadmat("CountyLevelMetrics.mat")
        
        # Extract and flatten the arrays (convert from 2D to 1D)
        return {
            "AWAREUSCF": metrics["AWAREUSCF"].flatten(),    # Water scarcity footprint
            "EFkgkWh": metrics["EFkgkWh"].flatten(),        # Carbon footprint
            "EWIF": metrics["EWIF"].flatten(),              # Water footprint
            "CountyFIPS": metrics["CountyFIPS"].flatten(),  # County identification codes
        }
    except FileNotFoundError:
        st.error("Data file 'CountyLevelMetrics.mat' not found. Please ensure it is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# -------------- MAIN APP --------------
def main():
    """Main application function that contains all the UI and logic."""
    
    # Load the data
    data = load_data()
    
    # App title and description
    st.title("🌍 Environmental Impact Explorer")
    st.markdown("*Visualize county-level environmental impacts across the United States*")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # (1) State selection dropdown
        state = st.selectbox(
            "Select a state:",
            options=[
                "USA", "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", 
                "Connecticut", "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana",
                "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
                "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
                "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", 
                "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", 
                "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
                "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
                "West Virginia", "Wisconsin", "Wyoming"
            ],
            help="Choose a specific state or 'USA' for the entire continental United States"
        )
        
        # (2) Metric selection
        metric_option = st.selectbox(
            "Select an environmental metric:",
            options=[
                "carbon footprint", 
                "scope 1 & 2 water footprint", 
                "water scarcity footprint"
            ],
            help="Choose which environmental impact to visualize"
        )
        
        # (3) On-site power input
        st.subheader("Facility Information")
        
        power_col1, power_col2 = st.columns([2, 1])
        with power_col1:
            power_value = st.text_input(
                "On-site power consumption:",
                placeholder="Enter power consumption",
                help="Enter your facility's power consumption"
            )
        with power_col2:
            power_unit = st.selectbox(
                "Power unit:",
                ["kWh/yr", "kWh/mo", "kW", "MW"],
                help="Select the unit for power consumption"
            )
        
        # (4) Water input
        water_col1, water_col2 = st.columns([2, 1])
        with water_col1:
            water_value = st.text_input(
                "On-site water consumption:",
                placeholder="Enter water consumption",
                help="Enter your facility's water consumption"
            )
        with water_col2:
            water_unit = st.selectbox(
                "Water unit:",
                ["L/yr", "L/mo", "L/s", "gpm", "gal/mo"],
                help="Select the unit for water consumption"
            )
        
        # Action buttons
        st.subheader("Actions")
        
        # Create button columns for better layout
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            # (5) About the Tool button
            if st.button("ℹ️ About", use_container_width=True):
                st.info("""
                    **About This Tool**
                    
                    This application helps estimate environmental impacts by visualizing 
                    county-level data for selected U.S. states.
                    
                    **Available Metrics:**
                    - **Carbon footprint**: kg CO₂ equivalent per kWh
                    - **Scope 1 & 2 water footprint**: Liters of water per kWh
                    - **Water scarcity footprint**: Liters water-equivalent per kWh
                    
                    **How to Use:**
                    1. Select a state and environmental metric
                    2. Enter your facility's power and water consumption
                    3. Click "Make Plot" to visualize county-level impacts
                    
                    **Color Coding:**
                    - 🟢 Green: Bottom 33% (lowest impact)
                    - 🟡 Yellow: Middle 33% (medium impact)  
                    - 🔴 Red: Top 33% (highest impact)
                """)
        
        with btn_col2:
            # (6) Make Plot button
            make_plot = st.button("📊 Make Plot", use_container_width=True, type="primary")
        
        with btn_col3:
            # (7) Exit button
            if st.button("🚪 Exit", use_container_width=True):
                st.warning("👋 Thank you for using the Environmental Impact Explorer!")
                st.balloons()
                st.stop()
    
    # Main content area
    with col2:
        if make_plot:
            # Validate inputs if provided
            power_valid = True
            water_valid = True
            
            if power_value.strip():
                power_valid, power_numeric = validate_numeric_input(power_value, "Power consumption")
            
            if water_value.strip():
                water_valid, water_numeric = validate_numeric_input(water_value, "Water consumption")
            
            if power_valid and water_valid:
                # Create the plot
                create_environmental_map(data, metric_option, state)
                
                # Display facility impact if inputs provided
                if power_value.strip() and water_value.strip():
                    calculate_facility_impact(power_numeric, power_unit, water_numeric, water_unit, metric_option)
        else:
            # Show instructions when no plot is displayed
            st.subheader("Welcome! 👋")
            st.markdown("""
                **Get Started:**
                1. Select your state and environmental metric on the left
                2. Optionally enter your facility's consumption data
                3. Click "Make Plot" to visualize environmental impacts
                
                **Features:**
                - Interactive county-level maps
                - Multiple environmental metrics
                - Facility impact calculations
                - Color-coded impact levels
            """)
            
            # Show a sample visualization placeholder
            st.image("https://via.placeholder.com/600x400/E8F4FD/1E88E5?text=Environmental+Impact+Map+Will+Appear+Here", 
                    caption="Your environmental impact map will appear here")

def create_environmental_map(data: Dict[str, Any], metric_option: str, state: str):
    """
    Create and display the environmental impact map.
    
    Args:
        data: Dictionary containing the environmental data
        metric_option: Selected environmental metric
        state: Selected state (or "USA" for all)
    """
    # Map metric names to data arrays
    metric_map = {
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }
    
    # Get the values for the selected metric
    values = metric_map[metric_option]
    fips = data["CountyFIPS"]
    
    # Create a DataFrame for easier manipulation
    # Convert FIPS codes to strings with leading zeros (5 digits total)
    fips_strings = [str(int(fips_code)).zfill(5) for fips_code in fips]
    
    df = pd.DataFrame({
        "fips": fips_strings,
        "value": values
    })
    
    # Remove any invalid values
    df = df.dropna()
    df = df[df["value"] > 0]  # Remove zero or negative values
    
    # Calculate percentiles for color categories
    low_percentile = np.percentile(df['value'], 33)
    high_percentile = np.percentile(df['value'], 66)
    
    # Create color categories
    def categorize_value(val):
        if val <= low_percentile:
            return "Low Impact"
        elif val <= high_percentile:
            return "Medium Impact"
        else:
            return "High Impact"
    
    df["category"] = df["value"].apply(categorize_value)
    df["formatted_value"] = df["value"].round(4)  # Round for display
    
    # Create the choropleth map
    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations="fips",
        color="category",
        color_discrete_map={
            "Low Impact": "#2E8B57",      # Sea Green
            "Medium Impact": "#FFD700",    # Gold
            "High Impact": "#DC143C"       # Crimson
        },
        scope="usa",
        labels={"category": "Impact Level", "formatted_value": f"{metric_option.title()}"},
        title=f"{metric_option.title()} by County",
        hover_data=["formatted_value"]
    )
    
    # Customize the map appearance
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Display the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    st.subheader("📊 Statistics")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric(
            "Low Impact Counties",
            f"{len(df[df['category'] == 'Low Impact'])} counties",
            f"≤ {low_percentile:.4f}"
        )
    
    with stat_col2:
        st.metric(
            "Medium Impact Counties",
            f"{len(df[df['category'] == 'Medium Impact'])} counties",
            f"{low_percentile:.4f} - {high_percentile:.4f}"
        )
    
    with stat_col3:
        st.metric(
            "High Impact Counties",
            f"{len(df[df['category'] == 'High Impact'])} counties",
            f"> {high_percentile:.4f}"
        )

def calculate_facility_impact(power_value: float, power_unit: str, water_value: float, water_unit: str, metric_option: str):
    """
    Calculate and display the environmental impact of the user's facility.
    
    Args:
        power_value: Power consumption value
        power_unit: Power consumption unit
        water_value: Water consumption value
        water_unit: Water consumption unit
        metric_option: Selected environmental metric
    """
    # Convert to standard units
    power_kwh_per_year = convert_power_to_kwh_per_year(power_value, power_unit)
    water_liters_per_year = convert_water_to_liters_per_year(water_value, water_unit)
    
    st.subheader("🏭 Your Facility's Impact")
    
    # Display converted values
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        st.metric(
            "Annual Power Consumption",
            f"{power_kwh_per_year:,.0f} kWh/year",
            f"From {power_value} {power_unit}"
        )
    
    with impact_col2:
        st.metric(
            "Annual Water Consumption",
            f"{water_liters_per_year:,.0f} L/year",
            f"From {water_value} {water_unit}"
        )
    
    # Note about impact calculations
    st.info("""
        💡 **Note**: To calculate your facility's total environmental impact, 
        multiply your annual power consumption by the county-specific factor 
        shown in the map above for your facility's location.
    """)

# -------------- RUN THE APP --------------
if __name__ == "__main__":
    main()
