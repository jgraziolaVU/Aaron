# app.py - Environmental Impact Explorer with Debug Features
# A beginner-friendly Streamlit app for visualizing environmental impacts with detailed debugging

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple

# -------------- CONFIGURATION --------------
# Set up the page configuration (this should be the first Streamlit command)
st.set_page_config(
    page_title="Environmental Impact Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------- HELPER FUNCTIONS --------------
def convert_power_to_kwh_per_year(value: float, unit: str) -> Tuple[float, Dict[str, Any]]:
    """
    Convert different power units to kWh/year for calculations.
    
    Args:
        value: The numerical value
        unit: The unit of measurement
    
    Returns:
        Tuple[float, Dict]: (converted_value, debug_info)
    """
    debug_info = {
        "input_value": value,
        "input_unit": unit,
        "conversion_factor": None,
        "calculation_steps": [],
        "output_value": 0,
        "output_unit": "kWh/yr"
    }
    
    if unit == "kWh/yr":
        debug_info["conversion_factor"] = 1
        debug_info["calculation_steps"].append(f"{value} kWh/yr × 1 = {value} kWh/yr")
        result = value
    elif unit == "kWh/mo":
        debug_info["conversion_factor"] = 12
        debug_info["calculation_steps"].append(f"{value} kWh/mo × 12 months/year = {value * 12} kWh/yr")
        result = value * 12
    elif unit == "kW":
        debug_info["conversion_factor"] = 8760
        debug_info["calculation_steps"].extend([
            f"Hours per year = 365.25 days/year × 24 hours/day = 8760 hours/year",
            f"{value} kW × 8760 hours/year = {value * 8760} kWh/yr"
        ])
        result = value * 8760
    elif unit == "MW":
        debug_info["conversion_factor"] = 1000 * 8760
        debug_info["calculation_steps"].extend([
            f"Convert MW to kW: {value} MW × 1000 kW/MW = {value * 1000} kW",
            f"Hours per year = 365.25 days/year × 24 hours/day = 8760 hours/year",
            f"{value * 1000} kW × 8760 hours/year = {value * 1000 * 8760} kWh/yr"
        ])
        result = value * 1000 * 8760
    else:
        debug_info["calculation_steps"].append(f"Unknown unit '{unit}' - returning 0")
        result = 0
    
    debug_info["output_value"] = result
    return result, debug_info

def convert_water_to_liters_per_year(value: float, unit: str) -> Tuple[float, Dict[str, Any]]:
    """
    Convert different water units to liters/year for calculations.
    
    Args:
        value: The numerical value
        unit: The unit of measurement
    
    Returns:
        Tuple[float, Dict]: (converted_value, debug_info)
    """
    debug_info = {
        "input_value": value,
        "input_unit": unit,
        "conversion_factor": None,
        "calculation_steps": [],
        "output_value": 0,
        "output_unit": "L/yr"
    }
    
    if unit == "L/yr":
        debug_info["conversion_factor"] = 1
        debug_info["calculation_steps"].append(f"{value} L/yr × 1 = {value} L/yr")
        result = value
    elif unit == "L/mo":
        debug_info["conversion_factor"] = 12
        debug_info["calculation_steps"].append(f"{value} L/mo × 12 months/year = {value * 12} L/yr")
        result = value * 12
    elif unit == "L/s":
        seconds_per_year = 31536000  # 365.25 * 24 * 3600
        debug_info["conversion_factor"] = seconds_per_year
        debug_info["calculation_steps"].extend([
            f"Seconds per year = 365.25 days/year × 24 hours/day × 3600 seconds/hour = {seconds_per_year:,} seconds/year",
            f"{value} L/s × {seconds_per_year:,} seconds/year = {value * seconds_per_year:,.0f} L/yr"
        ])
        result = value * seconds_per_year
    elif unit == "gpm":  # gallons per minute
        minutes_per_year = 525600  # 365.25 * 24 * 60
        liters_per_gallon = 3.78541
        debug_info["conversion_factor"] = minutes_per_year * liters_per_gallon
        debug_info["calculation_steps"].extend([
            f"Minutes per year = 365.25 days/year × 24 hours/day × 60 minutes/hour = {minutes_per_year:,} minutes/year",
            f"Liters per gallon = {liters_per_gallon} L/gal",
            f"{value} gpm × {minutes_per_year:,} minutes/year × {liters_per_gallon} L/gal = {value * minutes_per_year * liters_per_gallon:,.0f} L/yr"
        ])
        result = value * minutes_per_year * liters_per_gallon
    elif unit == "gal/mo":  # gallons per month
        months_per_year = 12
        liters_per_gallon = 3.78541
        debug_info["conversion_factor"] = months_per_year * liters_per_gallon
        debug_info["calculation_steps"].extend([
            f"Months per year = {months_per_year} months/year",
            f"Liters per gallon = {liters_per_gallon} L/gal",
            f"{value} gal/mo × {months_per_year} months/year × {liters_per_gallon} L/gal = {value * months_per_year * liters_per_gallon:,.1f} L/yr"
        ])
        result = value * months_per_year * liters_per_gallon
    else:
        debug_info["calculation_steps"].append(f"Unknown unit '{unit}' - returning 0")
        result = 0
    
    debug_info["output_value"] = result
    return result, debug_info

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

def analyze_data_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the quality and characteristics of the loaded data.
    
    Args:
        data: Dictionary containing the environmental data
        
    Returns:
        Dict containing data quality analysis
    """
    analysis = {
        "total_counties": len(data["CountyFIPS"]),
        "metrics_analysis": {},
        "data_ranges": {},
        "missing_data": {},
        "statistical_summary": {}
    }
    
    metrics = ["AWAREUSCF", "EFkgkWh", "EWIF"]
    metric_names = ["Water Scarcity Footprint", "Carbon Footprint", "Water Footprint"]
    
    for i, metric in enumerate(metrics):
        values = data[metric]
        valid_values = values[~np.isnan(values) & (values > 0)]
        
        analysis["metrics_analysis"][metric] = {
            "name": metric_names[i],
            "total_values": len(values),
            "valid_values": len(valid_values),
            "invalid_values": len(values) - len(valid_values),
            "percent_valid": (len(valid_values) / len(values)) * 100 if len(values) > 0 else 0
        }
        
        if len(valid_values) > 0:
            analysis["data_ranges"][metric] = {
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "mean": float(np.mean(valid_values)),
                "median": float(np.median(valid_values)),
                "std": float(np.std(valid_values))
            }
            
            analysis["statistical_summary"][metric] = {
                "percentile_33": float(np.percentile(valid_values, 33)),
                "percentile_66": float(np.percentile(valid_values, 66)),
                "percentile_95": float(np.percentile(valid_values, 95))
            }
        else:
            analysis["data_ranges"][metric] = None
            analysis["statistical_summary"][metric] = None
    
    return analysis

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
        data = {
            "AWAREUSCF": metrics["AWAREUSCF"].flatten(),    # Water scarcity footprint
            "EFkgkWh": metrics["EFkgkWh"].flatten(),        # Carbon footprint
            "EWIF": metrics["EWIF"].flatten(),              # Water footprint
            "CountyFIPS": metrics["CountyFIPS"].flatten(),  # County identification codes
        }
        
        # Add data quality analysis
        data["_quality_analysis"] = analyze_data_quality(data)
        
        return data
    except FileNotFoundError:
        st.error("Data file 'CountyLevelMetrics.mat' not found. Please ensure it is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def generate_debug_report(debug_data: Dict[str, Any]) -> str:
    """
    Generate a comprehensive debug report in text format.
    
    Args:
        debug_data: Dictionary containing all debug information
        
    Returns:
        str: Formatted debug report
    """
    report = f"""
ENVIRONMENTAL IMPACT CALCULATOR - DEBUG REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=================================================================

INPUT PARAMETERS
----------------
Selected State: {debug_data.get('state', 'N/A')}
Selected Metric: {debug_data.get('metric', 'N/A')}
Power Input: {debug_data.get('power_input', {}).get('input_value', 'N/A')} {debug_data.get('power_input', {}).get('input_unit', '')}
Water Input: {debug_data.get('water_input', {}).get('input_value', 'N/A')} {debug_data.get('water_input', {}).get('input_unit', '')}

POWER CONSUMPTION CONVERSION
----------------------------
"""
    
    if 'power_conversion' in debug_data:
        power = debug_data['power_conversion']
        report += f"Input: {power['input_value']} {power['input_unit']}\n"
        report += f"Conversion Factor: {power['conversion_factor']}\n"
        report += f"Calculation Steps:\n"
        for step in power['calculation_steps']:
            report += f"  • {step}\n"
        report += f"Final Result: {power['output_value']:,.2f} {power['output_unit']}\n\n"
    
    report += """
WATER CONSUMPTION CONVERSION
----------------------------
"""
    
    if 'water_conversion' in debug_data:
        water = debug_data['water_conversion']
        report += f"Input: {water['input_value']} {water['input_unit']}\n"
        report += f"Conversion Factor: {water['conversion_factor']}\n"
        report += f"Calculation Steps:\n"
        for step in water['calculation_steps']:
            report += f"  • {step}\n"
        report += f"Final Result: {water['output_value']:,.2f} {water['output_unit']}\n\n"
    
    if 'data_analysis' in debug_data:
        analysis = debug_data['data_analysis']
        report += f"""
DATA QUALITY ANALYSIS
---------------------
Total Counties in Dataset: {analysis['total_counties']:,}

Metric Quality Summary:
"""
        for metric, info in analysis['metrics_analysis'].items():
            report += f"""
{info['name']} ({metric}):
  • Total Values: {info['total_values']:,}
  • Valid Values: {info['valid_values']:,} ({info['percent_valid']:.1f}%)
  • Invalid Values: {info['invalid_values']:,}
"""
            
            if metric in analysis['data_ranges'] and analysis['data_ranges'][metric]:
                ranges = analysis['data_ranges'][metric]
                report += f"""  • Data Range: {ranges['min']:.6f} to {ranges['max']:.6f}
  • Mean: {ranges['mean']:.6f}
  • Median: {ranges['median']:.6f}
  • Standard Deviation: {ranges['std']:.6f}
"""
                
                if metric in analysis['statistical_summary']:
                    stats = analysis['statistical_summary'][metric]
                    report += f"""  • 33rd Percentile: {stats['percentile_33']:.6f}
  • 66th Percentile: {stats['percentile_66']:.6f}
  • 95th Percentile: {stats['percentile_95']:.6f}
"""
    
    if 'map_data' in debug_data:
        map_data = debug_data['map_data']
        report += f"""
MAP DATA PROCESSING
-------------------
Counties Processed: {map_data.get('counties_processed', 'N/A')}
Counties with Valid Data: {map_data.get('valid_counties', 'N/A')}
Data Filtering Steps:
"""
        for step in map_data.get('filtering_steps', []):
            report += f"  • {step}\n"
        
        if 'percentile_thresholds' in map_data:
            thresholds = map_data['percentile_thresholds']
            report += f"""
Impact Category Thresholds:
  • Low Impact (Bottom 33%): ≤ {thresholds['low']:.6f}
  • Medium Impact (Middle 33%): {thresholds['low']:.6f} to {thresholds['high']:.6f}
  • High Impact (Top 33%): > {thresholds['high']:.6f}

County Distribution:
  • Low Impact Counties: {map_data.get('low_impact_count', 0)}
  • Medium Impact Counties: {map_data.get('medium_impact_count', 0)}
  • High Impact Counties: {map_data.get('high_impact_count', 0)}
"""
    
    if 'facility_impact' in debug_data:
        facility = debug_data['facility_impact']
        report += f"""
FACILITY IMPACT CALCULATION
----------------------------
Annual Power Consumption: {facility.get('annual_power_kwh', 0):,.2f} kWh/year
Annual Water Consumption: {facility.get('annual_water_liters', 0):,.2f} L/year

To Calculate Total Environmental Impact:
1. Find your facility's county on the map
2. Note the county's impact factor for the selected metric
3. Multiply: Annual Power (kWh/year) × County Factor = Total Impact

Example Calculation (using median county factor):
  Power Impact = {facility.get('annual_power_kwh', 0):,.0f} kWh/year × [County Factor] = [Total Impact]
"""
    
    report += f"""
=================================================================
End of Debug Report
Generated by Environmental Impact Explorer v2.0
"""
    
    return report

# -------------- MAIN APP --------------
def main():
    """Main application function that contains all the UI and logic."""
    
    # Load the data
    data = load_data()
    
    # Initialize debug data storage
    if 'debug_data' not in st.session_state:
        st.session_state.debug_data = {}
    
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
        
        # Debug options
        st.subheader("Debug Options")
        
        debug_col1, debug_col2 = st.columns(2)
        with debug_col1:
            show_debug = st.checkbox("🔍 Show Debug Info", 
                                   help="Display detailed calculation steps and data analysis")
        with debug_col2:
            show_data_quality = st.checkbox("📊 Show Data Quality", 
                                          help="Display data quality analysis and statistics")
        
        # Action buttons
        st.subheader("Actions")
        
        # Create button columns for better layout
        btn_col1, btn_col2 = st.columns(2)
        
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
                    4. Enable debug options for detailed calculations
                    
                    **Color Coding:**
                    - 🟢 Green: Bottom 33% (lowest impact)
                    - 🟡 Yellow: Middle 33% (medium impact)  
                    - 🔴 Red: Top 33% (highest impact)
                """)
        
        with btn_col2:
            # (6) Make Plot button
            make_plot = st.button("📊 Make Plot", use_container_width=True, type="primary")
        
        # Download debug report button (full width)
        if st.session_state.debug_data and st.button("📄 Download Debug Report", use_container_width=True):
            debug_report = generate_debug_report(st.session_state.debug_data)
            st.download_button(
                label="💾 Download Report",
                data=debug_report,
                file_name=f"environmental_impact_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # (7) Exit button
        if st.button("🚪 Exit", use_container_width=True):
            st.warning("👋 Thank you for using the Environmental Impact Explorer!")
            st.balloons()
            st.stop()
    
    # Main content area
    with col2:
        if make_plot:
            # Clear previous debug data
            st.session_state.debug_data = {
                "state": state,
                "metric": metric_option,
                "timestamp": datetime.now().isoformat(),
                "data_analysis": data["_quality_analysis"]
            }
            
            # Validate inputs if provided
            power_valid = True
            water_valid = True
            power_numeric = 0
            water_numeric = 0
            
            if power_value.strip():
                power_valid, power_numeric = validate_numeric_input(power_value, "Power consumption")
                if power_valid:
                    st.session_state.debug_data["power_input"] = {
                        "input_value": power_numeric,
                        "input_unit": power_unit
                    }
            
            if water_value.strip():
                water_valid, water_numeric = validate_numeric_input(water_value, "Water consumption")
                if water_valid:
                    st.session_state.debug_data["water_input"] = {
                        "input_value": water_numeric,
                        "input_unit": water_unit
                    }
            
            if power_valid and water_valid:
                # Create the plot
                create_environmental_map(data, metric_option, state, show_debug, show_data_quality)
                
                # Display facility impact if inputs provided
                if power_value.strip() and water_value.strip():
                    calculate_facility_impact(power_numeric, power_unit, water_numeric, water_unit, metric_option, show_debug)
        else:
            # Show instructions when no plot is displayed
            st.subheader("Welcome! 👋")
            st.markdown("""
                **Get Started:**
                1. Select your state and environmental metric on the left
                2. Optionally enter your facility's consumption data
                3. Click "Make Plot" to visualize environmental impacts
                
                **New Debug Features:**
                - ✅ **Show Debug Info**: See detailed calculation steps
                - ✅ **Show Data Quality**: View data statistics and validation
                - ✅ **Download Debug Report**: Get a comprehensive analysis file
                
                **Features:**
                - Interactive county-level maps
                - Multiple environmental metrics
                - Facility impact calculations
                - Color-coded impact levels
                - Detailed debugging and validation
            """)
            
            # Show a sample visualization placeholder
            st.image("https://via.placeholder.com/600x400/E8F4FD/1E88E5?text=Environmental+Impact+Map+Will+Appear+Here", 
                    caption="Your environmental impact map will appear here")

def create_environmental_map(data: Dict[str, Any], metric_option: str, state: str, show_debug: bool, show_data_quality: bool):
    """
    Create and display the environmental impact map with debug information.
    
    Args:
        data: Dictionary containing the environmental data
        metric_option: Selected environmental metric
        state: Selected state (or "USA" for all)
        show_debug: Whether to show debug information
        show_data_quality: Whether to show data quality information
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
    
    # Debug: Track data processing steps
    debug_info = {
        "counties_processed": len(fips),
        "filtering_steps": [],
        "valid_counties": 0,
        "percentile_thresholds": {},
        "low_impact_count": 0,
        "medium_impact_count": 0,
        "high_impact_count": 0
    }
    
    debug_info["filtering_steps"].append(f"Initial dataset: {len(fips)} counties")
    
    # Create a DataFrame for easier manipulation
    # Convert FIPS codes to strings with leading zeros (5 digits total)
    fips_strings = [str(int(fips_code)).zfill(5) for fips_code in fips]
    
    df = pd.DataFrame({
        "fips": fips_strings,
        "value": values
    })
    
    debug_info["filtering_steps"].append(f"After creating DataFrame: {len(df)} rows")
    
    # Remove any invalid values
    df = df.dropna()
    debug_info["filtering_steps"].append(f"After removing NaN values: {len(df)} rows")
    
    df = df[df["value"] > 0]  # Remove zero or negative values
    debug_info["filtering_steps"].append(f"After removing zero/negative values: {len(df)} rows")
    debug_info["valid_counties"] = len(df)
    
    if len(df) == 0:
        st.error("No valid data found for the selected metric.")
        return
    
    # Calculate percentiles for color categories
    low_percentile = np.percentile(df['value'], 33)
    high_percentile = np.percentile(df['value'], 66)
    
    debug_info["percentile_thresholds"] = {
        "low": low_percentile,
        "high": high_percentile
    }
    
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
    
    # Count categories for debug
    debug_info["low_impact_count"] = len(df[df["category"] == "Low Impact"])
    debug_info["medium_impact_count"] = len(df[df["category"] == "Medium Impact"])
    debug_info["high_impact_count"] = len(df[df["category"] == "High Impact"])
    
    # Store debug info in session state
    st.session_state.debug_data["map_data"] = debug_info
    
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
            f"{debug_info['low_impact_count']} counties",
            f"≤ {low_percentile:.4f}"
        )
    
    with stat_col2:
        st.metric(
            "Medium Impact Counties",
            f"{debug_info['medium_impact_count']} counties",
            f"{low_percentile:.4f} - {high_percentile:.4f}"
        )
    
    with stat_col3:
        st.metric(
            "High Impact Counties",
            f"{debug_info['high_impact_count']} counties",
            f"> {high_percentile:.4f}"
        )
    
    # Show debug information if requested
    if show_debug:
        with st.expander("🔍 Debug Information - Map Processing", expanded=True):
            st.subheader("Data Processing Steps")
            for i, step in enumerate(debug_info["filtering_steps"], 1):
                st.write(f"{i}. {step}")
            
            st.subheader("Percentile Calculations")
            st.write(f"**33rd Percentile (Low/Medium threshold):** {low_percentile:.6f}")
            st.write(f"**66th Percentile (Medium/High threshold):** {high_percentile:.6f}")
            
            st.subheader("Category Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Impact", f"{debug_info['low_impact_count']}")
            with col2:
                st.metric("Medium Impact", f"{debug_info['medium_impact_count']}")
            with col3:
                st.metric("High Impact", f"{debug_info['high_impact_count']}")
    
    # Show data quality information if requested
    if show_data_quality:
        with st.expander("📊 Data Quality Analysis", expanded=True):
            quality = data["_quality_analysis"]
            
            st.subheader("Dataset Overview")
            st.metric("Total Counties", f"{quality['total_counties']:,}")
            
            st.subheader("Metric Quality Summary")
            for metric, info in quality["metrics_analysis"].items():
                if metric_option in info["name"].lower() or metric in metric_map:
                    st.write(f"**{info['name']}:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Valid Values", f"{info['valid_values']:,}")
                    with col2:
                        st.metric("Invalid Values", f"{info['invalid_values']:,}")
                    with col3:
                        st.metric("Validity Rate", f"{info['percent_valid']:.1f}%")
                    
                    if metric in quality["data_ranges"] and quality["data_ranges"][metric]:
                        ranges = quality["data_ranges"][metric]
                        st.write("**Statistical Summary:**")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric("Min", f"{ranges['min']:.6f}")
                        with stat_col2:
                            st.metric("Max", f"{ranges['max']:.6f}")
                        with stat_col3:
                            st.metric("Mean", f"{ranges['mean']:.6f}")
                        with stat_col4:
                            st.metric("Median", f"{ranges['median']:.6f}")

def calculate_facility_impact(power_value: float, power_unit: str, water_value: float, water_unit: str, metric_option: str, show_debug: bool):
    """
    Calculate and display the environmental impact of the user's facility with debug information.
    
    Args:
        power_value: Power consumption value
        power_unit: Power consumption unit
        water_value: Water consumption value
        water_unit: Water consumption unit
        metric_option: Selected environmental metric
        show_debug: Whether to show debug information
    """
    # Convert to standard units with debug info
    power_kwh_per_year, power_debug = convert_power_to_kwh_per_year(power_value, power_unit)
    water_liters_per_year, water_debug = convert_water_to_liters_per_year(water_value, water_unit)
    
    # Store debug info
    st.session_state.debug_data["power_conversion"] = power_debug
    st.session_state.debug_data["water_conversion"] = water_debug
    st.session_state.debug_data["facility_impact"] = {
        "annual_power_kwh": power_kwh_per_year,
        "annual_water_liters": water_liters_per_year
    }
    
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
    
    # Show debug information for conversions if requested
    if show_debug:
        with st.expander("🔍 Debug Information - Unit Conversions", expanded=True):
            st.subheader("Power Consumption Conversion")
            st.write(f"**Input:** {power_debug['input_value']} {power_debug['input_unit']}")
            st.write(f"**Conversion Factor:** {power_debug['conversion_factor']}")
            st.write("**Calculation Steps:**")
            for step in power_debug['calculation_steps']:
                st.write(f"• {step}")
            st.write(f"**Final Result:** {power_debug['output_value']:,.2f} {power_debug['output_unit']}")
            
            st.subheader("Water Consumption Conversion")
            st.write(f"**Input:** {water_debug['input_value']} {water_debug['input_unit']}")
            st.write(f"**Conversion Factor:** {water_debug['conversion_factor']}")
            st.write("**Calculation Steps:**")
            for step in water_debug['calculation_steps']:
                st.write(f"• {step}")
            st.write(f"**Final Result:** {water_debug['output_value']:,.2f} {water_debug['output_unit']}")
    
    # Note about impact calculations
    st.info("""
        💡 **Note**: To calculate your facility's total environmental impact, 
        multiply your annual power consumption by the county-specific factor 
        shown in the map above for your facility's location.
        
        **Formula:** Annual Power (kWh/year) × County Factor = Total Environmental Impact
    """)

# -------------- RUN THE APP --------------
if __name__ == "__main__":
    main()
