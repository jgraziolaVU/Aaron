# app.py - Enhanced Environmental Impact Explorer with Complete Debug Features
# A comprehensive Streamlit app for calculating and visualizing environmental impacts with detailed debugging

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# -------------- CONFIGURATION --------------
# Set up the page configuration (this should be the first Streamlit command)
st.set_page_config(
    page_title="Environmental Impact Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------- CONSTANTS --------------
# Engineering benchmarks for context
FACILITY_BENCHMARKS = {
    "residential_small": {"power_kwh": 5000, "description": "Small residential home"},
    "residential_large": {"power_kwh": 15000, "description": "Large residential home"},
    "commercial_small": {"power_kwh": 50000, "description": "Small commercial building"},
    "commercial_large": {"power_kwh": 200000, "description": "Large commercial building"},
    "industrial_small": {"power_kwh": 500000, "description": "Small industrial facility"},
    "industrial_large": {"power_kwh": 5000000, "description": "Large industrial facility"}
}

STATE_FIPS_MAPPING = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05", "California": "06",
    "Colorado": "08", "Connecticut": "09", "Delaware": "10", "Florida": "12", "Georgia": "13",
    "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19", "Kansas": "20",
    "Kentucky": "21", "Louisiana": "22", "Maine": "23", "Maryland": "24", "Massachusetts": "25",
    "Michigan": "26", "Minnesota": "27", "Mississippi": "28", "Missouri": "29", "Montana": "30",
    "Nebraska": "31", "Nevada": "32", "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35",
    "New York": "36", "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40",
    "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45",
    "South Dakota": "46", "Tennessee": "47", "Texas": "48", "Utah": "49", "Vermont": "50",
    "Virginia": "51", "Washington": "53", "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56"
}

# -------------- HELPER FUNCTIONS --------------
def categorize_facility_size(power_kwh_per_year: float) -> Dict[str, Any]:
    """
    Categorize facility size based on annual power consumption with engineering context.
    
    Args:
        power_kwh_per_year: Annual power consumption in kWh/year
        
    Returns:
        Dict containing facility category and context
    """
    if power_kwh_per_year < 10000:
        return {
            "category": "Residential Scale",
            "benchmark": "residential_small",
            "context": "Similar to a small to medium residential home",
            "typical_range": "2,000-15,000 kWh/year",
            "engineering_notes": "Very low consumption - check if this is correct for an industrial analysis"
        }
    elif power_kwh_per_year < 30000:
        return {
            "category": "Large Residential",
            "benchmark": "residential_large", 
            "context": "Similar to a large residential home or very small business",
            "typical_range": "10,000-30,000 kWh/year",
            "engineering_notes": "Residential scale - unusual for industrial facility analysis"
        }
    elif power_kwh_per_year < 100000:
        return {
            "category": "Small Commercial",
            "benchmark": "commercial_small",
            "context": "Small office building, retail store, or light manufacturing",
            "typical_range": "30,000-200,000 kWh/year",
            "engineering_notes": "Light commercial load profile"
        }
    elif power_kwh_per_year < 1000000:
        return {
            "category": "Large Commercial/Light Industrial",
            "benchmark": "commercial_large",
            "context": "Large commercial building, warehouse, or light industrial facility",
            "typical_range": "100,000-1,000,000 kWh/year", 
            "engineering_notes": "Moderate industrial load - check capacity factor assumptions"
        }
    elif power_kwh_per_year < 10000000:
        return {
            "category": "Industrial Facility",
            "benchmark": "industrial_small",
            "context": "Manufacturing plant, processing facility, or heavy industrial operation",
            "typical_range": "1,000,000-50,000,000 kWh/year",
            "engineering_notes": "Industrial scale - verify 24/7 operation assumptions"
        }
    else:
        return {
            "category": "Large Industrial Complex",
            "benchmark": "industrial_large",
            "context": "Major manufacturing complex, refinery, or industrial campus",
            "typical_range": ">10,000,000 kWh/year",
            "engineering_notes": "Very large facility - confirm power consumption accuracy"
        }

def convert_power_to_kwh_per_year(value: float, unit: str, capacity_factor: float = 1.0) -> Tuple[float, Dict[str, Any]]:
    """
    Convert different power units to kWh/year for calculations with capacity factor consideration.
    
    Args:
        value: The numerical value
        unit: The unit of measurement
        capacity_factor: Operating capacity factor (0.0 to 1.0), defaults to 1.0 for kWh units
        
    Returns:
        Tuple[float, Dict]: (converted_value, debug_info)
    """
    debug_info = {
        "input_value": value,
        "input_unit": unit,
        "capacity_factor": capacity_factor,
        "conversion_factor": None,
        "calculation_steps": [],
        "output_value": 0,
        "output_unit": "kWh/yr",
        "engineering_notes": []
    }
    
    if unit == "kWh/yr":
        debug_info["conversion_factor"] = 1
        debug_info["calculation_steps"].append(f"{value} kWh/yr × 1 = {value} kWh/yr")
        debug_info["engineering_notes"].append("Direct energy consumption - no capacity factor applied")
        result = value
    elif unit == "kWh/mo":
        debug_info["conversion_factor"] = 12
        debug_info["calculation_steps"].append(f"{value} kWh/mo × 12 months/year = {value * 12} kWh/yr")
        debug_info["engineering_notes"].append("Monthly energy consumption scaled to annual")
        result = value * 12
    elif unit == "kW":
        hours_per_year = 8760
        debug_info["conversion_factor"] = hours_per_year * capacity_factor
        debug_info["calculation_steps"].extend([
            f"Hours per year = 365.25 days/year × 24 hours/day = {hours_per_year:,} hours/year",
            f"Applying capacity factor of {capacity_factor:.1%} for realistic operation",
            f"{value} kW × {hours_per_year:,} hours/year × {capacity_factor:.3f} = {value * hours_per_year * capacity_factor:,.0f} kWh/yr"
        ])
        debug_info["engineering_notes"].extend([
            f"Power rating converted to energy using {capacity_factor:.1%} capacity factor",
            "Industrial facilities typically operate at 70-85% capacity factor",
            "24/7 operation (100% capacity factor) is rare except for continuous processes"
        ])
        result = value * hours_per_year * capacity_factor
    elif unit == "MW":
        hours_per_year = 8760
        kw_conversion = 1000
        debug_info["conversion_factor"] = kw_conversion * hours_per_year * capacity_factor
        debug_info["calculation_steps"].extend([
            f"Convert MW to kW: {value} MW × {kw_conversion} kW/MW = {value * kw_conversion} kW",
            f"Hours per year = 365.25 days/year × 24 hours/day = {hours_per_year:,} hours/year",
            f"Applying capacity factor of {capacity_factor:.1%} for realistic operation",
            f"{value * kw_conversion} kW × {hours_per_year:,} hours/year × {capacity_factor:.3f} = {value * kw_conversion * hours_per_year * capacity_factor:,.0f} kWh/yr"
        ])
        debug_info["engineering_notes"].extend([
            f"Large power rating converted with {capacity_factor:.1%} capacity factor",
            "MW-scale facilities require careful capacity factor analysis",
            "Consider load profiles, maintenance downtime, and operational patterns"
        ])
        result = value * kw_conversion * hours_per_year * capacity_factor
    else:
        debug_info["calculation_steps"].append(f"Unknown unit '{unit}' - returning 0")
        debug_info["engineering_notes"].append("ERROR: Unknown power unit provided")
        result = 0
    
    debug_info["output_value"] = result
    return result, debug_info

def convert_water_to_liters_per_year(value: float, unit: str) -> Tuple[float, Dict[str, Any]]:
    """
    Convert different water units to liters/year for calculations with engineering validation.
    
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
        "output_unit": "L/yr",
        "engineering_notes": []
    }
    
    if unit == "L/yr":
        debug_info["conversion_factor"] = 1
        debug_info["calculation_steps"].append(f"{value} L/yr × 1 = {value} L/yr")
        debug_info["engineering_notes"].append("Direct annual water consumption")
        result = value
    elif unit == "L/mo":
        debug_info["conversion_factor"] = 12
        debug_info["calculation_steps"].append(f"{value} L/mo × 12 months/year = {value * 12} L/yr")
        debug_info["engineering_notes"].append("Monthly consumption scaled to annual - consider seasonal variations")
        result = value * 12
    elif unit == "L/s":
        seconds_per_year = 31536000  # 365.25 * 24 * 3600
        debug_info["conversion_factor"] = seconds_per_year
        debug_info["calculation_steps"].extend([
            f"Seconds per year = 365.25 days/year × 24 hours/day × 3600 seconds/hour = {seconds_per_year:,} seconds/year",
            f"{value} L/s × {seconds_per_year:,} seconds/year = {value * seconds_per_year:,.0f} L/yr"
        ])
        debug_info["engineering_notes"].extend([
            "Flow rate converted assuming continuous 24/7/365 operation",
            "Industrial processes rarely operate at constant flow rates",
            "Consider peak vs. average flow rates and operational schedules"
        ])
        result = value * seconds_per_year
    elif unit == "gpm":  # gallons per minute
        minutes_per_year = 525600  # 365.25 * 24 * 60
        liters_per_gallon = 3.78541
        debug_info["conversion_factor"] = minutes_per_year * liters_per_gallon
        debug_info["calculation_steps"].extend([
            f"Minutes per year = 365.25 days/year × 24 hours/day × 60 minutes/hour = {minutes_per_year:,} minutes/year",
            f"Liters per gallon = {liters_per_gallon} L/gal (US gallon)",
            f"{value} gpm × {minutes_per_year:,} minutes/year × {liters_per_gallon} L/gal = {value * minutes_per_year * liters_per_gallon:,.0f} L/yr"
        ])
        debug_info["engineering_notes"].extend([
            "Flow rate in US gallons per minute converted to annual consumption",
            "Assumes continuous 24/7/365 operation - verify operational schedule",
            "Consider if this represents peak, average, or design flow rate"
        ])
        result = value * minutes_per_year * liters_per_gallon
    elif unit == "gal/mo":  # gallons per month
        months_per_year = 12
        liters_per_gallon = 3.78541
        debug_info["conversion_factor"] = months_per_year * liters_per_gallon
        debug_info["calculation_steps"].extend([
            f"Months per year = {months_per_year} months/year",
            f"Liters per gallon = {liters_per_gallon} L/gal (US gallon)",
            f"{value} gal/mo × {months_per_year} months/year × {liters_per_gallon} L/gal = {value * months_per_year * liters_per_gallon:,.1f} L/yr"
        ])
        debug_info["engineering_notes"].extend([
            "Monthly gallons scaled to annual consumption",
            "Consider seasonal variations in water usage patterns"
        ])
        result = value * months_per_year * liters_per_gallon
    else:
        debug_info["calculation_steps"].append(f"Unknown unit '{unit}' - returning 0")
        debug_info["engineering_notes"].append("ERROR: Unknown water unit provided")
        result = 0
    
    debug_info["output_value"] = result
    return result, debug_info

def calculate_environmental_impact(power_kwh_per_year: float, metric_values: np.ndarray, 
                                 metric_name: str, state: str = "USA") -> Dict[str, Any]:
    """
    Calculate the actual environmental impact using facility consumption and regional factors.
    
    Args:
        power_kwh_per_year: Annual power consumption in kWh/year
        metric_values: Array of environmental impact factors by county
        metric_name: Name of the environmental metric
        state: Selected state or "USA" for national
        
    Returns:
        Dict containing detailed impact calculations
    """
    # Remove invalid values from metric data
    valid_values = metric_values[~np.isnan(metric_values) & (metric_values > 0)]
    
    if len(valid_values) == 0:
        return {
            "error": "No valid environmental data available",
            "impact_range": {"min": 0, "max": 0, "median": 0},
            "facility_impact": {"min": 0, "max": 0, "median": 0, "unit": ""}
        }
    
    # Calculate statistical summary
    impact_stats = {
        "min_factor": float(np.min(valid_values)),
        "max_factor": float(np.max(valid_values)),
        "mean_factor": float(np.mean(valid_values)),
        "median_factor": float(np.median(valid_values)),
        "std_factor": float(np.std(valid_values)),
        "percentile_25": float(np.percentile(valid_values, 25)),
        "percentile_75": float(np.percentile(valid_values, 75))
    }
    
    # Calculate facility impact range
    facility_impact = {
        "min_impact": power_kwh_per_year * impact_stats["min_factor"],
        "max_impact": power_kwh_per_year * impact_stats["max_factor"],
        "mean_impact": power_kwh_per_year * impact_stats["mean_factor"],
        "median_impact": power_kwh_per_year * impact_stats["median_factor"]
    }
    
    # Determine impact units and interpretation
    if "carbon" in metric_name.lower():
        impact_unit = "kg CO₂ equiv/year"
        facility_impact["tons_co2_equiv"] = facility_impact["median_impact"] / 1000
        interpretation = f"Your facility produces approximately {facility_impact['tons_co2_equiv']:.2f} metric tons of CO₂ equivalent per year"
    elif "water" in metric_name.lower():
        impact_unit = "L water/year"
        facility_impact["megaliters"] = facility_impact["median_impact"] / 1000000
        interpretation = f"Your facility has a water footprint of approximately {facility_impact['megaliters']:.2f} megaliters per year"
    else:
        impact_unit = "impact units/year"
        interpretation = f"Your facility's environmental impact is {facility_impact['median_impact']:.0f} {impact_unit}"
    
    # Engineering assessment
    facility_size = categorize_facility_size(power_kwh_per_year)
    
    return {
        "impact_statistics": impact_stats,
        "facility_impact": facility_impact,
        "impact_unit": impact_unit,
        "interpretation": interpretation,
        "facility_assessment": facility_size,
        "calculation_details": {
            "power_consumption_kwh": power_kwh_per_year,
            "counties_analyzed": len(valid_values),
            "median_factor": impact_stats["median_factor"],
            "calculation": f"{power_kwh_per_year:,.0f} kWh/year × {impact_stats['median_factor']:.6f} = {facility_impact['median_impact']:.2f} {impact_unit}"
        }
    }

def validate_numeric_input(value: str, field_name: str) -> tuple[bool, float]:
    """
    Validate that a text input contains a valid positive number with engineering context.
    """
    if not value.strip():
        return False, 0.0
    
    try:
        numeric_value = float(value)
        if numeric_value < 0:
            st.error(f"{field_name} must be a positive number")
            return False, 0.0
        elif numeric_value == 0:
            st.warning(f"{field_name} is zero - this will result in no environmental impact")
        return True, numeric_value
    except ValueError:
        st.error(f"{field_name} must be a valid number")
        return False, 0.0

def analyze_data_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the quality and characteristics of the loaded data with engineering insights.
    """
    analysis = {
        "total_counties": len(data["CountyFIPS"]),
        "metrics_analysis": {},
        "data_ranges": {},
        "missing_data": {},
        "statistical_summary": {},
        "engineering_assessment": {}
    }
    
    metrics = ["AWAREUSCF", "EFkgkWh", "EWIF"]
    metric_names = ["Water Scarcity Footprint", "Carbon Footprint", "Water Footprint"]
    metric_units = ["L water-equiv/kWh", "kg CO₂ equiv/kWh", "L water/kWh"]
    
    for i, metric in enumerate(metrics):
        values = data[metric]
        valid_values = values[~np.isnan(values) & (values > 0)]
        
        analysis["metrics_analysis"][metric] = {
            "name": metric_names[i],
            "unit": metric_units[i],
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
                "percentile_10": float(np.percentile(valid_values, 10)),
                "percentile_25": float(np.percentile(valid_values, 25)),
                "percentile_33": float(np.percentile(valid_values, 33)),
                "percentile_66": float(np.percentile(valid_values, 66)),
                "percentile_75": float(np.percentile(valid_values, 75)),
                "percentile_90": float(np.percentile(valid_values, 90)),
                "percentile_95": float(np.percentile(valid_values, 95))
            }
            
            # Engineering assessment of data quality
            coefficient_of_variation = analysis["data_ranges"][metric]["std"] / analysis["data_ranges"][metric]["mean"]
            analysis["engineering_assessment"][metric] = {
                "coefficient_of_variation": coefficient_of_variation,
                "data_spread": "High" if coefficient_of_variation > 1.0 else "Medium" if coefficient_of_variation > 0.5 else "Low",
                "outlier_potential": "High" if analysis["data_ranges"][metric]["max"] > 10 * analysis["data_ranges"][metric]["median"] else "Low"
            }
        else:
            analysis["data_ranges"][metric] = None
            analysis["statistical_summary"][metric] = None
            analysis["engineering_assessment"][metric] = {"error": "No valid data"}
    
    return analysis

# -------------- DATA LOADING --------------
@st.cache_data
def load_data() -> Dict[str, Any]:
    """
    Load the environmental data from the .mat file with comprehensive error handling.
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
        
        # Add metadata
        data["_metadata"] = {
            "file_loaded": datetime.now().isoformat(),
            "data_source": "CountyLevelMetrics.mat",
            "total_counties": len(data["CountyFIPS"]),
            "metrics_available": ["AWAREUSCF", "EFkgkWh", "EWIF"]
        }
        
        return data
    except FileNotFoundError:
        st.error("Data file 'CountyLevelMetrics.mat' not found. Please ensure it is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

async def perform_ai_debug_analysis(debug_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use Claude API to analyze debug data and identify potential issues.
    """
    try:
        # Prepare the debug data for AI analysis
        analysis_prompt = f"""
You are an expert mechanical engineer and data analyst reviewing environmental impact calculations. 

Please analyze the following debug data and identify any potential issues, inconsistencies, or areas for improvement:

DEBUG DATA:
{json.dumps(debug_data, indent=2, default=str)}

Please provide a comprehensive analysis covering:

1. **Unit Conversion Accuracy**: Are the conversion factors correct? Any potential errors?

2. **Data Quality Issues**: What concerns exist with the underlying dataset?

3. **Statistical Method Validation**: Are the percentile-based categorizations appropriate?

4. **Calculation Logic**: Any flaws in the mathematical approaches used?

5. **Potential Sources of Error**: What could be causing incorrect results?

6. **Recommendations**: Specific steps to improve accuracy and reliability.

7. **Engineering Perspective**: From a mechanical engineering standpoint, what seems wrong or suspicious?

Format your response as a JSON object with the following structure:
{{
  "overall_assessment": "Brief summary of findings",
  "unit_conversion_issues": ["list of issues found"],
  "data_quality_concerns": ["list of concerns"],
  "statistical_method_review": ["observations about methods"],
  "calculation_errors": ["potential mathematical errors"],
  "error_sources": ["likely sources of problems"],
  "recommendations": ["specific actionable recommendations"],
  "engineering_insights": ["engineering-specific observations"],
  "confidence_level": "high/medium/low",
  "priority_issues": ["most critical issues to address first"]
}}

Respond ONLY with the JSON object. Do not include any text outside of the JSON structure.
"""

        # Make API call to Claude
        response = await fetch("https://api.anthropic.com/v1/messages", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model: "claude-sonnet-4-20250514",
                max_tokens: 2000,
                messages: [
                    { role: "user", content: analysis_prompt }
                ]
            })
        })
        
        if not response.ok:
            st.error(f"AI analysis failed: HTTP {response.status}")
            return None
        
        data = await response.json()
        response_text = data.content[0].text.strip()
        
        # Clean up the response to ensure it's valid JSON
        response_text = response_text.replace('```json\n', '').replace('\n```', '').strip()
        
        # Parse the JSON response
        ai_analysis = json.loads(response_text)
        
        return ai_analysis
        
    except json.JSONDecodeError as e:
        st.error(f"AI response parsing error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"AI analysis error: {str(e)}")
        return None

def display_ai_analysis(ai_analysis: Dict[str, Any]):
    """
    Display the AI debugging analysis in a user-friendly format.
    """
    st.subheader("🤖 AI Debug Analysis")
    
    # Overall assessment with confidence level
    confidence_color = {
        "high": "green",
        "medium": "orange", 
        "low": "red"
    }.get(ai_analysis.get("confidence_level", "medium"), "blue")
    
    st.markdown(f"""
    **Overall Assessment** (Confidence: :{confidence_color}[{ai_analysis.get('confidence_level', 'medium').upper()}]):
    
    {ai_analysis.get('overall_assessment', 'No assessment available')}
    """)
    
    # Priority Issues (Most Important)
    if ai_analysis.get("priority_issues"):
        st.error("🚨 **PRIORITY ISSUES - Address These First:**")
        for issue in ai_analysis["priority_issues"]:
            st.write(f"• {issue}")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Technical Issues", "📊 Data Quality", "💡 Recommendations", "🔬 Engineering Review"])
    
    with tab1:
        st.subheader("Unit Conversion Issues")
        if ai_analysis.get("unit_conversion_issues"):
            for issue in ai_analysis["unit_conversion_issues"]:
                st.warning(f"⚠️ {issue}")
        else:
            st.success("✅ No unit conversion issues detected")
        
        st.subheader("Calculation Errors")
        if ai_analysis.get("calculation_errors"):
            for error in ai_analysis["calculation_errors"]:
                st.error(f"❌ {error}")
        else:
            st.success("✅ No calculation errors detected")
        
        st.subheader("Statistical Method Review")
        if ai_analysis.get("statistical_method_review"):
            for observation in ai_analysis["statistical_method_review"]:
                st.info(f"📈 {observation}")
    
    with tab2:
        st.subheader("Data Quality Concerns")
        if ai_analysis.get("data_quality_concerns"):
            for concern in ai_analysis["data_quality_concerns"]:
                st.warning(f"⚠️ {concern}")
        else:
            st.success("✅ No major data quality concerns detected")
        
        st.subheader("Potential Error Sources")
        if ai_analysis.get("error_sources"):
            for source in ai_analysis["error_sources"]:
                st.error(f"🎯 {source}")
    
    with tab3:
        st.subheader("Actionable Recommendations")
        if ai_analysis.get("recommendations"):
            for i, recommendation in enumerate(ai_analysis["recommendations"], 1):
                st.success(f"**{i}.** {recommendation}")
        else:
            st.info("No specific recommendations provided")
    
    with tab4:
        st.subheader("Engineering Perspective")
        if ai_analysis.get("engineering_insights"):
            for insight in ai_analysis["engineering_insights"]:
                st.info(f"🔬 {insight}")
        else:
            st.info("No engineering-specific insights provided")

def generate_enhanced_debug_report(debug_data: Dict[str, Any]) -> str:
    """
    Generate a comprehensive debug report with complete impact calculations and code snippets.
    """
    report = f"""
ENHANCED ENVIRONMENTAL IMPACT CALCULATOR - COMPLETE DEBUG REPORT
===============================================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: COMPREHENSIVE ENGINEERING ANALYSIS WITH CODE SNIPPETS
===============================================================

INPUT PARAMETERS
----------------
Selected State: {debug_data.get('state', 'N/A')}
Selected Metric: {debug_data.get('metric', 'N/A')}
Power Input: {debug_data.get('power_input', {}).get('input_value', 'N/A')} {debug_data.get('power_input', {}).get('input_unit', '')}
Water Input: {debug_data.get('water_input', {}).get('input_value', 'N/A')} {debug_data.get('water_input', {}).get('input_unit', '')}
Capacity Factor: {debug_data.get('capacity_factor', 1.0):.1%}

POWER CONSUMPTION CONVERSION - DETAILED WITH CODE
=================================================
"""
    
    if 'power_conversion' in debug_data:
        power = debug_data['power_conversion']
        report += f"""
INPUT PARAMETERS:
  • Input Value: {power['input_value']} {power['input_unit']}
  • Capacity Factor Applied: {power.get('capacity_factor', 1.0):.1%}
  • Conversion Factor: {power['conversion_factor']}

CALCULATION STEPS:
"""
        for step in power['calculation_steps']:
            report += f"  • {step}\n"
        
        # Add code snippet
        report += f"""
CODE SNIPPET - Power Conversion Function:
----------------------------------------
def convert_power_to_kwh_per_year(value: float, unit: str, capacity_factor: float = 1.0):
    if unit == "kW":
        hours_per_year = 8760  # 365.25 * 24
        result = value * hours_per_year * capacity_factor
        # Your calculation: {power['input_value']} * 8760 * {power.get('capacity_factor', 1.0)} = {power['output_value']}
    elif unit == "MW":
        kw_conversion = 1000
        hours_per_year = 8760
        result = value * kw_conversion * hours_per_year * capacity_factor
        # Your calculation: {power['input_value']} * 1000 * 8760 * {power.get('capacity_factor', 1.0)} = {power['output_value']}
    elif unit == "kWh/yr":
        result = value  # Direct conversion
    elif unit == "kWh/mo":
        result = value * 12  # 12 months per year
    return result
"""
        
        # Add engineering notes
        if 'engineering_notes' in power:
            report += f"\nENGINEERING VALIDATION NOTES:\n"
            for note in power['engineering_notes']:
                report += f"  ⚠️  {note}\n"
        
        report += f"\nFINAL POWER RESULT: {power['output_value']:,.2f} {power['output_unit']}\n\n"
    
    report += """
WATER CONSUMPTION CONVERSION - DETAILED WITH CODE
=================================================
"""
    
    if 'water_conversion' in debug_data:
        water = debug_data['water_conversion']
        report += f"""
INPUT PARAMETERS:
  • Input Value: {water['input_value']} {water['input_unit']}
  • Conversion Factor: {water['conversion_factor']}

CALCULATION STEPS:
"""
        for step in water['calculation_steps']:
            report += f"  • {step}\n"
            
        # Add code snippet for water conversion
        report += f"""
CODE SNIPPET - Water Conversion Function:
-----------------------------------------
def convert_water_to_liters_per_year(value: float, unit: str):
    if unit == "L/s":
        seconds_per_year = 31536000  # 365.25 * 24 * 3600
        result = value * seconds_per_year
        # Your calculation: {water['input_value']} * 31,536,000 = {water['output_value']}
    elif unit == "gpm":  # gallons per minute
        minutes_per_year = 525600  # 365.25 * 24 * 60
        liters_per_gallon = 3.78541
        result = value * minutes_per_year * liters_per_gallon
        # Your calculation: {water['input_value']} * 525,600 * 3.78541 = {water['output_value']}
    elif unit == "L/yr":
        result = value  # Direct conversion
    elif unit == "L/mo":
        result = value * 12  # 12 months per year
    return result
"""
        
        # Add engineering notes
        if 'engineering_notes' in water:
            report += f"\nENGINEERING VALIDATION NOTES:\n"
            for note in water['engineering_notes']:
                report += f"  ⚠️  {note}\n"
        
        report += f"\nFINAL WATER RESULT: {water['output_value']:,.2f} {water['output_unit']}\n\n"
    
    # Enhanced facility impact section with code
    if 'environmental_impact' in debug_data:
        impact = debug_data['environmental_impact']
        report += f"""
COMPLETE ENVIRONMENTAL IMPACT CALCULATION WITH CODE
===================================================
Annual Power Consumption: {impact['calculation_details']['power_consumption_kwh']:,.0f} kWh/year
Environmental Metric: {debug_data.get('metric', 'N/A')}

STATISTICAL ANALYSIS OF REGIONAL FACTORS:
  • Counties Analyzed: {impact['calculation_details']['counties_analyzed']:,}
  • Minimum Factor: {impact['impact_statistics']['min_factor']:.8f}
  • Maximum Factor: {impact['impact_statistics']['max_factor']:.8f}
  • Median Factor: {impact['impact_statistics']['median_factor']:.8f}
  • Mean Factor: {impact['impact_statistics']['mean_factor']:.8f}
  • Standard Deviation: {impact['impact_statistics']['std_factor']:.8f}
  • 25th Percentile: {impact['impact_statistics']['percentile_25']:.8f}
  • 75th Percentile: {impact['impact_statistics']['percentile_75']:.8f}

FACILITY ENVIRONMENTAL IMPACT CALCULATION:
  • Primary Calculation: {impact['calculation_details']['calculation']}
  • Result: {impact['facility_impact']['median_impact']:.2f} {impact['impact_unit']}

CODE SNIPPET - Environmental Impact Calculation:
------------------------------------------------
def calculate_environmental_impact(power_kwh_per_year, metric_values, metric_name):
    # Remove invalid values
    valid_values = metric_values[~np.isnan(metric_values) & (metric_values > 0)]
    
    # Calculate statistics
    median_factor = np.median(valid_values)  # Your median: {impact['impact_statistics']['median_factor']:.8f}
    min_factor = np.min(valid_values)        # Your min: {impact['impact_statistics']['min_factor']:.8f}
    max_factor = np.max(valid_values)        # Your max: {impact['impact_statistics']['max_factor']:.8f}
    
    # Calculate facility impact
    median_impact = power_kwh_per_year * median_factor
    min_impact = power_kwh_per_year * min_factor
    max_impact = power_kwh_per_year * max_factor
    
    # Your specific calculation:
    # {impact['calculation_details']['power_consumption_kwh']:,.0f} kWh/year × {impact['impact_statistics']['median_factor']:.8f} = {impact['facility_impact']['median_impact']:.2f}
    
    return median_impact, min_impact, max_impact

IMPACT RANGE ANALYSIS:
  • Best Case (Min Factor): {impact['facility_impact']['min_impact']:.2f} {impact['impact_unit']}
  • Most Likely (Median): {impact['facility_impact']['median_impact']:.2f} {impact['impact_unit']}
  • Worst Case (Max Factor): {impact['facility_impact']['max_impact']:.2f} {impact['impact_unit']}
  • Mean Impact: {impact['facility_impact']['mean_impact']:.2f} {impact['impact_unit']}

FACILITY SCALE ASSESSMENT:
  • Scale Category: {impact['facility_assessment']['category']}
  • Context: {impact['facility_assessment']['context']}
  • Typical Range: {impact['facility_assessment']['typical_range']}
  • Engineering Assessment: {impact['facility_assessment']['engineering_notes']}

INTERPRETATION:
{impact['interpretation']}
"""

        # Add facility categorization code
        report += f"""
CODE SNIPPET - Facility Size Categorization:
--------------------------------------------
def categorize_facility_size(power_kwh_per_year):
    # Your facility: {impact['calculation_details']['power_consumption_kwh']:,.0f} kWh/year
    
    if power_kwh_per_year < 10000:
        return "Residential Scale"           # < 10,000 kWh/year
    elif power_kwh_per_year < 30000:
        return "Large Residential"          # 10,000-30,000 kWh/year
    elif power_kwh_per_year < 100000:
        return "Small Commercial"           # 30,000-100,000 kWh/year
    elif power_kwh_per_year < 1000000:
        return "Large Commercial"           # 100,000-1,000,000 kWh/year
    elif power_kwh_per_year < 10000000:
        return "Industrial Facility"       # 1M-10M kWh/year
    else:
        return "Large Industrial Complex"   # >10M kWh/year
    
    # Your facility categorized as: {impact['facility_assessment']['category']}
"""
    
    # Add data quality analysis with code
    if 'data_analysis' in debug_data:
        analysis = debug_data['data_analysis']
        report += f"""
DATA QUALITY ANALYSIS - COMPREHENSIVE WITH CODE
===============================================
Total Counties in Dataset: {analysis['total_counties']:,}
Data Source: {analysis.get('_metadata', {}).get('data_source', 'Unknown')}
File Loaded: {analysis.get('_metadata', {}).get('file_loaded', 'Unknown')}

CODE SNIPPET - Data Quality Analysis:
-------------------------------------
import scipy.io
import numpy as np

def analyze_data_quality(data):
    # Load data
    metrics = scipy.io.loadmat("CountyLevelMetrics.mat")
    
    # Extract arrays - YOUR DATA:
    AWAREUSCF = metrics["AWAREUSCF"].flatten()  # Water scarcity: {analysis['total_counties']:,} counties
    EFkgkWh = metrics["EFkgkWh"].flatten()      # Carbon footprint: {analysis['total_counties']:,} counties  
    EWIF = metrics["EWIF"].flatten()            # Water footprint: {analysis['total_counties']:,} counties
    CountyFIPS = metrics["CountyFIPS"].flatten() # County codes: {analysis['total_counties']:,} counties
    
    # Data validation
    for metric_name, values in [("AWAREUSCF", AWAREUSCF), ("EFkgkWh", EFkgkWh), ("EWIF", EWIF)]:
        valid_values = values[~np.isnan(values) & (values > 0)]
        validity_rate = (len(valid_values) / len(values)) * 100
        print(f"{{metric_name}}: {{len(valid_values):,}}/{{len(values):,}} valid ({{validity_rate:.1f}}%)")

METRIC QUALITY SUMMARY:
"""
        for metric, info in analysis['metrics_analysis'].items():
            report += f"""
{info['name']} ({metric}) - Units: {info.get('unit', 'Unknown')}:
  • Total Values: {info['total_values']:,}
  • Valid Values: {info['valid_values']:,} ({info['percent_valid']:.1f}%)
  • Invalid Values: {info['invalid_values']:,}
"""
            
            if metric in analysis['data_ranges'] and analysis['data_ranges'][metric]:
                ranges = analysis['data_ranges'][metric]
                report += f"""  • Data Range: {ranges['min']:.8f} to {ranges['max']:.8f}
  • Mean: {ranges['mean']:.8f} ± {ranges['std']:.8f} (std dev)
  • Median: {ranges['median']:.8f}
  • Coefficient of Variation: {(ranges['std']/ranges['mean']):.4f}
"""
                
                if metric in analysis['statistical_summary']:
                    stats = analysis['statistical_summary'][metric]
                    report += f"""  • 10th Percentile: {stats['percentile_10']:.8f}
  • 25th Percentile: {stats['percentile_25']:.8f}
  • 33rd Percentile: {stats['percentile_33']:.8f}
  • 66th Percentile: {stats['percentile_66']:.8f}
  • 75th Percentile: {stats['percentile_75']:.8f}
  • 90th Percentile: {stats['percentile_90']:.8f}
  • 95th Percentile: {stats['percentile_95']:.8f}
"""
                
                if metric in analysis.get('engineering_assessment', {}):
                    eng = analysis['engineering_assessment'][metric]
                    if 'error' not in eng:
                        report += f"""  • Data Spread Assessment: {eng['data_spread']} (CV: {eng['coefficient_of_variation']:.4f})
  • Outlier Potential: {eng['outlier_potential']}
"""
    
    # Map processing details with code
    if 'map_data' in debug_data:
        map_data = debug_data['map_data']
        report += f"""
MAP DATA PROCESSING - DETAILED WITH CODE
========================================
Counties Processed: {map_data.get('counties_processed', 'N/A')}
Counties with Valid Data: {map_data.get('valid_counties', 'N/A')}

CODE SNIPPET - Map Data Processing:
-----------------------------------
import pandas as pd
import numpy as np

def create_environmental_map(data, metric_option):
    # Get metric values
    metric_map = {{
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"], 
        "water scarcity footprint": data["AWAREUSCF"]
    }}
    
    values = metric_map[metric_option]  # Your selected: {debug_data.get('metric', 'N/A')}
    fips = data["CountyFIPS"]
    
    # Create DataFrame
    fips_strings = [str(int(fips_code)).zfill(5) for fips_code in fips]
    df = pd.DataFrame({{"fips": fips_strings, "value": values}})
    
    # Data filtering pipeline - YOUR PROCESSING:
    print(f"Initial dataset: {{len(df)}} counties")           # {map_data.get('counties_processed', 'N/A')}
    
    df = df.dropna()
    print(f"After removing NaN: {{len(df)}} counties")        # {map_data.get('valid_counties', 'N/A')} 
    
    df = df[df["value"] > 0]
    print(f"After removing ≤0: {{len(df)}} counties")         # {map_data.get('valid_counties', 'N/A')}
    
    # Calculate percentile thresholds
    low_percentile = np.percentile(df['value'], 33)           # Your value: {map_data.get('percentile_thresholds', {}).get('low', 'N/A')}
    high_percentile = np.percentile(df['value'], 66)          # Your value: {map_data.get('percentile_thresholds', {}).get('high', 'N/A')}

DATA FILTERING PIPELINE:
"""
        for i, step in enumerate(map_data.get('filtering_steps', []), 1):
            report += f"  {i}. {step}\n"
        
        if 'percentile_thresholds' in map_data:
            thresholds = map_data['percentile_thresholds']
            report += f"""
IMPACT CATEGORY THRESHOLDS (33rd/66th Percentile Method):
  • Low Impact (Bottom 33%): ≤ {thresholds['low']:.8f}
  • Medium Impact (Middle 33%): {thresholds['low']:.8f} to {thresholds['high']:.8f}
  • High Impact (Top 33%): > {thresholds['high']:.8f}

COUNTY DISTRIBUTION BY IMPACT CATEGORY:
  • Low Impact Counties: {map_data.get('low_impact_count', 0):,} ({(map_data.get('low_impact_count', 0) / map_data.get('valid_counties', 1)) * 100:.1f}%)
  • Medium Impact Counties: {map_data.get('medium_impact_count', 0):,} ({(map_data.get('medium_impact_count', 0) / map_data.get('valid_counties', 1)) * 100:.1f}%)
  • High Impact Counties: {map_data.get('high_impact_count', 0):,} ({(map_data.get('high_impact_count', 0) / map_data.get('valid_counties', 1)) * 100:.1f}%)

CODE SNIPPET - Category Assignment:
-----------------------------------
def categorize_value(val, low_percentile, high_percentile):
    if val <= low_percentile:
        return "Low Impact"      # ≤ {thresholds['low']:.8f}
    elif val <= high_percentile:
        return "Medium Impact"   # {thresholds['low']:.8f} to {thresholds['high']:.8f}
    else:
        return "High Impact"     # > {thresholds['high']:.8f}

# Apply to your data:
df["category"] = df["value"].apply(lambda x: categorize_value(x, {thresholds['low']:.8f}, {thresholds['high']:.8f}))
"""

        # Add statistical analysis code
        if 'statistical_summary' in map_data:
            stats = map_data['statistical_summary']
            report += f"""
STATISTICAL ANALYSIS CODE VERIFICATION:
---------------------------------------
import numpy as np

# Your processed data statistics:
valid_values = df["value"].values  # {map_data.get('valid_counties', 'N/A')} counties

# Statistical calculations - VERIFY YOUR RESULTS:
min_value = np.min(valid_values)      # Should be: {stats.get('min', 'N/A')}
max_value = np.max(valid_values)      # Should be: {stats.get('max', 'N/A')}
mean_value = np.mean(valid_values)    # Should be: {stats.get('mean', 'N/A')}
median_value = np.median(valid_values) # Should be: {stats.get('median', 'N/A')}
std_value = np.std(valid_values)      # Should be: {stats.get('std', 'N/A')}
q1_value = np.percentile(valid_values, 25)  # Should be: {stats.get('q1', 'N/A')}
q3_value = np.percentile(valid_values, 75)  # Should be: {stats.get('q3', 'N/A')}
iqr_value = q3_value - q1_value       # Should be: {stats.get('iqr', 'N/A')}
"""
    
    report += f"""
COMPLETE CODE INTEGRATION EXAMPLE
=================================
# Full workflow for reproducing your results:

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd

def main():
    # 1. Load data
    metrics = scipy.io.loadmat("CountyLevelMetrics.mat")
    data = {{
        "AWAREUSCF": metrics["AWAREUSCF"].flatten(),
        "EFkgkWh": metrics["EFkgkWh"].flatten(),
        "EWIF": metrics["EWIF"].flatten(),
        "CountyFIPS": metrics["CountyFIPS"].flatten()
    }}
    
    # 2. Convert power consumption
    power_kwh_per_year = convert_power_to_kwh_per_year({debug_data.get('power_input', {}).get('input_value', 0)}, "{debug_data.get('power_input', {}).get('input_unit', 'kWh/yr')}", {debug_data.get('capacity_factor', 1.0)})
    
    # 3. Get environmental factors
    metric_map = {{
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }}
    environmental_factors = metric_map["{debug_data.get('metric', 'N/A')}"]
    
    # 4. Calculate impact
    valid_factors = environmental_factors[~np.isnan(environmental_factors) & (environmental_factors > 0)]
    median_factor = np.median(valid_factors)
    facility_impact = power_kwh_per_year * median_factor
    
    print(f"Your facility impact: {{facility_impact:.2f}}")
    
    return facility_impact

# Run the calculation
if __name__ == "__main__":
    result = main()
    
RECOMMENDATIONS FOR FURTHER ANALYSIS
====================================
1. VALIDATION CHECKLIST:
   □ Verify input values against actual facility operation data
   □ Check capacity factor assumptions for your equipment type
   □ Validate units are correct (kW vs kWh, L/s vs gpm)
   □ Review county selection matches actual facility location
   □ Compare results with industry benchmarks

2. CODE VERIFICATION:
   □ Run the provided code snippets independently 
   □ Verify statistical calculations match your data
   □ Check conversion factors against engineering references
   □ Validate percentile calculations and thresholds

3. ENGINEERING IMPROVEMENTS:
   □ Consider seasonal variations in consumption patterns
   □ Account for equipment efficiency curves and load factors
   □ Review operational schedules vs. 24/7 assumptions
   □ Assess regional grid mix variations for accuracy

4. DEBUGGING STEPS:
   □ Test with known benchmark facilities
   □ Verify data file integrity and version
   □ Check for regional data filtering if state-specific
   □ Validate county FIPS codes for your area

=================================================================
END OF COMPREHENSIVE DEBUG REPORT WITH COMPLETE CODE ANALYSIS
Generated by Environmental Impact Explorer v3.0 - Enhanced Engineering Edition
File: COMPLETE_ENV_DEBUG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt
=================================================================
    
TECHNICAL SUPPORT:
If you need to validate these calculations or have questions about the methodology,
this report contains all the code snippets and calculation steps needed for 
independent verification by your engineering team.

All functions, formulas, and data processing steps are documented above with
the exact code that produced your results.
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
    st.title("🌍 Enhanced Environmental Impact Explorer")
    st.markdown("*Calculate and visualize comprehensive environmental impacts with detailed engineering analysis*")
    
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
        
        # (3) Enhanced facility information
        st.subheader("Facility Information")
        
        # Power input with capacity factor
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
        
        # Capacity factor for power units (kW, MW)
        if power_unit in ["kW", "MW"]:
            capacity_factor = st.slider(
                "Capacity Factor (%)",
                min_value=10,
                max_value=100,
                value=80,
                step=5,
                help="Operating capacity factor - typical industrial facilities: 70-85%"
            ) / 100.0
        else:
            capacity_factor = 1.0
        
        # Water input
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
        
        # Enhanced debug options
        st.subheader("Debug Options")
        
        debug_col1, debug_col2 = st.columns(2)
        with debug_col1:
            show_debug = st.checkbox("🔍 Show Debug Info", 
                                   help="Display detailed calculation steps and data analysis")
        with debug_col2:
            show_data_quality = st.checkbox("📊 Show Data Quality", 
                                          help="Display data quality analysis and statistics")
        
        show_engineering = st.checkbox("🔧 Engineering Analysis", 
                                     help="Show engineering context and validation")
        
        # PROMINENT DEBUG REPORT DOWNLOAD SECTION
        st.subheader("🚨 DEBUG REPORT DOWNLOAD")
        
        if st.session_state.debug_data:
            st.success("✅ Debug data available for download!")
            
            # Make download button VERY prominent
            if st.button("📥 DOWNLOAD COMPLETE DEBUG REPORT", 
                        use_container_width=True, 
                        type="primary",
                        help="Download comprehensive calculation analysis with code snippets"):
                debug_report = generate_enhanced_debug_report(st.session_state.debug_data)
                st.download_button(
                    label="💾 SAVE DEBUG FILE NOW",
                    data=debug_report,
                    file_name=f"COMPLETE_ENV_DEBUG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    type="primary"
                )
                st.balloons()  # Visual confirmation
        else:
            st.info("ℹ️ Run a calculation first to generate debug data")
        
        # Action buttons
        st.subheader("Actions")
        
        # Create button columns for better layout
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            # About button
            if st.button("ℹ️ About", use_container_width=True):
                st.info("""
                    **Enhanced Environmental Impact Explorer**
                    
                    This application calculates precise environmental impacts using 
                    county-level data with engineering validation and debugging.
                    
                    **Available Metrics:**
                    - **Carbon footprint**: kg CO₂ equivalent per kWh
                    - **Scope 1 & 2 water footprint**: Liters of water per kWh
                    - **Water scarcity footprint**: Liters water-equivalent per kWh
                    
                    **Enhanced Features:**
                    - Capacity factor adjustments for realistic power calculations
                    - Engineering context and facility size assessment
                    - Complete impact calculations with statistical analysis
                    - Comprehensive debugging and validation
                    
                    **How to Use:**
                    1. Select your state and environmental metric
                    2. Enter facility power and water consumption
                    3. Adjust capacity factor if using kW/MW units
                    4. Enable debug options for detailed analysis
                    5. Click "Calculate Impact" for complete results
                """)
        
        with btn_col2:
            # Main calculation button
            calculate_impact = st.button("🧮 Calculate Impact", use_container_width=True, type="primary")
        
        # Download debug report button (full width)
        if st.session_state.debug_data and st.button("📄 Download Enhanced Debug Report", use_container_width=True):
            debug_report = generate_enhanced_debug_report(st.session_state.debug_data)
            st.download_button(
                label="💾 Download Report",
                data=debug_report,
                file_name=f"enhanced_environmental_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Download debug report button (full width)
        if st.session_state.debug_data and st.button("📄 Download Enhanced Debug Report", use_container_width=True):
            debug_report = generate_enhanced_debug_report(st.session_state.debug_data)
            st.download_button(
                label="💾 Download Report",
                data=debug_report,
                file_name=f"enhanced_environmental_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # AI Debug Analysis button
        if st.session_state.debug_data and st.button("🤖 AI Debug Analysis", use_container_width=True):
            with st.spinner("🔍 AI is analyzing your calculations..."):
                # Create a synchronous wrapper for the async function
                try:
                    # Try to get existing event loop, create new one if none exists
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    ai_analysis = loop.run_until_complete(perform_ai_debug_analysis(st.session_state.debug_data))
                    if ai_analysis:
                        st.session_state.ai_analysis = ai_analysis
                    else:
                        st.error("AI analysis failed. Please check your data and try again.")
                except Exception as e:
                    st.error(f"AI analysis error: {str(e)}")
                    # Fallback: provide basic analysis
                    st.session_state.ai_analysis = {
                        "overall_assessment": "AI analysis temporarily unavailable. Please use manual debug features.",
                        "recommendations": ["Check unit conversions manually", "Verify data quality statistics", "Review calculation steps in debug mode"],
                        "confidence_level": "low"
                    }
        
        # Exit button
        if st.button("🚪 Exit", use_container_width=True):
            st.warning("👋 Thank you for using the Enhanced Environmental Impact Explorer!")
            st.balloons()
            st.stop()
    
    # Main content area
    with col2:
        if calculate_impact:
            # Clear previous debug data
            st.session_state.debug_data = {
                "state": state,
                "metric": metric_option,
                "timestamp": datetime.now().isoformat(),
                "data_analysis": data["_quality_analysis"],
                "capacity_factor": capacity_factor
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
            
            if power_valid and water_valid and power_value.strip():
                # Create the plot
                create_environmental_map(data, metric_option, state, show_debug, show_data_quality)
                
                # Calculate complete facility impact
                if power_value.strip():
                    calculate_complete_facility_impact(
                        power_numeric, power_unit, capacity_factor,
                        water_numeric if water_value.strip() else 0, water_unit,
                        metric_option, data, show_debug, show_engineering
                    )
                
                # Display AI analysis if available
                if 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
                    display_ai_analysis(st.session_state.ai_analysis)
            
            elif not power_value.strip():
                st.warning("⚠️ Please enter power consumption to calculate environmental impact.")
        else:
            # Show enhanced instructions when no calculation is displayed
            st.subheader("Welcome to Enhanced Impact Analysis! 🚀")
            st.markdown("""
                **Get Started:**
                1. Select your state and environmental metric on the left
                2. Enter your facility's power consumption (required)
                3. Optionally enter water consumption data
                4. Adjust capacity factor for kW/MW units (typically 70-85% for industrial)
                5. Click "Calculate Impact" for complete environmental analysis
                
                **Enhanced Features:**
                - ✅ **Complete Impact Calculations**: Get actual environmental impact numbers
                - ✅ **Engineering Validation**: Facility size assessment and engineering context
                - ✅ **Capacity Factor Adjustments**: Realistic power consumption modeling
                - ✅ **Statistical Analysis**: Comprehensive regional factor analysis
                - ✅ **Detailed Debug Reports**: Export complete calculation documentation
                - ✅ **AI-Powered Analysis**: Automated issue detection and recommendations
                
                **Engineering Benefits:**
                - Realistic operational assumptions
                - Industry benchmark comparisons
                - Data quality validation
                - Complete calculation transparency
                - Professional documentation
            """)
            
            # Show enhanced example
            st.info("""
                **Example Usage:**
                - Power: `500 kW` with `75%` capacity factor = 3,285,000 kWh/year
                - Results in: Complete carbon footprint calculation with county-specific factors
                - Output: Actual tons CO₂ equivalent per year + engineering assessment
            """)

def create_environmental_map(data: Dict[str, Any], metric_option: str, state: str, show_debug: bool, show_data_quality: bool):
    """
    Create and display the environmental impact map with enhanced debugging.
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
    
    # Enhanced debug tracking
    debug_info = {
        "counties_processed": len(fips),
        "filtering_steps": [],
        "valid_counties": 0,
        "percentile_thresholds": {},
        "low_impact_count": 0,
        "medium_impact_count": 0,
        "high_impact_count": 0,
        "statistical_summary": {}
    }
    
    debug_info["filtering_steps"].append(f"Initial dataset: {len(fips)} counties")
    
    # Create a DataFrame for easier manipulation
    fips_strings = [str(int(fips_code)).zfill(5) for fips_code in fips]
    
    df = pd.DataFrame({
        "fips": fips_strings,
        "value": values
    })
    
    debug_info["filtering_steps"].append(f"After creating DataFrame: {len(df)} rows")
    
    # Enhanced data filtering with statistics
    initial_count = len(df)
    df = df.dropna()
    nan_removed = initial_count - len(df)
    debug_info["filtering_steps"].append(f"After removing NaN values: {len(df)} rows ({nan_removed} NaN values removed)")
    
    zero_negative_count = len(df[df["value"] <= 0])
    df = df[df["value"] > 0]
    debug_info["filtering_steps"].append(f"After removing zero/negative values: {len(df)} rows ({zero_negative_count} zero/negative values removed)")
    debug_info["valid_counties"] = len(df)
    
    if len(df) == 0:
        st.error("No valid data found for the selected metric.")
        return
    
    # Enhanced statistical analysis
    debug_info["statistical_summary"] = {
        "min": float(df['value'].min()),
        "max": float(df['value'].max()),
        "mean": float(df['value'].mean()),
        "median": float(df['value'].median()),
        "std": float(df['value'].std()),
        "q1": float(df['value'].quantile(0.25)),
        "q3": float(df['value'].quantile(0.75)),
        "iqr": float(df['value'].quantile(0.75) - df['value'].quantile(0.25))
    }
    
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
    df["formatted_value"] = df["value"].round(6)  # More precision for display
    
    # Count categories for debug
    debug_info["low_impact_count"] = len(df[df["category"] == "Low Impact"])
    debug_info["medium_impact_count"] = len(df[df["category"] == "Medium Impact"])
    debug_info["high_impact_count"] = len(df[df["category"] == "High Impact"])
    
    # Store enhanced debug info in session state
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
        title=f"{metric_option.title()} by County - Statistical Distribution",
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
    
    # Enhanced statistics display
    st.subheader("📊 Enhanced Statistical Analysis")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric(
            "Low Impact Counties",
            f"{debug_info['low_impact_count']} counties",
            f"≤ {low_percentile:.6f}"
        )
    
    with stat_col2:
        st.metric(
            "Medium Impact Counties",
            f"{debug_info['medium_impact_count']} counties",
            f"{low_percentile:.6f} - {high_percentile:.6f}"
        )
    
    with stat_col3:
        st.metric(
            "High Impact Counties",
            f"{debug_info['high_impact_count']} counties",
            f"> {high_percentile:.6f}"
        )
        
    with stat_col4:
        st.metric(
            "Data Quality",
            f"{(debug_info['valid_counties']/debug_info['counties_processed'])*100:.1f}%",
            f"{debug_info['valid_counties']}/{debug_info['counties_processed']} valid"
        )
    
    # Show enhanced debug information if requested
    if show_debug:
        with st.expander("🔍 Enhanced Debug Information - Map Processing", expanded=True):
            st.subheader("Data Processing Pipeline")
            for i, step in enumerate(debug_info["filtering_steps"], 1):
                st.write(f"{i}. {step}")
            
            st.subheader("Statistical Summary")
            stats = debug_info["statistical_summary"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Value", f"{stats['min']:.6f}")
                st.metric("Q1 (25%)", f"{stats['q1']:.6f}")
            with col2:
                st.metric("Median", f"{stats['median']:.6f}")
                st.metric("Mean", f"{stats['mean']:.6f}")
            with col3:
                st.metric("Max Value", f"{stats['max']:.6f}")
                st.metric("Q3 (75%)", f"{stats['q3']:.6f}")
            
            st.subheader("Percentile Thresholds")
            st.write(f"**33rd Percentile (Low/Medium threshold):** {low_percentile:.8f}")
            st.write(f"**66th Percentile (Medium/High threshold):** {high_percentile:.8f}")
            st.write(f"**Interquartile Range (IQR):** {stats['iqr']:.6f}")
            st.write(f"**Coefficient of Variation:** {(stats['std']/stats['mean']):.3f}")
    
    # Enhanced data quality information
    if show_data_quality:
        with st.expander("📊 Enhanced Data Quality Analysis", expanded=True):
            quality = data["_quality_analysis"]
            
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Counties", f"{quality['total_counties']:,}")
            with col2:
                st.metric("Data Source", quality.get('_metadata', {}).get('data_source', 'Unknown'))
            with col3:
                st.metric("Valid Data Rate", f"{(debug_info['valid_counties']/quality['total_counties'])*100:.1f}%")
            
            st.subheader("Metric-Specific Quality Analysis")
            for metric, info in quality["metrics_analysis"].items():
                if metric_option in info["name"].lower():
                    st.write(f"**{info['name']} ({info.get('unit', 'Unknown units')}):**")
                    
                    qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
                    with qual_col1:
                        st.metric("Valid Values", f"{info['valid_values']:,}")
                    with qual_col2:
                        st.metric("Invalid Values", f"{info['invalid_values']:,}")
                    with qual_col3:
                        st.metric("Validity Rate", f"{info['percent_valid']:.1f}%")
                    with qual_col4:
                        if metric in quality.get('engineering_assessment', {}):
                            eng = quality['engineering_assessment'][metric]
                            if 'data_spread' in eng:
                                st.metric("Data Spread", eng['data_spread'])
                    
                    if metric in quality["data_ranges"] and quality["data_ranges"][metric]:
                        ranges = quality["data_ranges"][metric]
                        st.write("**Full Statistical Summary:**")
                        
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        with stats_col1:
                            st.metric("Minimum", f"{ranges['min']:.6f}")
                            st.metric("10th %ile", f"{quality['statistical_summary'][metric]['percentile_10']:.6f}")
                        with stats_col2:
                            st.metric("25th %ile", f"{quality['statistical_summary'][metric]['percentile_25']:.6f}")
                            st.metric("Median", f"{ranges['median']:.6f}")
                        with stats_col3:
                            st.metric("75th %ile", f"{quality['statistical_summary'][metric]['percentile_75']:.6f}")
                            st.metric("Mean", f"{ranges['mean']:.6f}")
                        with stats_col4:
                            st.metric("90th %ile", f"{quality['statistical_summary'][metric]['percentile_90']:.6f}")
                            st.metric("Maximum", f"{ranges['max']:.6f}")

def calculate_complete_facility_impact(power_value: float, power_unit: str, capacity_factor: float,
                                     water_value: float, water_unit: str, metric_option: str, 
                                     data: Dict[str, Any], show_debug: bool, show_engineering: bool):
    """
    Calculate complete environmental impact with engineering analysis.
    """
    # Convert to standard units with debug info
    power_kwh_per_year, power_debug = convert_power_to_kwh_per_year(power_value, power_unit, capacity_factor)
    
    if water_value > 0:
        water_liters_per_year, water_debug = convert_water_to_liters_per_year(water_value, water_unit)
        st.session_state.debug_data["water_conversion"] = water_debug
    else:
        water_liters_per_year = 0
        water_debug = None
    
    # Calculate actual environmental impact
    metric_map = {
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }
    
    environmental_impact = calculate_environmental_impact(
        power_kwh_per_year, 
        metric_map[metric_option], 
        metric_option
    )
    
    # Store comprehensive debug info
    st.session_state.debug_data["power_conversion"] = power_debug
    st.session_state.debug_data["environmental_impact"] = environmental_impact
    st.session_state.debug_data["facility_impact"] = {
        "annual_power_kwh": power_kwh_per_year,
        "annual_water_liters": water_liters_per_year,
        "capacity_factor_used": capacity_factor
    }
    
    st.subheader("🏭 Complete Facility Environmental Impact Analysis")
    
    # Display facility scale assessment
    facility_assessment = environmental_impact["facility_assessment"]
    
    # Engineering context display
    if show_engineering:
        st.info(f"""
        **🔧 Engineering Assessment:**
        
        **Facility Scale:** {facility_assessment['category']}
        
        **Context:** {facility_assessment['context']}
        
        **Typical Range:** {facility_assessment['typical_range']}
        
        **Engineering Notes:** {facility_assessment['engineering_notes']}
        """)
    
    # Main results display
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        st.metric(
            "Annual Power Consumption",
            f"{power_kwh_per_year:,.0f} kWh/year",
            f"From {power_value} {power_unit} @ {capacity_factor:.0%} CF"
        )
        
        if "error" not in environmental_impact:
            if "carbon" in metric_option.lower():
                st.metric(
                    "Carbon Footprint",
                    f"{environmental_impact['facility_impact']['tons_co2_equiv']:.2f} metric tons CO₂/year",
                    f"{environmental_impact['facility_impact']['median_impact']:.0f} kg CO₂ equiv/year"
                )
            else:
                st.metric(
                    f"{metric_option.title()} Impact",
                    f"{environmental_impact['facility_impact']['median_impact']:,.0f}",
                    environmental_impact['impact_unit']
                )
    
    with impact_col2:
        if water_value > 0:
            st.metric(
                "Annual Water Consumption",
                f"{water_liters_per_year:,.0f} L/year",
                f"From {water_value} {water_unit}"
            )
        else:
            st.metric(
                "Water Data",
                "Not provided",
                "Optional input"
            )
        
        if "error" not in environmental_impact:
            # Show impact range
            st.metric(
                "Impact Range",
                f"{environmental_impact['facility_impact']['min_impact']:.0f} - {environmental_impact['facility_impact']['max_impact']:.0f}",
                f"Based on {environmental_impact['calculation_details']['counties_analyzed']:,} counties"
            )
    
    # Impact interpretation
    if "error" not in environmental_impact:
        st.success(f"**🎯 {environmental_impact['interpretation']}**")
        
        # Detailed calculation display
        with st.expander("📊 Detailed Impact Calculation", expanded=True):
            calc_details = environmental_impact['calculation_details']
            
            st.write("**Calculation Method:**")
            st.code(calc_details['calculation'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Counties Analyzed", f"{calc_details['counties_analyzed']:,}")
            with col2:
                st.metric("Median Factor", f"{calc_details['median_factor']:.6f}")
            with col3:
                st.metric("Power Consumption", f"{calc_details['power_consumption_kwh']:,.0f} kWh/year")
            
            # Impact statistics
            st.subheader("Statistical Analysis of Impact Factors")
            stats = environmental_impact['impact_statistics']
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Minimum Factor", f"{stats['min_factor']:.6f}")
                st.metric("Your Min Impact", f"{environmental_impact['facility_impact']['min_impact']:.0f}")
            with stat_col2:
                st.metric("25th Percentile", f"{stats['percentile_25']:.6f}")
                st.metric("Q1 Impact", f"{power_kwh_per_year * stats['percentile_25']:.0f}")
            with stat_col3:
                st.metric("Median Factor", f"{stats['median_factor']:.6f}")
                st.metric("Median Impact", f"{environmental_impact['facility_impact']['median_impact']:.0f}")
            with stat_col4:
                st.metric("Maximum Factor", f"{stats['max_factor']:.6f}")
                st.metric("Your Max Impact", f"{environmental_impact['facility_impact']['max_impact']:.0f}")
        
        # Benchmarking section
        if show_engineering:
            with st.expander("🏆 Industry Benchmarking", expanded=False):
                st.subheader("Facility Size Comparison")
                
                benchmark_data = []
                for key, benchmark in FACILITY_BENCHMARKS.items():
                    if "carbon" in metric_option.lower():
                        benchmark_impact = benchmark['power_kwh'] * stats['median_factor'] / 1000  # Convert to tons
                        benchmark_data.append({
                            "Category": benchmark['description'],
                            "Power (kWh/year)": f"{benchmark['power_kwh']:,}",
                            "CO₂ Impact (tons/year)": f"{benchmark_impact:.1f}"
                        })
                    else:
                        benchmark_impact = benchmark['power_kwh'] * stats['median_factor']
                        benchmark_data.append({
                            "Category": benchmark['description'],
                            "Power (kWh/year)": f"{benchmark['power_kwh']:,}",
                            f"{metric_option.title()} Impact": f"{benchmark_impact:,.0f}"
                        })
                
                benchmark_df = pd.DataFrame(benchmark_data)
                st.dataframe(benchmark_df, use_container_width=True)
                
                # Your facility comparison
                st.write("**Your Facility vs. Benchmarks:**")
                your_category = facility_assessment['category']
                if "carbon" in metric_option.lower():
                    your_impact = environmental_impact['facility_impact']['tons_co2_equiv']
                    st.write(f"• Your facility ({your_category}): **{your_impact:.2f} metric tons CO₂/year**")
                else:
                    your_impact = environmental_impact['facility_impact']['median_impact']
                    st.write(f"• Your facility ({your_category}): **{your_impact:,.0f} {environmental_impact['impact_unit']}**")
    else:
        st.error(f"❌ {environmental_impact['error']}")
    
    # Enhanced debug information
    if show_debug:
        with st.expander("🔍 Complete Debug Information - Impact Calculations", expanded=True):
            st.subheader("Power Conversion Details")
            st.write(f"**Input:** {power_debug['input_value']} {power_debug['input_unit']}")
            if capacity_factor != 1.0:
                st.write(f"**Capacity Factor Applied:** {capacity_factor:.1%}")
            st.write(f"**Conversion Factor:** {power_debug['conversion_factor']}")
            st.write("**Calculation Steps:**")
            for step in power_debug['calculation_steps']:
                st.write(f"• {step}")
            
            if 'engineering_notes' in power_debug:
                st.write("**Engineering Notes:**")
                for note in power_debug['engineering_notes']:
                    st.warning(f"⚠️ {note}")
            
            st.write(f"**Final Power Result:** {power_debug['output_value']:,.2f} {power_debug['output_unit']}")
            
            if water_debug:
                st.subheader("Water Conversion Details")
                st.write(f"**Input:** {water_debug['input_value']} {water_debug['input_unit']}")
                st.write(f"**Conversion Factor:** {water_debug['conversion_factor']}")
                st.write("**Calculation Steps:**")
                for step in water_debug['calculation_steps']:
                    st.write(f"• {step}")
                
                if 'engineering_notes' in water_debug:
                    st.write("**Engineering Notes:**")
                    for note in water_debug['engineering_notes']:
                        st.warning(f"⚠️ {note}")
                
                st.write(f"**Final Water Result:** {water_debug['output_value']:,.2f} {water_debug['output_unit']}")
            
            if "error" not in environmental_impact:
                st.subheader("Environmental Impact Calculation Details")
                st.json(environmental_impact['calculation_details'])
                
                st.subheader("Facility Assessment Details")
                st.json(facility_assessment)
    
    # Recommendations section
    st.subheader("💡 Engineering Recommendations")
    
    recommendations = []
    
    # Power-related recommendations
    if capacity_factor == 1.0 and power_unit in ["kW", "MW"]:
        recommendations.append("⚠️ **Consider realistic capacity factor**: 100% capacity factor assumes continuous 24/7/365 operation, which is rare in industry.")
    
    if facility_assessment['category'] == "Residential Scale" and power_kwh_per_year < 5000:
        recommendations.append("🔍 **Verify power consumption**: This seems low for an industrial facility. Check if units are correct.")
    
    if power_kwh_per_year > 10000000:  # 10 million kWh/year
        recommendations.append("🏭 **Large facility detected**: Consider detailed load profiling and energy efficiency assessments.")
    
    # Impact-related recommendations
    if "error" not in environmental_impact:
        if environmental_impact['facility_impact']['median_impact'] > environmental_impact['facility_impact']['mean_impact'] * 1.5:
            recommendations.append("📊 **Above-average impact**: Your facility's impact is higher than typical - consider location-specific factors.")
    
    # Water-related recommendations
    if water_value == 0:
        recommendations.append("💧 **Consider adding water data**: Water consumption data would provide more complete environmental impact assessment.")
    
    # Display recommendations
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ **No immediate concerns identified** - Your inputs appear reasonable for the facility type.")
    
    # Final note about calculations
    st.info("""
        **📋 Next Steps:**
        1. Verify your input values against actual facility data
        2. Consider seasonal variations in consumption patterns
        3. Review county-specific factors if you know your exact location
        4. Use the debug report for detailed documentation
        5. Compare results with industry benchmarks for your facility type
    """)

# -------------- RUN THE APP --------------
if __name__ == "__main__":
    main()
