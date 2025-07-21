# app.py - Enhanced Environmental Impact Explorer with Complete Debug Features and Validation
# A comprehensive Streamlit app for calculating and visualizing environmental impacts with detailed debugging

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

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
            "engineering_notes": "Very low consumption - check if this is correct for an industrial analysis",
            "concern_level": "high"
        }
    elif power_kwh_per_year < 30000:
        return {
            "category": "Large Residential",
            "benchmark": "residential_large", 
            "context": "Similar to a large residential home or very small business",
            "typical_range": "10,000-30,000 kWh/year",
            "engineering_notes": "Residential scale - unusual for industrial facility analysis",
            "concern_level": "medium"
        }
    elif power_kwh_per_year < 100000:
        return {
            "category": "Small Commercial",
            "benchmark": "commercial_small",
            "context": "Small office building, retail store, or light manufacturing",
            "typical_range": "30,000-200,000 kWh/year",
            "engineering_notes": "Light commercial load profile",
            "concern_level": "low"
        }
    elif power_kwh_per_year < 1000000:
        return {
            "category": "Large Commercial/Light Industrial",
            "benchmark": "commercial_large",
            "context": "Large commercial building, warehouse, or light industrial facility",
            "typical_range": "100,000-1,000,000 kWh/year", 
            "engineering_notes": "Moderate industrial load - check capacity factor assumptions",
            "concern_level": "none"
        }
    elif power_kwh_per_year < 10000000:
        return {
            "category": "Industrial Facility",
            "benchmark": "industrial_small",
            "context": "Manufacturing plant, processing facility, or heavy industrial operation",
            "typical_range": "1,000,000-50,000,000 kWh/year",
            "engineering_notes": "Industrial scale - verify 24/7 operation assumptions",
            "concern_level": "none"
        }
    else:
        return {
            "category": "Large Industrial Complex",
            "benchmark": "industrial_large",
            "context": "Major manufacturing complex, refinery, or industrial campus",
            "typical_range": ">10,000,000 kWh/year",
            "engineering_notes": "Very large facility - confirm power consumption accuracy",
            "concern_level": "none"
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
        "percentile_75": float(np.percentile(valid_values, 75)),
        "coefficient_of_variation": float(np.std(valid_values) / np.mean(valid_values))
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

def validate_industrial_inputs(debug_data: Dict[str, Any]) -> List[str]:
    """
    Validate inputs for industrial-scale analysis and return warning messages.
    """
    warnings = []
    
    if 'power_conversion' in debug_data:
        annual_power = debug_data['power_conversion']['output_value']
        
        # Check for unrealistically low consumption
        if annual_power < 50000:  # Less than 50 MWh/year
            warnings.append(f"🚨 CRITICAL: Power consumption ({annual_power:,.0f} kWh/year) is very low for industrial analysis")
            warnings.append("   → This is residential/small commercial scale, not industrial")
            warnings.append("   → Typical industrial facilities: 500,000+ kWh/year")
        elif annual_power < 200000:  # Less than 200 MWh/year
            warnings.append(f"⚠️  Power consumption ({annual_power:,.0f} kWh/year) appears to be commercial scale")
            warnings.append("   → Consider if this is correct for industrial environmental analysis")
        
        # Check capacity factor
        if debug_data.get('capacity_factor', 1.0) == 1.0:
            unit = debug_data.get('power_input', {}).get('input_unit', '')
            if unit in ['kW', 'MW']:
                warnings.append("⚠️  100% capacity factor is unrealistic for most industrial operations")
                warnings.append("   → Typical industrial capacity factors: 70-85%")
                warnings.append("   → 100% assumes perfect 24/7/365 operation with no downtime")
    
    # Check facility categorization
    if 'environmental_impact' in debug_data:
        facility_assessment = debug_data['environmental_impact']['facility_assessment']
        category = facility_assessment['category']
        concern_level = facility_assessment.get('concern_level', 'none')
        
        if concern_level == 'high':
            warnings.append(f"🚨 FACILITY SCALE MISMATCH: Categorized as '{category}'")
            warnings.append("   → This is unusual for industrial environmental impact analysis")
            warnings.append("   → Double-check your power consumption values and units")
        elif concern_level == 'medium':
            warnings.append(f"⚠️  Facility categorized as '{category}'")
            warnings.append("   → Verify this is appropriate for your analysis type")
    
    return warnings

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
            
            # Enhanced engineering assessment of data quality
            coefficient_of_variation = analysis["data_ranges"][metric]["std"] / analysis["data_ranges"][metric]["mean"]
            outlier_ratio = analysis["data_ranges"][metric]["max"] / analysis["data_ranges"][metric]["median"]
            
            # Data quality flags
            high_variability = coefficient_of_variation > 2.0
            extreme_outliers = outlier_ratio > 50
            suspicious_max = analysis["data_ranges"][metric]["max"] % 10 == 0 and analysis["data_ranges"][metric]["max"] > 50  # Suspiciously round numbers
            
            analysis["engineering_assessment"][metric] = {
                "coefficient_of_variation": coefficient_of_variation,
                "outlier_ratio": outlier_ratio,
                "data_spread": "Very High" if coefficient_of_variation > 2.0 else "High" if coefficient_of_variation > 1.0 else "Medium" if coefficient_of_variation > 0.5 else "Low",
                "outlier_potential": "Very High" if outlier_ratio > 50 else "High" if outlier_ratio > 10 else "Medium" if outlier_ratio > 5 else "Low",
                "quality_flags": {
                    "high_variability": high_variability,
                    "extreme_outliers": extreme_outliers,
                    "suspicious_max": suspicious_max
                }
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
    Generate a comprehensive debug report with CORRECTED calculations and validation warnings.
    """
    report = f"""
ENHANCED ENVIRONMENTAL IMPACT CALCULATOR - COMPLETE DEBUG REPORT
===============================================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: COMPREHENSIVE ENGINEERING ANALYSIS WITH CODE SNIPPETS AND VALIDATION
===============================================================

INPUT PARAMETERS
----------------
Selected State: {debug_data.get('state', 'N/A')}
Selected Metric: {debug_data.get('metric', 'N/A')}
Power Input: {debug_data.get('power_input', {}).get('input_value', 'N/A')} {debug_data.get('power_input', {}).get('input_unit', '')}
Water Input: {debug_data.get('water_input', {}).get('input_value', 'N/A')} {debug_data.get('water_input', {}).get('input_unit', '')}
Capacity Factor: {debug_data.get('capacity_factor', 1.0):.1%}

🚨 VALIDATION WARNINGS AND RECOMMENDATIONS
==========================================
"""
    
    # Add validation warnings
    warnings = validate_industrial_inputs(debug_data)
    if warnings:
        for warning in warnings:
            report += f"{warning}\n"
        report += "\n"
    else:
        report += "✅ No critical validation issues detected.\n\n"
    
    # Add data quality warnings
    if 'environmental_impact' in debug_data:
        impact = debug_data['environmental_impact']
        if 'coefficient_of_variation' in impact['impact_statistics']:
            cv = impact['impact_statistics']['coefficient_of_variation']
            if cv > 2.0:
                report += f"⚠️  HIGH DATA VARIABILITY WARNING:\n"
                report += f"   • Coefficient of Variation: {cv:.2f}\n"
                report += f"   • This indicates very high variability in the environmental data\n"
                report += f"   • Results may be less reliable than typical\n\n"
    
    report += f"""
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
        
        # CORRECTED code snippet with proper calculations
        report += f"""
CODE SNIPPET - Power Conversion Function (CORRECTED):
---------------------------------------------------
def convert_power_to_kwh_per_year(value: float, unit: str, capacity_factor: float = 1.0):
    if unit == "kW":
        hours_per_year = 8760  # 365.25 * 24
        result = value * hours_per_year * capacity_factor
        # CORRECT calculation: {power['input_value']} * 8760 * {power.get('capacity_factor', 1.0)} = {power['input_value'] * 8760 * power.get('capacity_factor', 1.0):,.0f}
    elif unit == "MW":
        kw_conversion = 1000
        hours_per_year = 8760
        result = value * kw_conversion * hours_per_year * capacity_factor  
        # CORRECT calculation: {power['input_value']} * 1000 * 8760 * {power.get('capacity_factor', 1.0)} = {power['input_value'] * 1000 * 8760 * power.get('capacity_factor', 1.0):,.0f}
    elif unit == "kWh/yr":
        result = value  # Direct conversion
        # ACTUAL result: {power['output_value']:,.0f}
    elif unit == "kWh/mo":
        result = value * 12  # 12 months per year
        # CORRECT calculation: {power['input_value']} * 12 = {power['input_value'] * 12:,.0f}
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
            
        # CORRECTED water conversion code snippet
        report += f"""
CODE SNIPPET - Water Conversion Function (CORRECTED):
----------------------------------------------------
def convert_water_to_liters_per_year(value: float, unit: str):
    if unit == "L/s":
        seconds_per_year = 31536000  # 365.25 * 24 * 3600
        result = value * seconds_per_year
        # CORRECT calculation: {water['input_value']} * 31,536,000 = {water['input_value'] * 31536000:,.0f}
    elif unit == "gpm":  # gallons per minute
        minutes_per_year = 525600  # 365.25 * 24 * 60
        liters_per_gallon = 3.78541
        result = value * minutes_per_year * liters_per_gallon
        # CORRECT calculation: {water['input_value']} * 525,600 * 3.78541 = {water['input_value'] * 525600 * 3.78541:,.0f}
    elif unit == "L/yr":
        result = value  # Direct conversion
        # ACTUAL result: {water['output_value']:,.0f}
    elif unit == "L/mo":
        result = value * 12  # 12 months per year
        # CORRECT calculation: {water['input_value']} * 12 = {water['input_value'] * 12:,.0f}
    return result
"""
        
        # Add engineering notes
        if 'engineering_notes' in water:
            report += f"\nENGINEERING VALIDATION NOTES:\n"
            for note in water['engineering_notes']:
                report += f"  ⚠️  {note}\n"
        
        report += f"\nFINAL WATER RESULT: {water['output_value']:,.2f} {water['output_unit']}\n\n"

    # Continue with the rest of the report...
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
  • Coefficient of Variation: {impact['impact_statistics'].get('coefficient_of_variation', 0):.4f}

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
  • Concern Level: {impact['facility_assessment'].get('concern_level', 'none').upper()}

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
    
    # Add data quality analysis with enhanced warnings
    if 'data_analysis' in debug_data:
        analysis = debug_data['data_analysis']
        report += f"""
DATA QUALITY ANALYSIS - COMPREHENSIVE WITH WARNINGS
===================================================
Total Counties in Dataset: {analysis['total_counties']:,}
Data Source: {analysis.get('_metadata', {}).get('data_source', 'CountyLevelMetrics.mat')}
File Loaded: {analysis.get('_metadata', {}).get('file_loaded', 'Unknown')}

ENHANCED DATA QUALITY WARNINGS:
"""
        for metric, info in analysis['metrics_analysis'].items():
            if metric in analysis.get('engineering_assessment', {}):
                eng = analysis['engineering_assessment'][metric]
                if 'quality_flags' in eng:
                    flags = eng['quality_flags']
                    if flags.get('high_variability', False):
                        report += f"🚨 {info['name']}: Very high variability detected (CV: {eng['coefficient_of_variation']:.2f})\n"
                    if flags.get('extreme_outliers', False):
                        report += f"⚠️  {info['name']}: Extreme outliers detected (ratio: {eng['outlier_ratio']:.1f})\n"
                    if flags.get('suspicious_max', False):
                        report += f"⚠️  {info['name']}: Suspicious maximum value detected\n"

        report += f"""
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
                
                if metric in analysis['engineering_assessment']:
                    eng = analysis['engineering_assessment'][metric]
                    if 'error' not in eng:
                        report += f"""  • Data Spread Assessment: {eng['data_spread']} (CV: {eng['coefficient_of_variation']:.4f})
  • Outlier Potential: {eng['outlier_potential']} (Ratio: {eng['outlier_ratio']:.1f})
"""
    
    report += f"""
COMPLETE CODE INTEGRATION EXAMPLE - CORRECTED VERSION
====================================================
# Full workflow for reproducing your results with validation:

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd

def validate_inputs(power_kwh_per_year, capacity_factor):
    \"\"\"Validate inputs and provide warnings\"\"\"
    warnings = []
    if power_kwh_per_year < 50000:
        warnings.append("WARNING: Very low power consumption for industrial analysis")
    if capacity_factor == 1.0:
        warnings.append("WARNING: 100% capacity factor may be unrealistic")
    return warnings

def main():
    # 1. Load data
    metrics = scipy.io.loadmat("CountyLevelMetrics.mat")
    data = {{
        "AWAREUSCF": metrics["AWAREUSCF"].flatten(),
        "EFkgkWh": metrics["EFkgkWh"].flatten(),
        "EWIF": metrics["EWIF"].flatten(),
        "CountyFIPS": metrics["CountyFIPS"].flatten()
    }}
    
    # 2. Convert power consumption with validation
    power_kwh_per_year = convert_power_to_kwh_per_year(
        {debug_data.get('power_input', {}).get('input_value', 0)}, 
        "{debug_data.get('power_input', {}).get('input_unit', 'kWh/yr')}", 
        {debug_data.get('capacity_factor', 1.0)}
    )
    
    # 3. Validate inputs
    validation_warnings = validate_inputs(power_kwh_per_year, {debug_data.get('capacity_factor', 1.0)})
    for warning in validation_warnings:
        print(warning)
    
    # 4. Get environmental factors
    metric_map = {{
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }}
    environmental_factors = metric_map["{debug_data.get('metric', 'N/A')}"]
    
    # 5. Calculate impact with data quality check
    valid_factors = environmental_factors[~np.isnan(environmental_factors) & (environmental_factors > 0)]
    if len(valid_factors) == 0:
        print("ERROR: No valid environmental data")
        return None
    
    median_factor = np.median(valid_factors)
    cv = np.std(valid_factors) / np.mean(valid_factors)
    
    if cv > 2.0:
        print(f"WARNING: High data variability (CV: {{cv:.2f}})")
    
    facility_impact = power_kwh_per_year * median_factor
    
    print(f"Your facility impact: {{facility_impact:.2f}}")
    print(f"Data quality coefficient of variation: {{cv:.2f}}")
    
    return facility_impact

# Run the calculation with validation
if __name__ == "__main__":
    result = main()
    
ENHANCED RECOMMENDATIONS FOR FURTHER ANALYSIS
============================================
1. CRITICAL INPUT VALIDATION:
   □ Verify power consumption values are appropriate for your facility type
   □ Use realistic capacity factors (70-85% for most industrial operations)
   □ Double-check units (kW vs kWh, MW vs MWh)
   □ Confirm facility scale matches analysis intent

2. DATA QUALITY VERIFICATION:
   □ Check coefficient of variation - values >2.0 indicate poor data quality
   □ Investigate extreme outliers and suspicious maximum values
   □ Consider regional data filtering if available
   □ Validate results against independent sources

3. ENGINEERING VALIDATION:
   □ Compare results with industry benchmarks for your facility type
   □ Consider seasonal variations and operational patterns
   □ Account for equipment efficiency and load profiles
   □ Review capacity factor assumptions against actual operations

4. ENHANCED DEBUGGING:
   □ Run validation functions on all inputs
   □ Check calculation intermediate steps manually
   □ Verify statistical methods are appropriate for your data
   □ Test with known reference cases

=================================================================
END OF ENHANCED DEBUG REPORT WITH VALIDATION AND CORRECTIONS
Generated by Environmental Impact Explorer v4.0 - Professional Engineering Edition
=================================================================
    
TECHNICAL SUPPORT:
This enhanced report includes validation warnings, corrected calculations,
and comprehensive data quality assessments. All mathematical errors in 
previous versions have been corrected, and validation checks have been added
to identify potential issues with inputs and data quality.

All functions, formulas, and validation steps are documented above with
the exact corrected code that should produce accurate results.
"""
    
    return report

def generate_quick_summary(debug_data: Dict[str, Any]) -> str:
    """Generate a concise calculation summary with validation warnings."""
    summary = f"""
ENVIRONMENTAL IMPACT CALCULATION SUMMARY WITH VALIDATION
========================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: QUICK REFERENCE SUMMARY WITH WARNINGS

INPUT PARAMETERS:
-----------------
State Selected: {debug_data.get('state', 'N/A')}
Environmental Metric: {debug_data.get('metric', 'N/A')}
Power Input: {debug_data.get('power_input', {}).get('input_value', 'N/A')} {debug_data.get('power_input', {}).get('input_unit', '')}
Capacity Factor: {debug_data.get('capacity_factor', 1.0):.1%}
"""
    
    # Add validation warnings at the top
    warnings = validate_industrial_inputs(debug_data)
    if warnings:
        summary += f"\n🚨 VALIDATION WARNINGS:\n"
        for warning in warnings:
            summary += f"{warning}\n"
        summary += "\n"
    
    if 'water_input' in debug_data:
        summary += f"Water Input: {debug_data['water_input'].get('input_value', 'N/A')} {debug_data['water_input'].get('input_unit', '')}\n"
    
    summary += "\nCONVERSION RESULTS:\n"
    summary += "-" * 19 + "\n"
    
    if 'power_conversion' in debug_data:
        power = debug_data['power_conversion']
        summary += f"Annual Power Consumption: {power['output_value']:,.0f} {power['output_unit']}\n"
    
    if 'water_conversion' in debug_data:
        water = debug_data['water_conversion']
        summary += f"Annual Water Consumption: {water['output_value']:,.0f} {water['output_unit']}\n"
    
    summary += "\nENVIRONMENTAL IMPACT RESULTS:\n"
    summary += "-" * 29 + "\n"
    
    if 'environmental_impact' in debug_data:
        impact = debug_data['environmental_impact']
        if 'error' not in impact:
            summary += f"Primary Impact: {impact['facility_impact']['median_impact']:.2f} {impact['impact_unit']}\n"
            summary += f"Impact Range: {impact['facility_impact']['min_impact']:.0f} - {impact['facility_impact']['max_impact']:.0f}\n"
            summary += f"Facility Category: {impact['facility_assessment']['category']}\n"
            summary += f"Counties Analyzed: {impact['calculation_details']['counties_analyzed']:,}\n"
            
            # Add data quality indicator
            if 'coefficient_of_variation' in impact['impact_statistics']:
                cv = impact['impact_statistics']['coefficient_of_variation']
                summary += f"Data Quality (CV): {cv:.2f} {'(POOR)' if cv > 2.0 else '(GOOD)' if cv < 0.5 else '(FAIR)'}\n"
            
            summary += f"\nKEY CALCULATION:\n"
            summary += f"{impact['calculation_details']['calculation']}\n"
            
            summary += f"\nINTERPRETATION:\n"
            summary += f"{impact['interpretation']}\n"
    
    summary += f"""
VALIDATION SUMMARY:
------------------
• Input Validation: {'❌ ISSUES FOUND' if warnings else '✅ PASSED'}
• Facility Scale: {debug_data.get('environmental_impact', {}).get('facility_assessment', {}).get('category', 'N/A')}
• Concern Level: {debug_data.get('environmental_impact', {}).get('facility_assessment', {}).get('concern_level', 'unknown').upper()}

For complete validation details and corrected code, see the full debug report.
"""
    
    return summary

def generate_engineering_report(debug_data: Dict[str, Any]) -> str:
    """Generate enhanced engineering-focused analysis report with validation."""
    report = f"""
ENHANCED ENGINEERING ANALYSIS REPORT
====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: PROFESSIONAL ENGINEERING ASSESSMENT WITH VALIDATION

EXECUTIVE SUMMARY:
==================
"""
    
    # Add executive summary with key warnings
    warnings = validate_industrial_inputs(debug_data)
    if warnings:
        report += "🚨 CRITICAL ISSUES IDENTIFIED - IMMEDIATE ATTENTION REQUIRED\n"
        for warning in warnings[:3]:  # Show top 3 warnings
            report += f"{warning}\n"
        report += "\n"
    else:
        report += "✅ No critical validation issues detected. Analysis appears appropriate.\n\n"
    
    report += f"""
FACILITY SCALE ASSESSMENT:
==========================
"""
    
    if 'environmental_impact' in debug_data and 'facility_assessment' in debug_data['environmental_impact']:
        assessment = debug_data['environmental_impact']['facility_assessment']
        report += f"""
Scale Category: {assessment['category']}
Engineering Context: {assessment['context']}
Typical Range: {assessment['typical_range']}
Assessment Notes: {assessment['engineering_notes']}
Concern Level: {assessment.get('concern_level', 'none').upper()}

SCALE APPROPRIATENESS:
"""
        concern_level = assessment.get('concern_level', 'none')
        if concern_level == 'high':
            report += "🚨 CRITICAL: Facility scale appears inappropriate for industrial analysis\n"
        elif concern_level == 'medium':
            report += "⚠️  CAUTION: Verify facility scale is appropriate for intended analysis\n"
        else:
            report += "✅ Facility scale appears appropriate for industrial analysis\n"
    
    report += f"\nPOWER CONSUMPTION ANALYSIS:\n"
    report += "=" * 28 + "\n"
    
    if 'power_conversion' in debug_data:
        power = debug_data['power_conversion']
        report += f"""
Input Power: {power['input_value']} {power['input_unit']}
Capacity Factor Applied: {debug_data.get('capacity_factor', 1.0):.1%}
Annual Power Consumption: {power['output_value']:,.0f} {power['output_unit']}

CAPACITY FACTOR ANALYSIS:
"""
        cf = debug_data.get('capacity_factor', 1.0)
        unit = debug_data.get('power_input', {}).get('input_unit', '')
        
        if cf == 1.0 and unit in ['kW', 'MW']:
            report += "❌ 100% capacity factor is unrealistic - suggests perfect 24/7/365 operation\n"
            report += "   Recommended: 70-85% for most industrial facilities\n"
        elif cf > 0.9:
            report += "⚠️  Very high capacity factor - verify this is realistic for your operation\n"
        elif 0.7 <= cf <= 0.85:
            report += "✅ Capacity factor is within typical industrial range\n"
        else:
            report += f"ℹ️  Capacity factor of {cf:.1%} - verify appropriateness for your application\n"
        
        report += f"\nENGINEERING VALIDATION NOTES:\n"
        for note in power.get('engineering_notes', []):
            report += f"• {note}\n"
        
        report += f"\nCONVERSION METHODOLOGY:\n"
        for step in power.get('calculation_steps', []):
            report += f"• {step}\n"
    
    if 'water_conversion' in debug_data:
        water = debug_data['water_conversion']
        report += f"\nWATER CONSUMPTION ANALYSIS:\n"
        report += "=" * 28 + "\n"
        report += f"""
Input Water: {water['input_value']} {water['input_unit']}
Annual Water Consumption: {water['output_value']:,.0f} {water['output_unit']}

ENGINEERING VALIDATION NOTES:
"""
        for note in water.get('engineering_notes', []):
            report += f"• {note}\n"
    
    if 'environmental_impact' in debug_data and 'error' not in debug_data['environmental_impact']:
        impact = debug_data['environmental_impact']
        report += f"\nENVIRONMENTAL IMPACT ENGINEERING ANALYSIS:\n"
        report += "=" * 40 + "\n"
        
        stats = impact['impact_statistics']
        report += f"""
Statistical Robustness:
• Data Points: {impact['calculation_details']['counties_analyzed']:,} counties
• Coefficient of Variation: {stats.get('coefficient_of_variation', 0):.3f}
• Factor Range: {stats['min_factor']:.6f} to {stats['max_factor']:.6f}
• Median Factor Used: {stats['median_factor']:.6f}

DATA QUALITY ASSESSMENT:
"""
        cv = stats.get('coefficient_of_variation', 0)
        if cv > 2.0:
            report += f"🚨 CRITICAL: Very high data variability (CV: {cv:.2f}) - results may be unreliable\n"
        elif cv > 1.0:
            report += f"⚠️  WARNING: High data variability (CV: {cv:.2f}) - interpret results with caution\n"
        elif cv > 0.5:
            report += f"ℹ️  Moderate data variability (CV: {cv:.2f}) - acceptable for analysis\n"
        else:
            report += f"✅ Low data variability (CV: {cv:.2f}) - good data quality\n"
        
        report += f"""
Impact Assessment:
• Best Case (Min): {impact['facility_impact']['min_impact']:.2f} {impact['impact_unit']}
• Most Likely (Median): {impact['facility_impact']['median_impact']:.2f} {impact['impact_unit']}
• Worst Case (Max): {impact['facility_impact']['max_impact']:.2f} {impact['impact_unit']}
• Mean Impact: {impact['facility_impact']['mean_impact']:.2f} {impact['impact_unit']}

Percentile Analysis:
• 25th Percentile Factor: {stats['percentile_25']:.6f}
• 75th Percentile Factor: {stats['percentile_75']:.6f}
• Interquartile Range: {stats['percentile_75'] - stats['percentile_25']:.6f}
"""
    
    # Add data quality assessment
    if 'data_analysis' in debug_data:
        analysis = debug_data['data_analysis']
        report += f"\nDATA QUALITY ENGINEERING ASSESSMENT:\n"
        report += "=" * 36 + "\n"
        report += f"Total Dataset Size: {analysis['total_counties']:,} counties\n"
        
        for metric, info in analysis['metrics_analysis'].items():
            if debug_data.get('metric', '').lower() in info['name'].lower():
                report += f"""
{info['name']} Quality Assessment:
• Validity Rate: {info['percent_valid']:.1f}%
• Valid Values: {info['valid_values']:,}/{info['total_values']:,}
• Units: {info.get('unit', 'Unknown')}
"""
                
                if metric in analysis.get('engineering_assessment', {}):
                    eng = analysis['engineering_assessment'][metric]
                    if 'error' not in eng:
                        report += f"• Data Spread: {eng['data_spread']} (CV: {eng['coefficient_of_variation']:.4f})\n"
                        report += f"• Outlier Risk: {eng['outlier_potential']} (Ratio: {eng.get('outlier_ratio', 0):.1f})\n"
                        
                        # Add quality flags
                        if 'quality_flags' in eng:
                            flags = eng['quality_flags']
                            if any(flags.values()):
                                report += "• Quality Concerns:\n"
                                if flags.get('high_variability'):
                                    report += "  - High variability detected\n"
                                if flags.get('extreme_outliers'):
                                    report += "  - Extreme outliers present\n"
                                if flags.get('suspicious_max'):
                                    report += "  - Suspicious maximum values\n"
    
    report += f"""
ENGINEERING RECOMMENDATIONS:
============================
"""
    
    recommendations = []
    
    # Check capacity factor
    if debug_data.get('capacity_factor', 1.0) == 1.0:
        recommendations.append("1. CRITICAL: Use realistic capacity factor (<100%) for power calculations")
        recommendations.append("   → Typical industrial: 70-85%, Continuous process: 85-95%")
    
    # Check facility size
    if 'environmental_impact' in debug_data:
        concern_level = debug_data['environmental_impact']['facility_assessment'].get('concern_level', 'none')
        category = debug_data['environmental_impact']['facility_assessment']['category']
        if concern_level == 'high':
            recommendations.append("2. CRITICAL: Verify power consumption - classified as residential scale")
            recommendations.append("   → Check units (kW vs kWh, MW vs MWh)")
            recommendations.append("   → Confirm facility type matches analysis intent")
        elif concern_level == 'medium':
            recommendations.append("2. WARNING: Facility appears to be commercial scale")
            recommendations.append("   → Verify this is appropriate for industrial analysis")
    
    # Check data quality
    if 'environmental_impact' in debug_data:
        cv = debug_data['environmental_impact']['impact_statistics'].get('coefficient_of_variation', 0)
        if cv > 2.0:
            recommendations.append("3. DATA QUALITY: Very high variability in environmental factors")
            recommendations.append("   → Results may be unreliable")
            recommendations.append("   → Consider alternative data sources or methods")
        elif cv > 1.0:
            recommendations.append("3. DATA QUALITY: Moderate variability in environmental factors")
            recommendations.append("   → Interpret results with appropriate uncertainty bounds")
    
    # Add general recommendations
    recommendations.extend([
        "4. VALIDATION: Compare results with industry benchmarks",
        "5. OPERATIONAL: Consider seasonal variations and load profiles",
        "6. ACCURACY: Verify all input values against actual facility data"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        if not rec.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
            report += f"   {rec}\n"
        else:
            report += f"{rec}\n"
    
    report += f"""
VALIDATION CHECKLIST:
====================
IMMEDIATE ACTIONS:
□ Verify power consumption values are correct and appropriate
□ Check capacity factor assumptions for your equipment type
□ Confirm facility scale matches analysis intent
□ Review units are correct (kW vs kWh, L/s vs gpm)

TECHNICAL VALIDATION:
□ Compare results with industry benchmarks for your facility type
□ Validate against independent data sources where possible
□ Check coefficient of variation for data quality assessment
□ Review percentile calculations and thresholds

OPERATIONAL CONSIDERATIONS:
□ Consider seasonal variations in consumption patterns
□ Account for equipment efficiency curves and load factors
□ Review operational schedules vs. continuous operation assumptions
□ Assess regional grid mix variations for accuracy

For complete calculation details and corrected code, see the full debug report.
"""
    
    return report

def extract_code_snippets(debug_data: Dict[str, Any]) -> str:
    """Extract corrected Python code snippets from debug data."""
    code = f'''"""
ENVIRONMENTAL IMPACT CALCULATION CODE SNIPPETS - CORRECTED VERSION
==================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Python code for reproducing the calculations with validation
"""

import numpy as np
import pandas as pd
import scipy.io
from typing import Dict, Any, List

# CONFIGURATION AND CONSTANTS
# ============================

FACILITY_BENCHMARKS = {{
    "residential_small": {{"power_kwh": 5000, "description": "Small residential home"}},
    "residential_large": {{"power_kwh": 15000, "description": "Large residential home"}},
    "commercial_small": {{"power_kwh": 50000, "description": "Small commercial building"}},
    "commercial_large": {{"power_kwh": 200000, "description": "Large commercial building"}},
    "industrial_small": {{"power_kwh": 500000, "description": "Small industrial facility"}},
    "industrial_large": {{"power_kwh": 5000000, "description": "Large industrial facility"}}
}}

# VALIDATION FUNCTIONS
# ====================

def validate_inputs(power_kwh_per_year: float, capacity_factor: float) -> List[str]:
    """Validate inputs and return warning messages."""
    warnings = []
    
    if power_kwh_per_year < 50000:
        warnings.append("CRITICAL: Very low power consumption for industrial analysis")
        warnings.append(f"Input: {{power_kwh_per_year:,.0f}} kWh/year (typical industrial: 500,000+ kWh/year)")
    
    if capacity_factor == 1.0:
        warnings.append("WARNING: 100% capacity factor may be unrealistic")
        warnings.append("Typical industrial: 70-85%, Continuous process: 85-95%")
    
    return warnings

def assess_data_quality(values: np.ndarray) -> Dict[str, Any]:
    """Assess quality of environmental data."""
    valid_values = values[~np.isnan(values) & (values > 0)]
    
    if len(valid_values) == 0:
        return {{"error": "No valid data"}}
    
    mean_val = np.mean(valid_values)
    std_val = np.std(valid_values)
    cv = std_val / mean_val
    
    quality = "POOR" if cv > 2.0 else "FAIR" if cv > 1.0 else "GOOD"
    
    return {{
        "coefficient_of_variation": cv,
        "quality_rating": quality,
        "valid_count": len(valid_values),
        "total_count": len(values)
    }}

# POWER CONVERSION FUNCTION - CORRECTED
# =====================================

def convert_power_to_kwh_per_year(value: float, unit: str, capacity_factor: float = 1.0) -> float:
    """Convert different power units to kWh/year with capacity factor."""
    
    if unit == "kWh/yr":
        return value
    elif unit == "kWh/mo":
        return value * 12  # 12 months per year
    elif unit == "kW":
        hours_per_year = 8760  # 365.25 * 24
        return value * hours_per_year * capacity_factor
    elif unit == "MW":
        kw_conversion = 1000
        hours_per_year = 8760
        return value * kw_conversion * hours_per_year * capacity_factor
    else:
        return 0

# WATER CONVERSION FUNCTION - CORRECTED
# =====================================

def convert_water_to_liters_per_year(value: float, unit: str) -> float:
    """Convert different water units to liters/year."""
    
    if unit == "L/yr":
        return value
    elif unit == "L/mo":
        return value * 12
    elif unit == "L/s":
        seconds_per_year = 31536000  # 365.25 * 24 * 3600
        return value * seconds_per_year
    elif unit == "gpm":  # gallons per minute
        minutes_per_year = 525600  # 365.25 * 24 * 60
        liters_per_gallon = 3.78541
        return value * minutes_per_year * liters_per_gallon
    elif unit == "gal/mo":
        months_per_year = 12
        liters_per_gallon = 3.78541
        return value * months_per_year * liters_per_gallon
    else:
        return 0

# FACILITY CATEGORIZATION FUNCTION
# =================================

def categorize_facility_size(power_kwh_per_year: float) -> Dict[str, str]:
    """Categorize facility size based on annual power consumption."""
    
    if power_kwh_per_year < 10000:
        return {{
            "category": "Residential Scale",
            "concern_level": "high",
            "notes": "Very low for industrial analysis"
        }}
    elif power_kwh_per_year < 30000:
        return {{
            "category": "Large Residential",
            "concern_level": "medium",
            "notes": "Unusual for industrial facility analysis"
        }}
    elif power_kwh_per_year < 100000:
        return {{
            "category": "Small Commercial",
            "concern_level": "low",
            "notes": "Light commercial load profile"
        }}
    elif power_kwh_per_year < 1000000:
        return {{
            "category": "Large Commercial/Light Industrial",
            "concern_level": "none",
            "notes": "Moderate industrial load"
        }}
    elif power_kwh_per_year < 10000000:
        return {{
            "category": "Industrial Facility",
            "concern_level": "none",
            "notes": "Industrial scale"
        }}
    else:
        return {{
            "category": "Large Industrial Complex",
            "concern_level": "none",
            "notes": "Very large facility"
        }}

# ENVIRONMENTAL IMPACT CALCULATION FUNCTION - ENHANCED
# ====================================================

def calculate_environmental_impact(power_kwh_per_year: float, metric_values: np.ndarray) -> Dict[str, Any]:
    """Calculate environmental impact with data quality assessment."""
    
    # Remove invalid values
    valid_values = metric_values[~np.isnan(metric_values) & (metric_values > 0)]
    
    if len(valid_values) == 0:
        return {{"error": "No valid environmental data"}}
    
    # Calculate statistics
    min_factor = np.min(valid_values)
    max_factor = np.max(valid_values)
    median_factor = np.median(valid_values)
    mean_factor = np.mean(valid_values)
    std_factor = np.std(valid_values)
    cv = std_factor / mean_factor
    
    # Calculate facility impact
    min_impact = power_kwh_per_year * min_factor
    max_impact = power_kwh_per_year * max_factor
    median_impact = power_kwh_per_year * median_factor
    mean_impact = power_kwh_per_year * mean_factor
    
    return {{
        'min_impact': min_impact,
        'max_impact': max_impact,
        'median_impact': median_impact,
        'mean_impact': mean_impact,
        'median_factor': median_factor,
        'coefficient_of_variation': cv,
        'data_quality': 'POOR' if cv > 2.0 else 'FAIR' if cv > 1.0 else 'GOOD',
        'counties_analyzed': len(valid_values)
    }}

# MAIN CALCULATION WORKFLOW WITH VALIDATION
# =========================================

def main_calculation_with_validation():
    """Main workflow with comprehensive validation."""
    
    # 1. Load data
    metrics = scipy.io.loadmat("CountyLevelMetrics.mat")
    data = {{
        "AWAREUSCF": metrics["AWAREUSCF"].flatten(),  # Water scarcity footprint
        "EFkgkWh": metrics["EFkgkWh"].flatten(),      # Carbon footprint
        "EWIF": metrics["EWIF"].flatten(),            # Water footprint
        "CountyFIPS": metrics["CountyFIPS"].flatten() # County codes
    }}
    
    # 2. Your specific inputs (replace with actual values)
    power_value = {debug_data.get('power_input', {}).get('input_value', 'YOUR_VALUE')}
    power_unit = "{debug_data.get('power_input', {}).get('input_unit', 'YOUR_UNIT')}"
    capacity_factor = {debug_data.get('capacity_factor', 1.0)}
    metric_selected = "{debug_data.get('metric', 'carbon footprint')}"
    
    # 3. Convert power to standard units
    annual_power_kwh = convert_power_to_kwh_per_year(power_value, power_unit, capacity_factor)
    
    # 4. Validate inputs
    validation_warnings = validate_inputs(annual_power_kwh, capacity_factor)
    if validation_warnings:
        print("VALIDATION WARNINGS:")
        for warning in validation_warnings:
            print(f"  - {{warning}}")
        print()
    
    # 5. Categorize facility
    facility_info = categorize_facility_size(annual_power_kwh)
    print(f"Facility Category: {{facility_info['category']}}")
    if facility_info['concern_level'] != 'none':
        print(f"CONCERN: {{facility_info['notes']}}")
    print()
    
    # 6. Get environmental factors
    metric_map = {{
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }}
    environmental_factors = metric_map[metric_selected]
    
    # 7. Assess data quality
    data_quality = assess_data_quality(environmental_factors)
    if "error" not in data_quality:
        print(f"Data Quality: {{data_quality['quality_rating']}} (CV: {{data_quality['coefficient_of_variation']:.3f}})")
        print(f"Valid Data: {{data_quality['valid_count']:,}}/{{data_quality['total_count']:,}} counties")
        print()
    
    # 8. Calculate impact
    impact_results = calculate_environmental_impact(annual_power_kwh, environmental_factors)
    
    if "error" not in impact_results:
        print("RESULTS:")
        print(f"Annual Power: {{annual_power_kwh:,.0f}} kWh/year")
        print(f"Environmental Impact: {{impact_results['median_impact']:.2f}} (median)")
        print(f"Impact Range: {{impact_results['min_impact']:.0f}} - {{impact_results['max_impact']:.0f}}")
        print(f"Counties Analyzed: {{impact_results['counties_analyzed']:,}}")
        print(f"Data Quality: {{impact_results['data_quality']}}")
        
        return {{
            'annual_power_kwh': annual_power_kwh,
            'impact_results': impact_results,
            'facility_info': facility_info,
            'validation_warnings': validation_warnings,
            'data_quality': data_quality
        }}
    else:
        print("ERROR: No valid environmental data available")
        return None

# EXAMPLE USAGE WITH YOUR DATA
# ============================

if __name__ == "__main__":
    print("Environmental Impact Calculator - Enhanced Version")
    print("=" * 50)
    results = main_calculation_with_validation()
    
    if results:
        print("\\nCalculation completed successfully!")
        print("Check validation warnings and data quality assessment above.")
    else:
        print("\\nCalculation failed - check data file and inputs.")

# DEBUGGING HELPER FUNCTIONS
# ==========================

def debug_conversion(value: float, unit: str, capacity_factor: float = 1.0):
    """Debug power conversion step by step."""
    print(f"Converting {{value}} {{unit}} with capacity factor {{capacity_factor:.1%}}")
    
    if unit == "kW":
        result = value * 8760 * capacity_factor
        print(f"  {{value}} kW × 8,760 hours/year × {{capacity_factor}} = {{result:,.0f}} kWh/year")
    elif unit == "MW":
        result = value * 1000 * 8760 * capacity_factor  
        print(f"  {{value}} MW × 1,000 × 8,760 hours/year × {{capacity_factor}} = {{result:,.0f}} kWh/year")
    elif unit == "kWh/yr":
        result = value
        print(f"  {{value}} kWh/year (direct)")
    else:
        result = 0
        print(f"  Unknown unit: {{unit}}")
    
    return result

def verify_calculation(power_kwh: float, factor: float):
    """Verify environmental impact calculation."""
    result = power_kwh * factor
    print(f"Impact calculation: {{power_kwh:,.0f}} × {{factor:.6f}} = {{result:.2f}}")
    return result

'''
    
    return code

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
    st.markdown("*Calculate and visualize comprehensive environmental impacts with detailed engineering analysis and validation*")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # Status indicator at top of sidebar
        if st.session_state.debug_data:
            # Add validation status
            warnings = validate_industrial_inputs(st.session_state.debug_data)
            if warnings:
                st.error("🚨 Validation Issues")
                st.caption(f"{len(warnings)} warnings found")
            else:
                st.success("🟢 Debug Data Ready")
            debug_size = len(str(st.session_state.debug_data))
            st.caption(f"Data size: {debug_size:,} chars")
        else:
            st.info("🔴 No Debug Data")
            st.caption("Run calculation first")
        
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
        
        # (3) Enhanced facility information with validation hints
        st.subheader("Facility Information")
        st.info("💡 **Industrial facilities typically consume 500,000+ kWh/year**")
        
        # Power input with capacity factor
        power_col1, power_col2 = st.columns([2, 1])
        with power_col1:
            power_value = st.text_input(
                "On-site power consumption:",
                placeholder="e.g., 750000 for industrial",
                help="Enter your facility's power consumption. Industrial facilities typically use 500,000+ kWh/year"
            )
        with power_col2:
            power_unit = st.selectbox(
                "Power unit:",
                ["kWh/yr", "kWh/mo", "kW", "MW"],
                help="Select the unit for power consumption"
            )
        
        # Enhanced capacity factor with recommendations
        if power_unit in ["kW", "MW"]:
            st.markdown("**Capacity Factor Guidelines:**")
            st.markdown("- 🏭 Industrial: 70-85%")
            st.markdown("- ⚡ Continuous Process: 85-95%")
            st.markdown("- 🏢 Commercial: 40-70%")
            
            capacity_factor = st.slider(
                "Capacity Factor (%)",
                min_value=10,
                max_value=100,
                value=80,
                step=5,
                help="Operating capacity factor - 100% assumes perfect 24/7/365 operation (unrealistic for most facilities)"
            ) / 100.0
            
            if capacity_factor == 1.0:
                st.warning("⚠️ 100% capacity factor assumes perfect 24/7/365 operation - this is unrealistic for most facilities!")
        else:
            capacity_factor = 1.0
        
        # Water input
        water_col1, water_col2 = st.columns([2, 1])
        with water_col1:
            water_value = st.text_input(
                "On-site water consumption:",
                placeholder="Enter water consumption",
                help="Enter your facility's water consumption (optional)"
            )
        with water_col2:
            water_unit = st.selectbox(
                "Water unit:",
                ["L/yr", "L/mo", "L/s", "gpm", "gal/mo"],
                help="Select the unit for water consumption"
            )
        
        # Enhanced debug options
        st.subheader("🔍 Analysis Options")
        
        debug_col1, debug_col2 = st.columns(2)
        with debug_col1:
            show_debug = st.checkbox("🔍 Show Debug Info", 
                                   help="Display detailed calculation steps and data analysis")
        with debug_col2:
            show_data_quality = st.checkbox("📊 Show Data Quality", 
                                          help="Display data quality analysis and statistics")
        
        show_engineering = st.checkbox("🔧 Engineering Analysis", 
                                     help="Show engineering context and validation")
        
        # ============================================================================
        # ENHANCED DEBUG REPORT DOWNLOAD SECTION WITH VALIDATION STATUS
        # ============================================================================
        st.subheader("📥 DEBUG REPORTS & CALCULATIONS")
        
        # Status indicator
        if st.session_state.debug_data:
            # Show validation status
            warnings = validate_industrial_inputs(st.session_state.debug_data)
            if warnings:
                st.error(f"⚠️ {len(warnings)} validation issues detected!")
                with st.expander("View Issues", expanded=False):
                    for warning in warnings[:5]:  # Show first 5 warnings
                        st.write(f"• {warning}")
            else:
                st.success("✅ Debug data available - validation passed!")
            
            debug_size = len(str(st.session_state.debug_data))
            st.caption(f"📊 Data captured: {debug_size:,} characters of debug information")
            
            # Main comprehensive report download
            st.markdown("**📋 Complete Debug Report with Validation:**")
            debug_report = generate_enhanced_debug_report(st.session_state.debug_data)
            
            # Calculate file size
            file_size_kb = len(debug_report.encode('utf-8')) / 1024
            
            # Prominent download button
            download_col1, download_col2 = st.columns([3, 1])
            with download_col1:
                st.download_button(
                    label=f"📥 DOWNLOAD ENHANCED DEBUG REPORT ({file_size_kb:.1f} KB)",
                    data=debug_report,
                    file_name=f"ENHANCED_ENV_DEBUG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    type="primary",
                    help="Downloads comprehensive report with validation warnings, corrected calculations, and code snippets"
                )
            with download_col2:
                if st.button("👀 Preview", help="Preview report contents"):
                    st.session_state.show_preview = not st.session_state.get('show_preview', False)
            
            # Preview toggle
            if st.session_state.get('show_preview', False):
                with st.expander("📄 Debug Report Preview", expanded=True):
                    st.text_area(
                        "Report Contents (first 2000 characters):",
                        debug_report[:2000] + "\n\n... [Full report available in download] ...",
                        height=300,
                        disabled=True
                    )
            
            # Additional download options
            st.markdown("**📊 Additional Downloads:**")
            
            # Quick summary and engineering analysis side by side
            add_col1, add_col2 = st.columns(2)
            
            with add_col1:
                quick_summary = generate_quick_summary(st.session_state.debug_data)
                st.download_button(
                    label="📋 Quick Summary",
                    data=quick_summary,
                    file_name=f"calculation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Concise summary with validation warnings and key results"
                )
            
            with add_col2:
                eng_report = generate_engineering_report(st.session_state.debug_data)
                st.download_button(
                    label="🔬 Engineering Analysis",
                    data=eng_report,
                    file_name=f"engineering_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Professional engineering assessment with validation"
                )
            
            # Code-only download
            if st.button("💻 Download Corrected Code", use_container_width=True):
                code_snippets = extract_code_snippets(st.session_state.debug_data)
                st.download_button(
                    label="💾 Save Corrected Code",
                    data=code_snippets,
                    file_name=f"corrected_calculation_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                    mime="text/x-python",
                    use_container_width=True,
                    key="code_download"
                )
            
        else:
            st.info("ℹ️ Run a calculation first to generate debug data")
            st.markdown("""
            **📋 Available Reports After Calculation:**
            - **Enhanced Debug Report**: All calculations, validation warnings, and corrected code
            - **Quick Summary**: Concise results with validation status
            - **Engineering Analysis**: Professional facility assessment
            - **Corrected Code**: Python code with validation functions
            """)
        
        # Action buttons
        st.subheader("Actions")
        
        # Create button columns for better layout
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            # About button
            if st.button("ℹ️ About", use_container_width=True):
                st.info("""
                    **Enhanced Environmental Impact Explorer v4.0**
                    
                    Professional-grade environmental impact calculator with:
                    
                    **🔧 Engineering Validation:**
                    - Input validation for industrial-scale analysis
                    - Capacity factor recommendations
                    - Facility scale assessment
                    
                    **📊 Data Quality Analysis:**
                    - Coefficient of variation calculation
                    - Outlier detection
                    - Data reliability assessment
                    
                    **🔍 Enhanced Debugging:**
                    - Corrected calculation code
                    - Comprehensive validation warnings
                    - Professional engineering reports
                    
                    **Available Metrics:**
                    - **Carbon footprint**: kg CO₂ equivalent per kWh
                    - **Scope 1 & 2 water footprint**: Liters of water per kWh
                    - **Water scarcity footprint**: Liters water-equivalent per kWh
                """)
        
        with btn_col2:
            # Main calculation button
            calculate_impact = st.button("🧮 Calculate Impact", use_container_width=True, type="primary")
        
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
                # Show validation warnings prominently
                warnings = validate_industrial_inputs(st.session_state.debug_data)
                if warnings:
                    st.error("🚨 **VALIDATION WARNINGS DETECTED**")
                    for warning in warnings:
                        st.warning(warning)
                    st.markdown("---")
                
                # Create the plot
                create_environmental_map(data, metric_option, state, show_debug, show_data_quality)
                
                # Calculate complete facility impact
                if power_value.strip():
                    calculate_complete_facility_impact(
                        power_numeric, power_unit, capacity_factor,
                        water_numeric if water_value.strip() else 0, water_unit,
                        metric_option, data, show_debug, show_engineering
                    )
            
            elif not power_value.strip():
                st.warning("⚠️ Please enter power consumption to calculate environmental impact.")
        else:
            # Show enhanced instructions when no calculation is displayed
            st.subheader("Welcome to Enhanced Impact Analysis! 🚀")
            st.markdown("""
                **Professional Environmental Impact Calculator v4.0**
                
                **🎯 Get Started:**
                1. Select your state and environmental metric on the left
                2. Enter your facility's power consumption (**Industrial: 500,000+ kWh/year**)
                3. Set realistic capacity factor (70-85% for most industrial facilities)
                4. Optionally enter water consumption data
                5. Click "Calculate Impact" for complete environmental analysis
                
                **✨ Enhanced Features:**
                - ✅ **Input Validation**: Automatic detection of unrealistic values
                - ✅ **Engineering Context**: Professional facility scale assessment  
                - ✅ **Data Quality Analysis**: Comprehensive statistical validation
                - ✅ **Corrected Calculations**: All mathematical errors fixed
                - ✅ **Professional Reports**: Export-ready documentation
                
                **🔧 Professional Benefits:**
                - Industry-standard capacity factor recommendations
                - Comprehensive validation warnings
                - Data quality assessments with coefficient of variation
                - Professional engineering documentation
                - Corrected code snippets for verification
            """)
            
            # Show enhanced examples with validation context
            st.info("""
                **💡 Realistic Examples:**
                - **Small Industrial**: `500 kW` with `80%` capacity = 3,504,000 kWh/year
                - **Large Industrial**: `2 MW` with `75%` capacity = 13,140,000 kWh/year
                - **Results**: Actual environmental impact + professional validation
            """)
            
            # Add validation hints
            st.success("""
                **🎯 Validation Tips:**
                - Industrial facilities: 500,000+ kWh/year
                - Capacity factors: 70-85% (not 100%)
                - Check units carefully (kW vs kWh)
                - Review all validation warnings in debug reports
            """)

def create_environmental_map(data: Dict[str, Any], metric_option: str, state: str, show_debug: bool, show_data_quality: bool):
    """
    Create and display the environmental impact map with enhanced debugging and validation.
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
    
    # Enhanced debug tracking with data quality assessment
    debug_info = {
        "counties_processed": len(fips),
        "filtering_steps": [],
        "valid_counties": 0,
        "percentile_thresholds": {},
        "low_impact_count": 0,
        "medium_impact_count": 0,
        "high_impact_count": 0,
        "statistical_summary": {},
        "data_quality_flags": {}
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
    
    # Enhanced statistical analysis with quality assessment
    debug_info["statistical_summary"] = {
        "min": float(df['value'].min()),
        "max": float(df['value'].max()),
        "mean": float(df['value'].mean()),
        "median": float(df['value'].median()),
        "std": float(df['value'].std()),
        "q1": float(df['value'].quantile(0.25)),
        "q3": float(df['value'].quantile(0.75)),
        "iqr": float(df['value'].quantile(0.75) - df['value'].quantile(0.25)),
        "coefficient_of_variation": float(df['value'].std() / df['value'].mean())
    }
    
    # Data quality flags
    cv = debug_info["statistical_summary"]["coefficient_of_variation"]
    outlier_ratio = debug_info["statistical_summary"]["max"] / debug_info["statistical_summary"]["median"]
    
    debug_info["data_quality_flags"] = {
        "high_variability": cv > 2.0,
        "extreme_outliers": outlier_ratio > 50,
        "suspicious_max": debug_info["statistical_summary"]["max"] % 10 == 0 and debug_info["statistical_summary"]["max"] > 50,
        "quality_rating": "POOR" if cv > 2.0 else "FAIR" if cv > 1.0 else "GOOD"
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
    
    # Show data quality warnings prominently
    if debug_info["data_quality_flags"]["quality_rating"] != "GOOD":
        quality_rating = debug_info["data_quality_flags"]["quality_rating"]
        st.warning(f"⚠️ **Data Quality: {quality_rating}** (CV: {cv:.2f}) - Results may be less reliable")
    
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
    
    # Enhanced statistics display with data quality indicators
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
        quality_color = "🟢" if cv < 1.0 else "🟡" if cv < 2.0 else "🔴"
        st.metric(
            "Data Quality",
            f"{quality_color} {debug_info['data_quality_flags']['quality_rating']}",
            f"CV: {cv:.2f}"
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
            
            st.subheader("Data Quality Assessment")
            flags = debug_info["data_quality_flags"]
            if flags["high_variability"]:
                st.error(f"❌ High variability detected (CV: {cv:.3f})")
            if flags["extreme_outliers"]:
                st.warning(f"⚠️ Extreme outliers detected (ratio: {outlier_ratio:.1f})")
            if flags["suspicious_max"]:
                st.warning("⚠️ Suspicious maximum values detected")
            
            st.subheader("Percentile Thresholds")
            st.write(f"**33rd Percentile (Low/Medium threshold):** {low_percentile:.8f}")
            st.write(f"**66th Percentile (Medium/High threshold):** {high_percentile:.8f}")
            st.write(f"**Interquartile Range (IQR):** {stats['iqr']:.6f}")
            st.write(f"**Coefficient of Variation:** {cv:.3f} ({flags['quality_rating']})")
    
    # Enhanced data quality information
    if show_data_quality:
        with st.expander("📊 Comprehensive Data Quality Analysis", expanded=True):
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
    Calculate complete environmental impact with enhanced validation and engineering analysis.
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
    
    # Display facility scale assessment with validation warnings
    facility_assessment = environmental_impact["facility_assessment"]
    concern_level = facility_assessment.get("concern_level", "none")
    
    # Show validation warnings prominently
    if concern_level == "high":
        st.error(f"🚨 **CRITICAL FACILITY SCALE ISSUE**: {facility_assessment['category']}")
        st.error(f"⚠️ {facility_assessment['engineering_notes']}")
    elif concern_level == "medium":
        st.warning(f"⚠️ **Facility Scale Note**: {facility_assessment['category']}")
        st.warning(f"ℹ️ {facility_assessment['engineering_notes']}")
    
    # Engineering context display
    if show_engineering:
        st.info(f"""
        **🔧 Engineering Assessment:**
        
        **Facility Scale:** {facility_assessment['category']}
        
        **Context:** {facility_assessment['context']}
        
        **Typical Range:** {facility_assessment['typical_range']}
        
        **Engineering Notes:** {facility_assessment['engineering_notes']}
        
        **Validation Status:** {concern_level.upper()}
        """)
    
    # Main results display with validation context
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        # Add validation context to power display
        power_status = "🟢" if power_kwh_per_year >= 500000 else "🟡" if power_kwh_per_year >= 50000 else "🔴"
        
        st.metric(
            "Annual Power Consumption",
            f"{power_status} {power_kwh_per_year:,.0f} kWh/year",
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
            # Add data quality indicator to impact range
            cv = environmental_impact['impact_statistics'].get('coefficient_of_variation', 0)
            quality_indicator = "🟢" if cv < 1.0 else "🟡" if cv < 2.0 else "🔴"
            
            st.metric(
                "Impact Range",
                f"{quality_indicator} {environmental_impact['facility_impact']['min_impact']:.0f} - {environmental_impact['facility_impact']['max_impact']:.0f}",
                f"Based on {environmental_impact['calculation_details']['counties_analyzed']:,} counties"
            )
    
    # Impact interpretation with validation context
    if "error" not in environmental_impact:
        st.success(f"**🎯 {environmental_impact['interpretation']}**")
        
        # Add data quality warning if needed
        cv = environmental_impact['impact_statistics'].get('coefficient_of_variation', 0)
        if cv > 2.0:
            st.error(f"⚠️ **Data Quality Warning**: High variability detected (CV: {cv:.2f}) - results may be unreliable")
        elif cv > 1.0:
            st.warning(f"ℹ️ **Data Quality Note**: Moderate variability detected (CV: {cv:.2f}) - interpret with caution")
        
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
                st.metric("Data Quality (CV)", f"{cv:.3f}")
            
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
    
    # Enhanced recommendations section with validation context
    st.subheader("💡 Professional Engineering Recommendations")
    
    recommendations = []
    
    # Power-related recommendations with validation
    if capacity_factor == 1.0 and power_unit in ["kW", "MW"]:
        recommendations.append("🚨 **CRITICAL: Use realistic capacity factor**: 100% assumes perfect 24/7/365 operation - unrealistic for most facilities")
        recommendations.append("   → Recommended: 70-85% for industrial, 85-95% for continuous processes")
    
    if concern_level == "high":
        recommendations.append("🚨 **CRITICAL: Verify facility scale**: Power consumption appears too low for industrial analysis")
        recommendations.append("   → Check units (kW vs kWh, MW vs MWh)")
        recommendations.append("   → Typical industrial facilities: 500,000+ kWh/year")
    elif concern_level == "medium":
        recommendations.append("⚠️ **Review facility classification**: Currently classified as residential/commercial scale")
    
    if power_kwh_per_year > 10000000:  # 10 million kWh/year
        recommendations.append("🏭 **Large facility detected**: Consider detailed load profiling and energy efficiency assessments")
    
    # Data quality recommendations
    if "error" not in environmental_impact:
        cv = environmental_impact['impact_statistics'].get('coefficient_of_variation', 0)
        if cv > 2.0:
            recommendations.append("⚠️ **Data Quality Concern**: Very high variability in environmental data (CV > 2.0)")
            recommendations.append("   → Results may be unreliable - consider alternative data sources")
        elif cv > 1.0:
            recommendations.append("ℹ️ **Data Quality Note**: Moderate variability in environmental data")
            recommendations.append("   → Interpret results with appropriate uncertainty bounds")
    
    # Water-related recommendations
    if water_value == 0:
        recommendations.append("💧 **Consider adding water data**: Water consumption data would provide more complete environmental impact assessment")
    
    # Display recommendations
    if recommendations:
        for rec in recommendations:
            if rec.startswith("🚨"):
                st.error(rec)
            elif rec.startswith("⚠️"):
                st.warning(rec)
            elif rec.startswith("   →"):
                st.info(rec)
            else:
                st.info(rec)
    else:
        st.success("✅ **No critical concerns identified** - Your inputs appear reasonable for the facility type.")
    
    # Final note about calculations with validation context
    st.info("""
        **📋 Enhanced Analysis Summary:**
        1. **Validation Status**: Check warnings above for input validation issues
        2. **Data Quality**: Review coefficient of variation for reliability assessment  
        3. **Facility Scale**: Confirm power consumption is appropriate for analysis type
        4. **Capacity Factor**: Verify realistic operational assumptions (70-85% typical)
        5. **Documentation**: Use enhanced debug reports for complete technical validation
    """)

# -------------- RUN THE APP --------------
if __name__ == "__main__":
    main()
