import streamlit as st
import numpy as np
import math
import openai
import pandas as pd
import altair as alt
import base64
from datetime import datetime, time, timedelta # Added time and timedelta

# Optional imports - Bayesian functionality
try:
    import scipy.optimize as optimize
    from scipy.stats import norm
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scipy not installed. Bayesian estimation will not be available.")

# Optional imports - FAISS for guideline embedding (if needed)
try:
    import faiss
    import pickle
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Guideline embeddings will not be available.")

st.set_page_config(page_title="Antimicrobial TDM App", layout="wide")

# ===== API CONFIGURATION =====
# Securely access the API key from streamlit secrets
try:
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.

    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== LOAD EMBEDDED GUIDELINE =====
@st.cache_resource
def load_guideline_embeddings():
    if not FAISS_AVAILABLE:
        return None, ["FAISS not installed. Guideline embeddings not available."]

    # Replace this with your actual FAISS index and guideline chunks
    index = faiss.IndexFlatL2(768)  # Dummy placeholder index
    chunks = ["Guideline excerpt 1", "Guideline excerpt 2", "Guideline excerpt 3"]
    return index, chunks

guideline_index, guideline_chunks = None, []
if FAISS_AVAILABLE:
    guideline_index, guideline_chunks = load_guideline_embeddings()

# ===== Global Practical Dosing Regimens =====
practical_intervals = "6hr, 8hr, 12hr, 24hr, every other day"

# ===== HELPER FUNCTION: Updated Practical Dose Adjustment =====
def suggest_adjustment(parameter, target_min, target_max, label="Parameter", intervals=practical_intervals):
    # Ensure targets are valid numbers before comparison
    if isinstance(target_min, (int, float)) and isinstance(target_max, (int, float)) and isinstance(parameter, (int, float)):
        if parameter < target_min:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is low. Target: {target_min:.1f}-{target_max:.1f}. Consider increasing dose or shortening interval ({intervals}).")
        elif parameter > target_max:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is high. Target: {target_min:.1f}-{target_max:.1f}. Consider reducing dose or lengthening interval ({intervals}).")
        else:
            st.success(f"✅ {label} ({parameter:.1f}) is within target range ({target_min:.1f}-{target_max:.1f}).")
    else:
        st.info(f"{label}: {parameter}. Target range: {target_min}-{target_max}. (Comparison skipped due to non-numeric values).")


# ===== PDF GENERATION FUNCTIONS (REMOVED) =====
# create_recommendation_pdf, get_pdf_download_link, display_pdf_download_button functions removed.

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization

    Parameters:
    - peak: Peak concentration (mg/L)
    - trough: Trough concentration (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - t_peak: Time to peak after start of infusion (hr) - assumed end of infusion
    - infusion_time: Duration of infusion (hr)

    Returns:
    - Altair chart object
    """
    # Generate time points for the curve
    times = np.linspace(0, tau*1.5, 150)  # Generate points for 1.5 intervals to show next dose

    # Generate concentrations for each time point using steady-state equations
    concentrations = []
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * T_inf)) * exp(-ke * (t - T_inf)) / (1 - exp(-ke * tau)) -- Post-infusion
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * t)) / (1 - exp(-ke * tau)) -- During infusion (simplified, assumes Cmin=0 start)
    # Let's use the provided peak and trough which represent Cmax (at t=infusion_time) and Cmin (at t=tau)

    for t_cycle in np.linspace(0, tau*1.5, 150): # Iterate through time points
        # Determine concentration based on time within the dosing cycle (modulo tau)
        t = t_cycle % tau
        num_cycles = int(t_cycle // tau) # Which cycle we are in (0, 1, ...)

        conc = 0
        if t <= infusion_time:
            # During infusion: Assume linear rise from previous trough to current peak
            # This is an approximation but visually represents the infusion period
            conc = trough + (peak - trough) * (t / infusion_time)
        else:
            # After infusion: Exponential decay from peak
            time_since_peak = t - infusion_time # Time elapsed since the peak concentration (end of infusion)
            conc = peak * math.exp(-ke * time_since_peak)

        concentrations.append(max(0, conc)) # Ensure concentration doesn't go below 0


    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Time (hr)': times,
        'Concentration (mg/L)': concentrations
    })

    # --- Create Target Bands ---
    target_bands = []
    # Determine drug type based on typical levels for band coloring
    if peak > 45 or trough > 20:  # Likely vancomycin
        # Vancomycin Peak Target - Empiric vs Definitive
        if trough <= 15:  # Likely empiric (target trough 10-15)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [20], 'y2': [30]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Empiric)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [10], 'y2': [15]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Empiric)")))
        else:  # Likely definitive (target trough 15-20)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [25], 'y2': [40]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Definitive)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [15], 'y2': [20]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Definitive)")))
    else:  # Likely aminoglycoside (e.g., Gentamicin)
        # Aminoglycoside Peak Target (e.g., 5-10 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [5], 'y2': [10]}))
                           .mark_rect(opacity=0.15, color='lightblue')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Amino)")))
        # Aminoglycoside Trough Target (e.g., <2 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [0], 'y2': [2]}))
                           .mark_rect(opacity=0.15, color='lightgreen')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Amino)")))


    # --- Create Concentration Line ---
    line = alt.Chart(df).mark_line(color='firebrick').encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)', scale=alt.Scale(zero=True)), # Ensure Y axis starts at 0
        tooltip=['Time (hr)', alt.Tooltip('Concentration (mg/L)', format=".1f")]
    )

    # --- Add Vertical Lines for Key Events ---
    vertical_lines_data = []
    # Mark end of infusion for each cycle shown
    for i in range(int(tau*1.5 / tau) + 1):
        inf_end_time = i * tau + infusion_time
        if inf_end_time <= tau*1.5:
             vertical_lines_data.append({'Time': inf_end_time, 'Event': 'Infusion End'})
    # Mark start of next dose for each cycle shown
    for i in range(1, int(tau*1.5 / tau) + 1):
         dose_time = i * tau
         if dose_time <= tau*1.5:
              vertical_lines_data.append({'Time': dose_time, 'Event': 'Next Dose'})

    vertical_lines_df = pd.DataFrame(vertical_lines_data)

    vertical_rules = alt.Chart(vertical_lines_df).mark_rule(strokeDash=[4, 4]).encode(
        x='Time',
        color=alt.Color('Event', scale=alt.Scale(domain=['Infusion End', 'Next Dose'], range=['gray', 'black'])),
        tooltip=['Event', 'Time']
    )

    # --- Combine Charts ---
    chart = alt.layer(*target_bands, line, vertical_rules).properties(
        width=alt.Step(4), # Adjust width automatically
        height=400,
        title=f'Estimated Concentration-Time Profile (Tau={tau} hr)'
    ).interactive() # Make chart interactive (zoom/pan)

    return chart


# ===== VANCOMYCIN AUC CALCULATION (TRAPEZOIDAL METHOD) =====
def calculate_vancomycin_auc_trapezoidal(cmax, cmin, ke, tau, infusion_duration):
    """
    Calculate vancomycin AUC24 using the linear-log trapezoidal method.
    
    This method is recommended for vancomycin TDM as per the guidelines.
    
    Parameters:
    - cmax: Max concentration at end of infusion (mg/L)
    - cmin: Min concentration at end of interval (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - infusion_duration: Duration of infusion (hr)
    
    Returns:
    - AUC24: 24-hour area under the curve (mg·hr/L)
    """
    # Calculate concentration at start of infusion (C0)
    c0 = cmax * math.exp(ke * infusion_duration)
    
    # Calculate AUC during infusion phase (linear trapezoid)
    auc_inf = infusion_duration * (c0 + cmax) / 2
    
    # Calculate AUC during elimination phase (log trapezoid)
    if ke > 0 and cmax > cmin:
        auc_elim = (cmax - cmin) / ke
    else:
        # Fallback to linear trapezoid if ke is very small
        auc_elim = (tau - infusion_duration) * (cmax + cmin) / 2
    
    # Calculate total AUC for one dosing interval
    auc_interval = auc_inf + auc_elim
    
    # Convert to AUC24
    auc24 = auc_interval * (24 / tau)
    
    return auc24

# ===== BAYESIAN PARAMETER ESTIMATION =====
def bayesian_parameter_estimation(measured_levels, sample_times, dose, tau, weight, age, crcl, gender):
    """
    Bayesian estimation of PK parameters based on measured levels

    Parameters:
    - measured_levels: List of measured drug concentrations (mg/L)
    - sample_times: List of times when samples were taken (hr after dose start)
    - dose: Dose administered (mg)
    - tau: Dosing interval (hr)
    - weight: Patient weight (kg)
    - age: Patient age (years)
    - crcl: Creatinine clearance (mL/min)
    - gender: Patient gender ("Male" or "Female")

    Returns:
    - Dictionary with estimated PK parameters
    """
    if not BAYESIAN_AVAILABLE:
        st.error("Bayesian estimation requires scipy. Please install it with 'pip install scipy'")
        return None

    # Prior population parameters for vancomycin (adjust if needed for aminoglycosides)
    # Mean values
    vd_pop_mean = 0.7  # L/kg (Vancomycin specific, adjust for aminoglycosides if used)
    ke_pop_mean = 0.00083 * crcl + 0.0044 # hr^-1 (Vancomycin specific - ensure crcl is used correctly)
    ke_pop_mean = max(0.01, ke_pop_mean) # Ensure Ke isn't too low

    # Standard deviations for population parameters
    vd_pop_sd = 0.2  # L/kg
    ke_pop_sd = 0.05 # Increased SD for Ke prior to allow more flexibility

    # Define objective function to minimize (negative log likelihood)
    def objective_function(params):
        vd_ind, ke_ind = params # Individual parameters to estimate
        vd_total = vd_ind * weight

        # Calculate expected concentrations at sample times using steady-state infusion model
        expected_concs = []
        infusion_time = 1.0 # Assume 1 hour infusion, make adjustable if needed

        for t in sample_times:
            # Steady State Concentration Equation (1-compartment, intermittent infusion)
            term_dose_vd = dose / vd_total
            term_ke_tinf = ke_ind * infusion_time
            term_ke_tau = ke_ind * tau

            try:
                exp_ke_tinf = math.exp(-term_ke_tinf)
                exp_ke_tau = math.exp(-term_ke_tau)

                if abs(1.0 - exp_ke_tau) < 1e-9: # Avoid division by zero if tau is very long or ke very small
                    # Handle as if continuous infusion or single dose if tau is effectively infinite
                    conc = 0 # Simplified - needs better handling for edge cases
                else:
                    common_factor = (term_dose_vd / term_ke_tinf) * (1.0 - exp_ke_tinf) / (1.0 - exp_ke_tau)

                    if t <= infusion_time: # During infusion phase
                        conc = common_factor * (1.0 - math.exp(-ke_ind * t))
                    else: # Post-infusion phase
                        conc = common_factor * math.exp(-ke_ind * (t - infusion_time))

            except OverflowError:
                 conc = float('inf') # Handle potential overflow with large ke/t values
            except ValueError:
                 conc = 0 # Handle math domain errors

            expected_concs.append(max(0, conc)) # Ensure non-negative

        # Calculate negative log likelihood
        # Measurement error model (e.g., proportional + additive)
        # sd = sqrt(sigma_add^2 + (sigma_prop * expected_conc)^2)
        sigma_add = 1.0  # Additive SD (mg/L)
        sigma_prop = 0.1 # Proportional SD (10%)
        nll = 0
        for i in range(len(measured_levels)):
            expected = expected_concs[i]
            measurement_sd = math.sqrt(sigma_add**2 + (sigma_prop * expected)**2)
            if measurement_sd < 1e-6: measurement_sd = 1e-6 # Prevent division by zero in logpdf

            # Add contribution from measurement likelihood
            # Use logpdf for robustness, especially with low concentrations
            nll += -norm.logpdf(measured_levels[i], loc=expected, scale=measurement_sd)

        # Add contribution from parameter priors (log scale often more stable for Ke)
        # Prior for Vd (Normal)
        nll += -norm.logpdf(vd_ind, loc=vd_pop_mean, scale=vd_pop_sd)
        # Prior for Ke (Log-Normal might be better, but using Normal for simplicity)
        nll += -norm.logpdf(ke_ind, loc=ke_pop_mean, scale=ke_pop_sd)

        # Penalize non-physical parameters slightly if optimization strays
        if vd_ind <= 0 or ke_ind <= 0:
             nll += 1e6 # Add large penalty

        return nll

    # Initial guess based on population values
    initial_params = [vd_pop_mean, ke_pop_mean]

    # Parameter bounds (physical constraints)
    bounds = [(0.1, 2.5), (0.001, 0.5)]  # Reasonable bounds for Vd (L/kg) and Ke (hr^-1)

    # Perform optimization using a robust method
    try:
        result = optimize.minimize(
            objective_function,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B', # Suitable for bound constraints
            options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 500} # Adjust tolerances/iterations
        )
    except Exception as e:
         st.error(f"Optimization failed: {e}")
         return None

    if not result.success:
        st.warning(f"Bayesian optimization did not converge: {result.message} (Function evaluations: {result.nfev})")
        # Optionally return population estimates or None
        return None # Indicate failure

    # Extract optimized parameters
    vd_opt_kg, ke_opt = result.x
    # Ensure parameters are within bounds post-optimization (should be handled by L-BFGS-B, but double-check)
    vd_opt_kg = max(bounds[0][0], min(bounds[0][1], vd_opt_kg))
    ke_opt = max(bounds[1][0], min(bounds[1][1], ke_opt))

    vd_total_opt = vd_opt_kg * weight
    cl_opt = ke_opt * vd_total_opt
    t_half_opt = 0.693 / ke_opt if ke_opt > 0 else float('inf')

    return {
        'vd': vd_opt_kg, # Vd per kg
        'vd_total': vd_total_opt, # Total Vd in L
        'ke': ke_opt,
        'cl': cl_opt,
        't_half': t_half_opt,
        'optimization_success': result.success,
        'final_nll': result.fun # Final negative log-likelihood value
    }


# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM.
    Uses OpenAI API if available, otherwise provides a simulated response.

    Parameters:
    - prompt: The clinical data prompt including calculated values and context.
    - patient_data: Dictionary with patient information (used for context).
    """
    # Extract the drug type from the prompt for context
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
    else:
        drug = "Antimicrobial"

    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Include patient context if relevant (e.g., renal function).
            Format your response with these exact sections:

            ## CLINICAL ASSESSMENT
            📊 **MEASURED/ESTIMATED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **ASSESSMENT:** (state if appropriately dosed, underdosed, or overdosed based on levels and targets)

            ## RECOMMENDATIONS
            🔵 **DOSING:** (action-oriented recommendations using verbs like CONTINUE, ADJUST, HOLD, INCREASE, DECREASE. Suggest practical regimens where possible.)
            🔵 **MONITORING:** (specific monitoring parameters and schedule, e.g., recheck levels, renal function)
            ⚠️ **CAUTIONS:** (relevant warnings, e.g., toxicity risk, renal impairment)

            Here is the case:
            --- Patient Context ---
            Age: {patient_data.get('age', 'N/A')} years, Gender: {patient_data.get('gender', 'N/A')}
            Weight: {patient_data.get('weight', 'N/A')} kg, Height: {patient_data.get('height', 'N/A')} cm
            Patient ID: {patient_data.get('patient_id', 'N/A')}, Ward: {patient_data.get('ward', 'N/A')}
            Serum Cr: {patient_data.get('serum_cr', 'N/A')} µmol/L, CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})
            Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}
            Current Regimen: {patient_data.get('current_dose_regimen', 'N/A')}
            Notes: {patient_data.get('notes', 'N/A')}
            --- TDM Data & Calculations ---
            {prompt}
            --- End of Case ---
            """

            # Call OpenAI API - updated for openai v1.0.0+
            response = openai.chat.completions.create(
                model="gpt-4",  # or your preferred model like gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear, actionable recommendations in the specified format."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3, # Lower temperature for more deterministic clinical advice
                max_tokens=600 # Increased token limit for detailed response
            )
            llm_response = response.choices[0].message.content

            st.subheader("Clinical Interpretation (LLM)")
            st.markdown(llm_response) # Display the formatted response directly
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")

            # No PDF generation needed here

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
            # Fall through to standardized interpretation if API fails

    # If OpenAI is not available/fails, use the standardized interpretation
    if not (OPENAI_AVAILABLE and openai.api_key): # Or if the API call failed above
        st.subheader("Clinical Interpretation (Simulated)")
        interpretation_data = generate_standardized_interpretation(prompt, drug, patient_data)

        # If the interpretation_data is a string (error message), just display it
        if isinstance(interpretation_data, str):
            st.write(interpretation_data)
            return

        # Unpack the interpretation data
        levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data

        # Display the formatted interpretation
        formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
        st.markdown(formatted_interpretation) # Use markdown for better formatting

        # Add note about simulated response
        st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")

    # Add the raw prompt at the bottom for debugging/transparency
    with st.expander("Raw Analysis Data Sent to LLM (or used for Simulation)", expanded=False):
        st.code(prompt)


def generate_standardized_interpretation(prompt, drug, patient_data):
    """
    Generate a standardized interpretation based on drug type and prompt content.
    Includes patient context for better recommendations.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    crcl = patient_data.get('crcl', None) # Get CrCl for context

    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt, crcl)
    elif drug == "Aminoglycoside":
        return generate_aminoglycoside_interpretation(prompt, crcl)
    else:
        # For generic, create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        if crcl and crcl < 60:
             cautions.append(f"Renal function (CrCl: {crcl:.1f} mL/min) may impact dosing.")

        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy using Markdown.

    Parameters:
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string (e.g., "appropriately dosed")
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - cautions: Optional list of caution strings

    Returns:
    - Formatted markdown string
    """
    # Format measured levels with status indicators
    levels_md = "📊 **MEASURED/ESTIMATED LEVELS:**\n"
    if not levels_data or (len(levels_data) == 1 and levels_data[0][0] == "Not available"):
         levels_md += "- No levels data available for interpretation.\n"
    else:
        for name, value, target, status in levels_data:
            icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴" # Red for above
            # Format value appropriately (e.g., 1 decimal for levels, 0 for AUC)
            value_str = f"{value:.1f}" if isinstance(value, (int, float)) and "AUC" not in name else f"{value:.0f}" if isinstance(value, (int, float)) else str(value)
            levels_md += f"- {name}: {value_str} (Target: {target}) {icon}\n"


    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is **{assessment.upper()}**"

    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## RECOMMENDATIONS

🔵 **DOSING:**
"""
    if dosing_recs:
        for rec in dosing_recs:
            output += f"- {rec}\n"
    else:
        output += "- No specific dosing recommendations generated.\n"

    output += "\n🔵 **MONITORING:**\n"
    if monitoring_recs:
        for rec in monitoring_recs:
            output += f"- {rec}\n"
    else:
        output += "- Standard monitoring applies.\n"

    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"

    return output

def generate_vancomycin_interpretation(prompt, crcl=None):
    """
    Generate standardized vancomycin interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    # Extract key values from the prompt using regex for robustness
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract levels (measured or estimated)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Trough.*?([\d.,]+)\s*mg/L", prompt)
peak_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Peak.*?([\d.,]+)\s*mg/L", prompt)
    auc_val = extract_float(r"(?:Estimated|Predicted)\s+AUC24.*?([\d.,]+)\s*mg.hr/L", prompt)

    # Extract targets
    target_auc_str = extract_string(r"Target\s+AUC24.*?(\d+\s*-\s*\d+)\s*mg.hr/L", prompt, "400-600")
    target_trough_str = extract_string(r"(?:Target|Secondary Target)\s+Trough.*?([\d.]+\s*-\s*[\d.]+)\s*mg/L", prompt, "10-15")

    # Extract current/new regimen details
    current_dose_interval = extract_float(r"Current\s+Regimen.*?([\d,]+)\s*mg.*?q\s*(\d+)", prompt)
    current_interval = extract_float(r"Current\s+Regimen.*?q\s*(\d+)", prompt)
    new_dose_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?([\d,]+)\s*mg.*?q\s*(\d+)", prompt)
    new_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?q\s*(\d+)", prompt)


    # Parse target ranges
    auc_target_min, auc_target_max = 400, 600
    auc_match = re.match(r"(\d+)\s*-\s*(\d+)", target_auc_str)
    if auc_match: auc_target_min, auc_target_max = int(auc_match.group(1)), int(auc_match.group(2))
    auc_target_formatted = f"{auc_target_min}-{auc_target_max} mg·hr/L"

    trough_target_min, trough_target_max = 10, 15
    trough_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_trough_str)
    if trough_match:
        try:
            trough_target_min = float(trough_match.group(1))
            trough_target_max = float(trough_match.group(2))
        except ValueError: pass
    trough_target_formatted = f"{trough_target_min:.1f}-{trough_target_max:.1f} mg/L"


    # Check if essential values for assessment were extracted
    if trough_val is None and auc_val is None:
        return "Insufficient level data (Trough or AUC) in prompt for standardized vancomycin interpretation."

    # --- Start Interpretation Logic ---
    levels_data = []
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    status = "assessment uncertain" # Default status

    # Assess Trough Level
    trough_status = "N/A"
    if trough_val is not None:
        if trough_val < trough_target_min: trough_status = "below"
        elif trough_val > trough_target_max: trough_status = "above"
        else: trough_status = "within"
        levels_data.append(("Trough", trough_val, trough_target_formatted, trough_status))

    # Assess AUC Level
    auc_status = "N/A"
    if auc_val is not None:
        if auc_val < auc_target_min: auc_status = "below"
        elif auc_val > auc_target_max: auc_status = "above"
        else: auc_status = "within"
        levels_data.append(("AUC24", auc_val, auc_target_formatted, auc_status))

    # Assess Peak Level (if available)
    peak_status = "N/A"
    if peak_val is not None:
        # Define peak range based on empiric vs definitive therapy
        # Assuming trough level helps determine empiric vs definitive
        if trough_val is not None and trough_val <= 15:  # Likely empiric therapy
            peak_target_min, peak_target_max = 20, 30
            peak_target_formatted = f"{peak_target_min}-{peak_target_max} mg/L (Empiric)"
        else:  # Likely definitive therapy
            peak_target_min, peak_target_max = 25, 40
            peak_target_formatted = f"{peak_target_min}-{peak_target_max} mg/L (Definitive)"
        
        if peak_val < peak_target_min: peak_status = "below"
        elif peak_val > peak_target_max: peak_status = "above"
        else: peak_status = "within"
        levels_data.append(("Peak", peak_val, peak_target_formatted, peak_status))


    # Determine overall assessment status (prioritize AUC, then Trough)
    if auc_status == "within" and trough_status != "above": status = "appropriately dosed (AUC target met)"
    elif auc_status == "within" and trough_status == "above": status = "potentially overdosed (AUC ok, trough high)"
    elif auc_status == "below": status = "underdosed (AUC below target)"
    elif auc_status == "above": status = "overdosed (AUC above target)"
    elif auc_status == "N/A": # If AUC not available, use trough
         if trough_status == "within": status = "likely appropriately dosed (trough target met)"
         elif trough_status == "below": status = "likely underdosed (trough below target)"
         elif trough_status == "above": status = "likely overdosed (trough above target)"


    # Generate recommendations based on status
    if "appropriately dosed" in status:
        dosing_recs.append("CONTINUE current regimen.")
        if current_dose_interval and current_interval:
             dosing_recs.append(f"MAINTAIN {current_dose_interval:.0f}mg q{current_interval:.0f}h.")
        monitoring_recs.append("MONITOR renal function per protocol (e.g., 2-3 times weekly).")
        monitoring_recs.append("REPEAT levels if clinical status or renal function changes significantly.")
        if status == "potentially overdosed (AUC ok, trough high)": # Add caution if trough high despite AUC ok
             cautions.append("Trough is elevated, increasing potential nephrotoxicity risk despite acceptable AUC. Monitor renal function closely.")
             monitoring_recs.append("Consider rechecking trough sooner if renal function declines.")

    else: # Underdosed or Overdosed
        if "underdosed" in status:
             dosing_recs.append("INCREASE dose and/or shorten interval to achieve target AUC.")
             monitoring_recs.append("RECHECK levels after 3-5 doses of new regimen (allow steady state).")
        elif "overdosed" in status:
             if trough_val is not None and trough_val > 25: # Significantly high trough
                 dosing_recs.append("HOLD next dose(s) until trough is acceptable (e.g., < 20 mg/L).")
                 cautions.append("Significantly elevated trough increases nephrotoxicity risk.")
             dosing_recs.append("DECREASE dose and/or lengthen interval to achieve target AUC.")
             monitoring_recs.append("RECHECK levels within 24-48 hours after adjustment (or before next dose if interval long).")
             monitoring_recs.append("MONITOR renal function daily until stable.")

        # Suggest new regimen if provided in prompt
        if new_dose_interval and new_interval:
             # Suggest practical regimens based on new TDD
             new_tdd_calc = new_dose_interval * (24 / new_interval)
             suggested_regimens = []
             for practical_interval_opt in [8, 12, 24, 36, 48]: # Common intervals
                 dose_per_interval_opt = new_tdd_calc / (24 / practical_interval_opt)
                 # Round dose per interval to nearest 250mg
                 rounded_dose_opt = round(dose_per_interval_opt / 250) * 250
                 if rounded_dose_opt > 0:
                     # Check if this option is close to the suggested one
                     is_suggested = abs(practical_interval_opt - new_interval) < 1 and abs(rounded_dose_opt - new_dose_interval) < 125
                     prefix = "➡️" if is_suggested else "  -"
                     suggested_regimens.append(f"{prefix} {rounded_dose_opt:.0f}mg q{practical_interval_opt}h (approx. {rounded_dose_opt * (24/practical_interval_opt):.0f}mg/day)")

             if suggested_regimens:
                 dosing_recs.append(f"ADJUST regimen towards target AUC ({auc_target_formatted}). Consider practical options:")
                 # Add the explicitly suggested regimen first if found
                 explicit_suggestion = f"{new_dose_interval:.0f}mg q{new_interval:.0f}h"
                 if not any(explicit_suggestion in reg for reg in suggested_regimens):
                      dosing_recs.append(f"➡️ {explicit_suggestion} (Calculated)") # Add if not already covered by rounding
                 for reg in suggested_regimens:
                     dosing_recs.append(reg)

             else: # Fallback if no practical options generated
                  dosing_recs.append(f"ADJUST regimen to {new_dose_interval:.0f}mg q{new_interval:.0f}h as calculated.")
        else: # If no new dose calculated in prompt
             dosing_recs.append("ADJUST regimen based on clinical judgment and estimated PK to achieve target AUC.")


    # Add renal function caution if relevant
    if crcl is not None:
        renal_status = ""
        if crcl < 15: renal_status = "Kidney Failure"
        elif crcl < 30: renal_status = "Severe Impairment"
        elif crcl < 60: renal_status = "Moderate Impairment"
        elif crcl < 90: renal_status = "Mild Impairment"

        if crcl < 60: # Add caution for moderate to severe impairment
            cautions.append(f"{renal_status} (CrCl: {crcl:.1f} mL/min). Increased risk of accumulation and toxicity. Monitor levels and renal function closely.")
            if "overdosed" in status or (trough_val is not None and trough_val > target_trough_max):
                 monitoring_recs.append("MONITOR renal function at least daily.")
            else:
                 monitoring_recs.append("MONITOR renal function frequently (e.g., every 1-2 days).")

    cautions.append("Ensure appropriate infusion duration (e.g., ≥ 1 hour per gram, max rate 1g/hr) to minimize infusion reactions.")
    cautions.append("Consider potential drug interactions affecting vancomycin clearance or toxicity (e.g., piperacillin-tazobactam, loop diuretics, other nephrotoxins).")


    return levels_data, status, dosing_recs, monitoring_recs, cautions


def generate_aminoglycoside_interpretation(prompt, crcl=None):
    """
    Generate standardized aminoglycoside interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract drug name
    drug_match = re.search(r"Drug:\s*(Gentamicin|Amikacin)", prompt, re.IGNORECASE)
    drug_name = drug_match.group(1).lower() if drug_match else "aminoglycoside"

    # Extract levels (measured or estimated)
    peak_val = extract_float(r"(?:Measured|Estimated|Predicted|Est\.)\s+(?:Peak|Cmax).*?([\d.,]+)\s*mg/L", prompt)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted|Est\.)\s+(?:Trough|Cmin|C1).*?([\d.,]+)\s*mg/L", prompt) # Allow C1 as trough

    # Extract targets
    target_peak_str = extract_string(r"Target\s+Peak.*?((?:[\d.]+\s*-\s*[\d.]+|[<>]?\s*[\d.]+|N/A|Not routinely targeted))\s*mg/L", prompt, "N/A")
    target_trough_str = extract_string(r"Target\s+Trough.*?((?:[\d.]+\s*-\s*[\d.]+|[<>]?\s*[\d.]+|N/A))\s*mg/L", prompt, "N/A")

    # Extract current/new regimen details
    current_dose = extract_float(r"Current\s+Regimen.*?([\d,]+)\s*mg", prompt)
    current_interval = extract_float(r"Current\s+Regimen.*?q\s*(\d+)", prompt)
    new_dose = extract_float(r"(?:Suggested|New)\s+Regimen.*?([\d,]+)\s*mg", prompt)
    new_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?q\s*(\d+)", prompt)


    # --- Parse Target Ranges ---
    peak_min, peak_max = 0, 100 # Default wide range
    trough_limit_type = "max" # Assume target is '< max' by default
    trough_max = 100 # Default wide range

    # Parse Peak Target String
    if "N/A" in target_peak_str or "not targeted" in target_peak_str:
        peak_min, peak_max = None, None # Indicate not applicable
    else:
        peak_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_peak_str)
        if peak_match:
            try: peak_min, peak_max = float(peak_match.group(1)), float(peak_match.group(2))
            except ValueError: pass # Keep defaults if parsing fails

    # Parse Trough Target String
    if "N/A" in target_trough_str:
        trough_max = None # Indicate not applicable
    else:
        trough_match_less = re.match(r"<\s*([\d.]+)", target_trough_str)
        trough_match_range = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_trough_str)
        if trough_match_less:
            try: trough_max = float(trough_match_less.group(1)); trough_limit_type = "max"
            except ValueError: pass
        elif trough_match_range: # Handle if a range is given for trough (less common for amino)
             try: trough_max = float(trough_match_range.group(2)); trough_limit_type = "range"; trough_min = float(trough_match_range.group(1))
             except ValueError: pass # Default to max limit if range parsing fails


    # Check if essential level values were extracted
    if peak_val is None or trough_val is None:
        # Allow interpretation if only trough is available for HD patients
        if not ("Hemodialysis" in prompt and trough_val is not None):
             return "Insufficient level data (Peak or Trough) in prompt for standardized aminoglycoside interpretation."


    # --- Start Interpretation Logic ---
    levels_data = []
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    status = "assessment uncertain" # Default status

    # Assess Peak Level
    peak_status = "N/A"
    if peak_min is not None and peak_max is not None and peak_val is not None:
        if peak_val < peak_min: peak_status = "below"
        elif peak_val > peak_max: peak_status = "above"
        else: peak_status = "within"
        levels_data.append(("Peak", peak_val, target_peak_str, peak_status))
    elif peak_val is not None: # If target is N/A but value exists
         levels_data.append(("Peak", peak_val, target_peak_str, "N/A"))


    # Assess Trough Level
    trough_status = "N/A"
    if trough_max is not None and trough_val is not None:
        if trough_limit_type == "max":
            if trough_val >= trough_max: trough_status = "above" # At or above the max limit
            else: trough_status = "within" # Below the max limit
        elif trough_limit_type == "range":
             if trough_val < trough_min: trough_status = "below" # Below the range min (unlikely target for amino)
             elif trough_val > trough_max: trough_status = "above" # Above the range max
             else: trough_status = "within"
        levels_data.append(("Trough", trough_val, target_trough_str, trough_status))
    elif trough_val is not None: # If target is N/A but value exists
        levels_data.append(("Trough", trough_val, target_trough_str, "N/A"))


    # Determine overall assessment status
    # Prioritize avoiding toxicity (high trough), then achieving efficacy (adequate peak)
    if trough_status == "above":
        status = "potentially toxic (elevated trough)"
        if peak_status == "below": status = "ineffective and potentially toxic" # Worst case
    elif peak_status == "below":
        status = "subtherapeutic (inadequate peak)"
    elif peak_status == "above": # Peak high, trough ok
        status = "potentially supratherapeutic (high peak)"
    elif peak_status == "within" and trough_status == "within":
        status = "appropriately dosed"
    elif peak_status == "N/A" and trough_status == "within": # e.g., HD patient trough ok
         status = "likely appropriate (trough acceptable)"
    elif peak_status == "N/A" and trough_status == "above": # e.g., HD patient trough high
         status = "potentially toxic (elevated trough)"


    # Generate recommendations
    if "appropriately dosed" in status or "likely appropriate" in status :
        dosing_recs.append("CONTINUE current regimen.")
        if current_dose and current_interval: dosing_recs.append(f"MAINTAIN {current_dose:.0f}mg q{current_interval:.0f}h.")
        monitoring_recs.append("MONITOR renal function regularly (e.g., 2-3 times weekly or per HD schedule).")
        monitoring_recs.append("REPEAT levels if clinical status, renal function, or dialysis schedule changes.")
    elif status == "assessment not applicable": # Synergy Amikacin
         dosing_recs.append("Follow specific institutional protocol for Synergy Amikacin dosing.")
         monitoring_recs.append("MONITOR renal function and clinical status.")
    else: # Adjustments needed
        if status == "ineffective and potentially toxic":
             dosing_recs.append("HOLD next dose(s).")
             dosing_recs.append("INCREASE dose AND EXTEND interval significantly once resumed.")
             monitoring_recs.append("RECHECK levels (peak & trough) before resuming and after 2-3 doses of new regimen.")
             cautions.append("High risk of toxicity and low efficacy with current levels.")
        elif status == "subtherapeutic (inadequate peak)":
             dosing_recs.append("INCREASE dose.")
             dosing_recs.append("MAINTAIN current interval (unless trough also borderline high).")
             monitoring_recs.append("RECHECK peak and trough after 1-2 doses of new regimen.")
             cautions.append("Inadequate peak may compromise efficacy, especially for gram-negative infections.")
        elif status == "potentially toxic (elevated trough)":
             dosing_recs.append("EXTEND dosing interval.")
             dosing_recs.append("MAINTAIN current dose amount (or consider slight reduction if peak also high/borderline).")
             monitoring_recs.append("RECHECK trough level before next scheduled dose.")
             cautions.append("Elevated trough increases risk of nephrotoxicity and ototoxicity. Hold dose if trough significantly elevated.")
        elif status == "potentially supratherapeutic (high peak)": # High peak, trough ok
             dosing_recs.append("DECREASE dose.")
             dosing_recs.append("MAINTAIN current interval.")
             monitoring_recs.append("RECHECK peak and trough after 1-2 doses of new regimen.")
             cautions.append("Elevated peak may increase toxicity risk slightly, though trough is primary driver. Ensure trough remains acceptable.")

        # Suggest new regimen if provided in prompt
        if new_dose and new_interval:
             # Round new dose to nearest 10mg or 20mg
             rounding = 20 if drug_name == "gentamicin" else 50 if drug_name == "amikacin" else 10
             practical_new_dose = round(new_dose / rounding) * rounding
             if practical_new_dose > 0:
                 dosing_recs.append(f"Consider adjusting regimen towards: {practical_new_dose:.0f}mg q{new_interval:.0f}h.")
        else:
             dosing_recs.append("ADJUST regimen based on clinical judgment and estimated PK to achieve target levels.")


    # Add general monitoring and cautions
    monitoring_recs.append("MONITOR renal function (SCr, BUN, UOP) at least 2-3 times weekly, or more frequently if unstable, trough elevated, or on concomitant nephrotoxins.")
    monitoring_recs.append("MONITOR for signs/symptoms of nephrotoxicity (rising SCr, decreased UOP) and ototoxicity (hearing changes, tinnitus, vertigo).")
    cautions.append(f"{drug_name.capitalize()} carries risk of nephrotoxicity and ototoxicity.")
    cautions.append("Risk increases with prolonged therapy (>7-10 days), pre-existing renal impairment, high troughs, large cumulative dose, and concomitant nephrotoxins (e.g., vancomycin, diuretics, contrast).")
    if crcl is not None:
         renal_status = ""
         if crcl < 15: renal_status = "Kidney Failure"
         elif crcl < 30: renal_status = "Severe Impairment"
         elif crcl < 60: renal_status = "Moderate Impairment"
         elif crcl < 90: renal_status = "Mild Impairment"
         if crcl < 60:
             cautions.append(f"{renal_status} (CrCl: {crcl:.1f} mL/min) significantly increases toxicity risk. Adjust dose/interval carefully and monitor very closely.")


    return levels_data, status, dosing_recs, monitoring_recs, cautions


# ===== SIDEBAR: NAVIGATION AND PATIENT INFO =====
def setup_sidebar_and_navigation():
    st.sidebar.title("📊 Navigation")
    # Sidebar radio for selecting the module
    page = st.sidebar.radio("Select Module", [
        "Aminoglycoside: Initial Dose",
        "Aminoglycoside: Conventional Dosing (C1/C2)",
        "Vancomycin AUC-based Dosing"
    ])

    st.sidebar.title("🩺 Patient Demographics")
    # ADDED Patient ID and Ward
    patient_id = st.sidebar.text_input("Patient ID", value="N/A")
    ward = st.sidebar.text_input("Ward", value="N/A")
    # --- Existing fields ---
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=65)
    height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=165)
    weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1, format="%.1f")
    serum_cr = st.sidebar.number_input("Serum Creatinine (µmol/L)", min_value=10.0, max_value=2000.0, value=90.0, step=1.0)

    # Calculate Cockcroft-Gault Creatinine Clearance
    crcl = 0.0 # Default value
    renal_function = "N/A"
    if age > 0 and weight > 0 and serum_cr > 0: # Avoid division by zero or negative age
        # Cockcroft-Gault Formula
        crcl_factor = (140 - age) * weight
        crcl_gender_mult = 1.23 if gender == "Male" else 1.04
        crcl = (crcl_factor * crcl_gender_mult) / serum_cr
        crcl = max(0, crcl) # Ensure CrCl is not negative

        # Renal function category based on CrCl
        if crcl >= 90: renal_function = "Normal (≥90)"
        elif crcl >= 60: renal_function = "Mild Impairment (60-89)"
        elif crcl >= 30: renal_function = "Moderate Impairment (30-59)"
        elif crcl >= 15: renal_function = "Severe Impairment (15-29)"
        else: renal_function = "Kidney Failure (<15)"

    with st.sidebar.expander("Creatinine Clearance (Cockcroft-Gault)", expanded=True):
        if age > 0 and weight > 0 and serum_cr > 0:
            st.success(f"CrCl: {crcl:.1f} mL/min")
            st.info(f"Renal Function: {renal_function}")
        else:
            st.warning("Enter valid Age (>0), Weight (>0), and SCr (>0) to calculate CrCl.")


    st.sidebar.title("🩺 Clinical Information")
    clinical_diagnosis = st.sidebar.text_input("Diagnosis / Indication", placeholder="e.g., Pneumonia, Sepsis")
    current_dose_regimen = st.sidebar.text_area("Current Dosing Regimen", value="1g IV q12h", placeholder="e.g., Gentamicin 120mg IV q8h")
    notes = st.sidebar.text_area("Other Clinical Notes", value="No known allergies.", placeholder="e.g., Fluid status, interacting meds")

    # UPDATED clinical_summary
    clinical_summary = (
        f"Patient ID: {patient_id}, Ward: {ward}\n"
        f"Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} cm\n"
        f"SCr: {serum_cr} µmol/L\n"
        f"Diagnosis: {clinical_diagnosis}\n"
        f"Renal function: {renal_function} (Est. CrCl: {crcl:.1f} mL/min)\n"
        f"Current regimen: {current_dose_regimen}\n"
        f"Notes: {notes}"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Antimicrobial TDM App v1.2**

    Developed for therapeutic drug monitoring of antimicrobials.

    Provides PK estimates, AUC calculations, and dosing recommendations
    for vancomycin and aminoglycosides. Includes optional LLM interpretation.

    **Disclaimer:** This tool assists clinical decision making but does not replace
    professional judgment. Verify all calculations and recommendations.
    """)

    # Return all the data entered in the sidebar
    return {
        'page': page,
        'patient_id': patient_id, # Added
        'ward': ward,           # Added
        'gender': gender,
        'age': age,
        'height': height,
        'weight': weight,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen,
        'notes': notes,
        'clinical_summary': clinical_summary # Updated summary string
    }

# ===== MODULE 1: Aminoglycoside Initial Dose =====
def aminoglycoside_initial_dose(patient_data):
    st.title("🧮 Aminoglycoside Initial Dose (Population PK)")

    # Unpack patient data for easier access
    gender = patient_data['gender']
    age = patient_data['age']
    height = patient_data['height']
    weight = patient_data['weight']
    crcl = patient_data['crcl']
    notes = patient_data['notes']

    drug = st.selectbox("Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Dosing Strategy / Goal", ["Extended Interval (Once Daily - SDD)", "Traditional (Multiple Daily - MDD)", "Synergy (e.g., Endocarditis)", "Hemodialysis", "Neonates (Use with caution)"])

    # Map selection to internal codes
    regimen_code = "SDD" if "Extended" in regimen \
              else "MDD" if "Traditional" in regimen \
              else "Synergy" if "Synergy" in regimen \
              else "Hemodialysis" if "Hemodialysis" in regimen \
              else "Neonates" if "Neonates" in regimen \
              else "MDD" # Default

    # --- Set default target ranges based on regimen and drug ---
    target_peak_info = "N/A"
    target_trough_info = "N/A"
    default_peak = 0.0
    default_trough = 0.0

    if drug == "Gentamicin":import streamlit as st
import numpy as np
import math
import openai
import pandas as pd
import altair as alt
import base64
from datetime import datetime, time, timedelta # Added time and timedelta

# Optional imports - Bayesian functionality
try:
    import scipy.optimize as optimize
    from scipy.stats import norm
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scipy not installed. Bayesian estimation will not be available.")

# Optional imports - FAISS for guideline embedding (if needed)
try:
    import faiss
    import pickle
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Guideline embeddings will not be available.")

st.set_page_config(page_title="Antimicrobial TDM App", layout="wide")

# ===== API CONFIGURATION =====
# Securely access the API key from streamlit secrets
try:
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.

    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== LOAD EMBEDDED GUIDELINE =====
@st.cache_resource
def load_guideline_embeddings():
    if not FAISS_AVAILABLE:
        return None, ["FAISS not installed. Guideline embeddings not available."]

    # Replace this with your actual FAISS index and guideline chunks
    index = faiss.IndexFlatL2(768)  # Dummy placeholder index
    chunks = ["Guideline excerpt 1", "Guideline excerpt 2", "Guideline excerpt 3"]
    return index, chunks

guideline_index, guideline_chunks = None, []
if FAISS_AVAILABLE:
    guideline_index, guideline_chunks = load_guideline_embeddings()

# ===== Global Practical Dosing Regimens =====
practical_intervals = "6hr, 8hr, 12hr, 24hr, every other day"

# ===== HELPER FUNCTION: Updated Practical Dose Adjustment =====
def suggest_adjustment(parameter, target_min, target_max, label="Parameter", intervals=practical_intervals):
    # Ensure targets are valid numbers before comparison
    if isinstance(target_min, (int, float)) and isinstance(target_max, (int, float)) and isinstance(parameter, (int, float)):
        if parameter < target_min:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is low. Target: {target_min:.1f}-{target_max:.1f}. Consider increasing dose or shortening interval ({intervals}).")
        elif parameter > target_max:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is high. Target: {target_min:.1f}-{target_max:.1f}. Consider reducing dose or lengthening interval ({intervals}).")
        else:
            st.success(f"✅ {label} ({parameter:.1f}) is within target range ({target_min:.1f}-{target_max:.1f}).")
    else:
        st.info(f"{label}: {parameter}. Target range: {target_min}-{target_max}. (Comparison skipped due to non-numeric values).")


# ===== PDF GENERATION FUNCTIONS (REMOVED) =====
# create_recommendation_pdf, get_pdf_download_link, display_pdf_download_button functions removed.

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization

    Parameters:
    - peak: Peak concentration (mg/L)
    - trough: Trough concentration (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - t_peak: Time to peak after start of infusion (hr) - assumed end of infusion
    - infusion_time: Duration of infusion (hr)

    Returns:
    - Altair chart object
    """
    # Generate time points for the curve
    times = np.linspace(0, tau*1.5, 150)  # Generate points for 1.5 intervals to show next dose

    # Generate concentrations for each time point using steady-state equations
    concentrations = []
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * T_inf)) * exp(-ke * (t - T_inf)) / (1 - exp(-ke * tau)) -- Post-infusion
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * t)) / (1 - exp(-ke * tau)) -- During infusion (simplified, assumes Cmin=0 start)
    # Let's use the provided peak and trough which represent Cmax (at t=infusion_time) and Cmin (at t=tau)

    for t_cycle in np.linspace(0, tau*1.5, 150): # Iterate through time points
        # Determine concentration based on time within the dosing cycle (modulo tau)
        t = t_cycle % tau
        num_cycles = int(t_cycle // tau) # Which cycle we are in (0, 1, ...)

        conc = 0
        if t <= infusion_time:
            # During infusion: Assume linear rise from previous trough to current peak
            # This is an approximation but visually represents the infusion period
            conc = trough + (peak - trough) * (t / infusion_time)
        else:
            # After infusion: Exponential decay from peak
            time_since_peak = t - infusion_time # Time elapsed since the peak concentration (end of infusion)
            conc = peak * math.exp(-ke * time_since_peak)

        concentrations.append(max(0, conc)) # Ensure concentration doesn't go below 0


    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Time (hr)': times,
        'Concentration (mg/L)': concentrations
    })

    # --- Create Target Bands ---
    target_bands = []
    # Determine drug type based on typical levels for band coloring
    if peak > 45 or trough > 20:  # Likely vancomycin
        # Vancomycin Peak Target - Empiric vs Definitive
        if trough <= 15:  # Likely empiric (target trough 10-15)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [20], 'y2': [30]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Empiric)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [10], 'y2': [15]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Empiric)")))
        else:  # Likely definitive (target trough 15-20)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [25], 'y2': [40]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Definitive)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [15], 'y2': [20]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Definitive)")))
    else:  # Likely aminoglycoside (e.g., Gentamicin)
        # Aminoglycoside Peak Target (e.g., 5-10 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [5], 'y2': [10]}))
                           .mark_rect(opacity=0.15, color='lightblue')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Amino)")))
        # Aminoglycoside Trough Target (e.g., <2 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [0], 'y2': [2]}))
                           .mark_rect(opacity=0.15, color='lightgreen')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Amino)")))


    # --- Create Concentration Line ---
    line = alt.Chart(df).mark_line(color='firebrick').encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)', scale=alt.Scale(zero=True)), # Ensure Y axis starts at 0
        tooltip=['Time (hr)', alt.Tooltip('Concentration (mg/L)', format=".1f")]
    )

    # --- Add Vertical Lines for Key Events ---
    vertical_lines_data = []
    # Mark end of infusion for each cycle shown
    for i in range(int(tau*1.5 / tau) + 1):
        inf_end_time = i * tau + infusion_time
        if inf_end_time <= tau*1.5:
             vertical_lines_data.append({'Time': inf_end_time, 'Event': 'Infusion End'})
    # Mark start of next dose for each cycle shown
    for i in range(1, int(tau*1.5 / tau) + 1):
         dose_time = i * tau
         if dose_time <= tau*1.5:
              vertical_lines_data.append({'Time': dose_time, 'Event': 'Next Dose'})

    vertical_lines_df = pd.DataFrame(vertical_lines_data)

    vertical_rules = alt.Chart(vertical_lines_df).mark_rule(strokeDash=[4, 4]).encode(
        x='Time',
        color=alt.Color('Event', scale=alt.Scale(domain=['Infusion End', 'Next Dose'], range=['gray', 'black'])),
        tooltip=['Event', 'Time']
    )

    # --- Combine Charts ---
    chart = alt.layer(*target_bands, line, vertical_rules).properties(
        width=alt.Step(4), # Adjust width automatically
        height=400,
        title=f'Estimated Concentration-Time Profile (Tau={tau} hr)'
    ).interactive() # Make chart interactive (zoom/pan)

    return chart


# ===== VANCOMYCIN AUC CALCULATION (TRAPEZOIDAL METHOD) =====
def calculate_vancomycin_auc_trapezoidal(cmax, cmin, ke, tau, infusion_duration):
    """
    Calculate vancomycin AUC24 using the linear-log trapezoidal method.
    
    This method is recommended for vancomycin TDM as per the guidelines.
    
    Parameters:
    - cmax: Max concentration at end of infusion (mg/L)
    - cmin: Min concentration at end of interval (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - infusion_duration: Duration of infusion (hr)
    
    Returns:
    - AUC24: 24-hour area under the curve (mg·hr/L)
    """
    # Calculate concentration at start of infusion (C0)
    c0 = cmax * math.exp(ke * infusion_duration)
    
    # Calculate AUC during infusion phase (linear trapezoid)
    auc_inf = infusion_duration * (c0 + cmax) / 2
    
    # Calculate AUC during elimination phase (log trapezoid)
    if ke > 0 and cmax > cmin:
        auc_elim = (cmax - cmin) / ke
    else:
        # Fallback to linear trapezoid if ke is very small
        auc_elim = (tau - infusion_duration) * (cmax + cmin) / 2
    
    # Calculate total AUC for one dosing interval
    auc_interval = auc_inf + auc_elim
    
    # Convert to AUC24
    auc24 = auc_interval * (24 / tau)
    
    return auc24

# ===== BAYESIAN PARAMETER ESTIMATION =====
def bayesian_parameter_estimation(measured_levels, sample_times, dose, tau, weight, age, crcl, gender):
    """
    Bayesian estimation of PK parameters based on measured levels

    Parameters:
    - measured_levels: List of measured drug concentrations (mg/L)
    - sample_times: List of times when samples were taken (hr after dose start)
    - dose: Dose administered (mg)
    - tau: Dosing interval (hr)
    - weight: Patient weight (kg)
    - age: Patient age (years)
    - crcl: Creatinine clearance (mL/min)
    - gender: Patient gender ("Male" or "Female")

    Returns:
    - Dictionary with estimated PK parameters
    """
    if not BAYESIAN_AVAILABLE:
        st.error("Bayesian estimation requires scipy. Please install it with 'pip install scipy'")
        return None

    # Prior population parameters for vancomycin (adjust if needed for aminoglycosides)
    # Mean values
    vd_pop_mean = 0.7  # L/kg (Vancomycin specific, adjust for aminoglycosides if used)
    ke_pop_mean = 0.00083 * crcl + 0.0044 # hr^-1 (Vancomycin specific - ensure crcl is used correctly)
    ke_pop_mean = max(0.01, ke_pop_mean) # Ensure Ke isn't too low

    # Standard deviations for population parameters
    vd_pop_sd = 0.2  # L/kg
    ke_pop_sd = 0.05 # Increased SD for Ke prior to allow more flexibility

    # Define objective function to minimize (negative log likelihood)
    def objective_function(params):
        vd_ind, ke_ind = params # Individual parameters to estimate
        vd_total = vd_ind * weight

        # Calculate expected concentrations at sample times using steady-state infusion model
        expected_concs = []
        infusion_time = 1.0 # Assume 1 hour infusion, make adjustable if needed

        for t in sample_times:
            # Steady State Concentration Equation (1-compartment, intermittent infusion)
            term_dose_vd = dose / vd_total
            term_ke_tinf = ke_ind * infusion_time
            term_ke_tau = ke_ind * tau

            try:
                exp_ke_tinf = math.exp(-term_ke_tinf)
                exp_ke_tau = math.exp(-term_ke_tau)

                if abs(1.0 - exp_ke_tau) < 1e-9: # Avoid division by zero if tau is very long or ke very small
                    # Handle as if continuous infusion or single dose if tau is effectively infinite
                    conc = 0 # Simplified - needs better handling for edge cases
                else:
                    common_factor = (term_dose_vd / term_ke_tinf) * (1.0 - exp_ke_tinf) / (1.0 - exp_ke_tau)

                    if t <= infusion_time: # During infusion phase
                        conc = common_factor * (1.0 - math.exp(-ke_ind * t))
                    else: # Post-infusion phase
                        conc = common_factor * math.exp(-ke_ind * (t - infusion_time))

            except OverflowError:
                 conc = float('inf') # Handle potential overflow with large ke/t values
            except ValueError:
                 conc = 0 # Handle math domain errors

            expected_concs.append(max(0, conc)) # Ensure non-negative

        # Calculate negative log likelihood
        # Measurement error model (e.g., proportional + additive)
        # sd = sqrt(sigma_add^2 + (sigma_prop * expected_conc)^2)
        sigma_add = 1.0  # Additive SD (mg/L)
        sigma_prop = 0.1 # Proportional SD (10%)
        nll = 0
        for i in range(len(measured_levels)):
            expected = expected_concs[i]
            measurement_sd = math.sqrt(sigma_add**2 + (sigma_prop * expected)**2)
            if measurement_sd < 1e-6: measurement_sd = 1e-6 # Prevent division by zero in logpdf

            # Add contribution from measurement likelihood
            # Use logpdf for robustness, especially with low concentrations
            nll += -norm.logpdf(measured_levels[i], loc=expected, scale=measurement_sd)

        # Add contribution from parameter priors (log scale often more stable for Ke)
        # Prior for Vd (Normal)
        nll += -norm.logpdf(vd_ind, loc=vd_pop_mean, scale=vd_pop_sd)
        # Prior for Ke (Log-Normal might be better, but using Normal for simplicity)
        nll += -norm.logpdf(ke_ind, loc=ke_pop_mean, scale=ke_pop_sd)

        # Penalize non-physical parameters slightly if optimization strays
        if vd_ind <= 0 or ke_ind <= 0:
             nll += 1e6 # Add large penalty

        return nll

    # Initial guess based on population values
    initial_params = [vd_pop_mean, ke_pop_mean]

    # Parameter bounds (physical constraints)
    bounds = [(0.1, 2.5), (0.001, 0.5)]  # Reasonable bounds for Vd (L/kg) and Ke (hr^-1)

    # Perform optimization using a robust method
    try:
        result = optimize.minimize(
            objective_function,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B', # Suitable for bound constraints
            options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 500} # Adjust tolerances/iterations
        )
    except Exception as e:
         st.error(f"Optimization failed: {e}")
         return None

    if not result.success:
        st.warning(f"Bayesian optimization did not converge: {result.message} (Function evaluations: {result.nfev})")
        # Optionally return population estimates or None
        return None # Indicate failure

    # Extract optimized parameters
    vd_opt_kg, ke_opt = result.x
    # Ensure parameters are within bounds post-optimization (should be handled by L-BFGS-B, but double-check)
    vd_opt_kg = max(bounds[0][0], min(bounds[0][1], vd_opt_kg))
    ke_opt = max(bounds[1][0], min(bounds[1][1], ke_opt))

    vd_total_opt = vd_opt_kg * weight
    cl_opt = ke_opt * vd_total_opt
    t_half_opt = 0.693 / ke_opt if ke_opt > 0 else float('inf')

    return {
        'vd': vd_opt_kg, # Vd per kg
        'vd_total': vd_total_opt, # Total Vd in L
        'ke': ke_opt,
        'cl': cl_opt,
        't_half': t_half_opt,
        'optimization_success': result.success,
        'final_nll': result.fun # Final negative log-likelihood value
    }


# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM.
    Uses OpenAI API if available, otherwise provides a simulated response.

    Parameters:
    - prompt: The clinical data prompt including calculated values and context.
    - patient_data: Dictionary with patient information (used for context).
    """
    # Extract the drug type from the prompt for context
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
    else:
        drug = "Antimicrobial"

    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Include patient context if relevant (e.g., renal function).
            Format your response with these exact sections:

            ## CLINICAL ASSESSMENT
            📊 **MEASURED/ESTIMATED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **ASSESSMENT:** (state if appropriately dosed, underdosed, or overdosed based on levels and targets)

            ## RECOMMENDATIONS
            🔵 **DOSING:** (action-oriented recommendations using verbs like CONTINUE, ADJUST, HOLD, INCREASE, DECREASE. Suggest practical regimens where possible.)
            🔵 **MONITORING:** (specific monitoring parameters and schedule, e.g., recheck levels, renal function)
            ⚠️ **CAUTIONS:** (relevant warnings, e.g., toxicity risk, renal impairment)

            Here is the case:
            --- Patient Context ---
            Age: {patient_data.get('age', 'N/A')} years, Gender: {patient_data.get('gender', 'N/A')}
            Weight: {patient_data.get('weight', 'N/A')} kg, Height: {patient_data.get('height', 'N/A')} cm
            Patient ID: {patient_data.get('patient_id', 'N/A')}, Ward: {patient_data.get('ward', 'N/A')}
            Serum Cr: {patient_data.get('serum_cr', 'N/A')} µmol/L, CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})
            Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}
            Current Regimen: {patient_data.get('current_dose_regimen', 'N/A')}
            Notes: {patient_data.get('notes', 'N/A')}
            --- TDM Data & Calculations ---
            {prompt}
            --- End of Case ---
            """

            # Call OpenAI API - updated for openai v1.0.0+
            response = openai.chat.completions.create(
                model="gpt-4",  # or your preferred model like gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear, actionable recommendations in the specified format."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3, # Lower temperature for more deterministic clinical advice
                max_tokens=600 # Increased token limit for detailed response
            )
            llm_response = response.choices[0].message.content

            st.subheader("Clinical Interpretation (LLM)")
            st.markdown(llm_response) # Display the formatted response directly
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")

            # No PDF generation needed here

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
            # Fall through to standardized interpretation if API fails

    # If OpenAI is not available/fails, use the standardized interpretation
    if not (OPENAI_AVAILABLE and openai.api_key): # Or if the API call failed above
        st.subheader("Clinical Interpretation (Simulated)")
        interpretation_data = generate_standardized_interpretation(prompt, drug, patient_data)

        # If the interpretation_data is a string (error message), just display it
        if isinstance(interpretation_data, str):
            st.write(interpretation_data)
            return

        # Unpack the interpretation data
        levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data

        # Display the formatted interpretation
        formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
        st.markdown(formatted_interpretation) # Use markdown for better formatting

        # Add note about simulated response
        st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")

    # Add the raw prompt at the bottom for debugging/transparency
    with st.expander("Raw Analysis Data Sent to LLM (or used for Simulation)", expanded=False):
        st.code(prompt)


def generate_standardized_interpretation(prompt, drug, patient_data):
    """
    Generate a standardized interpretation based on drug type and prompt content.
    Includes patient context for better recommendations.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    crcl = patient_data.get('crcl', None) # Get CrCl for context

    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt, crcl)
    elif drug == "Aminoglycoside":
        return generate_aminoglycoside_interpretation(prompt, crcl)
    else:
        # For generic, create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        if crcl and crcl < 60:
             cautions.append(f"Renal function (CrCl: {crcl:.1f} mL/min) may impact dosing.")

        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy using Markdown.

    Parameters:
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string (e.g., "appropriately dosed")
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - cautions: Optional list of caution strings

    Returns:
    - Formatted markdown string
    """
    # Format measured levels with status indicators
    levels_md = "📊 **MEASURED/ESTIMATED LEVELS:**\n"
    if not levels_data or (len(levels_data) == 1 and levels_data[0][0] == "Not available"):
         levels_md += "- No levels data available for interpretation.\n"
    else:
        for name, value, target, status in levels_data:
            icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴" # Red for above
            # Format value appropriately (e.g., 1 decimal for levels, 0 for AUC)
            value_str = f"{value:.1f}" if isinstance(value, (int, float)) and "AUC" not in name else f"{value:.0f}" if isinstance(value, (int, float)) else str(value)
            levels_md += f"- {name}: {value_str} (Target: {target}) {icon}\n"


    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is **{assessment.upper()}**"

    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## RECOMMENDATIONS

🔵 **DOSING:**
"""
    if dosing_recs:
        for rec in dosing_recs:
            output += f"- {rec}\n"
    else:
        output += "- No specific dosing recommendations generated.\n"

    output += "\n🔵 **MONITORING:**\n"
    if monitoring_recs:
        for rec in monitoring_recs:
            output += f"- {rec}\n"
    else:
        output += "- Standard monitoring applies.\n"

    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"

    return output

def generate_vancomycin_interpretation(prompt, crcl=None):
    """
    Generate standardized vancomycin interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    # Extract key values from the prompt using regex for robustness
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract levels (measured or estimated)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Trough.*?([\d.,]+)\s*mg/L", prompt)
    peak_val
    if drug == "Gentamicin":
        if regimen_code == "MDD": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 1.0, "5-10 mg/L", "<2 mg/L"
        elif regimen_code == "SDD": default_peak, default_trough, target_peak_info, target_trough_info = 20.0, 0.5, "15-30 mg/L (or 10x MIC)", "<1 mg/L (often undetectable)"
        elif regimen_code == "Synergy": default_peak, default_trough, target_peak_info, target_trough_info = 4.0, 0.5, "3-5 mg/L", "<1 mg/L"
        elif regimen_code == "Hemodialysis": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 1.0, "Peak not routinely targeted", "<2 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 0.5, "5-12 mg/L", "<1 mg/L"
    else:  # Amikacin
        if regimen_code == "MDD": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 5.0, "20-30 mg/L", "<10 mg/L"
        elif regimen_code == "SDD": default_peak, default_trough, target_peak_info, target_trough_info = 60.0, 2.0, "50-70 mg/L (or 10x MIC)", "<5 mg/L (often undetectable)"
        elif regimen_code == "Synergy": default_peak, default_trough, target_peak_info, target_trough_info = 0.0, 0.0, "N/A", "N/A"
        elif regimen_code == "Hemodialysis": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 5.0, "Peak not routinely targeted", "<10 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 2.5, "20-30 mg/L", "<5 mg/L"

    st.info(f"Typical Targets for {regimen}: Peak {target_peak_info}, Trough {target_trough_info}")

    # MIC input for SDD regimens
    mic = 1.0 # Default MIC
    if regimen_code == "SDD":
        st.markdown("*Note: Target peak for Extended Interval is often 10x MIC.*")
        mic = st.number_input("Enter MIC (mg/L)", min_value=0.1, value=1.0, step=0.1, format="%.1f")
        recommended_peak_mic = mic * 10
        if recommended_peak_mic > default_peak:
            default_peak = recommended_peak_mic
        st.info(f"Based on MIC, target peak is ≥ {recommended_peak_mic:.1f} mg/L")

    # Allow user override of targets
    col1, col2 = st.columns(2)
    with col1:
        target_cmax = st.number_input("Target Peak (Cmax, mg/L)", value=default_peak, format="%.1f")
    with col2:
        target_cmin = st.number_input("Target Trough (Cmin, mg/L)", value=default_trough, format="%.1f")

    # Default tau based on regimen
    default_tau = 24 if regimen_code == "SDD" \
             else 8 if regimen_code == "MDD" \
             else 12 if regimen_code == "Synergy" \
             else 48 # Default for HD (q48h common) / Neonates (adjust based on age/PMA)
    tau = st.number_input("Desired Dosing Interval (hr)", min_value=4, max_value=72, value=default_tau, step=4)

    # Infusion duration
    infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)

    # Special handling notes
    if regimen_code == "Hemodialysis":
        st.info("For hemodialysis, dose is typically given post-dialysis. Interval depends on dialysis schedule (e.g., q48h, q72h). Calculations assume dose given after dialysis.")
    if regimen_code == "Neonates":
        st.warning("Neonatal PK varies significantly. These calculations use adult population estimates. CONSULT a pediatric pharmacist.")

    # --- Calculations ---
    # Calculate IBW and dosing weight (using standard formulas)
    ibw = 0.0
    if height > 152.4: # Height threshold for formulas (60 inches)
        ibw = (50 if gender == "Male" else 45.5) + 2.3 * (height / 2.54 - 60)
    ibw = max(0, ibw) # Ensure IBW is not negative

    dosing_weight = weight # Default to actual body weight
    weight_used = "Actual Body Weight"
    if ibw > 0: # Only adjust if IBW is calculable and patient is not underweight
        if weight / ibw > 1.3: # Obese threshold (e.g., >130% IBW)
            dosing_weight = ibw + 0.4 * (weight - ibw) # Adjusted BW
            weight_used = "Adjusted Body Weight"
        elif weight < ibw: # Underweight: Use Actual BW (common practice)
             dosing_weight = weight
             weight_used = "Actual Body Weight (using ABW as < IBW)"
        else: # Normal weight: Use Actual or Ideal (Using Actual here)
             dosing_weight = weight
             weight_used = "Actual Body Weight"


    st.markdown(f"**IBW:** {ibw:.1f} kg | **Dosing Weight Used:** {dosing_weight:.1f} kg ({weight_used})")

    # Population PK parameters (adjust Vd based on clinical factors if needed)
    base_vd_per_kg = 0.3 if drug == "Amikacin" else 0.26 # L/kg
    vd_adjustment = 1.0 # Default
    # Simple adjustments based on notes (can be refined)
    notes_lower = notes.lower()
    if any(term in notes_lower for term in ["ascites", "edema", "fluid overload", "anasarca", "chf exacerbation"]): vd_adjustment = 1.15; st.info("Vd increased by 15% due to potential fluid overload.")
    if any(term in notes_lower for term in ["septic", "sepsis", "burn", "icu patient"]): vd_adjustment = 1.20; st.info("Vd increased by 20% due to potential sepsis/burn/critical illness.")
    if any(term in notes_lower for term in ["dehydrated", "volume depleted"]): vd_adjustment = 0.90; st.info("Vd decreased by 10% due to potential dehydration.")

    vd = base_vd_per_kg * dosing_weight * vd_adjustment # Liters
    vd = max(1.0, vd) # Ensure Vd is at least 1L

    # Calculate Ke and Cl based on CrCl (population estimate)
    # Using published relationships might be better, e.g., Ke = a + b * CrCl
    # Simplified approach: CL (L/hr) ≈ CrCl (mL/min) * factor (e.g., 0.05 for Gentamicin)
    # Ke = CL / Vd
    cl_pop = 0.0
    if crcl > 0:
        # Example: Gentamicin CL ≈ 0.05 * CrCl (L/hr if CrCl in mL/min) - Highly simplified
        # Example: Amikacin CL might be slightly higher
        cl_factor = 0.06 if drug == "Amikacin" else 0.05
        cl_pop = cl_factor * crcl
    cl_pop = max(0.1, cl_pop) # Minimum clearance estimate

    ke = cl_pop / vd if vd > 0 else 0.01
    ke = max(0.005, ke) # Ensure ke is not excessively low

    t_half = 0.693 / ke if ke > 0 else float('inf')

    st.markdown(f"**Est. Vd:** {vd:.2f} L | **Est. Ke:** {ke:.4f} hr⁻¹ | **Est. t½:** {t_half:.2f} hr | **Est. CL:** {cl_pop:.2f} L/hr")

    # Calculate Dose needed to achieve target Cmax (using steady-state infusion equation)
    # Dose = Cmax * Vd * ke * T_inf * (1 - exp(-ke * tau)) / (1 - exp(-ke * T_inf))
    dose = 0.0
    try:
        term_ke_tinf = ke * infusion_duration
        term_ke_tau = ke * tau
        exp_ke_tinf = math.exp(-term_ke_tinf)
        exp_ke_tau = math.exp(-term_ke_tau)

        numerator = target_cmax * vd * term_ke_tinf * (1.0 - exp_ke_tau)
        denominator = (1.0 - exp_ke_tinf)

        if abs(denominator) > 1e-9:
            dose = numerator / denominator
        else: # Handle bolus case approximation if T_inf is very small
            dose = target_cmax * vd * (1.0 - exp_ke_tau) / (1.0 - exp_ke_tinf) # Recheck this derivation
            # Simpler Bolus: Dose = Cmax * Vd * (1-exp(-ke*tau)) -> This assumes Cmax is achieved instantly
            st.warning("Infusion duration is very short or Ke is very low; using approximation for dose calculation.")
            # Let's stick to the rearranged infusion formula, checking denominator

    except (OverflowError, ValueError) as math_err:
         st.error(f"Math error during dose calculation: {math_err}. Check PK parameters.")
         dose = 0 # Prevent further calculation

    # Calculate expected levels with the calculated dose
    expected_cmax = 0.0
    expected_cmin = 0.0
    if dose > 0 and vd > 0 and ke > 0 and infusion_duration > 0 and tau > 0:
        try:
            term_ke_tinf = ke * infusion_duration
            term_ke_tau = ke * tau
            exp_ke_tinf = math.exp(-term_ke_tinf)
            exp_ke_tau = math.exp(-term_ke_tau)

            common_factor = (dose / (vd * term_ke_tinf)) * (1.0 - exp_ke_tinf) / (1.0 - exp_ke_tau)

            # Cmax occurs at the end of infusion (t = infusion_duration)
            # Using the "during infusion" part of the SS equation at t=Tinf
            # C(t_inf) = common_factor * (1 - exp(-ke*t_inf)) -> simplifies to peak formula
            expected_cmax = (dose / (vd * ke * infusion_duration)) * (1 - exp_ke_tinf) / (1 - exp_ke_tau) * (1 - exp_ke_tinf) # Recheck needed
            # Let's use the simpler Cmax definition from rearrangement:
            # Cmax = Dose * (1 - exp(-ke * T_inf)) / [Vd * ke * T_inf * (1 - exp(-ke * tau))]
            denominator_cmax = vd * ke * infusion_duration * (1 - exp_ke_tau)
            if abs(denominator_cmax) > 1e-9:
                 expected_cmax = dose * (1 - exp_ke_tinf) / denominator_cmax

            # Cmin occurs at the end of the interval (t = tau)
            # Cmin = Cmax * exp(-ke * (tau - T_inf))
            expected_cmin = expected_cmax * math.exp(-ke * (tau - infusion_duration))

        except (OverflowError, ValueError) as math_err_levels:
             st.warning(f"Could not predict levels due to math error: {math_err_levels}")


    # Round the dose to a practical value (e.g., nearest 10mg or 20mg)
    rounding_base = 20 if drug == "Gentamicin" else 50 if drug == "Amikacin" else 10
    practical_dose = round(dose / rounding_base) * rounding_base
    practical_dose = max(rounding_base, practical_dose) # Ensure dose is at least the rounding base

    st.success(f"Recommended Initial Dose: **{practical_dose:.0f} mg** IV every **{tau:.0f}** hours (infused over {infusion_duration} hr)")
    st.info(f"Predicted Peak (end of infusion): ~{expected_cmax:.1f} mg/L")
    st.info(f"Predicted Trough (end of interval): ~{expected_cmin:.2f} mg/L")


    # Suggest loading dose if applicable (e.g., for SDD or severe infections)
    if regimen_code == "SDD" or "sepsis" in notes.lower() or "critical" in notes.lower():
        # Loading Dose ≈ Target Peak * Vd
        loading_dose = target_cmax * vd
        practical_loading_dose = round(loading_dose / rounding_base) * rounding_base
        practical_loading_dose = max(rounding_base, practical_loading_dose)
        st.warning(f"Consider Loading Dose: **~{practical_loading_dose:.0f} mg** IV x 1 dose to rapidly achieve target peak.")

    # Check if expected levels meet targets
    suggest_adjustment(expected_cmax, target_cmax * 0.85, target_cmax * 1.15, label="Predicted Peak") # Tighter range for check
    # Check trough against target_cmin (which is usually the max allowed trough)
    if expected_cmin > target_cmin: # Target Cmin here represents the upper limit for trough
         st.warning(f"⚠️ Predicted Trough ({expected_cmin:.2f} mg/L) may exceed target ({target_trough_info}). Consider lengthening interval if clinically appropriate.")
    else:
         st.success(f"✅ Predicted Trough ({expected_cmin:.2f} mg/L) likely below target ({target_trough_info}).")

    # Add visualization option
    if st.checkbox("Show Estimated Concentration-Time Curve"):
        if expected_cmax > 0 and expected_cmin >= 0 and ke > 0 and tau > 0:
            chart = plot_concentration_time_curve(
                peak=expected_cmax,
                trough=expected_cmin,
                ke=ke,
                tau=tau,
                t_peak=infusion_duration, # Assume peak occurs at end of infusion
                infusion_time=infusion_duration
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Cannot display curve due to invalid calculated parameters.")


    if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
        prompt = (f"Aminoglycoside Initial Dose Calculation:\n"
                  f"Drug: {drug}, Regimen Goal: {regimen}\n"
                  f"Target Peak: {target_cmax:.1f} mg/L, Target Trough: {target_cmin:.1f} mg/L (Typical: Peak {target_peak_info}, Trough {target_trough_info})\n"
                  f"Desired Interval (tau): {tau} hr, Infusion Duration: {infusion_duration} hr\n"
                  f"Calculated Dose: {practical_dose:.0f} mg\n"
                  f"Estimated PK: Vd={vd:.2f} L, Ke={ke:.4f} hr⁻¹, t½={t_half:.2f} hr, CL={cl_pop:.2f} L/hr\n"
                  f"Predicted Levels: Peak≈{expected_cmax:.1f} mg/L, Trough≈{expected_cmin:.2f} mg/L")
        interpret_with_llm(prompt, patient_data)


# ===== MODULE 2: Aminoglycoside Conventional Dosing (C1/C2) =====
def aminoglycoside_conventional_dosing(patient_data):
    st.title("📊 Aminoglycoside Dose Adjustment (using Levels)")

    drug = st.selectbox("Select Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Therapeutic Goal / Strategy", ["Traditional (Multiple Daily - MDD)", "Extended Interval (Once Daily - SDD)", "Synergy (e.g., Endocarditis)", "Hemodialysis", "Neonates (Use with caution)"])

    # Map selection to internal codes
    regimen_code = "SDD" if "Extended" in regimen \
              else "MDD" if "Traditional" in regimen \
              else "Synergy" if "Synergy" in regimen \
              else "Hemodialysis" if "Hemodialysis" in regimen \
              else "Neonates" if "Neonates" in regimen \
              else "MDD" # Default

    # --- Set target ranges based on chosen regimen and drug ---
    target_peak_info = "N/A"
    target_trough_info = "N/A"
    target_peak_min, target_peak_max = 0.0, 100.0
    target_trough_max = 100.0 # Represents the upper limit for trough

    if drug == "Gentamicin":
        if regimen_code == "MDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 5, 10, 2, "5-10 mg/L", "<2 mg/L"
        elif regimen_code == "SDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 15, 30, 1, "15-30 mg/L (or 10x MIC)", "<1 mg/L (often undetectable)"
        elif regimen_code == "Synergy": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 3, 5, 1, "3-5 mg/L", "<1 mg/L"
        elif regimen_code == "Hemodialysis": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 2, "Peak not routinely targeted", "<2 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 5, 12, 1, "5-12 mg/L", "<1 mg/L"
    else:  # Amikacin
        if regimen_code == "MDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 20, 30, 10, "20-30 mg/L", "<10 mg/L"
        elif regimen_code == "SDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 50, 70, 5, "50-70 mg/L (or 10x MIC)", "<5 mg/L (often undetectable)"
        elif regimen_code == "Synergy": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 0, "N/A", "N/A"
        elif regimen_code == "Hemodialysis": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 10, "Peak not routinely targeted", "<10 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 20, 30, 5, "20-30 mg/L", "<5 mg/L"

    st.markdown("### Target Concentration Ranges:")
    col_t1, col_t2 = st.columns(2)
    with col_t1: st.markdown(f"**Peak Target:** {target_peak_info}")
    with col_t2: st.markdown(f"**Trough Target:** {target_trough_info}")

    # MIC input for SDD regimens
    mic = 1.0 # Default MIC
    if regimen_code == "SDD":
        st.markdown("*Note: Target peak for Extended Interval is often 10x MIC.*")
        mic = st.number_input("Enter MIC (mg/L)", min_value=0.1, value=1.0, step=0.1, format="%.1f")
        recommended_peak_mic = mic * 10
        st.info(f"Based on MIC, target peak is ≥ {recommended_peak_mic:.1f} mg/L. Adjust target below if needed.")
        # Update target peak min based on MIC if higher
        target_peak_min = max(target_peak_min, recommended_peak_mic)


    st.markdown("### Dosing and Sampling Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        dose = st.number_input("Dose Administered (mg)", min_value=10.0, value = 120.0, step=5.0)
        infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    with col2:
        # Use current date as default, allow user to change if needed
        default_date = datetime.now().date()
        dose_start_datetime_dt = st.datetime_input("Date & Time of Dose Start", value=datetime.combine(default_date, time(12,0)), step=timedelta(minutes=15)) # Combine date and time(12,0)
    with col3:
        tau = st.number_input("Current Dosing Interval (hr)", min_value=4, max_value=72, value=8, step=4)

    st.markdown("### Measured Levels and Sample Times")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        c1 = st.number_input("Trough Level (C1, mg/L)", min_value=0.0, value=1.0, step=0.1, format="%.1f", help="Usually pre-dose level")
        c1_sample_datetime_dt = st.datetime_input("Date & Time of Trough Sample", value=datetime.combine(default_date, time(11,30)), step=timedelta(minutes=15)) # Default 30 min before 12pm dose
    with col_l2:
        c2 = st.number_input("Peak Level (C2, mg/L)", min_value=0.0, value=8.0, step=0.1, format="%.1f", help="Usually post-infusion level")
        c2_sample_datetime_dt = st.datetime_input("Date & Time of Peak Sample", value=datetime.combine(default_date, time(13,30)), step=timedelta(minutes=15)) # Default 30 min after 1hr infusion ends (1:30pm)


    # --- Calculate t1 and t2 relative to dose start time ---
    t1 = (c1_sample_datetime_dt - dose_start_datetime_dt).total_seconds() / 3600.0 # Time of C1 sample relative to dose start in hours
    t2 = (c2_sample_datetime_dt - dose_start_datetime_dt).total_seconds() / 3600.0 # Time of C2 sample relative to dose start in hours

    st.markdown(f"*Calculated time from dose start to Trough (C1) sample (t1): {t1:.2f} hr*")
    st.markdown(f"*Calculated time from dose start to Peak (C2) sample (t2): {t2:.2f} hr*")

    # Validate timings
    valid_times = True
    if t1 >= t2:
        st.error("❌ Trough sample time (C1) must be before Peak sample time (C2). Please check the dates and times.")
        valid_times = False
    if t2 <= infusion_duration:
        st.warning(f"⚠️ Peak sample time (t2={t2:.2f} hr) is during or before the end of infusion ({infusion_duration:.1f} hr). Calculated Cmax will be extrapolated; accuracy may be reduced.")
    if t1 > 0 and t1 < infusion_duration:
         st.warning(f"⚠️ Trough sample time (t1={t1:.2f} hr) appears to be during the infusion. Ensure C1 is a true pre-dose trough for most accurate calculations.")

    # --- Perform PK Calculations ---
    st.markdown("### Calculated Pharmacokinetic Parameters")
    results_calculated = False
    ke, t_half, vd, cl = 0, float('inf'), 0, 0
    cmax_extrapolated, cmin_extrapolated = 0, 0

    if valid_times:
        try:
            # Ensure levels are positive for log calculation
            if c1 <= 0 or c2 <= 0:
                st.error("❌ Measured levels (C1 and C2) must be greater than 0 for calculation.")

            else:
                # Calculate Ke using two levels (Sawchuk-Zaske method adaptation)
                # Assumes levels are in the elimination phase relative to each other
                delta_t = t2 - t1
                if delta_t <= 0: raise ValueError("Time difference between samples (t2-t1) must be positive.")

                # Check if both points are likely post-infusion for simple Ke calculation
                if t1 >= infusion_duration:
                     ke = (math.log(c1) - math.log(c2)) / delta_t # ln(C1/C2) / (t2-t1)
                else:
                     # If t1 is during infusion or pre-dose, Ke calculation is more complex.
                     # Using the simple formula introduces error. A Bayesian approach or iterative method is better.
                     # For this tool, we'll proceed with the simple formula but add a warning.
                     ke = (math.log(c1) - math.log(c2)) / delta_t
                     st.warning("⚠️ Ke calculated assuming log-linear decay between C1 and C2. Accuracy reduced if C1 is not post-infusion.")

                ke = max(1e-6, ke) # Ensure ke is positive and non-zero
                t_half = 0.693 / ke if ke > 0 else float('inf')

                # Extrapolate to find Cmax (at end of infusion) and Cmin (at end of interval)
                # C_t = C_known * exp(-ke * (t - t_known))
                # Cmax = C2 * exp(ke * (t2 - infusion_duration)) # Extrapolate C2 back to end of infusion
                cmax_extrapolated = c2 * math.exp(ke * (t2 - infusion_duration))

                # Cmin = Cmax_extrapolated * exp(-ke * (tau - infusion_duration)) # Trough at end of interval
                cmin_extrapolated = cmax_extrapolated * math.exp(-ke * (tau - infusion_duration))

                # Calculate Vd using Cmax and dose (steady-state infusion formula)
                # Vd = Dose * (1 - exp(-ke * T_inf)) / [Cmax * ke * T_inf * (1 - exp(-ke * tau))]
                term_inf = (1 - math.exp(-ke * infusion_duration))
                term_tau = (1 - math.exp(-ke * tau))
                denominator_vd = cmax_extrapolated * ke * infusion_duration * term_tau
                vd = 0.0
                if abs(denominator_vd) > 1e-9 and abs(term_inf) > 1e-9 : # Avoid division by zero
                    vd = (dose * term_inf) / denominator_vd
                    vd = max(1.0, vd) # Ensure Vd is at least 1L
                else:
                    st.warning("Could not calculate Vd accurately due to near-zero terms (check Ke, Tau, Infusion Duration).")

                cl = ke * vd if vd > 0 else 0.0

                st.markdown(f"**Individualized Ke:** {ke:.4f} hr⁻¹ | **t½:** {t_half:.2f} hr")
                st.markdown(f"**Est. Cmax (end of infusion):** {cmax_extrapolated:.1f} mg/L | **Est. Cmin (end of interval):** {cmin_extrapolated:.2f} mg/L")
                if vd > 0:
                     st.markdown(f"**Est. Vd:** {vd:.2f} L | **Est. CL:** {cl:.2f} L/hr")
                else:
                     st.markdown("**Est. Vd & CL:** Could not be calculated accurately.")

                results_calculated = True

                # --- Dose Recommendation ---
                st.markdown("### Dose Adjustment Recommendation")
                if vd <= 0 or ke <=0:
                     st.warning("Cannot calculate new dose recommendation due to invalid PK parameters.")
                else:
                    # Ask for desired target levels (default to mid-point of range or target min)
                    default_desired_peak = target_peak_min if regimen_code == "SDD" else (target_peak_min + target_peak_max) / 2
                    desired_peak = st.number_input("Desired Target    peak_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Peak.*?([\d.,]+)\s*mg/L", prompt)
    auc_val = extract_float(r"(?:Estimated|Predicted)\s+AUC24.*?([\d.,]+)\s*mg.hr/L", prompt)

    # Extract targets
    target_auc_str = extract_string(r"Target\s+AUC24.*?(\d+\s*-\s*\d+)\s*mg.hr/L", prompt, "400-600")
    target_trough_str = extract_string(r"(?:Target|Secondary Target)\s+Trough.*?([\d.]+\s*-\s*[\d.]+)\s*mg/L", prompt, "10-15")

    # Extract current/new regimen details
    current_dose_interval = extract_float(r"Current\s+Regimen.*?([\d,]+)\s*mg.*?q\s*(\d+)", prompt)
    current_interval = extract_float(r"Current\s+Regimen.*?q\s*(\d+)", prompt)
    new_dose_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?([\d,]+)\s*mg.*?q\s*(\d+)", prompt)
    new_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?q\s*(\d+)", prompt)


    # Parse target ranges
    auc_target_min, auc_target_max = 400, 600
    auc_match = re.match(r"(\d+)\s*-\s*(\d+)", target_auc_str)
    if auc_match: auc_target_min, auc_target_max = int(auc_match.group(1)), int(auc_match.group(2))
    auc_target_formatted = f"{auc_target_min}-{auc_target_max} mg·hr/L"

    trough_target_min, trough_target_max = 10, 15
    trough_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_trough_str)
    if trough_match:
        try:
            trough_target_min = float(trough_match.group(1))
            trough_target_max = float(trough_match.group(2))
        except ValueError: pass
    trough_target_formatted = f"{trough_target_min:.1f}-{trough_target_max:.1f} mg/L"


    # Check if essential values for assessment were extracted
    if trough_val is None and auc_val is None:
        return "Insufficient level data (Trough or AUC) in prompt for standardized vancomycin interpretation."

    # --- Start Interpretation Logic ---
    levels_data = []
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    status = "assessment uncertain" # Default status

    # Assess Trough Level
    trough_status = "N/A"
    if trough_val is not None:
        if trough_val < trough_target_min: trough_status = "below"
        elif trough_val > trough_target_max: trough_status = "above"
        else: trough_status = "within"
        levels_data.append(("Trough", trough_val, trough_target_formatted, trough_status))

    # Assess AUC Level
    auc_status = "N/A"
    if auc_val is not None:
        if auc_val < auc_target_min: auc_status = "below"
        elif auc_val > auc_target_max: auc_status = "above"
        else: auc_status = "within"
        levels_data.append(("AUC24", auc_val, auc_target_formatted, auc_status))

    # Assess Peak Level (if available)
    peak_status = "N/A"
    if peak_val is not None:
        # Define peak range based on empiric vs definitive therapy
        # Assuming trough level helps determine empiric vs definitive
        if trough_val is not None and trough_val <= 15:  # Likely empiric therapy
            peak_target_min, peak_target_max = 20, 30
            peak_target_formatted = f"{peak_target_min}-{peak_target_max} mg/L (Empiric)"
        else:  # Likely definitive therapy
            peak_target_min, peak_target_max = 25, 40
            peak_target_formatted = f"{peak_target_min}-{peak_target_max} mg/L (Definitive)"
        
        if peak_val < peak_target_min: peak_status = "below"
        elif peak_val > peak_target_max: peak_status = "above"
        else: peak_status = "within"
        levels_data.append(("Peak", peak_val, peak_target_formatted, peak_status))


    # Determine overall assessment status (prioritize AUC, then Trough)
    if auc_status == "within" and trough_status != "above": status = "appropriately dosed (AUC target met)"
    elif auc_status == "within" and trough_status == "above": status = "potentially overdosed (AUC ok, trough high)"
    elif auc_status == "below": status = "underdosed (AUC below target)"
    elif auc_status == "above": status = "overdosed (AUC above target)"
    elif auc_status == "N/A": # If AUC not available, use trough
         if trough_status == "within": status = "likely appropriately dosed (trough target met)"
         elif trough_status == "below": status = "likely underdosed (trough below target)"
         elif trough_status == "above": status = "likely overdosed (trough above target)"


    # Generate recommendations based on status
    if "appropriately dosed" in status:
        dosing_recs.append("CONTINUE current regimen.")
        if current_dose_interval and current_interval:
             dosing_recs.append(f"MAINTAIN {current_dose_interval:.0f}mg q{current_interval:.0f}h.")
        monitoring_recs.append("MONITOR renal function per protocol (e.g., 2-3 times weekly).")
        monitoring_recs.append("REPEAT levels if clinical status or renal function changes significantly.")
        if status == "potentially overdosed (AUC ok, trough high)": # Add caution if trough high despite AUC ok
             cautions.append("Trough is elevated, increasing potential nephrotoxicity risk despite acceptable AUC. Monitor renal function closely.")
             monitoring_recs.append("Consider rechecking trough sooner if renal function declines.")

    else: # Underdosed or Overdosed
        if "underdosed" in status:
             dosing_recs.append("INCREASE dose and/or shorten interval to achieve target AUC.")
             monitoring_recs.append("RECHECK levels after 3-5 doses of new regimen (allow steady state).")
        elif "overdosed" in status:
             if trough_val is not None and trough_val > 25: # Significantly high trough
                 dosing_recs.append("HOLD next dose(s) until trough is acceptable (e.g., < 20 mg/L).")
                 cautions.append("Significantly elevated trough increases nephrotoxicity risk.")
             dosing_recs.append("DECREASE dose and/or lengthen interval to achieve target AUC.")
             monitoring_recs.append("RECHECK levels within 24-48 hours after adjustment (or before next dose if interval long).")
             monitoring_recs.append("MONITOR renal function daily until stable.")

        # Suggest new regimen if provided in prompt
        if new_dose_interval and new_interval:
             # Suggest practical regimens based on new TDD
             new_tdd_calc = new_dose_interval * (24 / new_interval)
             suggested_regimens = []
             for practical_interval_opt in [8, 12, 24, 36, 48]: # Common intervals
                 dose_per_interval_opt = new_tdd_calc / (24 / practical_interval_opt)
                 # Round dose per interval to nearest 250mg
                 rounded_dose_opt = round(dose_per_interval_opt / 250) * 250
                 if rounded_dose_opt > 0:
                     # Check if this option is close to the suggested one
                     is_suggested = abs(practical_interval_opt - new_interval) < 1 and abs(rounded_dose_opt - new_dose_interval) < 125
                     prefix = "➡️" if is_suggested else "  -"
                     suggested_regimens.append(f"{prefix} {rounded_dose_opt:.0f}mg q{practical_interval_opt}h (approx. {rounded_dose_opt * (24/practical_interval_opt):.0f}mg/day)")

             if suggested_regimens:
                 dosing_recs.append(f"ADJUST regimen towards target AUC ({auc_target_formatted}). Consider practical options:")
                 # Add the explicitly suggested regimen first if found
                 explicit_suggestion = f"{new_dose_interval:.0f}mg q{new_interval:.0f}h"
                 if not any(explicit_suggestion in reg for reg in suggested_regimens):
                      dosing_recs.append(f"➡️ {explicit_suggestion} (Calculated)") # Add if not already covered by rounding
                 for reg in suggested_regimens:
                     dosing_recs.append(reg)

             else: # Fallback if no practical options generated
                  dosing_recs.append(f"ADJUST regimen to {new_dose_interval:.0f}mg q{new_interval:.0f}h as calculated.")
        else: # If no new dose calculated in prompt
             dosing_recs.append("ADJUST regimen based on clinical judgment and estimated PK to achieve target AUC.")


    # Add renal function caution if relevant
    if crcl is not None:
        renal_status = ""
        if crcl < 15: renal_status = "Kidney Failure"
        elif crcl < 30: renal_status = "Severe Impairment"
        elif crcl < 60: renal_status = "Moderate Impairment"
        elif crcl < 90: renal_status = "Mild Impairment"

        if crcl < 60: # Add caution for moderate to severe impairment
            cautions.append(f"{renal_status} (CrCl: {crcl:.1f} mL/min). Increased risk of accumulation and toxicity. Monitor levels and renal function closely.")
            if "overdosed" in status or (trough_val is not None and trough_val > target_trough_max):
                 monitoring_recs.append("MONITOR renal function at least daily.")
            else:
                 monitoring_recs.append("MONITOR renal function frequently (e.g., every 1-2 days).")

    cautions.append("Ensure appropriate infusion duration (e.g., ≥ 1 hour per gram, max rate 1g/hr) to minimize infusion reactions.")
    cautions.append("Consider potential drug interactions affecting vancomycin clearance or toxicity (e.g., piperacillin-tazobactam, loop diuretics, other nephrotoxins).")


    return levels_data, status, dosing_recs, monitoring_recs, cautions


def generate_aminoglycoside_interpretation(prompt, crcl=None):
    """
    Generate standardized aminoglycoside interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract drug name
    drug_match = re.search(r"Drug:\s*(Gentamicin|Amikacin)", prompt, re.IGNORECASE)
    drug_name = drug_match.group(1).lower() if drug_match else "aminoglycoside"

    # Extract levels (measured or estimated)
    peak_val = extract_float(r"(?:Measured|Estimated|Predicted|Est\.)\s+(?:Peak|Cmax).*?([\d.,]+)\s*mg/L", prompt)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted|Est\.)\s+(?:Trough|Cmin|C1).*?([\d.,]+)\s*mg/L", prompt) # Allow C1 as trough

    # Extract targets
    target_peak_str = extract_string(r"Target\s+Peak.*?((?:[\d.]+\s*-\s*[\d.]+|[<>]?\s*[\d.]+|N/A|Not routinely targeted))\s*mg/L", prompt, "N/A")
    target_trough_str = extract_string(r"Target\s+Trough.*?((?:[\d.]+\s*-\s*[\d.]+|[<>]?\s*[\d.]+|N/A))\s*mg/L", prompt, "N/A")

    # Extract current/new regimen details
    current_dose = extract_float(r"Current\s+Regimen.*?([\d,]+)\s*mg", prompt)
    current_interval = extract_float(r"Current\s+Regimen.*?q\s*(\d+)", prompt)
    new_dose = extract_float(r"(?:Suggested|New)\s+Regimen.*?([\d,]+)\s*mg", prompt)
    new_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?q\s*(\d+)", prompt)


    # --- Parse Target Ranges ---
    peak_min, peak_max = 0, 100 # Default wide range
    trough_limit_type = "max" # Assume target is '< max' by default
    trough_max = 100 # Default wide range

    # Parse Peak Target String
    if "N/A" in target_peak_str or "not targeted" in target_peak_str:
        peak_min, peak_max = None, None # Indicate not applicable
    else:
        peak_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_peak_str)
        if peak_match:
            try: peak_min, peak_max = float(peak_match.group(1)), float(peak_match.group(2))
            except ValueError: pass # Keep defaults if parsing fails

    # Parse Trough Target String
    if "N/A" in target_trough_str:
        trough_max = None # Indicate not applicable
    else:
        trough_match_less = re.match(r"<\s*([\d.]+)", target_trough_str)
        trough_match_range = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_trough_str)
        if trough_match_less:
            try: trough_max = float(trough_match_less.group(1)); trough_limit_type = "max"
            except ValueError: pass
        elif trough_match_range: # Handle if a range is given for trough (less common for amino)
             try: trough_max = float(trough_match_range.group(2)); trough_limit_type = "range"; trough_min = float(trough_match_range.group(1))
             except ValueError: pass # Default to max limit if range parsing fails


    # Check if essential level values were extracted
    if peak_val is None or trough_val is None:
        # Allow interpretation if only trough is available for HD patients
        if not ("Hemodialysis" in prompt and trough_val is not None):
             return "Insufficient level data (Peak or Trough) in prompt for standardized aminoglycoside interpretation."


    # --- Start Interpretation Logic ---
    levels_data = []
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    status = "assessment uncertain" # Default status

    # Assess Peak Level
    peak_status = "N/A"
    if peak_min is not None and peak_max is not None and peak_val is not None:
        if peak_val < peak_min: peak_status = "below"
        elif peak_val > peak_max: peak_status = "above"
        else: peak_status = "within"
        levels_data.append(("Peak", peak_val, target_peak_str, peak_status))
    elif peak_val is not None: # If target is N/A but value exists
         levels_data.append(("Peak", peak_val, target_peak_str, "N/A"))


    # Assess Trough Level
    trough_status = "N/A"
    if trough_max is not None and trough_val is not None:
        if trough_limit_type == "max":
            if trough_val >= trough_max: trough_status = "above" # At or above the max limit
            else: trough_status = "within" # Below the max limit
        elif trough_limit_type == "range":
             if trough_val < trough_min: trough_status = "below" # Below the range min (unlikely target for amino)
             elif trough_val > trough_max: trough_status = "above" # Above the range max
             else: trough_status = "within"
        levels_data.append(("Trough", trough_val, target_trough_str, trough_status))
    elif trough_val is not None: # If target is N/A but value exists
        levels_data.append(("Trough", trough_val, target_trough_str, "N/A"))


    # Determine overall assessment status
    # Prioritize avoiding toxicity (high trough), then achieving efficacy (adequate peak)
    if trough_status == "above":
        status = "potentially toxic (elevated trough)"
        if peak_status == "below": status = "ineffective and potentially toxic" # Worst case
    elif peak_status == "below":
        status = "subtherapeutic (inadequate peak)"
    elif peak_status == "above": # Peak high, trough ok
        status = "potentially supratherapeutic (high peak)"
    elif peak_status == "within" and trough_status == "within":
        status = "appropriately dosed"
    elif peak_status == "N/A" and trough_status == "within": # e.g., HD patient trough ok
         status = "likely appropriate (trough acceptable)"
    elif peak_status == "N/A" and trough_status == "above": # e.g., HD patient trough high
         status = "potentially toxic (elevated trough)"


    # Generate recommendations
    if "appropriately dosed" in status or "likely appropriate" in status :
        dosing_recs.append("CONTINUE current regimen.")
        if current_dose and current_interval: dosing_recs.append(f"MAINTAIN {current_dose:.0f}mg q{current_interval:.0f}h.")
        monitoring_recs.append("MONITOR renal function regularly (e.g., 2-3 times weekly or per HD schedule).")
        monitoring_recs.append("REPEAT levels if clinical status, renal function, or dialysis schedule changes.")
    elif status == "assessment not applicable": # Synergy Amikacin
         dosing_recs.append("Follow specific institutional protocol for Synergy Amikacin dosing.")
         monitoring_recs.append("MONITOR renal function and clinical status.")
    else: # Adjustments needed
        if status == "ineffective and potentially toxic":
             dosing_recs.append("HOLD next dose(s).")
             dosing_recs.append("INCREASE dose AND EXTEND interval significantly once resumed.")
             monitoring_recs.append("RECHECK levels (peak & trough) before resuming and after 2-3 doses of new regimen.")
             cautions.append("High risk of toxicity and low efficacy with current levels.")
        elif status == "subtherapeutic (inadequate peak)":
             dosing_recs.append("INCREASE dose.")
             dosing_recs.append("MAINTAIN current interval (unless trough also borderline high).")
             monitoring_recs.append("RECHECK peak and trough after 1-2 doses of new regimen.")
             cautions.append("Inadequate peak may compromise efficacy, especially for gram-negative infections.")
        elif status == "potentially toxic (elevated trough)":
             dosing_recs.append("EXTEND dosing interval.")
             dosing_recs.append("MAINTAIN current dose amount (or consider slight reduction if peak also high/borderline).")
             monitoring_recs.append("RECHECK trough level before next scheduled dose.")
             cautions.append("Elevated trough increases risk of nephrotoxicity and ototoxicity. Hold dose if trough significantly elevated.")
        elif status == "potentially supratherapeutic (high peak)": # High peak, trough ok
             dosing_recs.append("DECREASE dose.")
             dosing_recs.append("MAINTAIN current interval.")
             monitoring_recs.append("RECHECK peak and trough after 1-2 doses of new regimen.")
             cautions.append("Elevated peak may increase toxicity risk slightly, though trough is primary driver. Ensure trough remains acceptable.")

        # Suggest new regimen if provided in prompt
        if new_dose and new_interval:
             # Round new dose to nearest 10mg or 20mg
             rounding = 20 if drug_name == "gentamicin" else 50 if drug_name == "amikacin" else 10
             practical_new_dose = round(new_dose / rounding) * rounding
             if practical_new_dose > 0:
                 dosing_recs.append(f"Consider adjusting regimen towards: {practical_new_dose:.0f}mg q{new_interval:.0f}h.")
        else:
             dosing_recs.append("ADJUST regimen based on clinical judgment and estimated PK to achieve target levels.")


    # Add general monitoring and cautions
    monitoring_recs.append("MONITOR renal function (SCr, BUN, UOP) at least 2-3 times weekly, or more frequently if unstable, trough elevated, or on concomitant nephrotoxins.")
    monitoring_recs.append("MONITOR for signs/symptoms of nephrotoxicity (rising SCr, decreased UOP) and ototoxicity (hearing changes, tinnitus, vertigo).")
    cautions.append(f"{drug_name.capitalize()} carries risk of nephrotoxicity and ototoxicity.")
    cautions.append("Risk increases with prolonged therapy (>7-10 days), pre-existing renal impairment, high troughs, large cumulative dose, and concomitant nephrotoxins (e.g., vancomycin, diuretics, contrast).")
    if crcl is not None:
         renal_status = ""
         if crcl < 15: renal_status = "Kidney Failure"
         elif crcl < 30: renal_status = "Severe Impairment"
         elif crcl < 60: renal_status = "Moderate Impairment"
         elif crcl < 90: renal_status = "Mild Impairment"
         if crcl < 60:
             cautions.append(f"{renal_status} (CrCl: {crcl:.1f} mL/min) significantly increases toxicity risk. Adjust dose/interval carefully and monitor very closely.")


    return levels_data, status, dosing_recs, monitoring_recs, cautions


# ===== SIDEBAR: NAVIGATION AND PATIENT INFO =====
def setup_sidebar_and_navigation():
    st.sidebar.title("📊 Navigation")
    # Sidebar radio for selecting the module
    page = st.sidebar.radio("Select Module", [
        "Aminoglycoside: Initial Dose",
        "Aminoglycoside: Conventional Dosing (C1/C2)",
        "Vancomycin AUC-based Dosing"
    ])

    st.sidebar.title("🩺 Patient Demographics")
    # ADDED Patient ID and Ward
    patient_id = st.sidebar.text_input("Patient ID", value="N/A")
    ward = st.sidebar.text_input("Ward", value="N/A")
    # --- Existing fields ---
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=65)
    height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=165)
    weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1, format="%.1f")
    serum_cr = st.sidebar.number_input("Serum Creatinine (µmol/L)", min_value=10.0, max_value=2000.0, value=90.0, step=1.0)

    # Calculate Cockcroft-Gault Creatinine Clearance
    crcl = 0.0 # Default value
    renal_function = "N/A"
    if age > 0 and weight > 0 and serum_cr > 0: # Avoid division by zero or negative age
        # Cockcroft-Gault Formula
        crcl_factor = (140 - age) * weight
        crcl_gender_mult = 1.23 if gender == "Male" else 1.04
        crcl = (crcl_factor * crcl_gender_mult) / serum_cr
        crcl = max(0, crcl) # Ensure CrCl is not negative

        # Renal function category based on CrCl
        if crcl >= 90: renal_function = "Normal (≥90)"
        elif crcl >= 60: renal_function = "Mild Impairment (60-89)"
        elif crcl >= 30: renal_function = "Moderate Impairment (30-59)"
        elif crcl >= 15: renal_function = "Severe Impairment (15-29)"
        else: renal_function = "Kidney Failure (<15)"

    with st.sidebar.expander("Creatinine Clearance (Cockcroft-Gault)", expanded=True):
        if age > 0 and weight > 0 and serum_cr > 0:
            st.success(f"CrCl: {crcl:.1f} mL/min")
            st.info(f"Renal Function: {renal_function}")
        else:
            st.warning("Enter valid Age (>0), Weight (>0), and SCr (>0) to calculate CrCl.")


    st.sidebar.title("🩺 Clinical Information")
    clinical_diagnosis = st.sidebar.text_input("Diagnosis / Indication", placeholder="e.g., Pneumonia, Sepsis")
    current_dose_regimen = st.sidebar.text_area("Current Dosing Regimen", value="1g IV q12h", placeholder="e.g., Gentamicin 120mg IV q8h")
    notes = st.sidebar.text_area("Other Clinical Notes", value="No known allergies.", placeholder="e.g., Fluid status, interacting meds")

    # UPDATED clinical_summary
    clinical_summary = (
        f"Patient ID: {patient_id}, Ward: {ward}\n"
        f"Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} cm\n"
        f"SCr: {serum_cr} µmol/L\n"
        f"Diagnosis: {clinical_diagnosis}\n"
        f"Renal function: {renal_function} (Est. CrCl: {crcl:.1f} mL/min)\n"
        f"Current regimen: {current_dose_regimen}\n"
        f"Notes: {notes}"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Antimicrobial TDM App v1.2**

    Developed for therapeutic drug monitoring of antimicrobials.

    Provides PK estimates, AUC calculations, and dosing recommendations
    for vancomycin and aminoglycosides. Includes optional LLM interpretation.

    **Disclaimer:** This tool assists clinical decision making but does not replace
    professional judgment. Verify all calculations and recommendations.
    """)

    # Return all the data entered in the sidebar
    return {
        'page': page,
        'patient_id': patient_id, # Added
        'ward': ward,           # Added
        'gender': gender,
        'age': age,
        'height': height,
        'weight': weight,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen,
        'notes': notes,
        'clinical_summary': clinical_summary # Updated summary string
    }

# ===== MODULE 1: Aminoglycoside Initial Dose =====
def aminoglycoside_initial_dose(patient_data):
    st.title("🧮 Aminoglycoside Initial Dose (Population PK)")

    # Unpack patient data for easier access
    gender = patient_data['gender']
    age = patient_data['age']
    height = patient_data['height']
    weight = patient_data['weight']
    crcl = patient_data['crcl']
    notes = patient_data['notes']

    drug = st.selectbox("Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Dosing Strategy / Goal", ["Extended Interval (Once Daily - SDD)", "Traditional (Multiple Daily - MDD)", "Synergy (e.g., Endocarditis)", "Hemodialysis", "Neonates (Use with caution)"])

    # Map selection to internal codes
    regimen_code = "SDD" if "Extended" in regimen \
              else "MDD" if "Traditional" in regimen \
              else "Synergy" if "Synergy" in regimen \
              else "Hemodialysis" if "Hemodialysis" in regimen \
              else "Neonates" if "Neonates" in regimen \
              else "MDD" # Default

    # --- Set default target ranges based on regimen and drug ---
    target_peak_info = "N/A"
    target_trough_info = "N/A"
    default_peak = 0.0
    default_trough = 0.0

    if drug == "Gentamicin":import streamlit as st
import numpy as np
import math
import openai
import pandas as pd
import altair as alt
import base64
from datetime import datetime, time, timedelta # Added time and timedelta

# Optional imports - Bayesian functionality
try:
    import scipy.optimize as optimize
    from scipy.stats import norm
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scipy not installed. Bayesian estimation will not be available.")

# Optional imports - FAISS for guideline embedding (if needed)
try:
    import faiss
    import pickle
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Guideline embeddings will not be available.")

st.set_page_config(page_title="Antimicrobial TDM App", layout="wide")

# ===== API CONFIGURATION =====
# Securely access the API key from streamlit secrets
try:
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.

    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== LOAD EMBEDDED GUIDELINE =====
@st.cache_resource
def load_guideline_embeddings():
    if not FAISS_AVAILABLE:
        return None, ["FAISS not installed. Guideline embeddings not available."]

    # Replace this with your actual FAISS index and guideline chunks
    index = faiss.IndexFlatL2(768)  # Dummy placeholder index
    chunks = ["Guideline excerpt 1", "Guideline excerpt 2", "Guideline excerpt 3"]
    return index, chunks

guideline_index, guideline_chunks = None, []
if FAISS_AVAILABLE:
    guideline_index, guideline_chunks = load_guideline_embeddings()

# ===== Global Practical Dosing Regimens =====
practical_intervals = "6hr, 8hr, 12hr, 24hr, every other day"

# ===== HELPER FUNCTION: Updated Practical Dose Adjustment =====
def suggest_adjustment(parameter, target_min, target_max, label="Parameter", intervals=practical_intervals):
    # Ensure targets are valid numbers before comparison
    if isinstance(target_min, (int, float)) and isinstance(target_max, (int, float)) and isinstance(parameter, (int, float)):
        if parameter < target_min:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is low. Target: {target_min:.1f}-{target_max:.1f}. Consider increasing dose or shortening interval ({intervals}).")
        elif parameter > target_max:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is high. Target: {target_min:.1f}-{target_max:.1f}. Consider reducing dose or lengthening interval ({intervals}).")
        else:
            st.success(f"✅ {label} ({parameter:.1f}) is within target range ({target_min:.1f}-{target_max:.1f}).")
    else:
        st.info(f"{label}: {parameter}. Target range: {target_min}-{target_max}. (Comparison skipped due to non-numeric values).")


# ===== PDF GENERATION FUNCTIONS (REMOVED) =====
# create_recommendation_pdf, get_pdf_download_link, display_pdf_download_button functions removed.

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization

    Parameters:
    - peak: Peak concentration (mg/L)
    - trough: Trough concentration (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - t_peak: Time to peak after start of infusion (hr) - assumed end of infusion
    - infusion_time: Duration of infusion (hr)

    Returns:
    - Altair chart object
    """
    # Generate time points for the curve
    times = np.linspace(0, tau*1.5, 150)  # Generate points for 1.5 intervals to show next dose

    # Generate concentrations for each time point using steady-state equations
    concentrations = []
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * T_inf)) * exp(-ke * (t - T_inf)) / (1 - exp(-ke * tau)) -- Post-infusion
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * t)) / (1 - exp(-ke * tau)) -- During infusion (simplified, assumes Cmin=0 start)
    # Let's use the provided peak and trough which represent Cmax (at t=infusion_time) and Cmin (at t=tau)

    for t_cycle in np.linspace(0, tau*1.5, 150): # Iterate through time points
        # Determine concentration based on time within the dosing cycle (modulo tau)
        t = t_cycle % tau
        num_cycles = int(t_cycle // tau) # Which cycle we are in (0, 1, ...)

        conc = 0
        if t <= infusion_time:
            # During infusion: Assume linear rise from previous trough to current peak
            # This is an approximation but visually represents the infusion period
            conc = trough + (peak - trough) * (t / infusion_time)
        else:
            # After infusion: Exponential decay from peak
            time_since_peak = t - infusion_time # Time elapsed since the peak concentration (end of infusion)
            conc = peak * math.exp(-ke * time_since_peak)

        concentrations.append(max(0, conc)) # Ensure concentration doesn't go below 0


    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Time (hr)': times,
        'Concentration (mg/L)': concentrations
    })

    # --- Create Target Bands ---
    target_bands = []
    # Determine drug type based on typical levels for band coloring
    if peak > 45 or trough > 20:  # Likely vancomycin
        # Vancomycin Peak Target - Empiric vs Definitive
        if trough <= 15:  # Likely empiric (target trough 10-15)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [20], 'y2': [30]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Empiric)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [10], 'y2': [15]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Empiric)")))
        else:  # Likely definitive (target trough 15-20)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [25], 'y2': [40]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Definitive)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [15], 'y2': [20]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Definitive)")))
    else:  # Likely aminoglycoside (e.g., Gentamicin)
        # Aminoglycoside Peak Target (e.g., 5-10 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [5], 'y2': [10]}))
                           .mark_rect(opacity=0.15, color='lightblue')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Amino)")))
        # Aminoglycoside Trough Target (e.g., <2 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [0], 'y2': [2]}))
                           .mark_rect(opacity=0.15, color='lightgreen')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Amino)")))


    # --- Create Concentration Line ---
    line = alt.Chart(df).mark_line(color='firebrick').encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)', scale=alt.Scale(zero=True)), # Ensure Y axis starts at 0
        tooltip=['Time (hr)', alt.Tooltip('Concentration (mg/L)', format=".1f")]
    )

    # --- Add Vertical Lines for Key Events ---
    vertical_lines_data = []
    # Mark end of infusion for each cycle shown
    for i in range(int(tau*1.5 / tau) + 1):
        inf_end_time = i * tau + infusion_time
        if inf_end_time <= tau*1.5:
             vertical_lines_data.append({'Time': inf_end_time, 'Event': 'Infusion End'})
    # Mark start of next dose for each cycle shown
    for i in range(1, int(tau*1.5 / tau) + 1):
         dose_time = i * tau
         if dose_time <= tau*1.5:
              vertical_lines_data.append({'Time': dose_time, 'Event': 'Next Dose'})

    vertical_lines_df = pd.DataFrame(vertical_lines_data)

    vertical_rules = alt.Chart(vertical_lines_df).mark_rule(strokeDash=[4, 4]).encode(
        x='Time',
        color=alt.Color('Event', scale=alt.Scale(domain=['Infusion End', 'Next Dose'], range=['gray', 'black'])),
        tooltip=['Event', 'Time']
    )

    # --- Combine Charts ---
    chart = alt.layer(*target_bands, line, vertical_rules).properties(
        width=alt.Step(4), # Adjust width automatically
        height=400,
        title=f'Estimated Concentration-Time Profile (Tau={tau} hr)'
    ).interactive() # Make chart interactive (zoom/pan)

    return chart


# ===== VANCOMYCIN AUC CALCULATION (TRAPEZOIDAL METHOD) =====
def calculate_vancomycin_auc_trapezoidal(cmax, cmin, ke, tau, infusion_duration):
    """
    Calculate vancomycin AUC24 using the linear-log trapezoidal method.
    
    This method is recommended for vancomycin TDM as per the guidelines.
    
    Parameters:
    - cmax: Max concentration at end of infusion (mg/L)
    - cmin: Min concentration at end of interval (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - infusion_duration: Duration of infusion (hr)
    
    Returns:
    - AUC24: 24-hour area under the curve (mg·hr/L)
    """
    # Calculate concentration at start of infusion (C0)
    c0 = cmax * math.exp(ke * infusion_duration)
    
    # Calculate AUC during infusion phase (linear trapezoid)
    auc_inf = infusion_duration * (c0 + cmax) / 2
    
    # Calculate AUC during elimination phase (log trapezoid)
    if ke > 0 and cmax > cmin:
        auc_elim = (cmax - cmin) / ke
    else:
        # Fallback to linear trapezoid if ke is very small
        auc_elim = (tau - infusion_duration) * (cmax + cmin) / 2
    
    # Calculate total AUC for one dosing interval
    auc_interval = auc_inf + auc_elim
    
    # Convert to AUC24
    auc24 = auc_interval * (24 / tau)
    
    return auc24

# ===== BAYESIAN PARAMETER ESTIMATION =====
def bayesian_parameter_estimation(measured_levels, sample_times, dose, tau, weight, age, crcl, gender):
    """
    Bayesian estimation of PK parameters based on measured levels

    Parameters:
    - measured_levels: List of measured drug concentrations (mg/L)
    - sample_times: List of times when samples were taken (hr after dose start)
    - dose: Dose administered (mg)
    - tau: Dosing interval (hr)
    - weight: Patient weight (kg)
    - age: Patient age (years)
    - crcl: Creatinine clearance (mL/min)
    - gender: Patient gender ("Male" or "Female")

    Returns:
    - Dictionary with estimated PK parameters
    """
    if not BAYESIAN_AVAILABLE:
        st.error("Bayesian estimation requires scipy. Please install it with 'pip install scipy'")
        return None

    # Prior population parameters for vancomycin (adjust if needed for aminoglycosides)
    # Mean values
    vd_pop_mean = 0.7  # L/kg (Vancomycin specific, adjust for aminoglycosides if used)
    ke_pop_mean = 0.00083 * crcl + 0.0044 # hr^-1 (Vancomycin specific - ensure crcl is used correctly)
    ke_pop_mean = max(0.01, ke_pop_mean) # Ensure Ke isn't too low

    # Standard deviations for population parameters
    vd_pop_sd = 0.2  # L/kg
    ke_pop_sd = 0.05 # Increased SD for Ke prior to allow more flexibility

    # Define objective function to minimize (negative log likelihood)
    def objective_function(params):
        vd_ind, ke_ind = params # Individual parameters to estimate
        vd_total = vd_ind * weight

        # Calculate expected concentrations at sample times using steady-state infusion model
        expected_concs = []
        infusion_time = 1.0 # Assume 1 hour infusion, make adjustable if needed

        for t in sample_times:
            # Steady State Concentration Equation (1-compartment, intermittent infusion)
            term_dose_vd = dose / vd_total
            term_ke_tinf = ke_ind * infusion_time
            term_ke_tau = ke_ind * tau

            try:
                exp_ke_tinf = math.exp(-term_ke_tinf)
                exp_ke_tau = math.exp(-term_ke_tau)

                if abs(1.0 - exp_ke_tau) < 1e-9: # Avoid division by zero if tau is very long or ke very small
                    # Handle as if continuous infusion or single dose if tau is effectively infinite
                    conc = 0 # Simplified - needs better handling for edge cases
                else:
                    common_factor = (term_dose_vd / term_ke_tinf) * (1.0 - exp_ke_tinf) / (1.0 - exp_ke_tau)

                    if t <= infusion_time: # During infusion phase
                        conc = common_factor * (1.0 - math.exp(-ke_ind * t))
                    else: # Post-infusion phase
                        conc = common_factor * math.exp(-ke_ind * (t - infusion_time))

            except OverflowError:
                 conc = float('inf') # Handle potential overflow with large ke/t values
            except ValueError:
                 conc = 0 # Handle math domain errors

            expected_concs.append(max(0, conc)) # Ensure non-negative

        # Calculate negative log likelihood
        # Measurement error model (e.g., proportional + additive)
        # sd = sqrt(sigma_add^2 + (sigma_prop * expected_conc)^2)
        sigma_add = 1.0  # Additive SD (mg/L)
        sigma_prop = 0.1 # Proportional SD (10%)
        nll = 0
        for i in range(len(measured_levels)):
            expected = expected_concs[i]
            measurement_sd = math.sqrt(sigma_add**2 + (sigma_prop * expected)**2)
            if measurement_sd < 1e-6: measurement_sd = 1e-6 # Prevent division by zero in logpdf

            # Add contribution from measurement likelihood
            # Use logpdf for robustness, especially with low concentrations
            nll += -norm.logpdf(measured_levels[i], loc=expected, scale=measurement_sd)

        # Add contribution from parameter priors (log scale often more stable for Ke)
        # Prior for Vd (Normal)
        nll += -norm.logpdf(vd_ind, loc=vd_pop_mean, scale=vd_pop_sd)
        # Prior for Ke (Log-Normal might be better, but using Normal for simplicity)
        nll += -norm.logpdf(ke_ind, loc=ke_pop_mean, scale=ke_pop_sd)

        # Penalize non-physical parameters slightly if optimization strays
        if vd_ind <= 0 or ke_ind <= 0:
             nll += 1e6 # Add large penalty

        return nll

    # Initial guess based on population values
    initial_params = [vd_pop_mean, ke_pop_mean]

    # Parameter bounds (physical constraints)
    bounds = [(0.1, 2.5), (0.001, 0.5)]  # Reasonable bounds for Vd (L/kg) and Ke (hr^-1)

    # Perform optimization using a robust method
    try:
        result = optimize.minimize(
            objective_function,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B', # Suitable for bound constraints
            options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 500} # Adjust tolerances/iterations
        )
    except Exception as e:
         st.error(f"Optimization failed: {e}")
         return None

    if not result.success:
        st.warning(f"Bayesian optimization did not converge: {result.message} (Function evaluations: {result.nfev})")
        # Optionally return population estimates or None
        return None # Indicate failure

    # Extract optimized parameters
    vd_opt_kg, ke_opt = result.x
    # Ensure parameters are within bounds post-optimization (should be handled by L-BFGS-B, but double-check)
    vd_opt_kg = max(bounds[0][0], min(bounds[0][1], vd_opt_kg))
    ke_opt = max(bounds[1][0], min(bounds[1][1], ke_opt))

    vd_total_opt = vd_opt_kg * weight
    cl_opt = ke_opt * vd_total_opt
    t_half_opt = 0.693 / ke_opt if ke_opt > 0 else float('inf')

    return {
        'vd': vd_opt_kg, # Vd per kg
        'vd_total': vd_total_opt, # Total Vd in L
        'ke': ke_opt,
        'cl': cl_opt,
        't_half': t_half_opt,
        'optimization_success': result.success,
        'final_nll': result.fun # Final negative log-likelihood value
    }


# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM.
    Uses OpenAI API if available, otherwise provides a simulated response.

    Parameters:
    - prompt: The clinical data prompt including calculated values and context.
    - patient_data: Dictionary with patient information (used for context).
    """
    # Extract the drug type from the prompt for context
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
    else:
        drug = "Antimicrobial"

    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Include patient context if relevant (e.g., renal function).
            Format your response with these exact sections:

            ## CLINICAL ASSESSMENT
            📊 **MEASURED/ESTIMATED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **ASSESSMENT:** (state if appropriately dosed, underdosed, or overdosed based on levels and targets)

            ## RECOMMENDATIONS
            🔵 **DOSING:** (action-oriented recommendations using verbs like CONTINUE, ADJUST, HOLD, INCREASE, DECREASE. Suggest practical regimens where possible.)
            🔵 **MONITORING:** (specific monitoring parameters and schedule, e.g., recheck levels, renal function)
            ⚠️ **CAUTIONS:** (relevant warnings, e.g., toxicity risk, renal impairment)

            Here is the case:
            --- Patient Context ---
            Age: {patient_data.get('age', 'N/A')} years, Gender: {patient_data.get('gender', 'N/A')}
            Weight: {patient_data.get('weight', 'N/A')} kg, Height: {patient_data.get('height', 'N/A')} cm
            Patient ID: {patient_data.get('patient_id', 'N/A')}, Ward: {patient_data.get('ward', 'N/A')}
            Serum Cr: {patient_data.get('serum_cr', 'N/A')} µmol/L, CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})
            Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}
            Current Regimen: {patient_data.get('current_dose_regimen', 'N/A')}
            Notes: {patient_data.get('notes', 'N/A')}
            --- TDM Data & Calculations ---
            {prompt}
            --- End of Case ---
            """

            # Call OpenAI API - updated for openai v1.0.0+
            response = openai.chat.completions.create(
                model="gpt-4",  # or your preferred model like gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear, actionable recommendations in the specified format."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3, # Lower temperature for more deterministic clinical advice
                max_tokens=600 # Increased token limit for detailed response
            )
            llm_response = response.choices[0].message.content

            st.subheader("Clinical Interpretation (LLM)")
            st.markdown(llm_response) # Display the formatted response directly
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")

            # No PDF generation needed here

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
            # Fall through to standardized interpretation if API fails

    # If OpenAI is not available/fails, use the standardized interpretation
    if not (OPENAI_AVAILABLE and openai.api_key): # Or if the API call failed above
        st.subheader("Clinical Interpretation (Simulated)")
        interpretation_data = generate_standardized_interpretation(prompt, drug, patient_data)

        # If the interpretation_data is a string (error message), just display it
        if isinstance(interpretation_data, str):
            st.write(interpretation_data)
            return

        # Unpack the interpretation data
        levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data

        # Display the formatted interpretation
        formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
        st.markdown(formatted_interpretation) # Use markdown for better formatting

        # Add note about simulated response
        st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")

    # Add the raw prompt at the bottom for debugging/transparency
    with st.expander("Raw Analysis Data Sent to LLM (or used for Simulation)", expanded=False):
        st.code(prompt)


def generate_standardized_interpretation(prompt, drug, patient_data):
    """
    Generate a standardized interpretation based on drug type and prompt content.
    Includes patient context for better recommendations.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    crcl = patient_data.get('crcl', None) # Get CrCl for context

    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt, crcl)
    elif drug == "Aminoglycoside":
        return generate_aminoglycoside_interpretation(prompt, crcl)
    else:
        # For generic, create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        if crcl and crcl < 60:
             cautions.append(f"Renal function (CrCl: {crcl:.1f} mL/min) may impact dosing.")

        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy using Markdown.

    Parameters:
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string (e.g., "appropriately dosed")
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - cautions: Optional list of caution strings

    Returns:
    - Formatted markdown string
    """
    # Format measured levels with status indicators
    levels_md = "📊 **MEASURED/ESTIMATED LEVELS:**\n"
    if not levels_data or (len(levels_data) == 1 and levels_data[0][0] == "Not available"):
         levels_md += "- No levels data available for interpretation.\n"
    else:
        for name, value, target, status in levels_data:
            icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴" # Red for above
            # Format value appropriately (e.g., 1 decimal for levels, 0 for AUC)
            value_str = f"{value:.1f}" if isinstance(value, (int, float)) and "AUC" not in name else f"{value:.0f}" if isinstance(value, (int, float)) else str(value)
            levels_md += f"- {name}: {value_str} (Target: {target}) {icon}\n"


    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is **{assessment.upper()}**"

    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## RECOMMENDATIONS

🔵 **DOSING:**
"""
    if dosing_recs:
        for rec in dosing_recs:
            output += f"- {rec}\n"
    else:
        output += "- No specific dosing recommendations generated.\n"

    output += "\n🔵 **MONITORING:**\n"
    if monitoring_recs:
        for rec in monitoring_recs:
            output += f"- {rec}\n"
    else:
        output += "- Standard monitoring applies.\n"

    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"

    return output

def generate_vancomycin_interpretation(prompt, crcl=None):
    """
    Generate standardized vancomycin interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    # Extract key values from the prompt using regex for robustness
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract levels (measured or estimated)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Trough.*?([\d.,]+)\s*mg/L", prompt)
    peak_val
    desired_peak = st.number_input("Desired Target Peak (mg/L)", min_value=0.0, value=default_desired_peak, format="%.1f")
                    desired_interval = st.number_input("Desired Target Interval (hr)", min_value=4, max_value=72, value=tau, step=4) # Default to current interval

                    # Calculate new dose needed for desired peak at desired interval
                    # Dose = Cmax_desired * Vd * ke * T_inf * (1 - exp(-ke * Tau_desired)) / (1 - exp(-ke * T_inf))
                    new_dose = 0.0
                    try:
                        new_term_ke_tinf = ke * infusion_duration
                        new_term_ke_tau = ke * desired_interval
                        new_exp_ke_tinf = math.exp(-new_term_ke_tinf)
                        new_exp_ke_tau = math.exp(-new_term_ke_tau)

                        new_numerator = desired_peak * vd * new_term_ke_tinf * (1.0 - new_exp_ke_tau)
                        new_denominator = (1.0 - new_exp_ke_tinf)

                        if abs(new_denominator) > 1e-9:
                             new_dose = new_numerator / new_denominator
                        else:
                             st.warning("Could not calculate new dose accurately due to near-zero denominator.")

                    except (OverflowError, ValueError) as math_err_newdose:
                         st.error(f"Math error during new dose calculation: {math_err_newdose}")


                    # Round new dose
                    rounding_base = 20 if drug == "Gentamicin" else 50 if drug == "Amikacin" else 10
                    practical_new_dose = round(new_dose / rounding_base) * rounding_base
                    practical_new_dose = max(rounding_base, practical_new_dose)

                    # Predict peak and trough with new dose and interval
                    predicted_peak = 0.0
                    predicted_trough = 0.0
                    if practical_new_dose > 0 and vd > 0 and ke > 0 and infusion_duration > 0 and desired_interval > 0:
                        try:
                            pred_term_ke_tinf = ke * infusion_duration
                            pred_term_ke_tau = ke * desired_interval
                            pred_exp_ke_tinf = math.exp(-pred_term_ke_tinf)
                            pred_exp_ke_tau = math.exp(-pred_term_ke_tau)

                            pred_denominator_cmax = vd * ke * infusion_duration * (1.0 - pred_exp_ke_tau)
                            if abs(pred_denominator_cmax) > 1e-9:
                                 predicted_peak = practical_new_dose * (1.0 - pred_exp_ke_tinf) / pred_denominator_cmax

                            predicted_trough = predicted_peak * math.exp(-ke * (desired_interval - infusion_duration))

                        except (OverflowError, ValueError) as math_err_pred:
                             st.warning(f"Could not predict levels for new dose due to math error: {math_err_pred}")


                    st.success(f"Suggested New Regimen: **{practical_new_dose:.0f} mg** IV q **{desired_interval:.0f}h** (infused over {infusion_duration} hr)")
                    st.info(f"Predicted Peak: ~{predicted_peak:.1f} mg/L | Predicted Trough: ~{predicted_trough:.2f} mg/L")

                    # Check predicted levels against targets
                    suggest_adjustment(predicted_peak, target_peak_min, target_peak_max, label="Predicted Peak")
                    if predicted_trough >= target_trough_max:
                         st.warning(f"⚠️ Predicted Trough ({predicted_trough:.2f} mg/L) meets or exceeds target maximum ({target_trough_max} mg/L). Consider lengthening interval further.")
                    else:
                         st.success(f"✅ Predicted Trough ({predicted_trough:.2f} mg/L) is below target maximum ({target_trough_max} mg/L).")


        except ValueError as ve:
            st.error(f"Input Error: {ve}")
            results_calculated = False
        except Exception as e:
            st.error(f"Calculation Error: {e}. Please check inputs.")
            results_calculated = False # Ensure button doesn't show if error


    # --- Interpretation and Visualization ---
    if results_calculated:
        # Add visualization option
        if st.checkbox("Show Estimated Concentration-Time Curve (Based on Calculated Parameters)"):
            if cmax_extrapolated > 0 and cmin_extrapolated >= 0 and ke > 0 and tau > 0:
                 chart = plot_concentration_time_curve(
                     peak=cmax_extrapolated,
                     trough=cmin_extrapolated,
                     ke=ke,
                     tau=tau, # Show curve for the *current* interval
                     t_peak=infusion_duration,
                     infusion_time=infusion_duration
                 )
                 st.altair_chart(chart, use_container_width=True)
            else:
                 st.warning("Cannot display curve due to invalid calculated parameters.")


        if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
            # Prepare prompt for interpretation
            prompt = (f"Aminoglycoside TDM Adjustment:\n"
                      f"Drug: {drug}, Regimen Goal: {regimen}\n"
                      f"Current Regimen: {dose:.0f} mg IV q {tau:.0f}h (infused over {infusion_duration} hr)\n"
                      f"Measured Levels: Trough (C1)={c1:.1f} mg/L at {t1:.2f} hr post-start ({c1_sample_datetime_dt.strftime('%Y-%m-%d %H:%M')}); Peak (C2)={c2:.1f} mg/L at {t2:.2f} hr post-start ({c2_sample_datetime_dt.strftime('%Y-%m-%d %H:%M')}).\n"
                      f"Target Ranges: Peak {target_peak_info}, Trough {target_trough_info}\n"
                      f"Calculated PK: Ke={ke:.4f} hr⁻¹, t½={t_half:.2f} hr, Vd≈{vd:.2f} L, CL≈{cl:.2f} L/hr\n"
                      f"Estimated Levels (Current Regimen): Cmax≈{cmax_extrapolated:.1f} mg/L, Cmin≈{cmin_extrapolated:.2f} mg/L\n")
            if 'practical_new_dose' in locals() and 'desired_interval' in locals(): # Add recommendation if calculated
                 prompt += (f"Suggested Adjustment: {practical_new_dose:.0f} mg IV q {desired_interval:.0f}h\n"
                           f"Predicted Levels (New Regimen): Peak≈{predicted_peak:.1f} mg/L, Trough≈{predicted_trough:.2f} mg/L")

            interpret_with_llm(prompt, patient_data)


# ===== MODULE 3: Vancomycin AUC-based Dosing =====
def vancomycin_auc_dosing(patient_data):
    st.title("🧪 Vancomycin AUC-Based Dosing & Adjustment")
    st.info("AUC24 is calculated using the Linear-Log Trapezoidal method as recommended for vancomycin TDM")

    # Unpack patient data
    weight = patient_data['weight']
    crcl = patient_data['crcl']
    gender = patient_data['gender']
    age = patient_data['age']

    method = st.radio("Select Method / Scenario", ["Calculate Initial Dose (Population PK)", "Adjust Dose using Trough Level", "Adjust Dose using Peak & Trough Levels"], horizontal=True)

    # --- Target Selection ---
    st.markdown("### Target Selection")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        target_auc = st.slider("Target AUC24 (mg·hr/L)", min_value=300, max_value=700, value=500, step=10) # Default 400-600
        st.info("Typical Target AUC24: 400-600 mg·hr/L")
    with col_t2:
        # Trough target is secondary but often monitored
        target_trough_range = st.selectbox(
            "Secondary Target Trough Range (mg/L)",
            ["10-15 (Empiric)", "15-20 (Definitive)", "Custom"]
        )
        if target_trough_range == "Custom":
             target_cmin_min = st.number_input("Min Trough", min_value=5.0, value=10.0, step=0.5)
             target_cmin_max = st.number_input("Max Trough", min_value=target_cmin_min, value=15.0, step=0.5)
        elif "15-20" in target_trough_range:
             target_cmin_min, target_cmin_max = 15.0, 20.0
        else: # Default 10-15
             target_cmin_min, target_cmin_max = 10.0, 15.0
        st.info(f"Selected Target Trough: {target_cmin_min:.1f} - {target_cmin_max:.1f} mg/L")
    
    # Add Peak targets
    st.markdown("### Peak Target Range")
    if "Empiric" in target_trough_range:
        target_peak_min, target_peak_max = 20.0, 30.0
        st.info("Peak Target for Empiric: 20-30 mg/L")
    elif "Definitive" in target_trough_range:
        target_peak_min, target_peak_max = 25.0, 40.0
        st.info("Peak Target for Definitive: 25-40 mg/L")
    else:  # Custom
        target_peak_min = st.number_input("Min Peak", min_value=10.0, value=20.0, step=1.0)
        target_peak_max = st.number_input("Max Peak", min_value=target_peak_min, value=30.0, step=1.0)
        st.info(f"Custom Peak Target: {target_peak_min:.1f} - {target_peak_max:.1f} mg/L")

    # --- Input Fields based on Method ---
    st.markdown("### Enter Dosing and Level Information")

    # --- Initial Dose Calculation ---
    if "Initial Dose" in method:
        st.markdown("Using population PK estimates based on patient demographics.")
        desired_interval = st.selectbox("Desired Dosing Interval (hr)", [8, 12, 24, 36, 48], index=1) # Default q12h
        infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5, help="Typically 1 hr per 1g")

        # Population PK Estimates (Simplified Bayesian approach using CrCl)
        # Ref: Pai MP, Neely M, Rodvold KA, Lodise TP. Innovative approaches to optimizing the delivery of vancomycin in infected patients. Adv Drug Deliv Rev. 2014;
        # Vd (L) ≈ 0.7 L/kg * Weight (kg) -- Can refine based on age/obesity if needed
        # CLvanco (L/hr) ≈ CrCl (mL/min) * (0.75 to 0.9) * 60 / 1000 -- Renal clearance dominant
        vd_pop = 0.7 * weight
        cl_pop = crcl * 0.8 * 60 / 1000 # Using 0.8 as renal clearance fraction
        cl_pop = max(0.1, cl_pop) # Ensure CL is not zero
        ke_pop = cl_pop / vd_pop if vd_pop > 0 else 0
        ke_pop = max(1e-6, ke_pop) # Ensure Ke is positive
        t_half_pop = 0.693 / ke_pop if ke_pop > 0 else float('inf')

        st.markdown("#### Population PK Estimates:")
        st.markdown(f"**Est. Vd:** {vd_pop:.2f} L | **Est. CL:** {cl_pop:.2f} L/hr | **Est. Ke:** {ke_pop:.4f} hr⁻¹ | **Est. t½:** {t_half_pop:.2f} hr")

        # Calculate Dose needed for Target AUC
        # AUC24 = Dose_daily / CL => Dose_daily = AUC24 * CL
        # Dose_per_interval = Dose_daily / (24 / interval)
        target_dose_daily = target_auc * cl_pop
        target_dose_interval = target_dose_daily / (24 / desired_interval)

        # Round to practical dose (e.g., nearest 250mg)
        practical_dose = round(target_dose_interval / 250) * 250
        practical_dose = max(250, practical_dose) # Minimum practical dose

        # Predict levels with this practical dose
        # Using steady-state infusion equations
        predicted_peak = 0.0
        predicted_trough = 0.0
        if vd_pop > 0 and ke_pop > 0 and infusion_duration > 0 and desired_interval > 0:
            try:
                term_inf = (1 - math.exp(-ke_pop * infusion_duration))
                term_interval = (1 - math.exp(-ke_pop * desired_interval))
                denominator = vd_pop * ke_pop * infusion_duration * term_interval

                if abs(denominator) > 1e-9 and abs(term_inf) > 1e-9:
                    # Cmax = Dose * (1 - exp(-ke * T_inf)) / [Vd * ke * T_inf * (1 - exp(-ke * tau))]
                    predicted_peak = (practical_dose * term_inf) / denominator
                    predicted_trough = predicted_peak * math.exp(-ke_pop * (desired_interval - infusion_duration))
            except (OverflowError, ValueError):
                 st.warning("Could not predict levels due to math error.")


        st.markdown("#### Recommended Initial Dose:")
        st.success(f"Start with **{practical_dose:.0f} mg** IV q **{desired_interval:.0f}h** (infused over {infusion_duration} hr)")
        
        # Calculate predicted AUC using trapezoidal method
        predicted_auc24 = 0
        if predicted_peak > 0 and predicted_trough >= 0 and ke_pop > 0 and desired_interval > 0:
            predicted_auc24 = calculate_vancomycin_auc_trapezoidal(
                predicted_peak, predicted_trough, ke_pop, desired_interval, infusion_duration
            )
        else:
            # Fallback to simple calculation
            predicted_auc24 = (practical_dose * (24/desired_interval)) / cl_pop if cl_pop > 0 else 0
        
        st.info(f"Predicted AUC24: ~{predicted_auc24:.0f} mg·hr/L")
        st.info(f"Predicted Peak (end of infusion): ~{predicted_peak:.1f} mg/L")
        st.info(f"Predicted Trough (end of interval): ~{predicted_trough:.1f} mg/L")

        # Check predicted trough against secondary target
        if predicted_trough < target_cmin_min: st.warning("⚠️ Predicted trough may be below secondary target range.")
        elif predicted_trough > target_cmin_max: st.warning("⚠️ Predicted trough may be above secondary target range.")
        else: st.success("✅ Predicted trough is within secondary target range.")

        # Suggest Loading Dose for severe infections or high target AUC
        if target_auc >= 500 or "sepsis" in patient_data.get('clinical_diagnosis', '').lower() or "meningitis" in patient_data.get('clinical_diagnosis', '').lower():
             loading_dose = 25 * weight # Common LD: 25-30 mg/kg (using actual weight)
             practical_loading_dose = round(loading_dose / 250) * 250
             st.warning(f"Consider Loading Dose: **~{practical_loading_dose:.0f} mg** IV x 1 dose (e.g., 25 mg/kg actual weight). Infuse over 1.5-2 hours.")

        # Visualization and Interpretation
        if st.checkbox("Show Estimated Concentration-Time Curve"):
            if predicted_peak > 0 and predicted_trough >= 0 and ke_pop > 0 and desired_interval > 0:
                chart = plot_concentration_time_curve(
                    peak=predicted_peak, trough=predicted_trough, ke=ke_pop, tau=desired_interval,
                    t_peak=infusion_duration, infusion_time=infusion_duration
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Cannot display curve due to invalid calculated parameters.")


        if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
            prompt = (f"Vancomycin Initial Dose Calculation:\n"
                      f"Target AUC24: {target_auc} mg·hr/L, Secondary Target Trough: {target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L\n"
                      f"Desired Interval: {desired_interval} hr, Infusion Duration: {infusion_duration} hr\n"
                      f"Population PK Estimates: Vd={vd_pop:.2f} L, CL={cl_pop:.2f} L/hr, Ke={ke_pop:.4f} hr⁻¹, t½={t_half_pop:.2f} hr\n"
                      f"Recommended Initial Dose: {practical_dose:.0f} mg IV q {desired_interval:.0f}h\n"
                      f"Predicted Levels: Peak≈{predicted_peak:.1f} mg/L, Trough≈{predicted_trough:.1f} mg/L, AUC24≈{predicted_auc24:.0f} mg·hr/L")
            interpret_with_llm(prompt, patient_data)


    # --- Adjustment using Trough Level ---
    elif "Trough Level" in method:
        st.markdown("Adjusting dose based on a measured trough level and population Vd estimate.")
        col1, col2 = st.columns(2)
        with col1:
            current_dose_interval = st.number_input("Current Dose per Interval (mg)", min_value=250.0, value=1000.0, step=50.0)
            current_interval = st.number_input("Current Dosing Interval (hr)", min_value=4, max_value=72, value=12, step=4)
            infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        with col2:
            measured_trough = st.number_input("Measured Trough Level (mg/L)", min_value=0.1, value=12.0, step=0.1, format="%.1f")
            trough_sample_time_rel = st.selectbox("Trough Sample Time Relative to Dose", ["Just Before Next Dose (Steady State)", "Other"], index=0)
            if trough_sample_time_rel == "Other":
                 st.warning("Calculation assumes steady-state trough taken just before the next dose. Interpretation may be less accurate otherwise.")

        # Estimate Ke and CL using measured trough and population Vd (Simplified Bayesian / Ratio Method)
        vd_pop = 0.7 * weight
        cl_pop = crcl * 0.8 * 60 / 1000
        cl_pop = max(0.1, cl_pop)
        ke_pop = cl_pop / vd_pop if vd_pop > 0 else 0
        ke_pop = max(1e-6, ke_pop)
        t_half_pop = 0.693 / ke_pop if ke_pop > 0 else float('inf')

        st.markdown("#### Population PK Estimates (Used as Prior):")
        st.markdown(f"**Est. Vd:** {vd_pop:.2f} L | **Est. CL:** {cl_pop:.2f} L/hr | **Est. Ke:** {ke_pop:.4f} hr⁻¹ | **Est. t½:** {t_half_pop:.2f} hr")

        # Calculate predicted trough using population PK
        predicted_trough_pop = 0.0
        if vd_pop > 0 and ke_pop > 0 and infusion_duration > 0 and current_interval > 0:
             try:
                 term_inf_pop = (1 - math.exp(-ke_pop * infusion_duration))
                 term_int_pop = (1 - math.exp(-ke_pop * current_interval))
                 denom_pop = vd_pop * ke_pop * infusion_duration * term_int_pop
                 if abs(denom_pop) > 1e-9 and abs(term_inf_pop) > 1e-9:
                     cmax_pred_pop = (current_dose_interval * term_inf_pop) / denom_pop
                     predicted_trough_pop = cmax_pred_pop * math.exp(-ke_pop * (current_interval - infusion_duration))
             except (OverflowError, ValueError): pass # Keep predicted_trough_pop as 0

        # Adjust CL based on ratio of measured trough to predicted population trough
        cl_adjusted = cl_pop
        if predicted_trough_pop > 0.5 and measured_trough > 0.1: # Avoid adjusting if predicted is very low or measured is zero
             # Ratio adjustment: New CL = Old CL * (Target / Measured) -> applied to AUC
             # Adjustment based on trough ratio: CL_adj = CL_pop * (Pred_Trough / Meas_Trough)
             cl_adjusted = cl_pop * (predicted_trough_pop / measured_trough)
             cl_adjusted = max(0.05, min(cl_adjusted, cl_pop * 5)) # Bound the adjustment
        elif measured_trough <= 0.1:
             st.warning("Measured trough is very low or zero, cannot reliably adjust CL based on ratio.")


        # Recalculate Ke based on adjusted CL and pop Vd
        ke_adjusted = cl_adjusted / vd_pop if vd_pop > 0 else ke_pop
        ke_adjusted = max(1e-6, ke_adjusted)
        t_half_adjusted = 0.693 / ke_adjusted if ke_adjusted > 0 else float('inf')

        st.markdown("#### Adjusted PK Estimates (Based on Trough):")
        st.markdown(f"**Adj. CL:** {cl_adjusted:.2f} L/hr | **Adj. Ke:** {ke_adjusted:.4f} hr⁻¹ | **Adj. t½:** {t_half_adjusted:.2f} hr")

        # Calculate current AUC24 using trapezoidal method if peaks and troughs available
        current_auc24 = 0
        if vd_pop > 0 and ke_adjusted > 0 and infusion_duration > 0 and current_interval > 0:
            try:
                # Calculate Cmax using adjusted PK parameters
                term_inf_adj = (1 - math.exp(-ke_adjusted * infusion_duration))
                term_int_adj = (1 - math.exp(-ke_adjusted * current_interval))
                denom_adj = vd_pop * ke_adjusted * infusion_duration * term_int_adj
                
                if abs(denom_adj) > 1e-9 and abs(term_inf_adj) > 1e-9:
                    cmax_calc = (current_dose_interval * term_inf_adj) / denom_adj
                    cmin_calc = cmax_calc * math.exp(-ke_adjusted * (current_interval - infusion_duration))
                    
                    # Use trapezoidal method
                    current_auc24 = calculate_vancomycin_auc_trapezoidal(
                        cmax_calc, cmin_calc, ke_adjusted, current_interval, infusion_duration
                    )
            except (OverflowError, ValueError):
                # Fallback to simple calculation
                current_dose_daily = current_dose_interval * (24 / current_interval)
                current_auc24 = current_dose_daily / cl_adjusted if cl_adjusted > 0 else 0

        st.markdown("#### Current Regimen Assessment:")
        st.markdown(f"Measured Trough: **{measured_trough:.1f} mg/L**")
        st.markdown(f"Estimated AUC24 (Current Dose): **{current_auc24:.0f} mg·hr/L**")

        # Check against targets
        if measured_trough < target_cmin_min: st.warning(f"⚠️ Measured Trough is BELOW target ({target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L).")
        elif measured_trough > target_cmin_max: st.warning(f"⚠️ Measured Trough is ABOVE target ({target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L).")
        else: st.success("✅ Measured Trough is WITHIN target range.")

        if current_auc24 < 400: st.warning(f"⚠️ Estimated AUC24 is LOW (<400 mg·hr/L).")
        elif current_auc24 > 600: st.warning(f"⚠️ Estimated AUC24 is HIGH (>600 mg·hr/L).")
        else: st.success("✅ Estimated AUC24 is WITHIN target range (400-600 mg·hr/L).")


        # Calculate New Dose for Target AUC
        st.markdown("#### Dose Adjustment Recommendation:")
        if cl_adjusted <= 0:
             st.warning("Cannot recommend new dose as adjusted Clearance is invalid.")
        else:
            desired_interval_adj = st.selectbox("Desired Target Interval (hr)", [8, 12, 24, 36, 48], index=[8, 12, 24, 36, 48].index(current_interval) if current_interval in [8,12,24,36,48] else 1)

            new_dose_daily = target_auc * cl_adjusted
            new_dose_interval = new_dose_daily / (24 / desired_interval_adj)

            # Round to practical dose
            practical_new_dose = round(new_dose_interval / 250) * 250
            practical_new_dose = max(250, practical_new_dose)

            # Predict levels with new practical dose using adjusted PK
            predicted_peak_new = 0.0
            predicted_trough_new = 0.0
            if vd_pop > 0 and ke_adjusted > 0 and infusion_duration > 0 and desired_interval_adj > 0:
                 try:
                     term_inf_adj = (1 - math.exp(-ke_adjusted * infusion_duration))
                     term_interval_adj = (1 - math.exp(-ke_adjusted * desired_interval_adj))
                     denominator_adj = vd_pop * ke_adjusted * infusion_duration * term_interval_adj

                     if abs(denominator_adj) > 1e-9 and abs(term_inf_adj) > 1e-9:
                         predicted_peak_new = (practical_new_dose * term_inf_adj) / denominator_adj
                         predicted_trough_new = predicted_peak_new * math.exp(-ke_adjusted * (desired_interval_adj - infusion_duration))
                 except (OverflowError, ValueError): pass # Keep levels as 0

            # Calculate predicted AUC using trapezoidal method if predicted levels available
            predicted_auc_new = 0
            if predicted_peak_new > 0 and predicted_trough_new >= 0 and ke_adjusted > 0 and desired_interval_adj > 0:
                predicted_auc_new = calculate_vancomycin_auc_trapezoidal(
                    predicted_peak_new, predicted_trough_new, ke_adjusted, desired_interval_adj, infusion_duration
                )
            else:
                # Fallback to simple calculation
                predicted_auc_new = (practical_new_dose * (24/desired_interval_adj)) / cl_adjusted if cl_adjusted > 0 else 0


            st.success(f"Adjust to: **{practical_new_dose:.0f} mg** IV q **{desired_interval_adj:.0f}h** (infused over {infusion_duration} hr)")
            st.info(f"Predicted AUC24: ~{predicted_auc_new:.0f} mg·hr/L")
            st.info(f"Predicted Trough: ~{predicted_trough_new:.1f} mg/L")

            # Check predicted trough against secondary target
            if predicted_trough_new < target_cmin_min: st.warning("⚠️ Predicted trough with new dose may be below secondary target range.")
            elif predicted_trough_new > target_cmin_max: st.warning("⚠️ Predicted trough with new dose may be above secondary target range.")
            else: st.success("✅ Predicted trough with new dose is within secondary target range.")

            # Visualization and Interpretation
            if st.checkbox("Show Estimated Concentration-Time Curve (Adjusted PK)"):
                 if predicted_peak_new > 0 and predicted_trough_new >= 0 and ke_adjusted > 0 and desired_interval_adj > 0:
                     chart = plot_concentration_time_curve(
                         peak=predicted_peak_new, trough=predicted_trough_new, ke=ke_adjusted, tau=desired_interval_adj,
                         t_peak=infusion_duration, infusion_time=infusion_duration
                     )
                     st.altair_chart(chart, use_container_width=True)
                 else:
                     st.warning("Cannot display curve due to invalid calculated parameters for new dose.")


            if st.button("🧠 Generate Clinical Interpretation (LLM/    if drug == "Gentamicin":
        if regimen_code == "MDD": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 1.0, "5-10 mg/L", "<2 mg/L"
        elif regimen_code == "SDD": default_peak, default_trough, target_peak_info, target_trough_info = 20.0, 0.5, "15-30 mg/L (or 10x MIC)", "<1 mg/L (often undetectable)"
        elif regimen_code == "Synergy": default_peak, default_trough, target_peak_info, target_trough_info = 4.0, 0.5, "3-5 mg/L", "<1 mg/L"
        elif regimen_code == "Hemodialysis": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 1.0, "Peak not routinely targeted", "<2 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 0.5, "5-12 mg/L", "<1 mg/L"
    else:  # Amikacin
        if regimen_code == "MDD": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 5.0, "20-30 mg/L", "<10 mg/L"
        elif regimen_code == "SDD": default_peak, default_trough, target_peak_info, target_trough_info = 60.0, 2.0, "50-70 mg/L (or 10x MIC)", "<5 mg/L (often undetectable)"
        elif regimen_code == "Synergy": default_peak, default_trough, target_peak_info, target_trough_info = 0.0, 0.0, "N/A", "N/A"
        elif regimen_code == "Hemodialysis": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 5.0, "Peak not routinely targeted", "<10 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 2.5, "20-30 mg/L", "<5 mg/L"

    st.info(f"Typical Targets for {regimen}: Peak {target_peak_info}, Trough {target_trough_info}")

    # MIC input for SDD regimens
    mic = 1.0 # Default MIC
    if regimen_code == "SDD":
        st.markdown("*Note: Target peak for Extended Interval is often 10x MIC.*")
        mic = st.number_input("Enter MIC (mg/L)", min_value=0.1, value=1.0, step=0.1, format="%.1f")
        recommended_peak_mic = mic * 10
        if recommended_peak_mic > default_peak:
            default_peak = recommended_peak_mic
        st.info(f"Based on MIC, target peak is ≥ {recommended_peak_mic:.1f} mg/L")

    # Allow user override of targets
    col1, col2 = st.columns(2)
    with col1:
        target_cmax = st.number_input("Target Peak (Cmax, mg/L)", value=default_peak, format="%.1f")
    with col2:
        target_cmin = st.number_input("Target Trough (Cmin, mg/L)", value=default_trough, format="%.1f")

    # Default tau based on regimen
    default_tau = 24 if regimen_code == "SDD" \
             else 8 if regimen_code == "MDD" \
             else 12 if regimen_code == "Synergy" \
             else 48 # Default for HD (q48h common) / Neonates (adjust based on age/PMA)
    tau = st.number_input("Desired Dosing Interval (hr)", min_value=4, max_value=72, value=default_tau, step=4)

    # Infusion duration
    infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)

    # Special handling notes
    if regimen_code == "Hemodialysis":
        st.info("For hemodialysis, dose is typically given post-dialysis. Interval depends on dialysis schedule (e.g., q48h, q72h). Calculations assume dose given after dialysis.")
    if regimen_code == "Neonates":
        st.warning("Neonatal PK varies significantly. These calculations use adult population estimates. CONSULT a pediatric pharmacist.")

    # --- Calculations ---
    # Calculate IBW and dosing weight (using standard formulas)
    ibw = 0.0
    if height > 152.4: # Height threshold for formulas (60 inches)
        ibw = (50 if gender == "Male" else 45.5) + 2.3 * (height / 2.54 - 60)
    ibw = max(0, ibw) # Ensure IBW is not negative

    dosing_weight = weight # Default to actual body weight
    weight_used = "Actual Body Weight"
    if ibw > 0: # Only adjust if IBW is calculable and patient is not underweight
        if weight / ibw > 1.3: # Obese threshold (e.g., >130% IBW)
            dosing_weight = ibw + 0.4 * (weight - ibw) # Adjusted BW
            weight_used = "Adjusted Body Weight"
        elif weight < ibw: # Underweight: Use Actual BW (common practice)
             dosing_weight = weight
             weight_used = "Actual Body Weight (using ABW as < IBW)"
        else: # Normal weight: Use Actual or Ideal (Using Actual here)
             dosing_weight = weight
             weight_used = "Actual Body Weight"


    st.markdown(f"**IBW:** {ibw:.1f} kg | **Dosing Weight Used:** {dosing_weight:.1f} kg ({weight_used})")

    # Population PK parameters (adjust Vd based on clinical factors if needed)
    base_vd_per_kg = 0.3 if drug == "Amikacin" else 0.26 # L/kg
    vd_adjustment = 1.0 # Default
    # Simple adjustments based on notes (can be refined)
    notes_lower = notes.lower()
    if any(term in notes_lower for term in ["ascites", "edema", "fluid overload", "anasarca", "chf exacerbation"]): vd_adjustment = 1.15; st.info("Vd increased by 15% due to potential fluid overload.")
    if any(term in notes_lower for term in ["septic", "sepsis", "burn", "icu patient"]): vd_adjustment = 1.20; st.info("Vd increased by 20% due to potential sepsis/burn/critical illness.")
    if any(term in notes_lower for term in ["dehydrated", "volume depleted"]): vd_adjustment = 0.90; st.info("Vd decreased by 10% due to potential dehydration.")

    vd = base_vd_per_kg * dosing_weight * vd_adjustment # Liters
    vd = max(1.0, vd) # Ensure Vd is at least 1L

    # Calculate Ke and Cl based on CrCl (population estimate)
    # Using published relationships might be better, e.g., Ke = a + b * CrCl
    # Simplified approach: CL (L/hr) ≈ CrCl (mL/min) * factor (e.g., 0.05 for Gentamicin)
    # Ke = CL / Vd
    cl_pop = 0.0
    if crcl > 0:
        # Example: Gentamicin CL ≈ 0.05 * CrCl (L/hr if CrCl in mL/min) - Highly simplified
        # Example: Amikacin CL might be slightly higher
        cl_factor = 0.06 if drug == "Amikacin" else 0.05
        cl_pop = cl_factor * crcl
    cl_pop = max(0.1, cl_pop) # Minimum clearance estimate

    ke = cl_pop / vd if vd > 0 else 0.01
    ke = max(0.005, ke) # Ensure ke is not excessively low

    t_half = 0.693 / ke if ke > 0 else float('inf')

    st.markdown(f"**Est. Vd:** {vd:.2f} L | **Est. Ke:** {ke:.4f} hr⁻¹ | **Est. t½:** {t_half:.2f} hr | **Est. CL:** {cl_pop:.2f} L/hr")

    # Calculate Dose needed to achieve target Cmax (using steady-state infusion equation)
    # Dose = Cmax * Vd * ke * T_inf * (1 - exp(-ke * tau)) / (1 - exp(-ke * T_inf))
    dose = 0.0
    try:
        term_ke_tinf = ke * infusion_duration
        term_ke_tau = ke * tau
        exp_ke_tinf = math.exp(-term_ke_tinf)
        exp_ke_tau = math.exp(-term_ke_tau)

        numerator = target_cmax * vd * term_ke_tinf * (1.0 - exp_ke_tau)
        denominator = (1.0 - exp_ke_tinf)

        if abs(denominator) > 1e-9:
            dose = numerator / denominator
        else: # Handle bolus case approximation if T_inf is very small
            dose = target_cmax * vd * (1.0 - exp_ke_tau) / (1.0 - exp_ke_tinf) # Recheck this derivation
            # Simpler Bolus: Dose = Cmax * Vd * (1-exp(-ke*tau)) -> This assumes Cmax is achieved instantly
            st.warning("Infusion duration is very short or Ke is very low; using approximation for dose calculation.")
            # Let's stick to the rearranged infusion formula, checking denominator

    except (OverflowError, ValueError) as math_err:
         st.error(f"Math error during dose calculation: {math_err}. Check PK parameters.")
         dose = 0 # Prevent further calculation

    # Calculate expected levels with the calculated dose
    expected_cmax = 0.0
    expected_cmin = 0.0
    if dose > 0 and vd > 0 and ke > 0 and infusion_duration > 0 and tau > 0:
        try:
            term_ke_tinf = ke * infusion_duration
            term_ke_tau = ke * tau
            exp_ke_tinf = math.exp(-term_ke_tinf)
            exp_ke_tau = math.exp(-term_ke_tau)

            common_factor = (dose / (vd * term_ke_tinf)) * (1.0 - exp_ke_tinf) / (1.0 - exp_ke_tau)

            # Cmax occurs at the end of infusion (t = infusion_duration)
            # Using the "during infusion" part of the SS equation at t=Tinf
            # C(t_inf) = common_factor * (1 - exp(-ke*t_inf)) -> simplifies to peak formula
            expected_cmax = (dose / (vd * ke * infusion_duration)) * (1 - exp_ke_tinf) / (1 - exp_ke_tau) * (1 - exp_ke_tinf) # Recheck needed
            # Let's use the simpler Cmax definition from rearrangement:
            # Cmax = Dose * (1 - exp(-ke * T_inf)) / [Vd * ke * T_inf * (1 - exp(-ke * tau))]
            denominator_cmax = vd * ke * infusion_duration * (1 - exp_ke_tau)
            if abs(denominator_cmax) > 1e-9:
                 expected_cmax = dose * (1 - exp_ke_tinf) / denominator_cmax

            # Cmin occurs at the end of the interval (t = tau)
            # Cmin = Cmax * exp(-ke * (tau - T_inf))
            expected_cmin = expected_cmax * math.exp(-ke * (tau - infusion_duration))

        except (OverflowError, ValueError) as math_err_levels:
             st.warning(f"Could not predict levels due to math error: {math_err_levels}")


    # Round the dose to a practical value (e.g., nearest 10mg or 20mg)
    rounding_base = 20 if drug == "Gentamicin" else 50 if drug == "Amikacin" else 10
    practical_dose = round(dose / rounding_base) * rounding_base
    practical_dose = max(rounding_base, practical_dose) # Ensure dose is at least the rounding base

    st.success(f"Recommended Initial Dose: **{practical_dose:.0f} mg** IV every **{tau:.0f}** hours (infused over {infusion_duration} hr)")
    st.info(f"Predicted Peak (end of infusion): ~{expected_cmax:.1f} mg/L")
    st.info(f"Predicted Trough (end of interval): ~{expected_cmin:.2f} mg/L")


    # Suggest loading dose if applicable (e.g., for SDD or severe infections)
    if regimen_code == "SDD" or "sepsis" in notes.lower() or "critical" in notes.lower():
        # Loading Dose ≈ Target Peak * Vd
        loading_dose = target_cmax * vd
        practical_loading_dose = round(loading_dose / rounding_base) * rounding_base
        practical_loading_dose = max(rounding_base, practical_loading_dose)
        st.warning(f"Consider Loading Dose: **~{practical_loading_dose:.0f} mg** IV x 1 dose to rapidly achieve target peak.")

    # Check if expected levels meet targets
    suggest_adjustment(expected_cmax, target_cmax * 0.85, target_cmax * 1.15, label="Predicted Peak") # Tighter range for check
    # Check trough against target_cmin (which is usually the max allowed trough)
    if expected_cmin > target_cmin: # Target Cmin here represents the upper limit for trough
         st.warning(f"⚠️ Predicted Trough ({expected_cmin:.2f} mg/L) may exceed target ({target_trough_info}). Consider lengthening interval if clinically appropriate.")
    else:
         st.success(f"✅ Predicted Trough ({expected_cmin:.2f} mg/L) likely below target ({target_trough_info}).")

    # Add visualization option
    if st.checkbox("Show Estimated Concentration-Time Curve"):
        if expected_cmax > 0 and expected_cmin >= 0 and ke > 0 and tau > 0:
            chart = plot_concentration_time_curve(
                peak=expected_cmax,
                trough=expected_cmin,
                ke=ke,
                tau=tau,
                t_peak=infusion_duration, # Assume peak occurs at end of infusion
                infusion_time=infusion_duration
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Cannot display curve due to invalid calculated parameters.")


    if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
        prompt = (f"Aminoglycoside Initial Dose Calculation:\n"
                  f"Drug: {drug}, Regimen Goal: {regimen}\n"
                  f"Target Peak: {target_cmax:.1f} mg/L, Target Trough: {target_cmin:.1f} mg/L (Typical: Peak {target_peak_info}, Trough {target_trough_info})\n"
                  f"Desired Interval (tau): {tau} hr, Infusion Duration: {infusion_duration} hr\n"
                  f"Calculated Dose: {practical_dose:.0f} mg\n"
                  f"Estimated PK: Vd={vd:.2f} L, Ke={ke:.4f} hr⁻¹, t½={t_half:.2f} hr, CL={cl_pop:.2f} L/hr\n"
                  f"Predicted Levels: Peak≈{expected_cmax:.1f} mg/L, Trough≈{expected_cmin:.2f} mg/L")
        interpret_with_llm(prompt, patient_data)


# ===== MODULE 2: Aminoglycoside Conventional Dosing (C1/C2) =====
def aminoglycoside_conventional_dosing(patient_data):
    st.title("📊 Aminoglycoside Dose Adjustment (using Levels)")

    drug = st.selectbox("Select Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Therapeutic Goal / Strategy", ["Traditional (Multiple Daily - MDD)", "Extended Interval (Once Daily - SDD)", "Synergy (e.g., Endocarditis)", "Hemodialysis", "Neonates (Use with caution)"])

    # Map selection to internal codes
    regimen_code = "SDD" if "Extended" in regimen \
              else "MDD" if "Traditional" in regimen \
              else "Synergy" if "Synergy" in regimen \
              else "Hemodialysis" if "Hemodialysis" in regimen \
              else "Neonates" if "Neonates" in regimen \
              else "MDD" # Default

    # --- Set target ranges based on chosen regimen and drug ---
    target_peak_info = "N/A"
    target_trough_info = "N/A"
    target_peak_min, target_peak_max = 0.0, 100.0
    target_trough_max = 100.0 # Represents the upper limit for trough

    if drug == "Gentamicin":
        if regimen_code == "MDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 5, 10, 2, "5-10 mg/L", "<2 mg/L"
        elif regimen_code == "SDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 15, 30, 1, "15-30 mg/L (or 10x MIC)", "<1 mg/L (often undetectable)"
        elif regimen_code == "Synergy": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 3, 5, 1, "3-5 mg/L", "<1 mg/L"
        elif regimen_code == "Hemodialysis": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 2, "Peak not routinely targeted", "<2 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 5, 12, 1, "5-12 mg/L", "<1 mg/L"
    else:  # Amikacin
        if regimen_code == "MDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 20, 30, 10, "20-30 mg/L", "<10 mg/L"
        elif regimen_code == "SDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 50, 70, 5, "50-70 mg/L (or 10x MIC)", "<5 mg/L (often undetectable)"
        elif regimen_code == "Synergy": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 0, "N/A", "N/A"
        elif regimen_code == "Hemodialysis": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 10, "Peak not routinely targeted", "<10 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 20, 30, 5, "20-30 mg/L", "<5 mg/L"

    st.markdown("### Target Concentration Ranges:")
    col_t1, col_t2 = st.columns(2)
    with col_t1: st.markdown(f"**Peak Target:** {target_peak_info}")
    with col_t2: st.markdown(f"**Trough Target:** {target_trough_info}")

    # MIC input for SDD regimens
    mic = 1.0 # Default MIC
    if regimen_code == "SDD":
        st.markdown("*Note: Target peak for Extended Interval is often 10x MIC.*")
        mic = st.number_input("Enter MIC (mg/L)", min_value=0.1, value=1.0, step=0.1, format="%.1f")
        recommended_peak_mic = mic * 10
        st.info(f"Based on MIC, target peak is ≥ {recommended_peak_mic:.1f} mg/L. Adjust target below if needed.")
        # Update target peak min based on MIC if higher
        target_peak_min = max(target_peak_min, recommended_peak_mic)


    st.markdown("### Dosing and Sampling Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        dose = st.number_input("Dose Administered (mg)", min_value=10.0, value = 120.0, step=5.0)
        infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    with col2:
        # Use current date as default, allow user to change if needed
        default_date = datetime.now().date()
        dose_start_datetime_dt = st.datetime_input("Date & Time of Dose Start", value=datetime.combine(default_date, time(12,0)), step=timedelta(minutes=15)) # Combine date and time(12,0)
    with col3:
        tau = st.number_input("Current Dosing Interval (hr)", min_value=4, max_value=72, value=8, step=4)

    st.markdown("### Measured Levels and Sample Times")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        c1 = st.number_input("Trough Level (C1, mg/L)", min_value=0.0, value=1.0, step=0.1, format="%.1f", help="Usually pre-dose level")
        c1_sample_datetime_dt = st.datetime_input("Date & Time of Trough Sample", value=datetime.combine(default_date, time(11,30)), step=timedelta(minutes=15)) # Default 30 min before 12pm dose
    with col_l2:
        c2 = st.number_input("Peak Level (C2, mg/L)", min_value=0.0, value=8.0, step=0.1, format="%.1f", help="Usually post-infusion level")
        c2_sample_datetime_dt = st.datetime_input("Date & Time of Peak Sample", value=datetime.combine(default_date, time(13,30)), step=timedelta(minutes=15)) # Default 30 min after 1hr infusion ends (1:30pm)


    # --- Calculate t1 and t2 relative to dose start time ---
    t1 = (c1_sample_datetime_dt - dose_start_datetime_dt).total_seconds() / 3600.0 # Time of C1 sample relative to dose start in hours
    t2 = (c2_sample_datetime_dt - dose_start_datetime_dt).total_seconds() / 3600.0 # Time of C2 sample relative to dose start in hours

    st.markdown(f"*Calculated time from dose start to Trough (C1) sample (t1): {t1:.2f} hr*")
    st.markdown(f"*Calculated time from dose start to Peak (C2) sample (t2): {t2:.2f} hr*")

    # Validate timings
    valid_times = True
    if t1 >= t2:
        st.error("❌ Trough sample time (C1) must be before Peak sample time (C2). Please check the dates and times.")
        valid_times = False
    if t2 <= infusion_duration:
        st.warning(f"⚠️ Peak sample time (t2={t2:.2f} hr) is during or before the end of infusion ({infusion_duration:.1f} hr). Calculated Cmax will be extrapolated; accuracy may be reduced.")
    if t1 > 0 and t1 < infusion_duration:
         st.warning(f"⚠️ Trough sample time (t1={t1:.2f} hr) appears to be during the infusion. Ensure C1 is a true pre-dose trough for most accurate calculations.")

    # --- Perform PK Calculations ---
    st.markdown("### Calculated Pharmacokinetic Parameters")
    results_calculated = False
    ke, t_half, vd, cl = 0, float('inf'), 0, 0
    cmax_extrapolated, cmin_extrapolated = 0, 0

    if valid_times:
        try:
            # Ensure levels are positive for log calculation
            if c1 <= 0 or c2 <= 0:
                st.error("❌ Measured levels (C1 and C2) must be greater than 0 for calculation.")

            else:
                # Calculate Ke using two levels (Sawchuk-Zaske method adaptation)
                # Assumes levels are in the elimination phase relative to each other
                delta_t = t2 - t1
                if delta_t <= 0: raise ValueError("Time difference between samples (t2-t1) must be positive.")

                # Check if both points are likely post-infusion for simple Ke calculation
                if t1 >= infusion_duration:
                     ke = (math.log(c1) - math.log(c2)) / delta_t # ln(C1/C2) / (t2-t1)
                else:
                     # If t1 is during infusion or pre-dose, Ke calculation is more complex.
                     # Using the simple formula introduces error. A Bayesian approach or iterative method is better.
                     # For this tool, we'll proceed with the simple formula but add a warning.
                     ke = (math.log(c1) - math.log(c2)) / delta_t
                     st.warning("⚠️ Ke calculated assuming log-linear decay between C1 and C2. Accuracy reduced if C1 is not post-infusion.")

                ke = max(1e-6, ke) # Ensure ke is positive and non-zero
                t_half = 0.693 / ke if ke > 0 else float('inf')

                # Extrapolate to find Cmax (at end of infusion) and Cmin (at end of interval)
                # C_t = C_known * exp(-ke * (t - t_known))
                # Cmax = C2 * exp(ke * (t2 - infusion_duration)) # Extrapolate C2 back to end of infusion
                cmax_extrapolated = c2 * math.exp(ke * (t2 - infusion_duration))

                # Cmin = Cmax_extrapolated * exp(-ke * (tau - infusion_duration)) # Trough at end of interval
                cmin_extrapolated = cmax_extrapolated * math.exp(-ke * (tau - infusion_duration))

                # Calculate Vd using Cmax and dose (steady-state infusion formula)
                # Vd = Dose * (1 - exp(-ke * T_inf)) / [Cmax * ke * T_inf * (1 - exp(-ke * tau))]
                term_inf = (1 - math.exp(-ke * infusion_duration))
                term_tau = (1 - math.exp(-ke * tau))
                denominator_vd = cmax_extrapolated * ke * infusion_duration * term_tau
                vd = 0.0
                if abs(denominator_vd) > 1e-9 and abs(term_inf) > 1e-9 : # Avoid division by zero
                    vd = (dose * term_inf) / denominator_vd
                    vd = max(1.0, vd) # Ensure Vd is at least 1L
                else:
                    st.warning("Could not calculate Vd accurately due to near-zero terms (check Ke, Tau, Infusion Duration).")

                cl = ke * vd if vd > 0 else 0.0

                st.markdown(f"**Individualized Ke:** {ke:.4f} hr⁻¹ | **t½:** {t_half:.2f} hr")
                st.markdown(f"**Est. Cmax (end of infusion):** {cmax_extrapolated:.1f} mg/L | **Est. Cmin (end of interval):** {cmin_extrapolated:.2f} mg/L")
                if vd > 0:
                     st.markdown(f"**Est. Vd:** {vd:.2f} L | **Est. CL:** {cl:.2f} L/hr")
                else:
                     st.markdown("**Est. Vd & CL:** Could not be calculated accurately.")

                results_calculated = True

                # --- Dose Recommendation ---
                st.markdown("### Dose Adjustment Recommendation")
                if vd <= 0 or ke <=0:
                     st.warning("Cannot calculate new dose recommendation due to invalid PK parameters.")
                else:
                    # Ask for desired target levels (default to mid-point of range or target min)
                    default_desired_peak = target_peak_min if regimen_code == "SDD" else (target_peak_min + target_peak_max) / 2
                    desired_peak = st.number_input("Desired Target    peak_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Peak.*?([\d.,]+)\s*mg/L", prompt)
    auc_val = extract_float(r"(?:Estimated|Predicted)\s+AUC24.*?([\d.,]+)\s*mg.hr/L", prompt)

    # Extract targets
    target_auc_str = extract_string(r"Target\s+AUC24.*?(\d+\s*-\s*\d+)\s*mg.hr/L", prompt, "400-600")
    target_trough_str = extract_string(r"(?:Target|Secondary Target)\s+Trough.*?([\d.]+\s*-\s*[\d.]+)\s*mg/L", prompt, "10-15")

    # Extract current/new regimen details
    current_dose_interval = extract_float(r"Current\s+Regimen.*?([\d,]+)\s*mg.*?q\s*(\d+)", prompt)
    current_interval = extract_float(r"Current\s+Regimen.*?q\s*(\d+)", prompt)
    new_dose_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?([\d,]+)\s*mg.*?q\s*(\d+)", prompt)
    new_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?q\s*(\d+)", prompt)


    # Parse target ranges
    auc_target_min, auc_target_max = 400, 600
    auc_match = re.match(r"(\d+)\s*-\s*(\d+)", target_auc_str)
    if auc_match: auc_target_min, auc_target_max = int(auc_match.group(1)), int(auc_match.group(2))
    auc_target_formatted = f"{auc_target_min}-{auc_target_max} mg·hr/L"

    trough_target_min, trough_target_max = 10, 15
    trough_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_trough_str)
    if trough_match:
        try:
            trough_target_min = float(trough_match.group(1))
            trough_target_max = float(trough_match.group(2))
        except ValueError: pass
    trough_target_formatted = f"{trough_target_min:.1f}-{trough_target_max:.1f} mg/L"


    # Check if essential values for assessment were extracted
    if trough_val is None and auc_val is None:
        return "Insufficient level data (Trough or AUC) in prompt for standardized vancomycin interpretation."

    # --- Start Interpretation Logic ---
    levels_data = []
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    status = "assessment uncertain" # Default status

    # Assess Trough Level
    trough_status = "N/A"
    if trough_val is not None:
        if trough_val < trough_target_min: trough_status = "below"
        elif trough_val > trough_target_max: trough_status = "above"
        else: trough_status = "within"
        levels_data.append(("Trough", trough_val, trough_target_formatted, trough_status))

    # Assess AUC Level
    auc_status = "N/A"
    if auc_val is not None:
        if auc_val < auc_target_min: auc_status = "below"
        elif auc_val > auc_target_max: auc_status = "above"
        else: auc_status = "within"
        levels_data.append(("AUC24", auc_val, auc_target_formatted, auc_status))

    # Assess Peak Level (if available)
    peak_status = "N/A"
    if peak_val is not None:
        # Define peak range based on empiric vs definitive therapy
        # Assuming trough level helps determine empiric vs definitive
        if trough_val is not None and trough_val <= 15:  # Likely empiric therapy
            peak_target_min, peak_target_max = 20, 30
            peak_target_formatted = f"{peak_target_min}-{peak_target_max} mg/L (Empiric)"
        else:  # Likely definitive therapy
            peak_target_min, peak_target_max = 25, 40
            peak_target_formatted = f"{peak_target_min}-{peak_target_max} mg/L (Definitive)"
        
        if peak_val < peak_target_min: peak_status = "below"
        elif peak_val > peak_target_max: peak_status = "above"
        else: peak_status = "within"
        levels_data.append(("Peak", peak_val, peak_target_formatted, peak_status))


    # Determine overall assessment status (prioritize AUC, then Trough)
    if auc_status == "within" and trough_status != "above": status = "appropriately dosed (AUC target met)"
    elif auc_status == "within" and trough_status == "above": status = "potentially overdosed (AUC ok, trough high)"
    elif auc_status == "below": status = "underdosed (AUC below target)"
    elif auc_status == "above": status = "overdosed (AUC above target)"
    elif auc_status == "N/A": # If AUC not available, use trough
         if trough_status == "within": status = "likely appropriately dosed (trough target met)"
         elif trough_status == "below": status = "likely underdosed (trough below target)"
         elif trough_status == "above": status = "likely overdosed (trough above target)"


    # Generate recommendations based on status
    if "appropriately dosed" in status:
        dosing_recs.append("CONTINUE current regimen.")
        if current_dose_interval and current_interval:
             dosing_recs.append(f"MAINTAIN {current_dose_interval:.0f}mg q{current_interval:.0f}h.")
        monitoring_recs.append("MONITOR renal function per protocol (e.g., 2-3 times weekly).")
        monitoring_recs.append("REPEAT levels if clinical status or renal function changes significantly.")
        if status == "potentially overdosed (AUC ok, trough high)": # Add caution if trough high despite AUC ok
             cautions.append("Trough is elevated, increasing potential nephrotoxicity risk despite acceptable AUC. Monitor renal function closely.")
             monitoring_recs.append("Consider rechecking trough sooner if renal function declines.")

    else: # Underdosed or Overdosed
        if "underdosed" in status:
             dosing_recs.append("INCREASE dose and/or shorten interval to achieve target AUC.")
             monitoring_recs.append("RECHECK levels after 3-5 doses of new regimen (allow steady state).")
        elif "overdosed" in status:
             if trough_val is not None and trough_val > 25: # Significantly high trough
                 dosing_recs.append("HOLD next dose(s) until trough is acceptable (e.g., < 20 mg/L).")
                 cautions.append("Significantly elevated trough increases nephrotoxicity risk.")
             dosing_recs.append("DECREASE dose and/or lengthen interval to achieve target AUC.")
             monitoring_recs.append("RECHECK levels within 24-48 hours after adjustment (or before next dose if interval long).")
             monitoring_recs.append("MONITOR renal function daily until stable.")

        # Suggest new regimen if provided in prompt
        if new_dose_interval and new_interval:
             # Suggest practical regimens based on new TDD
             new_tdd_calc = new_dose_interval * (24 / new_interval)
             suggested_regimens = []
             for practical_interval_opt in [8, 12, 24, 36, 48]: # Common intervals
                 dose_per_interval_opt = new_tdd_calc / (24 / practical_interval_opt)
                 # Round dose per interval to nearest 250mg
                 rounded_dose_opt = round(dose_per_interval_opt / 250) * 250
                 if rounded_dose_opt > 0:
                     # Check if this option is close to the suggested one
                     is_suggested = abs(practical_interval_opt - new_interval) < 1 and abs(rounded_dose_opt - new_dose_interval) < 125
                     prefix = "➡️" if is_suggested else "  -"
                     suggested_regimens.append(f"{prefix} {rounded_dose_opt:.0f}mg q{practical_interval_opt}h (approx. {rounded_dose_opt * (24/practical_interval_opt):.0f}mg/day)")

             if suggested_regimens:
                 dosing_recs.append(f"ADJUST regimen towards target AUC ({auc_target_formatted}). Consider practical options:")
                 # Add the explicitly suggested regimen first if found
                 explicit_suggestion = f"{new_dose_interval:.0f}mg q{new_interval:.0f}h"
                 if not any(explicit_suggestion in reg for reg in suggested_regimens):
                      dosing_recs.append(f"➡️ {explicit_suggestion} (Calculated)") # Add if not already covered by rounding
                 for reg in suggested_regimens:
                     dosing_recs.append(reg)

             else: # Fallback if no practical options generated
                  dosing_recs.append(f"ADJUST regimen to {new_dose_interval:.0f}mg q{new_interval:.0f}h as calculated.")
        else: # If no new dose calculated in prompt
             dosing_recs.append("ADJUST regimen based on clinical judgment and estimated PK to achieve target AUC.")


    # Add renal function caution if relevant
    if crcl is not None:
        renal_status = ""
        if crcl < 15: renal_status = "Kidney Failure"
        elif crcl < 30: renal_status = "Severe Impairment"
        elif crcl < 60: renal_status = "Moderate Impairment"
        elif crcl < 90: renal_status = "Mild Impairment"

        if crcl < 60: # Add caution for moderate to severe impairment
            cautions.append(f"{renal_status} (CrCl: {crcl:.1f} mL/min). Increased risk of accumulation and toxicity. Monitor levels and renal function closely.")
            if "overdosed" in status or (trough_val is not None and trough_val > target_trough_max):
                 monitoring_recs.append("MONITOR renal function at least daily.")
            else:
                 monitoring_recs.append("MONITOR renal function frequently (e.g., every 1-2 days).")

    cautions.append("Ensure appropriate infusion duration (e.g., ≥ 1 hour per gram, max rate 1g/hr) to minimize infusion reactions.")
    cautions.append("Consider potential drug interactions affecting vancomycin clearance or toxicity (e.g., piperacillin-tazobactam, loop diuretics, other nephrotoxins).")


    return levels_data, status, dosing_recs, monitoring_recs, cautions


def generate_aminoglycoside_interpretation(prompt, crcl=None):
    """
    Generate standardized aminoglycoside interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract drug name
    drug_match = re.search(r"Drug:\s*(Gentamicin|Amikacin)", prompt, re.IGNORECASE)
    drug_name = drug_match.group(1).lower() if drug_match else "aminoglycoside"

    # Extract levels (measured or estimated)
    peak_val = extract_float(r"(?:Measured|Estimated|Predicted|Est\.)\s+(?:Peak|Cmax).*?([\d.,]+)\s*mg/L", prompt)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted|Est\.)\s+(?:Trough|Cmin|C1).*?([\d.,]+)\s*mg/L", prompt) # Allow C1 as trough

    # Extract targets
    target_peak_str = extract_string(r"Target\s+Peak.*?((?:[\d.]+\s*-\s*[\d.]+|[<>]?\s*[\d.]+|N/A|Not routinely targeted))\s*mg/L", prompt, "N/A")
    target_trough_str = extract_string(r"Target\s+Trough.*?((?:[\d.]+\s*-\s*[\d.]+|[<>]?\s*[\d.]+|N/A))\s*mg/L", prompt, "N/A")

    # Extract current/new regimen details
    current_dose = extract_float(r"Current\s+Regimen.*?([\d,]+)\s*mg", prompt)
    current_interval = extract_float(r"Current\s+Regimen.*?q\s*(\d+)", prompt)
    new_dose = extract_float(r"(?:Suggested|New)\s+Regimen.*?([\d,]+)\s*mg", prompt)
    new_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?q\s*(\d+)", prompt)


    # --- Parse Target Ranges ---
    peak_min, peak_max = 0, 100 # Default wide range
    trough_limit_type = "max" # Assume target is '< max' by default
    trough_max = 100 # Default wide range

    # Parse Peak Target String
    if "N/A" in target_peak_str or "not targeted" in target_peak_str:
        peak_min, peak_max = None, None # Indicate not applicable
    else:
        peak_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_peak_str)
        if peak_match:
            try: peak_min, peak_max = float(peak_match.group(1)), float(peak_match.group(2))
            except ValueError: pass # Keep defaults if parsing fails

    # Parse Trough Target String
    if "N/A" in target_trough_str:
        trough_max = None # Indicate not applicable
    else:
        trough_match_less = re.match(r"<\s*([\d.]+)", target_trough_str)
        trough_match_range = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_trough_str)
        if trough_match_less:
            try: trough_max = float(trough_match_less.group(1)); trough_limit_type = "max"
            except ValueError: pass
        elif trough_match_range: # Handle if a range is given for trough (less common for amino)
             try: trough_max = float(trough_match_range.group(2)); trough_limit_type = "range"; trough_min = float(trough_match_range.group(1))
             except ValueError: pass # Default to max limit if range parsing fails


    # Check if essential level values were extracted
    if peak_val is None or trough_val is None:
        # Allow interpretation if only trough is available for HD patients
        if not ("Hemodialysis" in prompt and trough_val is not None):
             return "Insufficient level data (Peak or Trough) in prompt for standardized aminoglycoside interpretation."


    # --- Start Interpretation Logic ---
    levels_data = []
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    status = "assessment uncertain" # Default status

    # Assess Peak Level
    peak_status = "N/A"
    if peak_min is not None and peak_max is not None and peak_val is not None:
        if peak_val < peak_min: peak_status = "below"
        elif peak_val > peak_max: peak_status = "above"
        else: peak_status = "within"
        levels_data.append(("Peak", peak_val, target_peak_str, peak_status))
    elif peak_val is not None: # If target is N/A but value exists
         levels_data.append(("Peak", peak_val, target_peak_str, "N/A"))


    # Assess Trough Level
    trough_status = "N/A"
    if trough_max is not None and trough_val is not None:
        if trough_limit_type == "max":
            if trough_val >= trough_max: trough_status = "above" # At or above the max limit
            else: trough_status = "within" # Below the max limit
        elif trough_limit_type == "range":
             if trough_val < trough_min: trough_status = "below" # Below the range min (unlikely target for amino)
             elif trough_val > trough_max: trough_status = "above" # Above the range max
             else: trough_status = "within"
        levels_data.append(("Trough", trough_val, target_trough_str, trough_status))
    elif trough_val is not None: # If target is N/A but value exists
        levels_data.append(("Trough", trough_val, target_trough_str, "N/A"))


    # Determine overall assessment status
    # Prioritize avoiding toxicity (high trough), then achieving efficacy (adequate peak)
    if trough_status == "above":
        status = "potentially toxic (elevated trough)"
        if peak_status == "below": status = "ineffective and potentially toxic" # Worst case
    elif peak_status == "below":
        status = "subtherapeutic (inadequate peak)"
    elif peak_status == "above": # Peak high, trough ok
        status = "potentially supratherapeutic (high peak)"
    elif peak_status == "within" and trough_status == "within":
        status = "appropriately dosed"
    elif peak_status == "N/A" and trough_status == "within": # e.g., HD patient trough ok
         status = "likely appropriate (trough acceptable)"
    elif peak_status == "N/A" and trough_status == "above": # e.g., HD patient trough high
         status = "potentially toxic (elevated trough)"


    # Generate recommendations
    if "appropriately dosed" in status or "likely appropriate" in status :
        dosing_recs.append("CONTINUE current regimen.")
        if current_dose and current_interval: dosing_recs.append(f"MAINTAIN {current_dose:.0f}mg q{current_interval:.0f}h.")
        monitoring_recs.append("MONITOR renal function regularly (e.g., 2-3 times weekly or per HD schedule).")
        monitoring_recs.append("REPEAT levels if clinical status, renal function, or dialysis schedule changes.")
    elif status == "assessment not applicable": # Synergy Amikacin
         dosing_recs.append("Follow specific institutional protocol for Synergy Amikacin dosing.")
         monitoring_recs.append("MONITOR renal function and clinical status.")
    else: # Adjustments needed
        if status == "ineffective and potentially toxic":
             dosing_recs.append("HOLD next dose(s).")
             dosing_recs.append("INCREASE dose AND EXTEND interval significantly once resumed.")
             monitoring_recs.append("RECHECK levels (peak & trough) before resuming and after 2-3 doses of new regimen.")
             cautions.append("High risk of toxicity and low efficacy with current levels.")
        elif status == "subtherapeutic (inadequate peak)":
             dosing_recs.append("INCREASE dose.")
             dosing_recs.append("MAINTAIN current interval (unless trough also borderline high).")
             monitoring_recs.append("RECHECK peak and trough after 1-2 doses of new regimen.")
             cautions.append("Inadequate peak may compromise efficacy, especially for gram-negative infections.")
        elif status == "potentially toxic (elevated trough)":
             dosing_recs.append("EXTEND dosing interval.")
             dosing_recs.append("MAINTAIN current dose amount (or consider slight reduction if peak also high/borderline).")
             monitoring_recs.append("RECHECK trough level before next scheduled dose.")
             cautions.append("Elevated trough increases risk of nephrotoxicity and ototoxicity. Hold dose if trough significantly elevated.")
        elif status == "potentially supratherapeutic (high peak)": # High peak, trough ok
             dosing_recs.append("DECREASE dose.")
             dosing_recs.append("MAINTAIN current interval.")
             monitoring_recs.append("RECHECK peak and trough after 1-2 doses of new regimen.")
             cautions.append("Elevated peak may increase toxicity risk slightly, though trough is primary driver. Ensure trough remains acceptable.")

        # Suggest new regimen if provided in prompt
        if new_dose and new_interval:
             # Round new dose to nearest 10mg or 20mg
             rounding = 20 if drug_name == "gentamicin" else 50 if drug_name == "amikacin" else 10
             practical_new_dose = round(new_dose / rounding) * rounding
             if practical_new_dose > 0:
                 dosing_recs.append(f"Consider adjusting regimen towards: {practical_new_dose:.0f}mg q{new_interval:.0f}h.")
        else:
             dosing_recs.append("ADJUST regimen based on clinical judgment and estimated PK to achieve target levels.")


    # Add general monitoring and cautions
    monitoring_recs.append("MONITOR renal function (SCr, BUN, UOP) at least 2-3 times weekly, or more frequently if unstable, trough elevated, or on concomitant nephrotoxins.")
    monitoring_recs.append("MONITOR for signs/symptoms of nephrotoxicity (rising SCr, decreased UOP) and ototoxicity (hearing changes, tinnitus, vertigo).")
    cautions.append(f"{drug_name.capitalize()} carries risk of nephrotoxicity and ototoxicity.")
    cautions.append("Risk increases with prolonged therapy (>7-10 days), pre-existing renal impairment, high troughs, large cumulative dose, and concomitant nephrotoxins (e.g., vancomycin, diuretics, contrast).")
    if crcl is not None:
         renal_status = ""
         if crcl < 15: renal_status = "Kidney Failure"
         elif crcl < 30: renal_status = "Severe Impairment"
         elif crcl < 60: renal_status = "Moderate Impairment"
         elif crcl < 90: renal_status = "Mild Impairment"
         if crcl < 60:
             cautions.append(f"{renal_status} (CrCl: {crcl:.1f} mL/min) significantly increases toxicity risk. Adjust dose/interval carefully and monitor very closely.")


    return levels_data, status, dosing_recs, monitoring_recs, cautions


# ===== SIDEBAR: NAVIGATION AND PATIENT INFO =====
def setup_sidebar_and_navigation():
    st.sidebar.title("📊 Navigation")
    # Sidebar radio for selecting the module
    page = st.sidebar.radio("Select Module", [
        "Aminoglycoside: Initial Dose",
        "Aminoglycoside: Conventional Dosing (C1/C2)",
        "Vancomycin AUC-based Dosing"
    ])

    st.sidebar.title("🩺 Patient Demographics")
    # ADDED Patient ID and Ward
    patient_id = st.sidebar.text_input("Patient ID", value="N/A")
    ward = st.sidebar.text_input("Ward", value="N/A")
    # --- Existing fields ---
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=65)
    height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=165)
    weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1, format="%.1f")
    serum_cr = st.sidebar.number_input("Serum Creatinine (µmol/L)", min_value=10.0, max_value=2000.0, value=90.0, step=1.0)

    # Calculate Cockcroft-Gault Creatinine Clearance
    crcl = 0.0 # Default value
    renal_function = "N/A"
    if age > 0 and weight > 0 and serum_cr > 0: # Avoid division by zero or negative age
        # Cockcroft-Gault Formula
        crcl_factor = (140 - age) * weight
        crcl_gender_mult = 1.23 if gender == "Male" else 1.04
        crcl = (crcl_factor * crcl_gender_mult) / serum_cr
        crcl = max(0, crcl) # Ensure CrCl is not negative

        # Renal function category based on CrCl
        if crcl >= 90: renal_function = "Normal (≥90)"
        elif crcl >= 60: renal_function = "Mild Impairment (60-89)"
        elif crcl >= 30: renal_function = "Moderate Impairment (30-59)"
        elif crcl >= 15: renal_function = "Severe Impairment (15-29)"
        else: renal_function = "Kidney Failure (<15)"

    with st.sidebar.expander("Creatinine Clearance (Cockcroft-Gault)", expanded=True):
        if age > 0 and weight > 0 and serum_cr > 0:
            st.success(f"CrCl: {crcl:.1f} mL/min")
            st.info(f"Renal Function: {renal_function}")
        else:
            st.warning("Enter valid Age (>0), Weight (>0), and SCr (>0) to calculate CrCl.")


    st.sidebar.title("🩺 Clinical Information")
    clinical_diagnosis = st.sidebar.text_input("Diagnosis / Indication", placeholder="e.g., Pneumonia, Sepsis")
    current_dose_regimen = st.sidebar.text_area("Current Dosing Regimen", value="1g IV q12h", placeholder="e.g., Gentamicin 120mg IV q8h")
    notes = st.sidebar.text_area("Other Clinical Notes", value="No known allergies.", placeholder="e.g., Fluid status, interacting meds")

    # UPDATED clinical_summary
    clinical_summary = (
        f"Patient ID: {patient_id}, Ward: {ward}\n"
        f"Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} cm\n"
        f"SCr: {serum_cr} µmol/L\n"
        f"Diagnosis: {clinical_diagnosis}\n"
        f"Renal function: {renal_function} (Est. CrCl: {crcl:.1f} mL/min)\n"
        f"Current regimen: {current_dose_regimen}\n"
        f"Notes: {notes}"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Antimicrobial TDM App v1.2**

    Developed for therapeutic drug monitoring of antimicrobials.

    Provides PK estimates, AUC calculations, and dosing recommendations
    for vancomycin and aminoglycosides. Includes optional LLM interpretation.

    **Disclaimer:** This tool assists clinical decision making but does not replace
    professional judgment. Verify all calculations and recommendations.
    """)

    # Return all the data entered in the sidebar
    return {
        'page': page,
        'patient_id': patient_id, # Added
        'ward': ward,           # Added
        'gender': gender,
        'age': age,
        'height': height,
        'weight': weight,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen,
        'notes': notes,
        'clinical_summary': clinical_summary # Updated summary string
    }

# ===== MODULE 1: Aminoglycoside Initial Dose =====
def aminoglycoside_initial_dose(patient_data):
    st.title("🧮 Aminoglycoside Initial Dose (Population PK)")

    # Unpack patient data for easier access
    gender = patient_data['gender']
    age = patient_data['age']
    height = patient_data['height']
    weight = patient_data['weight']
    crcl = patient_data['crcl']
    notes = patient_data['notes']

    drug = st.selectbox("Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Dosing Strategy / Goal", ["Extended Interval (Once Daily - SDD)", "Traditional (Multiple Daily - MDD)", "Synergy (e.g., Endocarditis)", "Hemodialysis", "Neonates (Use with caution)"])

    # Map selection to internal codes
    regimen_code = "SDD" if "Extended" in regimen \
              else "MDD" if "Traditional" in regimen \
              else "Synergy" if "Synergy" in regimen \
              else "Hemodialysis" if "Hemodialysis" in regimen \
              else "Neonates" if "Neonates" in regimen \
              else "MDD" # Default

    # --- Set default target ranges based on regimen and drug ---
    target_peak_info = "N/A"
    target_trough_info = "N/A"
    default_peak = 0.0
    default_trough = 0.0

    if drug == "Gentamicin":import streamlit as st
import numpy as np
import math
import openai
import pandas as pd
import altair as alt
import base64
from datetime import datetime, time, timedelta # Added time and timedelta

# Optional imports - Bayesian functionality
try:
    import scipy.optimize as optimize
    from scipy.stats import norm
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scipy not installed. Bayesian estimation will not be available.")

# Optional imports - FAISS for guideline embedding (if needed)
try:
    import faiss
    import pickle
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Guideline embeddings will not be available.")

st.set_page_config(page_title="Antimicrobial TDM App", layout="wide")

# ===== API CONFIGURATION =====
# Securely access the API key from streamlit secrets
try:
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.

    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== LOAD EMBEDDED GUIDELINE =====
@st.cache_resource
def load_guideline_embeddings():
    if not FAISS_AVAILABLE:
        return None, ["FAISS not installed. Guideline embeddings not available."]

    # Replace this with your actual FAISS index and guideline chunks
    index = faiss.IndexFlatL2(768)  # Dummy placeholder index
    chunks = ["Guideline excerpt 1", "Guideline excerpt 2", "Guideline excerpt 3"]
    return index, chunks

guideline_index, guideline_chunks = None, []
if FAISS_AVAILABLE:
    guideline_index, guideline_chunks = load_guideline_embeddings()

# ===== Global Practical Dosing Regimens =====
practical_intervals = "6hr, 8hr, 12hr, 24hr, every other day"

# ===== HELPER FUNCTION: Updated Practical Dose Adjustment =====
def suggest_adjustment(parameter, target_min, target_max, label="Parameter", intervals=practical_intervals):
    # Ensure targets are valid numbers before comparison
    if isinstance(target_min, (int, float)) and isinstance(target_max, (int, float)) and isinstance(parameter, (int, float)):
        if parameter < target_min:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is low. Target: {target_min:.1f}-{target_max:.1f}. Consider increasing dose or shortening interval ({intervals}).")
        elif parameter > target_max:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is high. Target: {target_min:.1f}-{target_max:.1f}. Consider reducing dose or lengthening interval ({intervals}).")
        else:
            st.success(f"✅ {label} ({parameter:.1f}) is within target range ({target_min:.1f}-{target_max:.1f}).")
    else:
        st.info(f"{label}: {parameter}. Target range: {target_min}-{target_max}. (Comparison skipped due to non-numeric values).")


# ===== PDF GENERATION FUNCTIONS (REMOVED) =====
# create_recommendation_pdf, get_pdf_download_link, display_pdf_download_button functions removed.

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization

    Parameters:
    - peak: Peak concentration (mg/L)
    - trough: Trough concentration (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - t_peak: Time to peak after start of infusion (hr) - assumed end of infusion
    - infusion_time: Duration of infusion (hr)

    Returns:
    - Altair chart object
    """
    # Generate time points for the curve
    times = np.linspace(0, tau*1.5, 150)  # Generate points for 1.5 intervals to show next dose

    # Generate concentrations for each time point using steady-state equations
    concentrations = []
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * T_inf)) * exp(-ke * (t - T_inf)) / (1 - exp(-ke * tau)) -- Post-infusion
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * t)) / (1 - exp(-ke * tau)) -- During infusion (simplified, assumes Cmin=0 start)
    # Let's use the provided peak and trough which represent Cmax (at t=infusion_time) and Cmin (at t=tau)

    for t_cycle in np.linspace(0, tau*1.5, 150): # Iterate through time points
        # Determine concentration based on time within the dosing cycle (modulo tau)
        t = t_cycle % tau
        num_cycles = int(t_cycle // tau) # Which cycle we are in (0, 1, ...)

        conc = 0
        if t <= infusion_time:
            # During infusion: Assume linear rise from previous trough to current peak
            # This is an approximation but visually represents the infusion period
            conc = trough + (peak - trough) * (t / infusion_time)
        else:
            # After infusion: Exponential decay from peak
            time_since_peak = t - infusion_time # Time elapsed since the peak concentration (end of infusion)
            conc = peak * math.exp(-ke * time_since_peak)

        concentrations.append(max(0, conc)) # Ensure concentration doesn't go below 0


    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Time (hr)': times,
        'Concentration (mg/L)': concentrations
    })

    # --- Create Target Bands ---
    target_bands = []
    # Determine drug type based on typical levels for band coloring
    if peak > 45 or trough > 20:  # Likely vancomycin
        # Vancomycin Peak Target - Empiric vs Definitive
        if trough <= 15:  # Likely empiric (target trough 10-15)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [20], 'y2': [30]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Empiric)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [10], 'y2': [15]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Empiric)")))
        else:  # Likely definitive (target trough 15-20)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [25], 'y2': [40]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Definitive)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [15], 'y2': [20]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Definitive)")))
    else:  # Likely aminoglycoside (e.g., Gentamicin)
        # Aminoglycoside Peak Target (e.g., 5-10 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [5], 'y2': [10]}))
                           .mark_rect(opacity=0.15, color='lightblue')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Amino)")))
        # Aminoglycoside Trough Target (e.g., <2 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [0], 'y2': [2]}))
                           .mark_rect(opacity=0.15, color='lightgreen')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Amino)")))


    # --- Create Concentration Line ---
    line = alt.Chart(df).mark_line(color='firebrick').encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)', scale=alt.Scale(zero=True)), # Ensure Y axis starts at 0
        tooltip=['Time (hr)', alt.Tooltip('Concentration (mg/L)', format=".1f")]
    )

    # --- Add Vertical Lines for Key Events ---
    vertical_lines_data = []
    # Mark end of infusion for each cycle shown
    for i in range(int(tau*1.5 / tau) + 1):
        inf_end_time = i * tau + infusion_time
        if inf_end_time <= tau*1.5:
             vertical_lines_data.append({'Time': inf_end_time, 'Event': 'Infusion End'})
    # Mark start of next dose for each cycle shown
    for i in range(1, int(tau*1.5 / tau) + 1):
         dose_time = i * tau
         if dose_time <= tau*1.5:
              vertical_lines_data.append({'Time': dose_time, 'Event': 'Next Dose'})

    vertical_lines_df = pd.DataFrame(vertical_lines_data)

    vertical_rules = alt.Chart(vertical_lines_df).mark_rule(strokeDash=[4, 4]).encode(
        x='Time',
        color=alt.Color('Event', scale=alt.Scale(domain=['Infusion End', 'Next Dose'], range=['gray', 'black'])),
        tooltip=['Event', 'Time']
    )

    # --- Combine Charts ---
    chart = alt.layer(*target_bands, line, vertical_rules).properties(
        width=alt.Step(4), # Adjust width automatically
        height=400,
        title=f'Estimated Concentration-Time Profile (Tau={tau} hr)'
    ).interactive() # Make chart interactive (zoom/pan)

    return chart


# ===== VANCOMYCIN AUC CALCULATION (TRAPEZOIDAL METHOD) =====
def calculate_vancomycin_auc_trapezoidal(cmax, cmin, ke, tau, infusion_duration):
    """
    Calculate vancomycin AUC24 using the linear-log trapezoidal method.
    
    This method is recommended for vancomycin TDM as per the guidelines.
    
    Parameters:
    - cmax: Max concentration at end of infusion (mg/L)
    - cmin: Min concentration at end of interval (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - infusion_duration: Duration of infusion (hr)
    
    Returns:
    - AUC24: 24-hour area under the curve (mg·hr/L)
    """
    # Calculate concentration at start of infusion (C0)
    c0 = cmax * math.exp(ke * infusion_duration)
    
    # Calculate AUC during infusion phase (linear trapezoid)
    auc_inf = infusion_duration * (c0 + cmax) / 2
    
    # Calculate AUC during elimination phase (log trapezoid)
    if ke > 0 and cmax > cmin:
        auc_elim = (cmax - cmin) / ke
    else:
        # Fallback to linear trapezoid if ke is very small
        auc_elim = (tau - infusion_duration) * (cmax + cmin) / 2
    
    # Calculate total AUC for one dosing interval
    auc_interval = auc_inf + auc_elim
    
    # Convert to AUC24
    auc24 = auc_interval * (24 / tau)
    
    return auc24

# ===== BAYESIAN PARAMETER ESTIMATION =====
def bayesian_parameter_estimation(measured_levels, sample_times, dose, tau, weight, age, crcl, gender):
    """
    Bayesian estimation of PK parameters based on measured levels

    Parameters:
    - measured_levels: List of measured drug concentrations (mg/L)
    - sample_times: List of times when samples were taken (hr after dose start)
    - dose: Dose administered (mg)
    - tau: Dosing interval (hr)
    - weight: Patient weight (kg)
    - age: Patient age (years)
    - crcl: Creatinine clearance (mL/min)
    - gender: Patient gender ("Male" or "Female")

    Returns:
    - Dictionary with estimated PK parameters
    """
    if not BAYESIAN_AVAILABLE:
        st.error("Bayesian estimation requires scipy. Please install it with 'pip install scipy'")
        return None

    # Prior population parameters for vancomycin (adjust if needed for aminoglycosides)
    # Mean values
    vd_pop_mean = 0.7  # L/kg (Vancomycin specific, adjust for aminoglycosides if used)
    ke_pop_mean = 0.00083 * crcl + 0.0044 # hr^-1 (Vancomycin specific - ensure crcl is used correctly)
    ke_pop_mean = max(0.01, ke_pop_mean) # Ensure Ke isn't too low

    # Standard deviations for population parameters
    vd_pop_sd = 0.2  # L/kg
    ke_pop_sd = 0.05 # Increased SD for Ke prior to allow more flexibility

    # Define objective function to minimize (negative log likelihood)
    def objective_function(params):
        vd_ind, ke_ind = params # Individual parameters to estimate
        vd_total = vd_ind * weight

        # Calculate expected concentrations at sample times using steady-state infusion model
        expected_concs = []
        infusion_time = 1.0 # Assume 1 hour infusion, make adjustable if needed

        for t in sample_times:
            # Steady State Concentration Equation (1-compartment, intermittent infusion)
            term_dose_vd = dose / vd_total
            term_ke_tinf = ke_ind * infusion_time
            term_ke_tau = ke_ind * tau

            try:
                exp_ke_tinf = math.exp(-term_ke_tinf)
                exp_ke_tau = math.exp(-term_ke_tau)

                if abs(1.0 - exp_ke_tau) < 1e-9: # Avoid division by zero if tau is very long or ke very small
                    # Handle as if continuous infusion or single dose if tau is effectively infinite
                    conc = 0 # Simplified - needs better handling for edge cases
                else:
                    common_factor = (term_dose_vd / term_ke_tinf) * (1.0 - exp_ke_tinf) / (1.0 - exp_ke_tau)

                    if t <= infusion_time: # During infusion phase
                        conc = common_factor * (1.0 - math.exp(-ke_ind * t))
                    else: # Post-infusion phase
                        conc = common_factor * math.exp(-ke_ind * (t - infusion_time))

            except OverflowError:
                 conc = float('inf') # Handle potential overflow with large ke/t values
            except ValueError:
                 conc = 0 # Handle math domain errors

            expected_concs.append(max(0, conc)) # Ensure non-negative

        # Calculate negative log likelihood
        # Measurement error model (e.g., proportional + additive)
        # sd = sqrt(sigma_add^2 + (sigma_prop * expected_conc)^2)
        sigma_add = 1.0  # Additive SD (mg/L)
        sigma_prop = 0.1 # Proportional SD (10%)
        nll = 0
        for i in range(len(measured_levels)):
            expected = expected_concs[i]
            measurement_sd = math.sqrt(sigma_add**2 + (sigma_prop * expected)**2)
            if measurement_sd < 1e-6: measurement_sd = 1e-6 # Prevent division by zero in logpdf

            # Add contribution from measurement likelihood
            # Use logpdf for robustness, especially with low concentrations
            nll += -norm.logpdf(measured_levels[i], loc=expected, scale=measurement_sd)

        # Add contribution from parameter priors (log scale often more stable for Ke)
        # Prior for Vd (Normal)
        nll += -norm.logpdf(vd_ind, loc=vd_pop_mean, scale=vd_pop_sd)
        # Prior for Ke (Log-Normal might be better, but using Normal for simplicity)
        nll += -norm.logpdf(ke_ind, loc=ke_pop_mean, scale=ke_pop_sd)

        # Penalize non-physical parameters slightly if optimization strays
        if vd_ind <= 0 or ke_ind <= 0:
             nll += 1e6 # Add large penalty

        return nll

    # Initial guess based on population values
    initial_params = [vd_pop_mean, ke_pop_mean]

    # Parameter bounds (physical constraints)
    bounds = [(0.1, 2.5), (0.001, 0.5)]  # Reasonable bounds for Vd (L/kg) and Ke (hr^-1)

    # Perform optimization using a robust method
    try:
        result = optimize.minimize(
            objective_function,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B', # Suitable for bound constraints
            options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 500} # Adjust tolerances/iterations
        )
    except Exception as e:
         st.error(f"Optimization failed: {e}")
         return None

    if not result.success:
        st.warning(f"Bayesian optimization did not converge: {result.message} (Function evaluations: {result.nfev})")
        # Optionally return population estimates or None
        return None # Indicate failure

    # Extract optimized parameters
    vd_opt_kg, ke_opt = result.x
    # Ensure parameters are within bounds post-optimization (should be handled by L-BFGS-B, but double-check)
    vd_opt_kg = max(bounds[0][0], min(bounds[0][1], vd_opt_kg))
    ke_opt = max(bounds[1][0], min(bounds[1][1], ke_opt))

    vd_total_opt = vd_opt_kg * weight
    cl_opt = ke_opt * vd_total_opt
    t_half_opt = 0.693 / ke_opt if ke_opt > 0 else float('inf')

    return {
        'vd': vd_opt_kg, # Vd per kg
        'vd_total': vd_total_opt, # Total Vd in L
        'ke': ke_opt,
        'cl': cl_opt,
        't_half': t_half_opt,
        'optimization_success': result.success,
        'final_nll': result.fun # Final negative log-likelihood value
    }


# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM.
    Uses OpenAI API if available, otherwise provides a simulated response.

    Parameters:
    - prompt: The clinical data prompt including calculated values and context.
    - patient_data: Dictionary with patient information (used for context).
    """
    # Extract the drug type from the prompt for context
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
    else:
        drug = "Antimicrobial"

    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Include patient context if relevant (e.g., renal function).
            Format your response with these exact sections:

            ## CLINICAL ASSESSMENT
            📊 **MEASURED/ESTIMATED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **ASSESSMENT:** (state if appropriately dosed, underdosed, or overdosed based on levels and targets)

            ## RECOMMENDATIONS
            🔵 **DOSING:** (action-oriented recommendations using verbs like CONTINUE, ADJUST, HOLD, INCREASE, DECREASE. Suggest practical regimens where possible.)
            🔵 **MONITORING:** (specific monitoring parameters and schedule, e.g., recheck levels, renal function)
            ⚠️ **CAUTIONS:** (relevant warnings, e.g., toxicity risk, renal impairment)

            Here is the case:
            --- Patient Context ---
            Age: {patient_data.get('age', 'N/A')} years, Gender: {patient_data.get('gender', 'N/A')}
            Weight: {patient_data.get('weight', 'N/A')} kg, Height: {patient_data.get('height', 'N/A')} cm
            Patient ID: {patient_data.get('patient_id', 'N/A')}, Ward: {patient_data.get('ward', 'N/A')}
            Serum Cr: {patient_data.get('serum_cr', 'N/A')} µmol/L, CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})
            Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}
            Current Regimen: {patient_data.get('current_dose_regimen', 'N/A')}
            Notes: {patient_data.get('notes', 'N/A')}
            --- TDM Data & Calculations ---
            {prompt}
            --- End of Case ---
            """

            # Call OpenAI API - updated for openai v1.0.0+
            response = openai.chat.completions.create(
                model="gpt-4",  # or your preferred model like gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear, actionable recommendations in the specified format."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3, # Lower temperature for more deterministic clinical advice
                max_tokens=600 # Increased token limit for detailed response
            )
            llm_response = response.choices[0].message.content

            st.subheader("Clinical Interpretation (LLM)")
            st.markdown(llm_response) # Display the formatted response directly
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")

            # No PDF generation needed here

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
            # Fall through to standardized interpretation if API fails

    # If OpenAI is not available/fails, use the standardized interpretation
    if not (OPENAI_AVAILABLE and openai.api_key): # Or if the API call failed above
        st.subheader("Clinical Interpretation (Simulated)")
        interpretation_data = generate_standardized_interpretation(prompt, drug, patient_data)

        # If the interpretation_data is a string (error message), just display it
        if isinstance(interpretation_data, str):
            st.write(interpretation_data)
            return

        # Unpack the interpretation data
        levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data

        # Display the formatted interpretation
        formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
        st.markdown(formatted_interpretation) # Use markdown for better formatting

        # Add note about simulated response
        st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")

    # Add the raw prompt at the bottom for debugging/transparency
    with st.expander("Raw Analysis Data Sent to LLM (or used for Simulation)", expanded=False):
        st.code(prompt)


def generate_standardized_interpretation(prompt, drug, patient_data):
    """
    Generate a standardized interpretation based on drug type and prompt content.
    Includes patient context for better recommendations.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    crcl = patient_data.get('crcl', None) # Get CrCl for context

    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt, crcl)
    elif drug == "Aminoglycoside":
        return generate_aminoglycoside_interpretation(prompt, crcl)
    else:
        # For generic, create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        if crcl and crcl < 60:
             cautions.append(f"Renal function (CrCl: {crcl:.1f} mL/min) may impact dosing.")

        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy using Markdown.

    Parameters:
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string (e.g., "appropriately dosed")
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - cautions: Optional list of caution strings

    Returns:
    - Formatted markdown string
    """
    # Format measured levels with status indicators
    levels_md = "📊 **MEASURED/ESTIMATED LEVELS:**\n"
    if not levels_data or (len(levels_data) == 1 and levels_data[0][0] == "Not available"):
         levels_md += "- No levels data available for interpretation.\n"
    else:
        for name, value, target, status in levels_data:
            icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴" # Red for above
            # Format value appropriately (e.g., 1 decimal for levels, 0 for AUC)
            value_str = f"{value:.1f}" if isinstance(value, (int, float)) and "AUC" not in name else f"{value:.0f}" if isinstance(value, (int, float)) else str(value)
            levels_md += f"- {name}: {value_str} (Target: {target}) {icon}\n"


    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is **{assessment.upper()}**"

    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## RECOMMENDATIONS

🔵 **DOSING:**
"""
    if dosing_recs:
        for rec in dosing_recs:
            output += f"- {rec}\n"
    else:
        output += "- No specific dosing recommendations generated.\n"

    output += "\n🔵 **MONITORING:**\n"
    if monitoring_recs:
        for rec in monitoring_recs:
            output += f"- {rec}\n"
    else:
        output += "- Standard monitoring applies.\n"

    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"

    return output

def generate_vancomycin_interpretation(prompt, crcl=None):
    """
    Generate standardized vancomycin interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    # Extract key values from the prompt using regex for robustness
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract levels (measured or estimated)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Trough.*?([\d.,]+)\s*mg/L", prompt)
    peak_val
    if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
                 prompt = (f"Vancomycin TDM Adjustment (Trough Only):\n"
                           f"Current Regimen: {current_dose_interval:.0f} mg IV q {current_interval:.0f}h (infused over {infusion_duration} hr)\n"
                           f"Measured Trough: {measured_trough:.1f} mg/L\n"
                           f"Target AUC24: {target_auc} mg·hr/L, Secondary Target Trough: {target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L\n"
                           f"Adjusted PK Estimates (using Pop Vd): Vd≈{vd_pop:.2f} L, CL≈{cl_adjusted:.2f} L/hr, Ke≈{ke_adjusted:.4f} hr⁻¹, t½≈{t_half_adjusted:.2f} hr\n"
                           f"Estimated Current AUC24: {current_auc24:.0f} mg·hr/L\n"
                           f"Suggested Adjustment: {practical_new_dose:.0f} mg IV q {desired_interval_adj:.0f}h\n"
                           f"Predicted Levels (New Regimen): Trough≈{predicted_trough_new:.1f} mg/L, AUC24≈{predicted_auc_new:.0f} mg·hr/L")
                 interpret_with_llm(prompt, patient_data)


    # --- Adjustment using Peak & Trough Levels ---
    elif "Peak & Trough Levels" in method:
        st.markdown("Adjusting dose based on measured peak and trough levels.")
        col1, col2 = st.columns(2)
        with col1:
            current_dose_interval = st.number_input("Current Dose per Interval (mg)", min_value=250.0, value=1000.0, step=50.0)
            current_interval = st.number_input("Current Dosing Interval (hr)", min_value=4, max_value=72, value=12, step=4)
            infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        with col2:
             # Use datetime inputs for precision - UPDATED
            default_date = datetime.now().date()
            # Set default dose time (e.g., 9:00 AM)
            default_dose_time = datetime.combine(default_date, time(9, 0))
            dose_start_datetime_dt = st.datetime_input(
                "Date & Time of Dose Start",
                value=default_dose_time,
                step=timedelta(minutes=15)
            )
            # Default trough time (e.g., 30 min before dose)
            default_trough_time = default_dose_time - timedelta(minutes=30)
            trough_sample_datetime_dt = st.datetime_input(
                "Date & Time of Trough Sample",
                value=default_trough_time,
                step=timedelta(minutes=15)
            )
            # Default peak time (e.g., 1.5 hr after dose start = 30 min post 1hr infusion)
            default_peak_time = dose_start_datetime_dt + timedelta(hours=infusion_duration + 0.5)
            peak_sample_datetime_dt = st.datetime_input(
                "Date & Time of Peak Sample",
                 value=default_peak_time,
                 step=timedelta(minutes=15)
            )

        col_l1, col_l2 = st.columns(2)
        with col_l1:
             measured_trough = st.number_input("Measured Trough Level (mg/L)", min_value=0.1, value=12.0, step=0.1, format="%.1f")
        with col_l2:
             measured_peak = st.number_input("Measured Peak Level (mg/L)", min_value=0.1, value=30.0, step=0.1, format="%.1f")

        # Calculate relative sample times - UPDATED
        t_trough_rel = (trough_sample_datetime_dt - dose_start_datetime_dt).total_seconds() / 3600.0
        t_peak_rel = (peak_sample_datetime_dt - dose_start_datetime_dt).total_seconds() / 3600.0

        st.markdown(f"*Time from dose start to Trough sample (t_trough): {t_trough_rel:.2f} hr*")
        st.markdown(f"*Time from dose start to Peak sample (t_peak): {t_peak_rel:.2f} hr*")

        # Validate timings
        valid_times = True
        if t_trough_rel >= t_peak_rel:
            st.error("❌ Trough sample time must be before Peak sample time.")
            valid_times = False
        if t_peak_rel <= infusion_duration:
             st.warning(f"⚠️ Peak sample time (t_peak={t_peak_rel:.2f} hr) is during or before the end of infusion ({infusion_duration:.1f} hr). Calculations will extrapolate Cmax; accuracy may be reduced.")
        # Check if trough is truly pre-dose (t_trough_rel should be negative or close to zero)
        if t_trough_rel > 0.1: # Allow small positive window (e.g., sample taken exactly at dose time)
             st.warning(f"⚠️ Trough sample time (t_trough={t_trough_rel:.2f} hr) is after the dose start time. Ensure this is the intended trough measurement (e.g., from previous dose cycle). Calculations assume steady state.")


        # Calculate Individual PK parameters (Ke, Vd, CL)
        ke_ind, t_half_ind, vd_ind, cl_ind = 0, float('inf'), 0, 0
        cmax_extrap, cmin_extrap = 0, 0
        results_calculated = False

        if valid_times:
            try:
                # Ensure levels are positive
                if measured_trough <= 0 or measured_peak <= 0:
                    raise ValueError("Measured levels must be positive.")

                # Calculate Ke from the two levels using log-linear decay between the sample times
                time_diff = t_peak_rel - t_trough_rel
                if time_diff <= 0: raise ValueError("Peak sample time must be chronologically after trough sample time.")

                # Ke = ln(C_earlier / C_later) / (t_later - t_earlier)
                ke_ind = (math.log(measured_trough) - math.log(measured_peak)) / time_diff
                ke_ind = max(1e-6, ke_ind) # Ensure positive Ke

                t_half_ind = 0.693 / ke_ind if ke_ind > 0 else float('inf')

                # Extrapolate Cmax (at end of infusion) and Cmin (at end of interval)
                # Requires knowing the time relative to the *end* of infusion for peak sample
                time_from_inf_end_to_peak = t_peak_rel - infusion_duration
                cmax_extrap = measured_peak * math.exp(ke_ind * time_from_inf_end_to_peak)

                # Cmin = Cmax * exp(-ke * (Tau - T_inf))
                cmin_extrap = cmax_extrap * math.exp(-ke_ind * (current_interval - infusion_duration))

                # Calculate Vd using steady-state infusion equation and extrapolated Cmax
                term_inf_ind = (1 - math.exp(-ke_ind * infusion_duration))
                term_int_ind = (1 - math.exp(-ke_ind * current_interval))
                denom_vd_ind = cmax_extrap * ke_ind * infusion_duration * term_int_ind
                vd_ind = 0.0
                if abs(denom_vd_ind) > 1e-9 and abs(term_inf_ind) > 1e-9:
                     vd_ind = (current_dose_interval * term_inf_ind) / denom_vd_ind
                     vd_ind = max(1.0, vd_ind) # Ensure Vd is at least 1L
                else:
                     st.warning("Could not calculate Vd accurately.")

                cl_ind = ke_ind * vd_ind if vd_ind > 0 else 0.0

                st.markdown("#### Individualized PK Parameters:")
                st.markdown(f"**Ind. Ke:** {ke_ind:.4f} hr⁻¹ | **t½:** {t_half_ind:.2f} hr")
                st.markdown(f"**Est. Cmax (end of infusion):** {cmax_extrap:.1f} mg/L | **Est. Cmin (end of interval):** {cmin_extrap:.1f} mg/L")
                if vd_ind > 0:
                     st.markdown(f"**Est. Vd:** {vd_ind:.2f} L | **Est. CL:** {cl_ind:.2f} L/hr")
                else:
                     st.markdown("**Est. Vd & CL:** Could not be calculated accurately.")

                # Calculate current AUC24 using trapezoidal method with extrapolated values
                current_auc24_ind = 0
                if cmax_extrap > 0 and cmin_extrap >= 0 and ke_ind > 0 and current_interval > 0:
                    current_auc24_ind = calculate_vancomycin_auc_trapezoidal(
                        cmax_extrap, cmin_extrap, ke_ind, current_interval, infusion_duration
                    )
                else:
                    # Fallback to simple calculation if extrapolation failed
                    current_dose_daily = current_dose_interval * (24 / current_interval)
                    current_auc24_ind = current_dose_daily / cl_ind if cl_ind > 0 else 0

                st.markdown("#### Current Regimen Assessment:")
                st.markdown(f"Measured Peak: **{measured_peak:.1f} mg/L** (@ {t_peak_rel:.2f} hr) | Measured Trough: **{measured_trough:.1f} mg/L** (@ {t_trough_rel:.2f} hr)")
                st.markdown(f"Estimated AUC24 (Current Dose): **{current_auc24_ind:.0f} mg·hr/L**")

                # Check against targets
                if measured_trough < target_cmin_min: st.warning(f"⚠️ Measured Trough is BELOW target ({target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L).")
                elif measured_trough > target_cmin_max: st.warning(f"⚠️ Measured Trough is ABOVE target ({target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L).")
                else: st.success("✅ Measured Trough is WITHIN target range.")

                if current_auc24_ind < 400: st.warning(f"⚠️ Estimated AUC24 is LOW (<400 mg·hr/L).")
                elif current_auc24_ind > 600: st.warning(f"⚠️ Estimated AUC24 is HIGH (>600 mg·hr/L).")
                else: st.success("✅ Estimated AUC24 is WITHIN target range (400-600 mg·hr/L).")

                results_calculated = True

                # --- Dose Adjustment Recommendation ---
                st.markdown("#### Dose Adjustment Recommendation:")
                if cl_ind <= 0:
                     st.warning("Cannot recommend new dose as Clearance could not be calculated.")
                else:
                    desired_interval_adj = st.selectbox("Desired Target Interval (hr) ", [8, 12, 24, 36, 48], index=[8, 12, 24, 36, 48].index(current_interval) if current_interval in [8,12,24,36,48] else 1, key="interval_adj_pt") # Unique key

                    new_dose_daily = target_auc * cl_ind
                    new_dose_interval = new_dose_daily / (24 / desired_interval_adj)

                    # Round to practical dose
                    practical_new_dose = round(new_dose_interval / 250) * 250
                    practical_new_dose = max(250, practical_new_dose)

                    # Predict levels with new practical dose using individualized PK
                    predicted_peak_new = 0.0
                    predicted_trough_new = 0.0
                    if vd_ind > 0 and ke_ind > 0 and infusion_duration > 0 and desired_interval_adj > 0:
                        try:
                            term_inf_new = (1 - math.exp(-ke_ind * infusion_duration))
                            term_interval_new = (1 - math.exp(-ke_ind * desired_interval_adj))
                            denominator_new = vd_ind * ke_ind * infusion_duration * term_interval_new

                            if abs(denominator_new) > 1e-9 and abs(term_inf_new) > 1e-9:
                                predicted_peak_new = (practical_new_dose * term_inf_new) / denominator_new
                                predicted_trough_new = predicted_peak_new * math.exp(-ke_ind * (desired_interval_adj - infusion_duration))
                        except (OverflowError, ValueError): pass # Keep levels as 0

                    # Calculate predicted AUC using trapezoidal method
                    predicted_auc_new = 0
                    if predicted_peak_new > 0 and predicted_trough_new >= 0 and ke_ind > 0 and desired_interval_adj > 0:
                        predicted_auc_new = calculate_vancomycin_auc_trapezoidal(
                            predicted_peak_new, predicted_trough_new, ke_ind, desired_interval_adj, infusion_duration
                        )
                    else:
                        # Fallback to simple calculation
                        predicted_auc_new = (practical_new_dose * (24/desired_interval_adj)) / cl_ind if cl_ind > 0 else 0

                    st.success(f"Adjust to: **{practical_new_dose:.0f} mg** IV q **{desired_interval_adj:.0f}h** (infused over {infusion_duration} hr)")
                    st.info(f"Predicted AUC24: ~{predicted_auc_new:.0f} mg·hr/L")
                    st.info(f"Predicted Trough: ~{predicted_trough_new:.1f} mg/L")

                    # Check predicted trough against secondary target
                    if predicted_trough_new < target_cmin_min: st.warning("⚠️ Predicted trough with new dose may be below secondary target range.")
                    elif predicted_trough_new > target_cmin_max: st.warning("⚠️ Predicted trough with new dose may be above secondary target range.")
                    else: st.success("✅ Predicted trough with new dose is within secondary target range.")

            except ValueError as ve:
                st.error(f"Input Error: {ve}")
                results_calculated = False
            except Exception as e:
                st.error(f"Calculation Error: {e}. Please check inputs.")
                results_calculated = False

        # --- Visualization and Interpretation ---
        if results_calculated:
            if st.checkbox("Show Estimated Concentration-Time Curve (Individualized PK)"):
                if cmax_extrap > 0 and cmin_extrap >= 0 and ke_ind > 0 and current_interval > 0:
                     chart = plot_concentration_time_curve(
                         peak=cmax_extrap, trough=cmin_extrap, ke=ke_ind, tau=current_interval, # Show current interval curve
                         t_peak=infusion_duration, infusion_time=infusion_duration
                     )
                     st.altair_chart(chart, use_container_width=True)
                else:
                     st.warning("Cannot display curve due to invalid calculated parameters.")


            if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
                 prompt = (f"Vancomycin TDM Adjustment (Peak & Trough):\n"
                           f"Current Regimen: {current_dose_interval:.0f} mg IV q {current_interval:.0f}h (infused over {infusion_duration} hr)\n"
                           f"Measured Levels: Peak={measured_peak:.1f} mg/L at {t_peak_rel:.2f} hr post-start ({peak_sample_datetime_dt.strftime('%Y-%m-%d %H:%M')}); Trough={measured_trough:.1f} mg/L at {t_trough_rel:.2f} hr post-start ({trough_sample_datetime_dt.strftime('%Y-%m-%d %H:%M')}).\n"
                           f"Target AUC24: {target_auc} mg·hr/L, Secondary Target Trough: {target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L\n"
                           f"Individualized PK: Ke={ke_ind:.4f} hr⁻¹, t½={t_half_ind:.2f} hr, Vd≈{vd_ind:.2f} L, CL≈{cl_ind:.2f} L/hr\n"
                           f"Estimated Current AUC24: {current_auc24_ind:.0f} mg·hr/L\n")
                 # Add suggested adjustment if calculated
                 if 'practical_new_dose' in locals() and 'desired_interval_adj' in locals() and cl_ind > 0:
                      prompt += (f"Suggested Adjustment: {practical_new_dose:.0f} mg IV q {desired_interval_adj:.0f}h\n"
                                f"Predicted Levels (New Regimen): Trough≈{predicted_trough_new:.1f} mg/L, AUC24≈{predicted_auc_new:.0f} mg·hr/L")
                 interpret_with_llm(prompt, patient_data)


# ===== MAIN APPLICATION CODE =====
def main():
    # Set up sidebar and get patient data
    patient_data = setup_sidebar_and_navigation()

    # Route to appropriate module based on selected page
    if patient_data['page'] == "Aminoglycoside: Initial Dose":
        aminoglycoside_initial_dose(patient_data)
    elif patient_data['page'] == "Aminoglycoside: Conventional Dosing (C1/C2)":
        aminoglycoside_conventional_dosing(patient_data)
    elif patient_data['page'] == "Vancomycin AUC-based Dosing":
        vancomycin_auc_dosing(patient_data)
    else:
        st.error(f"Unknown page selected: {patient_data['page']}")

# Run the application
if __name__ == "__main__":
    main()                    desired_peak = st.number_input("Desired Target Peak (mg/L)", min_value=0.0, value=default_desired_peak, format="%.1f")
                    desired_interval = st.number_input("Desired Target Interval (hr)", min_value=4, max_value=72, value=tau, step=4) # Default to current interval

                    # Calculate new dose needed for desired peak at desired interval
                    # Dose = Cmax_desired * Vd * ke * T_inf * (1 - exp(-ke * Tau_desired)) / (1 - exp(-ke * T_inf))
                    new_dose = 0.0
                    try:
                        new_term_ke_tinf = ke * infusion_duration
                        new_term_ke_tau = ke * desired_interval
                        new_exp_ke_tinf = math.exp(-new_term_ke_tinf)
                        new_exp_ke_tau = math.exp(-new_term_ke_tau)

                        new_numerator = desired_peak * vd * new_term_ke_tinf * (1.0 - new_exp_ke_tau)
                        new_denominator = (1.0 - new_exp_ke_tinf)

                        if abs(new_denominator) > 1e-9:
                             new_dose = new_numerator / new_denominator
                        else:
                             st.warning("Could not calculate new dose accurately due to near-zero denominator.")

                    except (OverflowError, ValueError) as math_err_newdose:
                         st.error(f"Math error during new dose calculation: {math_err_newdose}")


                    # Round new dose
                    rounding_base = 20 if drug == "Gentamicin" else 50 if drug == "Amikacin" else 10
                    practical_new_dose = round(new_dose / rounding_base) * rounding_base
                    practical_new_dose = max(rounding_base, practical_new_dose)

                    # Predict peak and trough with new dose and interval
                    predicted_peak = 0.0
                    predicted_trough = 0.0
                    if practical_new_dose > 0 and vd > 0 and ke > 0 and infusion_duration > 0 and desired_interval > 0:
                        try:
                            pred_term_ke_tinf = ke * infusion_duration
                            pred_term_ke_tau = ke * desired_interval
                            pred_exp_ke_tinf = math.exp(-pred_term_ke_tinf)
                            pred_exp_ke_tau = math.exp(-pred_term_ke_tau)

                            pred_denominator_cmax = vd * ke * infusion_duration * (1.0 - pred_exp_ke_tau)
                            if abs(pred_denominator_cmax) > 1e-9:
                                 predicted_peak = practical_new_dose * (1.0 - pred_exp_ke_tinf) / pred_denominator_cmax

                            predicted_trough = predicted_peak * math.exp(-ke * (desired_interval - infusion_duration))

                        except (OverflowError, ValueError) as math_err_pred:
                             st.warning(f"Could not predict levels for new dose due to math error: {math_err_pred}")


                    st.success(f"Suggested New Regimen: **{practical_new_dose:.0f} mg** IV q **{desired_interval:.0f}h** (infused over {infusion_duration} hr)")
                    st.info(f"Predicted Peak: ~{predicted_peak:.1f} mg/L | Predicted Trough: ~{predicted_trough:.2f} mg/L")

                    # Check predicted levels against targets
                    suggest_adjustment(predicted_peak, target_peak_min, target_peak_max, label="Predicted Peak")
                    if predicted_trough >= target_trough_max:
                         st.warning(f"⚠️ Predicted Trough ({predicted_trough:.2f} mg/L) meets or exceeds target maximum ({target_trough_max} mg/L). Consider lengthening interval further.")
                    else:
                         st.success(f"✅ Predicted Trough ({predicted_trough:.2f} mg/L) is below target maximum ({target_trough_max} mg/L).")


        except ValueError as ve:
            st.error(f"Input Error: {ve}")
            results_calculated = False
        except Exception as e:
            st.error(f"Calculation Error: {e}. Please check inputs.")
            results_calculated = False # Ensure button doesn't show if error


    # --- Interpretation and Visualization ---
    if results_calculated:
        # Add visualization option
        if st.checkbox("Show Estimated Concentration-Time Curve (Based on Calculated Parameters)"):
            if cmax_extrapolated > 0 and cmin_extrapolated >= 0 and ke > 0 and tau > 0:
                 chart = plot_concentration_time_curve(
                     peak=cmax_extrapolated,
                     trough=cmin_extrapolated,
                     ke=ke,
                     tau=tau, # Show curve for the *current* interval
                     t_peak=infusion_duration,
                     infusion_time=infusion_duration
                 )
                 st.altair_chart(chart, use_container_width=True)
            else:
                 st.warning("Cannot display curve due to invalid calculated parameters.")


        if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
            # Prepare prompt for interpretation
            prompt = (f"Aminoglycoside TDM Adjustment:\n"
                      f"Drug: {drug}, Regimen Goal: {regimen}\n"
                      f"Current Regimen: {dose:.0f} mg IV q {tau:.0f}h (infused over {infusion_duration} hr)\n"
                      f"Measured Levels: Trough (C1)={c1:.1f} mg/L at {t1:.2f} hr post-start ({c1_sample_datetime_dt.strftime('%Y-%m-%d %H:%M')}); Peak (C2)={c2:.1f} mg/L at {t2:.2f} hr post-start ({c2_sample_datetime_dt.strftime('%Y-%m-%d %H:%M')}).\n"
                      f"Target Ranges: Peak {target_peak_info}, Trough {target_trough_info}\n"
                      f"Calculated PK: Ke={ke:.4f} hr⁻¹, t½={t_half:.2f} hr, Vd≈{vd:.2f} L, CL≈{cl:.2f} L/hr\n"
                      f"Estimated Levels (Current Regimen): Cmax≈{cmax_extrapolated:.1f} mg/L, Cmin≈{cmin_extrapolated:.2f} mg/L\n")
            if 'practical_new_dose' in locals() and 'desired_interval' in locals(): # Add recommendation if calculated
                 prompt += (f"Suggested Adjustment: {practical_new_dose:.0f} mg IV q {desired_interval:.0f}h\n"
                           f"Predicted Levels (New Regimen): Peak≈{predicted_peak:.1f} mg/L, Trough≈{predicted_trough:.2f} mg/L")

            interpret_with_llm(prompt, patient_data)


# ===== MODULE 3: Vancomycin AUC-based Dosing =====
def vancomycin_auc_dosing(patient_data):
    st.title("🧪 Vancomycin AUC-Based Dosing & Adjustment")
    st.info("AUC24 is calculated using the Linear-Log Trapezoidal method as recommended for vancomycin TDM")

    # Unpack patient data
    weight = patient_data['weight']
    crcl = patient_data['crcl']
    gender = patient_data['gender']
    age = patient_data['age']

    method = st.radio("Select Method / Scenario", ["Calculate Initial Dose (Population PK)", "Adjust Dose using Trough Level", "Adjust Dose using Peak & Trough Levels"], horizontal=True)

    # --- Target Selection ---
    st.markdown("### Target Selection")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        target_auc = st.slider("Target AUC24 (mg·hr/L)", min_value=300, max_value=700, value=500, step=10) # Default 400-600
        st.info("Typical Target AUC24: 400-600 mg·hr/L")
    with col_t2:
        # Trough target is secondary but often monitored
        target_trough_range = st.selectbox(
            "Secondary Target Trough Range (mg/L)",
            ["10-15 (Empiric)", "15-20 (Definitive)", "Custom"]
        )
        if target_trough_range == "Custom":
             target_cmin_min = st.number_input("Min Trough", min_value=5.0, value=10.0, step=0.5)
             target_cmin_max = st.number_input("Max Trough", min_value=target_cmin_min, value=15.0, step=0.5)
        elif "15-20" in target_trough_range:
             target_cmin_min, target_cmin_max = 15.0, 20.0
        else: # Default 10-15
             target_cmin_min, target_cmin_max = 10.0, 15.0
        st.info(f"Selected Target Trough: {target_cmin_min:.1f} - {target_cmin_max:.1f} mg/L")
    
    # Add Peak targets
    st.markdown("### Peak Target Range")
    if "Empiric" in target_trough_range:
        target_peak_min, target_peak_max = 20.0, 30.0
        st.info("Peak Target for Empiric: 20-30 mg/L")
    elif "Definitive" in target_trough_range:
        target_peak_min, target_peak_max = 25.0, 40.0
        st.info("Peak Target for Definitive: 25-40 mg/L")
    else:  # Custom
        target_peak_min = st.number_input("Min Peak", min_value=10.0, value=20.0, step=1.0)
        target_peak_max = st.number_input("Max Peak", min_value=target_peak_min, value=30.0, step=1.0)
        st.info(f"Custom Peak Target: {target_peak_min:.1f} - {target_peak_max:.1f} mg/L")

    # --- Input Fields based on Method ---
    st.markdown("### Enter Dosing and Level Information")

    # --- Initial Dose Calculation ---
    if "Initial Dose" in method:
        st.markdown("Using population PK estimates based on patient demographics.")
        desired_interval = st.selectbox("Desired Dosing Interval (hr)", [8, 12, 24, 36, 48], index=1) # Default q12h
        infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5, help="Typically 1 hr per 1g")

        # Population PK Estimates (Simplified Bayesian approach using CrCl)
        # Ref: Pai MP, Neely M, Rodvold KA, Lodise TP. Innovative approaches to optimizing the delivery of vancomycin in infected patients. Adv Drug Deliv Rev. 2014;
        # Vd (L) ≈ 0.7 L/kg * Weight (kg) -- Can refine based on age/obesity if needed
        # CLvanco (L/hr) ≈ CrCl (mL/min) * (0.75 to 0.9) * 60 / 1000 -- Renal clearance dominant
        vd_pop = 0.7 * weight
        cl_pop = crcl * 0.8 * 60 / 1000 # Using 0.8 as renal clearance fraction
        cl_pop = max(0.1, cl_pop) # Ensure CL is not zero
        ke_pop = cl_pop / vd_pop if vd_pop > 0 else 0
        ke_pop = max(1e-6, ke_pop) # Ensure Ke is positive
        t_half_pop = 0.693 / ke_pop if ke_pop > 0 else float('inf')

        st.markdown("#### Population PK Estimates:")
        st.markdown(f"**Est. Vd:** {vd_pop:.2f} L | **Est. CL:** {cl_pop:.2f} L/hr | **Est. Ke:** {ke_pop:.4f} hr⁻¹ | **Est. t½:** {t_half_pop:.2f} hr")

        # Calculate Dose needed for Target AUC
        # AUC24 = Dose_daily / CL => Dose_daily = AUC24 * CL
        # Dose_per_interval = Dose_daily / (24 / interval)
        target_dose_daily = target_auc * cl_pop
        target_dose_interval = target_dose_daily / (24 / desired_interval)

        # Round to practical dose (e.g., nearest 250mg)
        practical_dose = round(target_dose_interval / 250) * 250
        practical_dose = max(250, practical_dose) # Minimum practical dose

        # Predict levels with this practical dose
        # Using steady-state infusion equations
        predicted_peak = 0.0
        predicted_trough = 0.0
        if vd_pop > 0 and ke_pop > 0 and infusion_duration > 0 and desired_interval > 0:
            try:
                term_inf = (1 - math.exp(-ke_pop * infusion_duration))
                term_interval = (1 - math.exp(-ke_pop * desired_interval))
                denominator = vd_pop * ke_pop * infusion_duration * term_interval

                if abs(denominator) > 1e-9 and abs(term_inf) > 1e-9:
                    # Cmax = Dose * (1 - exp(-ke * T_inf)) / [Vd * ke * T_inf * (1 - exp(-ke * tau))]
                    predicted_peak = (practical_dose * term_inf) / denominator
                    predicted_trough = predicted_peak * math.exp(-ke_pop * (desired_interval - infusion_duration))
            except (OverflowError, ValueError):
                 st.warning("Could not predict levels due to math error.")


        st.markdown("#### Recommended Initial Dose:")
        st.success(f"Start with **{practical_dose:.0f} mg** IV q **{desired_interval:.0f}h** (infused over {infusion_duration} hr)")
        
        # Calculate predicted AUC using trapezoidal method
        predicted_auc24 = 0
        if predicted_peak > 0 and predicted_trough >= 0 and ke_pop > 0 and desired_interval > 0:
            predicted_auc24 = calculate_vancomycin_auc_trapezoidal(
                predicted_peak, predicted_trough, ke_pop, desired_interval, infusion_duration
            )
        else:
            # Fallback to simple calculation
            predicted_auc24 = (practical_dose * (24/desired_interval)) / cl_pop if cl_pop > 0 else 0
        
        st.info(f"Predicted AUC24: ~{predicted_auc24:.0f} mg·hr/L")
        st.info(f"Predicted Peak (end of infusion): ~{predicted_peak:.1f} mg/L")
        st.info(f"Predicted Trough (end of interval): ~{predicted_trough:.1f} mg/L")

        # Check predicted trough against secondary target
        if predicted_trough < target_cmin_min: st.warning("⚠️ Predicted trough may be below secondary target range.")
        elif predicted_trough > target_cmin_max: st.warning("⚠️ Predicted trough may be above secondary target range.")
        else: st.success("✅ Predicted trough is within secondary target range.")

        # Suggest Loading Dose for severe infections or high target AUC
        if target_auc >= 500 or "sepsis" in patient_data.get('clinical_diagnosis', '').lower() or "meningitis" in patient_data.get('clinical_diagnosis', '').lower():
             loading_dose = 25 * weight # Common LD: 25-30 mg/kg (using actual weight)
             practical_loading_dose = round(loading_dose / 250) * 250
             st.warning(f"Consider Loading Dose: **~{practical_loading_dose:.0f} mg** IV x 1 dose (e.g., 25 mg/kg actual weight). Infuse over 1.5-2 hours.")

        # Visualization and Interpretation
        if st.checkbox("Show Estimated Concentration-Time Curve"):
            if predicted_peak > 0 and predicted_trough >= 0 and ke_pop > 0 and desired_interval > 0:
                chart = plot_concentration_time_curve(
                    peak=predicted_peak, trough=predicted_trough, ke=ke_pop, tau=desired_interval,
                    t_peak=infusion_duration, infusion_time=infusion_duration
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Cannot display curve due to invalid calculated parameters.")


        if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
            prompt = (f"Vancomycin Initial Dose Calculation:\n"
                      f"Target AUC24: {target_auc} mg·hr/L, Secondary Target Trough: {target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L\n"
                      f"Desired Interval: {desired_interval} hr, Infusion Duration: {infusion_duration} hr\n"
                      f"Population PK Estimates: Vd={vd_pop:.2f} L, CL={cl_pop:.2f} L/hr, Ke={ke_pop:.4f} hr⁻¹, t½={t_half_pop:.2f} hr\n"
                      f"Recommended Initial Dose: {practical_dose:.0f} mg IV q {desired_interval:.0f}h\n"
                      f"Predicted Levels: Peak≈{predicted_peak:.1f} mg/L, Trough≈{predicted_trough:.1f} mg/L, AUC24≈{predicted_auc24:.0f} mg·hr/L")
            interpret_with_llm(prompt, patient_data)


    # --- Adjustment using Trough Level ---
    elif "Trough Level" in method:
        st.markdown("Adjusting dose based on a measured trough level and population Vd estimate.")
        col1, col2 = st.columns(2)
        with col1:
            current_dose_interval = st.number_input("Current Dose per Interval (mg)", min_value=250.0, value=1000.0, step=50.0)
            current_interval = st.number_input("Current Dosing Interval (hr)", min_value=4, max_value=72, value=12, step=4)
            infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        with col2:
            measured_trough = st.number_input("Measured Trough Level (mg/L)", min_value=0.1, value=12.0, step=0.1, format="%.1f")
            trough_sample_time_rel = st.selectbox("Trough Sample Time Relative to Dose", ["Just Before Next Dose (Steady State)", "Other"], index=0)
            if trough_sample_time_rel == "Other":
                 st.warning("Calculation assumes steady-state trough taken just before the next dose. Interpretation may be less accurate otherwise.")

        # Estimate Ke and CL using measured trough and population Vd (Simplified Bayesian / Ratio Method)
        vd_pop = 0.7 * weight
        cl_pop = crcl * 0.8 * 60 / 1000
        cl_pop = max(0.1, cl_pop)
        ke_pop = cl_pop / vd_pop if vd_pop > 0 else 0
        ke_pop = max(1e-6, ke_pop)
        t_half_pop = 0.693 / ke_pop if ke_pop > 0 else float('inf')

        st.markdown("#### Population PK Estimates (Used as Prior):")
        st.markdown(f"**Est. Vd:** {vd_pop:.2f} L | **Est. CL:** {cl_pop:.2f} L/hr | **Est. Ke:** {ke_pop:.4f} hr⁻¹ | **Est. t½:** {t_half_pop:.2f} hr")

        # Calculate predicted trough using population PK
        predicted_trough_pop = 0.0
        if vd_pop > 0 and ke_pop > 0 and infusion_duration > 0 and current_interval > 0:
             try:
                 term_inf_pop = (1 - math.exp(-ke_pop * infusion_duration))
                 term_int_pop = (1 - math.exp(-ke_pop * current_interval))
                 denom_pop = vd_pop * ke_pop * infusion_duration * term_int_pop
                 if abs(denom_pop) > 1e-9 and abs(term_inf_pop) > 1e-9:
                     cmax_pred_pop = (current_dose_interval * term_inf_pop) / denom_pop
                     predicted_trough_pop = cmax_pred_pop * math.exp(-ke_pop * (current_interval - infusion_duration))
             except (OverflowError, ValueError): pass # Keep predicted_trough_pop as 0

        # Adjust CL based on ratio of measured trough to predicted population trough
        cl_adjusted = cl_pop
        if predicted_trough_pop > 0.5 and measured_trough > 0.1: # Avoid adjusting if predicted is very low or measured is zero
             # Ratio adjustment: New CL = Old CL * (Target / Measured) -> applied to AUC
             # Adjustment based on trough ratio: CL_adj = CL_pop * (Pred_Trough / Meas_Trough)
             cl_adjusted = cl_pop * (predicted_trough_pop / measured_trough)
             cl_adjusted = max(0.05, min(cl_adjusted, cl_pop * 5)) # Bound the adjustment
        elif measured_trough <= 0.1:
             st.warning("Measured trough is very low or zero, cannot reliably adjust CL based on ratio.")


        # Recalculate Ke based on adjusted CL and pop Vd
        ke_adjusted = cl_adjusted / vd_pop if vd_pop > 0 else ke_pop
        ke_adjusted = max(1e-6, ke_adjusted)
        t_half_adjusted = 0.693 / ke_adjusted if ke_adjusted > 0 else float('inf')

        st.markdown("#### Adjusted PK Estimates (Based on Trough):")
        st.markdown(f"**Adj. CL:** {cl_adjusted:.2f} L/hr | **Adj. Ke:** {ke_adjusted:.4f} hr⁻¹ | **Adj. t½:** {t_half_adjusted:.2f} hr")

        # Calculate current AUC24 using trapezoidal method if peaks and troughs available
        current_auc24 = 0
        if vd_pop > 0 and ke_adjusted > 0 and infusion_duration > 0 and current_interval > 0:
            try:
                # Calculate Cmax using adjusted PK parameters
                term_inf_adj = (1 - math.exp(-ke_adjusted * infusion_duration))
                term_int_adj = (1 - math.exp(-ke_adjusted * current_interval))
                denom_adj = vd_pop * ke_adjusted * infusion_duration * term_int_adj
                
                if abs(denom_adj) > 1e-9 and abs(term_inf_adj) > 1e-9:
                    cmax_calc = (current_dose_interval * term_inf_adj) / denom_adj
                    cmin_calc = cmax_calc * math.exp(-ke_adjusted * (current_interval - infusion_duration))
                    
                    # Use trapezoidal method
                    current_auc24 = calculate_vancomycin_auc_trapezoidal(
                        cmax_calc, cmin_calc, ke_adjusted, current_interval, infusion_duration
                    )
            except (OverflowError, ValueError):
                # Fallback to simple calculation
                current_dose_daily = current_dose_interval * (24 / current_interval)
                current_auc24 = current_dose_daily / cl_adjusted if cl_adjusted > 0 else 0

        st.markdown("#### Current Regimen Assessment:")
        st.markdown(f"Measured Trough: **{measured_trough:.1f} mg/L**")
        st.markdown(f"Estimated AUC24 (Current Dose): **{current_auc24:.0f} mg·hr/L**")

        # Check against targets
        if measured_trough < target_cmin_min: st.warning(f"⚠️ Measured Trough is BELOW target ({target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L).")
        elif measured_trough > target_cmin_max: st.warning(f"⚠️ Measured Trough is ABOVE target ({target_cmin_min:.1f}-{target_cmin_max:.1f} mg/L).")
        else: st.success("✅ Measured Trough is WITHIN target range.")

        if current_auc24 < 400: st.warning(f"⚠️ Estimated AUC24 is LOW (<400 mg·hr/L).")
        elif current_auc24 > 600: st.warning(f"⚠️ Estimated AUC24 is HIGH (>600 mg·hr/L).")
        else: st.success("✅ Estimated AUC24 is WITHIN target range (400-600 mg·hr/L).")


        # Calculate New Dose for Target AUC
        st.markdown("#### Dose Adjustment Recommendation:")
        if cl_adjusted <= 0:
             st.warning("Cannot recommend new dose as adjusted Clearance is invalid.")
        else:
            desired_interval_adj = st.selectbox("Desired Target Interval (hr)", [8, 12, 24, 36, 48], index=[8, 12, 24, 36, 48].index(current_interval) if current_interval in [8,12,24,36,48] else 1)

            new_dose_daily = target_auc * cl_adjusted
            new_dose_interval = new_dose_daily / (24 / desired_interval_adj)

            # Round to practical dose
            practical_new_dose = round(new_dose_interval / 250) * 250
            practical_new_dose = max(250, practical_new_dose)

            # Predict levels with new practical dose using adjusted PK
            predicted_peak_new = 0.0
            predicted_trough_new = 0.0
            if vd_pop > 0 and ke_adjusted > 0 and infusion_duration > 0 and desired_interval_adj > 0:
                 try:
                     term_inf_adj = (1 - math.exp(-ke_adjusted * infusion_duration))
                     term_interval_adj = (1 - math.exp(-ke_adjusted * desired_interval_adj))
                     denominator_adj = vd_pop * ke_adjusted * infusion_duration * term_interval_adj

                     if abs(denominator_adj) > 1e-9 and abs(term_inf_adj) > 1e-9:
                         predicted_peak_new = (practical_new_dose * term_inf_adj) / denominator_adj
                         predicted_trough_new = predicted_peak_new * math.exp(-ke_adjusted * (desired_interval_adj - infusion_duration))
                 except (OverflowError, ValueError): pass # Keep levels as 0

            # Calculate predicted AUC using trapezoidal method if predicted levels available
            predicted_auc_new = 0
            if predicted_peak_new > 0 and predicted_trough_new >= 0 and ke_adjusted > 0 and desired_interval_adj > 0:
                predicted_auc_new = calculate_vancomycin_auc_trapezoidal(
                    predicted_peak_new, predicted_trough_new, ke_adjusted, desired_interval_adj, infusion_duration
                )
            else:
                # Fallback to simple calculation
                predicted_auc_new = (practical_new_dose * (24/desired_interval_adj)) / cl_adjusted if cl_adjusted > 0 else 0


            st.success(f"Adjust to: **{practical_new_dose:.0f} mg** IV q **{desired_interval_adj:.0f}h** (infused over {infusion_duration} hr)")
            st.info(f"Predicted AUC24: ~{predicted_auc_new:.0f} mg·hr/L")
            st.info(f"Predicted Trough: ~{predicted_trough_new:.1f} mg/L")

            # Check predicted trough against secondary target
            if predicted_trough_new < target_cmin_min: st.warning("⚠️ Predicted trough with new dose may be below secondary target range.")
            elif predicted_trough_new > target_cmin_max: st.warning("⚠️ Predicted trough with new dose may be above secondary target range.")
            else: st.success("✅ Predicted trough with new dose is within secondary target range.")

            # Visualization and Interpretation
            if st.checkbox("Show Estimated Concentration-Time Curve (Adjusted PK)"):
                 if predicted_peak_new > 0 and predicted_trough_new >= 0 and ke_adjusted > 0 and desired_interval_adj > 0:
                     chart = plot_concentration_time_curve(
                         peak=predicted_peak_new, trough=predicted_trough_new, ke=ke_adjusted, tau=desired_interval_adj,
                         t_peak=infusion_duration, infusion_time=infusion_duration
                     )
                     st.altair_chart(chart, use_container_width=True)
                 else:
                     st.warning("Cannot display curve due to invalid calculated parameters for new dose.")


            if st.button("🧠 Generate Clinical Interpretation (LLM/    if drug == "Gentamicin":
        if regimen_code == "MDD": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 1.0, "5-10 mg/L", "<2 mg/L"
        elif regimen_code == "SDD": default_peak, default_trough, target_peak_info, target_trough_info = 20.0, 0.5, "15-30 mg/L (or 10x MIC)", "<1 mg/L (often undetectable)"
        elif regimen_code == "Synergy": default_peak, default_trough, target_peak_info, target_trough_info = 4.0, 0.5, "3-5 mg/L", "<1 mg/L"
        elif regimen_code == "Hemodialysis": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 1.0, "Peak not routinely targeted", "<2 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": default_peak, default_trough, target_peak_info, target_trough_info = 8.0, 0.5, "5-12 mg/L", "<1 mg/L"
    else:  # Amikacin
        if regimen_code == "MDD": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 5.0, "20-30 mg/L", "<10 mg/L"
        elif regimen_code == "SDD": default_peak, default_trough, target_peak_info, target_trough_info = 60.0, 2.0, "50-70 mg/L (or 10x MIC)", "<5 mg/L (often undetectable)"
        elif regimen_code == "Synergy": default_peak, default_trough, target_peak_info, target_trough_info = 0.0, 0.0, "N/A", "N/A"
        elif regimen_code == "Hemodialysis": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 5.0, "Peak not routinely targeted", "<10 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": default_peak, default_trough, target_peak_info, target_trough_info = 25.0, 2.5, "20-30 mg/L", "<5 mg/L"

    st.info(f"Typical Targets for {regimen}: Peak {target_peak_info}, Trough {target_trough_info}")

    # MIC input for SDD regimens
    mic = 1.0 # Default MIC
    if regimen_code == "SDD":
        st.markdown("*Note: Target peak for Extended Interval is often 10x MIC.*")
        mic = st.number_input("Enter MIC (mg/L)", min_value=0.1, value=1.0, step=0.1, format="%.1f")
        recommended_peak_mic = mic * 10
        if recommended_peak_mic > default_peak:
            default_peak = recommended_peak_mic
        st.info(f"Based on MIC, target peak is ≥ {recommended_peak_mic:.1f} mg/L")

    # Allow user override of targets
    col1, col2 = st.columns(2)
    with col1:
        target_cmax = st.number_input("Target Peak (Cmax, mg/L)", value=default_peak, format="%.1f")
    with col2:
        target_cmin = st.number_input("Target Trough (Cmin, mg/L)", value=default_trough, format="%.1f")

    # Default tau based on regimen
    default_tau = 24 if regimen_code == "SDD" \
             else 8 if regimen_code == "MDD" \
             else 12 if regimen_code == "Synergy" \
             else 48 # Default for HD (q48h common) / Neonates (adjust based on age/PMA)
    tau = st.number_input("Desired Dosing Interval (hr)", min_value=4, max_value=72, value=default_tau, step=4)

    # Infusion duration
    infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)

    # Special handling notes
    if regimen_code == "Hemodialysis":
        st.info("For hemodialysis, dose is typically given post-dialysis. Interval depends on dialysis schedule (e.g., q48h, q72h). Calculations assume dose given after dialysis.")
    if regimen_code == "Neonates":
        st.warning("Neonatal PK varies significantly. These calculations use adult population estimates. CONSULT a pediatric pharmacist.")

    # --- Calculations ---
    # Calculate IBW and dosing weight (using standard formulas)
    ibw = 0.0
    if height > 152.4: # Height threshold for formulas (60 inches)
        ibw = (50 if gender == "Male" else 45.5) + 2.3 * (height / 2.54 - 60)
    ibw = max(0, ibw) # Ensure IBW is not negative

    dosing_weight = weight # Default to actual body weight
    weight_used = "Actual Body Weight"
    if ibw > 0: # Only adjust if IBW is calculable and patient is not underweight
        if weight / ibw > 1.3: # Obese threshold (e.g., >130% IBW)
            dosing_weight = ibw + 0.4 * (weight - ibw) # Adjusted BW
            weight_used = "Adjusted Body Weight"
        elif weight < ibw: # Underweight: Use Actual BW (common practice)
             dosing_weight = weight
             weight_used = "Actual Body Weight (using ABW as < IBW)"
        else: # Normal weight: Use Actual or Ideal (Using Actual here)
             dosing_weight = weight
             weight_used = "Actual Body Weight"


    st.markdown(f"**IBW:** {ibw:.1f} kg | **Dosing Weight Used:** {dosing_weight:.1f} kg ({weight_used})")

    # Population PK parameters (adjust Vd based on clinical factors if needed)
    base_vd_per_kg = 0.3 if drug == "Amikacin" else 0.26 # L/kg
    vd_adjustment = 1.0 # Default
    # Simple adjustments based on notes (can be refined)
    notes_lower = notes.lower()
    if any(term in notes_lower for term in ["ascites", "edema", "fluid overload", "anasarca", "chf exacerbation"]): vd_adjustment = 1.15; st.info("Vd increased by 15% due to potential fluid overload.")
    if any(term in notes_lower for term in ["septic", "sepsis", "burn", "icu patient"]): vd_adjustment = 1.20; st.info("Vd increased by 20% due to potential sepsis/burn/critical illness.")
    if any(term in notes_lower for term in ["dehydrated", "volume depleted"]): vd_adjustment = 0.90; st.info("Vd decreased by 10% due to potential dehydration.")

    vd = base_vd_per_kg * dosing_weight * vd_adjustment # Liters
    vd = max(1.0, vd) # Ensure Vd is at least 1L

    # Calculate Ke and Cl based on CrCl (population estimate)
    # Using published relationships might be better, e.g., Ke = a + b * CrCl
    # Simplified approach: CL (L/hr) ≈ CrCl (mL/min) * factor (e.g., 0.05 for Gentamicin)
    # Ke = CL / Vd
    cl_pop = 0.0
    if crcl > 0:
        # Example: Gentamicin CL ≈ 0.05 * CrCl (L/hr if CrCl in mL/min) - Highly simplified
        # Example: Amikacin CL might be slightly higher
        cl_factor = 0.06 if drug == "Amikacin" else 0.05
        cl_pop = cl_factor * crcl
    cl_pop = max(0.1, cl_pop) # Minimum clearance estimate

    ke = cl_pop / vd if vd > 0 else 0.01
    ke = max(0.005, ke) # Ensure ke is not excessively low

    t_half = 0.693 / ke if ke > 0 else float('inf')

    st.markdown(f"**Est. Vd:** {vd:.2f} L | **Est. Ke:** {ke:.4f} hr⁻¹ | **Est. t½:** {t_half:.2f} hr | **Est. CL:** {cl_pop:.2f} L/hr")

    # Calculate Dose needed to achieve target Cmax (using steady-state infusion equation)
    # Dose = Cmax * Vd * ke * T_inf * (1 - exp(-ke * tau)) / (1 - exp(-ke * T_inf))
    dose = 0.0
    try:
        term_ke_tinf = ke * infusion_duration
        term_ke_tau = ke * tau
        exp_ke_tinf = math.exp(-term_ke_tinf)
        exp_ke_tau = math.exp(-term_ke_tau)

        numerator = target_cmax * vd * term_ke_tinf * (1.0 - exp_ke_tau)
        denominator = (1.0 - exp_ke_tinf)

        if abs(denominator) > 1e-9:
            dose = numerator / denominator
        else: # Handle bolus case approximation if T_inf is very small
            dose = target_cmax * vd * (1.0 - exp_ke_tau) / (1.0 - exp_ke_tinf) # Recheck this derivation
            # Simpler Bolus: Dose = Cmax * Vd * (1-exp(-ke*tau)) -> This assumes Cmax is achieved instantly
            st.warning("Infusion duration is very short or Ke is very low; using approximation for dose calculation.")
            # Let's stick to the rearranged infusion formula, checking denominator

    except (OverflowError, ValueError) as math_err:
         st.error(f"Math error during dose calculation: {math_err}. Check PK parameters.")
         dose = 0 # Prevent further calculation

    # Calculate expected levels with the calculated dose
    expected_cmax = 0.0
    expected_cmin = 0.0
    if dose > 0 and vd > 0 and ke > 0 and infusion_duration > 0 and tau > 0:
        try:
            term_ke_tinf = ke * infusion_duration
            term_ke_tau = ke * tau
            exp_ke_tinf = math.exp(-term_ke_tinf)
            exp_ke_tau = math.exp(-term_ke_tau)

            common_factor = (dose / (vd * term_ke_tinf)) * (1.0 - exp_ke_tinf) / (1.0 - exp_ke_tau)

            # Cmax occurs at the end of infusion (t = infusion_duration)
            # Using the "during infusion" part of the SS equation at t=Tinf
            # C(t_inf) = common_factor * (1 - exp(-ke*t_inf)) -> simplifies to peak formula
            expected_cmax = (dose / (vd * ke * infusion_duration)) * (1 - exp_ke_tinf) / (1 - exp_ke_tau) * (1 - exp_ke_tinf) # Recheck needed
            # Let's use the simpler Cmax definition from rearrangement:
            # Cmax = Dose * (1 - exp(-ke * T_inf)) / [Vd * ke * T_inf * (1 - exp(-ke * tau))]
            denominator_cmax = vd * ke * infusion_duration * (1 - exp_ke_tau)
            if abs(denominator_cmax) > 1e-9:
                 expected_cmax = dose * (1 - exp_ke_tinf) / denominator_cmax

            # Cmin occurs at the end of the interval (t = tau)
            # Cmin = Cmax * exp(-ke * (tau - T_inf))
            expected_cmin = expected_cmax * math.exp(-ke * (tau - infusion_duration))

        except (OverflowError, ValueError) as math_err_levels:
             st.warning(f"Could not predict levels due to math error: {math_err_levels}")


    # Round the dose to a practical value (e.g., nearest 10mg or 20mg)
    rounding_base = 20 if drug == "Gentamicin" else 50 if drug == "Amikacin" else 10
    practical_dose = round(dose / rounding_base) * rounding_base
    practical_dose = max(rounding_base, practical_dose) # Ensure dose is at least the rounding base

    st.success(f"Recommended Initial Dose: **{practical_dose:.0f} mg** IV every **{tau:.0f}** hours (infused over {infusion_duration} hr)")
    st.info(f"Predicted Peak (end of infusion): ~{expected_cmax:.1f} mg/L")
    st.info(f"Predicted Trough (end of interval): ~{expected_cmin:.2f} mg/L")


    # Suggest loading dose if applicable (e.g., for SDD or severe infections)
    if regimen_code == "SDD" or "sepsis" in notes.lower() or "critical" in notes.lower():
        # Loading Dose ≈ Target Peak * Vd
        loading_dose = target_cmax * vd
        practical_loading_dose = round(loading_dose / rounding_base) * rounding_base
        practical_loading_dose = max(rounding_base, practical_loading_dose)
        st.warning(f"Consider Loading Dose: **~{practical_loading_dose:.0f} mg** IV x 1 dose to rapidly achieve target peak.")

    # Check if expected levels meet targets
    suggest_adjustment(expected_cmax, target_cmax * 0.85, target_cmax * 1.15, label="Predicted Peak") # Tighter range for check
    # Check trough against target_cmin (which is usually the max allowed trough)
    if expected_cmin > target_cmin: # Target Cmin here represents the upper limit for trough
         st.warning(f"⚠️ Predicted Trough ({expected_cmin:.2f} mg/L) may exceed target ({target_trough_info}). Consider lengthening interval if clinically appropriate.")
    else:
         st.success(f"✅ Predicted Trough ({expected_cmin:.2f} mg/L) likely below target ({target_trough_info}).")

    # Add visualization option
    if st.checkbox("Show Estimated Concentration-Time Curve"):
        if expected_cmax > 0 and expected_cmin >= 0 and ke > 0 and tau > 0:
            chart = plot_concentration_time_curve(
                peak=expected_cmax,
                trough=expected_cmin,
                ke=ke,
                tau=tau,
                t_peak=infusion_duration, # Assume peak occurs at end of infusion
                infusion_time=infusion_duration
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Cannot display curve due to invalid calculated parameters.")


    if st.button("🧠 Generate Clinical Interpretation (LLM/Simulated)"):
        prompt = (f"Aminoglycoside Initial Dose Calculation:\n"
                  f"Drug: {drug}, Regimen Goal: {regimen}\n"
                  f"Target Peak: {target_cmax:.1f} mg/L, Target Trough: {target_cmin:.1f} mg/L (Typical: Peak {target_peak_info}, Trough {target_trough_info})\n"
                  f"Desired Interval (tau): {tau} hr, Infusion Duration: {infusion_duration} hr\n"
                  f"Calculated Dose: {practical_dose:.0f} mg\n"
                  f"Estimated PK: Vd={vd:.2f} L, Ke={ke:.4f} hr⁻¹, t½={t_half:.2f} hr, CL={cl_pop:.2f} L/hr\n"
                  f"Predicted Levels: Peak≈{expected_cmax:.1f} mg/L, Trough≈{expected_cmin:.2f} mg/L")
        interpret_with_llm(prompt, patient_data)


# ===== MODULE 2: Aminoglycoside Conventional Dosing (C1/C2) =====
def aminoglycoside_conventional_dosing(patient_data):
    st.title("📊 Aminoglycoside Dose Adjustment (using Levels)")

    drug = st.selectbox("Select Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Therapeutic Goal / Strategy", ["Traditional (Multiple Daily - MDD)", "Extended Interval (Once Daily - SDD)", "Synergy (e.g., Endocarditis)", "Hemodialysis", "Neonates (Use with caution)"])

    # Map selection to internal codes
    regimen_code = "SDD" if "Extended" in regimen \
              else "MDD" if "Traditional" in regimen \
              else "Synergy" if "Synergy" in regimen \
              else "Hemodialysis" if "Hemodialysis" in regimen \
              else "Neonates" if "Neonates" in regimen \
              else "MDD" # Default

    # --- Set target ranges based on chosen regimen and drug ---
    target_peak_info = "N/A"
    target_trough_info = "N/A"
    target_peak_min, target_peak_max = 0.0, 100.0
    target_trough_max = 100.0 # Represents the upper limit for trough

    if drug == "Gentamicin":
        if regimen_code == "MDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 5, 10, 2, "5-10 mg/L", "<2 mg/L"
        elif regimen_code == "SDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 15, 30, 1, "15-30 mg/L (or 10x MIC)", "<1 mg/L (often undetectable)"
        elif regimen_code == "Synergy": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 3, 5, 1, "3-5 mg/L", "<1 mg/L"
        elif regimen_code == "Hemodialysis": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 2, "Peak not routinely targeted", "<2 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 5, 12, 1, "5-12 mg/L", "<1 mg/L"
    else:  # Amikacin
        if regimen_code == "MDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 20, 30, 10, "20-30 mg/L", "<10 mg/L"
        elif regimen_code == "SDD": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 50, 70, 5, "50-70 mg/L (or 10x MIC)", "<5 mg/L (often undetectable)"
        elif regimen_code == "Synergy": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 0, "N/A", "N/A"
        elif regimen_code == "Hemodialysis": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 0, 0, 10, "Peak not routinely targeted", "<10 mg/L (pre-dialysis)"
        elif regimen_code == "Neonates": target_peak_min, target_peak_max, target_trough_max, target_peak_info, target_trough_info = 20, 30, 5, "20-30 mg/L", "<5 mg/L"

    st.markdown("### Target Concentration Ranges:")
    col_t1, col_t2 = st.columns(2)
    with col_t1: st.markdown(f"**Peak Target:** {target_peak_info}")
    with col_t2: st.markdown(f"**Trough Target:** {target_trough_info}")

    # MIC input for SDD regimens
    mic = 1.0 # Default MIC
    if regimen_code == "SDD":
        st.markdown("*Note: Target peak for Extended Interval is often 10x MIC.*")
        mic = st.number_input("Enter MIC (mg/L)", min_value=0.1, value=1.0, step=0.1, format="%.1f")
        recommended_peak_mic = mic * 10
        st.info(f"Based on MIC, target peak is ≥ {recommended_peak_mic:.1f} mg/L. Adjust target below if needed.")
        # Update target peak min based on MIC if higher
        target_peak_min = max(target_peak_min, recommended_peak_mic)


    st.markdown("### Dosing and Sampling Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        dose = st.number_input("Dose Administered (mg)", min_value=10.0, value = 120.0, step=5.0)
        infusion_duration = st.number_input("Infusion Duration (hr)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    with col2:
        # Use current date as default, allow user to change if needed
        default_date = datetime.now().date()
        dose_start_datetime_dt = st.datetime_input("Date & Time of Dose Start", value=datetime.combine(default_date, time(12,0)), step=timedelta(minutes=15)) # Combine date and time(12,0)
    with col3:
        tau = st.number_input("Current Dosing Interval (hr)", min_value=4, max_value=72, value=8, step=4)

    st.markdown("### Measured Levels and Sample Times")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        c1 = st.number_input("Trough Level (C1, mg/L)", min_value=0.0, value=1.0, step=0.1, format="%.1f", help="Usually pre-dose level")
        c1_sample_datetime_dt = st.datetime_input("Date & Time of Trough Sample", value=datetime.combine(default_date, time(11,30)), step=timedelta(minutes=15)) # Default 30 min before 12pm dose
    with col_l2:
        c2 = st.number_input("Peak Level (C2, mg/L)", min_value=0.0, value=8.0, step=0.1, format="%.1f", help="Usually post-infusion level")
        c2_sample_datetime_dt = st.datetime_input("Date & Time of Peak Sample", value=datetime.combine(default_date, time(13,30)), step=timedelta(minutes=15)) # Default 30 min after 1hr infusion ends (1:30pm)


    # --- Calculate t1 and t2 relative to dose start time ---
    t1 = (c1_sample_datetime_dt - dose_start_datetime_dt).total_seconds() / 3600.0 # Time of C1 sample relative to dose start in hours
    t2 = (c2_sample_datetime_dt - dose_start_datetime_dt).total_seconds() / 3600.0 # Time of C2 sample relative to dose start in hours

    st.markdown(f"*Calculated time from dose start to Trough (C1) sample (t1): {t1:.2f} hr*")
    st.markdown(f"*Calculated time from dose start to Peak (C2) sample (t2): {t2:.2f} hr*")

    # Validate timings
    valid_times = True
    if t1 >= t2:
        st.error("❌ Trough sample time (C1) must be before Peak sample time (C2). Please check the dates and times.")
        valid_times = False
    if t2 <= infusion_duration:
        st.warning(f"⚠️ Peak sample time (t2={t2:.2f} hr) is during or before the end of infusion ({infusion_duration:.1f} hr). Calculated Cmax will be extrapolated; accuracy may be reduced.")
    if t1 > 0 and t1 < infusion_duration:
         st.warning(f"⚠️ Trough sample time (t1={t1:.2f} hr) appears to be during the infusion. Ensure C1 is a true pre-dose trough for most accurate calculations.")

    # --- Perform PK Calculations ---
    st.markdown("### Calculated Pharmacokinetic Parameters")
    results_calculated = False
    ke, t_half, vd, cl = 0, float('inf'), 0, 0
    cmax_extrapolated, cmin_extrapolated = 0, 0

    if valid_times:
        try:
            # Ensure levels are positive for log calculation
            if c1 <= 0 or c2 <= 0:
                st.error("❌ Measured levels (C1 and C2) must be greater than 0 for calculation.")

            else:
                # Calculate Ke using two levels (Sawchuk-Zaske method adaptation)
                # Assumes levels are in the elimination phase relative to each other
                delta_t = t2 - t1
                if delta_t <= 0: raise ValueError("Time difference between samples (t2-t1) must be positive.")

                # Check if both points are likely post-infusion for simple Ke calculation
                if t1 >= infusion_duration:
                     ke = (math.log(c1) - math.log(c2)) / delta_t # ln(C1/C2) / (t2-t1)
                else:
                     # If t1 is during infusion or pre-dose, Ke calculation is more complex.
                     # Using the simple formula introduces error. A Bayesian approach or iterative method is better.
                     # For this tool, we'll proceed with the simple formula but add a warning.
                     ke = (math.log(c1) - math.log(c2)) / delta_t
                     st.warning("⚠️ Ke calculated assuming log-linear decay between C1 and C2. Accuracy reduced if C1 is not post-infusion.")

                ke = max(1e-6, ke) # Ensure ke is positive and non-zero
                t_half = 0.693 / ke if ke > 0 else float('inf')

                # Extrapolate to find Cmax (at end of infusion) and Cmin (at end of interval)
                # C_t = C_known * exp(-ke * (t - t_known))
                # Cmax = C2 * exp(ke * (t2 - infusion_duration)) # Extrapolate C2 back to end of infusion
                cmax_extrapolated = c2 * math.exp(ke * (t2 - infusion_duration))

                # Cmin = Cmax_extrapolated * exp(-ke * (tau - infusion_duration)) # Trough at end of interval
                cmin_extrapolated = cmax_extrapolated * math.exp(-ke * (tau - infusion_duration))

                # Calculate Vd using Cmax and dose (steady-state infusion formula)
                # Vd = Dose * (1 - exp(-ke * T_inf)) / [Cmax * ke * T_inf * (1 - exp(-ke * tau))]
                term_inf = (1 - math.exp(-ke * infusion_duration))
                term_tau = (1 - math.exp(-ke * tau))
                denominator_vd = cmax_extrapolated * ke * infusion_duration * term_tau
                vd = 0.0
                if abs(denominator_vd) > 1e-9 and abs(term_inf) > 1e-9 : # Avoid division by zero
                    vd = (dose * term_inf) / denominator_vd
                    vd = max(1.0, vd) # Ensure Vd is at least 1L
                else:
                    st.warning("Could not calculate Vd accurately due to near-zero terms (check Ke, Tau, Infusion Duration).")

                cl = ke * vd if vd > 0 else 0.0

                st.markdown(f"**Individualized Ke:** {ke:.4f} hr⁻¹ | **t½:** {t_half:.2f} hr")
                st.markdown(f"**Est. Cmax (end of infusion):** {cmax_extrapolated:.1f} mg/L | **Est. Cmin (end of interval):** {cmin_extrapolated:.2f} mg/L")
                if vd > 0:
                     st.markdown(f"**Est. Vd:** {vd:.2f} L | **Est. CL:** {cl:.2f} L/hr")
                else:
                     st.markdown("**Est. Vd & CL:** Could not be calculated accurately.")

                results_calculated = True

                # --- Dose Recommendation ---
                st.markdown("### Dose Adjustment Recommendation")
                if vd <= 0 or ke <=0:
                     st.warning("Cannot calculate new dose recommendation due to invalid PK parameters.")
                else:
                    # Ask for desired target levels (default to mid-point of range or target min)
                    default_desired_peak = target_peak_min if regimen_code == "SDD" else (target_peak_min + target_peak_max) / 2
                    desired_peak = st.number_input("Desired Target    peak_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Peak.*?([\d.,]+)\s*mg/L", prompt)
    auc_val = extract_float(r"(?:Estimated|Predicted)\s+AUC24.*?([\d.,]+)\s*mg.hr/L", prompt)

    # Extract targets
    target_auc_str = extract_string(r"Target\s+AUC24.*?(\d+\s*-\s*\d+)\s*mg.hr/L", prompt, "400-600")
    target_trough_str = extract_string(r"(?:Target|Secondary Target)\s+Trough.*?([\d.]+\s*-\s*[\d.]+)\s*mg/L", prompt, "10-15")

    # Extract current/new regimen details
    current_dose_interval = extract_float(r"Current\s+Regimen.*?([\d,]+)\s*mg.*?q\s*(\d+)", prompt)
    current_interval = extract_float(r"Current\s+Regimen.*?q\s*(\d+)", prompt)
    new_dose_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?([\d,]+)\s*mg.*?q\s*(\d+)", prompt)
    new_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?q\s*(\d+)", prompt)


    # Parse target ranges
    auc_target_min, auc_target_max = 400, 600
    auc_match = re.match(r"(\d+)\s*-\s*(\d+)", target_auc_str)
    if auc_match: auc_target_min, auc_target_max = int(auc_match.group(1)), int(auc_match.group(2))
    auc_target_formatted = f"{auc_target_min}-{auc_target_max} mg·hr/L"

    trough_target_min, trough_target_max = 10, 15
    trough_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_trough_str)
    if trough_match:
        try:
            trough_target_min = float(trough_match.group(1))
            trough_target_max = float(trough_match.group(2))
        except ValueError: pass
    trough_target_formatted = f"{trough_target_min:.1f}-{trough_target_max:.1f} mg/L"


    # Check if essential values for assessment were extracted
    if trough_val is None and auc_val is None:
        return "Insufficient level data (Trough or AUC) in prompt for standardized vancomycin interpretation."

    # --- Start Interpretation Logic ---
    levels_data = []
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    status = "assessment uncertain" # Default status

    # Assess Trough Level
    trough_status = "N/A"
    if trough_val is not None:
        if trough_val < trough_target_min: trough_status = "below"
        elif trough_val > trough_target_max: trough_status = "above"
        else: trough_status = "within"
        levels_data.append(("Trough", trough_val, trough_target_formatted, trough_status))

    # Assess AUC Level
    auc_status = "N/A"
    if auc_val is not None:
        if auc_val < auc_target_min: auc_status = "below"
        elif auc_val > auc_target_max: auc_status = "above"
        else: auc_status = "within"
        levels_data.append(("AUC24", auc_val, auc_target_formatted, auc_status))

    # Assess Peak Level (if available)
    peak_status = "N/A"
    if peak_val is not None:
        # Define peak range based on empiric vs definitive therapy
        # Assuming trough level helps determine empiric vs definitive
        if trough_val is not None and trough_val <= 15:  # Likely empiric therapy
            peak_target_min, peak_target_max = 20, 30
            peak_target_formatted = f"{peak_target_min}-{peak_target_max} mg/L (Empiric)"
        else:  # Likely definitive therapy
            peak_target_min, peak_target_max = 25, 40
            peak_target_formatted = f"{peak_target_min}-{peak_target_max} mg/L (Definitive)"
        
        if peak_val < peak_target_min: peak_status = "below"
        elif peak_val > peak_target_max: peak_status = "above"
        else: peak_status = "within"
        levels_data.append(("Peak", peak_val, peak_target_formatted, peak_status))


    # Determine overall assessment status (prioritize AUC, then Trough)
    if auc_status == "within" and trough_status != "above": status = "appropriately dosed (AUC target met)"
    elif auc_status == "within" and trough_status == "above": status = "potentially overdosed (AUC ok, trough high)"
    elif auc_status == "below": status = "underdosed (AUC below target)"
    elif auc_status == "above": status = "overdosed (AUC above target)"
    elif auc_status == "N/A": # If AUC not available, use trough
         if trough_status == "within": status = "likely appropriately dosed (trough target met)"
         elif trough_status == "below": status = "likely underdosed (trough below target)"
         elif trough_status == "above": status = "likely overdosed (trough above target)"


    # Generate recommendations based on status
    if "appropriately dosed" in status:
        dosing_recs.append("CONTINUE current regimen.")
        if current_dose_interval and current_interval:
             dosing_recs.append(f"MAINTAIN {current_dose_interval:.0f}mg q{current_interval:.0f}h.")
        monitoring_recs.append("MONITOR renal function per protocol (e.g., 2-3 times weekly).")
        monitoring_recs.append("REPEAT levels if clinical status or renal function changes significantly.")
        if status == "potentially overdosed (AUC ok, trough high)": # Add caution if trough high despite AUC ok
             cautions.append("Trough is elevated, increasing potential nephrotoxicity risk despite acceptable AUC. Monitor renal function closely.")
             monitoring_recs.append("Consider rechecking trough sooner if renal function declines.")

    else: # Underdosed or Overdosed
        if "underdosed" in status:
             dosing_recs.append("INCREASE dose and/or shorten interval to achieve target AUC.")
             monitoring_recs.append("RECHECK levels after 3-5 doses of new regimen (allow steady state).")
        elif "overdosed" in status:
             if trough_val is not None and trough_val > 25: # Significantly high trough
                 dosing_recs.append("HOLD next dose(s) until trough is acceptable (e.g., < 20 mg/L).")
                 cautions.append("Significantly elevated trough increases nephrotoxicity risk.")
             dosing_recs.append("DECREASE dose and/or lengthen interval to achieve target AUC.")
             monitoring_recs.append("RECHECK levels within 24-48 hours after adjustment (or before next dose if interval long).")
             monitoring_recs.append("MONITOR renal function daily until stable.")

        # Suggest new regimen if provided in prompt
        if new_dose_interval and new_interval:
             # Suggest practical regimens based on new TDD
             new_tdd_calc = new_dose_interval * (24 / new_interval)
             suggested_regimens = []
             for practical_interval_opt in [8, 12, 24, 36, 48]: # Common intervals
                 dose_per_interval_opt = new_tdd_calc / (24 / practical_interval_opt)
                 # Round dose per interval to nearest 250mg
                 rounded_dose_opt = round(dose_per_interval_opt / 250) * 250
                 if rounded_dose_opt > 0:
                     # Check if this option is close to the suggested one
                     is_suggested = abs(practical_interval_opt - new_interval) < 1 and abs(rounded_dose_opt - new_dose_interval) < 125
                     prefix = "➡️" if is_suggested else "  -"
                     suggested_regimens.append(f"{prefix} {rounded_dose_opt:.0f}mg q{practical_interval_opt}h (approx. {rounded_dose_opt * (24/practical_interval_opt):.0f}mg/day)")

             if suggested_regimens:
                 dosing_recs.append(f"ADJUST regimen towards target AUC ({auc_target_formatted}). Consider practical options:")
                 # Add the explicitly suggested regimen first if found
                 explicit_suggestion = f"{new_dose_interval:.0f}mg q{new_interval:.0f}h"
                 if not any(explicit_suggestion in reg for reg in suggested_regimens):
                      dosing_recs.append(f"➡️ {explicit_suggestion} (Calculated)") # Add if not already covered by rounding
                 for reg in suggested_regimens:
                     dosing_recs.append(reg)

             else: # Fallback if no practical options generated
                  dosing_recs.append(f"ADJUST regimen to {new_dose_interval:.0f}mg q{new_interval:.0f}h as calculated.")
        else: # If no new dose calculated in prompt
             dosing_recs.append("ADJUST regimen based on clinical judgment and estimated PK to achieve target AUC.")


    # Add renal function caution if relevant
    if crcl is not None:
        renal_status = ""
        if crcl < 15: renal_status = "Kidney Failure"
        elif crcl < 30: renal_status = "Severe Impairment"
        elif crcl < 60: renal_status = "Moderate Impairment"
        elif crcl < 90: renal_status = "Mild Impairment"

        if crcl < 60: # Add caution for moderate to severe impairment
            cautions.append(f"{renal_status} (CrCl: {crcl:.1f} mL/min). Increased risk of accumulation and toxicity. Monitor levels and renal function closely.")
            if "overdosed" in status or (trough_val is not None and trough_val > target_trough_max):
                 monitoring_recs.append("MONITOR renal function at least daily.")
            else:
                 monitoring_recs.append("MONITOR renal function frequently (e.g., every 1-2 days).")

    cautions.append("Ensure appropriate infusion duration (e.g., ≥ 1 hour per gram, max rate 1g/hr) to minimize infusion reactions.")
    cautions.append("Consider potential drug interactions affecting vancomycin clearance or toxicity (e.g., piperacillin-tazobactam, loop diuretics, other nephrotoxins).")


    return levels_data, status, dosing_recs, monitoring_recs, cautions


def generate_aminoglycoside_interpretation(prompt, crcl=None):
    """
    Generate standardized aminoglycoside interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract drug name
    drug_match = re.search(r"Drug:\s*(Gentamicin|Amikacin)", prompt, re.IGNORECASE)
    drug_name = drug_match.group(1).lower() if drug_match else "aminoglycoside"

    # Extract levels (measured or estimated)
    peak_val = extract_float(r"(?:Measured|Estimated|Predicted|Est\.)\s+(?:Peak|Cmax).*?([\d.,]+)\s*mg/L", prompt)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted|Est\.)\s+(?:Trough|Cmin|C1).*?([\d.,]+)\s*mg/L", prompt) # Allow C1 as trough

    # Extract targets
    target_peak_str = extract_string(r"Target\s+Peak.*?((?:[\d.]+\s*-\s*[\d.]+|[<>]?\s*[\d.]+|N/A|Not routinely targeted))\s*mg/L", prompt, "N/A")
    target_trough_str = extract_string(r"Target\s+Trough.*?((?:[\d.]+\s*-\s*[\d.]+|[<>]?\s*[\d.]+|N/A))\s*mg/L", prompt, "N/A")

    # Extract current/new regimen details
    current_dose = extract_float(r"Current\s+Regimen.*?([\d,]+)\s*mg", prompt)
    current_interval = extract_float(r"Current\s+Regimen.*?q\s*(\d+)", prompt)
    new_dose = extract_float(r"(?:Suggested|New)\s+Regimen.*?([\d,]+)\s*mg", prompt)
    new_interval = extract_float(r"(?:Suggested|New)\s+Regimen.*?q\s*(\d+)", prompt)


    # --- Parse Target Ranges ---
    peak_min, peak_max = 0, 100 # Default wide range
    trough_limit_type = "max" # Assume target is '< max' by default
    trough_max = 100 # Default wide range

    # Parse Peak Target String
    if "N/A" in target_peak_str or "not targeted" in target_peak_str:
        peak_min, peak_max = None, None # Indicate not applicable
    else:
        peak_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_peak_str)
        if peak_match:
            try: peak_min, peak_max = float(peak_match.group(1)), float(peak_match.group(2))
            except ValueError: pass # Keep defaults if parsing fails

    # Parse Trough Target String
    if "N/A" in target_trough_str:
        trough_max = None # Indicate not applicable
    else:
        trough_match_less = re.match(r"<\s*([\d.]+)", target_trough_str)
        trough_match_range = re.match(r"([\d.]+)\s*-\s*([\d.]+)", target_trough_str)
        if trough_match_less:
            try: trough_max = float(trough_match_less.group(1)); trough_limit_type = "max"
            except ValueError: pass
        elif trough_match_range: # Handle if a range is given for trough (less common for amino)
             try: trough_max = float(trough_match_range.group(2)); trough_limit_type = "range"; trough_min = float(trough_match_range.group(1))
             except ValueError: pass # Default to max limit if range parsing fails


    # Check if essential level values were extracted
    if peak_val is None or trough_val is None:
        # Allow interpretation if only trough is available for HD patients
        if not ("Hemodialysis" in prompt and trough_val is not None):
             return "Insufficient level data (Peak or Trough) in prompt for standardized aminoglycoside interpretation."


    # --- Start Interpretation Logic ---
    levels_data = []
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    status = "assessment uncertain" # Default status

    # Assess Peak Level
    peak_status = "N/A"
    if peak_min is not None and peak_max is not None and peak_val is not None:
        if peak_val < peak_min: peak_status = "below"
        elif peak_val > peak_max: peak_status = "above"
        else: peak_status = "within"
        levels_data.append(("Peak", peak_val, target_peak_str, peak_status))
    elif peak_val is not None: # If target is N/A but value exists
         levels_data.append(("Peak", peak_val, target_peak_str, "N/A"))


    # Assess Trough Level
    trough_status = "N/A"
    if trough_max is not None and trough_val is not None:
        if trough_limit_type == "max":
            if trough_val >= trough_max: trough_status = "above" # At or above the max limit
            else: trough_status = "within" # Below the max limit
        elif trough_limit_type == "range":
             if trough_val < trough_min: trough_status = "below" # Below the range min (unlikely target for amino)
             elif trough_val > trough_max: trough_status = "above" # Above the range max
             else: trough_status = "within"
        levels_data.append(("Trough", trough_val, target_trough_str, trough_status))
    elif trough_val is not None: # If target is N/A but value exists
        levels_data.append(("Trough", trough_val, target_trough_str, "N/A"))


    # Determine overall assessment status
    # Prioritize avoiding toxicity (high trough), then achieving efficacy (adequate peak)
    if trough_status == "above":
        status = "potentially toxic (elevated trough)"
        if peak_status == "below": status = "ineffective and potentially toxic" # Worst case
    elif peak_status == "below":
        status = "subtherapeutic (inadequate peak)"
    elif peak_status == "above": # Peak high, trough ok
        status = "potentially supratherapeutic (high peak)"
    elif peak_status == "within" and trough_status == "within":
        status = "appropriately dosed"
    elif peak_status == "N/A" and trough_status == "within": # e.g., HD patient trough ok
         status = "likely appropriate (trough acceptable)"
    elif peak_status == "N/A" and trough_status == "above": # e.g., HD patient trough high
         status = "potentially toxic (elevated trough)"


    # Generate recommendations
    if "appropriately dosed" in status or "likely appropriate" in status :
        dosing_recs.append("CONTINUE current regimen.")
        if current_dose and current_interval: dosing_recs.append(f"MAINTAIN {current_dose:.0f}mg q{current_interval:.0f}h.")
        monitoring_recs.append("MONITOR renal function regularly (e.g., 2-3 times weekly or per HD schedule).")
        monitoring_recs.append("REPEAT levels if clinical status, renal function, or dialysis schedule changes.")
    elif status == "assessment not applicable": # Synergy Amikacin
         dosing_recs.append("Follow specific institutional protocol for Synergy Amikacin dosing.")
         monitoring_recs.append("MONITOR renal function and clinical status.")
    else: # Adjustments needed
        if status == "ineffective and potentially toxic":
             dosing_recs.append("HOLD next dose(s).")
             dosing_recs.append("INCREASE dose AND EXTEND interval significantly once resumed.")
             monitoring_recs.append("RECHECK levels (peak & trough) before resuming and after 2-3 doses of new regimen.")
             cautions.append("High risk of toxicity and low efficacy with current levels.")
        elif status == "subtherapeutic (inadequate peak)":
             dosing_recs.append("INCREASE dose.")
             dosing_recs.append("MAINTAIN current interval (unless trough also borderline high).")
             monitoring_recs.append("RECHECK peak and trough after 1-2 doses of new regimen.")
             cautions.append("Inadequate peak may compromise efficacy, especially for gram-negative infections.")
        elif status == "potentially toxic (elevated trough)":
             dosing_recs.append("EXTEND dosing interval.")
             dosing_recs.append("MAINTAIN current dose amount (or consider slight reduction if peak also high/borderline).")
             monitoring_recs.append("RECHECK trough level before next scheduled dose.")
             cautions.append("Elevated trough increases risk of nephrotoxicity and ototoxicity. Hold dose if trough significantly elevated.")
        elif status == "potentially supratherapeutic (high peak)": # High peak, trough ok
             dosing_recs.append("DECREASE dose.")
             dosing_recs.append("MAINTAIN current interval.")
             monitoring_recs.append("RECHECK peak and trough after 1-2 doses of new regimen.")
             cautions.append("Elevated peak may increase toxicity risk slightly, though trough is primary driver. Ensure trough remains acceptable.")

        # Suggest new regimen if provided in prompt
        if new_dose and new_interval:
             # Round new dose to nearest 10mg or 20mg
             rounding = 20 if drug_name == "gentamicin" else 50 if drug_name == "amikacin" else 10
             practical_new_dose = round(new_dose / rounding) * rounding
             if practical_new_dose > 0:
                 dosing_recs.append(f"Consider adjusting regimen towards: {practical_new_dose:.0f}mg q{new_interval:.0f}h.")
        else:
             dosing_recs.append("ADJUST regimen based on clinical judgment and estimated PK to achieve target levels.")


    # Add general monitoring and cautions
    monitoring_recs.append("MONITOR renal function (SCr, BUN, UOP) at least 2-3 times weekly, or more frequently if unstable, trough elevated, or on concomitant nephrotoxins.")
    monitoring_recs.append("MONITOR for signs/symptoms of nephrotoxicity (rising SCr, decreased UOP) and ototoxicity (hearing changes, tinnitus, vertigo).")
    cautions.append(f"{drug_name.capitalize()} carries risk of nephrotoxicity and ototoxicity.")
    cautions.append("Risk increases with prolonged therapy (>7-10 days), pre-existing renal impairment, high troughs, large cumulative dose, and concomitant nephrotoxins (e.g., vancomycin, diuretics, contrast).")
    if crcl is not None:
         renal_status = ""
         if crcl < 15: renal_status = "Kidney Failure"
         elif crcl < 30: renal_status = "Severe Impairment"
         elif crcl < 60: renal_status = "Moderate Impairment"
         elif crcl < 90: renal_status = "Mild Impairment"
         if crcl < 60:
             cautions.append(f"{renal_status} (CrCl: {crcl:.1f} mL/min) significantly increases toxicity risk. Adjust dose/interval carefully and monitor very closely.")


    return levels_data, status, dosing_recs, monitoring_recs, cautions


# ===== SIDEBAR: NAVIGATION AND PATIENT INFO =====
def setup_sidebar_and_navigation():
    st.sidebar.title("📊 Navigation")
    # Sidebar radio for selecting the module
    page = st.sidebar.radio("Select Module", [
        "Aminoglycoside: Initial Dose",
        "Aminoglycoside: Conventional Dosing (C1/C2)",
        "Vancomycin AUC-based Dosing"
    ])

    st.sidebar.title("🩺 Patient Demographics")
    # ADDED Patient ID and Ward
    patient_id = st.sidebar.text_input("Patient ID", value="N/A")
    ward = st.sidebar.text_input("Ward", value="N/A")
    # --- Existing fields ---
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=65)
    height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=165)
    weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1, format="%.1f")
    serum_cr = st.sidebar.number_input("Serum Creatinine (µmol/L)", min_value=10.0, max_value=2000.0, value=90.0, step=1.0)

    # Calculate Cockcroft-Gault Creatinine Clearance
    crcl = 0.0 # Default value
    renal_function = "N/A"
    if age > 0 and weight > 0 and serum_cr > 0: # Avoid division by zero or negative age
        # Cockcroft-Gault Formula
        crcl_factor = (140 - age) * weight
        crcl_gender_mult = 1.23 if gender == "Male" else 1.04
        crcl = (crcl_factor * crcl_gender_mult) / serum_cr
        crcl = max(0, crcl) # Ensure CrCl is not negative

        # Renal function category based on CrCl
        if crcl >= 90: renal_function = "Normal (≥90)"
        elif crcl >= 60: renal_function = "Mild Impairment (60-89)"
        elif crcl >= 30: renal_function = "Moderate Impairment (30-59)"
        elif crcl >= 15: renal_function = "Severe Impairment (15-29)"
        else: renal_function = "Kidney Failure (<15)"

    with st.sidebar.expander("Creatinine Clearance (Cockcroft-Gault)", expanded=True):
        if age > 0 and weight > 0 and serum_cr > 0:
            st.success(f"CrCl: {crcl:.1f} mL/min")
            st.info(f"Renal Function: {renal_function}")
        else:
            st.warning("Enter valid Age (>0), Weight (>0), and SCr (>0) to calculate CrCl.")


    st.sidebar.title("🩺 Clinical Information")
    clinical_diagnosis = st.sidebar.text_input("Diagnosis / Indication", placeholder="e.g., Pneumonia, Sepsis")
    current_dose_regimen = st.sidebar.text_area("Current Dosing Regimen", value="1g IV q12h", placeholder="e.g., Gentamicin 120mg IV q8h")
    notes = st.sidebar.text_area("Other Clinical Notes", value="No known allergies.", placeholder="e.g., Fluid status, interacting meds")

    # UPDATED clinical_summary
    clinical_summary = (
        f"Patient ID: {patient_id}, Ward: {ward}\n"
        f"Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} cm\n"
        f"SCr: {serum_cr} µmol/L\n"
        f"Diagnosis: {clinical_diagnosis}\n"
        f"Renal function: {renal_function} (Est. CrCl: {crcl:.1f} mL/min)\n"
        f"Current regimen: {current_dose_regimen}\n"
        f"Notes: {notes}"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Antimicrobial TDM App v1.2**

    Developed for therapeutic drug monitoring of antimicrobials.

    Provides PK estimates, AUC calculations, and dosing recommendations
    for vancomycin and aminoglycosides. Includes optional LLM interpretation.

    **Disclaimer:** This tool assists clinical decision making but does not replace
    professional judgment. Verify all calculations and recommendations.
    """)

    # Return all the data entered in the sidebar
    return {
        'page': page,
        'patient_id': patient_id, # Added
        'ward': ward,           # Added
        'gender': gender,
        'age': age,
        'height': height,
        'weight': weight,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen,
        'notes': notes,
        'clinical_summary': clinical_summary # Updated summary string
    }

# ===== MODULE 1: Aminoglycoside Initial Dose =====
def aminoglycoside_initial_dose(patient_data):
    st.title("🧮 Aminoglycoside Initial Dose (Population PK)")

    # Unpack patient data for easier access
    gender = patient_data['gender']
    age = patient_data['age']
    height = patient_data['height']
    weight = patient_data['weight']
    crcl = patient_data['crcl']
    notes = patient_data['notes']

    drug = st.selectbox("Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Dosing Strategy / Goal", ["Extended Interval (Once Daily - SDD)", "Traditional (Multiple Daily - MDD)", "Synergy (e.g., Endocarditis)", "Hemodialysis", "Neonates (Use with caution)"])

    # Map selection to internal codes
    regimen_code = "SDD" if "Extended" in regimen \
              else "MDD" if "Traditional" in regimen \
              else "Synergy" if "Synergy" in regimen \
              else "Hemodialysis" if "Hemodialysis" in regimen \
              else "Neonates" if "Neonates" in regimen \
              else "MDD" # Default

    # --- Set default target ranges based on regimen and drug ---
    target_peak_info = "N/A"
    target_trough_info = "N/A"
    default_peak = 0.0
    default_trough = 0.0

    if drug == "Gentamicin":import streamlit as st
import numpy as np
import math
import openai
import pandas as pd
import altair as alt
import base64
from datetime import datetime, time, timedelta # Added time and timedelta

# Optional imports - Bayesian functionality
try:
    import scipy.optimize as optimize
    from scipy.stats import norm
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scipy not installed. Bayesian estimation will not be available.")

# Optional imports - FAISS for guideline embedding (if needed)
try:
    import faiss
    import pickle
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Guideline embeddings will not be available.")

st.set_page_config(page_title="Antimicrobial TDM App", layout="wide")

# ===== API CONFIGURATION =====
# Securely access the API key from streamlit secrets
try:
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.

    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== LOAD EMBEDDED GUIDELINE =====
@st.cache_resource
def load_guideline_embeddings():
    if not FAISS_AVAILABLE:
        return None, ["FAISS not installed. Guideline embeddings not available."]

    # Replace this with your actual FAISS index and guideline chunks
    index = faiss.IndexFlatL2(768)  # Dummy placeholder index
    chunks = ["Guideline excerpt 1", "Guideline excerpt 2", "Guideline excerpt 3"]
    return index, chunks

guideline_index, guideline_chunks = None, []
if FAISS_AVAILABLE:
    guideline_index, guideline_chunks = load_guideline_embeddings()

# ===== Global Practical Dosing Regimens =====
practical_intervals = "6hr, 8hr, 12hr, 24hr, every other day"

# ===== HELPER FUNCTION: Updated Practical Dose Adjustment =====
def suggest_adjustment(parameter, target_min, target_max, label="Parameter", intervals=practical_intervals):
    # Ensure targets are valid numbers before comparison
    if isinstance(target_min, (int, float)) and isinstance(target_max, (int, float)) and isinstance(parameter, (int, float)):
        if parameter < target_min:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is low. Target: {target_min:.1f}-{target_max:.1f}. Consider increasing dose or shortening interval ({intervals}).")
        elif parameter > target_max:
            st.warning(f"⚠️ {label} ({parameter:.1f}) is high. Target: {target_min:.1f}-{target_max:.1f}. Consider reducing dose or lengthening interval ({intervals}).")
        else:
            st.success(f"✅ {label} ({parameter:.1f}) is within target range ({target_min:.1f}-{target_max:.1f}).")
    else:
        st.info(f"{label}: {parameter}. Target range: {target_min}-{target_max}. (Comparison skipped due to non-numeric values).")


# ===== PDF GENERATION FUNCTIONS (REMOVED) =====
# create_recommendation_pdf, get_pdf_download_link, display_pdf_download_button functions removed.

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization

    Parameters:
    - peak: Peak concentration (mg/L)
    - trough: Trough concentration (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - t_peak: Time to peak after start of infusion (hr) - assumed end of infusion
    - infusion_time: Duration of infusion (hr)

    Returns:
    - Altair chart object
    """
    # Generate time points for the curve
    times = np.linspace(0, tau*1.5, 150)  # Generate points for 1.5 intervals to show next dose

    # Generate concentrations for each time point using steady-state equations
    concentrations = []
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * T_inf)) * exp(-ke * (t - T_inf)) / (1 - exp(-ke * tau)) -- Post-infusion
    # C(t) = (Dose / (Vd * ke * T_inf)) * (1 - exp(-ke * t)) / (1 - exp(-ke * tau)) -- During infusion (simplified, assumes Cmin=0 start)
    # Let's use the provided peak and trough which represent Cmax (at t=infusion_time) and Cmin (at t=tau)

    for t_cycle in np.linspace(0, tau*1.5, 150): # Iterate through time points
        # Determine concentration based on time within the dosing cycle (modulo tau)
        t = t_cycle % tau
        num_cycles = int(t_cycle // tau) # Which cycle we are in (0, 1, ...)

        conc = 0
        if t <= infusion_time:
            # During infusion: Assume linear rise from previous trough to current peak
            # This is an approximation but visually represents the infusion period
            conc = trough + (peak - trough) * (t / infusion_time)
        else:
            # After infusion: Exponential decay from peak
            time_since_peak = t - infusion_time # Time elapsed since the peak concentration (end of infusion)
            conc = peak * math.exp(-ke * time_since_peak)

        concentrations.append(max(0, conc)) # Ensure concentration doesn't go below 0


    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Time (hr)': times,
        'Concentration (mg/L)': concentrations
    })

    # --- Create Target Bands ---
    target_bands = []
    # Determine drug type based on typical levels for band coloring
    if peak > 45 or trough > 20:  # Likely vancomycin
        # Vancomycin Peak Target - Empiric vs Definitive
        if trough <= 15:  # Likely empiric (target trough 10-15)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [20], 'y2': [30]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Empiric)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [10], 'y2': [15]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Empiric)")))
        else:  # Likely definitive (target trough 15-20)
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [25], 'y2': [40]}))
                               .mark_rect(opacity=0.15, color='lightblue')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Vanco Definitive)")))
            target_bands.append(alt.Chart(pd.DataFrame({'y1': [15], 'y2': [20]}))
                               .mark_rect(opacity=0.15, color='lightgreen')
                               .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Vanco Definitive)")))
    else:  # Likely aminoglycoside (e.g., Gentamicin)
        # Aminoglycoside Peak Target (e.g., 5-10 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [5], 'y2': [10]}))
                           .mark_rect(opacity=0.15, color='lightblue')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Peak Range (Amino)")))
        # Aminoglycoside Trough Target (e.g., <2 for Gent MDD)
        target_bands.append(alt.Chart(pd.DataFrame({'y1': [0], 'y2': [2]}))
                           .mark_rect(opacity=0.15, color='lightgreen')
                           .encode(y='y1', y2='y2', tooltip=alt.value("Target Trough Range (Amino)")))


    # --- Create Concentration Line ---
    line = alt.Chart(df).mark_line(color='firebrick').encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)', scale=alt.Scale(zero=True)), # Ensure Y axis starts at 0
        tooltip=['Time (hr)', alt.Tooltip('Concentration (mg/L)', format=".1f")]
    )

    # --- Add Vertical Lines for Key Events ---
    vertical_lines_data = []
    # Mark end of infusion for each cycle shown
    for i in range(int(tau*1.5 / tau) + 1):
        inf_end_time = i * tau + infusion_time
        if inf_end_time <= tau*1.5:
             vertical_lines_data.append({'Time': inf_end_time, 'Event': 'Infusion End'})
    # Mark start of next dose for each cycle shown
    for i in range(1, int(tau*1.5 / tau) + 1):
         dose_time = i * tau
         if dose_time <= tau*1.5:
              vertical_lines_data.append({'Time': dose_time, 'Event': 'Next Dose'})

    vertical_lines_df = pd.DataFrame(vertical_lines_data)

    vertical_rules = alt.Chart(vertical_lines_df).mark_rule(strokeDash=[4, 4]).encode(
        x='Time',
        color=alt.Color('Event', scale=alt.Scale(domain=['Infusion End', 'Next Dose'], range=['gray', 'black'])),
        tooltip=['Event', 'Time']
    )

    # --- Combine Charts ---
    chart = alt.layer(*target_bands, line, vertical_rules).properties(
        width=alt.Step(4), # Adjust width automatically
        height=400,
        title=f'Estimated Concentration-Time Profile (Tau={tau} hr)'
    ).interactive() # Make chart interactive (zoom/pan)

    return chart


# ===== VANCOMYCIN AUC CALCULATION (TRAPEZOIDAL METHOD) =====
def calculate_vancomycin_auc_trapezoidal(cmax, cmin, ke, tau, infusion_duration):
    """
    Calculate vancomycin AUC24 using the linear-log trapezoidal method.
    
    This method is recommended for vancomycin TDM as per the guidelines.
    
    Parameters:
    - cmax: Max concentration at end of infusion (mg/L)
    - cmin: Min concentration at end of interval (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - infusion_duration: Duration of infusion (hr)
    
    Returns:
    - AUC24: 24-hour area under the curve (mg·hr/L)
    """
    # Calculate concentration at start of infusion (C0)
    c0 = cmax * math.exp(ke * infusion_duration)
    
    # Calculate AUC during infusion phase (linear trapezoid)
    auc_inf = infusion_duration * (c0 + cmax) / 2
    
    # Calculate AUC during elimination phase (log trapezoid)
    if ke > 0 and cmax > cmin:
        auc_elim = (cmax - cmin) / ke
    else:
        # Fallback to linear trapezoid if ke is very small
        auc_elim = (tau - infusion_duration) * (cmax + cmin) / 2
    
    # Calculate total AUC for one dosing interval
    auc_interval = auc_inf + auc_elim
    
    # Convert to AUC24
    auc24 = auc_interval * (24 / tau)
    
    return auc24

# ===== BAYESIAN PARAMETER ESTIMATION =====
def bayesian_parameter_estimation(measured_levels, sample_times, dose, tau, weight, age, crcl, gender):
    """
    Bayesian estimation of PK parameters based on measured levels

    Parameters:
    - measured_levels: List of measured drug concentrations (mg/L)
    - sample_times: List of times when samples were taken (hr after dose start)
    - dose: Dose administered (mg)
    - tau: Dosing interval (hr)
    - weight: Patient weight (kg)
    - age: Patient age (years)
    - crcl: Creatinine clearance (mL/min)
    - gender: Patient gender ("Male" or "Female")

    Returns:
    - Dictionary with estimated PK parameters
    """
    if not BAYESIAN_AVAILABLE:
        st.error("Bayesian estimation requires scipy. Please install it with 'pip install scipy'")
        return None

    # Prior population parameters for vancomycin (adjust if needed for aminoglycosides)
    # Mean values
    vd_pop_mean = 0.7  # L/kg (Vancomycin specific, adjust for aminoglycosides if used)
    ke_pop_mean = 0.00083 * crcl + 0.0044 # hr^-1 (Vancomycin specific - ensure crcl is used correctly)
    ke_pop_mean = max(0.01, ke_pop_mean) # Ensure Ke isn't too low

    # Standard deviations for population parameters
    vd_pop_sd = 0.2  # L/kg
    ke_pop_sd = 0.05 # Increased SD for Ke prior to allow more flexibility

    # Define objective function to minimize (negative log likelihood)
    def objective_function(params):
        vd_ind, ke_ind = params # Individual parameters to estimate
        vd_total = vd_ind * weight

        # Calculate expected concentrations at sample times using steady-state infusion model
        expected_concs = []
        infusion_time = 1.0 # Assume 1 hour infusion, make adjustable if needed

        for t in sample_times:
            # Steady State Concentration Equation (1-compartment, intermittent infusion)
            term_dose_vd = dose / vd_total
            term_ke_tinf = ke_ind * infusion_time
            term_ke_tau = ke_ind * tau

            try:
                exp_ke_tinf = math.exp(-term_ke_tinf)
                exp_ke_tau = math.exp(-term_ke_tau)

                if abs(1.0 - exp_ke_tau) < 1e-9: # Avoid division by zero if tau is very long or ke very small
                    # Handle as if continuous infusion or single dose if tau is effectively infinite
                    conc = 0 # Simplified - needs better handling for edge cases
                else:
                    common_factor = (term_dose_vd / term_ke_tinf) * (1.0 - exp_ke_tinf) / (1.0 - exp_ke_tau)

                    if t <= infusion_time: # During infusion phase
                        conc = common_factor * (1.0 - math.exp(-ke_ind * t))
                    else: # Post-infusion phase
                        conc = common_factor * math.exp(-ke_ind * (t - infusion_time))

            except OverflowError:
                 conc = float('inf') # Handle potential overflow with large ke/t values
            except ValueError:
                 conc = 0 # Handle math domain errors

            expected_concs.append(max(0, conc)) # Ensure non-negative

        # Calculate negative log likelihood
        # Measurement error model (e.g., proportional + additive)
        # sd = sqrt(sigma_add^2 + (sigma_prop * expected_conc)^2)
        sigma_add = 1.0  # Additive SD (mg/L)
        sigma_prop = 0.1 # Proportional SD (10%)
        nll = 0
        for i in range(len(measured_levels)):
            expected = expected_concs[i]
            measurement_sd = math.sqrt(sigma_add**2 + (sigma_prop * expected)**2)
            if measurement_sd < 1e-6: measurement_sd = 1e-6 # Prevent division by zero in logpdf

            # Add contribution from measurement likelihood
            # Use logpdf for robustness, especially with low concentrations
            nll += -norm.logpdf(measured_levels[i], loc=expected, scale=measurement_sd)

        # Add contribution from parameter priors (log scale often more stable for Ke)
        # Prior for Vd (Normal)
        nll += -norm.logpdf(vd_ind, loc=vd_pop_mean, scale=vd_pop_sd)
        # Prior for Ke (Log-Normal might be better, but using Normal for simplicity)
        nll += -norm.logpdf(ke_ind, loc=ke_pop_mean, scale=ke_pop_sd)

        # Penalize non-physical parameters slightly if optimization strays
        if vd_ind <= 0 or ke_ind <= 0:
             nll += 1e6 # Add large penalty

        return nll

    # Initial guess based on population values
    initial_params = [vd_pop_mean, ke_pop_mean]

    # Parameter bounds (physical constraints)
    bounds = [(0.1, 2.5), (0.001, 0.5)]  # Reasonable bounds for Vd (L/kg) and Ke (hr^-1)

    # Perform optimization using a robust method
    try:
        result = optimize.minimize(
            objective_function,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B', # Suitable for bound constraints
            options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 500} # Adjust tolerances/iterations
        )
    except Exception as e:
         st.error(f"Optimization failed: {e}")
         return None

    if not result.success:
        st.warning(f"Bayesian optimization did not converge: {result.message} (Function evaluations: {result.nfev})")
        # Optionally return population estimates or None
        return None # Indicate failure

    # Extract optimized parameters
    vd_opt_kg, ke_opt = result.x
    # Ensure parameters are within bounds post-optimization (should be handled by L-BFGS-B, but double-check)
    vd_opt_kg = max(bounds[0][0], min(bounds[0][1], vd_opt_kg))
    ke_opt = max(bounds[1][0], min(bounds[1][1], ke_opt))

    vd_total_opt = vd_opt_kg * weight
    cl_opt = ke_opt * vd_total_opt
    t_half_opt = 0.693 / ke_opt if ke_opt > 0 else float('inf')

    return {
        'vd': vd_opt_kg, # Vd per kg
        'vd_total': vd_total_opt, # Total Vd in L
        'ke': ke_opt,
        'cl': cl_opt,
        't_half': t_half_opt,
        'optimization_success': result.success,
        'final_nll': result.fun # Final negative log-likelihood value
    }


# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM.
    Uses OpenAI API if available, otherwise provides a simulated response.

    Parameters:
    - prompt: The clinical data prompt including calculated values and context.
    - patient_data: Dictionary with patient information (used for context).
    """
    # Extract the drug type from the prompt for context
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
    else:
        drug = "Antimicrobial"

    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Include patient context if relevant (e.g., renal function).
            Format your response with these exact sections:

            ## CLINICAL ASSESSMENT
            📊 **MEASURED/ESTIMATED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **ASSESSMENT:** (state if appropriately dosed, underdosed, or overdosed based on levels and targets)

            ## RECOMMENDATIONS
            🔵 **DOSING:** (action-oriented recommendations using verbs like CONTINUE, ADJUST, HOLD, INCREASE, DECREASE. Suggest practical regimens where possible.)
            🔵 **MONITORING:** (specific monitoring parameters and schedule, e.g., recheck levels, renal function)
            ⚠️ **CAUTIONS:** (relevant warnings, e.g., toxicity risk, renal impairment)

            Here is the case:
            --- Patient Context ---
            Age: {patient_data.get('age', 'N/A')} years, Gender: {patient_data.get('gender', 'N/A')}
            Weight: {patient_data.get('weight', 'N/A')} kg, Height: {patient_data.get('height', 'N/A')} cm
            Patient ID: {patient_data.get('patient_id', 'N/A')}, Ward: {patient_data.get('ward', 'N/A')}
            Serum Cr: {patient_data.get('serum_cr', 'N/A')} µmol/L, CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})
            Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}
            Current Regimen: {patient_data.get('current_dose_regimen', 'N/A')}
            Notes: {patient_data.get('notes', 'N/A')}
            --- TDM Data & Calculations ---
            {prompt}
            --- End of Case ---
            """

            # Call OpenAI API - updated for openai v1.0.0+
            response = openai.chat.completions.create(
                model="gpt-4",  # or your preferred model like gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear, actionable recommendations in the specified format."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3, # Lower temperature for more deterministic clinical advice
                max_tokens=600 # Increased token limit for detailed response
            )
            llm_response = response.choices[0].message.content

            st.subheader("Clinical Interpretation (LLM)")
            st.markdown(llm_response) # Display the formatted response directly
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")

            # No PDF generation needed here

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
            # Fall through to standardized interpretation if API fails

    # If OpenAI is not available/fails, use the standardized interpretation
    if not (OPENAI_AVAILABLE and openai.api_key): # Or if the API call failed above
        st.subheader("Clinical Interpretation (Simulated)")
        interpretation_data = generate_standardized_interpretation(prompt, drug, patient_data)

        # If the interpretation_data is a string (error message), just display it
        if isinstance(interpretation_data, str):
            st.write(interpretation_data)
            return

        # Unpack the interpretation data
        levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data

        # Display the formatted interpretation
        formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
        st.markdown(formatted_interpretation) # Use markdown for better formatting

        # Add note about simulated response
        st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")

    # Add the raw prompt at the bottom for debugging/transparency
    with st.expander("Raw Analysis Data Sent to LLM (or used for Simulation)", expanded=False):
        st.code(prompt)


def generate_standardized_interpretation(prompt, drug, patient_data):
    """
    Generate a standardized interpretation based on drug type and prompt content.
    Includes patient context for better recommendations.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    crcl = patient_data.get('crcl', None) # Get CrCl for context

    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt, crcl)
    elif drug == "Aminoglycoside":
        return generate_aminoglycoside_interpretation(prompt, crcl)
    else:
        # For generic, create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        if crcl and crcl < 60:
             cautions.append(f"Renal function (CrCl: {crcl:.1f} mL/min) may impact dosing.")

        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy using Markdown.

    Parameters:
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string (e.g., "appropriately dosed")
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - cautions: Optional list of caution strings

    Returns:
    - Formatted markdown string
    """
    # Format measured levels with status indicators
    levels_md = "📊 **MEASURED/ESTIMATED LEVELS:**\n"
    if not levels_data or (len(levels_data) == 1 and levels_data[0][0] == "Not available"):
         levels_md += "- No levels data available for interpretation.\n"
    else:
        for name, value, target, status in levels_data:
            icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴" # Red for above
            # Format value appropriately (e.g., 1 decimal for levels, 0 for AUC)
            value_str = f"{value:.1f}" if isinstance(value, (int, float)) and "AUC" not in name else f"{value:.0f}" if isinstance(value, (int, float)) else str(value)
            levels_md += f"- {name}: {value_str} (Target: {target}) {icon}\n"


    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is **{assessment.upper()}**"

    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## RECOMMENDATIONS

🔵 **DOSING:**
"""
    if dosing_recs:
        for rec in dosing_recs:
            output += f"- {rec}\n"
    else:
        output += "- No specific dosing recommendations generated.\n"

    output += "\n🔵 **MONITORING:**\n"
    if monitoring_recs:
        for rec in monitoring_recs:
            output += f"- {rec}\n"
    else:
        output += "- Standard monitoring applies.\n"

    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"

    return output

def generate_vancomycin_interpretation(prompt, crcl=None):
    """
    Generate standardized vancomycin interpretation. Includes CrCl context.

    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - cautions: List of cautions

    Or returns a string if insufficient data
    """
    # Extract key values from the prompt using regex for robustness
    import re

    def extract_float(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE) # Ignore case
        try:
            # Handle potential commas in numbers
            return float(match.group(1).replace(',', '')) if match else default
        except (ValueError, IndexError, AttributeError):
            return default

    def extract_string(pattern, text, default="N/A"):
         match = re.search(pattern, text, re.IGNORECASE) # Ignore case
         return match.group(1).strip() if match else default

    # Extract levels (measured or estimated)
    trough_val = extract_float(r"(?:Measured|Estimated|Predicted)\s+Trough.*?([\d.,]+)\s*mg/L", prompt)
    peak_val
