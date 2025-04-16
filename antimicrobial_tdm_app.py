import streamlit as st
import numpy as np
import math
import openai
import pandas as pd
import altair as alt

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
    if parameter < target_min:
        st.warning(f"⚠️ {label} is low. Consider increasing the dose or shortening the interval to a practical regimen ({intervals}).")
    elif parameter > target_max:
        st.warning(f"⚠️ {label} is high. Consider reducing the dose or lengthening the interval to a practical regimen ({intervals}).")
    else:
        st.success(f"✅ {label} is within target range.")

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization
    
    Parameters:
    - peak: Peak concentration (mg/L)
    - trough: Trough concentration (mg/L)
    - ke: Elimination rate constant (hr^-1)
    - tau: Dosing interval (hr)
    - t_peak: Time to peak after start of infusion (hr)
    - infusion_time: Duration of infusion (hr)
    
    Returns:
    - Altair chart object
    """
    # Generate time points for the curve
    times = np.linspace(0, tau*1.5, 100)  # Generate points for 1.5 intervals to show next dose
    
    # Generate concentrations for each time point
    concentrations = []
    
    # Create time points and corresponding concentrations
    for t in times:
        # During first infusion
        if t <= infusion_time:
            # Linear increase during infusion
            conc = peak * (t / infusion_time)
        # After infusion, before next dose
        elif t <= tau:
            # Exponential decay after peak
            t_after_peak = t - t_peak
            conc = peak * np.exp(-ke * t_after_peak)
        # During second infusion
        elif t <= tau + infusion_time:
            # Second dose starts with trough and increases linearly during infusion
            t_in_second_infusion = t - tau
            conc = trough + (peak - trough) * (t_in_second_infusion / infusion_time)
        # After second infusion
        else:
            # Exponential decay after second peak
            t_after_second_peak = t - (tau + t_peak)
            conc = peak * np.exp(-ke * t_after_second_peak)
            
        concentrations.append(conc)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Time (hr)': times,
        'Concentration (mg/L)': concentrations
    })
    
    # Create horizontal bands for target ranges
    if peak > 50:  # Likely vancomycin
        target_peak_band = alt.Chart(pd.DataFrame({
            'y1': [20], 'y2': [40]  # Typical peak range for vancomycin
        })).mark_rect(opacity=0.2, color='green').encode(
            y='y1', y2='y2'
        )
        target_trough_band = alt.Chart(pd.DataFrame({
            'y1': [10], 'y2': [15]  # Typical trough range for vancomycin
        })).mark_rect(opacity=0.2, color='blue').encode(
            y='y1', y2='y2'
        )
    else:  # Likely aminoglycoside
        target_peak_band = alt.Chart(pd.DataFrame({
            'y1': [6], 'y2': [12]  # Typical peak range for gentamicin
        })).mark_rect(opacity=0.2, color='green').encode(
            y='y1', y2='y2'
        )
        target_trough_band = alt.Chart(pd.DataFrame({
            'y1': [0], 'y2': [2]  # Typical trough range for gentamicin
        })).mark_rect(opacity=0.2, color='blue').encode(
            y='y1', y2='y2'
        )
    
    # Create the concentration-time curve
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)')
    )
    
    # Add vertical lines for key time points
    infusion_end = alt.Chart(pd.DataFrame({'x': [infusion_time]})).mark_rule(
        strokeDash=[5, 5], color='gray'
    ).encode(x='x')
    
    next_dose = alt.Chart(pd.DataFrame({'x': [tau]})).mark_rule(
        strokeDash=[5, 5], color='red'
    ).encode(x='x')
    
    # Combine charts
    chart = alt.layer(
        target_peak_band, 
        target_trough_band, 
        line, 
        infusion_end,
        next_dose
    ).properties(
        width=600,
        height=400,
        title='Concentration-Time Profile'
    )
    
    return chart
    # ===== BAYESIAN PARAMETER ESTIMATION =====
def bayesian_parameter_estimation(measured_levels, sample_times, dose, tau, weight, age, crcl, gender):
    """
    Bayesian estimation of PK parameters based on measured levels
    
    Parameters:
    - measured_levels: List of measured drug concentrations (mg/L)
    - sample_times: List of times when samples were taken (hr after dose)
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
        
    # Prior population parameters for vancomycin
    # Mean values
    vd_pop_mean = 0.7  # L/kg
    ke_pop_mean = 0.0044 + 0.00083 * crcl  # hr^-1
    
    # Standard deviations for population parameters
    vd_pop_sd = 0.2  # L/kg 
    ke_pop_sd = 0.002  # hr^-1
    
    # Define objective function to minimize (negative log likelihood)
    def objective_function(params):
        vd, ke = params
        vd_total = vd * weight
        
        # Calculate expected concentrations at sample times
        expected_concs = []
        for t in sample_times:
            # Calculate which dosing interval this sample is in
            interval = int(t / tau)
            t_in_interval = t % tau
            
            # Calculate concentration at this time
            conc = 0
            for i in range(interval + 1):
                # Add contribution from each previous dose
                t_after_dose = t - i * tau
                if t_after_dose > 0:
                    # Assuming 1-hour infusion with immediate distribution
                    if t_after_dose <= 1:
                        # During infusion: linear increase
                        peak_conc = dose / vd_total
                        conc += peak_conc * (t_after_dose / 1)
                    else:
                        # After infusion: exponential decay
                        peak_conc = dose / vd_total
                        conc += peak_conc * np.exp(-ke * (t_after_dose - 1))
            
            expected_concs.append(conc)
        
        # Calculate negative log likelihood
        # Assuming measurement error is normally distributed
        measurement_error_sd = 2.0  # mg/L
        nll = 0
        for i in range(len(measured_levels)):
            # Add contribution from measurement likelihood
            nll += -norm.logpdf(measured_levels[i], expected_concs[i], measurement_error_sd)
        
        # Add contribution from parameter priors
        nll += -norm.logpdf(vd, vd_pop_mean, vd_pop_sd)
        nll += -norm.logpdf(ke, ke_pop_mean, ke_pop_sd)
        
        return nll
    
    # Initial guess based on population values
    initial_params = [vd_pop_mean, ke_pop_mean]
    
    # Parameter bounds
    bounds = [(0.1, 2.0), (0.001, 0.3)]  # Reasonable bounds for Vd and Ke
    
    # Perform optimization
    result = optimize.minimize(
        objective_function, 
        initial_params,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    # Extract optimized parameters
    vd_opt, ke_opt = result.x
    vd_total_opt = vd_opt * weight
    cl_opt = ke_opt * vd_total_opt
    t_half_opt = 0.693 / ke_opt
    
    return {
        'vd': vd_opt,
        'vd_total': vd_total_opt,
        'ke': ke_opt,
        'cl': cl_opt,
        't_half': t_half_opt,
        'optimization_success': result.success
    }
    # ===== INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt):
    """
    Enhanced clinical interpretation function for antimicrobial TDM
    
    This function can call the OpenAI API if configured, otherwise
    it will provide a simulated response with clinically relevant format.
    """
    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Call OpenAI API - uncomment and modify as needed
            response = openai.ChatCompletion.create(
                model="gpt-4",  # or your preferred model
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            llm_response = response.choices[0].message.content
            st.write("### Clinical Interpretation")
            st.write(llm_response)
            
            # Add a note about source
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")
            return
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
    
    # Extract the drug type from the prompt
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
    else:
        drug = "Antimicrobial"
    
    st.write("### Clinical Interpretation")
    
    # Simulated interpretation - create a more structured, clinically relevant format
    if drug == "Vancomycin":
        # Extract key values from the prompt
        trough_val = None
        auc_val = None
        target_range = None
        
        # Parse for measured or estimated trough
        if "Measured trough" in prompt:
            parts = prompt.split("Measured trough = ")
            if len(parts) > 1:
                trough_val = float(parts[1].split()[0])
        elif "Estimated trough" in prompt:
            parts = prompt.split("Estimated trough = ")
            if len(parts) > 1:
                trough_val = float(parts[1].split()[0])
        
        # Parse for AUC
        if "AUC24 = " in prompt:
            parts = prompt.split("AUC24 = ")
            if len(parts) > 1:
                auc_val = float(parts[1].split()[0])
        
        # Parse for target range
        if "Target trough range = " in prompt:
            parts = prompt.split("Target trough range = ")
            if len(parts) > 1:
                target_range = parts[1].strip()
        
        # Create a styled interpretation
        if trough_val is not None and auc_val is not None:
            # Create columns for Key Findings and Clinical Recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Key Findings:")
                # AUC analysis
                if auc_val < 400:
                    st.markdown("• **AUC:24 is subtherapeutic** at {:.1f} mg·hr/L (target: 400-600)".format(auc_val))
                    st.markdown("• This may lead to **treatment failure** and **antimicrobial resistance**")
                elif auc_val > 600:
                    st.markdown("• **AUC:24 is supratherapeutic** at {:.1f} mg·hr/L (target: 400-600)".format(auc_val))
                    st.markdown("• This increases the **risk of nephrotoxicity**")
                else:
                    st.markdown("• **AUC:24 is therapeutic** at {:.1f} mg·hr/L (target: 400-600)".format(auc_val))
                
                # Trough analysis
                if trough_val < 10:
                    st.markdown("• **Trough is subtherapeutic** at {:.1f} mg/L (target: {})".format(trough_val, target_range))
                elif trough_val > 20:
                    st.markdown("• **Trough is significantly elevated** at {:.1f} mg/L (target: {})".format(trough_val, target_range))
                    st.markdown("• **High risk of nephrotoxicity** with continued exposure to these levels")
                elif trough_val > 15:
                    st.markdown("• **Trough is mildly elevated** at {:.1f} mg/L (target: {})".format(trough_val, target_range))
                else:
                    st.markdown("• **Trough is therapeutic** at {:.1f} mg/L (target: {})".format(trough_val, target_range))
            
            with col2:
                st.markdown("#### Clinical Recommendations:")
                
                # Generate dosing recommendations based on AUC and trough
                if auc_val < 400 and trough_val < 10:
                    st.markdown("• **Increase dose** and consider shortening interval")
                    st.markdown("• **Consider loading dose** of 25-30 mg/kg if severe infection")
                    st.markdown("• Recheck levels after 3 doses (steady state)")
                elif auc_val > 600 and trough_val > 20:
                    st.markdown("• **Hold next dose** and reassess renal function")
                    st.markdown("• **Decrease dose by 25-30%** or extend interval")
                    st.markdown("• **Monitor renal function** closely")
                    st.markdown("• Recheck levels in 24-48 hours")
                elif auc_val > 600:
                    st.markdown("• **Decrease dose** by approximately 15-20%")
                    st.markdown("• Consider extending dosing interval")
                    st.markdown("• Recheck levels after 2-3 doses")
                elif auc_val < 400:
                    st.markdown("• **Increase dose** by approximately 15-20%")
                    st.markdown("• Recheck levels after 3 doses")
                else:
                    st.markdown("• **Continue current regimen**")
                    st.markdown("• Reassess need for continued TDM based on clinical response")
                    
                # Add special considerations
                if "CrCl" in prompt:
                    crcl_parts = prompt.split("CrCl = ")
                    if len(crcl_parts) > 1:
                        crcl_val = float(crcl_parts[1].split()[0])
                        if crcl_val < 30:
                            st.markdown("• **Caution:** Severe renal impairment. Consider alternative agents.")
                        elif crcl_val < 60:
                            st.markdown("• **Note:** Moderate renal impairment. Monitor renal function daily.")
            
            # Add overall assessment
            st.markdown("#### Clinical Context:")
            if "weight" in prompt.lower():
                weight_parts = prompt.split("Weight = ")
                if len(weight_parts) > 1:
                    weight_val = float(weight_parts[1].split()[0])
                    if weight_val > 120:
                        st.markdown("• Patient is obese. Consider using adjusted body weight for dosing calculations.")
            
            # Add educational information
            with st.expander("Educational Notes"):
                st.markdown("""
                * **Target AUC:24** for vancomycin is 400-600 mg·hr/L for most infections
                * **Target trough** depends on infection severity:
                    * 10-15 mg/L for uncomplicated infections
                    * 15-20 mg/L for severe infections (endocarditis, osteomyelitis, meningitis)
                * **Nephrotoxicity risk** increases significantly with troughs > 20 mg/L
                * **Steady state** is typically achieved after 3-4 half-lives (~24-36 hours in normal renal function)
                """)
    
    elif drug == "Aminoglycoside":
        # Create a structured interpretation for aminoglycosides
        st.markdown("#### Key Findings:")
        if "Cmax" in prompt and "Cmin" in prompt:
            cmax_parts = prompt.split("Cmax: ")
            cmin_parts = prompt.split("Cmin: ")
            if len(cmax_parts) > 1 and len(cmin_parts) > 1:
                cmax_val = float(cmax_parts[1].split(",")[0])
                cmin_val = float(cmin_parts[1].split(".")[0] + "." + cmin_parts[1].split(".")[1][0])
                
                drug_name = "aminoglycoside"
                if "Gentamicin" in prompt:
                    drug_name = "gentamicin"
                    peak_target = "5-10 mg/L"
                    trough_target = "<2 mg/L"
                elif "Amikacin" in prompt:
                    drug_name = "amikacin"
                    peak_target = "20-30 mg/L"
                    trough_target = "<10 mg/L"
                
                st.markdown(f"Analysis of {drug_name} levels:")
                st.markdown(f"• Peak: {cmax_val:.1f} mg/L (target: {peak_target})")
                st.markdown(f"• Trough: {cmin_val:.2f} mg/L (target: {trough_target})")
                
                st.markdown("#### Clinical Recommendations:")
                if drug_name == "gentamicin":
                    if cmax_val < 5:
                        st.markdown("• **Increase dose** to achieve therapeutic peak levels")
                    elif cmax_val > 10:
                        st.markdown("• **Consider decreasing dose** to avoid toxicity")
                    
                    if cmin_val > 2:
                        st.markdown("• **Extend dosing interval** to allow for adequate clearance")
                        st.markdown("• **Monitor renal function** for signs of nephrotoxicity")
                
                with st.expander("Educational Notes"):
                    st.markdown("""
                    * Aminoglycosides demonstrate **concentration-dependent killing**
                    * Higher peaks correlate with improved efficacy
                    * Elevated troughs correlate with increased toxicity risk
                    * Consider extended-interval dosing for most patients
                    """)
    
    # Add the raw prompt at the bottom for debugging
    with st.expander("Raw Analysis Data", expanded=False):
        st.code(prompt)
        
    # Add note about simulated response
    st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")
    # ===== SIDEBAR: NAVIGATION AND PATIENT INFO =====
def setup_sidebar_and_navigation():
    st.sidebar.title("📊 Navigation")
    # Sidebar radio for selecting the module – make sure the labels exactly match with your conditions below.
    page = st.sidebar.radio("Select Module", [
        "Aminoglycoside: Initial Dose",
        "Aminoglycoside: Conventional Dosing (C1/C2)",
        "Vancomycin AUC-based Dosing"
    ])

    st.sidebar.title("🩺 Patient Demographics")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age (years)", min_value=0, value=65)
    height = st.sidebar.number_input("Height (cm)", min_value=50, value=165)
    weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=70.0)
    serum_cr = st.sidebar.number_input("Serum Creatinine (µmol/L)", min_value=10.0, value=90.0)

    # Calculate Cockcroft-Gault Creatinine Clearance
    with st.sidebar.expander("Creatinine Clearance (Cockcroft-Gault)", expanded=True):
        # Calculate creatinine clearance using Cockcroft-Gault formula
        crcl = ((140 - age) * weight * (1.23 if gender == "Male" else 1.04)) / serum_cr
        st.success(f"CrCl: {crcl:.1f} mL/min")
        
        # Display renal function category
        if crcl >= 90:
            renal_function = "Normal"
        elif crcl >= 60:
            renal_function = "Mild Impairment"
        elif crcl >= 30:
            renal_function = "Moderate Impairment"
        elif crcl >= 15:
            renal_function = "Severe Impairment"
        else:
            renal_function = "Kidney Failure"
        
        st.info(f"Renal Function: {renal_function}")

    st.sidebar.title("🩺 Clinical Information")
    clinical_diagnosis = st.sidebar.text_input("Diagnosis")
    current_dose_regimen = st.sidebar.text_area("Current Dosing Regimen", value="1g IV q12h")
    notes = st.sidebar.text_area("Other Clinical Notes", value="No known allergies.")
    clinical_summary = f"Diagnosis: {clinical_diagnosis}\nRenal function: {renal_function} (CrCl: {crcl:.1f} mL/min)\nCurrent regimen: {current_dose_regimen}\nNotes: {notes}"
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Antimicrobial TDM App v1.0**
    
    Developed for therapeutic drug monitoring of antimicrobials.
    
    This app provides population PK estimates, AUC calculations, and dosing recommendations
    for vancomycin and aminoglycosides based on current best practices.
    
    **Disclaimer:** This tool is designed to assist clinical decision making but does not replace
    professional judgment. Always consult with a clinical pharmacist for complex cases.
    """)

    # Return all the data entered in the sidebar
    return {
        'page': page,
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
        'clinical_summary': clinical_summary
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
    
    # We'll use demographic data already entered in the sidebar
    drug = st.selectbox("Drug", ["Gentamicin", "Amikacin"])
    
    # Add dosing regimen selection
    regimen = st.selectbox("Therapeutic Goal", ["MDD", "SDD", "Synergy", "Hemodialysis", "Neonates"])
    
    # Set default target ranges based on regimen and drug
    if drug == "Gentamicin":
        if regimen == "MDD":
            default_peak = 10.0
            default_trough = 1.0
            st.info("Target: Peak 5-10 mg/L, Trough <2 mg/L")
        elif regimen == "SDD":
            default_peak = 20.0
            default_trough = 0.5
            st.info("Target: Peak 10-30 mg/L, Trough <1 mg/L")
        elif regimen == "Synergy":
            default_peak = 4.0
            default_trough = 0.5
            st.info("Target: Peak 3-5 mg/L, Trough <1 mg/L")
        elif regimen == "Hemodialysis":
            default_peak = 8.0
            default_trough = 1.0
            st.info("Target: Peak monitoring not necessary, Trough <2 mg/L")
        elif regimen == "Neonates":
            default_peak = 8.0
            default_trough = 0.5
            st.info("Target: Peak 5-12 mg/L, Trough <1 mg/L")
    else:  # Amikacin
        if regimen == "MDD":
            default_peak = 25.0
            default_trough = 5.0
            st.info("Target: Peak 20-30 mg/L, Trough <10 mg/L")
        elif regimen == "SDD":
            default_peak = 60.0
            default_trough = 0.5
            st.info("Target: Peak ~60 mg/L, Trough <1 mg/L")
        elif regimen == "Synergy":
            st.warning("Specific targets for Amikacin used in synergy are not established.")
            default_peak = 25.0
            default_trough = 5.0
        elif regimen == "Hemodialysis":
            default_peak = 25.0
            default_trough = 5.0
            st.info("Target: Peak monitoring not necessary, Trough <10 mg/L")
        elif regimen == "Neonates":
            default_peak = 25.0
            default_trough = 2.5
            st.info("Target: Peak 20-30 mg/L, Trough <5 mg/L")
    
    # For SDD regimens, add MIC input and adjust peak target
    if regimen == "SDD":
        st.markdown("*Note: Target peaks may be individualized based on MIC to achieve peak:MIC ratio of 10:1*")
        mic = st.number_input("MIC (mg/L)", min_value=0.0, value=1.0, step=0.25)
        recommended_peak = mic * 10
        if recommended_peak > default_peak:
            default_peak = recommended_peak
        st.info(f"For MIC of {mic} mg/L, recommended peak is ≥{recommended_peak} mg/L")
    
    # Allow user to override target values if needed
    col1, col2 = st.columns(2)
    with col1:
        target_cmax = st.number_input("Target Cmax (mg/L)", value=default_peak)
    with col2:
        target_cmin = st.number_input("Target Cmin (mg/L)", value=default_trough)
    
    # Special case for patients on hemodialysis
    is_hd = regimen == "Hemodialysis"
    if is_hd:
        st.info("For hemodialysis patients, doses are typically administered post-dialysis. Adjust interval based on dialysis schedule.")
    
    # Default tau based on regimen
    default_tau = 24 if regimen == "SDD" else 8 if regimen == "MDD" else 12
    tau = st.number_input("Dosing Interval (hr)", value=default_tau)

    # Adjust calculations for neonates
    if regimen == "Neonates":
        st.warning("This calculation uses adult PK parameters. Consult a pediatric clinical pharmacist for neonatal dosing.")
        
    # Calculate IBW and dosing weight
    ibw = 50 + 0.9 * (height - 152) if gender == "Male" else 45.5 + 0.9 * (height - 152)
    abw_ibw_ratio = weight / ibw
    
    if abw_ibw_ratio > 1.2:
        dosing_weight = ibw + 0.4 * (weight - ibw)
        weight_used = "Adjusted Body Weight"
    elif abw_ibw_ratio > 0.9:
        dosing_weight = weight
        weight_used = "Actual Body Weight"
    elif abw_ibw_ratio > 0.75:
        dosing_weight = ibw
        weight_used = "Ideal Body Weight"
    else:
        dosing_weight = weight * 1.13
        weight_used = "LBW x 1.13"

    # Adjust Vd based on patient factors
    base_vd = 0.3 if drug == "Amikacin" else 0.26
    vd_adjustment = 1.0  # default no adjustment
    
    # Volume adjustments for special cases
    if "ascites" in notes.lower() or "edema" in notes.lower() or "fluid overload" in notes.lower():
        vd_adjustment = 1.1  # 10% increase
        st.info("Vd adjusted for possible fluid overload based on clinical notes.")
    
    if "septic" in notes.lower() or "sepsis" in notes.lower():
        vd_adjustment = 1.15  # 15% increase
        st.info("Vd adjusted for possible sepsis based on clinical notes.")
    
    if "burn" in notes.lower():
        vd_adjustment = 1.2  # 20% increase
        st.info("Vd adjusted for possible burn injury based on clinical notes.")
    
    vd = base_vd * dosing_weight * vd_adjustment
    
    # Calculate clearance based on creatinine clearance
    clamg = (crcl * 60) / 1000
    ke = clamg / vd
    t_half = 0.693 / ke
    
    # Calculate dose
    dose = target_cmax * vd * (1 - np.exp(-ke * tau))
    expected_cmax = dose / (vd * (1 - np.exp(-ke * tau)))
    expected_cmin = expected_cmax * np.exp(-ke * tau)

    # Display results
    st.markdown(f"**IBW:** {ibw:.2f} kg  \n**Dosing Weight ({weight_used}):** {dosing_weight:.2f} kg  \n**CrCl:** {crcl:.2f} mL/min")
    
    # Round the dose to a practical value
    practical_dose = round(dose / 10) * 10  # Round to nearest 10mg
    if practical_dose < 100:
        practical_dose = round(dose / 5) * 5  # For low doses, round to nearest 5mg
    
    st.success(f"Recommended Initial Dose: **{practical_dose:.0f} mg** every **{tau:.0f}** hours")
    st.info(f"Expected Cmax: **{expected_cmax:.2f} mg/L**, Expected Cmin: **{expected_cmin:.2f} mg/L**")
    
    # Additional suggestions for loading dose
    if regimen == "SDD" or (is_hd and expected_cmax < target_cmax * 0.9):
        loading_dose = target_cmax * vd
        st.warning(f"Consider loading dose of **{round(loading_dose/10)*10:.0f} mg** to rapidly achieve target peak concentration.")

    suggest_adjustment(expected_cmax, target_cmax * 0.9, target_cmax * 1.1, label="Expected Cmax")
    
    # For trough target that's a "less than" value
    if expected_cmin > target_cmin:
        st.warning(f"⚠️ Expected Cmin ({expected_cmin:.2f} mg/L) is high. Target is <{target_cmin} mg/L. Consider lengthening the interval to a practical regimen ({practical_intervals}).")
    else:
        st.success(f"✅ Expected Cmin ({expected_cmin:.2f} mg/L) is below target of {target_cmin} mg/L.")

    # Add visualization option
    if st.checkbox("Show concentration-time curve"):
        chart = plot_concentration_time_curve(
            peak=expected_cmax, 
            trough=expected_cmin,
            ke=ke,
            tau=tau
        )
        st.altair_chart(chart, use_container_width=True)

    if st.button("🧠 Interpret with LLM"):
        prompt = (f"Aminoglycoside Initial Dose: Patient: {age} y/o {gender.lower()}, {height} cm, {weight} kg, SCr: {patient_data['serum_cr']} µmol/L. "
                  f"Drug: {drug}, Regimen: {regimen}, Target Cmax: {target_cmax}, Target Cmin: {target_cmin}, Interval: {tau} hr. "
                  f"Calculated: Weight {dosing_weight:.2f} kg ({weight_used}), Vd {vd:.2f} L, CrCl {crcl:.2f} mL/min, "
                  f"Ke {ke:.3f} hr⁻¹, t1/2 {t_half:.2f} hr, Dose {practical_dose:.0f} mg. "
                  f"Expected Cmax {expected_cmax:.2f} mg/L, Expected Cmin {expected_cmin:.2f} mg/L.")
        interpret_with_llm(prompt)
        # ===== MODULE 2: Aminoglycoside Conventional Dosing (C1/C2) =====
def aminoglycoside_conventional_dosing(patient_data):
    st.title("📊 Aminoglycoside Adjustment using C1/C2")
    
    drug = st.selectbox("Select Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Therapeutic Goal", ["MDD", "SDD", "Synergy", "Hemodialysis", "Neonates"])

    # Set target ranges based on chosen regimen and drug
    if drug == "Gentamicin":
        if regimen == "MDD":
            target_peak = (5, 10)
            target_trough = (0, 2)
        elif regimen == "SDD":
            target_peak = (10, 30)
            target_trough = (0, 1)
        elif regimen == "Synergy":
            target_peak = (3, 5)
            target_trough = (0, 1)
        elif regimen == "Hemodialysis":
            target_peak = (5, 10)  # Display "Not necessary" in UI
            target_trough = (0, 2)
        elif regimen == "Neonates":
            target_peak = (5, 12)
            target_trough = (0, 1)
    else:  # Amikacin
        if regimen == "MDD":
            target_peak = (20, 30)
            target_trough = (0, 10)
        elif regimen == "SDD":
            target_peak = (60, 60)
            target_trough = (0, 1)
        elif regimen == "Synergy":
            # Show N/A in UI
            target_peak = (0, 0)
            target_trough = (0, 0)
        elif regimen == "Hemodialysis":
            target_peak = (20, 30)  # Display "Not necessary" in UI
            target_trough = (0, 10)
        elif regimen == "Neonates":
            target_peak = (20, 30)
            target_trough = (0, 5)

    # Display target ranges with special cases
    st.markdown("### Target Concentration Ranges:")
    col1, col2 = st.columns(2)
    with col1:
        if regimen == "Synergy" and drug == "Amikacin":
            st.markdown("**Peak Target:** N/A")
        elif regimen == "Hemodialysis":
            st.markdown("**Peak Target:** Not necessary")
        else:
            st.markdown(f"**Peak Target:** {target_peak[0]} - {target_peak[1]} mg/L")
    
    with col2:
        if regimen == "Synergy" and drug == "Amikacin":
            st.markdown("**Trough Target:** N/A")
        else:
            if target_trough[1] == 1:
                st.markdown(f"**Trough Target:** <{target_trough[1]} mg/L")
            elif target_trough[1] == 2:
                st.markdown(f"**Trough Target:** <{target_trough[1]} mg/L")
            elif target_trough[1] == 5:
                st.markdown(f"**Trough Target:** <{target_trough[1]} mg/L")
            elif target_trough[1] == 10:
                st.markdown(f"**Trough Target:** <{target_trough[1]} mg/L")
            else:
                st.markdown(f"**Trough Target:** {target_trough[0]} - {target_trough[1]} mg/L")
    
    # Add MIC input for SDD regimens with note about peak:MIC ratio
    if regimen == "SDD":
        st.markdown("*Note: Target peaks may be individualized based on MIC to achieve peak:MIC ratio of 10:1*")
        mic = st.number_input("MIC (mg/L)", min_value=0.0, value=1.0, step=0.25)
        recommended_peak = mic * 10
        st.info(f"For MIC of {mic} mg/L, recommended peak is ≥{recommended_peak} mg/L")
    
    dose = st.number_input("Last Dose (mg)", min_value=0.0)
    c1 = st.number_input("Pre-dose Level (C1, mg/L)", min_value=0.0, value=1.0)
    c2 = st.number_input("Post-dose Level (C2, mg/L)", min_value=0.0, value=8.0)
    tau = st.number_input("Dosing Interval (hr)", value=8.0)
    t1 = st.number_input("C1 Sample Time After Dose (hr)", value=0.0)
    t2 = st.number_input("C2 Sample Time After Dose (hr)", value=1.0)
    t_post = st.number_input("Post-infusion Delay (hr)", value=0.5)

    try:
        if c1 > 0 and c2 > 0:
            ke = (math.log(c2) - math.log(c1)) / (tau - (t2 - t1))
            t_half = 0.693 / ke
            cmax = c2 * np.exp(ke * t_post)
            cmin = cmax * np.exp(-ke * tau)
            vd = dose / (cmax * (1 - np.exp(-ke * tau)))
            new_dose = cmax * vd * (1 - np.exp(-ke * tau))

            st.markdown(f"**Ke:** {ke:.3f} hr⁻¹  \n**Half-life:** {t_half:.2f} hr  \n**Cmax:** {cmax:.2f} mg/L, **Cmin:** {cmin:.2f} mg/L  \n**Vd:** {vd:.2f} L")
            st.success(f"Recommended New Dose: **{new_dose:.0f} mg**")
            
            # Special handling for regimens with special targets
            if regimen == "Hemodialysis":
                if cmin > target_trough[1]:
                    st.warning(f"⚠️ Trough is high ({cmin:.2f} mg/L). Consider lengthening interval or reducing dose.")
                else:
                    st.success(f"✅ Trough is acceptable ({cmin:.2f} mg/L).")
                st.info("Note: Peak monitoring is typically not necessary for hemodialysis patients.")
            elif regimen == "Synergy" and drug == "Amikacin":
                st.info("Note: Specific targets for Amikacin used in synergy are not established.")
            else:
                # Normal suggest_adjustment for other regimens
                if regimen == "SDD":
                    # For SDD, compare peak with MIC-based target
                    if cmax < recommended_peak:
                        st.warning(f"⚠️ Cmax ({cmax:.2f} mg/L) is below the recommended peak ({recommended_peak} mg/L) for the given MIC.")
                    else:
                        st.success(f"✅ Cmax ({cmax:.2f} mg/L) is adequate for the given MIC.")
                else:
                    suggest_adjustment(cmax, target_peak[0], target_peak[1], label="Cmax")
                
                # Handle trough targets that are "less than" values
                if target_trough[1] in [1, 2, 5, 10]:
                    if cmin > target_trough[1]:
                        st.warning(f"⚠️ Cmin is high ({cmin:.2f} mg/L). Target is <{target_trough[1]} mg/L.")
                    else:
                        st.success(f"✅ Cmin is acceptable ({cmin:.2f} mg/L).")
                else:
                    suggest_adjustment(cmin, target_trough[0], target_trough[1], label="Cmin")
            
            # Add visualization option
            if st.checkbox("Show concentration-time curve"):
                chart = plot_concentration_time_curve(
                    peak=cmax, 
                    trough=cmin,
                    ke=ke,
                    tau=tau
                )
                st.altair_chart(chart, use_container_width=True)

            if st.button("🧠 Interpret with LLM"):
                prompt = (f"Aminoglycoside TDM result: Drug: {drug}, Regimen: {regimen}, Dose: {dose} mg, "
                          f"C1: {c1} mg/L, C2: {c2} mg/L, Interval: {tau} hr. "
                          f"Ke: {ke:.3f}, t1/2: {t_half:.2f}, Vd: {vd:.2f}, Cmax: {cmax:.2f}, Cmin: {cmin:.2f}. "
                          f"Suggested new dose: {new_dose:.0f} mg.")
                if regimen == "SDD":
                    prompt += f" MIC: {mic} mg/L, Target peak:MIC ratio: 10:1"
                interpret_with_llm(prompt)
        else:
            st.error("❌ C1 and C2 must be greater than 0 to perform calculations.")
    except Exception as e:
        st.error(f"Calculation error: {e}")
     # ===== MODULE 3: Vancomycin AUC-based Dosing =====
def vancomycin_auc_dosing(patient_data):
    st.title("🧪 Vancomycin AUC-Based Dosing")
    
    # Unpack patient data for easier access
    weight = patient_data['weight']
    crcl = patient_data['crcl']
    gender = patient_data['gender']
    age = patient_data['age']
    
    method = st.radio("Select Method", ["Trough only", "Peak and Trough"], horizontal=True)
    
    # Global trough target selection (shown regardless of method)
    target_trough_strategy = st.selectbox(
        "Select Target Strategy", 
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"]
    )
    
    # Set targets based on selected strategy
    if "Empirical" in target_trough_strategy:
        target_cmin = (10, 15)
        target_peak = (20, 30) # Empirical peak targets
    else:
        target_cmin = (15, 20)
        target_peak = (30, 40) # Definitive peak targets
    
    st.markdown(f"**Target Trough Range:** {target_cmin[0]} - {target_cmin[1]} mg/L")
    
    # Only show peak targets for Peak and Trough method
    if method == "Peak and Trough":
        st.markdown(f"**Target Peak Range:** {target_peak[0]} - {target_peak[1]} mg/L")
    
    # Already have weight from sidebar
    
    st.info(f"Practical dosing intervals include: {practical_intervals}.")

    if method == "Trough only":
        current_dose = st.number_input("Current Total Daily Dose (mg)", value=2000)
        # Use CrCl from sidebar calculations
        tau = st.number_input("Dosing Interval (hr)", value=12.0)
        
        # Add measured trough input field
        measured_trough = st.number_input("Measured Trough Level (mg/L)", min_value=0.0, value=0.0)
        has_measured_trough = measured_trough > 0
        
        ke = 0.0044 + 0.00083 * crcl
        vd = 0.7 * weight
        cl = ke * vd
        
        # If we have a measured trough, use it for calculations
        if has_measured_trough:
            # Using formula from the image: Cmax = Cmin + Dose(mg)/V(L)
            estimated_dose_per_interval = current_dose/(24/tau)
            estimated_peak = measured_trough + estimated_dose_per_interval / vd
            estimated_trough = measured_trough
            
            # Calculate Ke from measured trough if possible (formula b in "Only trough level available")
            # This is more accurate than the population estimate
            # ln(Cmax-ln(Cmin))/T
            ke_measured = math.log(estimated_peak / measured_trough) / tau
            if 0.001 < ke_measured < 0.3:  # Sanity check on calculated Ke
                ke = ke_measured
                st.info(f"Using Ke calculated from measured trough: {ke:.4f} hr⁻¹")
        else:
            # Use population estimates if no measured trough
            estimated_peak = current_dose / (vd * (1 - np.exp(-ke * tau)))
            estimated_trough = estimated_peak * np.exp(-ke * tau)
            st.info("Using population parameters (no measured trough provided)")
            
        # Calculate AUC24 using linear trapezoidal method from the image
        # AUC(inf) = t' × (Cmin+Cmax)/2
        # AUC(elim) = (Cmax-Cmin)/Ke  
        # AUC24 = (AUCinf + AUCelim) × (24/T)
        
        infusion_time = 1  # Assuming standard 1-hour infusion
        
        # AUC during infusion phase using trapezoidal method
        auc_inf = infusion_time * (0 + estimated_peak) / 2
        
        # AUC during elimination phase
        auc_elim = (estimated_peak - estimated_trough) / ke
        
        # Total AUC for one interval
        auc_tau = auc_inf + auc_elim
        
        # Scale to 24 hours
        auc24 = auc_tau * (24 / tau)
        
        st.info(f"AUC24: {auc24:.1f} mg·hr/L")
        
        if has_measured_trough:
            st.markdown(f"Measured Trough: **{measured_trough:.1f} mg/L**")
        else:
            st.markdown(f"Estimated Trough: **{estimated_trough:.1f} mg/L**")
        
        # Enhanced suggest_adjustment with practical dosing recommendations
        trough_to_check = measured_trough if has_measured_trough else estimated_trough
        
        if trough_to_check < target_cmin[0]:
            practical_options = []
            # Try shorter intervals
            for interval in [6, 8, 12]:
                if interval < tau:
                    adj_dose = (current_dose / tau) * interval
                    new_peak = adj_dose / (vd * (1 - np.exp(-ke * interval)))
                    new_trough = new_peak * np.exp(-ke * interval)
                    if target_cmin[0] <= new_trough <= target_cmin[1]:
                        practical_options.append(f"🔷 {adj_dose:.0f}mg q{interval}h")
            # Try increased dose at same interval
            adj_dose = current_dose * (target_cmin[0] / trough_to_check) * 1.1  # 10% buffer
            new_peak = adj_dose / (vd * (1 - np.exp(-ke * tau)))
            new_trough = new_peak * np.exp(-ke * tau)
            if target_cmin[0] <= new_trough <= target_cmin[1]:
                practical_options.append(f"🔷 {adj_dose:.0f}mg q{tau}h")
                
            if practical_options:
                st.warning(f"⚠️ Trough is low ({trough_to_check:.1f} mg/L). Consider these practical options:")
                for option in practical_options:
                    st.markdown(option)
            else:
                st.warning(f"⚠️ Trough is low ({trough_to_check:.1f} mg/L). Consider increasing dose or shortening interval.")
                
        elif trough_to_check > target_cmin[1]:
            practical_options = []
            # Try longer intervals
            for interval in [8, 12, 24]:
                if interval > tau:
                    adj_dose = (current_dose / tau) * interval
                    new_peak = adj_dose / (vd * (1 - np.exp(-ke * interval)))
                    new_trough = new_peak * np.exp(-ke * interval)
                    if target_cmin[0] <= new_trough <= target_cmin[1]:
                        practical_options.append(f"🔷 {adj_dose:.0f}mg q{interval}h")
            # Try decreased dose at same interval
            adj_dose = current_dose * (target_cmin[1] / trough_to_check) * 0.9  # 10% buffer
            new_peak = adj_dose / (vd * (1 - np.exp(-ke * tau)))
            new_trough = new_peak * np.exp(-ke * tau)
            if target_cmin[0] <= new_trough <= target_cmin[1]:
                practical_options.append(f"🔷 {adj_dose:.0f}mg q{tau}h")
                
            if practical_options:
                st.warning(f"⚠️ Trough is high ({trough_to_check:.1f} mg/L). Consider these practical options:")
                for option in practical_options:
                    st.markdown(option)
            else:
                st.warning(f"⚠️ Trough is high ({trough_to_check:.1f} mg/L). Consider decreasing dose or lengthening interval.")
        else:
            st.success(f"✅ Trough is within target range ({trough_to_check:.1f} mg/L).")
        
        # Add AUC check as well
        if 400 <= auc24 <= 600:
            st.success(f"✅ AUC24 is within target range (400-600 mg·hr/L)")
        elif auc24 < 400:
            st.warning(f"⚠️ AUC24 is low ({auc24:.1f} mg·hr/L). Consider increasing dose.")
        else:
            st.warning(f"⚠️ AUC24 is high ({auc24:.1f} mg·hr/L). Consider decreasing dose.")
            
        # Calculate new TDD based on the formula in the image
        if has_measured_trough:
            desired_auc = 500  # Target middle of the range 400-600
            new_tdd = current_dose * (desired_auc / auc24)
            st.success(f"Recommended new TDD based on measured trough: **{new_tdd:.0f} mg/day**")
        
        # Add visualization option
        if st.checkbox("Show concentration-time curve"):
            chart = plot_concentration_time_curve(
                peak=estimated_peak, 
                trough=trough_to_check,
                ke=ke,
                tau=tau
            )
            st.altair_chart(chart, use_container_width=True)
            
        if st.button("🧠 Interpret with LLM"):
            trough_info = f"Measured trough = {measured_trough} mg/L" if has_measured_trough else f"Estimated trough = {estimated_trough:.1f} mg/L"
            prompt = (
                f"Vancomycin (Trough only): Current dose = {current_dose} mg/day, CrCl = {crcl:.1f} mL/min, "
                f"Weight = {weight} kg, Dosing interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, Vd = {vd:.2f} L, "
                f"AUC24 = {auc24:.1f} mg·hr/L, {trough_info}, "
                f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L."
            )
            interpret_with_llm(prompt)
    
    else:  # Peak and Trough method
        peak = st.number_input("Measured Peak (mg/L)", min_value=0.0)
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0)
        current_dose = st.number_input("Current Dose (mg)", min_value=0.0, value=1000.0)
        tau = st.number_input("Dosing Interval (hr)", value=12.0)
        t_peak = st.number_input("Time of Peak Sample (hr)", value=1.0)
        t_trough = st.number_input("Time of Trough Sample (hr)", value=tau)
        
        try:
            ke = (math.log(peak) - math.log(trough)) / (t_trough - t_peak)
            t_half = 0.693 / ke
            
            # Calculate AUC using the linear-log trapezoidal method as per reference:
            # AUC(inf) = t' × (Cmin+Cmax)/2
            # AUC(elim) = (Cmax-Cmin)/Ke
            # AUC24 = (AUCinf + AUCelim) × (24/T)
            
            t_prime = t_trough - t_peak  # Time between samples
            
            # Infusion phase (trapezoidal)
            auc_inf = t_prime * (trough + peak) / 2
            
            # Elimination phase
            auc_elim = (peak - trough) / ke
            
            # Total AUC for one dosing interval
            auc_tau = auc_inf + auc_elim
            
            # Scale to 24 hours
            auc24 = auc_tau * (24 / tau)
            
            st.info(f"Ke: {ke:.4f} hr⁻¹ | t½: {t_half:.2f} hr")
            st.success(f"AUC24: {auc24:.1f} mg·hr/L")
            
            # Enhanced suggestions for trough
            if trough < target_cmin[0]:
                practical_options = []
                # Try shorter intervals
                for interval in [6, 8, 12]:
                    if interval < tau:
                        adj_dose = current_dose * (target_cmin[0] / trough) * (1 - np.exp(-ke * interval)) / (1 - np.exp(-ke * tau))
                        new_trough = adj_dose * np.exp(-ke * interval) / (1 - np.exp(-ke * interval))
                        if target_cmin[0] <= new_trough <= target_cmin[1]:
                            practical_options.append(f"🔷 {adj_dose:.0f}mg q{interval}h")
                # Try increased dose at same interval
                adj_dose = current_dose * (target_cmin[0] / trough) * 1.1  # 10% buffer
                if practical_options:
                    st.warning(f"⚠️ Trough is low ({trough:.1f} mg/L). Consider these practical options:")
                    for option in practical_options:
                        st.markdown(option)
                else:
                    st.warning(f"⚠️ Trough is low ({trough:.1f} mg/L). Consider increasing dose to {adj_dose:.0f}mg or shortening interval.")
            elif trough > target_cmin[1]:
                practical_options = []
                # Try longer intervals
                for interval in [8, 12, 24]:
                    if interval > tau:
                        adj_dose = current_dose * (target_cmin[1] / trough) * (1 - np.exp(-ke * interval)) / (1 - np.exp(-ke * tau))
                        new_trough = adj_dose * np.exp(-ke * interval) / (1 - np.exp(-ke * interval))
                        if target_cmin[0] <= new_trough <= target_cmin[1]:
                            practical_options.append(f"🔷 {adj_dose:.0f}mg q{interval}h")
                # Try decreased dose at same interval
                adj_dose = current_dose * (target_cmin[1] / trough) * 0.9  # 10% buffer
                if practical_options:
                    st.warning(f"⚠️ Trough is high ({trough:.1f} mg/L). Consider these practical options:")
                    for option in practical_options:
                        st.markdown(option)
                else:
                    st.warning(f"⚠️ Trough is high ({trough:.1f} mg/L). Consider decreasing dose to {adj_dose:.0f}mg or lengthening interval.")
            else:
                st.success(f"✅ Trough is within target range ({trough:.1f} mg/L).")
            
            # Enhanced suggestions for peak
            if peak < target_peak[0]:
                st.warning(f"⚠️ Peak is low ({peak:.1f} mg/L). Target: {target_peak[0]}-{target_peak[1]} mg/L")
            elif peak > target_peak[1]:
                st.warning(f"⚠️ Peak is high ({peak:.1f} mg/L). Target: {target_peak[0]}-{target_peak[1]} mg/L")
            else:
                st.success(f"✅ Peak is within target range ({peak:.1f} mg/L).")
            
            # AUC check
            if 400 <= auc24 <= 600:
                st.success(f"✅ AUC24 is within target range (400-600 mg·hr/L)")
            elif auc24 < 400:
                st.warning(f"⚠️ AUC24 is low ({auc24:.1f} mg·hr/L). Consider increasing dose.")
            else:
                st.warning(f"⚠️ AUC24 is high ({auc24:.1f} mg·hr/L). Consider decreasing dose.")
            
            vd = current_dose / (peak * (1 - np.exp(-ke * tau)))
            st.info(f"Estimated Vd: {vd:.2f} L")
            
            # Calculate new dose based on target ranges
            target_min_trough = target_cmin[0]
            new_dose = target_min_trough * vd * (1 - np.exp(-ke * tau)) / np.exp(-ke * tau)
            
            st.success(f"Recommended Base Dose: **{new_dose:.0f} mg**")
            
            # Provide practical dosing options
            practical_doses = []
            for interval in [6, 8, 12, 24]:
                adj_dose = target_min_trough * vd * (1 - np.exp(-ke * interval)) / np.exp(-ke * interval)
                practical_doses.append((interval, adj_dose))
            
            st.subheader("Practical Dosing Options:")
            for interval, dose in practical_doses:
                expected_trough = dose * np.exp(-ke * interval) / (1 - np.exp(-ke * interval))
                expected_auc = ((dose - expected_trough) / ke + expected_trough * interval) * (24 / interval)
                if target_cmin[0] <= expected_trough <= target_cmin[1] and 400 <= expected_auc <= 600:
                    st.success(f"✅ {dose:.0f}mg q{interval}h (Est. trough: {expected_trough:.1f}, AUC: {expected_auc:.1f})")
                else:
                    st.info(f"🔹 {dose:.0f}mg q{interval}h (Est. trough: {expected_trough:.1f}, AUC: {expected_auc:.1f})")
            
            # Add bayesian estimation option if available
            if BAYESIAN_AVAILABLE:
                add_bayesian_estimation_ui()
            
            # Add visualization option
            if st.checkbox("Show concentration-time curve"):
                chart = plot_concentration_time_curve(
                    peak=peak, 
                    trough=trough,
                    ke=ke,
                    tau=tau
                )
                st.altair_chart(chart, use_container_width=True)
            
            if st.button("🧠 Interpret with LLM"):
                prompt = (
                    f"Vancomycin (Peak and Trough): Measured peak = {peak} mg/L, trough = {trough} mg/L, "
                    f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                    f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                    f"Target peak range = {target_peak[0]}-{target_peak[1]} mg/L, Recommended base dose = {new_dose:.0f} mg."
                )
                interpret_with_llm(prompt)
        except Exception as e:
            st.error(f"Calculation error: {e}")
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
        st.error(f"Unknown page: {patient_data['page']}")

# Run the application
if __name__ == "__main__":
    main()
