import streamlit as st
import numpy as np
import math
import pandas as pd
import altair as alt
import base64
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

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
    # Check for OpenAI API key
    import openai
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError, ImportError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.
    
    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== PATIENT INFO SECTION =====
def display_patient_info_section():
    """Display and collect patient information"""
    st.header("Patient Information")
    
    # Create a 2x2 grid for patient info
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", help="Enter the patient's unique identifier")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        serum_cr = st.number_input("Serum Creatinine (μmol/L)", min_value=10, max_value=1000, value=80)
    
    with col2:
        ward = st.text_input("Ward/Unit", help="Enter the patient's current location")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=40, max_value=250, value=170)
        clinical_diagnosis = st.text_input("Clinical Diagnosis", "Sepsis")
    
    # Calculate Creatinine Clearance
    # Cockcroft-Gault equation
    scr_mg = serum_cr / 88.4  # Convert μmol/L to mg/dL
    if gender == "Male":
        crcl = ((140 - age) * weight) / (72 * scr_mg)
    else:
        crcl = ((140 - age) * weight * 0.85) / (72 * scr_mg)
    
    # Determine renal function category
    if crcl >= 90:
        renal_function = "Normal renal function"
    elif crcl >= 60:
        renal_function = "Mild renal impairment"
    elif crcl >= 30:
        renal_function = "Moderate renal impairment"
    elif crcl >= 15:
        renal_function = "Severe renal impairment"
    else:
        renal_function = "Renal failure"
    
    # Display calculated CrCl and renal function
    st.metric("Estimated CrCl", f"{crcl:.1f} mL/min", renal_function)
    
    # Current medication regimen
    current_dose_regimen = st.text_input("Current Dosing Regimen", "Vancomycin 1000mg q12h")
    
    st.info(f"Patient {patient_id} is in {ward} with {renal_function.lower()} (CrCl: {crcl:.1f} mL/min)")
    
    # Return patient data as a dictionary
    return {
        'patient_id': patient_id,
        'ward': ward,
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen
    }

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details, peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization
    
    Parameters:
    - drug_info: String with drug name
    - levels_data: List of level data
    - assessment: Assessment string
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - calculation_details: String with calculation details
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
            conc = trough + (peak - trough) * (t / infusion_time)
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
    if "Vancomycin" in drug_info:  # Vancomycin
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
    elif "Gentamicin" in drug_info:  # Gentamicin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [10], 'y2': [30]  # Peak range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [5], 'y2': [10]  # Peak range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [2]  # Trough range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    elif "Amikacin" in drug_info:  # Amikacin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [60], 'y2': [80]  # Peak range for amikacin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for amikacin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [20], 'y2': [30]  # Peak range for amikacin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [10]  # Trough range for amikacin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    else:  # Default or unknown drug
        target_peak_band = alt.Chart(pd.DataFrame({
            'y1': [peak*0.8], 'y2': [peak*1.2]  # Default peak range ±20%
        })).mark_rect(opacity=0.2, color='green').encode(
            y='y1', y2='y2'
        )
        target_trough_band = alt.Chart(pd.DataFrame({
            'y1': [trough*0.5], 'y2': [trough*1.5]  # Default trough range ±50%
        })).mark_rect(opacity=0.2, color='blue').encode(
            y='y1', y2='y2'
        )
    
    # Create the concentration-time curve
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)')
    )
    
    # Add markers for actual measured peak and trough
    markers = alt.Chart(pd.DataFrame({
        'Time (hr)': [t_peak, tau],
        'Concentration (mg/L)': [peak, trough],
        'Label': ['Peak', 'Trough']
    })).mark_point(size=100, filled=True).encode(
        x='Time (hr)',
        y='Concentration (mg/L)',
        color=alt.Color('Label', scale=alt.Scale(domain=['Peak', 'Trough'], range=['green', 'blue']))
    )
    
    # Add vertical lines for key time points
    infusion_end = alt.Chart(pd.DataFrame({'x': [infusion_time]})).mark_rule(
        strokeDash=[5, 5], color='gray'
    ).encode(x='x')
    
    next_dose = alt.Chart(pd.DataFrame({'x': [tau]})).mark_rule(
        strokeDash=[5, 5], color='red'
    ).encode(x='x')
    
    # Add text annotations for key time points
    annotations = alt.Chart(pd.DataFrame({
        'x': [infusion_time/2, tau],
        'y': [peak*1.1, trough*0.9],
        'text': ['Infusion', 'Next Dose']
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Calculate half-life and display it
    half_life = 0.693 / ke
    half_life_text = alt.Chart(pd.DataFrame({
        'x': [tau/2],
        'y': [peak*0.5],
        'text': [f"t½ = {half_life:.1f} hr"]
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Combine charts
    chart = alt.layer(
        target_peak_band, 
        target_trough_band, 
        line, 
        markers,
        infusion_end,
        next_dose,
        annotations,
        half_life_text
    ).properties(
        width=600,
        height=400,
        title=f'{drug_info} Concentration-Time Profile'
    )
    
    # Display detailed calculation steps in an expander
    with st.expander("View Calculation Details", expanded=False):
        st.markdown("### PK Parameter Calculations")
        st.markdown(f"""
        **Key Parameters:**
        - Peak concentration (Cmax): {peak:.2f} mg/L
        - Trough concentration (Cmin): {trough:.2f} mg/L
        - Elimination rate constant (Ke): {ke:.4f} hr⁻¹
        - Half-life (t½): {half_life:.2f} hr
        - Dosing interval (τ): {tau} hr
        
        **Detailed Calculations:**
        ```
        Ke = -ln(Cmin/Cmax)/(τ - tpeak)
        Ke = -ln({trough:.2f}/{peak:.2f})/({tau} - {t_peak})
        Ke = {ke:.4f} hr⁻¹
        
        t½ = 0.693/Ke
        t½ = 0.693/{ke:.4f}
        t½ = {half_life:.2f} hr
        ```
        
        **Assessment:**
        {assessment}
        
        **Dosing Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in dosing_recs]) + """
        
        **Monitoring Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in monitoring_recs]))
        
        if calculation_details:
            st.markdown("**Additional Calculation Information:**")
            st.markdown(calculation_details)
    
    return chart

# ===== PDF GENERATION FUNCTIONS =====
def create_recommendation_pdf(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Create a downloadable PDF with the clinical recommendations
    
    Parameters:
    - patient_data: Dictionary with patient information
    - drug_info: String with drug name and method
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - calculation_details: Optional string with calculation details
    - cautions: Optional list of caution strings
    
    Returns:
    - base64 encoded PDF for download
    """
    # Create an in-memory PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom styles
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading3'],
        spaceAfter=6,
        textColor=colors.navy
    )
    
    # Create the content
    content = []
    
    # Add report title
    content.append(Paragraph("Antimicrobial TDM Report", title_style))
    content.append(Spacer(1, 12))
    
    # Add date and time
    now = datetime.now()
    content.append(Paragraph(f"Report Generated: {now.strftime('%Y-%m-%d %H:%M')}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add patient information
    content.append(Paragraph("Patient Information", heading_style))
    
    # Create patient info table with ID and Ward
    patient_info = []
    
    # Add patient ID and ward row
    patient_info.append([
        Paragraph("<b>Patient ID:</b>", normal_style),
        Paragraph(f"{patient_data.get('patient_id', 'N/A')}", normal_style),
        Paragraph("<b>Ward:</b>", normal_style),
        Paragraph(f"{patient_data.get('ward', 'N/A')}", normal_style)
    ])
    
    # First row
    patient_info.append([
        Paragraph("<b>Age:</b>", normal_style),
        Paragraph(f"{patient_data.get('age', 'N/A')} years", normal_style),
        Paragraph("<b>Gender:</b>", normal_style),
        Paragraph(f"{patient_data.get('gender', 'N/A')}", normal_style)
    ])
    
    # Second row
    patient_info.append([
        Paragraph("<b>Weight:</b>", normal_style),
        Paragraph(f"{patient_data.get('weight', 'N/A')} kg", normal_style),
        Paragraph("<b>Height:</b>", normal_style),
        Paragraph(f"{patient_data.get('height', 'N/A')} cm", normal_style)
    ])
    
    # Third row
    patient_info.append([
        Paragraph("<b>Serum Creatinine:</b>", normal_style),
        Paragraph(f"{patient_data.get('serum_cr', 'N/A')} µmol/L", normal_style),
        Paragraph("<b>CrCl:</b>", normal_style),
        Paragraph(f"{patient_data.get('crcl', 'N/A'):.1f} mL/min", normal_style)
    ])
    
    # Fourth row with diagnosis spanning full width
    patient_info.append([
        Paragraph("<b>Diagnosis:</b>", normal_style),
        Paragraph(f"{patient_data.get('clinical_diagnosis', 'N/A')}", normal_style),
        Paragraph("<b>Renal Function:</b>", normal_style),
        Paragraph(f"{patient_data.get('renal_function', 'N/A')}", normal_style)
    ])
    
    # Fifth row with regimen spanning full width
    patient_info.append([
        Paragraph("<b>Current Regimen:</b>", normal_style),
        Paragraph(f"{patient_data.get('current_dose_regimen', 'N/A')}", normal_style),
        Paragraph("", normal_style),
        Paragraph("", normal_style)
    ])
    
    # Create the table
    patient_table = Table(patient_info, colWidths=[100, 150, 100, 150])
    patient_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    content.append(patient_table)
    content.append(Spacer(1, 12))
    
    # Add drug information
    content.append(Paragraph("Drug Information", heading_style))
    content.append(Paragraph(drug_info, normal_style))
    content.append(Spacer(1, 12))
    
    # Add clinical assessment
    content.append(Paragraph("Clinical Assessment", heading_style))
    
    # Add measured levels
    content.append(Paragraph("Measured Levels:", section_style))
    
    # Create levels table
    levels_table_data = [["Parameter", "Value", "Target Range", "Status"]]
    
    for name, value, target, status in levels_data:
        # Determine status text and color
        if status == "within":
            status_text = "Within Range"
            status_color = colors.green
        elif status == "below":
            status_text = "Below Range"
            status_color = colors.orange
        else:  # above
            status_text = "Above Range"
            status_color = colors.red
        
        levels_table_data.append([name, value, target, status_text])
    
    levels_table = Table(levels_table_data)
    levels_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    # Add status color to each row in the table
    for i, (_, _, _, status) in enumerate(levels_data, 1):
        if status == "within":
            color = colors.lightgreen
        elif status == "below":
            color = colors.lightyellow
        else:  # above
            color = colors.mistyrose
        
        levels_table.setStyle(TableStyle([
            ('BACKGROUND', (3, i), (3, i), color),
        ]))
    
    content.append(levels_table)
    content.append(Spacer(1, 8))
    
    # Add assessment
    content.append(Paragraph("Assessment:", section_style))
    content.append(Paragraph(f"Patient is {assessment.upper()}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add calculations section if provided
    if calculation_details:
        content.append(Paragraph("Calculation Details:", section_style))
        content.append(Paragraph(calculation_details, normal_style))
        content.append(Spacer(1, 12))
    
    # Add recommendations
    content.append(Paragraph("Recommendations", heading_style))
    
    # Add dosing recommendations
    content.append(Paragraph("Dosing:", section_style))
    for rec in dosing_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add monitoring recommendations
    content.append(Paragraph("Monitoring:", section_style))
    for rec in monitoring_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add cautions if any
    if cautions and len(cautions) > 0:
        content.append(Paragraph("Cautions:", section_style))
        for caution in cautions:
            content.append(Paragraph(f"• {caution}", normal_style))
    
    # Add disclaimer
    content.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey
    )
    content.append(Paragraph("Disclaimer: This report is generated by an automated system and is intended to assist clinical decision making. Always use professional judgment when implementing recommendations.", disclaimer_style))
    
    # Build the PDF
    doc.build(content)
    
    # Get the PDF value from the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    # Encode the PDF to base64
    pdf_base64 = base64.b64encode(pdf_value).decode()
    
    return pdf_base64

# Function to create a download link for the PDF
def get_pdf_download_link(pdf_base64, filename="clinical_recommendations.pdf"):
    """Create a download link for a base64 encoded PDF"""
    href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="{filename}">Download Clinical Recommendations PDF</a>'
    return href

# Updated function to display buttons for printing and downloading recommendations
def display_pdf_download_button(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Display buttons to print/save recommendations as a PDF and print a summary
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Print/Save Full Report"):
            # Generate the PDF
            pdf_base64 = create_recommendation_pdf(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs,
                calculation_details,
                cautions
            )
            
            # Create the download link
            download_link = get_pdf_download_link(pdf_base64)
            
            # Display the download link
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Preview message
            st.success("PDF generated successfully. Click the link above to download.")
    
    with col2:
        if st.button("🖨️ Print Clinical Summary"):
            # Create a simple text printout of the assessment and recommendations
            assessment_text = create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            
            # Display in a text area that can be easily copied
            st.text_area("Copy this text to print", assessment_text, height=300)
            st.success("Assessment text generated. Copy and paste into your preferred document.")

# Enhanced function to create a printable text assessment
def create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """Create a plain text printable assessment for easy copying to clinical notes"""
    now = datetime.now()
    
    # Header
    text = f"ANTIMICROBIAL TDM ASSESSMENT - {now.strftime('%Y-%m-%d %H:%M')}\n"
    text += "=" * 50 + "\n\n"
    
    # Patient information - Make sure to include ID and ward
    text += f"Patient ID: {patient_data.get('patient_id', 'N/A')}\n"
    text += f"Ward: {patient_data.get('ward', 'N/A')}\n"
    text += f"Age: {patient_data.get('age', 'N/A')} years  |  "
    text += f"Gender: {patient_data.get('gender', 'N/A')}  |  "
    text += f"Weight: {patient_data.get('weight', 'N/A')} kg\n"
    text += f"Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}\n"
    text += f"CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})\n"
    text += f"Current regimen: {patient_data.get('current_dose_regimen', 'N/A')}\n\n"
    
    # Measured levels
    text += "MEASURED LEVELS:\n"
    for name, value, target, status in levels_data:
        status_text = "✓" if status == "within" else "↓" if status == "below" else "↑"
        text += f"- {name}: {value} (Target: {target}) {status_text}\n"
    
    # Assessment
    text += f"\nASSESSMENT: Patient is {assessment.upper()}\n\n"
    
    # PK Parameters (if available from calculation details)
    try:
        if "Half-life" in calculation_details or "t½" in calculation_details:
            text += "PHARMACOKINETIC PARAMETERS:\n"
            # Extract PK parameters from calculation details
            import re
            ke_match = re.search(r'Ke[\s=:]+([0-9.]+)', calculation_details)
            t_half_match = re.search(r't½[\s=:]+([0-9.]+)', calculation_details)
            
            if ke_match:
                ke = float(ke_match.group(1))
                text += f"- Elimination rate constant (Ke): {ke:.4f} hr⁻¹\n"
            
            if t_half_match:
                t_half = float(t_half_match.group(1))
                text += f"- Half-life (t½): {t_half:.2f} hr\n"
            
            text += "\n"
    except:
        pass  # Skip if unable to extract PK parameters
    
    # Recommendations
    text += "DOSING RECOMMENDATIONS:\n"
    for rec in dosing_recs:
        text += f"- {rec}\n"
    
    text += "\nMONITORING RECOMMENDATIONS:\n"
    for rec in monitoring_recs:
        text += f"- {rec}\n"
    
    # Cautions
    if cautions and len(cautions) > 0:
        text += "\nCAUTIONS:\n"
        for caution in cautions:
            text += f"-
            # ===== STANDARDIZED INTERPRETATION GENERATOR =====
def generate_standardized_interpretation(prompt, drug):
    """
    Generate a standardized interpretation based on drug type and prompt content
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt)
    elif "Aminoglycoside" in drug or "Gentamicin" in drug or "Amikacin" in drug:
        return generate_aminoglycoside_interpretation(prompt)
    else:
        # For generic, we'll create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None, calculation_details=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM with improved recommendation formatting
    and PDF printing capability
    
    This function can call the OpenAI API if configured, otherwise
    it will provide a simulated response with a standardized, clinically relevant format.
    
    Parameters:
    - prompt: The clinical data prompt
    - patient_data: Optional dictionary with patient information for PDF generation
    - calculation_details: Optional string with calculation details for PDF
    """
    # Extract the drug type from the prompt
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
        if "Trough only" in prompt:
            method = "Trough-only method"
        else:
            method = "Peak and Trough method"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
        if "Initial Dose" in prompt:
            method = "Initial dosing"
        else:
            method = "Conventional (C1/C2) method"
    else:
        drug = "Antimicrobial"
        method = "Standard method"
    
    drug_info = f"{drug} ({method})"
    
    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Format your response with these exact sections:
            
            ## CLINICAL ASSESSMENT
            📊 **MEASURED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **# ===== VANCOMYCIN INTERPRETATION FUNCTION =====
def generate_vancomycin_interpretation(prompt):
    """
    Generate standardized vancomycin interpretation
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    # Extract key values from the prompt
    peak_val = None
    trough_val = None
    auc24 = None
    
    # Extract peak and trough values
    if "Peak" in prompt:
        parts = prompt.split("Peak")
        if len(parts) > 1:
            peak_parts = parts[1].split("mg/L")
            if peak_parts:
                try:
                    peak_str = peak_parts[0].replace("=", "").replace(":", "").strip()
                    peak_val = float(peak_str)
                except ValueError:
                    pass
    
    if "Trough" in prompt:
        parts = prompt.split("Trough")
        if len(parts) > 1:
            trough_parts = parts[1].split("mg/L")
            if trough_parts:
                try:
                    trough_str = trough_parts[0].replace("=", "").replace(":", "").strip()
                    trough_val = float(trough_str)
                except ValueError:
                    pass
    
    # Extract AUC if available
    if "AUC24" in prompt:
        parts = prompt.split("AUC24")
        if len(parts) > 1:
            auc_parts = parts[1].split("mg·hr/L")
            if auc_parts:
                try:
                    auc_str = auc_parts[0].replace("=", "").replace(":", "").strip()
                    auc24 = float(auc_str)
                except ValueError:
                    pass
    
    # Extract trough target range
    trough_target_min, trough_target_max = 10, 20  # Default range
    if "Target trough range" in prompt:
        parts = prompt.split("Target trough range")
        if len(parts) > 1:
            range_parts = parts[1].strip().split("mg/L")
            if range_parts:
                try:
                    range_str = range_parts[0].replace("=", "").replace(":", "").strip()
                    if "-" in range_str:
                        min_max = range_str.split("-")
                        trough_target_min = float(min_max[0])
                        trough_target_max = float(min_max[1])
                except ValueError:
                    pass
    
    # Determine if empiric or definitive therapy based on trough target
    if trough_target_max <= 15:
        regimen = "Empiric"
        trough_target = f"{trough_target_min}-{trough_target_max} mg/L"
    else:
        regimen = "Definitive"
        trough_target = f"{trough_target_min}-{trough_target_max} mg/L"
    
    # Set AUC target based on indication
    if regimen == "Empiric":
        auc_target = "400-600 mg·hr/L"
        auc_min, auc_max = 400, 600
    else:  # Definitive
        auc_target = "400-800 mg·hr/L"
        auc_min, auc_max = 400, 800
    
    # Define peak target range
    peak_target = "20-40 mg/L"  # Typical peak range
    peak_min, peak_max = 20, 40
    
    # Determine vancomycin status
    status = "assessment not available"
    
    # If using trough-only monitoring
    if trough_val is not None and peak_val is None and auc24 is None:
        if trough_val < trough_target_min:
            status = "subtherapeutic (low trough)"
        elif trough_val > trough_target_max:
            status = "potentially supratherapeutic (high trough)"
        else:
            status = "appropriately dosed (trough-based)"
    
    # If using peak and trough monitoring
    elif trough_val is not None and peak_val is not None:
        if peak_val < peak_min and trough_val < trough_target_min:
            status = "subtherapeutic (inadequate peak and trough)"
        elif peak_val < peak_min:
            status = "potential underdosing (low peak)"
        elif trough_val < trough_target_min:
            status = "subtherapeutic (inadequate trough)"
        elif trough_val > trough_target_max:
            status = "potentially supratherapeutic (high trough)"
        elif peak_val > peak_max:
            status = "potentially supratherapeutic (high peak)"
        elif peak_min <= peak_val <= peak_max and trough_target_min <= trough_val <= trough_target_max:
            status = "appropriately dosed"
        else:
            status = "requires adjustment"
    
    # If using AUC monitoring
    elif auc24 is not None:
        if auc24 < auc_min:
            status = "subtherapeutic (low AUC)"
        elif auc24 > auc_max:
            status = "potentially supratherapeutic (high AUC)"
        else:
            status = "appropriately dosed (AUC-based)"
    
    # Create levels data based on available measurements
    levels_data = []
    
    if peak_val is not None:
        if peak_val < peak_min:
            peak_status = "below"
        elif peak_val > peak_max:
            peak_status = "above"
        else:
            peak_status = "within"
        levels_data.append(("Peak", f"{peak_val:.1f} mg/L", peak_target, peak_status))
    
    if trough_val is not None:
        if trough_val < trough_target_min:
            trough_status = "below"
        elif trough_val > trough_target_max:
            trough_status = "above"
        else:
            trough_status = "within"
        levels_data.append(("Trough", f"{trough_val:.1f} mg/L", trough_target, trough_status))
    
    if auc24 is not None:
        if auc24 < auc_min:
            auc_status = "below"
        elif auc24 > auc_max:
            auc_status = "above"
        else:
            auc_status = "within"
        levels_data.append(("AUC24", f"{auc24:.1f} mg·hr/L", auc_target, auc_status))
    
    # Generate recommendations based on status
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    
    # Check if we have enough data to provide recommendations
    if not levels_data:
        return "Insufficient data to generate interpretation. At least one measurement (peak, trough, or AUC) is required."
    
    # Extract new dose if available
    new_dose = None
    if "Recommended base dose" in prompt:
        parts = prompt.split("Recommended base dose")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose_str = dose_parts[0].replace("=", "").replace(":", "").strip()
                    new_dose = float(dose_str)
                except ValueError:
                    pass
    
    # Format new dose
    rounded_new_dose = None
    if new_dose:
        # Round to nearest 250mg for vancomycin
        rounded_new_dose = round(new_dose / 250) * 250
    
    # Generate recommendations based on status
    if status == "subtherapeutic (low trough)" or status == "subtherapeutic (inadequate trough)" or status == "subtherapeutic (low AUC)":
        if rounded_new_dose:
            dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("INCREASE dose by 25-30%")
        dosing_recs.append("CONSIDER shortening dosing interval")
        
        monitoring_recs.append("RECHECK levels after 3-4 doses (at steady state)")
        monitoring_recs.append("MONITOR renal function regularly")
        
        cautions.append("Subtherapeutic levels may lead to treatment failure")
        cautions.append("Ensure adequate hydration when increasing doses")
    
    elif status == "potentially supratherapeutic (high trough)" or status == "potentially supratherapeutic (high AUC)":
        if rounded_new_dose:
            dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("DECREASE dose by 20-25%")
        dosing_recs.append("CONSIDER extending dosing interval")
        
        monitoring_recs.append("RECHECK levels after 3-4 doses")
        monitoring_recs.append("MONITOR renal function closely")
        monitoring_recs.append("ASSESS for signs of nephrotoxicity")
        
        cautions.append("Risk of nephrotoxicity with elevated trough levels")
        cautions.append("Consider patient-specific risk factors for toxicity")
    
    elif status == "subtherapeutic (inadequate peak and trough)" or status == "potential underdosing (low peak)":
        if rounded_new_dose:
            dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("INCREASE dose by 30-40%")
        
        monitoring_recs.append("RECHECK peak and trough levels after 3-4 doses")
        monitoring_recs.append("VERIFY correct timing of sample collection")
        
        cautions.append("Significantly subtherapeutic levels increase risk of treatment failure")
        cautions.append("Consider evaluating for altered pharmacokinetics")
    
    elif status == "potentially supratherapeutic (high peak)":
        if rounded_new_dose:
            dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("DECREASE dose by 15-20%")
        dosing_recs.append("EXTEND dosing interval if appropriate")
        
        monitoring_recs.append("RECHECK levels after adjustment")
        monitoring_recs.append("MONITOR renal function")
        
        cautions.append("Risk of nephrotoxicity with excessive dosing")
    
    elif "appropriately dosed" in status:
        dosing_recs.append("CONTINUE current dosing regimen")
        
        monitoring_recs.append("MONITOR renal function regularly")
        monitoring_recs.append("REASSESS levels if clinical status changes")
        
        cautions.append("Even with therapeutic levels, monitor for adverse effects")
    
    else:  # requires adjustment
        if rounded_new_dose:
            dosing_recs.append(f"ADJUST dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("ADJUST dosing based on clinical response and levels")
        
        monitoring_recs.append("RECHECK levels after adjustment")
        monitoring_recs.append("MONITOR renal function")
        
        cautions.append("Individualize therapy based on clinical response")
    
    # Add standard monitoring recommendations
    if "MONITOR renal function" not in " ".join(monitoring_recs):
        monitoring_recs.append("MONITOR renal function every 2-3 days")
    
    return levels_data, assessment, dosing_recs, monitoring_recs, cautions# ===== AMINOGLYCOSIDE INTERPRETATION FUNCTION =====
def generate_aminoglycoside_interpretation(prompt):
    """
    Generate standardized aminoglycoside interpretation
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    # Extract key values from the prompt
    drug_name = "aminoglycoside"
    peak_val = None
    trough_val = None
    
    if "Gentamicin" in prompt:
        drug_name = "gentamicin"
    elif "Amikacin" in prompt:
        drug_name = "amikacin"
    
    # Extract peak and trough values
    if "Cmax:" in prompt:
        parts = prompt.split("Cmax:")
        if len(parts) > 1:
            peak_parts = parts[1].split(",")
            if peak_parts:
                try:
                    peak_val = float(peak_parts[0])
                except ValueError:
                    pass
    elif "Expected Cmax" in prompt:
        parts = prompt.split("Expected Cmax")
        if len(parts) > 1:
            peak_parts = parts[1].split("mg/L")
            if peak_parts:
                try:
                    peak_val = float(peak_parts[0].replace(":", "").strip())
                except ValueError:
                    pass
    
    if "Cmin:" in prompt:
        parts = prompt.split("Cmin:")
        if len(parts) > 1:
            trough_parts = parts[1].split(",")
            if trough_parts:
                try:
                    trough_val = float(trough_parts[0])
                except ValueError:
                    pass
    elif "Expected Cmin" in prompt:
        parts = prompt.split("Expected Cmin")
        if len(parts) > 1:
            trough_parts = parts[1].split("mg/L")
            if trough_parts:
                try:
                    trough_val = float(trough_parts[0].replace(":", "").strip())
                except ValueError:
                    pass
    
    # Extract dose
    dose = None
    if "Dose:" in prompt:
        parts = prompt.split("Dose:")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose = float(dose_parts[0])
                except ValueError:
                    pass
    elif "Dose " in prompt:
        parts = prompt.split("Dose ")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose = float(dose_parts[0])
                except ValueError:
                    pass
    
    # Extract suggested new dose
    new_dose = None
    if "Suggested new dose:" in prompt:
        parts = prompt.split("Suggested new dose:")
        if len(parts) > 1:
            new_dose_parts = parts[1].split("mg")
            if new_dose_parts:
                try:
                    new_dose = float(new_dose_parts[0])
                except ValueError:
                    pass
    elif "Recommended" in prompt and "Dose" in prompt:
        parts = prompt.split("Recommended")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    # Extract the number from this string
                    import re
                    numbers = re.findall(r'\d+', dose_parts[0])
                    if numbers:
                        new_dose = float(numbers[0])
                except ValueError:
                    pass
    
    # Extract target values based on regimen mention
    regimen = None
    if "SDD" in prompt:
        regimen = "SDD"
    elif "Synergy" in prompt:
        regimen = "Synergy"
    elif "MDD" in prompt:
        regimen = "MDD"
    
    # Set target ranges based on drug
    if drug_name == "gentamicin":
        if regimen == "SDD":
            peak_target = "10-30 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 10, 30
            trough_max = 1
        elif regimen == "Synergy":
            peak_target = "3-5 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 3, 5
            trough_max = 1
        else:  # Default to MDD
            peak_target = "5-10 mg/L"
            trough_target = "<2 mg/L"
            peak_min, peak_max = 5, 10
            trough_max = 2
    elif drug_name == "amikacin":
        if regimen == "SDD":
            peak_target = "60-80 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 60, 80
            trough_max = 1
        else:  # Default to MDD
            peak_target = "20-30 mg/L"
            trough_target = "<10 mg/L"
            peak_min, peak_max = 20, 30
            trough_max = 10
    else:
        peak_target = "varies by drug"
        trough_target = "varies by drug"
        peak_min, peak_max = 0, 100
        trough_max = 10
    
    # Determine aminoglycoside status
    status = "assessment not available"
    if peak_val and trough_val:
        if peak_val < peak_min and trough_val > trough_max:
            status = "ineffective and potentially toxic"
        elif peak_val < peak_min:
            status = "subtherapeutic (inadequate peak)"
        elif trough_val > trough_max:
            status = "potentially toxic (elevated trough)"
        elif peak_min <= peak_val <= peak_max and trough_val <= trough_max:
            status = "appropriately dosed"
        elif peak_val > peak_max:
            status = "potentially toxic (elevated peak)"
        else:
            status = "requires adjustment"
    
    # Format new dose
    rounded_new_dose = None
    if new_dose:
        # Round to nearest 10mg for most aminoglycosides
        rounded_new_dose = round(new_dose / 10) * 10
    
    # Create interpretation using standardized format
    if peak_val is not None and trough_val is not None:
        # Determine peak status
        if peak_val < peak_min:
            peak_status = "below"
        elif peak_val > peak_max:
            peak_status = "above"
        else:
            peak_status = "within"
        
        # Determine trough status
        if trough_val > trough_max:
            trough_status = "above"
        else:
            trough_status = "within"
        
        # Prepare data for standardized format
        levels_data = [
            (f"Peak", f"{peak_val:.1f} mg/L", peak_target, peak_status),
            (f"Trough", f"{trough_val:.2f} mg/L", trough_target, trough_status)
        ]
        
        # Generate recommendations based on status
        dosing_recs = []
        monitoring_recs = []
        cautions = []
        
        if status == "ineffective and potentially toxic":
            dosing_recs.append("HOLD next dose")
            dosing_recs.append("REASSESS renal function before resuming")
            if rounded_new_dose:
                dosing_recs.append(f"DECREASE to {rounded_new_dose}mg when resumed")
            dosing_recs.append("EXTEND dosing interval significantly")
            
            monitoring_recs.append("CHECK renal function before resuming therapy")
            monitoring_recs.append("RECHECK levels 2 doses after resumption")
            monitoring_recs.append("MONITOR for signs of ototoxicity and nephrotoxicity")
            
            cautions.append("Risk of nephrotoxicity and ototoxicity with elevated trough levels")
            cautions.append("Ineffective therapy may lead to treatment failure")
            
        elif status == "subtherapeutic (inadequate peak)":
            if rounded_new_dose:
                dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
            else:
                dosing_recs.append("INCREASE dose by 25-50%")
            
            monitoring_recs.append("RECHECK levels after 2-3 doses")
            monitoring_recs.append("VERIFY correct timing of peak sample collection")
            
            cautions.append("Subtherapeutic levels may lead to treatment failure")
            cautions.append("Consider other factors affecting drug disposition")
            
        elif status == "potentially toxic (elevated trough)":
            dosing_recs.append("EXTEND dosing interval")
            if rounded_new_dose:
                dosing_recs.append(f"CONSIDER dose reduction to {rounded_new_dose}mg")
            
            monitoring_recs.append("MONITOR renal function closely")
            monitoring_recs.append("RECHECK levels before next dose")
            monitoring_recs.append("ASSESS for signs of ototoxicity and nephrotoxicity")
            
            cautions.append("Risk of nephrotoxicity and ototoxicity with elevated trough levels")
            cautions.append("Consider patient-specific risk factors for toxicity")
            
        elif status == "potentially toxic (elevated peak)":
            if rounded_new_dose:
                dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
            else:
                dosing_recs.append("DECREASE dose by 20-25%")
            
            monitoring_recs.append("RECHECK levels after 2-3 doses")
            monitoring_recs.append("VERIFY correct timing of peak sample collection")
            monitoring_recs.append("MONITOR for signs of ototoxicity")
            
            cautions.append("Risk of ototoxicity with significantly elevated peak levels")
            
        elif status == "appropriately dosed":
            dosing_recs.append("CONTINUE current dosing regimen")
            
            monitoring_recs.append("MONITOR renal function regularly")
            monitoring_recs.append("REASSESS levels if clinical status changes")
            monitoring_recs.append("CONSIDER extended interval dosing for longer therapy")
            
            cautions.append("Even with therapeutic levels, monitor for adverse effects")
            
        else:  # requires adjustment
            dosing_recs.append("ADJUST dosing based on clinical response")
            if rounded_new_dose:
                dosing_recs.append(f"CONSIDER dose of {rounded_new_dose}mg")
            
            monitoring_recs.append("RECHECK levels after adjustment")
            monitoring_recs.append("MONITOR renal function")
            
            cautions.append("Individualize therapy based on clinical response")
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions
    else:
        return "Insufficient data to generate interpretation. Both peak and trough levels are required."# ===== FORMAT_CLINICAL_RECOMMENDATIONS FUNCTION =====
def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy
    
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
    levels_md = "📊 **MEASURED LEVELS:**\n"
    for name, value, target, status in levels_data:
        icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴"
        levels_md += f"- {name}: {value} (Target: {target}) {icon}\n"
    
    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is {assessment.upper()}"
    
    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## DETAILED RECOMMENDATIONS

🔵 **DOSING RECOMMENDATIONS:**
"""
    for rec in dosing_recs:
        output += f"- {rec}\n"
    
    output += "\n🔵 **MONITORING RECOMMENDATIONS:**\n"
    for rec in monitoring_recs:
        output += f"- {rec}\n"
    
    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS & CONSIDERATIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"
    
    # Add a summary section for quick reference
    output += "\n## QUICK SUMMARY\n"
    output += "**Status:** " + assessment.upper() + "\n"
    
    # Summarize key recommendations
    if len(dosing_recs) > 0:
        output += f"**Key Dosing Action:** {dosing_recs[0]}\n"
    
    if len(monitoring_recs) > 0:
        output += f"**Key Monitoring Action:** {monitoring_recs[0]}\n"
        
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    output += f"\n*Generated on: {timestamp}*"
    
    return outputimport streamlit as st
import numpy as np
import math
import pandas as pd
import altair as alt
import base64
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

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
    # Check for OpenAI API key
    import openai
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError, ImportError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.
    
    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== PATIENT INFO SECTION =====
def display_patient_info_section():
    """Display and collect patient information"""
    st.header("Patient Information")
    
    # Create a 2x2 grid for patient info
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", help="Enter the patient's unique identifier")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        serum_cr = st.number_input("Serum Creatinine (μmol/L)", min_value=10, max_value=1000, value=80)
    
    with col2:
        ward = st.text_input("Ward/Unit", help="Enter the patient's current location")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=40, max_value=250, value=170)
        clinical_diagnosis = st.text_input("Clinical Diagnosis", "Sepsis")
    
    # Calculate Creatinine Clearance
    # Cockcroft-Gault equation
    scr_mg = serum_cr / 88.4  # Convert μmol/L to mg/dL
    if gender == "Male":
        crcl = ((140 - age) * weight) / (72 * scr_mg)
    else:
        crcl = ((140 - age) * weight * 0.85) / (72 * scr_mg)
    
    # Determine renal function category
    if crcl >= 90:
        renal_function = "Normal renal function"
    elif crcl >= 60:
        renal_function = "Mild renal impairment"
    elif crcl >= 30:
        renal_function = "Moderate renal impairment"
    elif crcl >= 15:
        renal_function = "Severe renal impairment"
    else:
        renal_function = "Renal failure"
    
    # Display calculated CrCl and renal function
    st.metric("Estimated CrCl", f"{crcl:.1f} mL/min", renal_function)
    
    # Current medication regimen
    current_dose_regimen = st.text_input("Current Dosing Regimen", "Vancomycin 1000mg q12h")
    
    st.info(f"Patient {patient_id} is in {ward} with {renal_function.lower()} (CrCl: {crcl:.1f} mL/min)")
    
    # Return patient data as a dictionary
    return {
        'patient_id': patient_id,
        'ward': ward,
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen
    }

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details, peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization
    
    Parameters:
    - drug_info: String with drug name
    - levels_data: List of level data
    - assessment: Assessment string
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - calculation_details: String with calculation details
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
            conc = trough + (peak - trough) * (t / infusion_time)
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
    if "Vancomycin" in drug_info:  # Vancomycin
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
    elif "Gentamicin" in drug_info:  # Gentamicin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [10], 'y2': [30]  # Peak range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [5], 'y2': [10]  # Peak range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [2]  # Trough range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    elif "Amikacin" in drug_info:  # Amikacin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [60], 'y2': [80]  # Peak range for amikacin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for amikacin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [20], 'y2': [30]  # Peak range for amikacin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [10]  # Trough range for amikacin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    else:  # Default or unknown drug
        target_peak_band = alt.Chart(pd.DataFrame({
            'y1': [peak*0.8], 'y2': [peak*1.2]  # Default peak range ±20%
        })).mark_rect(opacity=0.2, color='green').encode(
            y='y1', y2='y2'
        )
        target_trough_band = alt.Chart(pd.DataFrame({
            'y1': [trough*0.5], 'y2': [trough*1.5]  # Default trough range ±50%
        })).mark_rect(opacity=0.2, color='blue').encode(
            y='y1', y2='y2'
        )
    
    # Create the concentration-time curve
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)')
    )
    
    # Add markers for actual measured peak and trough
    markers = alt.Chart(pd.DataFrame({
        'Time (hr)': [t_peak, tau],
        'Concentration (mg/L)': [peak, trough],
        'Label': ['Peak', 'Trough']
    })).mark_point(size=100, filled=True).encode(
        x='Time (hr)',
        y='Concentration (mg/L)',
        color=alt.Color('Label', scale=alt.Scale(domain=['Peak', 'Trough'], range=['green', 'blue']))
    )
    
    # Add vertical lines for key time points
    infusion_end = alt.Chart(pd.DataFrame({'x': [infusion_time]})).mark_rule(
        strokeDash=[5, 5], color='gray'
    ).encode(x='x')
    
    next_dose = alt.Chart(pd.DataFrame({'x': [tau]})).mark_rule(
        strokeDash=[5, 5], color='red'
    ).encode(x='x')
    
    # Add text annotations for key time points
    annotations = alt.Chart(pd.DataFrame({
        'x': [infusion_time/2, tau],
        'y': [peak*1.1, trough*0.9],
        'text': ['Infusion', 'Next Dose']
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Calculate half-life and display it
    half_life = 0.693 / ke
    half_life_text = alt.Chart(pd.DataFrame({
        'x': [tau/2],
        'y': [peak*0.5],
        'text': [f"t½ = {half_life:.1f} hr"]
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Combine charts
    chart = alt.layer(
        target_peak_band, 
        target_trough_band, 
        line, 
        markers,
        infusion_end,
        next_dose,
        annotations,
        half_life_text
    ).properties(
        width=600,
        height=400,
        title=f'{drug_info} Concentration-Time Profile'
    )
    
    # Display detailed calculation steps in an expander
    with st.expander("View Calculation Details", expanded=False):
        st.markdown("### PK Parameter Calculations")
        st.markdown(f"""
        **Key Parameters:**
        - Peak concentration (Cmax): {peak:.2f} mg/L
        - Trough concentration (Cmin): {trough:.2f} mg/L
        - Elimination rate constant (Ke): {ke:.4f} hr⁻¹
        - Half-life (t½): {half_life:.2f} hr
        - Dosing interval (τ): {tau} hr
        
        **Detailed Calculations:**
        ```
        Ke = -ln(Cmin/Cmax)/(τ - tpeak)
        Ke = -ln({trough:.2f}/{peak:.2f})/({tau} - {t_peak})
        Ke = {ke:.4f} hr⁻¹
        
        t½ = 0.693/Ke
        t½ = 0.693/{ke:.4f}
        t½ = {half_life:.2f} hr
        ```
        
        **Assessment:**
        {assessment}
        
        **Dosing Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in dosing_recs]) + """
        
        **Monitoring Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in monitoring_recs]))
        
        if calculation_details:
            st.markdown("**Additional Calculation Information:**")
            st.markdown(calculation_details)
    
    return chart

# ===== PDF GENERATION FUNCTIONS =====
def create_recommendation_pdf(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Create a downloadable PDF with the clinical recommendations
    
    Parameters:
    - patient_data: Dictionary with patient information
    - drug_info: String with drug name and method
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - calculation_details: Optional string with calculation details
    - cautions: Optional list of caution strings
    
    Returns:
    - base64 encoded PDF for download
    """
    # Create an in-memory PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom styles
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading3'],
        spaceAfter=6,
        textColor=colors.navy
    )
    
    # Create the content
    content = []
    
    # Add report title
    content.append(Paragraph("Antimicrobial TDM Report", title_style))
    content.append(Spacer(1, 12))
    
    # Add date and time
    now = datetime.now()
    content.append(Paragraph(f"Report Generated: {now.strftime('%Y-%m-%d %H:%M')}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add patient information
    content.append(Paragraph("Patient Information", heading_style))
    
    # Create patient info table with ID and Ward
    patient_info = []
    
    # Add patient ID and ward row
    patient_info.append([
        Paragraph("<b>Patient ID:</b>", normal_style),
        Paragraph(f"{patient_data.get('patient_id', 'N/A')}", normal_style),
        Paragraph("<b>Ward:</b>", normal_style),
        Paragraph(f"{patient_data.get('ward', 'N/A')}", normal_style)
    ])
    
    # First row
    patient_info.append([
        Paragraph("<b>Age:</b>", normal_style),
        Paragraph(f"{patient_data.get('age', 'N/A')} years", normal_style),
        Paragraph("<b>Gender:</b>", normal_style),
        Paragraph(f"{patient_data.get('gender', 'N/A')}", normal_style)
    ])
    
    # Second row
    patient_info.append([
        Paragraph("<b>Weight:</b>", normal_style),
        Paragraph(f"{patient_data.get('weight', 'N/A')} kg", normal_style),
        Paragraph("<b>Height:</b>", normal_style),
        Paragraph(f"{patient_data.get('height', 'N/A')} cm", normal_style)
    ])
    
    # Third row
    patient_info.append([
        Paragraph("<b>Serum Creatinine:</b>", normal_style),
        Paragraph(f"{patient_data.get('serum_cr', 'N/A')} µmol/L", normal_style),
        Paragraph("<b>CrCl:</b>", normal_style),
        Paragraph(f"{patient_data.get('crcl', 'N/A'):.1f} mL/min", normal_style)
    ])
    
    # Fourth row with diagnosis spanning full width
    patient_info.append([
        Paragraph("<b>Diagnosis:</b>", normal_style),
        Paragraph(f"{patient_data.get('clinical_diagnosis', 'N/A')}", normal_style),
        Paragraph("<b>Renal Function:</b>", normal_style),
        Paragraph(f"{patient_data.get('renal_function', 'N/A')}", normal_style)
    ])
    
    # Fifth row with regimen spanning full width
    patient_info.append([
        Paragraph("<b>Current Regimen:</b>", normal_style),
        Paragraph(f"{patient_data.get('current_dose_regimen', 'N/A')}", normal_style),
        Paragraph("", normal_style),
        Paragraph("", normal_style)
    ])
    
    # Create the table
    patient_table = Table(patient_info, colWidths=[100, 150, 100, 150])
    patient_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    content.append(patient_table)
    content.append(Spacer(1, 12))
    
    # Add drug information
    content.append(Paragraph("Drug Information", heading_style))
    content.append(Paragraph(drug_info, normal_style))
    content.append(Spacer(1, 12))
    
    # Add clinical assessment
    content.append(Paragraph("Clinical Assessment", heading_style))
    
    # Add measured levels
    content.append(Paragraph("Measured Levels:", section_style))
    
    # Create levels table
    levels_table_data = [["Parameter", "Value", "Target Range", "Status"]]
    
    for name, value, target, status in levels_data:
        # Determine status text and color
        if status == "within":
            status_text = "Within Range"
            status_color = colors.green
        elif status == "below":
            status_text = "Below Range"
            status_color = colors.orange
        else:  # above
            status_text = "Above Range"
            status_color = colors.red
        
        levels_table_data.append([name, value, target, status_text])
    
    levels_table = Table(levels_table_data)
    levels_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    # Add status color to each row in the table
    for i, (_, _, _, status) in enumerate(levels_data, 1):
        if status == "within":
            color = colors.lightgreen
        elif status == "below":
            color = colors.lightyellow
        else:  # above
            color = colors.mistyrose
        
        levels_table.setStyle(TableStyle([
            ('BACKGROUND', (3, i), (3, i), color),
        ]))
    
    content.append(levels_table)
    content.append(Spacer(1, 8))
    
    # Add assessment
    content.append(Paragraph("Assessment:", section_style))
    content.append(Paragraph(f"Patient is {assessment.upper()}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add calculations section if provided
    if calculation_details:
        content.append(Paragraph("Calculation Details:", section_style))
        content.append(Paragraph(calculation_details, normal_style))
        content.append(Spacer(1, 12))
    
    # Add recommendations
    content.append(Paragraph("Recommendations", heading_style))
    
    # Add dosing recommendations
    content.append(Paragraph("Dosing:", section_style))
    for rec in dosing_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add monitoring recommendations
    content.append(Paragraph("Monitoring:", section_style))
    for rec in monitoring_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add cautions if any
    if cautions and len(cautions) > 0:
        content.append(Paragraph("Cautions:", section_style))
        for caution in cautions:
            content.append(Paragraph(f"• {caution}", normal_style))
    
    # Add disclaimer
    content.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey
    )
    content.append(Paragraph("Disclaimer: This report is generated by an automated system and is intended to assist clinical decision making. Always use professional judgment when implementing recommendations.", disclaimer_style))
    
    # Build the PDF
    doc.build(content)
    
    # Get the PDF value from the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    # Encode the PDF to base64
    pdf_base64 = base64.b64encode(pdf_value).decode()
    
    return pdf_base64

# Function to create a download link for the PDF
def get_pdf_download_link(pdf_base64, filename="clinical_recommendations.pdf"):
    """Create a download link for a base64 encoded PDF"""
    href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="{filename}">Download Clinical Recommendations PDF</a>'
    return href

# Updated function to display buttons for printing and downloading recommendations
def display_pdf_download_button(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Display buttons to print/save recommendations as a PDF and print a summary
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Print/Save Full Report"):
            # Generate the PDF
            pdf_base64 = create_recommendation_pdf(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs,
                calculation_details,
                cautions
            )
            
            # Create the download link
            download_link = get_pdf_download_link(pdf_base64)
            
            # Display the download link
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Preview message
            st.success("PDF generated successfully. Click the link above to download.")
    
    with col2:
        if st.button("🖨️ Print Clinical Summary"):
            # Create a simple text printout of the assessment and recommendations
            assessment_text = create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            
            # Display in a text area that can be easily copied
            st.text_area("Copy this text to print", assessment_text, height=300)
            st.success("Assessment text generated. Copy and paste into your preferred document.")

# Enhanced function to create a printable text assessment
def create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """Create a plain text printable assessment for easy copying to clinical notes"""
    now = datetime.now()
    
    # Header
    text = f"ANTIMICROBIAL TDM ASSESSMENT - {now.strftime('%Y-%m-%d %H:%M')}\n"
    text += "=" * 50 + "\n\n"
    
    # Patient information - Make sure to include ID and ward
    text += f"Patient ID: {patient_data.get('patient_id', 'N/A')}\n"
    text += f"Ward: {patient_data.get('ward', 'N/A')}\n"
    text += f"Age: {patient_data.get('age', 'N/A')} years  |  "
    text += f"Gender: {patient_data.get('gender', 'N/A')}  |  "
    text += f"Weight: {patient_data.get('weight', 'N/A')} kg\n"
    text += f"Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}\n"
    text += f"CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})\n"
    text += f"Current regimen: {patient_data.get('current_dose_regimen', 'N/A')}\n\n"
    
    # Measured levels
    text += "MEASURED LEVELS:\n"
    for name, value, target, status in levels_data:
        status_text = "✓" if status == "within" else "↓" if status == "below" else "↑"
        text += f"- {name}: {value} (Target: {target}) {status_text}\n"
    
    # Assessment
    text += f"\nASSESSMENT: Patient is {assessment.upper()}\n\n"
    
    # PK Parameters (if available from calculation details)
    try:
        if "Half-life" in calculation_details or "t½" in calculation_details:
            text += "PHARMACOKINETIC PARAMETERS:\n"
            # Extract PK parameters from calculation details
            import re
            ke_match = re.search(r'Ke[\s=:]+([0-9.]+)', calculation_details)
            t_half_match = re.search(r't½[\s=:]+([0-9.]+)', calculation_details)
            
            if ke_match:
                ke = float(ke_match.group(1))
                text += f"- Elimination rate constant (Ke): {ke:.4f} hr⁻¹\n"
            
            if t_half_match:
                t_half = float(t_half_match.group(1))
                text += f"- Half-life (t½): {t_half:.2f} hr\n"
            
            text += "\n"
    except:
        pass  # Skip if unable to extract PK parameters
    
    # Recommendations
    text += "DOSING RECOMMENDATIONS:\n"
    for rec in dosing_recs:
        text += f"- {rec}\n"
    
    text += "\nMONITORING RECOMMENDATIONS:\n"
    for rec in monitoring_recs:
        text += f"- {rec}\n"
    
    # Cautions
    if cautions and len(cautions) > 0:
        text += "\nCAUTIONS:\n"
        for caution in cautions:
            text += f"- {caution}\n"
    
    # Footer
    text += "\n" + "=" * 50 + "\n"
    text += "This assessment is intended to assist clinical decision making.\n"
    text += "Always use professional judgment when implementing recommendations.\n"
    text += f"Generated by: Antimicrobial TDM App - {now.strftime('%Y-%m-%d')}"
    
    return text
    def vancomycin_auc_guided(patient_data):
    """Vancomycin AUC-guided monitoring method"""
    st.info("AUC-guided monitoring is the preferred approach according to recent guidelines")
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
        first_level = st.number_input("First Concentration (mg/L)", min_value=0.0, max_value=80.0, value=30.0, step=0.5)
        first_time = st.number_input("Time After Start of Infusion for First Sample (hours)", min_value=0.5, max_value=12.0, value=2.0, step=0.5)
    
    with col2:
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        second_level = st.number_input("Second Concentration (mg/L)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
        second_time = st.number_input("Time After Start of Infusion for Second Sample (hours)", min_value=2.0, max_value=24.0, value=8.0, step=0.5)
        
    # Target AUC selection
    target_auc_strategy = st.radio(
        "Target AUC24 Range",
        ["400-600 mg·hr/L (standard infections)", "500-700 mg·hr/L (serious infections)"],
        help="Select appropriate target based on severity of infection"
    )
    
    # Set target AUC range based on selection
    if "400-600" in target_auc_strategy:
        target_auc = (400, 600)
    else:
        target_auc = (500, 700)
    
    # Calculate button
    if st.button("Calculate Vancomycin AUC Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters from two-point sampling
            
            # Calculate elimination rate constant
            delta_time = second_time - first_time
            ke = -math.log(second_level/first_level)/delta_time
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Estimate Cmax (peak) - Assuming first sample is post-distribution
            t_after_infusion = first_time - infusion_time
            if t_after_infusion < 0:
                t_after_infusion = 0  # If first sample is during infusion
            
            estimated_peak = first_level * math.exp(ke * t_after_infusion)
            
            # Estimate Cmin (trough) - Before next dose
            t_to_next_dose = interval - second_time
            estimated_trough = second_level * math.exp(-ke * t_to_next_dose)
            
            # Calculate volume of distribution
            weight = patient_data.get('weight', 70)
            
            # Adjusted for infusion time
            vd_est = dose / (estimated_peak * (1 - math.exp(-ke * infusion_time)))
            
            # Calculate clearance
            cl = ke * vd_est
            
            # Calculate AUC for one dosing interval
            auc_tau = dose / cl
            
            # Calculate AUC24
            auc24 = auc_tau * (24 / interval)
            
            # Calculate new dose to reach target AUC24
            target_auc24 = (target_auc[0] + target_auc[1]) / 2  # Midpoint of target range
            new_dose = (target_auc24 * cl * interval) / 24
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin AUC Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("First Level", f"{first_level:.1f} mg/L at {first_time}h")
                st.metric("Second Level", f"{second_level:.1f} mg/L at {second_time}h")
                st.metric("Estimated Trough", f"{estimated_trough:.1f} mg/L")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Calculated AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd_est:.1f} L ({vd_est/weight:.2f} L/kg)")
                st.metric("Clearance", f"{cl:.2f} L/hr")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{interval}h")
            
            # Show AUC target status
            if auc24 < target_auc[0]:
                st.warning(f"⚠️ AUC24 ({auc24:.1f} mg·hr/L) is below target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            elif auc24 > target_auc[1]:
                st.warning(f"⚠️ AUC24 ({auc24:.1f} mg·hr/L) is above target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            else:
                st.success(f"✅ AUC24 ({auc24:.1f} mg·hr/L) is within target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            
            # Visualization
            st.subheader("Concentration-Time Curve with AUC")
            
            # Create data for visualization
            times = np.linspace(0, interval*1.5, 100)
            concentrations = []
            
            # Calculate concentration at each time point
            for t in times:
                if t <= infusion_time:
                    # During infusion
                    conc = estimated_peak * (t / infusion_timedef vancomycin_peak_trough(patient_data):
    """Vancomycin peak and trough monitoring method"""
    st.info("Peak and trough monitoring provides better insight into vancomycin pharmacokinetics")
    
    # Target trough selection
    target_trough_strategy = st.radio(
        "Target Trough Range",
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"],
        help="Select appropriate target based on indication"
    )
    
    # Set target ranges based on selection
    if "Empirical" in target_trough_strategy:
        target_cmin = (10, 15)
        target_peak = (20, 30)
    else:
        target_cmin = (15, 20)
        target_peak = (25, 40)
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
        peak = st.number_input("Measured Peak (mg/L)", min_value=5.0, max_value=80.0, value=25.0, step=0.5)
    
    with col2:
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        peak_draw_time = st.number_input("Time After Start of Infusion for Peak (hours)", min_value=0.5, max_value=6.0, value=1.5, step=0.5)
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=50.0, value=12.5, step=0.5)
    
    # Calculate button
    if st.button("Calculate Vancomycin Peak-Trough Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters based on peak and trough
            
            # Calculate elimination rate constant
            t_peak = peak_draw_time
            tau = interval
            ke = -math.log(trough/peak)/(tau - t_peak)
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Calculate volume of distribution
            weight = patient_data.get('weight', 70)
            
            # Adjusted peak based on timing (if peak drawn after end of infusion)
            if t_peak > infusion_time:
                # Backextrapolate to the end of infusion
                adjusted_peak = peak * math.exp(ke * (t_peak - infusion_time))
            else:
                adjusted_peak = peak
            
            # Calculate Vd using the adjusted peak
            vd = dose / adjusted_peak
            
            # Calculate clearance
            cl = ke * vd
            
            # Calculate AUC24
            auc24 = (dose * 24) / (cl * tau)
            
            # Calculate new dose to reach target trough
            target_trough = (target_cmin[0] + target_cmin[1]) / 2  # Midpoint of target range
            new_dose = (target_trough * cl * tau) / (24/tau)
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin Peak-Trough Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Measured Peak", f"{peak:.1f} mg/L")
                st.metric("Measured Trough", f"{trough:.1f} mg/L")
                
                # Show status based on peak and trough
                if peak < target_peak[0]:
                    st.warning("⚠️ Peak below target range")
                elif peak > target_peak[1]:
                    st.warning("⚠️ Peak above target range")
                else:
                    st.success("✅ Peak within target range")
                
                if trough < target_cmin[0]:
                    st.warning("⚠️ Trough below target range")
                elif trough > target_cmin[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough within target range")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Est. AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd:.1f} L ({vd/weight:.2f} L/kg)")
                st.metric("Clearance", f"{cl:.2f} L/hr")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{tau}h")
            
            # Visualization
            st.subheader("Concentration-Time Curve")
            
            # Plot concentration-time curve
            chart = plot_concentration_time_curve(
                "Vancomycin (Peak-Trough method)",
                [], "", [], [], "",
                peak=peak, 
                trough=trough,
                ke=ke,
                tau=tau
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd:.1f} L ({vd/weight:.2f} L/kg)
            Cl = {cl:.2f} L/hr
            Current peak = {peak:.1f} mg/L
            Current trough = {trough:.1f} mg/L
            Target peak = {target_peak[0]}-{target_peak[1]} mg/L
            Target trough = {target_cmin[0]}-{target_cmin[1]} mg/L
            Recommended dose = {practical_new_dose:.0f} mg q{tau}h
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"Vancomycin (Peak and Trough): Measured peak = {peak} mg/L, trough = {trough} mg/L, "
                f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                f"Target peak range = {target_peak[0]}-{target_peak[1]} mg/L, Recommended base dose = {practical_new_dose:.0f} mg."
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate interpretation
            interpretation_data = generate_standardized_interpretation(prompt, "Vancomycin")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            if "Empirical" in target_trough_strategy:
                drug_info = "Vancomycin (Peak and Trough method, Empirical dosing 10-15 mg/L)"
            else:
                drug_info = "Vancomycin (Peak and Trough method, Definitive dosing 15-20 mg/L)"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )# ===== MAIN APP LAYOUT =====
def main():
    """Main application layout and functionality"""
    st.title("🧪 Advanced Antimicrobial TDM Calculator")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📋 Patient TDM", "📈 PK Analysis", "📚 References"])
    
    with tab1:
        # Collect patient information
        patient_data = display_patient_info_section()
        
        # Select antimicrobial
        st.header("Antimicrobial Selection")
        antimicrobial = st.selectbox(
            "Select Antimicrobial", 
            ["Vancomycin", "Gentamicin", "Amikacin", "Other Aminoglycoside"]
        )
        
        # Conditionally display appropriate input fields based on selection
        if "Vancomycin" in antimicrobial:
            vancomycin_section(patient_data)
        elif any(drug in antimicrobial for drug in ["Gentamicin", "Amikacin", "Aminoglycoside"]):
            aminoglycoside_section(patient_data, drug_name=antimicrobial.lower())
        else:
            st.info("Please select an antimicrobial agent")
    
    with tab2:
        pharmacokinetic_analysis_section()
    
    with tab3:
        display_references()

# ===== VANCOMYCIN SECTION =====
def vancomycin_section(patient_data):
    """Display vancomycin-specific input fields and calculations"""
    st.subheader("Vancomycin TDM")
    
    # Vancomycin Monitoring Method
    monitoring_method = st.radio(
        "Monitoring Method",
        ["Trough-only", "Peak and Trough", "AUC-guided"],
        help="Select the monitoring approach for vancomycin"
    )
    
    # Input fields based on monitoring method
    if monitoring_method == "Trough-only":
        vancomycin_trough_only(patient_data)
    elif monitoring_method == "Peak and Trough":
        vancomycin_peak_trough(patient_data)
    else:  # AUC-guided
        vancomycin_auc_guided(patient_data)

def vancomycin_trough_only(patient_data):
    """Vancomycin trough-only monitoring method"""
    st.info("Trough-only monitoring is a traditional approach for vancomycin dosing")
    
    # Target trough selection
    target_trough_strategy = st.radio(
        "Target Trough Range",
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"],
        help="Select appropriate target based on indication"
    )
    
    # Set target trough range based on selection
    if "Empirical" in target_trough_strategy:
        target_cmin = (10, 15)
    else:
        target_cmin = (15, 20)
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
    
    with col2:
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=50.0, value=12.5, step=0.5)
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    
    # Add timing details
    timing_info = st.checkbox("Add Timing Information")
    if timing_info:
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Date of Last Dose", placeholder="YYYY-MM-DD")
            st.text_input("Time of Last Dose", placeholder="HH:MM")
        with col2:
            st.text_input("Date of Blood Sample", placeholder="YYYY-MM-DD")
            st.text_input("Time of Blood Sample", placeholder="HH:MM")
    
    # Calculate button
    if st.button("Calculate Vancomycin Trough Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters
            # Assume one-compartment model for simplicity
            
            # Determine patient CrCl
            crcl = patient_data.get('crcl', 100)
            weight = patient_data.get('weight', 70)
            
            # Estimate Ke based on renal function
            ke = 0.00083 * crcl + 0.0044
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Estimate volume of distribution (standard population value)
            vd = 0.7 * weight
            
            # Calculate trough concentration at steady state
            tau = interval
            
            # Calculate clearance
            cl = ke * vd
            
            # Calculate AUC24
            auc24 = (dose * 24) / (cl * tau)
            
            # Estimate peak concentration (simple model)
            peak = (dose / vd) * (1 - math.exp(-ke * infusion_time))
            
            # Calculate new dose to reach target trough
            target_trough = (target_cmin[0] + target_cmin[1]) / 2  # Midpoint of target range
            new_dose = (target_trough * cl * tau) / (24/tau)
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin Trough Analysis Complete")
            
            # Create three columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Trough", f"{trough:.1f} mg/L")
                st.metric("Target Trough", f"{target_cmin[0]}-{target_cmin[1]} mg/L")
                
                # Show status icon based on trough
                if trough < target_cmin[0]:
                    st.warning("⚠️ Trough below target range")
                elif trough > target_cmin[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough within target range")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Est. AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Est. Peak", f"{peak:.1f} mg/L")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg")
                st.metric("Volume of Distribution", f"{vd:.1f} L")
            
            # Create detailed calculation steps in an expander
            with st.expander("Show Calculation Details", expanded=False):
                st.write("### Pharmacokinetic Calculations")
                st.write(f"""
                **Patient Parameters:**
                - Weight: {weight} kg
                - CrCl: {crcl:.1f} mL/min
                
                **Estimated PK Parameters:**
                - Ke = 0.00083 × CrCl + 0.0044
                - Ke = 0.00083 × {crcl:.1f} + 0.0044 = {ke:.4f} hr⁻¹
                - t½ = 0.693 / Ke = 0.693 / {ke:.4f} = {t_half:.1f} hr
                - Vd = 0.7 × Weight = 0.7 × {weight} = {vd:.1f} L
                - Cl = Ke × Vd = {ke:.4f} × {vd:.1f} = {cl:.2f} L/hr
                
                **Dose Calculations:**
                - Current dose: {dose} mg every {tau} hr
                - Current trough: {trough:.1f} mg/L
                - Target trough: {target_trough:.1f} mg/L
                - New dose = (Target × Cl × Tau) / (24/Tau)
                - New dose = ({target_trough:.1f} × {cl:.2f} × {tau}) / (24/{tau})
                - New dose = {new_dose:.1f} mg
                - Practical dose: {practical_new_dose:.0f} mg
                
                **AUC Calculation:**
                - AUC24 = (Dose × 24) / (Cl × Tau)
                - AUC24 = ({dose} × 24) / ({cl:.2f} × {tau})
                - AUC24 = {auc24:.1f} mg·hr/L
                """)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd:.1f} L
            Cl = {cl:.2f} L/hr
            Current trough = {trough:.1f} mg/L
            Target trough = {target_cmin[0]}-{target_cmin[1]} mg/L
            Recommended dose = {practical_new_dose:.0f} mg q{tau}h
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"Vancomycin (Trough only): Measured trough = {trough} mg/L, "
                f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                f"Recommended base dose = {practical_new_dose:.0f} mg."
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate and display interpretation
            interpretation_data = generate_standardized_interpretation(prompt, "Vancomycin")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            if "Empirical" in target_trough_strategy:
                drug_info = "Vancomycin (Trough-only method, Empirical dosing 10-15 mg/L)"
            else:
                drug_info = "Vancomycin (Trough-only method, Definitive dosing 15-20 mg/L)"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )# ===== STANDARDIZED INTERPRETATION GENERATOR =====
def generate_standardized_interpretation(prompt, drug):
    """
    Generate a standardized interpretation based on drug type and prompt content
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt)
    elif "Aminoglycoside" in drug or "Gentamicin" in drug or "Amikacin" in drug:
        return generate_aminoglycoside_interpretation(prompt)
    else:
        # For generic, we'll create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None, calculation_details=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM with improved recommendation formatting
    and PDF printing capability
    
    This function can call the OpenAI API if configured, otherwise
    it will provide a simulated response with a standardized, clinically relevant format.
    
    Parameters:
    - prompt: The clinical data prompt
    - patient_data: Optional dictionary with patient information for PDF generation
    - calculation_details: Optional string with calculation details for PDF
    """
    # Extract the drug type from the prompt
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
        if "Trough only" in prompt:
            method = "Trough-only method"
        else:
            method = "Peak and Trough method"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
        if "Initial Dose" in prompt:
            method = "Initial dosing"
        else:
            method = "Conventional (C1/C2) method"
    else:
        drug = "Antimicrobial"
        method = "Standard method"
    
    drug_info = f"{drug} ({method})"
    
    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Format your response with these exact sections:
            
            ## CLINICAL ASSESSMENT
            📊 **MEASURED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **ASSESSMENT:** (state if appropriately dosed, underdosed, or overdosed)
            
            ## RECOMMENDATIONS
            🔵 **DOSING:** (action-oriented recommendations using verbs like CONTINUE, ADJUST, HOLD)
            🔵 **MONITORING:** (specific monitoring parameters and schedule)
            ⚠️ **CAUTIONS:** (relevant warnings, if any)
            
            Here is the case: {prompt}
            """
            
            # Call OpenAI API - updated for openai v1.0.0+
            response = openai.chat.completions.create(
                model="gpt-4",  # or your preferred model
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear recommendations."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            llm_response = response.choices[0].message.content
            st.write(llm_response)
            
            # Add a note about source
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")
            
            # We can't easily extract the structured data from the LLM response for PDF generation
            # So we'll skip the PDF option for the OpenAI path for now
            return
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
    
    # Format the standardized clinical interpretation
    interpretation_data = generate_standardized_interpretation(prompt, drug)
    
    # If the interpretation_data is a string (error message), just display it and return
    if isinstance(interpretation_data, str):
        st.write(interpretation_data)
        return
    
    # Unpack the interpretation data
    levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
    
    # Display the formatted interpretation
    formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
    st.write(formatted_interpretation)
    
    # Add the PDF download button if patient_data is provided
    if patient_data:
        display_pdf_download_button(
            patient_data, 
            drug_info, 
            levels_data, 
            assessment, 
            dosing_recs, 
            monitoring_recs, 
            calculation_details,
            cautions
        )
    
    # Add the raw prompt at the bottom for debugging
    with st.expander("Raw Analysis Data", expanded=False):
        st.code(prompt)
        
    # Add note about simulated response
    st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")# ===== VANCOMYCIN INTERPRETATION FUNCTION =====
def generate_vancomycin_interpretation(prompt):
    """
    Generate standardized vancomycin interpretation
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    # Extract key values from the prompt
    peak_val = None
    trough_val = None
    auc24 = None
    
    # Extract peak and trough values
    if "Peak" in prompt:
        parts = prompt.split("Peak")
        if len(parts) > 1:
            peak_parts = parts[1].split("mg/L")
            if peak_parts:
                try:
                    peak_str = peak_parts[0].replace("=", "").replace(":", "").strip()
                    peak_val = float(peak_str)
                except ValueError:
                    pass
    
    if "Trough" in prompt:
        parts = prompt.split("Trough")
        if len(parts) > 1:
            trough_parts = parts[1].split("mg/L")
            if trough_parts:
                try:
                    trough_str = trough_parts[0].replace("=", "").replace(":", "").strip()
                    trough_val = float(trough_str)
                except ValueError:
                    pass
    
    # Extract AUC if available
    if "AUC24" in prompt:
        parts = prompt.split("AUC24")
        if len(parts) > 1:
            auc_parts = parts[1].split("mg·hr/L")
            if auc_parts:
                try:
                    auc_str = auc_parts[0].replace("=", "").replace(":", "").strip()
                    auc24 = float(auc_str)
                except ValueError:
                    pass
    
    # Extract trough target range
    trough_target_min, trough_target_max = 10, 20  # Default range
    if "Target trough range" in prompt:
        parts = prompt.split("Target trough range")
        if len(parts) > 1:
            range_parts = parts[1].strip().split("mg/L")
            if range_parts:
                try:
                    range_str = range_parts[0].replace("=", "").replace(":", "").strip()
                    if "-" in range_str:
                        min_max = range_str.split("-")
                        trough_target_min = float(min_max[0])
                        trough_target_max = float(min_max[1])
                except ValueError:
                    pass
    
    # Determine if empiric or definitive therapy based on trough target
    if trough_target_max <= 15:
        regimen = "Empiric"
        trough_target = f"{trough_target_min}-{trough_target_max} mg/L"
    else:
        regimen = "Definitive"
        trough_target = f"{trough_target_min}-{trough_target_max} mg/L"
    
    # Set AUC target based on indication
    if regimen == "Empiric":
        auc_target = "400-600 mg·hr/L"
        auc_min, auc_max = 400, 600
    else:  # Definitive
        auc_target = "400-800 mg·hr/L"
        auc_min, auc_max = 400, 800
    
    # Define peak target range
    peak_target = "20-40 mg/L"  # Typical peak range
    peak_min, peak_max = 20, 40
    
    # Determine vancomycin status
    status = "assessment not available"
    
    # If using trough-only monitoring
    if trough_val is not None and peak_val is None and auc24 is None:
        if trough_val < trough_target_min:
            status = "subtherapeutic (low trough)"
        elif trough_val > trough_target_max:
            status = "potentially supratherapeutic (high trough)"
        else:
            status = "appropriately dosed (trough-based)"
    
    # If using peak and trough monitoring
    elif trough_val is not None and peak_val is not None:
        if peak_val < peak_min and trough_val < trough_target_min:
            status = "subtherapeutic (inadequate peak and trough)"
        elif peak_val < peak_min:
            status = "potential underdosing (low peak)"
        elif trough_val < trough_target_min:
            status = "subtherapeutic (inadequate trough)"
        elif trough_val > trough_target_max:
            status = "potentially supratherapeutic (high trough)"
        elif peak_val > peak_max:
            status = "potentially supratherapeutic (high peak)"
        elif peak_min <= peak_val <= peak_max and trough_target_min <= trough_val <= trough_target_max:
            status = "appropriately dosed"
        else:
            status = "requires adjustment"
    
    # If using AUC monitoring
    elif auc24 is not None:
        if auc24 < auc_min:
            status = "subtherapeutic (low AUC)"
        elif auc24 > auc_max:
            status = "potentially supratherapeutic (high AUC)"
        else:
            status = "appropriately dosed (AUC-based)"
    
    # Create levels data based on available measurements
    levels_data = []
    
    if peak_val is not None:
        if peak_val < peak_min:
            peak_status = "below"
        elif peak_val > peak_max:
            peak_status = "above"
        else:
            peak_status = "within"
        levels_data.append(("Peak", f"{peak_val:.1f} mg/L", peak_target, peak_status))
    
    if trough_val is not None:
        if trough_val < trough_target_min:
            trough_status = "below"
        elif trough_val > trough_target_max:
            trough_status = "above"
        else:
            trough_status = "within"
        levels_data.append(("Trough", f"{trough_val:.1f} mg/L", trough_target, trough_status))
    
    if auc24 is not None:
        if auc24 < auc_min:
            auc_status = "below"
        elif auc24 > auc_max:
            auc_status = "above"
        else:
            auc_status = "within"
        levels_data.append(("AUC24", f"{auc24:.1f} mg·hr/L", auc_target, auc_status))
    
    # Generate recommendations based on status
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    
    # Check if we have enough data to provide recommendations
    if not levels_data:
        return "Insufficient data to generate interpretation. At least one measurement (peak, trough, or AUC) is required."
    
    # Extract new dose if available
    new_dose = None
    if "Recommended base dose" in prompt:
        parts = prompt.split("Recommended base dose")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose_str = dose_parts[0].replace("=", "").replace(":", "").strip()
                    new_dose = float(dose_str)
                except ValueError:
                    pass
    
    # Format new dose
    rounded_new_dose = None
    if new_dose:
        # Round to nearest 250mg for vancomycin
        rounded_new_dose = round(new_dose / 250) * 250
    
    # Generate recommendations based on status
    if status == "subtherapeutic (low trough)" or status == "subtherapeutic (inadequate trough)" or status == "subtherapeutic (low AUC)":
        if rounded_new_dose:
            dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("INCREASE dose by 25-30%")
        dosing_recs.append("CONSIDER shortening dosing interval")
        
        monitoring_recs.append("RECHECK levels after 3-4 doses (at steady state)")
        monitoring_recs.append("MONITOR renal function regularly")
        
        cautions.append("Subtherapeutic levels may lead to treatment failure")
        cautions.append("Ensure adequate hydration when increasing doses")
    
    elif status == "potentially supratherapeutic (high trough)" or status == "potentially supratherapeutic (high AUC)":
        if rounded_new_dose:
            dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("DECREASE dose by 20-25%")
        dosing_recs.append("CONSIDER extending dosing interval")
        
        monitoring_recs.append("RECHECK levels after 3-4 doses")
        monitoring_recs.append("MONITOR renal function closely")
        monitoring_recs.append("ASSESS for signs of nephrotoxicity")
        
        cautions.append("Risk of nephrotoxicity with elevated trough levels")
        cautions.append("Consider patient-specific risk factors for toxicity")
    
    elif status == "subtherapeutic (inadequate peak and trough)" or status == "potential underdosing (low peak)":
        if rounded_new_dose:
            dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("INCREASE dose by 30-40%")
        
        monitoring_recs.append("RECHECK peak and trough levels after 3-4 doses")
        monitoring_recs.append("VERIFY correct timing of sample collection")
        
        cautions.append("Significantly subtherapeutic levels increase risk of treatment failure")
        cautions.append("Consider evaluating for altered pharmacokinetics")
    
    elif status == "potentially supratherapeutic (high peak)":
        if rounded_new_dose:
            dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("DECREASE dose by 15-20%")
        dosing_recs.append("EXTEND dosing interval if appropriate")
        
        monitoring_recs.append("RECHECK levels after adjustment")
        monitoring_recs.append("MONITOR renal function")
        
        cautions.append("Risk of nephrotoxicity with excessive dosing")
    
    elif "appropriately dosed" in status:
        dosing_recs.append("CONTINUE current dosing regimen")
        
        monitoring_recs.append("MONITOR renal function regularly")
        monitoring_recs.append("REASSESS levels if clinical status changes")
        
        cautions.append("Even with therapeutic levels, monitor for adverse effects")
    
    else:  # requires adjustment
        if rounded_new_dose:
            dosing_recs.append(f"ADJUST dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("ADJUST dosing based on clinical response and levels")
        
        monitoring_recs.append("RECHECK levels after adjustment")
        monitoring_recs.append("MONITOR renal function")
        
        cautions.append("Individualize therapy based on clinical response")
    
    # Add standard monitoring recommendations
    if "MONITOR renal function" not in " ".join(monitoring_recs):
        monitoring_recs.append("MONITOR renal function every 2-3 days")
    
    return levels_data, assessment, dosing_recs, monitoring_recs, cautions# ===== AMINOGLYCOSIDE INTERPRETATION FUNCTION =====
def generate_aminoglycoside_interpretation(prompt):
    """
    Generate standardized aminoglycoside interpretation
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    # Extract key values from the prompt
    drug_name = "aminoglycoside"
    peak_val = None
    trough_val = None
    
    if "Gentamicin" in prompt:
        drug_name = "gentamicin"
    elif "Amikacin" in prompt:
        drug_name = "amikacin"
    
    # Extract peak and trough values
    if "Cmax:" in prompt:
        parts = prompt.split("Cmax:")
        if len(parts) > 1:
            peak_parts = parts[1].split(",")
            if peak_parts:
                try:
                    peak_val = float(peak_parts[0])
                except ValueError:
                    pass
    elif "Expected Cmax" in prompt:
        parts = prompt.split("Expected Cmax")
        if len(parts) > 1:
            peak_parts = parts[1].split("mg/L")
            if peak_parts:
                try:
                    peak_val = float(peak_parts[0].replace(":", "").strip())
                except ValueError:
                    pass
    
    if "Cmin:" in prompt:
        parts = prompt.split("Cmin:")
        if len(parts) > 1:
            trough_parts = parts[1].split(",")
            if trough_parts:
                try:
                    trough_val = float(trough_parts[0])
                except ValueError:
                    pass
    elif "Expected Cmin" in prompt:
        parts = prompt.split("Expected Cmin")
        if len(parts) > 1:
            trough_parts = parts[1].split("mg/L")
            if trough_parts:
                try:
                    trough_val = float(trough_parts[0].replace(":", "").strip())
                except ValueError:
                    pass
    
    # Extract dose
    dose = None
    if "Dose:" in prompt:
        parts = prompt.split("Dose:")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose = float(dose_parts[0])
                except ValueError:
                    pass
    elif "Dose " in prompt:
        parts = prompt.split("Dose ")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose = float(dose_parts[0])
                except ValueError:
                    pass
    
    # Extract suggested new dose
    new_dose = None
    if "Suggested new dose:" in prompt:
        parts = prompt.split("Suggested new dose:")
        if len(parts) > 1:
            new_dose_parts = parts[1].split("mg")
            if new_dose_parts:
                try:
                    new_dose = float(new_dose_parts[0])
                except ValueError:
                    pass
    elif "Recommended" in prompt and "Dose" in prompt:
        parts = prompt.split("Recommended")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    # Extract the number from this string
                    import re
                    numbers = re.findall(r'\d+', dose_parts[0])
                    if numbers:
                        new_dose = float(numbers[0])
                except ValueError:
                    pass
    
    # Extract target values based on regimen mention
    regimen = None
    if "SDD" in prompt:
        regimen = "SDD"
    elif "Synergy" in prompt:
        regimen = "Synergy"
    elif "MDD" in prompt:
        regimen = "MDD"
    
    # Set target ranges based on drug
    if drug_name == "gentamicin":
        if regimen == "SDD":
            peak_target = "10-30 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 10, 30
            trough_max = 1
        elif regimen == "Synergy":
            peak_target = "3-5 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 3, 5
            trough_max = 1
        else:  # Default to MDD
            peak_target = "5-10 mg/L"
            trough_target = "<2 mg/L"
            peak_min, peak_max = 5, 10
            trough_max = 2
    elif drug_name == "amikacin":
        if regimen == "SDD":
            peak_target = "60-80 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 60, 80
            trough_max = 1
        else:  # Default to MDD
            peak_target = "20-30 mg/L"
            trough_target = "<10 mg/L"
            peak_min, peak_max = 20, 30
            trough_max = 10
    else:
        peak_target = "varies by drug"
        trough_target = "varies by drug"
        peak_min, peak_max = 0, 100
        trough_max = 10
    
    # Determine aminoglycoside status
    status = "assessment not available"
    if peak_val and trough_val:
        if peak_val < peak_min and trough_val > trough_max:
            status = "ineffective and potentially toxic"
        elif peak_val < peak_min:
            status = "subtherapeutic (inadequate peak)"
        elif trough_val > trough_max:
            status = "potentially toxic (elevated trough)"
        elif peak_min <= peak_val <= peak_max and trough_val <= trough_max:
            status = "appropriately dosed"
        elif peak_val > peak_max:
            status = "potentially toxic (elevated peak)"
        else:
            status = "requires adjustment"
    
    # Format new dose
    rounded_new_dose = None
    if new_dose:
        # Round to nearest 10mg for most aminoglycosides
        rounded_new_dose = round(new_dose / 10) * 10
    
    # Create interpretation using standardized format
    if peak_val is not None and trough_val is not None:
        # Determine peak status
        if peak_val < peak_min:
            peak_status = "below"
        elif peak_val > peak_max:
            peak_status = "above"
        else:
            peak_status = "within"
        
        # Determine trough status
        if trough_val > trough_max:
            trough_status = "above"
        else:
            trough_status = "within"
        
        # Prepare data for standardized format
        levels_data = [
            (f"Peak", f"{peak_val:.1f} mg/L", peak_target, peak_status),
            (f"Trough", f"{trough_val:.2f} mg/L", trough_target, trough_status)
        ]
        
        # Generate recommendations based on status
        dosing_recs = []
        monitoring_recs = []
        cautions = []
        
        if status == "ineffective and potentially toxic":
            dosing_recs.append("HOLD next dose")
            dosing_recs.append("REASSESS renal function before resuming")
            if rounded_new_dose:
                dosing_recs.append(f"DECREASE to {rounded_new_dose}mg when resumed")
            dosing_recs.append("EXTEND dosing interval significantly")
            
            monitoring_recs.append("CHECK renal function before resuming therapy")
            monitoring_recs.append("RECHECK levels 2 doses after resumption")
            monitoring_recs.append("MONITOR for signs of ototoxicity and nephrotoxicity")
            
            cautions.append("Risk of nephrotoxicity and ototoxicity with elevated trough levels")
            cautions.append("Ineffective therapy may lead to treatment failure")
            
        elif status == "subtherapeutic (inadequate peak)":
            if rounded_new_dose:
                dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
            else:
                dosing_recs.append("INCREASE dose by 25-50%")
            
            monitoring_recs.append("RECHECK levels after 2-3 doses")
            monitoring_recs.append("VERIFY correct timing of peak sample collection")
            
            cautions.append("Subtherapeutic levels may lead to treatment failure")
            cautions.append("Consider other factors affecting drug disposition")
            
        elif status == "potentially toxic (elevated trough)":
            dosing_recs.append("EXTEND dosing interval")
            if rounded_new_dose:
                dosing_recs.append(f"CONSIDER dose reduction to {rounded_new_dose}mg")
            
            monitoring_recs.append("MONITOR renal function closely")
            monitoring_recs.append("RECHECK levels before next dose")
            monitoring_recs.append("ASSESS for signs of ototoxicity and nephrotoxicity")
            
            cautions.append("Risk of nephrotoxicity and ototoxicity with elevated trough levels")
            cautions.append("Consider patient-specific risk factors for toxicity")
            
        elif status == "potentially toxic (elevated peak)":
            if rounded_new_dose:
                dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
            else:
                dosing_recs.append("DECREASE dose by 20-25%")
            
            monitoring_recs.append("RECHECK levels after 2-3 doses")
            monitoring_recs.append("VERIFY correct timing of peak sample collection")
            monitoring_recs.append("MONITOR for signs of ototoxicity")
            
            cautions.append("Risk of ototoxicity with significantly elevated peak levels")
            
        elif status == "appropriately dosed":
            dosing_recs.append("CONTINUE current dosing regimen")
            
            monitoring_recs.append("MONITOR renal function regularly")
            monitoring_recs.append("REASSESS levels if clinical status changes")
            monitoring_recs.append("CONSIDER extended interval dosing for longer therapy")
            
            cautions.append("Even with therapeutic levels, monitor for adverse effects")
            
        else:  # requires adjustment
            dosing_recs.append("ADJUST dosing based on clinical response")
            if rounded_new_dose:
                dosing_recs.append(f"CONSIDER dose of {rounded_new_dose}mg")
            
            monitoring_recs.append("RECHECK levels after adjustment")
            monitoring_recs.append("MONITOR renal function")
            
            cautions.append("Individualize therapy based on clinical response")
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions
    else:
        return "Insufficient data to generate interpretation. Both peak and trough levels are required."# ===== FORMAT_CLINICAL_RECOMMENDATIONS FUNCTION =====
def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy
    
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
    levels_md = "📊 **MEASURED LEVELS:**\n"
    for name, value, target, status in levels_data:
        icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴"
        levels_md += f"- {name}: {value} (Target: {target}) {icon}\n"
    
    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is {assessment.upper()}"
    
    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## DETAILED RECOMMENDATIONS

🔵 **DOSING RECOMMENDATIONS:**
"""
    for rec in dosing_recs:
        output += f"- {rec}\n"
    
    output += "\n🔵 **MONITORING RECOMMENDATIONS:**\n"
    for rec in monitoring_recs:
        output += f"- {rec}\n"
    
    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS & CONSIDERATIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"
    
    # Add a summary section for quick reference
    output += "\n## QUICK SUMMARY\n"
    output += "**Status:** " + assessment.upper() + "\n"
    
    # Summarize key recommendations
    if len(dosing_recs) > 0:
        output += f"**Key Dosing Action:** {dosing_recs[0]}\n"
    
    if len(monitoring_recs) > 0:
        output += f"**Key Monitoring Action:** {monitoring_recs[0]}\n"
        
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    output += f"\n*Generated on: {timestamp}*"
    
    return outputimport streamlit as st
import numpy as np
import math
import pandas as pd
import altair as alt
import base64
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

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
    # Check for OpenAI API key
    import openai
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError, ImportError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.
    
    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== PATIENT INFO SECTION =====
def display_patient_info_section():
    """Display and collect patient information"""
    st.header("Patient Information")
    
    # Create a 2x2 grid for patient info
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", help="Enter the patient's unique identifier")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        serum_cr = st.number_input("Serum Creatinine (μmol/L)", min_value=10, max_value=1000, value=80)
    
    with col2:
        ward = st.text_input("Ward/Unit", help="Enter the patient's current location")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=40, max_value=250, value=170)
        clinical_diagnosis = st.text_input("Clinical Diagnosis", "Sepsis")
    
    # Calculate Creatinine Clearance
    # Cockcroft-Gault equation
    scr_mg = serum_cr / 88.4  # Convert μmol/L to mg/dL
    if gender == "Male":
        crcl = ((140 - age) * weight) / (72 * scr_mg)
    else:
        crcl = ((140 - age) * weight * 0.85) / (72 * scr_mg)
    
    # Determine renal function category
    if crcl >= 90:
        renal_function = "Normal renal function"
    elif crcl >= 60:
        renal_function = "Mild renal impairment"
    elif crcl >= 30:
        renal_function = "Moderate renal impairment"
    elif crcl >= 15:
        renal_function = "Severe renal impairment"
    else:
        renal_function = "Renal failure"
    
    # Display calculated CrCl and renal function
    st.metric("Estimated CrCl", f"{crcl:.1f} mL/min", renal_function)
    
    # Current medication regimen
    current_dose_regimen = st.text_input("Current Dosing Regimen", "Vancomycin 1000mg q12h")
    
    st.info(f"Patient {patient_id} is in {ward} with {renal_function.lower()} (CrCl: {crcl:.1f} mL/min)")
    
    # Return patient data as a dictionary
    return {
        'patient_id': patient_id,
        'ward': ward,
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen
    }

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details, peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization
    
    Parameters:
    - drug_info: String with drug name
    - levels_data: List of level data
    - assessment: Assessment string
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - calculation_details: String with calculation details
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
            conc = trough + (peak - trough) * (t / infusion_time)
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
    if "Vancomycin" in drug_info:  # Vancomycin
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
    elif "Gentamicin" in drug_info:  # Gentamicin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [10], 'y2': [30]  # Peak range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [5], 'y2': [10]  # Peak range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [2]  # Trough range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    elif "Amikacin" in drug_info:  # Amikacin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [60], 'y2': [80]  # Peak range for amikacin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for amikacin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [20], 'y2': [30]  # Peak range for amikacin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [10]  # Trough range for amikacin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    else:  # Default or unknown drug
        target_peak_band = alt.Chart(pd.DataFrame({
            'y1': [peak*0.8], 'y2': [peak*1.2]  # Default peak range ±20%
        })).mark_rect(opacity=0.2, color='green').encode(
            y='y1', y2='y2'
        )
        target_trough_band = alt.Chart(pd.DataFrame({
            'y1': [trough*0.5], 'y2': [trough*1.5]  # Default trough range ±50%
        })).mark_rect(opacity=0.2, color='blue').encode(
            y='y1', y2='y2'
        )
    
    # Create the concentration-time curve
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)')
    )
    
    # Add markers for actual measured peak and trough
    markers = alt.Chart(pd.DataFrame({
        'Time (hr)': [t_peak, tau],
        'Concentration (mg/L)': [peak, trough],
        'Label': ['Peak', 'Trough']
    })).mark_point(size=100, filled=True).encode(
        x='Time (hr)',
        y='Concentration (mg/L)',
        color=alt.Color('Label', scale=alt.Scale(domain=['Peak', 'Trough'], range=['green', 'blue']))
    )
    
    # Add vertical lines for key time points
    infusion_end = alt.Chart(pd.DataFrame({'x': [infusion_time]})).mark_rule(
        strokeDash=[5, 5], color='gray'
    ).encode(x='x')
    
    next_dose = alt.Chart(pd.DataFrame({'x': [tau]})).mark_rule(
        strokeDash=[5, 5], color='red'
    ).encode(x='x')
    
    # Add text annotations for key time points
    annotations = alt.Chart(pd.DataFrame({
        'x': [infusion_time/2, tau],
        'y': [peak*1.1, trough*0.9],
        'text': ['Infusion', 'Next Dose']
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Calculate half-life and display it
    half_life = 0.693 / ke
    half_life_text = alt.Chart(pd.DataFrame({
        'x': [tau/2],
        'y': [peak*0.5],
        'text': [f"t½ = {half_life:.1f} hr"]
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Combine charts
    chart = alt.layer(
        target_peak_band, 
        target_trough_band, 
        line, 
        markers,
        infusion_end,
        next_dose,
        annotations,
        half_life_text
    ).properties(
        width=600,
        height=400,
        title=f'{drug_info} Concentration-Time Profile'
    )
    
    # Display detailed calculation steps in an expander
    with st.expander("View Calculation Details", expanded=False):
        st.markdown("### PK Parameter Calculations")
        st.markdown(f"""
        **Key Parameters:**
        - Peak concentration (Cmax): {peak:.2f} mg/L
        - Trough concentration (Cmin): {trough:.2f} mg/L
        - Elimination rate constant (Ke): {ke:.4f} hr⁻¹
        - Half-life (t½): {half_life:.2f} hr
        - Dosing interval (τ): {tau} hr
        
        **Detailed Calculations:**
        ```
        Ke = -ln(Cmin/Cmax)/(τ - tpeak)
        Ke = -ln({trough:.2f}/{peak:.2f})/({tau} - {t_peak})
        Ke = {ke:.4f} hr⁻¹
        
        t½ = 0.693/Ke
        t½ = 0.693/{ke:.4f}
        t½ = {half_life:.2f} hr
        ```
        
        **Assessment:**
        {assessment}
        
        **Dosing Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in dosing_recs]) + """
        
        **Monitoring Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in monitoring_recs]))
        
        if calculation_details:
            st.markdown("**Additional Calculation Information:**")
            st.markdown(calculation_details)
    
    return chart

# ===== PDF GENERATION FUNCTIONS =====
def create_recommendation_pdf(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Create a downloadable PDF with the clinical recommendations
    
    Parameters:
    - patient_data: Dictionary with patient information
    - drug_info: String with drug name and method
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - calculation_details: Optional string with calculation details
    - cautions: Optional list of caution strings
    
    Returns:
    - base64 encoded PDF for download
    """
    # Create an in-memory PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom styles
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading3'],
        spaceAfter=6,
        textColor=colors.navy
    )
    
    # Create the content
    content = []
    
    # Add report title
    content.append(Paragraph("Antimicrobial TDM Report", title_style))
    content.append(Spacer(1, 12))
    
    # Add date and time
    now = datetime.now()
    content.append(Paragraph(f"Report Generated: {now.strftime('%Y-%m-%d %H:%M')}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add patient information
    content.append(Paragraph("Patient Information", heading_style))
    
    # Create patient info table with ID and Ward
    patient_info = []
    
    # Add patient ID and ward row
    patient_info.append([
        Paragraph("<b>Patient ID:</b>", normal_style),
        Paragraph(f"{patient_data.get('patient_id', 'N/A')}", normal_style),
        Paragraph("<b>Ward:</b>", normal_style),
        Paragraph(f"{patient_data.get('ward', 'N/A')}", normal_style)
    ])
    
    # First row
    patient_info.append([
        Paragraph("<b>Age:</b>", normal_style),
        Paragraph(f"{patient_data.get('age', 'N/A')} years", normal_style),
        Paragraph("<b>Gender:</b>", normal_style),
        Paragraph(f"{patient_data.get('gender', 'N/A')}", normal_style)
    ])
    
    # Second row
    patient_info.append([
        Paragraph("<b>Weight:</b>", normal_style),
        Paragraph(f"{patient_data.get('weight', 'N/A')} kg", normal_style),
        Paragraph("<b>Height:</b>", normal_style),
        Paragraph(f"{patient_data.get('height', 'N/A')} cm", normal_style)
    ])
    
    # Third row
    patient_info.append([
        Paragraph("<b>Serum Creatinine:</b>", normal_style),
        Paragraph(f"{patient_data.get('serum_cr', 'N/A')} µmol/L", normal_style),
        Paragraph("<b>CrCl:</b>", normal_style),
        Paragraph(f"{patient_data.get('crcl', 'N/A'):.1f} mL/min", normal_style)
    ])
    
    # Fourth row with diagnosis spanning full width
    patient_info.append([
        Paragraph("<b>Diagnosis:</b>", normal_style),
        Paragraph(f"{patient_data.get('clinical_diagnosis', 'N/A')}", normal_style),
        Paragraph("<b>Renal Function:</b>", normal_style),
        Paragraph(f"{patient_data.get('renal_function', 'N/A')}", normal_style)
    ])
    
    # Fifth row with regimen spanning full width
    patient_info.append([
        Paragraph("<b>Current Regimen:</b>", normal_style),
        Paragraph(f"{patient_data.get('current_dose_regimen', 'N/A')}", normal_style),
        Paragraph("", normal_style),
        Paragraph("", normal_style)
    ])
    
    # Create the table
    patient_table = Table(patient_info, colWidths=[100, 150, 100, 150])
    patient_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    content.append(patient_table)
    content.append(Spacer(1, 12))
    
    # Add drug information
    content.append(Paragraph("Drug Information", heading_style))
    content.append(Paragraph(drug_info, normal_style))
    content.append(Spacer(1, 12))
    
    # Add clinical assessment
    content.append(Paragraph("Clinical Assessment", heading_style))
    
    # Add measured levels
    content.append(Paragraph("Measured Levels:", section_style))
    
    # Create levels table
    levels_table_data = [["Parameter", "Value", "Target Range", "Status"]]
    
    for name, value, target, status in levels_data:
        # Determine status text and color
        if status == "within":
            status_text = "Within Range"
            status_color = colors.green
        elif status == "below":
            status_text = "Below Range"
            status_color = colors.orange
        else:  # above
            status_text = "Above Range"
            status_color = colors.red
        
        levels_table_data.append([name, value, target, status_text])
    
    levels_table = Table(levels_table_data)
    levels_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    # Add status color to each row in the table
    for i, (_, _, _, status) in enumerate(levels_data, 1):
        if status == "within":
            color = colors.lightgreen
        elif status == "below":
            color = colors.lightyellow
        else:  # above
            color = colors.mistyrose
        
        levels_table.setStyle(TableStyle([
            ('BACKGROUND', (3, i), (3, i), color),
        ]))
    
    content.append(levels_table)
    content.append(Spacer(1, 8))
    
    # Add assessment
    content.append(Paragraph("Assessment:", section_style))
    content.append(Paragraph(f"Patient is {assessment.upper()}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add calculations section if provided
    if calculation_details:
        content.append(Paragraph("Calculation Details:", section_style))
        content.append(Paragraph(calculation_details, normal_style))
        content.append(Spacer(1, 12))
    
    # Add recommendations
    content.append(Paragraph("Recommendations", heading_style))
    
    # Add dosing recommendations
    content.append(Paragraph("Dosing:", section_style))
    for rec in dosing_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add monitoring recommendations
    content.append(Paragraph("Monitoring:", section_style))
    for rec in monitoring_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add cautions if any
    if cautions and len(cautions) > 0:
        content.append(Paragraph("Cautions:", section_style))
        for caution in cautions:
            content.append(Paragraph(f"• {caution}", normal_style))
    
    # Add disclaimer
    content.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey
    )
    content.append(Paragraph("Disclaimer: This report is generated by an automated system and is intended to assist clinical decision making. Always use professional judgment when implementing recommendations.", disclaimer_style))
    
    # Build the PDF
    doc.build(content)
    
    # Get the PDF value from the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    # Encode the PDF to base64
    pdf_base64 = base64.b64encode(pdf_value).decode()
    
    return pdf_base64

# Function to create a download link for the PDF
def get_pdf_download_link(pdf_base64, filename="clinical_recommendations.pdf"):
    """Create a download link for a base64 encoded PDF"""
    href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="{filename}">Download Clinical Recommendations PDF</a>'
    return href

# Updated function to display buttons for printing and downloading recommendations
def display_pdf_download_button(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Display buttons to print/save recommendations as a PDF and print a summary
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Print/Save Full Report"):
            # Generate the PDF
            pdf_base64 = create_recommendation_pdf(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs,
                calculation_details,
                cautions
            )
            
            # Create the download link
            download_link = get_pdf_download_link(pdf_base64)
            
            # Display the download link
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Preview message
            st.success("PDF generated successfully. Click the link above to download.")
    
    with col2:
        if st.button("🖨️ Print Clinical Summary"):
            # Create a simple text printout of the assessment and recommendations
            assessment_text = create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            
            # Display in a text area that can be easily copied
            st.text_area("Copy this text to print", assessment_text, height=300)
            st.success("Assessment text generated. Copy and paste into your preferred document.")

# Enhanced function to create a printable text assessment
def create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """Create a plain text printable assessment for easy copying to clinical notes"""
    now = datetime.now()
    
    # Header
    text = f"ANTIMICROBIAL TDM ASSESSMENT - {now.strftime('%Y-%m-%d %H:%M')}\n"
    text += "=" * 50 + "\n\n"
    
    # Patient information - Make sure to include ID and ward
    text += f"Patient ID: {patient_data.get('patient_id', 'N/A')}\n"
    text += f"Ward: {patient_data.get('ward', 'N/A')}\n"
    text += f"Age: {patient_data.get('age', 'N/A')} years  |  "
    text += f"Gender: {patient_data.get('gender', 'N/A')}  |  "
    text += f"Weight: {patient_data.get('weight', 'N/A')} kg\n"
    text += f"Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}\n"
    text += f"CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})\n"
    text += f"Current regimen: {patient_data.get('current_dose_regimen', 'N/A')}\n\n"
    
    # Measured levels
    text += "MEASURED LEVELS:\n"
    for name, value, target, status in levels_data:
        status_text = "✓" if status == "within" else "↓" if status == "below" else "↑"
        text += f"- {name}: {value} (Target: {target}) {status_text}\n"
    
    # Assessment
    text += f"\nASSESSMENT: Patient is {assessment.upper()}\n\n"
    
    # PK Parameters (if available from calculation details)
    try:
        if "Half-life" in calculation_details or "t½" in calculation_details:
            text += "PHARMACOKINETIC PARAMETERS:\n"
            # Extract PK parameters from calculation details
            import re
            ke_match = re.search(r'Ke[\s=:]+([0-9.]+)', calculation_details)
            t_half_match = re.search(r't½[\s=:]+([0-9.]+)', calculation_details)
            
            if ke_match:
                ke = float(ke_match.group(1))
                text += f"- Elimination rate constant (Ke): {ke:.4f} hr⁻¹\n"
            
            if t_half_match:
                t_half = float(t_half_match.group(1))
                text += f"- Half-life (t½): {t_half:.2f} hr\n"
            
            text += "\n"
    except:
        pass  # Skip if unable to extract PK parameters
    
    # Recommendations
    text += "DOSING RECOMMENDATIONS:\n"
    for rec in dosing_recs:
        text += f"- {rec}\n"
    
    text += "\nMONITORING RECOMMENDATIONS:\n"
    for rec in monitoring_recs:
        text += f"- {rec}\n"
    
    # Cautions
    if cautions and len(cautions) > 0:
        text += "\nCAUTIONS:\n"
        for caution in cautions:
            text += f"- {caution}\n"
    
    # Footer
    text += "\n" + "=" * 50 + "\n"
    text += "This assessment is intended to assist clinical decision making.\n"
    text += "Always use professional judgment when implementing recommendations.\n"
    text += f"Generated by: Antimicrobial TDM App - {now.strftime('%Y-%m-%d')}"
    
    return text
    def aminoglycoside_synergy_dosing(patient_data, drug_name):
    """Aminoglycoside synergy dosing method (low-dose for synergistic effect)"""
    st.info("Synergy dosing involves lower doses for synergistic effect with other antimicrobials")
    
    # Set target ranges based on drug for synergy
    if drug_name.lower() == "gentamicin":
        initial_dose = 60  # mg fixed for synergy
        target_peak_range = (3,def aminoglycoside_extended_interval(patient_data, drug_name):
    """Aminoglycoside extended interval (once daily) dosing method"""
    st.info("Extended interval (SDD) dosing involves once-daily dosing with optional level monitoring")
    
    # Set target ranges based on drug for SDD
    if drug_name.lower() == "gentamicin":
        initial_dose_per_kg = 5  # mg/kg
        target_peak_range = (10, 30)
        target_trough_range = (0, 1)
        peak_target_str = "10-30 mg/L"
        trough_target_str = "<1 mg/L"
    elif drug_name.lower() == "amikacin":
        initial_dose_per_kg = 15  # mg/kg
        target_peak_range = (60, 80)
        target_trough_range = (0, 1)
        peak_target_str = "60-80 mg/L"
        trough_target_str = "<1 mg/L"
    else:  # Default - gentamicin-like
        initial_dose_per_kg = 5  # mg/kg
        target_peak_range = (10, 30)
        target_trough_range = (0, 1)
        peak_target_str = "10-30 mg/L"
        trough_target_str = "<1 mg/L"
    
    # Weight and renal function
    weight = patient_data.get('weight', 70)
    crcl = patient_data.get('crcl', 90)
    
    # Suggested initial dose based on weight
    suggested_initial_dose = round(weight * initial_dose_per_kg / 10) * 10  # Round to nearest 10mg
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=50, max_value=2000, value=suggested_initial_dose, step=10)
        interval = st.number_input("Dosing Interval (hours)", min_value=12, max_value=48, value=24, step=12)
        
        # Show dose per kg
        dose_per_kg = dose / weight
        st.info(f"Current dose: {dose_per_kg:.1f} mg/kg")
    
    with col2:
        has_levels = st.checkbox("Have measured levels?", value=False)
        
        if has_levels:
            level_time = st.number_input("Time After Start of Dose (hours)", min_value=1.0, max_value=24.0, value=6.0, step=1.0)
            level_value = st.number_input("Measured Level (mg/L)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    
    # Calculate button
    if st.button(f"Calculate {drug_name.capitalize()} SDD Dosing"):
        with st.spinner("Performing calculations..."):
            # Determine interval based on renal function
            suggested_interval = 24  # Default for normal renal function
            
            if crcl < 20:
                suggested_interval = 48
            elif crcl < 40:
                suggested_interval = 36
            
            # Initial calculation based on population parameters
            
            # Estimate pharmacokinetic parameters
            ke_est = 0.00293 * crcl + 0.014  # Estimated Ke based on CrCl
            vd_est = 0.3 * weight  # Estimated Vd based on weight
            
            # Calculate half-life
            t_half_est = 0.693 / ke_est
            
            # Calculate clearance
            cl_est = ke_est * vd_est
            
            # Estimate peak/trough based on population parameters
            peak_est = dose / vd_est
            trough_est = peak_est * math.exp(-ke_est * interval)
            
            # If we have measured levels, refine the estimates
            if has_levels:
                # Estimate Ke from the measured level
                # Assuming infusion over 30 min and sample after distribution phase
                infusion_time = 0.5  # 30 min in hours
                
                # Estimate peak (assuming Vd and first-order elimination)
                estimated_peak = dose / vd_est
                
                # Back-calculate Ke from the measured level
                time_after_peak = level_time - infusion_time
                if time_after_peak > 0:
                    ke_from_level = -math.log(level_value / estimated_peak) / time_after_peak
                    
                    # Use the measured Ke if it seems reasonable
                    if 0.01 <= ke_from_level <= 0.5:
                        ke_est = ke_from_level
                        
                        # Recalculate derived parameters
                        t_half_est = 0.693 / ke_est
                        cl_est = ke_est * vd_est
                        trough_est = level_value * math.exp(-ke_est * (interval - level_time))
            
            # Calculate the Hartford nomogram range (if available)
            hartford_interval = "24h"
            
            if level_time > 5 and level_time < 15:
                if level_value > 10:
                    hartford_interval = "72h (high risk)"
                elif level_value > 6:
                    hartford_interval = "48h" 
                elif level_value > 3:
                    hartford_interval = "36h"
                elif level_value > 2:
                    hartford_interval = "24h"
                else:
                    hartford_interval = "Consider increased dose"
            
            # Calculate new dose to reach target peak
            target_peak = (target_peak_range[0] + target_peak_range[1]) / 2  # Midpoint of target range
            new_dose = target_peak * vd_est
            
            # Round to practical values
            practical_new_dose = round(new_dose / 10) * 10  # Round to nearest 10mg
            
            # Display results in a nice format
            st.success(f"{drug_name.capitalize()} SDD Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if has_levels:
                    st.metric("Measured Level", f"{level_value:.1f} mg/L at {level_time}h")
                    st.metric("Estimated Trough", f"{trough_est:.2f} mg/L")
                else:
                    st.metric("Estimated Peak", f"{peak_est:.1f} mg/L")
                    st.metric("Estimated Trough", f"{trough_est:.2f} mg/L")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke_est:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half_est:.1f} hr")
                st.metric("Clearance", f"{cl_est:.2f} L/hr")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd_est:.1f} L ({vd_est/weight:.2f} L/kg)")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg")
                
                if has_levels:
                    st.metric("Hartford Nomogram", hartford_interval)
                else:
                    st.metric("Recommended Interval", f"{suggested_interval} hr")
            
            # Visualization
            st.subheader("Concentration-Time Curve")
            
            # Plot concentration-time curve
            if has_levels:
                # Estimate peak for plotting
                peak_for_plot = level_value * math.exp(ke_est * (level_time - infusion_time))
                trough_for_plot = trough_est
            else:
                peak_for_plot = peak_est
                trough_for_plot = trough_est
            
            chart = plot_concentration_time_curve(
                f"{drug_name.capitalize()} (SDD method)",
                [], "", [], [], "",
                peak=peak_for_plot, 
                trough=trough_for_plot,
                ke=ke_est,
                tau=interval
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke_est:.4f} hr⁻¹
            t½ = {t_half_est:.1f} hr
            Vd = {vd_est:.1f} L ({vd_est/weight:.2f} L/kg)
            Cl = {cl_est:.2f} L/hr
            """
            
            if has_levels:
                calculation_details += f"""
                Measured level = {level_value:.1f} mg/L at {level_time}h
                Estimated trough = {trough_est:.2f} mg/L
                Hartford nomogram recommendation = {hartford_interval}
                """
            else:
                calculation_details += f"""
                Estimated peak = {peak_est:.1f} mg/L
                Estimated trough = {trough_est:.2f} mg/L
                """
            
            calculation_details += f"""
            Target peak = {peak_target_str}
            Target trough = {trough_target_str}
            Recommended dose = {practical_new_dose:.0f} mg
            Recommended interval = {suggested_interval} hr
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"{drug_name.capitalize()} (SDD): "
            )
            
            if has_levels:
                prompt += f"Measured level = {level_value} mg/L at {level_time}h, "
            else:
                prompt += f"Expected Cmax = {peak_est:.1f} mg/L, Expected Cmin = {trough_est:.2f} mg/L, "
            
            prompt += (
                f"Interval = {interval} hr, Ke = {ke_est:.4f} hr⁻¹, "
                f"Target peak range = {target_peak_range[0]}-{target_peak_range[1]} mg/L, "
                f"Target trough = <{target_trough_range[1]} mg/L, "
                f"Suggested new dose: {practical_new_dose:.0f} mg, "
                f"Suggested new interval: {suggested_interval} hr, "
                f"SDD"
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate interpretation
            interpretation_data = generate_standardized_interpretation(prompt, f"{drug_name.capitalize()}")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            drug_info = f"{drug_name.capitalize()} (SDD method, peak {peak_target_str}, trough {trough_target_str})"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )# ===== AMINOGLYCOSIDE SECTION =====
def aminoglycoside_section(patient_data, drug_name="gentamicin"):
    """Display aminoglycoside-specific input fields and calculations"""
    st.subheader(f"{drug_name.capitalize()} TDM")
    
    # Aminoglycoside Dosing Method
    dosing_method = st.radio(
        "Dosing Method",
        ["Conventional (Multiple Daily Dosing)", "Extended Interval (Once Daily)", "Synergy Dosing"],
        help="Select the dosing approach for aminoglycoside"
    )
    
    # Input fields based on dosing method
    if dosing_method == "Conventional (Multiple Daily Dosing)":
        aminoglycoside_conventional_dosing(patient_data, drug_name)
    elif dosing_method == "Extended Interval (Once Daily)":
        aminoglycoside_extended_interval(patient_data, drug_name)
    else:  # Synergy Dosing
        aminoglycoside_synergy_dosing(patient_data, drug_name)

def aminoglycoside_conventional_dosing(patient_data, drug_name):
    """Aminoglycoside conventional (multiple daily dosing) method"""
    st.info("Conventional (MDD) dosing involves multiple daily doses with peak and trough monitoring")
    
    # Set target ranges based on drug
    if drug_name.lower() == "gentamicin":
        target_peak_range = (5, 10)
        target_trough_range = (0, 2)
        peak_target_str = "5-10 mg/L"
        trough_target_str = "<2 mg/L"
    elif drug_name.lower() == "amikacin":
        target_peak_range = (20, 30)
        target_trough_range = (0, 10)
        peak_target_str = "20-30 mg/L"
        trough_target_str = "<10 mg/L"
    else:  # Default - gentamicin-like
        target_peak_range = (5, 10)
        target_trough_range = (0, 2)
        peak_target_str = "5-10 mg/L"
        trough_target_str = "<2 mg/L"
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=10, max_value=1000, value=80, step=10)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=24, value=8, step=2)
        peak = st.number_input("Measured Peak (mg/L)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
    
    with col2:
        infusion_time = st.number_input("Infusion Duration (minutes)", min_value=15, max_value=60, value=30, step=5) / 60  # Convert to hours
        peak_draw_time = st.number_input("Time After Start of Infusion for Peak (hours)", min_value=0.5, max_value=2.0, value=1.0, step=0.25)
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
    
    # Calculate button
    if st.button(f"Calculate {drug_name.capitalize()} MDD Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters
            
            # Calculate elimination rate constant
            t_peak = peak_draw_time
            tau = interval
            ke = -math.log(trough/peak)/(tau - t_peak)
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Calculate volume of distribution
            weight = patient_data.get('weight', 70)
            
            # Adjusted peak based on timing (if peak drawn after end of infusion)
            if t_peak > infusion_time:
                # Backextrapolate to the end of infusion
                adjusted_peak = peak * math.exp(ke * (t_peak - infusion_time))
            else:
                adjusted_peak = peak
            
            # Calculate Vd using dose and adjusted peak
            vd = dose / adjusted_peak
            
            # Calculate clearance
            cl = ke * vd
            
            # Calculate new dose to reach target peak
            target_peak = (target_peak_range[0] + target_peak_range[1]) / 2  # Midpoint of target range
            new_dose = target_peak * vd
            
            # Calculate new interval to ensure trough below threshold
            target_trough = target_trough_range[1] * 0.8  # Aim slightly below maximum
            new_interval = -math.log(target_trough/target_peak) / ke
            
            # Round to practical values
            practical_new_dose = round(new_dose / 10) * 10  # Round to nearest 10mg
            
            # Round interval to practical values (6, 8, 12, 24 hours)
            practical_intervals = [6, 8, 12, 24]
            practical_new_interval = min(practical_intervals, key=lambda x: abs(x - new_interval))
            
            # Display results in a nice format
            st.success(f"{drug_name.capitalize()} MDD Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Measured Peak", f"{peak:.1f} mg/L")
                st.metric("Measured Trough", f"{trough:.2f} mg/L")
                
                # Show status based on peak and trough
                if peak < target_peak_range[0]:
                    st.warning("⚠️ Peak below target range")
                elif peak > target_peak_range[1]:
                    st.warning("⚠️ Peak above target range")
                else:
                    st.success("✅ Peak within target range")
                
                if trough > target_trough_range[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough below threshold")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Clearance", f"{cl:.2f} L/hr")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd:.1f} L ({vd/weight:.2f} L/kg)")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg")
                st.metric("Recommended Interval", f"{practical_new_interval} hr")
            
            # Visualization
            st.subheader("Concentration-Time Curve")
            
            # Plot concentration-time curve
            chart = plot_concentration_time_curve(
                f"{drug_name.capitalize()} (MDD method)",
                [], "", [], [], "",
                peak=peak, 
                trough=trough,
                ke=ke,
                tau=tau
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd:.1f} L ({vd/weight:.2f} L/kg)
            Cl = {cl:.2f} L/hr
            Current peak = {peak:.1f} mg/L
            Current trough = {trough:.2f} mg/L
            Target peak = {peak_target_str}
            Target trough = {trough_target_str}
            Recommended dose = {practical_new_dose:.0f} mg
            Recommended interval = {practical_new_interval} hr
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"{drug_name.capitalize()} (MDD): Measured peak = {peak} mg/L, trough = {trough} mg/L, "
                f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, "
                f"Target peak range = {target_peak_range[0]}-{target_peak_range[1]} mg/L, "
                f"Target trough = <{target_trough_range[1]} mg/L, "
                f"Suggested new dose: {practical_new_dose:.0f} mg, "
                f"Suggested new interval: {practical_new_interval} hr"
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate interpretation
            interpretation_data = generate_standardized_interpretation(prompt, f"{drug_name.capitalize()}")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            drug_info = f"{drug_name.capitalize()} (MDD method, peak {peak_target_str}, trough {trough_target_str})"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )def vancomycin_auc_guided(patient_data):
    """Vancomycin AUC-guided monitoring method"""
    st.info("AUC-guided monitoring is the preferred approach according to recent guidelines")
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
        first_level = st.number_input("First Concentration (mg/L)", min_value=0.0, max_value=80.0, value=30.0, step=0.5)
        first_time = st.number_input("Time After Start of Infusion for First Sample (hours)", min_value=0.5, max_value=12.0, value=2.0, step=0.5)
    
    with col2:
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        second_level = st.number_input("Second Concentration (mg/L)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
        second_time = st.number_input("Time After Start of Infusion for Second Sample (hours)", min_value=2.0, max_value=24.0, value=8.0, step=0.5)
        
    # Target AUC selection
    target_auc_strategy = st.radio(
        "Target AUC24 Range",
        ["400-600 mg·hr/L (standard infections)", "500-700 mg·hr/L (serious infections)"],
        help="Select appropriate target based on severity of infection"
    )
    
    # Set target AUC range based on selection
    if "400-600" in target_auc_strategy:
        target_auc = (400, 600)
    else:
        target_auc = (500, 700)
    
    # Calculate button
    if st.button("Calculate Vancomycin AUC Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters from two-point sampling
            
            # Calculate elimination rate constant
            delta_time = second_time - first_time
            ke = -math.log(second_level/first_level)/delta_time
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Estimate Cmax (peak) - Assuming first sample is post-distribution
            t_after_infusion = first_time - infusion_time
            if t_after_infusion < 0:
                t_after_infusion = 0  # If first sample is during infusion
            
            estimated_peak = first_level * math.exp(ke * t_after_infusion)
            
            # Estimate Cmin (trough) - Before next dose
            t_to_next_dose = interval - second_time
            estimated_trough = second_level * math.exp(-ke * t_to_next_dose)
            
            # Calculate volume of distribution
            weight = patient_data.get('weight', 70)
            
            # Adjusted for infusion time
            vd_est = dose / (estimated_peak * (1 - math.exp(-ke * infusion_time)))
            
            # Calculate clearance
            cl = ke * vd_est
            
            # Calculate AUC for one dosing interval
            auc_tau = dose / cl
            
            # Calculate AUC24
            auc24 = auc_tau * (24 / interval)
            
            # Calculate new dose to reach target AUC24
            target_auc24 = (target_auc[0] + target_auc[1]) / 2  # Midpoint of target range
            new_dose = (target_auc24 * cl * interval) / 24
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin AUC Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("First Level", f"{first_level:.1f} mg/L at {first_time}h")
                st.metric("Second Level", f"{second_level:.1f} mg/L at {second_time}h")
                st.metric("Estimated Trough", f"{estimated_trough:.1f} mg/L")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Calculated AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd_est:.1f} L ({vd_est/weight:.2f} L/kg)")
                st.metric("Clearance", f"{cl:.2f} L/hr")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{interval}h")
            
            # Show AUC target status
            if auc24 < target_auc[0]:
                st.warning(f"⚠️ AUC24 ({auc24:.1f} mg·hr/L) is below target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            elif auc24 > target_auc[1]:
                st.warning(f"⚠️ AUC24 ({auc24:.1f} mg·hr/L) is above target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            else:
                st.success(f"✅ AUC24 ({auc24:.1f} mg·hr/L) is within target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            
            # Visualization
            st.subheader("Concentration-Time Curve with AUC")
            
            # Create data for visualization
            times = np.linspace(0, interval*1.5, 100)
            concentrations = []
            
            # Calculate concentration at each time point
            for t in times:
                if t <= infusion_time:
                    # During infusion
                    conc = estimated_peak * (t / infusion_time)
                else:
                    # After infusion
                    conc = estimated_peak * math.exp(-ke * (t - infusion_time))
                concentrations.append(conc)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Time (hr)': times,
                'Concentration (mg/L)': concentrations
            })
            
            # Create area chart to visualize AUC
            base = alt.Chart(df).encode(
                x=alt.X('Time (hr)', title='Time (hours)')
            )
            
            # Line for concentration
            line = base.mark_line(color='blue').encode(
                y=alt.Y('Concentration (mg/L)', title='Vancomycin Concentration (mg/L)')
            )
            
            # Area for AUC visualization
            area = base.mark_area(opacity=0.3, color='green').encode(
                y=alt.Y('Concentration (mg/L)', title='Vancomycin Concentration (mg/L)')
            )
            
            # Points for measured levels
            points = alt.Chart(pd.DataFrame({
                'Time (hr)': [first_time, second_time],
                'Concentration (mg/L)': [first_level, second_level],
                'Label': ['First Sample', 'Second Sample']
            })).mark_point(size=100).encode(
                x='Time (hr)',
                y='Concentration (mg/L)',
                color='Label'
            )
            
            # Add AUC target range as text
            target_text = alt.Chart(pd.DataFrame({
                'x': [interval / 2],
                'y': [max(concentrations) * 0.9],
                'text': [f"Target AUC24: {target_auc[0]}-{target_auc[1]} mg·hr/L\nCalculated AUC24: {auc24:.1f} mg·hr/L"]
            })).mark_text(align='center').encode(
                x='x',
                y='y',
                text='text'
            )
            
            # Combine the charts
            chart = (line + area + points + target_text).properties(
                width=600,
                height=400,
                title='Vancomycin Concentration-Time Profile with AUC'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            First level: {first_level:.1f} mg/L at {first_time} hrs
            Second level: {second_level:.1f} mg/L at {second_time} hrs
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd_est:.1f} L ({vd_est/weight:.2f} L/kg)
            Cl = {cl:.2f} L/hr
            Estimated trough = {estimated_trough:.1f} mg/L
            AUC24 = {auc24:.1f} mg·hr/L
            Target AUC = {target_auc[0]}-{target_auc[1]} mg·hr/L
            Recommended dose = {practical_new_dose:.0f} mg q{interval}h
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"Vancomycin (AUC-guided): First level = {first_level} mg/L at {first_time}h, "
                f"Second level = {second_level} mg/L at {second_time}h, "
                f"Interval = {interval} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                f"Target AUC range = {target_auc[0]}-{target_auc[1]} mg·hr/L, "
                f"Recommended base dose = {practical_new_dose:.0f} mg."
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Create AUC-specific levels data
            levels_data = [
                ("AUC24", f"{auc24:.1f} mg·hr/L", f"{target_auc[0]}-{target_auc[1]} mg·hr/L", 
                 "within" if target_auc[0] <= auc24 <= target_auc[1] else "below" if auc24 < target_auc[0] else "above"),
                ("Estimated Trough", f"{estimated_trough:.1f} mg/L", "10-20 mg/L", 
                 "within" if 10 <= estimated_trough <= 20 else "below" if estimated_trough < 10 else "above")
            ]
            
            # Determine assessment based on AUC
            if auc24 < target_auc[0]:
                assessment = "subtherapeutic (low AUC)"
                dosing_recs = [
                    f"INCREASE dose to {practical_new_dose} mg every {interval} hours",
                    "REASSESS AUC after 3-4 doses (steady state)",
                    "CONSIDER more frequent monitoring in critical infections"
                ]
                monitoring_recs = [
                    "REPEAT two-level sampling after dose adjustment",
                    "MONITOR renal function every 2-3 days",
                    "ASSESS clinical response daily"
                ]
                cautions = [
                    "Subtherapeutic exposure may lead to treatment failure",
                    "AUC-guided monitoring is preferred for serious MRSA infections"
                ]
            elif auc24 > target_auc[1]:
                assessment = "potentially supratherapeutic (high AUC)"
                dosing_recs = [
                    f"DECREASE dose to {practical_new_dose} mg every {interval} hours",
                    "REASSESS AUC after 3-4 doses",
                    "CONSIDER extending interval if nephrotoxicity risk is high"
                ]
                monitoring_recs = [
                    "MONITOR renal function daily",
                    "REPEAT two-level sampling after 3-4 doses",
                    "ASSESS for signs of nephrotoxicity"
                ]
                cautions = [
                    "Risk of nephrotoxicity increases with AUC > 700 mg·hr/L",
                    "Consider patient-specific risk factors for nephrotoxicity"
                ]
            else:
                assessment = "appropriately dosed (AUC-based)"
                dosing_recs = [
                    "CONTINUE current dosing regimen",
                    f"MAINTAIN dose of {practical_new_dose} mg every {interval} hours",
                    "REASSESS if renal function changes"
                ]
                monitoring_recs = [
                    "MONITOR renal function every 2-3 days",
                    "REPEAT AUC calculation if clinical status changes",
                    "ASSESS clinical response regularly"
                ]
                cautions = [
                    "Even with therapeutic AUC, monitor for nephrotoxicity",
                    "Consider more frequent monitoring in critically ill patients"
                ]
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            drug_info = f"Vancomycin (AUC-guided method, Target {target_auc[0]}-{target_auc[1]} mg·hr/L)"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )def vancomycin_peak_trough(patient_data):
    """Vancomycin peak and trough monitoring method"""
    st.info("Peak and trough monitoring provides better insight into vancomycin pharmacokinetics")
    
    # Target trough selection
    target_trough_strategy = st.radio(
        "Target Trough Range",
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"],
        help="Select appropriate target based on indication"
    )
    
    # Set target ranges based on selection
    if "Empirical" in target_trough_strategy:
        target_cmin = (10, 15)
        target_peak = (20, 30)
    else:
        target_cmin = (15, 20)
        target_peak = (25, 40)
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
        peak = st.number_input("Measured Peak (mg/L)", min_value=5.0, max_value=80.0, value=25.0, step=0.5)
    
    with col2:
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        peak_draw_time = st.number_input("Time After Start of Infusion for Peak (hours)", min_value=0.5, max_value=6.0, value=1.5, step=0.5)
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=50.0, value=12.5, step=0.5)
    
    # Calculate button
    if st.button("Calculate Vancomycin Peak-Trough Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters based on peak and trough
            
            # Calculate elimination rate constant
            t_peak = peak_draw_time
            tau = interval
            ke = -math.log(trough/peak)/(tau - t_peak)
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Calculate volume of distribution
            weight = patient_data.get('weight', 70)
            
            # Adjusted peak based on timing (if peak drawn after end of infusion)
            if t_peak > infusion_time:
                # Backextrapolate to the end of infusion
                adjusted_peak = peak * math.exp(ke * (t_peak - infusion_time))
            else:
                adjusted_peak = peak
            
            # Calculate Vd using the adjusted peak
            vd = dose / adjusted_peak
            
            # Calculate clearance
            cl = ke * vd
            
            # Calculate AUC24
            auc24 = (dose * 24) / (cl * tau)
            
            # Calculate new dose to reach target trough
            target_trough = (target_cmin[0] + target_cmin[1]) / 2  # Midpoint of target range
            new_dose = (target_trough * cl * tau) / (24/tau)
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin Peak-Trough Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Measured Peak", f"{peak:.1f} mg/L")
                st.metric("Measured Trough", f"{trough:.1f} mg/L")
                
                # Show status based on peak and trough
                if peak < target_peak[0]:
                    st.warning("⚠️ Peak below target range")
                elif peak > target_peak[1]:
                    st.warning("⚠️ Peak above target range")
                else:
                    st.success("✅ Peak within target range")
                
                if trough < target_cmin[0]:
                    st.warning("⚠️ Trough below target range")
                elif trough > target_cmin[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough within target range")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Est. AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd:.1f} L ({vd/weight:.2f} L/kg)")
                st.metric("Clearance", f"{cl:.2f} L/hr")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{tau}h")
            
            # Visualization
            st.subheader("Concentration-Time Curve")
            
            # Plot concentration-time curve
            chart = plot_concentration_time_curve(
                "Vancomycin (Peak-Trough method)",
                [], "", [], [], "",
                peak=peak, 
                trough=trough,
                ke=ke,
                tau=tau
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd:.1f} L ({vd/weight:.2f} L/kg)
            Cl = {cl:.2f} L/hr
            Current peak = {peak:.1f} mg/L
            Current trough = {trough:.1f} mg/L
            Target peak = {target_peak[0]}-{target_peak[1]} mg/L
            Target trough = {target_cmin[0]}-{target_cmin[1]} mg/L
            Recommended dose = {practical_new_dose:.0f} mg q{tau}h
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"Vancomycin (Peak and Trough): Measured peak = {peak} mg/L, trough = {trough} mg/L, "
                f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                f"Target peak range = {target_peak[0]}-{target_peak[1]} mg/L, Recommended base dose = {practical_new_dose:.0f} mg."
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate interpretation
            interpretation_data = generate_standardized_interpretation(prompt, "Vancomycin")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            if "Empirical" in target_trough_strategy:
                drug_info = "Vancomycin (Peak and Trough method, Empirical dosing 10-15 mg/L)"
            else:
                drug_info = "Vancomycin (Peak and Trough method, Definitive dosing 15-20 mg/L)"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )# ===== MAIN APP LAYOUT =====
def main():
    """Main application layout and functionality"""
    st.title("🧪 Advanced Antimicrobial TDM Calculator")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📋 Patient TDM", "📈 PK Analysis", "📚 References"])
    
    with tab1:
        # Collect patient information
        patient_data = display_patient_info_section()
        
        # Select antimicrobial
        st.header("Antimicrobial Selection")
        antimicrobial = st.selectbox(
            "Select Antimicrobial", 
            ["Vancomycin", "Gentamicin", "Amikacin", "Other Aminoglycoside"]
        )
        
        # Conditionally display appropriate input fields based on selection
        if "Vancomycin" in antimicrobial:
            vancomycin_section(patient_data)
        elif any(drug in antimicrobial for drug in ["Gentamicin", "Amikacin", "Aminoglycoside"]):
            aminoglycoside_section(patient_data, drug_name=antimicrobial.lower())
        else:
            st.info("Please select an antimicrobial agent")
    
    with tab2:
        pharmacokinetic_analysis_section()
    
    with tab3:
        display_references()

# ===== VANCOMYCIN SECTION =====
def vancomycin_section(patient_data):
    """Display vancomycin-specific input fields and calculations"""
    st.subheader("Vancomycin TDM")
    
    # Vancomycin Monitoring Method
    monitoring_method = st.radio(
        "Monitoring Method",
        ["Trough-only", "Peak and Trough", "AUC-guided"],
        help="Select the monitoring approach for vancomycin"
    )
    
    # Input fields based on monitoring method
    if monitoring_method == "Trough-only":
        vancomycin_trough_only(patient_data)
    elif monitoring_method == "Peak and Trough":
        vancomycin_peak_trough(patient_data)
    else:  # AUC-guided
        vancomycin_auc_guided(patient_data)

def vancomycin_trough_only(patient_data):
    """Vancomycin trough-only monitoring method"""
    st.info("Trough-only monitoring is a traditional approach for vancomycin dosing")
    
    # Target trough selection
    target_trough_strategy = st.radio(
        "Target Trough Range",
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"],
        help="Select appropriate target based on indication"
    )
    
    # Set target trough range based on selection
    if "Empirical" in target_trough_strategy:
        target_cmin = (10, 15)
    else:
        target_cmin = (15, 20)
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
    
    with col2:
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=50.0, value=12.5, step=0.5)
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    
    # Add timing details
    timing_info = st.checkbox("Add Timing Information")
    if timing_info:
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Date of Last Dose", placeholder="YYYY-MM-DD")
            st.text_input("Time of Last Dose", placeholder="HH:MM")
        with col2:
            st.text_input("Date of Blood Sample", placeholder="YYYY-MM-DD")
            st.text_input("Time of Blood Sample", placeholder="HH:MM")
    
    # Calculate button
    if st.button("Calculate Vancomycin Trough Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters
            # Assume one-compartment model for simplicity
            
            # Determine patient CrCl
            crcl = patient_data.get('crcl', 100)
            weight = patient_data.get('weight', 70)
            
            # Estimate Ke based on renal function
            ke = 0.00083 * crcl + 0.0044
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Estimate volume of distribution (standard population value)
            vd = 0.7 * weight
            
            # Calculate trough concentration at steady state
            tau = interval
            
            # Calculate clearance
            cl = ke * vd
            
            # Calculate AUC24
            auc24 = (dose * 24) / (cl * tau)
            
            # Estimate peak concentration (simple model)
            peak = (dose / vd) * (1 - math.exp(-ke * infusion_time))
            
            # Calculate new dose to reach target trough
            target_trough = (target_cmin[0] + target_cmin[1]) / 2  # Midpoint of target range
            new_dose = (target_trough * cl * tau) / (24/tau)
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin Trough Analysis Complete")
            
            # Create three columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Trough", f"{trough:.1f} mg/L")
                st.metric("Target Trough", f"{target_cmin[0]}-{target_cmin[1]} mg/L")
                
                # Show status icon based on trough
                if trough < target_cmin[0]:
                    st.warning("⚠️ Trough below target range")
                elif trough > target_cmin[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough within target range")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Est. AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Est. Peak", f"{peak:.1f} mg/L")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg")
                st.metric("Volume of Distribution", f"{vd:.1f} L")
            
            # Create detailed calculation steps in an expander
            with st.expander("Show Calculation Details", expanded=False):
                st.write("### Pharmacokinetic Calculations")
                st.write(f"""
                **Patient Parameters:**
                - Weight: {weight} kg
                - CrCl: {crcl:.1f} mL/min
                
                **Estimated PK Parameters:**
                - Ke = 0.00083 × CrCl + 0.0044
                - Ke = 0.00083 × {crcl:.1f} + 0.0044 = {ke:.4f} hr⁻¹
                - t½ = 0.693 / Ke = 0.693 / {ke:.4f} = {t_half:.1f} hr
                - Vd = 0.7 × Weight = 0.7 × {weight} = {vd:.1f} L
                - Cl = Ke × Vd = {ke:.4f} × {vd:.1f} = {cl:.2f} L/hr
                
                **Dose Calculations:**
                - Current dose: {dose} mg every {tau} hr
                - Current trough: {trough:.1f} mg/L
                - Target trough: {target_trough:.1f} mg/L
                - New dose = (Target × Cl × Tau) / (24/Tau)
                - New dose = ({target_trough:.1f} × {cl:.2f} × {tau}) / (24/{tau})
                - New dose = {new_dose:.1f} mg
                - Practical dose: {practical_new_dose:.0f} mg
                
                **AUC Calculation:**
                - AUC24 = (Dose × 24) / (Cl × Tau)
                - AUC24 = ({dose} × 24) / ({cl:.2f} × {tau})
                - AUC24 = {auc24:.1f} mg·hr/L
                """)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd:.1f} L
            Cl = {cl:.2f} L/hr
            Current trough = {trough:.1f} mg/L
            Target trough = {target_cmin[0]}-{target_cmin[1]} mg/L
            Recommended dose = {practical_new_dose:.0f} mg q{tau}h
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"Vancomycin (Trough only): Measured trough = {trough} mg/L, "
                f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                f"Recommended base dose = {practical_new_dose:.0f} mg."
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate and display interpretation
            interpretation_data = generate_standardized_interpretation(prompt, "Vancomycin")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            if "Empirical" in target_trough_strategy:
                drug_info = "Vancomycin (Trough-only method, Empirical dosing 10-15 mg/L)"
            else:
                drug_info = "Vancomycin (Trough-only method, Definitive dosing 15-20 mg/L)"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )# ===== STANDARDIZED INTERPRETATION GENERATOR =====
def generate_standardized_interpretation(prompt, drug):
    """
    Generate a standardized interpretation based on drug type and prompt content
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt)
    elif "Aminoglycoside" in drug or "Gentamicin" in drug or "Amikacin" in drug:
        return generate_aminoglycoside_interpretation(prompt)
    else:
        # For generic, we'll create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None, calculation_details=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM with improved recommendation formatting
    and PDF printing capability
    
    This function can call the OpenAI API if configured, otherwise
    it will provide a simulated response with a standardized, clinically relevant format.
    
    Parameters:
    - prompt: The clinical data prompt
    - patient_data: Optional dictionary with patient information for PDF generation
    - calculation_details: Optional string with calculation details for PDF
    """
    # Extract the drug type from the prompt
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
        if "Trough only" in prompt:
            method = "Trough-only method"
        else:
            method = "Peak and Trough method"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
        if "Initial Dose" in prompt:
            method = "Initial dosing"
        else:
            method = "Conventional (C1/C2) method"
    else:
        drug = "Antimicrobial"
        method = "Standard method"
    
    drug_info = f"{drug} ({method})"
    
    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Format your response with these exact sections:
            
            ## CLINICAL ASSESSMENT
            📊 **MEASURED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **ASSESSMENT:** (state if appropriately dosed, underdosed, or overdosed)
            
            ## RECOMMENDATIONS
            🔵 **DOSING:** (action-oriented recommendations using verbs like CONTINUE, ADJUST, HOLD)
            🔵 **MONITORING:** (specific monitoring parameters and schedule)
            ⚠️ **CAUTIONS:** (relevant warnings, if any)
            
            Here is the case: {prompt}
            """
            
            # Call OpenAI API - updated for openai v1.0.0+
            response = openai.chat.completions.create(
                model="gpt-4",  # or your preferred model
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear recommendations."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            llm_response = response.choices[0].message.content
            st.write(llm_response)
            
            # Add a note about source
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")
            
            # We can't easily extract the structured data from the LLM response for PDF generation
            # So we'll skip the PDF option for the OpenAI path for now
            return
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
    
    # Format the standardized clinical interpretation
    interpretation_data = generate_standardized_interpretation(prompt, drug)
    
    # If the interpretation_data is a string (error message), just display it and return
    if isinstance(interpretation_data, str):
        st.write(interpretation_data)
        return
    
    # Unpack the interpretation data
    levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
    
    # Display the formatted interpretation
    formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
    st.write(formatted_interpretation)
    
    # Add the PDF download button if patient_data is provided
    if patient_data:
        display_pdf_download_button(
            patient_data, 
            drug_info, 
            levels_data, 
            assessment, 
            dosing_recs, 
            monitoring_recs, 
            calculation_details,
            cautions
        )
    
    # Add the raw prompt at the bottom for debugging
    with st.expander("Raw Analysis Data", expanded=False):
        st.code(prompt)
        
    # Add note about simulated response
    st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")# ===== VANCOMYCIN INTERPRETATION FUNCTION =====
def generate_vancomycin_interpretation(prompt):
    """
    Generate standardized vancomycin interpretation
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    # Extract key values from the prompt
    peak_val = None
    trough_val = None
    auc24 = None
    
    # Extract peak and trough values
    if "Peak" in prompt:
        parts = prompt.split("Peak")
        if len(parts) > 1:
            peak_parts = parts[1].split("mg/L")
            if peak_parts:
                try:
                    peak_str = peak_parts[0].replace("=", "").replace(":", "").strip()
                    peak_val = float(peak_str)
                except ValueError:
                    pass
    
    if "Trough" in prompt:
        parts = prompt.split("Trough")
        if len(parts) > 1:
            trough_parts = parts[1].split("mg/L")
            if trough_parts:
                try:
                    trough_str = trough_parts[0].replace("=", "").replace(":", "").strip()
                    trough_val = float(trough_str)
                except ValueError:
                    pass
    
    # Extract AUC if available
    if "AUC24" in prompt:
        parts = prompt.split("AUC24")
        if len(parts) > 1:
            auc_parts = parts[1].split("mg·hr/L")
            if auc_parts:
                try:
                    auc_str = auc_parts[0].replace("=", "").replace(":", "").strip()
                    auc24 = float(auc_str)
                except ValueError:
                    pass
    
    # Extract trough target range
    trough_target_min, trough_target_max = 10, 20  # Default range
    if "Target trough range" in prompt:
        parts = prompt.split("Target trough range")
        if len(parts) > 1:
            range_parts = parts[1].strip().split("mg/L")
            if range_parts:
                try:
                    range_str = range_parts[0].replace("=", "").replace(":", "").strip()
                    if "-" in range_str:
                        min_max = range_str.split("-")
                        trough_target_min = float(min_max[0])
                        trough_target_max = float(min_max[1])
                except ValueError:
                    pass
    
    # Determine if empiric or definitive therapy based on trough target
    if trough_target_max <= 15:
        regimen = "Empiric"
        trough_target = f"{trough_target_min}-{trough_target_max} mg/L"
    else:
        regimen = "Definitive"
        trough_target = f"{trough_target_min}-{trough_target_max} mg/L"
    
    # Set AUC target based on indication
    if regimen == "Empiric":
        auc_target = "400-600 mg·hr/L"
        auc_min, auc_max = 400, 600
    else:  # Definitive
        auc_target = "400-800 mg·hr/L"
        auc_min, auc_max = 400, 800
    
    # Define peak target range
    peak_target = "20-40 mg/L"  # Typical peak range
    peak_min, peak_max = 20, 40
    
    # Determine vancomycin status
    status = "assessment not available"
    
    # If using trough-only monitoring
    if trough_val is not None and peak_val is None and auc24 is None:
        if trough_val < trough_target_min:
            status = "subtherapeutic (low trough)"
        elif trough_val > trough_target_max:
            status = "potentially supratherapeutic (high trough)"
        else:
            status = "appropriately dosed (trough-based)"
    
    # If using peak and trough monitoring
    elif trough_val is not None and peak_val is not None:
        if peak_val < peak_min and trough_val < trough_target_min:
            status = "subtherapeutic (inadequate peak and trough)"
        elif peak_val < peak_min:
            status = "potential underdosing (low peak)"
        elif trough_val < trough_target_min:
            status = "subtherapeutic (inadequate trough)"
        elif trough_val > trough_target_max:
            status = "potentially supratherapeutic (high trough)"
        elif peak_val > peak_max:
            status = "potentially supratherapeutic (high peak)"
        elif peak_min <= peak_val <= peak_max and trough_target_min <= trough_val <= trough_target_max:
            status = "appropriately dosed"
        else:
            status = "requires adjustment"
    
    # If using AUC monitoring
    elif auc24 is not None:
        if auc24 < auc_min:
            status = "subtherapeutic (low AUC)"
        elif auc24 > auc_max:
            status = "potentially supratherapeutic (high AUC)"
        else:
            status = "appropriately dosed (AUC-based)"
    
    # Create levels data based on available measurements
    levels_data = []
    
    if peak_val is not None:
        if peak_val < peak_min:
            peak_status = "below"
        elif peak_val > peak_max:
            peak_status = "above"
        else:
            peak_status = "within"
        levels_data.append(("Peak", f"{peak_val:.1f} mg/L", peak_target, peak_status))
    
    if trough_val is not None:
        if trough_val < trough_target_min:
            trough_status = "below"
        elif trough_val > trough_target_max:
            trough_status = "above"
        else:
            trough_status = "within"
        levels_data.append(("Trough", f"{trough_val:.1f} mg/L", trough_target, trough_status))
    
    if auc24 is not None:
        if auc24 < auc_min:
            auc_status = "below"
        elif auc24 > auc_max:
            auc_status = "above"
        else:
            auc_status = "within"
        levels_data.append(("AUC24", f"{auc24:.1f} mg·hr/L", auc_target, auc_status))
    
    # Generate recommendations based on status
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    
    # Check if we have enough data to provide recommendations
    if not levels_data:
        return "Insufficient data to generate interpretation. At least one measurement (peak, trough, or AUC) is required."
    
    # Extract new dose if available
    new_dose = None
    if "Recommended base dose" in prompt:
        parts = prompt.split("Recommended base dose")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose_str = dose_parts[0].replace("=", "").replace(":", "").strip()
                    new_dose = float(dose_str)
                except ValueError:
                    pass
    
    # Format new dose
    rounded_new_dose = None
    if new_dose:
        # Round to nearest 250mg for vancomycin
        rounded_new_dose = round(new_dose / 250) * 250
    
    # Generate recommendations based on status
    if status == "subtherapeutic (low trough)" or status == "subtherapeutic (inadequate trough)" or status == "subtherapeutic (low AUC)":
        if rounded_new_dose:
            dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("INCREASE dose by 25-30%")
        dosing_recs.append("CONSIDER shortening dosing interval")
        
        monitoring_recs.append("RECHECK levels after 3-4 doses (at steady state)")
        monitoring_recs.append("MONITOR renal function regularly")
        
        cautions.append("Subtherapeutic levels may lead to treatment failure")
        cautions.append("Ensure adequate hydration when increasing doses")
    
    elif status == "potentially supratherapeutic (high trough)" or status == "potentially supratherapeutic (high AUC)":
        if rounded_new_dose:
            dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("DECREASE dose by 20-25%")
        dosing_recs.append("CONSIDER extending dosing interval")
        
        monitoring_recs.append("RECHECK levels after 3-4 doses")
        monitoring_recs.append("MONITOR renal function closely")
        monitoring_recs.append("ASSESS for signs of nephrotoxicity")
        
        cautions.append("Risk of nephrotoxicity with elevated trough levels")
        cautions.append("Consider patient-specific risk factors for toxicity")
    
    elif status == "subtherapeutic (inadequate peak and trough)" or status == "potential underdosing (low peak)":
        if rounded_new_dose:
            dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("INCREASE dose by 30-40%")
        
        monitoring_recs.append("RECHECK peak and trough levels after 3-4 doses")
        monitoring_recs.append("VERIFY correct timing of sample collection")
        
        cautions.append("Significantly subtherapeutic levels increase risk of treatment failure")
        cautions.append("Consider evaluating for altered pharmacokinetics")
    
    elif status == "potentially supratherapeutic (high peak)":
        if rounded_new_dose:
            dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("DECREASE dose by 15-20%")
        dosing_recs.append("EXTEND dosing interval if appropriate")
        
        monitoring_recs.append("RECHECK levels after adjustment")
        monitoring_recs.append("MONITOR renal function")
        
        cautions.append("Risk of nephrotoxicity with excessive dosing")
    
    elif "appropriately dosed" in status:
        dosing_recs.append("CONTINUE current dosing regimen")
        
        monitoring_recs.append("MONITOR renal function regularly")
        monitoring_recs.append("REASSESS levels if clinical status changes")
        
        cautions.append("Even with therapeutic levels, monitor for adverse effects")
    
    else:  # requires adjustment
        if rounded_new_dose:
            dosing_recs.append(f"ADJUST dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("ADJUST dosing based on clinical response and levels")
        
        monitoring_recs.append("RECHECK levels after adjustment")
        monitoring_recs.append("MONITOR renal function")
        
        cautions.append("Individualize therapy based on clinical response")
    
    # Add standard monitoring recommendations
    if "MONITOR renal function" not in " ".join(monitoring_recs):
        monitoring_recs.append("MONITOR renal function every 2-3 days")
    
    return levels_data, assessment, dosing_recs, monitoring_recs, cautions# ===== AMINOGLYCOSIDE INTERPRETATION FUNCTION =====
def generate_aminoglycoside_interpretation(prompt):
    """
    Generate standardized aminoglycoside interpretation
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    # Extract key values from the prompt
    drug_name = "aminoglycoside"
    peak_val = None
    trough_val = None
    
    if "Gentamicin" in prompt:
        drug_name = "gentamicin"
    elif "Amikacin" in prompt:
        drug_name = "amikacin"
    
    # Extract peak and trough values
    if "Cmax:" in prompt:
        parts = prompt.split("Cmax:")
        if len(parts) > 1:
            peak_parts = parts[1].split(",")
            if peak_parts:
                try:
                    peak_val = float(peak_parts[0])
                except ValueError:
                    pass
    elif "Expected Cmax" in prompt:
        parts = prompt.split("Expected Cmax")
        if len(parts) > 1:
            peak_parts = parts[1].split("mg/L")
            if peak_parts:
                try:
                    peak_val = float(peak_parts[0].replace(":", "").strip())
                except ValueError:
                    pass
    
    if "Cmin:" in prompt:
        parts = prompt.split("Cmin:")
        if len(parts) > 1:
            trough_parts = parts[1].split(",")
            if trough_parts:
                try:
                    trough_val = float(trough_parts[0])
                except ValueError:
                    pass
    elif "Expected Cmin" in prompt:
        parts = prompt.split("Expected Cmin")
        if len(parts) > 1:
            trough_parts = parts[1].split("mg/L")
            if trough_parts:
                try:
                    trough_val = float(trough_parts[0].replace(":", "").strip())
                except ValueError:
                    pass
    
    # Extract dose
    dose = None
    if "Dose:" in prompt:
        parts = prompt.split("Dose:")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose = float(dose_parts[0])
                except ValueError:
                    pass
    elif "Dose " in prompt:
        parts = prompt.split("Dose ")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose = float(dose_parts[0])
                except ValueError:
                    pass
    
    # Extract suggested new dose
    new_dose = None
    if "Suggested new dose:" in prompt:
        parts = prompt.split("Suggested new dose:")
        if len(parts) > 1:
            new_dose_parts = parts[1].split("mg")
            if new_dose_parts:
                try:
                    new_dose = float(new_dose_parts[0])
                except ValueError:
                    pass
    elif "Recommended" in prompt and "Dose" in prompt:
        parts = prompt.split("Recommended")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    # Extract the number from this string
                    import re
                    numbers = re.findall(r'\d+', dose_parts[0])
                    if numbers:
                        new_dose = float(numbers[0])
                except ValueError:
                    pass
    
    # Extract target values based on regimen mention
    regimen = None
    if "SDD" in prompt:
        regimen = "SDD"
    elif "Synergy" in prompt:
        regimen = "Synergy"
    elif "MDD" in prompt:
        regimen = "MDD"
    
    # Set target ranges based on drug
    if drug_name == "gentamicin":
        if regimen == "SDD":
            peak_target = "10-30 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 10, 30
            trough_max = 1
        elif regimen == "Synergy":
            peak_target = "3-5 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 3, 5
            trough_max = 1
        else:  # Default to MDD
            peak_target = "5-10 mg/L"
            trough_target = "<2 mg/L"
            peak_min, peak_max = 5, 10
            trough_max = 2
    elif drug_name == "amikacin":
        if regimen == "SDD":
            peak_target = "60-80 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 60, 80
            trough_max = 1
        else:  # Default to MDD
            peak_target = "20-30 mg/L"
            trough_target = "<10 mg/L"
            peak_min, peak_max = 20, 30
            trough_max = 10
    else:
        peak_target = "varies by drug"
        trough_target = "varies by drug"
        peak_min, peak_max = 0, 100
        trough_max = 10
    
    # Determine aminoglycoside status
    status = "assessment not available"
    if peak_val and trough_val:
        if peak_val < peak_min and trough_val > trough_max:
            status = "ineffective and potentially toxic"
        elif peak_val < peak_min:
            status = "subtherapeutic (inadequate peak)"
        elif trough_val > trough_max:
            status = "potentially toxic (elevated trough)"
        elif peak_min <= peak_val <= peak_max and trough_val <= trough_max:
            status = "appropriately dosed"
        elif peak_val > peak_max:
            status = "potentially toxic (elevated peak)"
        else:
            status = "requires adjustment"
    
    # Format new dose
    rounded_new_dose = None
    if new_dose:
        # Round to nearest 10mg for most aminoglycosides
        rounded_new_dose = round(new_dose / 10) * 10
    
    # Create interpretation using standardized format
    if peak_val is not None and trough_val is not None:
        # Determine peak status
        if peak_val < peak_min:
            peak_status = "below"
        elif peak_val > peak_max:
            peak_status = "above"
        else:
            peak_status = "within"
        
        # Determine trough status
        if trough_val > trough_max:
            trough_status = "above"
        else:
            trough_status = "within"
        
        # Prepare data for standardized format
        levels_data = [
            (f"Peak", f"{peak_val:.1f} mg/L", peak_target, peak_status),
            (f"Trough", f"{trough_val:.2f} mg/L", trough_target, trough_status)
        ]
        
        # Generate recommendations based on status
        dosing_recs = []
        monitoring_recs = []
        cautions = []
        
        if status == "ineffective and potentially toxic":
            dosing_recs.append("HOLD next dose")
            dosing_recs.append("REASSESS renal function before resuming")
            if rounded_new_dose:
                dosing_recs.append(f"DECREASE to {rounded_new_dose}mg when resumed")
            dosing_recs.append("EXTEND dosing interval significantly")
            
            monitoring_recs.append("CHECK renal function before resuming therapy")
            monitoring_recs.append("RECHECK levels 2 doses after resumption")
            monitoring_recs.append("MONITOR for signs of ototoxicity and nephrotoxicity")
            
            cautions.append("Risk of nephrotoxicity and ototoxicity with elevated trough levels")
            cautions.append("Ineffective therapy may lead to treatment failure")
            
        elif status == "subtherapeutic (inadequate peak)":
            if rounded_new_dose:
                dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
            else:
                dosing_recs.append("INCREASE dose by 25-50%")
            
            monitoring_recs.append("RECHECK levels after 2-3 doses")
            monitoring_recs.append("VERIFY correct timing of peak sample collection")
            
            cautions.append("Subtherapeutic levels may lead to treatment failure")
            cautions.append("Consider other factors affecting drug disposition")
            
        elif status == "potentially toxic (elevated trough)":
            dosing_recs.append("EXTEND dosing interval")
            if rounded_new_dose:
                dosing_recs.append(f"CONSIDER dose reduction to {rounded_new_dose}mg")
            
            monitoring_recs.append("MONITOR renal function closely")
            monitoring_recs.append("RECHECK levels before next dose")
            monitoring_recs.append("ASSESS for signs of ototoxicity and nephrotoxicity")
            
            cautions.append("Risk of nephrotoxicity and ototoxicity with elevated trough levels")
            cautions.append("Consider patient-specific risk factors for toxicity")
            
        elif status == "potentially toxic (elevated peak)":
            if rounded_new_dose:
                dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
            else:
                dosing_recs.append("DECREASE dose by 20-25%")
            
            monitoring_recs.append("RECHECK levels after 2-3 doses")
            monitoring_recs.append("VERIFY correct timing of peak sample collection")
            monitoring_recs.append("MONITOR for signs of ototoxicity")
            
            cautions.append("Risk of ototoxicity with significantly elevated peak levels")
            
        elif status == "appropriately dosed":
            dosing_recs.append("CONTINUE current dosing regimen")
            
            monitoring_recs.append("MONITOR renal function regularly")
            monitoring_recs.append("REASSESS levels if clinical status changes")
            monitoring_recs.append("CONSIDER extended interval dosing for longer therapy")
            
            cautions.append("Even with therapeutic levels, monitor for adverse effects")
            
        else:  # requires adjustment
            dosing_recs.append("ADJUST dosing based on clinical response")
            if rounded_new_dose:
                dosing_recs.append(f"CONSIDER dose of {rounded_new_dose}mg")
            
            monitoring_recs.append("RECHECK levels after adjustment")
            monitoring_recs.append("MONITOR renal function")
            
            cautions.append("Individualize therapy based on clinical response")
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions
    else:
        return "Insufficient data to generate interpretation. Both peak and trough levels are required."# ===== FORMAT_CLINICAL_RECOMMENDATIONS FUNCTION =====
def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy
    
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
    levels_md = "📊 **MEASURED LEVELS:**\n"
    for name, value, target, status in levels_data:
        icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴"
        levels_md += f"- {name}: {value} (Target: {target}) {icon}\n"
    
    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is {assessment.upper()}"
    
    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## DETAILED RECOMMENDATIONS

🔵 **DOSING RECOMMENDATIONS:**
"""
    for rec in dosing_recs:
        output += f"- {rec}\n"
    
    output += "\n🔵 **MONITORING RECOMMENDATIONS:**\n"
    for rec in monitoring_recs:
        output += f"- {rec}\n"
    
    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS & CONSIDERATIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"
    
    # Add a summary section for quick reference
    output += "\n## QUICK SUMMARY\n"
    output += "**Status:** " + assessment.upper() + "\n"
    
    # Summarize key recommendations
    if len(dosing_recs) > 0:
        output += f"**Key Dosing Action:** {dosing_recs[0]}\n"
    
    if len(monitoring_recs) > 0:
        output += f"**Key Monitoring Action:** {monitoring_recs[0]}\n"
        
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    output += f"\n*Generated on: {timestamp}*"
    
    return outputimport streamlit as st
import numpy as np
import math
import pandas as pd
import altair as alt
import base64
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

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
    # Check for OpenAI API key
    import openai
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError, ImportError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.
    
    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== PATIENT INFO SECTION =====
def display_patient_info_section():
    """Display and collect patient information"""
    st.header("Patient Information")
    
    # Create a 2x2 grid for patient info
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", help="Enter the patient's unique identifier")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        serum_cr = st.number_input("Serum Creatinine (μmol/L)", min_value=10, max_value=1000, value=80)
    
    with col2:
        ward = st.text_input("Ward/Unit", help="Enter the patient's current location")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=40, max_value=250, value=170)
        clinical_diagnosis = st.text_input("Clinical Diagnosis", "Sepsis")
    
    # Calculate Creatinine Clearance
    # Cockcroft-Gault equation
    scr_mg = serum_cr / 88.4  # Convert μmol/L to mg/dL
    if gender == "Male":
        crcl = ((140 - age) * weight) / (72 * scr_mg)
    else:
        crcl = ((140 - age) * weight * 0.85) / (72 * scr_mg)
    
    # Determine renal function category
    if crcl >= 90:
        renal_function = "Normal renal function"
    elif crcl >= 60:
        renal_function = "Mild renal impairment"
    elif crcl >= 30:
        renal_function = "Moderate renal impairment"
    elif crcl >= 15:
        renal_function = "Severe renal impairment"
    else:
        renal_function = "Renal failure"
    
    # Display calculated CrCl and renal function
    st.metric("Estimated CrCl", f"{crcl:.1f} mL/min", renal_function)
    
    # Current medication regimen
    current_dose_regimen = st.text_input("Current Dosing Regimen", "Vancomycin 1000mg q12h")
    
    st.info(f"Patient {patient_id} is in {ward} with {renal_function.lower()} (CrCl: {crcl:.1f} mL/min)")
    
    # Return patient data as a dictionary
    return {
        'patient_id': patient_id,
        'ward': ward,
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen
    }

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details, peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization
    
    Parameters:
    - drug_info: String with drug name
    - levels_data: List of level data
    - assessment: Assessment string
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - calculation_details: String with calculation details
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
            conc = trough + (peak - trough) * (t / infusion_time)
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
    if "Vancomycin" in drug_info:  # Vancomycin
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
    elif "Gentamicin" in drug_info:  # Gentamicin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [10], 'y2': [30]  # Peak range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [5], 'y2': [10]  # Peak range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [2]  # Trough range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    elif "Amikacin" in drug_info:  # Amikacin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [60], 'y2': [80]  # Peak range for amikacin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for amikacin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [20], 'y2': [30]  # Peak range for amikacin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [10]  # Trough range for amikacin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    else:  # Default or unknown drug
        target_peak_band = alt.Chart(pd.DataFrame({
            'y1': [peak*0.8], 'y2': [peak*1.2]  # Default peak range ±20%
        })).mark_rect(opacity=0.2, color='green').encode(
            y='y1', y2='y2'
        )
        target_trough_band = alt.Chart(pd.DataFrame({
            'y1': [trough*0.5], 'y2': [trough*1.5]  # Default trough range ±50%
        })).mark_rect(opacity=0.2, color='blue').encode(
            y='y1', y2='y2'
        )
    
    # Create the concentration-time curve
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)')
    )
    
    # Add markers for actual measured peak and trough
    markers = alt.Chart(pd.DataFrame({
        'Time (hr)': [t_peak, tau],
        'Concentration (mg/L)': [peak, trough],
        'Label': ['Peak', 'Trough']
    })).mark_point(size=100, filled=True).encode(
        x='Time (hr)',
        y='Concentration (mg/L)',
        color=alt.Color('Label', scale=alt.Scale(domain=['Peak', 'Trough'], range=['green', 'blue']))
    )
    
    # Add vertical lines for key time points
    infusion_end = alt.Chart(pd.DataFrame({'x': [infusion_time]})).mark_rule(
        strokeDash=[5, 5], color='gray'
    ).encode(x='x')
    
    next_dose = alt.Chart(pd.DataFrame({'x': [tau]})).mark_rule(
        strokeDash=[5, 5], color='red'
    ).encode(x='x')
    
    # Add text annotations for key time points
    annotations = alt.Chart(pd.DataFrame({
        'x': [infusion_time/2, tau],
        'y': [peak*1.1, trough*0.9],
        'text': ['Infusion', 'Next Dose']
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Calculate half-life and display it
    half_life = 0.693 / ke
    half_life_text = alt.Chart(pd.DataFrame({
        'x': [tau/2],
        'y': [peak*0.5],
        'text': [f"t½ = {half_life:.1f} hr"]
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Combine charts
    chart = alt.layer(
        target_peak_band, 
        target_trough_band, 
        line, 
        markers,
        infusion_end,
        next_dose,
        annotations,
        half_life_text
    ).properties(
        width=600,
        height=400,
        title=f'{drug_info} Concentration-Time Profile'
    )
    
    # Display detailed calculation steps in an expander
    with st.expander("View Calculation Details", expanded=False):
        st.markdown("### PK Parameter Calculations")
        st.markdown(f"""
        **Key Parameters:**
        - Peak concentration (Cmax): {peak:.2f} mg/L
        - Trough concentration (Cmin): {trough:.2f} mg/L
        - Elimination rate constant (Ke): {ke:.4f} hr⁻¹
        - Half-life (t½): {half_life:.2f} hr
        - Dosing interval (τ): {tau} hr
        
        **Detailed Calculations:**
        ```
        Ke = -ln(Cmin/Cmax)/(τ - tpeak)
        Ke = -ln({trough:.2f}/{peak:.2f})/({tau} - {t_peak})
        Ke = {ke:.4f} hr⁻¹
        
        t½ = 0.693/Ke
        t½ = 0.693/{ke:.4f}
        t½ = {half_life:.2f} hr
        ```
        
        **Assessment:**
        {assessment}
        
        **Dosing Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in dosing_recs]) + """
        
        **Monitoring Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in monitoring_recs]))
        
        if calculation_details:
            st.markdown("**Additional Calculation Information:**")
            st.markdown(calculation_details)
    
    return chart

# ===== PDF GENERATION FUNCTIONS =====
def create_recommendation_pdf(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Create a downloadable PDF with the clinical recommendations
    
    Parameters:
    - patient_data: Dictionary with patient information
    - drug_info: String with drug name and method
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - calculation_details: Optional string with calculation details
    - cautions: Optional list of caution strings
    
    Returns:
    - base64 encoded PDF for download
    """
    # Create an in-memory PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom styles
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading3'],
        spaceAfter=6,
        textColor=colors.navy
    )
    
    # Create the content
    content = []
    
    # Add report title
    content.append(Paragraph("Antimicrobial TDM Report", title_style))
    content.append(Spacer(1, 12))
    
    # Add date and time
    now = datetime.now()
    content.append(Paragraph(f"Report Generated: {now.strftime('%Y-%m-%d %H:%M')}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add patient information
    content.append(Paragraph("Patient Information", heading_style))
    
    # Create patient info table with ID and Ward
    patient_info = []
    
    # Add patient ID and ward row
    patient_info.append([
        Paragraph("<b>Patient ID:</b>", normal_style),
        Paragraph(f"{patient_data.get('patient_id', 'N/A')}", normal_style),
        Paragraph("<b>Ward:</b>", normal_style),
        Paragraph(f"{patient_data.get('ward', 'N/A')}", normal_style)
    ])
    
    # First row
    patient_info.append([
        Paragraph("<b>Age:</b>", normal_style),
        Paragraph(f"{patient_data.get('age', 'N/A')} years", normal_style),
        Paragraph("<b>Gender:</b>", normal_style),
        Paragraph(f"{patient_data.get('gender', 'N/A')}", normal_style)
    ])
    
    # Second row
    patient_info.append([
        Paragraph("<b>Weight:</b>", normal_style),
        Paragraph(f"{patient_data.get('weight', 'N/A')} kg", normal_style),
        Paragraph("<b>Height:</b>", normal_style),
        Paragraph(f"{patient_data.get('height', 'N/A')} cm", normal_style)
    ])
    
    # Third row
    patient_info.append([
        Paragraph("<b>Serum Creatinine:</b>", normal_style),
        Paragraph(f"{patient_data.get('serum_cr', 'N/A')} µmol/L", normal_style),
        Paragraph("<b>CrCl:</b>", normal_style),
        Paragraph(f"{patient_data.get('crcl', 'N/A'):.1f} mL/min", normal_style)
    ])
    
    # Fourth row with diagnosis spanning full width
    patient_info.append([
        Paragraph("<b>Diagnosis:</b>", normal_style),
        Paragraph(f"{patient_data.get('clinical_diagnosis', 'N/A')}", normal_style),
        Paragraph("<b>Renal Function:</b>", normal_style),
        Paragraph(f"{patient_data.get('renal_function', 'N/A')}", normal_style)
    ])
    
    # Fifth row with regimen spanning full width
    patient_info.append([
        Paragraph("<b>Current Regimen:</b>", normal_style),
        Paragraph(f"{patient_data.get('current_dose_regimen', 'N/A')}", normal_style),
        Paragraph("", normal_style),
        Paragraph("", normal_style)
    ])
    
    # Create the table
    patient_table = Table(patient_info, colWidths=[100, 150, 100, 150])
    patient_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    content.append(patient_table)
    content.append(Spacer(1, 12))
    
    # Add drug information
    content.append(Paragraph("Drug Information", heading_style))
    content.append(Paragraph(drug_info, normal_style))
    content.append(Spacer(1, 12))
    
    # Add clinical assessment
    content.append(Paragraph("Clinical Assessment", heading_style))
    
    # Add measured levels
    content.append(Paragraph("Measured Levels:", section_style))
    
    # Create levels table
    levels_table_data = [["Parameter", "Value", "Target Range", "Status"]]
    
    for name, value, target, status in levels_data:
        # Determine status text and color
        if status == "within":
            status_text = "Within Range"
            status_color = colors.green
        elif status == "below":
            status_text = "Below Range"
            status_color = colors.orange
        else:  # above
            status_text = "Above Range"
            status_color = colors.red
        
        levels_table_data.append([name, value, target, status_text])
    
    levels_table = Table(levels_table_data)
    levels_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    # Add status color to each row in the table
    for i, (_, _, _, status) in enumerate(levels_data, 1):
        if status == "within":
            color = colors.lightgreen
        elif status == "below":
            color = colors.lightyellow
        else:  # above
            color = colors.mistyrose
        
        levels_table.setStyle(TableStyle([
            ('BACKGROUND', (3, i), (3, i), color),
        ]))
    
    content.append(levels_table)
    content.append(Spacer(1, 8))
    
    # Add assessment
    content.append(Paragraph("Assessment:", section_style))
    content.append(Paragraph(f"Patient is {assessment.upper()}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add calculations section if provided
    if calculation_details:
        content.append(Paragraph("Calculation Details:", section_style))
        content.append(Paragraph(calculation_details, normal_style))
        content.append(Spacer(1, 12))
    
    # Add recommendations
    content.append(Paragraph("Recommendations", heading_style))
    
    # Add dosing recommendations
    content.append(Paragraph("Dosing:", section_style))
    for rec in dosing_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add monitoring recommendations
    content.append(Paragraph("Monitoring:", section_style))
    for rec in monitoring_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add cautions if any
    if cautions and len(cautions) > 0:
        content.append(Paragraph("Cautions:", section_style))
        for caution in cautions:
            content.append(Paragraph(f"• {caution}", normal_style))
    
    # Add disclaimer
    content.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey
    )
    content.append(Paragraph("Disclaimer: This report is generated by an automated system and is intended to assist clinical decision making. Always use professional judgment when implementing recommendations.", disclaimer_style))
    
    # Build the PDF
    doc.build(content)
    
    # Get the PDF value from the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    # Encode the PDF to base64
    pdf_base64 = base64.b64encode(pdf_value).decode()
    
    return pdf_base64

# Function to create a download link for the PDF
def get_pdf_download_link(pdf_base64, filename="clinical_recommendations.pdf"):
    """Create a download link for a base64 encoded PDF"""
    href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="{filename}">Download Clinical Recommendations PDF</a>'
    return href

# Updated function to display buttons for printing and downloading recommendations
def display_pdf_download_button(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Display buttons to print/save recommendations as a PDF and print a summary
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Print/Save Full Report"):
            # Generate the PDF
            pdf_base64 = create_recommendation_pdf(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs,
                calculation_details,
                cautions
            )
            
            # Create the download link
            download_link = get_pdf_download_link(pdf_base64)
            
            # Display the download link
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Preview message
            st.success("PDF generated successfully. Click the link above to download.")
    
    with col2:
        if st.button("🖨️ Print Clinical Summary"):
            # Create a simple text printout of the assessment and recommendations
            assessment_text = create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            
            # Display in a text area that can be easily copied
            st.text_area("Copy this text to print", assessment_text, height=300)
            st.success("Assessment text generated. Copy and paste into your preferred document.")

# Enhanced function to create a printable text assessment
def create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """Create a plain text printable assessment for easy copying to clinical notes"""
    now = datetime.now()
    
    # Header
    text = f"ANTIMICROBIAL TDM ASSESSMENT - {now.strftime('%Y-%m-%d %H:%M')}\n"
    text += "=" * 50 + "\n\n"
    
    # Patient information - Make sure to include ID and ward
    text += f"Patient ID: {patient_data.get('patient_id', 'N/A')}\n"
    text += f"Ward: {patient_data.get('ward', 'N/A')}\n"
    text += f"Age: {patient_data.get('age', 'N/A')} years  |  "
    text += f"Gender: {patient_data.get('gender', 'N/A')}  |  "
    text += f"Weight: {patient_data.get('weight', 'N/A')} kg\n"
    text += f"Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}\n"
    text += f"CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})\n"
    text += f"Current regimen: {patient_data.get('current_dose_regimen', 'N/A')}\n\n"
    
    # Measured levels
    text += "MEASURED LEVELS:\n"
    for name, value, target, status in levels_data:
        status_text = "✓" if status == "within" else "↓" if status == "below" else "↑"
        text += f"- {name}: {value} (Target: {target}) {status_text}\n"
    
    # Assessment
    text += f"\nASSESSMENT: Patient is {assessment.upper()}\n\n"
    
    # PK Parameters (if available from calculation details)
    try:
        if "Half-life" in calculation_details or "t½" in calculation_details:
            text += "PHARMACOKINETIC PARAMETERS:\n"
            # Extract PK parameters from calculation details
            import re
            ke_match = re.search(r'Ke[\s=:]+([0-9.]+)', calculation_details)
            t_half_match = re.search(r't½[\s=:]+([0-9.]+)', calculation_details)
            
            if ke_match:
                ke = float(ke_match.group(1))
                text += f"- Elimination rate constant (Ke): {ke:.4f} hr⁻¹\n"
            
            if t_half_match:
                t_half = float(t_half_match.group(1))
                text += f"- Half-life (t½): {t_half:.2f} hr\n"
            
            text += "\n"
    except:
        pass  # Skip if unable to extract PK parameters
    
    # Recommendations
    text += "DOSING RECOMMENDATIONS:\n"
    for rec in dosing_recs:
        text += f"- {rec}\n"
    
    text += "\nMONITORING RECOMMENDATIONS:\n"
    for rec in monitoring_recs:
        text += f"- {rec}\n"
    
    # Cautions
    if cautions and len(cautions) > 0:
        text += "\nCAUTIONS:\n"
        for caution in cautions:
            text += f"- {caution}\n"
    
    # Footer
    text += "\n" + "=" * 50 + "\n"
    text += "This assessment is intended to assist clinical decision making.\n"
    text += "Always use professional judgment when implementing recommendations.\n"
    text += f"Generated by: Antimicrobial TDM App - {now.strftime('%Y-%m-%d')}"
    
    return text
    # ===== PHARMACOKINETIC ANALYSIS SECTION =====
def pharmacokinetic_analysis_section():
    """Advanced pharmacokinetic analysis tools"""
    st.header("Advanced PK Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Bayesian Parameter Estimation", "Monte Carlo Simulation", "PK Model Comparison"]
    )
    
    if analysis_type == "Bayesian Parameter Estimation":
        st.info("Bayesian estimation allows incorporation of prior information with measured levels")
        
        if not BAYESIAN_AVAILABLE:
            st.error("Bayesian estimation requires the scipy library. Please install it.")
            return
        
        # Placeholder for now
        st.write("This feature is under development.")
        
    elif analysis_type == "Monte Carlo Simulation":
        st.info("Monte Carlo simulation allows evaluation of dosing regimens across virtual patient populations")
        
        # Placeholder for now
        st.write("This feature is under development.")
        
    else:  # PK Model Comparison
        st.info("Compare different PK models (e.g., one vs. two compartment)")
        
        # Placeholder for now
        st.write("This feature is under development.")

# ===== REFERENCES SECTION =====
def display_references():
    """Display important references and guidelines"""
    st.header("References & Guidelines")
    
    st.markdown("""
    ### Vancomycin Monitoring
    * **2020 Consensus Guidelines:** Rybak MJ, et al. "Therapeutic monitoring of vancomycin for serious methicillin-resistant Staphylococcus aureus infections: A revised consensus guideline and review by the American Society of Health-System Pharmacists, the Infectious Diseases Society of America, the Pediatric Infectious Diseases Society, and the Society of Infectious Diseases Pharmacists." Am J Health Syst Pharm. 2020.
    * **AUC Monitoring:** Neely MN, et al. "Are vancomycin trough concentrations adequate for optimal dosing?" Antimicrob Agents Chemother. 2014.
    
    ### Aminoglycoside Monitoring
    * **Extended Interval Dosing:** Nicolau DP, et al. "Experience with a once-daily aminoglycoside program administered to 2,184 adult patients." Antimicrob Agents Chemother. 1995.
    * **Hartford Nomogram:** Nicolau DP, et al. "Experience with a once-daily aminoglycoside program administered to 2,184 adult patients." Antimicrob Agents Chemother. 1995.
    * **Synergy Dosing:** Drusano GL, Louie A. "Optimization of aminoglycoside therapy." Antimicrob Agents Chemother. 2011.
    
    ### General TDM Principles
    * **Clinical Pharmacokinetics:** Bauer LA. "Applied Clinical Pharmacokinetics." 3rd edition. McGraw Hill. 2014.
    * **Population Pharmacokinetics:** Jelliffe RW, et al. "Individualized drug dosage regimens based on population pharmacokinetic models, Bayesian feedback and minimally invasive blood sampling: the MM-USCPACK software." Ther Drug Monit. 1993.
    
    ### Software Development
    * This application was built with Python and Streamlit
    * Pharmacokinetic calculations are based on established principles from clinical pharmacokinetics literature
    * Automated interpretation follows consensus guidelines from major infectious diseases and pharmacy organizations
    """)
    
    st.info("This application is intended as a clinical decision support tool and should be used in conjunction with clinical judgment.")

# ===== APP EXECUTION =====
if __name__ == "__main__":
    main()def aminoglycoside_synergy_dosing(patient_data, drug_name):
    """Aminoglycoside synergy dosing method (low-dose for synergistic effect)"""
    st.info("Synergy dosing involves lower doses for synergistic effect with other antimicrobials")
    
    # Set target ranges based on drug for synergy
    if drug_name.lower() == "gentamicin":
        initial_dose = 60  # mg fixed for synergy
        target_peak_range = (3, 5)
        target_trough_range = (0, 1)
        peak_target_str = "3-5 mg/L"
        trough_target_str = "<1 mg/L"
    else:  # Default/other aminoglycosides not typically used for synergy
        initial_dose = 80  # mg fixed for synergy
        target_peak_range = (3, 5)
        target_trough_range = (0, 1)
        peak_target_str = "3-5 mg/L"
        trough_target_str = "<1 mg/L"
    
    # Weight and renal function
    weight = patient_data.get('weight', 70)
    crcl = patient_data.get('crcl', 90)
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=30, max_value=150, value=initial_dose, step=10)
        interval = st.number_input("Dosing Interval (hours)", min_value=8, max_value=24, value=12, step=4)
        peak = st.number_input("Measured Peak (mg/L)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    
    with col2:
        synergy_with = st.selectbox("Synergy with", ["Penicillin", "Vancomycin", "Other β-lactam"])
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
    
    # Calculate button
    if st.button(f"Calculate {drug_name.capitalize()} Synergy Dosing"):
        with st.spinner("Performing calculations..."):
            # Determine appropriate interval based on renal function
            suggested_interval = 12  # Default interval for synergy
            
            if crcl < 30:
                suggested_interval = 24
            elif crcl < 60:
                suggested_interval = 18
                
            # Calculate pharmacokinetic parameters
            
            # Estimate Ke based on CrCl
            ke_est = 0.00293 * crcl + 0.014
            
            # Calculate half-life
            t_half_est = 0.693 / ke_est
            
            # Estimate volume of distribution
            vd_est = 0.25 * weight  # Lower Vd for synergy dosing
            
            # Calculate clearance
            cl_est = ke_est * vd_est
            
            # If we have measured peak and trough, refine estimates
            if peak > 0 and trough > 0:
                # Calculate Ke from measured levels
                ke_calc = -math.log(trough/peak)/(interval - 1)  # Assuming peak at 1 hour
                
                # Use calculated Ke if reasonable
                if 0.01 <= ke_calc <= 0.5:
                    ke_est = ke_calc
                    t_half_est = 0.693 / ke_est
                    
                    # Estimate Vd from peak
                    vd_est = dose / peak
                    cl_est = ke_est * vd_est
            
            # Calculate new dose to reach target peak
            target_peak = (target_peak_range[0] + target_peak_range[1]) / 2  # Midpoint of target range
            new_dose = target_peak * vd_est
            
            # Round to practical values
            practical_new_dose = round(new_dose / 10) * 10  # Round to nearest 10mg
            
            # Display results in a nice format
            st.success(f"{drug_name.capitalize()} Synergy Dosing Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Measured Peak", f"{peak:.1f} mg/L")
                st.metric("Measured Trough", f"{trough:.2f} mg/L")
                
                # Show status based on peak and trough
                if peak < target_peak_range[0]:
                    st.warning("⚠️ Peak below target range")
                elif peak > target_peak_range[1]:
                    st.warning("⚠️ Peak above target range")
                else:
                    st.success("✅ Peak within target range")
                
                if trough > target_trough_range[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough below threshold")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke_est:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half_est:.1f} hr")
                st.metric("Clearance", f"{cl_est:.2f} L/hr")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd_est:.1f} L ({vd_est/weight:.2f} L/kg)")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg")
                st.metric("Recommended Interval", f"{suggested_interval} hr")
            
            # Visualization
            st.subheader("Concentration-Time Curve")
            
            # Plot concentration-time curve
            chart = plot_concentration_time_curve(
                f"{drug_name.capitalize()} (Synergy dosing)",
                [], "", [], [], "",
                peak=peak, 
                trough=trough,
                ke=ke_est,
                tau=interval
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke_est:.4f} hr⁻¹
            t½ = {t_half_est:.1f} hr
            Vd = {vd_est:.1f} L ({vd_est/weight:.2f} L/kg)
            Cl = {cl_est:.2f} L/hr
            Current peak = {peak:.1f} mg/L
            Current trough = {trough:.2f} mg/L
            Target peak = {peak_target_str}
            Target trough = {trough_target_str}
            Recommended dose = {practical_new_dose:.0f} mg
            Recommended interval = {suggested_interval} hr
            Synergy with {synergy_with}
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"{drug_name.capitalize()} (Synergy): Measured peak = {peak} mg/L, trough = {trough} mg/L, "
                f"Interval = {interval} hr, Ke = {ke_est:.4f} hr⁻¹, "
                f"Target peak range = {target_peak_range[0]}-{target_peak_range[1]} mg/L, "
                f"Target trough = <{target_trough_range[1]} mg/L, "
                f"Suggested new dose: {practical_new_dose:.0f} mg, "
                f"Suggested new interval: {suggested_interval} hr, "
                f"Synergy"
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate interpretation
            interpretation_data = generate_standardized_interpretation(prompt, f"{drug_name.capitalize()}")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Modify recommendations for synergy specific advice
            synergy_dosing_recs = [
                f"CONTINUE {drug_name} for synergistic effect with {synergy_with}",
                f"ADJUST dose to {practical_new_dose} mg every {suggested_interval} hours",
                f"MAINTAIN synergy dosing for duration of therapy"
            ]
            
            synergy_monitoring_recs = [
                "MONITOR renal function every 48 hours",
                "CONSIDER repeat level after 3-4 doses",
                f"VERIFY efficacy of primary agent ({synergy_with})"
            ]
            
            synergy_cautions = [
                "Even at lower doses, aminoglycosides can cause nephrotoxicity",
                "Synergy dosing is NOT for primary therapy of serious gram-negative infections",
                f"Ensure therapeutic levels of {synergy_with}"
            ]
            
            # Display the formatted interpretation with synergy modifications
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, synergy_dosing_recs, synergy_monitoring_recs, synergy_cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            drug_info = f"{drug_name.capitalize()} (Synergy dosing with {synergy_with}, peak {peak_target_str}, trough {trough_target_str})"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                synergy_dosing_recs, 
                synergy_monitoring_recs, 
                calculation_details,
                synergy_cautions
            )def aminoglycoside_extended_interval(patient_data, drug_name):
    """Aminoglycoside extended interval (once daily) dosing method"""
    st.info("Extended interval (SDD) dosing involves once-daily dosing with optional level monitoring")
    
    # Set target ranges based on drug for SDD
    if drug_name.lower() == "gentamicin":
        initial_dose_per_kg = 5  # mg/kg
        target_peak_range = (10, 30)
        target_trough_range = (0, 1)
        peak_target_str = "10-30 mg/L"
        trough_target_str = "<1 mg/L"
    elif drug_name.lower() == "amikacin":
        initial_dose_per_kg = 15  # mg/kg
        target_peak_range = (60, 80)
        target_trough_range = (0, 1)
        peak_target_str = "60-80 mg/L"
        trough_target_str = "<1 mg/L"
    else:  # Default - gentamicin-like
        initial_dose_per_kg = 5  # mg/kg
        target_peak_range = (10, 30)
        target_trough_range = (0, 1)
        peak_target_str = "10-30 mg/L"
        trough_target_str = "<1 mg/L"
    
    # Weight and renal function
    weight = patient_data.get('weight', 70)
    crcl = patient_data.get('crcl', 90)
    
    # Suggested initial dose based on weight
    suggested_initial_dose = round(weight * initial_dose_per_kg / 10) * 10  # Round to nearest 10mg
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=50, max_value=2000, value=suggested_initial_dose, step=10)
        interval = st.number_input("Dosing Interval (hours)", min_value=12, max_value=48, value=24, step=12)
        
        # Show dose per kg
        dose_per_kg = dose / weight
        st.info(f"Current dose: {dose_per_kg:.1f} mg/kg")
    
    with col2:
        has_levels = st.checkbox("Have measured levels?", value=False)
        
        if has_levels:
            level_time = st.number_input("Time After Start of Dose (hours)", min_value=1.0, max_value=24.0, value=6.0, step=1.0)
            level_value = st.number_input("Measured Level (mg/L)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    
    # Calculate button
    if st.button(f"Calculate {drug_name.capitalize()} SDD Dosing"):
        with st.spinner("Performing calculations..."):
            # Determine interval based on renal function
            suggested_interval = 24  # Default for normal renal function
            
            if crcl < 20:
                suggested_interval = 48
            elif crcl < 40:
                suggested_interval = 36
            
            # Initial calculation based on population parameters
            
            # Estimate pharmacokinetic parameters
            ke_est = 0.00293 * crcl + 0.014  # Estimated Ke based on CrCl
            vd_est = 0.3 * weight  # Estimated Vd based on weight
            
            # Calculate half-life
            t_half_est = 0.693 / ke_est
            
            # Calculate clearance
            cl_est = ke_est * vd_est
            
            # Estimate peak/trough based on population parameters
            peak_est = dose / vd_est
            trough_est = peak_est * math.exp(-ke_est * interval)
            
            # If we have measured levels, refine the estimates
            if has_levels:
                # Estimate Ke from the measured level
                # Assuming infusion over 30 min and sample after distribution phase
                infusion_time = 0.5  # 30 min in hours
                
                # Estimate peak (assuming Vd and first-order elimination)
                estimated_peak = dose / vd_est
                
                # Back-calculate Ke from the measured level
                time_after_peak = level_time - infusion_time
                if time_after_peak > 0:
                    ke_from_level = -math.log(level_value / estimated_peak) / time_after_peak
                    
                    # Use the measured Ke if it seems reasonable
                    if 0.01 <= ke_from_level <= 0.5:
                        ke_est = ke_from_level
                        
                        # Recalculate derived parameters
                        t_half_est = 0.693 / ke_est
                        cl_est = ke_est * vd_est
                        trough_est = level_value * math.exp(-ke_est * (interval - level_time))
            
            # Calculate the Hartford nomogram range (if available)
            hartford_interval = "24h"
            
            if level_time > 5 and level_time < 15:
                if level_value > 10:
                    hartford_interval = "72h (high risk)"
                elif level_value > 6:
                    hartford_interval = "48h" 
                elif level_value > 3:
                    hartford_interval = "36h"
                elif level_value > 2:
                    hartford_interval = "24h"
                else:
                    hartford_interval = "Consider increased dose"
            
            # Calculate new dose to reach target peak
            target_peak = (target_peak_range[0] + target_peak_range[1]) / 2  # Midpoint of target range
            new_dose = target_peak * vd_est
            
            # Round to practical values
            practical_new_dose = round(new_dose / 10) * 10  # Round to nearest 10mg
            
            # Display results in a nice format
            st.success(f"{drug_name.capitalize()} SDD Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if has_levels:
                    st.metric("Measured Level", f"{level_value:.1f} mg/L at {level_time}h")
                    st.metric("Estimated Trough", f"{trough_est:.2f} mg/L")
                else:
                    st.metric("Estimated Peak", f"{peak_est:.1f} mg/L")
                    st.metric("Estimated Trough", f"{trough_est:.2f} mg/L")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke_est:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half_est:.1f} hr")
                st.metric("Clearance", f"{cl_est:.2f} L/hr")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd_est:.1f} L ({vd_est/weight:.2f} L/kg)")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg")
                
                if has_levels:
                    st.metric("Hartford Nomogram", hartford_interval)
                else:
                    st.metric("Recommended Interval", f"{suggested_interval} hr")
            
            # Visualization
            st.subheader("Concentration-Time Curve")
            
            # Plot concentration-time curve
            if has_levels:
                # Estimate peak for plotting
                peak_for_plot = level_value * math.exp(ke_est * (level_time - infusion_time))
                trough_for_plot = trough_est
            else:
                peak_for_plot = peak_est
                trough_for_plot = trough_est
            
            chart = plot_concentration_time_curve(
                f"{drug_name.capitalize()} (SDD method)",
                [], "", [], [], "",
                peak=peak_for_plot, 
                trough=trough_for_plot,
                ke=ke_est,
                tau=interval
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke_est:.4f} hr⁻¹
            t½ = {t_half_est:.1f} hr
            Vd = {vd_est:.1f} L ({vd_est/weight:.2f} L/kg)
            Cl = {cl_est:.2f} L/hr
            """
            
            if has_levels:
                calculation_details += f"""
                Measured level = {level_value:.1f} mg/L at {level_time}h
                Estimated trough = {trough_est:.2f} mg/L
                Hartford nomogram recommendation = {hartford_interval}
                """
            else:
                calculation_details += f"""
                Estimated peak = {peak_est:.1f} mg/L
                Estimated trough = {trough_est:.2f} mg/L
                """
            
            calculation_details += f"""
            Target peak = {peak_target_str}
            Target trough = {trough_target_str}
            Recommended dose = {practical_new_dose:.0f} mg
            Recommended interval = {suggested_interval} hr
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"{drug_name.capitalize()} (SDD): "
            )
            
            if has_levels:
                prompt += f"Measured level = {level_value} mg/L at {level_time}h, "
            else:
                prompt += f"Expected Cmax = {peak_est:.1f} mg/L, Expected Cmin = {trough_est:.2f} mg/L, "
            
            prompt += (
                f"Interval = {interval} hr, Ke = {ke_est:.4f} hr⁻¹, "
                f"Target peak range = {target_peak_range[0]}-{target_peak_range[1]} mg/L, "
                f"Target trough = <{target_trough_range[1]} mg/L, "
                f"Suggested new dose: {practical_new_dose:.0f} mg, "
                f"Suggested new interval: {suggested_interval} hr, "
                f"SDD"
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate interpretation
            interpretation_data = generate_standardized_interpretation(prompt, f"{drug_name.capitalize()}")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            drug_info = f"{drug_name.capitalize()} (SDD method, peak {peak_target_str}, trough {trough_target_str})"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )# ===== AMINOGLYCOSIDE SECTION =====
def aminoglycoside_section(patient_data, drug_name="gentamicin"):
    """Display aminoglycoside-specific input fields and calculations"""
    st.subheader(f"{drug_name.capitalize()} TDM")
    
    # Aminoglycoside Dosing Method
    dosing_method = st.radio(
        "Dosing Method",
        ["Conventional (Multiple Daily Dosing)", "Extended Interval (Once Daily)", "Synergy Dosing"],
        help="Select the dosing approach for aminoglycoside"
    )
    
    # Input fields based on dosing method
    if dosing_method == "Conventional (Multiple Daily Dosing)":
        aminoglycoside_conventional_dosing(patient_data, drug_name)
    elif dosing_method == "Extended Interval (Once Daily)":
        aminoglycoside_extended_interval(patient_data, drug_name)
    else:  # Synergy Dosing
        aminoglycoside_synergy_dosing(patient_data, drug_name)

def aminoglycoside_conventional_dosing(patient_data, drug_name):
    """Aminoglycoside conventional (multiple daily dosing) method"""
    st.info("Conventional (MDD) dosing involves multiple daily doses with peak and trough monitoring")
    
    # Set target ranges based on drug
    if drug_name.lower() == "gentamicin":
        target_peak_range = (5, 10)
        target_trough_range = (0, 2)
        peak_target_str = "5-10 mg/L"
        trough_target_str = "<2 mg/L"
    elif drug_name.lower() == "amikacin":
        target_peak_range = (20, 30)
        target_trough_range = (0, 10)
        peak_target_str = "20-30 mg/L"
        trough_target_str = "<10 mg/L"
    else:  # Default - gentamicin-like
        target_peak_range = (5, 10)
        target_trough_range = (0, 2)
        peak_target_str = "5-10 mg/L"
        trough_target_str = "<2 mg/L"
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=10, max_value=1000, value=80, step=10)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=24, value=8, step=2)
        peak = st.number_input("Measured Peak (mg/L)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
    
    with col2:
        infusion_time = st.number_input("Infusion Duration (minutes)", min_value=15, max_value=60, value=30, step=5) / 60  # Convert to hours
        peak_draw_time = st.number_input("Time After Start of Infusion for Peak (hours)", min_value=0.5, max_value=2.0, value=1.0, step=0.25)
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
    
    # Calculate button
    if st.button(f"Calculate {drug_name.capitalize()} MDD Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters
            
            # Calculate elimination rate constant
            t_peak = peak_draw_time
            tau = interval
            ke = -math.log(trough/peak)/(tau - t_peak)
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Calculate volume of distribution
            weight = patient_data.get('weight', 70)
            
            # Adjusted peak based on timing (if peak drawn after end of infusion)
            if t_peak > infusion_time:
                # Backextrapolate to the end of infusion
                adjusted_peak = peak * math.exp(ke * (t_peak - infusion_time))
            else:
                adjusted_peak = peak
            
            # Calculate Vd using dose and adjusted peak
            vd = dose / adjusted_peak
            
            # Calculate clearance
            cl = ke * vd
            
            # Calculate new dose to reach target peak
            target_peak = (target_peak_range[0] + target_peak_range[1]) / 2  # Midpoint of target range
            new_dose = target_peak * vd
            
            # Calculate new interval to ensure trough below threshold
            target_trough = target_trough_range[1] * 0.8  # Aim slightly below maximum
            new_interval = -math.log(target_trough/target_peak) / ke
            
            # Round to practical values
            practical_new_dose = round(new_dose / 10) * 10  # Round to nearest 10mg
            
            # Round interval to practical values (6, 8, 12, 24 hours)
            practical_intervals = [6, 8, 12, 24]
            practical_new_interval = min(practical_intervals, key=lambda x: abs(x - new_interval))
            
            # Display results in a nice format
            st.success(f"{drug_name.capitalize()} MDD Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Measured Peak", f"{peak:.1f} mg/L")
                st.metric("Measured Trough", f"{trough:.2f} mg/L")
                
                # Show status based on peak and trough
                if peak < target_peak_range[0]:
                    st.warning("⚠️ Peak below target range")
                elif peak > target_peak_range[1]:
                    st.warning("⚠️ Peak above target range")
                else:
                    st.success("✅ Peak within target range")
                
                if trough > target_trough_range[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough below threshold")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Clearance", f"{cl:.2f} L/hr")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd:.1f} L ({vd/weight:.2f} L/kg)")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg")
                st.metric("Recommended Interval", f"{practical_new_interval} hr")
            
            # Visualization
            st.subheader("Concentration-Time Curve")
            
            # Plot concentration-time curve
            chart = plot_concentration_time_curve(
                f"{drug_name.capitalize()} (MDD method)",
                [], "", [], [], "",
                peak=peak, 
                trough=trough,
                ke=ke,
                tau=tau
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd:.1f} L ({vd/weight:.2f} L/kg)
            Cl = {cl:.2f} L/hr
            Current peak = {peak:.1f} mg/L
            Current trough = {trough:.2f} mg/L
            Target peak = {peak_target_str}
            Target trough = {trough_target_str}
            Recommended dose = {practical_new_dose:.0f} mg
            Recommended interval = {practical_new_interval} hr
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"{drug_name.capitalize()} (MDD): Measured peak = {peak} mg/L, trough = {trough} mg/L, "
                f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, "
                f"Target peak range = {target_peak_range[0]}-{target_peak_range[1]} mg/L, "
                f"Target trough = <{target_trough_range[1]} mg/L, "
                f"Suggested new dose: {practical_new_dose:.0f} mg, "
                f"Suggested new interval: {practical_new_interval} hr"
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate interpretation
            interpretation_data = generate_standardized_interpretation(prompt, f"{drug_name.capitalize()}")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            drug_info = f"{drug_name.capitalize()} (MDD method, peak {peak_target_str}, trough {trough_target_str})"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )def vancomycin_auc_guided(patient_data):
    """Vancomycin AUC-guided monitoring method"""
    st.info("AUC-guided monitoring is the preferred approach according to recent guidelines")
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
        first_level = st.number_input("First Concentration (mg/L)", min_value=0.0, max_value=80.0, value=30.0, step=0.5)
        first_time = st.number_input("Time After Start of Infusion for First Sample (hours)", min_value=0.5, max_value=12.0, value=2.0, step=0.5)
    
    with col2:
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        second_level = st.number_input("Second Concentration (mg/L)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
        second_time = st.number_input("Time After Start of Infusion for Second Sample (hours)", min_value=2.0, max_value=24.0, value=8.0, step=0.5)
        
    # Target AUC selection
    target_auc_strategy = st.radio(
        "Target AUC24 Range",
        ["400-600 mg·hr/L (standard infections)", "500-700 mg·hr/L (serious infections)"],
        help="Select appropriate target based on severity of infection"
    )
    
    # Set target AUC range based on selection
    if "400-600" in target_auc_strategy:
        target_auc = (400, 600)
    else:
        target_auc = (500, 700)
    
    # Calculate button
    if st.button("Calculate Vancomycin AUC Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters from two-point sampling
            
            # Calculate elimination rate constant
            delta_time = second_time - first_time
            ke = -math.log(second_level/first_level)/delta_time
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Estimate Cmax (peak) - Assuming first sample is post-distribution
            t_after_infusion = first_time - infusion_time
            if t_after_infusion < 0:
                t_after_infusion = 0  # If first sample is during infusion
            
            estimated_peak = first_level * math.exp(ke * t_after_infusion)
            
            # Estimate Cmin (trough) - Before next dose
            t_to_next_dose = interval - second_time
            estimated_trough = second_level * math.exp(-ke * t_to_next_dose)
            
            # Calculate volume of distribution
            weight = patient_data.get('weight', 70)
            
            # Adjusted for infusion time
            vd_est = dose / (estimated_peak * (1 - math.exp(-ke * infusion_time)))
            
            # Calculate clearance
            cl = ke * vd_est
            
            # Calculate AUC for one dosing interval
            auc_tau = dose / cl
            
            # Calculate AUC24
            auc24 = auc_tau * (24 / interval)
            
            # Calculate new dose to reach target AUC24
            target_auc24 = (target_auc[0] + target_auc[1]) / 2  # Midpoint of target range
            new_dose = (target_auc24 * cl * interval) / 24
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin AUC Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("First Level", f"{first_level:.1f} mg/L at {first_time}h")
                st.metric("Second Level", f"{second_level:.1f} mg/L at {second_time}h")
                st.metric("Estimated Trough", f"{estimated_trough:.1f} mg/L")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Calculated AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd_est:.1f} L ({vd_est/weight:.2f} L/kg)")
                st.metric("Clearance", f"{cl:.2f} L/hr")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{interval}h")
            
            # Show AUC target status
            if auc24 < target_auc[0]:
                st.warning(f"⚠️ AUC24 ({auc24:.1f} mg·hr/L) is below target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            elif auc24 > target_auc[1]:
                st.warning(f"⚠️ AUC24 ({auc24:.1f} mg·hr/L) is above target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            else:
                st.success(f"✅ AUC24 ({auc24:.1f} mg·hr/L) is within target range ({target_auc[0]}-{target_auc[1]} mg·hr/L)")
            
            # Visualization
            st.subheader("Concentration-Time Curve with AUC")
            
            # Create data for visualization
            times = np.linspace(0, interval*1.5, 100)
            concentrations = []
            
            # Calculate concentration at each time point
            for t in times:
                if t <= infusion_time:
                    # During infusion
                    conc = estimated_peak * (t / infusion_time)
                else:
                    # After infusion
                    conc = estimated_peak * math.exp(-ke * (t - infusion_time))
                concentrations.append(conc)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Time (hr)': times,
                'Concentration (mg/L)': concentrations
            })
            
            # Create area chart to visualize AUC
            base = alt.Chart(df).encode(
                x=alt.X('Time (hr)', title='Time (hours)')
            )
            
            # Line for concentration
            line = base.mark_line(color='blue').encode(
                y=alt.Y('Concentration (mg/L)', title='Vancomycin Concentration (mg/L)')
            )
            
            # Area for AUC visualization
            area = base.mark_area(opacity=0.3, color='green').encode(
                y=alt.Y('Concentration (mg/L)', title='Vancomycin Concentration (mg/L)')
            )
            
            # Points for measured levels
            points = alt.Chart(pd.DataFrame({
                'Time (hr)': [first_time, second_time],
                'Concentration (mg/L)': [first_level, second_level],
                'Label': ['First Sample', 'Second Sample']
            })).mark_point(size=100).encode(
                x='Time (hr)',
                y='Concentration (mg/L)',
                color='Label'
            )
            
            # Add AUC target range as text
            target_text = alt.Chart(pd.DataFrame({
                'x': [interval / 2],
                'y': [max(concentrations) * 0.9],
                'text': [f"Target AUC24: {target_auc[0]}-{target_auc[1]} mg·hr/L\nCalculated AUC24: {auc24:.1f} mg·hr/L"]
            })).mark_text(align='center').encode(
                x='x',
                y='y',
                text='text'
            )
            
            # Combine the charts
            chart = (line + area + points + target_text).properties(
                width=600,
                height=400,
                title='Vancomycin Concentration-Time Profile with AUC'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            First level: {first_level:.1f} mg/L at {first_time} hrs
            Second level: {second_level:.1f} mg/L at {second_time} hrs
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd_est:.1f} L ({vd_est/weight:.2f} L/kg)
            Cl = {cl:.2f} L/hr
            Estimated trough = {estimated_trough:.1f} mg/L
            AUC24 = {auc24:.1f} mg·hr/L
            Target AUC = {target_auc[0]}-{target_auc[1]} mg·hr/L
            Recommended dose = {practical_new_dose:.0f} mg q{interval}h
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"Vancomycin (AUC-guided): First level = {first_level} mg/L at {first_time}h, "
                f"Second level = {second_level} mg/L at {second_time}h, "
                f"Interval = {interval} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                f"Target AUC range = {target_auc[0]}-{target_auc[1]} mg·hr/L, "
                f"Recommended base dose = {practical_new_dose:.0f} mg."
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Create AUC-specific levels data
            levels_data = [
                ("AUC24", f"{auc24:.1f} mg·hr/L", f"{target_auc[0]}-{target_auc[1]} mg·hr/L", 
                 "within" if target_auc[0] <= auc24 <= target_auc[1] else "below" if auc24 < target_auc[0] else "above"),
                ("Estimated Trough", f"{estimated_trough:.1f} mg/L", "10-20 mg/L", 
                 "within" if 10 <= estimated_trough <= 20 else "below" if estimated_trough < 10 else "above")
            ]
            
            # Determine assessment based on AUC
            if auc24 < target_auc[0]:
                assessment = "subtherapeutic (low AUC)"
                dosing_recs = [
                    f"INCREASE dose to {practical_new_dose} mg every {interval} hours",
                    "REASSESS AUC after 3-4 doses (steady state)",
                    "CONSIDER more frequent monitoring in critical infections"
                ]
                monitoring_recs = [
                    "REPEAT two-level sampling after dose adjustment",
                    "MONITOR renal function every 2-3 days",
                    "ASSESS clinical response daily"
                ]
                cautions = [
                    "Subtherapeutic exposure may lead to treatment failure",
                    "AUC-guided monitoring is preferred for serious MRSA infections"
                ]
            elif auc24 > target_auc[1]:
                assessment = "potentially supratherapeutic (high AUC)"
                dosing_recs = [
                    f"DECREASE dose to {practical_new_dose} mg every {interval} hours",
                    "REASSESS AUC after 3-4 doses",
                    "CONSIDER extending interval if nephrotoxicity risk is high"
                ]
                monitoring_recs = [
                    "MONITOR renal function daily",
                    "REPEAT two-level sampling after 3-4 doses",
                    "ASSESS for signs of nephrotoxicity"
                ]
                cautions = [
                    "Risk of nephrotoxicity increases with AUC > 700 mg·hr/L",
                    "Consider patient-specific risk factors for nephrotoxicity"
                ]
            else:
                assessment = "appropriately dosed (AUC-based)"
                dosing_recs = [
                    "CONTINUE current dosing regimen",
                    f"MAINTAIN dose of {practical_new_dose} mg every {interval} hours",
                    "REASSESS if renal function changes"
                ]
                monitoring_recs = [
                    "MONITOR renal function every 2-3 days",
                    "REPEAT AUC calculation if clinical status changes",
                    "ASSESS clinical response regularly"
                ]
                cautions = [
                    "Even with therapeutic AUC, monitor for nephrotoxicity",
                    "Consider more frequent monitoring in critically ill patients"
                ]
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            drug_info = f"Vancomycin (AUC-guided method, Target {target_auc[0]}-{target_auc[1]} mg·hr/L)"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )def vancomycin_peak_trough(patient_data):
    """Vancomycin peak and trough monitoring method"""
    st.info("Peak and trough monitoring provides better insight into vancomycin pharmacokinetics")
    
    # Target trough selection
    target_trough_strategy = st.radio(
        "Target Trough Range",
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"],
        help="Select appropriate target based on indication"
    )
    
    # Set target ranges based on selection
    if "Empirical" in target_trough_strategy:
        target_cmin = (10, 15)
        target_peak = (20, 30)
    else:
        target_cmin = (15, 20)
        target_peak = (25, 40)
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
        peak = st.number_input("Measured Peak (mg/L)", min_value=5.0, max_value=80.0, value=25.0, step=0.5)
    
    with col2:
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
        peak_draw_time = st.number_input("Time After Start of Infusion for Peak (hours)", min_value=0.5, max_value=6.0, value=1.5, step=0.5)
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=50.0, value=12.5, step=0.5)
    
    # Calculate button
    if st.button("Calculate Vancomycin Peak-Trough Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters based on peak and trough
            
            # Calculate elimination rate constant
            t_peak = peak_draw_time
            tau = interval
            ke = -math.log(trough/peak)/(tau - t_peak)
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Calculate volume of distribution
            weight = patient_data.get('weight', 70)
            
            # Adjusted peak based on timing (if peak drawn after end of infusion)
            if t_peak > infusion_time:
                # Backextrapolate to the end of infusion
                adjusted_peak = peak * math.exp(ke * (t_peak - infusion_time))
            else:
                adjusted_peak = peak
            
            # Calculate Vd using the adjusted peak
            vd = dose / adjusted_peak
            
            # Calculate clearance
            cl = ke * vd
            
            # Calculate AUC24
            auc24 = (dose * 24) / (cl * tau)
            
            # Calculate new dose to reach target trough
            target_trough = (target_cmin[0] + target_cmin[1]) / 2  # Midpoint of target range
            new_dose = (target_trough * cl * tau) / (24/tau)
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin Peak-Trough Analysis Complete")
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Measured Peak", f"{peak:.1f} mg/L")
                st.metric("Measured Trough", f"{trough:.1f} mg/L")
                
                # Show status based on peak and trough
                if peak < target_peak[0]:
                    st.warning("⚠️ Peak below target range")
                elif peak > target_peak[1]:
                    st.warning("⚠️ Peak above target range")
                else:
                    st.success("✅ Peak within target range")
                
                if trough < target_cmin[0]:
                    st.warning("⚠️ Trough below target range")
                elif trough > target_cmin[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough within target range")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Est. AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Volume of Distribution", f"{vd:.1f} L ({vd/weight:.2f} L/kg)")
                st.metric("Clearance", f"{cl:.2f} L/hr")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{tau}h")
            
            # Visualization
            st.subheader("Concentration-Time Curve")
            
            # Plot concentration-time curve
            chart = plot_concentration_time_curve(
                "Vancomycin (Peak-Trough method)",
                [], "", [], [], "",
                peak=peak, 
                trough=trough,
                ke=ke,
                tau=tau
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd:.1f} L ({vd/weight:.2f} L/kg)
            Cl = {cl:.2f} L/hr
            Current peak = {peak:.1f} mg/L
            Current trough = {trough:.1f} mg/L
            Target peak = {target_peak[0]}-{target_peak[1]} mg/L
            Target trough = {target_cmin[0]}-{target_cmin[1]} mg/L
            Recommended dose = {practical_new_dose:.0f} mg q{tau}h
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"Vancomycin (Peak and Trough): Measured peak = {peak} mg/L, trough = {trough} mg/L, "
                f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                f"Target peak range = {target_peak[0]}-{target_peak[1]} mg/L, Recommended base dose = {practical_new_dose:.0f} mg."
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate interpretation
            interpretation_data = generate_standardized_interpretation(prompt, "Vancomycin")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            if "Empirical" in target_trough_strategy:
                drug_info = "Vancomycin (Peak and Trough method, Empirical dosing 10-15 mg/L)"
            else:
                drug_info = "Vancomycin (Peak and Trough method, Definitive dosing 15-20 mg/L)"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )# ===== MAIN APP LAYOUT =====
def main():
    """Main application layout and functionality"""
    st.title("🧪 Advanced Antimicrobial TDM Calculator")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📋 Patient TDM", "📈 PK Analysis", "📚 References"])
    
    with tab1:
        # Collect patient information
        patient_data = display_patient_info_section()
        
        # Select antimicrobial
        st.header("Antimicrobial Selection")
        antimicrobial = st.selectbox(
            "Select Antimicrobial", 
            ["Vancomycin", "Gentamicin", "Amikacin", "Other Aminoglycoside"]
        )
        
        # Conditionally display appropriate input fields based on selection
        if "Vancomycin" in antimicrobial:
            vancomycin_section(patient_data)
        elif any(drug in antimicrobial for drug in ["Gentamicin", "Amikacin", "Aminoglycoside"]):
            aminoglycoside_section(patient_data, drug_name=antimicrobial.lower())
        else:
            st.info("Please select an antimicrobial agent")
    
    with tab2:
        pharmacokinetic_analysis_section()
    
    with tab3:
        display_references()

# ===== VANCOMYCIN SECTION =====
def vancomycin_section(patient_data):
    """Display vancomycin-specific input fields and calculations"""
    st.subheader("Vancomycin TDM")
    
    # Vancomycin Monitoring Method
    monitoring_method = st.radio(
        "Monitoring Method",
        ["Trough-only", "Peak and Trough", "AUC-guided"],
        help="Select the monitoring approach for vancomycin"
    )
    
    # Input fields based on monitoring method
    if monitoring_method == "Trough-only":
        vancomycin_trough_only(patient_data)
    elif monitoring_method == "Peak and Trough":
        vancomycin_peak_trough(patient_data)
    else:  # AUC-guided
        vancomycin_auc_guided(patient_data)

def vancomycin_trough_only(patient_data):
    """Vancomycin trough-only monitoring method"""
    st.info("Trough-only monitoring is a traditional approach for vancomycin dosing")
    
    # Target trough selection
    target_trough_strategy = st.radio(
        "Target Trough Range",
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"],
        help="Select appropriate target based on indication"
    )
    
    # Set target trough range based on selection
    if "Empirical" in target_trough_strategy:
        target_cmin = (10, 15)
    else:
        target_cmin = (15, 20)
    
    # Current regimen details
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=3000, value=1000, step=250)
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=48, value=12, step=6)
    
    with col2:
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=50.0, value=12.5, step=0.5)
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    
    # Add timing details
    timing_info = st.checkbox("Add Timing Information")
    if timing_info:
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Date of Last Dose", placeholder="YYYY-MM-DD")
            st.text_input("Time of Last Dose", placeholder="HH:MM")
        with col2:
            st.text_input("Date of Blood Sample", placeholder="YYYY-MM-DD")
            st.text_input("Time of Blood Sample", placeholder="HH:MM")
    
    # Calculate button
    if st.button("Calculate Vancomycin Trough Dosing"):
        with st.spinner("Performing calculations..."):
            # Calculate pharmacokinetic parameters
            # Assume one-compartment model for simplicity
            
            # Determine patient CrCl
            crcl = patient_data.get('crcl', 100)
            weight = patient_data.get('weight', 70)
            
            # Estimate Ke based on renal function
            ke = 0.00083 * crcl + 0.0044
            
            # Calculate half-life
            t_half = 0.693 / ke
            
            # Estimate volume of distribution (standard population value)
            vd = 0.7 * weight
            
            # Calculate trough concentration at steady state
            tau = interval
            
            # Calculate clearance
            cl = ke * vd
            
            # Calculate AUC24
            auc24 = (dose * 24) / (cl * tau)
            
            # Estimate peak concentration (simple model)
            peak = (dose / vd) * (1 - math.exp(-ke * infusion_time))
            
            # Calculate new dose to reach target trough
            target_trough = (target_cmin[0] + target_cmin[1]) / 2  # Midpoint of target range
            new_dose = (target_trough * cl * tau) / (24/tau)
            
            # Round to nearest practical dose
            practical_new_dose = round(new_dose / 250) * 250
            
            # Display results in a nice format
            st.success("Vancomycin Trough Analysis Complete")
            
            # Create three columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Trough", f"{trough:.1f} mg/L")
                st.metric("Target Trough", f"{target_cmin[0]}-{target_cmin[1]} mg/L")
                
                # Show status icon based on trough
                if trough < target_cmin[0]:
                    st.warning("⚠️ Trough below target range")
                elif trough > target_cmin[1]:
                    st.warning("⚠️ Trough above target range")
                else:
                    st.success("✅ Trough within target range")
            
            with col2:
                st.metric("Elimination Rate (Ke)", f"{ke:.4f} hr⁻¹")
                st.metric("Half-life (t½)", f"{t_half:.1f} hr")
                st.metric("Est. AUC24", f"{auc24:.1f} mg·hr/L")
            
            with col3:
                st.metric("Est. Peak", f"{peak:.1f} mg/L")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg")
                st.metric("Volume of Distribution", f"{vd:.1f} L")
            
            # Create detailed calculation steps in an expander
            with st.expander("Show Calculation Details", expanded=False):
                st.write("### Pharmacokinetic Calculations")
                st.write(f"""
                **Patient Parameters:**
                - Weight: {weight} kg
                - CrCl: {crcl:.1f} mL/min
                
                **Estimated PK Parameters:**
                - Ke = 0.00083 × CrCl + 0.0044
                - Ke = 0.00083 × {crcl:.1f} + 0.0044 = {ke:.4f} hr⁻¹
                - t½ = 0.693 / Ke = 0.693 / {ke:.4f} = {t_half:.1f} hr
                - Vd = 0.7 × Weight = 0.7 × {weight} = {vd:.1f} L
                - Cl = Ke × Vd = {ke:.4f} × {vd:.1f} = {cl:.2f} L/hr
                
                **Dose Calculations:**
                - Current dose: {dose} mg every {tau} hr
                - Current trough: {trough:.1f} mg/L
                - Target trough: {target_trough:.1f} mg/L
                - New dose = (Target × Cl × Tau) / (24/Tau)
                - New dose = ({target_trough:.1f} × {cl:.2f} × {tau}) / (24/{tau})
                - New dose = {new_dose:.1f} mg
                - Practical dose: {practical_new_dose:.0f} mg
                
                **AUC Calculation:**
                - AUC24 = (Dose × 24) / (Cl × Tau)
                - AUC24 = ({dose} × 24) / ({cl:.2f} × {tau})
                - AUC24 = {auc24:.1f} mg·hr/L
                """)
            
            # Generate clinical interpretation
            calculation_details = f"""
            Ke = {ke:.4f} hr⁻¹
            t½ = {t_half:.1f} hr
            Vd = {vd:.1f} L
            Cl = {cl:.2f} L/hr
            Current trough = {trough:.1f} mg/L
            Target trough = {target_cmin[0]}-{target_cmin[1]} mg/L
            Recommended dose = {practical_new_dose:.0f} mg q{tau}h
            """
            
            # Generate the clinical interpretation prompt
            prompt = (
                f"Vancomycin (Trough only): Measured trough = {trough} mg/L, "
                f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.1f} mg·hr/L, "
                f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                f"Recommended base dose = {practical_new_dose:.0f} mg."
            )
            
            # Display professional recommendation
            st.subheader("Clinical Interpretation")
            
            # Generate and display interpretation
            interpretation_data = generate_standardized_interpretation(prompt, "Vancomycin")
            
            # If the interpretation_data is a string (error message), just display it and return
            if isinstance(interpretation_data, str):
                st.write(interpretation_data)
                return
            
            # Unpack the interpretation data
            levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
            
            # Display the formatted interpretation
            formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            st.write(formatted_interpretation)
            
            # Get drug info
            if "Empirical" in target_trough_strategy:
                drug_info = "Vancomycin (Trough-only method, Empirical dosing 10-15 mg/L)"
            else:
                drug_info = "Vancomycin (Trough-only method, Definitive dosing 15-20 mg/L)"
            
            # Add PDF and print buttons
            display_pdf_download_button(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs, 
                calculation_details,
                cautions
            )# ===== STANDARDIZED INTERPRETATION GENERATOR =====
def generate_standardized_interpretation(prompt, drug):
    """
    Generate a standardized interpretation based on drug type and prompt content
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    if drug == "Vancomycin":
        return generate_vancomycin_interpretation(prompt)
    elif "Aminoglycoside" in drug or "Gentamicin" in drug or "Amikacin" in drug:
        return generate_aminoglycoside_interpretation(prompt)
    else:
        # For generic, we'll create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt, patient_data=None, calculation_details=None):
    """
    Enhanced clinical interpretation function for antimicrobial TDM with improved recommendation formatting
    and PDF printing capability
    
    This function can call the OpenAI API if configured, otherwise
    it will provide a simulated response with a standardized, clinically relevant format.
    
    Parameters:
    - prompt: The clinical data prompt
    - patient_data: Optional dictionary with patient information for PDF generation
    - calculation_details: Optional string with calculation details for PDF
    """
    # Extract the drug type from the prompt
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
        if "Trough only" in prompt:
            method = "Trough-only method"
        else:
            method = "Peak and Trough method"
    elif "Aminoglycoside" in prompt:
        drug = "Aminoglycoside"
        if "Initial Dose" in prompt:
            method = "Initial dosing"
        else:
            method = "Conventional (C1/C2) method"
    else:
        drug = "Antimicrobial"
        method = "Standard method"
    
    drug_info = f"{drug} ({method})"
    
    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            # Updated prompt to guide the LLM to provide structured outputs
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Format your response with these exact sections:
            
            ## CLINICAL ASSESSMENT
            📊 **MEASURED LEVELS:** (list each with target range and status icon ✅⚠️🔴)
            ⚕️ **ASSESSMENT:** (state if appropriately dosed, underdosed, or overdosed)
            
            ## RECOMMENDATIONS
            🔵 **DOSING:** (action-oriented recommendations using verbs like CONTINUE, ADJUST, HOLD)
            🔵 **MONITORING:** (specific monitoring parameters and schedule)
            ⚠️ **CAUTIONS:** (relevant warnings, if any)
            
            Here is the case: {prompt}
            """
            
            # Call OpenAI API - updated for openai v1.0.0+
            response = openai.chat.completions.create(
                model="gpt-4",  # or your preferred model
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in therapeutic drug monitoring. Provide concise, evidence-based interpretations with clear recommendations."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            llm_response = response.choices[0].message.content
            st.write(llm_response)
            
            # Add a note about source
            st.info("Interpretation provided by OpenAI GPT-4. Always verify with clinical judgment.")
            
            # We can't easily extract the structured data from the LLM response for PDF generation
            # So we'll skip the PDF option for the OpenAI path for now
            return
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to simulated clinical interpretation.")
    
    # Format the standardized clinical interpretation
    interpretation_data = generate_standardized_interpretation(prompt, drug)
    
    # If the interpretation_data is a string (error message), just display it and return
    if isinstance(interpretation_data, str):
        st.write(interpretation_data)
        return
    
    # Unpack the interpretation data
    levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data
    
    # Display the formatted interpretation
    formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
    st.write(formatted_interpretation)
    
    # Add the PDF download button if patient_data is provided
    if patient_data:
        display_pdf_download_button(
            patient_data, 
            drug_info, 
            levels_data, 
            assessment, 
            dosing_recs, 
            monitoring_recs, 
            calculation_details,
            cautions
        )
    
    # Add the raw prompt at the bottom for debugging
    with st.expander("Raw Analysis Data", expanded=False):
        st.code(prompt)
        
    # Add note about simulated response
    st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")# ===== VANCOMYCIN INTERPRETATION FUNCTION =====
def generate_vancomycin_interpretation(prompt):
    """
    Generate standardized vancomycin interpretation
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    # Extract key values from the prompt
    peak_val = None
    trough_val = None
    auc24 = None
    
    # Extract peak and trough values
    if "Peak" in prompt:
        parts = prompt.split("Peak")
        if len(parts) > 1:
            peak_parts = parts[1].split("mg/L")
            if peak_parts:
                try:
                    peak_str = peak_parts[0].replace("=", "").replace(":", "").strip()
                    peak_val = float(peak_str)
                except ValueError:
                    pass
    
    if "Trough" in prompt:
        parts = prompt.split("Trough")
        if len(parts) > 1:
            trough_parts = parts[1].split("mg/L")
            if trough_parts:
                try:
                    trough_str = trough_parts[0].replace("=", "").replace(":", "").strip()
                    trough_val = float(trough_str)
                except ValueError:
                    pass
    
    # Extract AUC if available
    if "AUC24" in prompt:
        parts = prompt.split("AUC24")
        if len(parts) > 1:
            auc_parts = parts[1].split("mg·hr/L")
            if auc_parts:
                try:
                    auc_str = auc_parts[0].replace("=", "").replace(":", "").strip()
                    auc24 = float(auc_str)
                except ValueError:
                    pass
    
    # Extract trough target range
    trough_target_min, trough_target_max = 10, 20  # Default range
    if "Target trough range" in prompt:
        parts = prompt.split("Target trough range")
        if len(parts) > 1:
            range_parts = parts[1].strip().split("mg/L")
            if range_parts:
                try:
                    range_str = range_parts[0].replace("=", "").replace(":", "").strip()
                    if "-" in range_str:
                        min_max = range_str.split("-")
                        trough_target_min = float(min_max[0])
                        trough_target_max = float(min_max[1])
                except ValueError:
                    pass
    
    # Determine if empiric or definitive therapy based on trough target
    if trough_target_max <= 15:
        regimen = "Empiric"
        trough_target = f"{trough_target_min}-{trough_target_max} mg/L"
    else:
        regimen = "Definitive"
        trough_target = f"{trough_target_min}-{trough_target_max} mg/L"
    
    # Set AUC target based on indication
    if regimen == "Empiric":
        auc_target = "400-600 mg·hr/L"
        auc_min, auc_max = 400, 600
    else:  # Definitive
        auc_target = "400-800 mg·hr/L"
        auc_min, auc_max = 400, 800
    
    # Define peak target range
    peak_target = "20-40 mg/L"  # Typical peak range
    peak_min, peak_max = 20, 40
    
    # Determine vancomycin status
    status = "assessment not available"
    
    # If using trough-only monitoring
    if trough_val is not None and peak_val is None and auc24 is None:
        if trough_val < trough_target_min:
            status = "subtherapeutic (low trough)"
        elif trough_val > trough_target_max:
            status = "potentially supratherapeutic (high trough)"
        else:
            status = "appropriately dosed (trough-based)"
    
    # If using peak and trough monitoring
    elif trough_val is not None and peak_val is not None:
        if peak_val < peak_min and trough_val < trough_target_min:
            status = "subtherapeutic (inadequate peak and trough)"
        elif peak_val < peak_min:
            status = "potential underdosing (low peak)"
        elif trough_val < trough_target_min:
            status = "subtherapeutic (inadequate trough)"
        elif trough_val > trough_target_max:
            status = "potentially supratherapeutic (high trough)"
        elif peak_val > peak_max:
            status = "potentially supratherapeutic (high peak)"
        elif peak_min <= peak_val <= peak_max and trough_target_min <= trough_val <= trough_target_max:
            status = "appropriately dosed"
        else:
            status = "requires adjustment"
    
    # If using AUC monitoring
    elif auc24 is not None:
        if auc24 < auc_min:
            status = "subtherapeutic (low AUC)"
        elif auc24 > auc_max:
            status = "potentially supratherapeutic (high AUC)"
        else:
            status = "appropriately dosed (AUC-based)"
    
    # Create levels data based on available measurements
    levels_data = []
    
    if peak_val is not None:
        if peak_val < peak_min:
            peak_status = "below"
        elif peak_val > peak_max:
            peak_status = "above"
        else:
            peak_status = "within"
        levels_data.append(("Peak", f"{peak_val:.1f} mg/L", peak_target, peak_status))
    
    if trough_val is not None:
        if trough_val < trough_target_min:
            trough_status = "below"
        elif trough_val > trough_target_max:
            trough_status = "above"
        else:
            trough_status = "within"
        levels_data.append(("Trough", f"{trough_val:.1f} mg/L", trough_target, trough_status))
    
    if auc24 is not None:
        if auc24 < auc_min:
            auc_status = "below"
        elif auc24 > auc_max:
            auc_status = "above"
        else:
            auc_status = "within"
        levels_data.append(("AUC24", f"{auc24:.1f} mg·hr/L", auc_target, auc_status))
    
    # Generate recommendations based on status
    dosing_recs = []
    monitoring_recs = []
    cautions = []
    
    # Check if we have enough data to provide recommendations
    if not levels_data:
        return "Insufficient data to generate interpretation. At least one measurement (peak, trough, or AUC) is required."
    
    # Extract new dose if available
    new_dose = None
    if "Recommended base dose" in prompt:
        parts = prompt.split("Recommended base dose")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose_str = dose_parts[0].replace("=", "").replace(":", "").strip()
                    new_dose = float(dose_str)
                except ValueError:
                    pass
    
    # Format new dose
    rounded_new_dose = None
    if new_dose:
        # Round to nearest 250mg for vancomycin
        rounded_new_dose = round(new_dose / 250) * 250
    
    # Generate recommendations based on status
    if status == "subtherapeutic (low trough)" or status == "subtherapeutic (inadequate trough)" or status == "subtherapeutic (low AUC)":
        if rounded_new_dose:
            dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("INCREASE dose by 25-30%")
        dosing_recs.append("CONSIDER shortening dosing interval")
        
        monitoring_recs.append("RECHECK levels after 3-4 doses (at steady state)")
        monitoring_recs.append("MONITOR renal function regularly")
        
        cautions.append("Subtherapeutic levels may lead to treatment failure")
        cautions.append("Ensure adequate hydration when increasing doses")
    
    elif status == "potentially supratherapeutic (high trough)" or status == "potentially supratherapeutic (high AUC)":
        if rounded_new_dose:
            dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("DECREASE dose by 20-25%")
        dosing_recs.append("CONSIDER extending dosing interval")
        
        monitoring_recs.append("RECHECK levels after 3-4 doses")
        monitoring_recs.append("MONITOR renal function closely")
        monitoring_recs.append("ASSESS for signs of nephrotoxicity")
        
        cautions.append("Risk of nephrotoxicity with elevated trough levels")
        cautions.append("Consider patient-specific risk factors for toxicity")
    
    elif status == "subtherapeutic (inadequate peak and trough)" or status == "potential underdosing (low peak)":
        if rounded_new_dose:
            dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("INCREASE dose by 30-40%")
        
        monitoring_recs.append("RECHECK peak and trough levels after 3-4 doses")
        monitoring_recs.append("VERIFY correct timing of sample collection")
        
        cautions.append("Significantly subtherapeutic levels increase risk of treatment failure")
        cautions.append("Consider evaluating for altered pharmacokinetics")
    
    elif status == "potentially supratherapeutic (high peak)":
        if rounded_new_dose:
            dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("DECREASE dose by 15-20%")
        dosing_recs.append("EXTEND dosing interval if appropriate")
        
        monitoring_recs.append("RECHECK levels after adjustment")
        monitoring_recs.append("MONITOR renal function")
        
        cautions.append("Risk of nephrotoxicity with excessive dosing")
    
    elif "appropriately dosed" in status:
        dosing_recs.append("CONTINUE current dosing regimen")
        
        monitoring_recs.append("MONITOR renal function regularly")
        monitoring_recs.append("REASSESS levels if clinical status changes")
        
        cautions.append("Even with therapeutic levels, monitor for adverse effects")
    
    else:  # requires adjustment
        if rounded_new_dose:
            dosing_recs.append(f"ADJUST dose to {rounded_new_dose}mg")
        else:
            dosing_recs.append("ADJUST dosing based on clinical response and levels")
        
        monitoring_recs.append("RECHECK levels after adjustment")
        monitoring_recs.append("MONITOR renal function")
        
        cautions.append("Individualize therapy based on clinical response")
    
    # Add standard monitoring recommendations
    if "MONITOR renal function" not in " ".join(monitoring_recs):
        monitoring_recs.append("MONITOR renal function every 2-3 days")
    
    return levels_data, assessment, dosing_recs, monitoring_recs, cautions# ===== AMINOGLYCOSIDE INTERPRETATION FUNCTION =====
def generate_aminoglycoside_interpretation(prompt):
    """
    Generate standardized aminoglycoside interpretation
    
    Returns a tuple of:
    - levels_data: List of tuples (name, value, target, status)
    - assessment: String of assessment
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations 
    - cautions: List of cautions
    
    Or returns a string if insufficient data
    """
    # Extract key values from the prompt
    drug_name = "aminoglycoside"
    peak_val = None
    trough_val = None
    
    if "Gentamicin" in prompt:
        drug_name = "gentamicin"
    elif "Amikacin" in prompt:
        drug_name = "amikacin"
    
    # Extract peak and trough values
    if "Cmax:" in prompt:
        parts = prompt.split("Cmax:")
        if len(parts) > 1:
            peak_parts = parts[1].split(",")
            if peak_parts:
                try:
                    peak_val = float(peak_parts[0])
                except ValueError:
                    pass
    elif "Expected Cmax" in prompt:
        parts = prompt.split("Expected Cmax")
        if len(parts) > 1:
            peak_parts = parts[1].split("mg/L")
            if peak_parts:
                try:
                    peak_val = float(peak_parts[0].replace(":", "").strip())
                except ValueError:
                    pass
    
    if "Cmin:" in prompt:
        parts = prompt.split("Cmin:")
        if len(parts) > 1:
            trough_parts = parts[1].split(",")
            if trough_parts:
                try:
                    trough_val = float(trough_parts[0])
                except ValueError:
                    pass
    elif "Expected Cmin" in prompt:
        parts = prompt.split("Expected Cmin")
        if len(parts) > 1:
            trough_parts = parts[1].split("mg/L")
            if trough_parts:
                try:
                    trough_val = float(trough_parts[0].replace(":", "").strip())
                except ValueError:
                    pass
    
    # Extract dose
    dose = None
    if "Dose:" in prompt:
        parts = prompt.split("Dose:")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose = float(dose_parts[0])
                except ValueError:
                    pass
    elif "Dose " in prompt:
        parts = prompt.split("Dose ")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    dose = float(dose_parts[0])
                except ValueError:
                    pass
    
    # Extract suggested new dose
    new_dose = None
    if "Suggested new dose:" in prompt:
        parts = prompt.split("Suggested new dose:")
        if len(parts) > 1:
            new_dose_parts = parts[1].split("mg")
            if new_dose_parts:
                try:
                    new_dose = float(new_dose_parts[0])
                except ValueError:
                    pass
    elif "Recommended" in prompt and "Dose" in prompt:
        parts = prompt.split("Recommended")
        if len(parts) > 1:
            dose_parts = parts[1].split("mg")
            if dose_parts:
                try:
                    # Extract the number from this string
                    import re
                    numbers = re.findall(r'\d+', dose_parts[0])
                    if numbers:
                        new_dose = float(numbers[0])
                except ValueError:
                    pass
    
    # Extract target values based on regimen mention
    regimen = None
    if "SDD" in prompt:
        regimen = "SDD"
    elif "Synergy" in prompt:
        regimen = "Synergy"
    elif "MDD" in prompt:
        regimen = "MDD"
    
    # Set target ranges based on drug
    if drug_name == "gentamicin":
        if regimen == "SDD":
            peak_target = "10-30 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 10, 30
            trough_max = 1
        elif regimen == "Synergy":
            peak_target = "3-5 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 3, 5
            trough_max = 1
        else:  # Default to MDD
            peak_target = "5-10 mg/L"
            trough_target = "<2 mg/L"
            peak_min, peak_max = 5, 10
            trough_max = 2
    elif drug_name == "amikacin":
        if regimen == "SDD":
            peak_target = "60-80 mg/L"
            trough_target = "<1 mg/L"
            peak_min, peak_max = 60, 80
            trough_max = 1
        else:  # Default to MDD
            peak_target = "20-30 mg/L"
            trough_target = "<10 mg/L"
            peak_min, peak_max = 20, 30
            trough_max = 10
    else:
        peak_target = "varies by drug"
        trough_target = "varies by drug"
        peak_min, peak_max = 0, 100
        trough_max = 10
    
    # Determine aminoglycoside status
    status = "assessment not available"
    if peak_val and trough_val:
        if peak_val < peak_min and trough_val > trough_max:
            status = "ineffective and potentially toxic"
        elif peak_val < peak_min:
            status = "subtherapeutic (inadequate peak)"
        elif trough_val > trough_max:
            status = "potentially toxic (elevated trough)"
        elif peak_min <= peak_val <= peak_max and trough_val <= trough_max:
            status = "appropriately dosed"
        elif peak_val > peak_max:
            status = "potentially toxic (elevated peak)"
        else:
            status = "requires adjustment"
    
    # Format new dose
    rounded_new_dose = None
    if new_dose:
        # Round to nearest 10mg for most aminoglycosides
        rounded_new_dose = round(new_dose / 10) * 10
    
    # Create interpretation using standardized format
    if peak_val is not None and trough_val is not None:
        # Determine peak status
        if peak_val < peak_min:
            peak_status = "below"
        elif peak_val > peak_max:
            peak_status = "above"
        else:
            peak_status = "within"
        
        # Determine trough status
        if trough_val > trough_max:
            trough_status = "above"
        else:
            trough_status = "within"
        
        # Prepare data for standardized format
        levels_data = [
            (f"Peak", f"{peak_val:.1f} mg/L", peak_target, peak_status),
            (f"Trough", f"{trough_val:.2f} mg/L", trough_target, trough_status)
        ]
        
        # Generate recommendations based on status
        dosing_recs = []
        monitoring_recs = []
        cautions = []
        
        if status == "ineffective and potentially toxic":
            dosing_recs.append("HOLD next dose")
            dosing_recs.append("REASSESS renal function before resuming")
            if rounded_new_dose:
                dosing_recs.append(f"DECREASE to {rounded_new_dose}mg when resumed")
            dosing_recs.append("EXTEND dosing interval significantly")
            
            monitoring_recs.append("CHECK renal function before resuming therapy")
            monitoring_recs.append("RECHECK levels 2 doses after resumption")
            monitoring_recs.append("MONITOR for signs of ototoxicity and nephrotoxicity")
            
            cautions.append("Risk of nephrotoxicity and ototoxicity with elevated trough levels")
            cautions.append("Ineffective therapy may lead to treatment failure")
            
        elif status == "subtherapeutic (inadequate peak)":
            if rounded_new_dose:
                dosing_recs.append(f"INCREASE dose to {rounded_new_dose}mg")
            else:
                dosing_recs.append("INCREASE dose by 25-50%")
            
            monitoring_recs.append("RECHECK levels after 2-3 doses")
            monitoring_recs.append("VERIFY correct timing of peak sample collection")
            
            cautions.append("Subtherapeutic levels may lead to treatment failure")
            cautions.append("Consider other factors affecting drug disposition")
            
        elif status == "potentially toxic (elevated trough)":
            dosing_recs.append("EXTEND dosing interval")
            if rounded_new_dose:
                dosing_recs.append(f"CONSIDER dose reduction to {rounded_new_dose}mg")
            
            monitoring_recs.append("MONITOR renal function closely")
            monitoring_recs.append("RECHECK levels before next dose")
            monitoring_recs.append("ASSESS for signs of ototoxicity and nephrotoxicity")
            
            cautions.append("Risk of nephrotoxicity and ototoxicity with elevated trough levels")
            cautions.append("Consider patient-specific risk factors for toxicity")
            
        elif status == "potentially toxic (elevated peak)":
            if rounded_new_dose:
                dosing_recs.append(f"DECREASE dose to {rounded_new_dose}mg")
            else:
                dosing_recs.append("DECREASE dose by 20-25%")
            
            monitoring_recs.append("RECHECK levels after 2-3 doses")
            monitoring_recs.append("VERIFY correct timing of peak sample collection")
            monitoring_recs.append("MONITOR for signs of ototoxicity")
            
            cautions.append("Risk of ototoxicity with significantly elevated peak levels")
            
        elif status == "appropriately dosed":
            dosing_recs.append("CONTINUE current dosing regimen")
            
            monitoring_recs.append("MONITOR renal function regularly")
            monitoring_recs.append("REASSESS levels if clinical status changes")
            monitoring_recs.append("CONSIDER extended interval dosing for longer therapy")
            
            cautions.append("Even with therapeutic levels, monitor for adverse effects")
            
        else:  # requires adjustment
            dosing_recs.append("ADJUST dosing based on clinical response")
            if rounded_new_dose:
                dosing_recs.append(f"CONSIDER dose of {rounded_new_dose}mg")
            
            monitoring_recs.append("RECHECK levels after adjustment")
            monitoring_recs.append("MONITOR renal function")
            
            cautions.append("Individualize therapy based on clinical response")
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions
    else:
        return "Insufficient data to generate interpretation. Both peak and trough levels are required."# ===== FORMAT_CLINICAL_RECOMMENDATIONS FUNCTION =====
def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy
    
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
    levels_md = "📊 **MEASURED LEVELS:**\n"
    for name, value, target, status in levels_data:
        icon = "✅" if status == "within" else "⚠️" if status == "below" else "🔴"
        levels_md += f"- {name}: {value} (Target: {target}) {icon}\n"
    
    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\nPatient is {assessment.upper()}"
    
    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT

{levels_md}
{assessment_md}

## DETAILED RECOMMENDATIONS

🔵 **DOSING RECOMMENDATIONS:**
"""
    for rec in dosing_recs:
        output += f"- {rec}\n"
    
    output += "\n🔵 **MONITORING RECOMMENDATIONS:**\n"
    for rec in monitoring_recs:
        output += f"- {rec}\n"
    
    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS & CONSIDERATIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"
    
    # Add a summary section for quick reference
    output += "\n## QUICK SUMMARY\n"
    output += "**Status:** " + assessment.upper() + "\n"
    
    # Summarize key recommendations
    if len(dosing_recs) > 0:
        output += f"**Key Dosing Action:** {dosing_recs[0]}\n"
    
    if len(monitoring_recs) > 0:
        output += f"**Key Monitoring Action:** {monitoring_recs[0]}\n"
        
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    output += f"\n*Generated on: {timestamp}*"
    
    return outputimport streamlit as st
import numpy as np
import math
import pandas as pd
import altair as alt
import base64
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

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
    # Check for OpenAI API key
    import openai
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError, ImportError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found in Streamlit secrets. LLM interpretation will not be available.
    
    To enable this feature:
    1. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    2. Or in Streamlit Cloud, add the secret in the dashboard
    """)

# ===== PATIENT INFO SECTION =====
def display_patient_info_section():
    """Display and collect patient information"""
    st.header("Patient Information")
    
    # Create a 2x2 grid for patient info
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", help="Enter the patient's unique identifier")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        serum_cr = st.number_input("Serum Creatinine (μmol/L)", min_value=10, max_value=1000, value=80)
    
    with col2:
        ward = st.text_input("Ward/Unit", help="Enter the patient's current location")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=40, max_value=250, value=170)
        clinical_diagnosis = st.text_input("Clinical Diagnosis", "Sepsis")
    
    # Calculate Creatinine Clearance
    # Cockcroft-Gault equation
    scr_mg = serum_cr / 88.4  # Convert μmol/L to mg/dL
    if gender == "Male":
        crcl = ((140 - age) * weight) / (72 * scr_mg)
    else:
        crcl = ((140 - age) * weight * 0.85) / (72 * scr_mg)
    
    # Determine renal function category
    if crcl >= 90:
        renal_function = "Normal renal function"
    elif crcl >= 60:
        renal_function = "Mild renal impairment"
    elif crcl >= 30:
        renal_function = "Moderate renal impairment"
    elif crcl >= 15:
        renal_function = "Severe renal impairment"
    else:
        renal_function = "Renal failure"
    
    # Display calculated CrCl and renal function
    st.metric("Estimated CrCl", f"{crcl:.1f} mL/min", renal_function)
    
    # Current medication regimen
    current_dose_regimen = st.text_input("Current Dosing Regimen", "Vancomycin 1000mg q12h")
    
    st.info(f"Patient {patient_id} is in {ward} with {renal_function.lower()} (CrCl: {crcl:.1f} mL/min)")
    
    # Return patient data as a dictionary
    return {
        'patient_id': patient_id,
        'ward': ward,
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'serum_cr': serum_cr,
        'crcl': crcl,
        'renal_function': renal_function,
        'clinical_diagnosis': clinical_diagnosis,
        'current_dose_regimen': current_dose_regimen
    }

# ===== CONCENTRATION-TIME CURVE VISUALIZATION =====
def plot_concentration_time_curve(drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details, peak, trough, ke, tau, t_peak=1.0, infusion_time=1.0):
    """
    Generate a concentration-time curve visualization
    
    Parameters:
    - drug_info: String with drug name
    - levels_data: List of level data
    - assessment: Assessment string
    - dosing_recs: List of dosing recommendations
    - monitoring_recs: List of monitoring recommendations
    - calculation_details: String with calculation details
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
            conc = trough + (peak - trough) * (t / infusion_time)
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
    if "Vancomycin" in drug_info:  # Vancomycin
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
    elif "Gentamicin" in drug_info:  # Gentamicin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [10], 'y2': [30]  # Peak range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for gentamicin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [5], 'y2': [10]  # Peak range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [2]  # Trough range for gentamicin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    elif "Amikacin" in drug_info:  # Amikacin
        if "SDD" in drug_info:  # Once-daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [60], 'y2': [80]  # Peak range for amikacin SDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [1]  # Trough range for amikacin SDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
        else:  # Multiple daily dosing
            target_peak_band = alt.Chart(pd.DataFrame({
                'y1': [20], 'y2': [30]  # Peak range for amikacin MDD
            })).mark_rect(opacity=0.2, color='green').encode(
                y='y1', y2='y2'
            )
            target_trough_band = alt.Chart(pd.DataFrame({
                'y1': [0], 'y2': [10]  # Trough range for amikacin MDD
            })).mark_rect(opacity=0.2, color='blue').encode(
                y='y1', y2='y2'
            )
    else:  # Default or unknown drug
        target_peak_band = alt.Chart(pd.DataFrame({
            'y1': [peak*0.8], 'y2': [peak*1.2]  # Default peak range ±20%
        })).mark_rect(opacity=0.2, color='green').encode(
            y='y1', y2='y2'
        )
        target_trough_band = alt.Chart(pd.DataFrame({
            'y1': [trough*0.5], 'y2': [trough*1.5]  # Default trough range ±50%
        })).mark_rect(opacity=0.2, color='blue').encode(
            y='y1', y2='y2'
        )
    
    # Create the concentration-time curve
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)')
    )
    
    # Add markers for actual measured peak and trough
    markers = alt.Chart(pd.DataFrame({
        'Time (hr)': [t_peak, tau],
        'Concentration (mg/L)': [peak, trough],
        'Label': ['Peak', 'Trough']
    })).mark_point(size=100, filled=True).encode(
        x='Time (hr)',
        y='Concentration (mg/L)',
        color=alt.Color('Label', scale=alt.Scale(domain=['Peak', 'Trough'], range=['green', 'blue']))
    )
    
    # Add vertical lines for key time points
    infusion_end = alt.Chart(pd.DataFrame({'x': [infusion_time]})).mark_rule(
        strokeDash=[5, 5], color='gray'
    ).encode(x='x')
    
    next_dose = alt.Chart(pd.DataFrame({'x': [tau]})).mark_rule(
        strokeDash=[5, 5], color='red'
    ).encode(x='x')
    
    # Add text annotations for key time points
    annotations = alt.Chart(pd.DataFrame({
        'x': [infusion_time/2, tau],
        'y': [peak*1.1, trough*0.9],
        'text': ['Infusion', 'Next Dose']
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Calculate half-life and display it
    half_life = 0.693 / ke
    half_life_text = alt.Chart(pd.DataFrame({
        'x': [tau/2],
        'y': [peak*0.5],
        'text': [f"t½ = {half_life:.1f} hr"]
    })).mark_text().encode(
        x='x',
        y='y',
        text='text'
    )
    
    # Combine charts
    chart = alt.layer(
        target_peak_band, 
        target_trough_band, 
        line, 
        markers,
        infusion_end,
        next_dose,
        annotations,
        half_life_text
    ).properties(
        width=600,
        height=400,
        title=f'{drug_info} Concentration-Time Profile'
    )
    
    # Display detailed calculation steps in an expander
    with st.expander("View Calculation Details", expanded=False):
        st.markdown("### PK Parameter Calculations")
        st.markdown(f"""
        **Key Parameters:**
        - Peak concentration (Cmax): {peak:.2f} mg/L
        - Trough concentration (Cmin): {trough:.2f} mg/L
        - Elimination rate constant (Ke): {ke:.4f} hr⁻¹
        - Half-life (t½): {half_life:.2f} hr
        - Dosing interval (τ): {tau} hr
        
        **Detailed Calculations:**
        ```
        Ke = -ln(Cmin/Cmax)/(τ - tpeak)
        Ke = -ln({trough:.2f}/{peak:.2f})/({tau} - {t_peak})
        Ke = {ke:.4f} hr⁻¹
        
        t½ = 0.693/Ke
        t½ = 0.693/{ke:.4f}
        t½ = {half_life:.2f} hr
        ```
        
        **Assessment:**
        {assessment}
        
        **Dosing Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in dosing_recs]) + """
        
        **Monitoring Recommendations:**
        """ + "\n".join([f"- {rec}" for rec in monitoring_recs]))
        
        if calculation_details:
            st.markdown("**Additional Calculation Information:**")
            st.markdown(calculation_details)
    
    return chart

# ===== PDF GENERATION FUNCTIONS =====
def create_recommendation_pdf(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Create a downloadable PDF with the clinical recommendations
    
    Parameters:
    - patient_data: Dictionary with patient information
    - drug_info: String with drug name and method
    - levels_data: List of tuples (name, value, target, status) for each measured level
    - assessment: Overall assessment string
    - dosing_recs: List of dosing recommendation strings
    - monitoring_recs: List of monitoring recommendation strings
    - calculation_details: Optional string with calculation details
    - cautions: Optional list of caution strings
    
    Returns:
    - base64 encoded PDF for download
    """
    # Create an in-memory PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom styles
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading3'],
        spaceAfter=6,
        textColor=colors.navy
    )
    
    # Create the content
    content = []
    
    # Add report title
    content.append(Paragraph("Antimicrobial TDM Report", title_style))
    content.append(Spacer(1, 12))
    
    # Add date and time
    now = datetime.now()
    content.append(Paragraph(f"Report Generated: {now.strftime('%Y-%m-%d %H:%M')}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add patient information
    content.append(Paragraph("Patient Information", heading_style))
    
    # Create patient info table with ID and Ward
    patient_info = []
    
    # Add patient ID and ward row
    patient_info.append([
        Paragraph("<b>Patient ID:</b>", normal_style),
        Paragraph(f"{patient_data.get('patient_id', 'N/A')}", normal_style),
        Paragraph("<b>Ward:</b>", normal_style),
        Paragraph(f"{patient_data.get('ward', 'N/A')}", normal_style)
    ])
    
    # First row
    patient_info.append([
        Paragraph("<b>Age:</b>", normal_style),
        Paragraph(f"{patient_data.get('age', 'N/A')} years", normal_style),
        Paragraph("<b>Gender:</b>", normal_style),
        Paragraph(f"{patient_data.get('gender', 'N/A')}", normal_style)
    ])
    
    # Second row
    patient_info.append([
        Paragraph("<b>Weight:</b>", normal_style),
        Paragraph(f"{patient_data.get('weight', 'N/A')} kg", normal_style),
        Paragraph("<b>Height:</b>", normal_style),
        Paragraph(f"{patient_data.get('height', 'N/A')} cm", normal_style)
    ])
    
    # Third row
    patient_info.append([
        Paragraph("<b>Serum Creatinine:</b>", normal_style),
        Paragraph(f"{patient_data.get('serum_cr', 'N/A')} µmol/L", normal_style),
        Paragraph("<b>CrCl:</b>", normal_style),
        Paragraph(f"{patient_data.get('crcl', 'N/A'):.1f} mL/min", normal_style)
    ])
    
    # Fourth row with diagnosis spanning full width
    patient_info.append([
        Paragraph("<b>Diagnosis:</b>", normal_style),
        Paragraph(f"{patient_data.get('clinical_diagnosis', 'N/A')}", normal_style),
        Paragraph("<b>Renal Function:</b>", normal_style),
        Paragraph(f"{patient_data.get('renal_function', 'N/A')}", normal_style)
    ])
    
    # Fifth row with regimen spanning full width
    patient_info.append([
        Paragraph("<b>Current Regimen:</b>", normal_style),
        Paragraph(f"{patient_data.get('current_dose_regimen', 'N/A')}", normal_style),
        Paragraph("", normal_style),
        Paragraph("", normal_style)
    ])
    
    # Create the table
    patient_table = Table(patient_info, colWidths=[100, 150, 100, 150])
    patient_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    content.append(patient_table)
    content.append(Spacer(1, 12))
    
    # Add drug information
    content.append(Paragraph("Drug Information", heading_style))
    content.append(Paragraph(drug_info, normal_style))
    content.append(Spacer(1, 12))
    
    # Add clinical assessment
    content.append(Paragraph("Clinical Assessment", heading_style))
    
    # Add measured levels
    content.append(Paragraph("Measured Levels:", section_style))
    
    # Create levels table
    levels_table_data = [["Parameter", "Value", "Target Range", "Status"]]
    
    for name, value, target, status in levels_data:
        # Determine status text and color
        if status == "within":
            status_text = "Within Range"
            status_color = colors.green
        elif status == "below":
            status_text = "Below Range"
            status_color = colors.orange
        else:  # above
            status_text = "Above Range"
            status_color = colors.red
        
        levels_table_data.append([name, value, target, status_text])
    
    levels_table = Table(levels_table_data)
    levels_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    # Add status color to each row in the table
    for i, (_, _, _, status) in enumerate(levels_data, 1):
        if status == "within":
            color = colors.lightgreen
        elif status == "below":
            color = colors.lightyellow
        else:  # above
            color = colors.mistyrose
        
        levels_table.setStyle(TableStyle([
            ('BACKGROUND', (3, i), (3, i), color),
        ]))
    
    content.append(levels_table)
    content.append(Spacer(1, 8))
    
    # Add assessment
    content.append(Paragraph("Assessment:", section_style))
    content.append(Paragraph(f"Patient is {assessment.upper()}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add calculations section if provided
    if calculation_details:
        content.append(Paragraph("Calculation Details:", section_style))
        content.append(Paragraph(calculation_details, normal_style))
        content.append(Spacer(1, 12))
    
    # Add recommendations
    content.append(Paragraph("Recommendations", heading_style))
    
    # Add dosing recommendations
    content.append(Paragraph("Dosing:", section_style))
    for rec in dosing_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add monitoring recommendations
    content.append(Paragraph("Monitoring:", section_style))
    for rec in monitoring_recs:
        content.append(Paragraph(f"• {rec}", normal_style))
    content.append(Spacer(1, 8))
    
    # Add cautions if any
    if cautions and len(cautions) > 0:
        content.append(Paragraph("Cautions:", section_style))
        for caution in cautions:
            content.append(Paragraph(f"• {caution}", normal_style))
    
    # Add disclaimer
    content.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey
    )
    content.append(Paragraph("Disclaimer: This report is generated by an automated system and is intended to assist clinical decision making. Always use professional judgment when implementing recommendations.", disclaimer_style))
    
    # Build the PDF
    doc.build(content)
    
    # Get the PDF value from the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    # Encode the PDF to base64
    pdf_base64 = base64.b64encode(pdf_value).decode()
    
    return pdf_base64

# Function to create a download link for the PDF
def get_pdf_download_link(pdf_base64, filename="clinical_recommendations.pdf"):
    """Create a download link for a base64 encoded PDF"""
    href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="{filename}">Download Clinical Recommendations PDF</a>'
    return href

# Updated function to display buttons for printing and downloading recommendations
def display_pdf_download_button(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Display buttons to print/save recommendations as a PDF and print a summary
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Print/Save Full Report"):
            # Generate the PDF
            pdf_base64 = create_recommendation_pdf(
                patient_data, 
                drug_info, 
                levels_data, 
                assessment, 
                dosing_recs, 
                monitoring_recs,
                calculation_details,
                cautions
            )
            
            # Create the download link
            download_link = get_pdf_download_link(pdf_base64)
            
            # Display the download link
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Preview message
            st.success("PDF generated successfully. Click the link above to download.")
    
    with col2:
        if st.button("🖨️ Print Clinical Summary"):
            # Create a simple text printout of the assessment and recommendations
            assessment_text = create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            
            # Display in a text area that can be easily copied
            st.text_area("Copy this text to print", assessment_text, height=300)
            st.success("Assessment text generated. Copy and paste into your preferred document.")

# Enhanced function to create a printable text assessment
def create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """Create a plain text printable assessment for easy copying to clinical notes"""
    now = datetime.now()
    
    # Header
    text = f"ANTIMICROBIAL TDM ASSESSMENT - {now.strftime('%Y-%m-%d %H:%M')}\n"
    text += "=" * 50 + "\n\n"
    
    # Patient information - Make sure to include ID and ward
    text += f"Patient ID: {patient_data.get('patient_id', 'N/A')}\n"
    text += f"Ward: {patient_data.get('ward', 'N/A')}\n"
    text += f"Age: {patient_data.get('age', 'N/A')} years  |  "
    text += f"Gender: {patient_data.get('gender', 'N/A')}  |  "
    text += f"Weight: {patient_data.get('weight', 'N/A')} kg\n"
    text += f"Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}\n"
    text += f"CrCl: {patient_data.get('crcl', 'N/A'):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})\n"
    text += f"Current regimen: {patient_data.get('current_dose_regimen', 'N/A')}\n\n"
    
    # Measured levels
    text += "MEASURED LEVELS:\n"
    for name, value, target, status in levels_data:
        status_text = "✓" if status == "within" else "↓" if status == "below" else "↑"
        text += f"- {name}: {value} (Target: {target}) {status_text}\n"
    
    # Assessment
    text += f"\nASSESSMENT: Patient is {assessment.upper()}\n\n"
    
    # PK Parameters (if available from calculation details)
    try:
        if "Half-life" in calculation_details or "t½" in calculation_details:
            text += "PHARMACOKINETIC PARAMETERS:\n"
            # Extract PK parameters from calculation details
            import re
            ke_match = re.search(r'Ke[\s=:]+([0-9.]+)', calculation_details)
            t_half_match = re.search(r't½[\s=:]+([0-9.]+)', calculation_details)
            
            if ke_match:
                ke = float(ke_match.group(1))
                text += f"- Elimination rate constant (Ke): {ke:.4f} hr⁻¹\n"
            
            if t_half_match:
                t_half = float(t_half_match.group(1))
                text += f"- Half-life (t½): {t_half:.2f} hr\n"
            
            text += "\n"
    except:
        pass  # Skip if unable to extract PK parameters
    
    # Recommendations
    text += "DOSING RECOMMENDATIONS:\n"
    for rec in dosing_recs:
        text += f"- {rec}\n"
    
    text += "\nMONITORING RECOMMENDATIONS:\n"
    for rec in monitoring_recs:
        text += f"- {rec}\n"
    
    # Cautions
    if cautions and len(cautions) > 0:
        text += "\nCAUTIONS:\n"
        for caution in cautions:
            text += f"- {caution}\n"
    
    # Footer
    text += "\n" + "=" * 50 + "\n"
    text += "This assessment is intended to assist clinical decision making.\n"
    text += "Always use professional judgment when implementing recommendations.\n"
    text += f"Generated by: Antimicrobial TDM App - {now.strftime('%Y-%m-%d')}"
    
    return text
