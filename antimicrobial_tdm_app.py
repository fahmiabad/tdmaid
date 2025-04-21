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
            monitoring_recs.append("RECHECK levels 2 doses after resumption")            # Add visualization option
            if st.checkbox("Show concentration-time curve"):
                chart = plot_concentration_time_curve(
                    drug_info, 
                    levels_data, 
                    assessment, 
                    dosing_recs, 
                    monitoring_recs, 
                    calculation_details,
                    cautions
                )
                
        except Exception as e:
            st.error(f"Calculation error: {e}")peak=peak, 
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
                    f"Target peak range = {target_peak[0]}-{target_peak[1]} mg/L, Recommended base dose = {practical_new_dose:.0f} mg."
                )
                
                # Generate clinical interpretation
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
                    import streamlit as st
import numpy as np
import math
import openai
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
    
    # Fifth row with notes spanning full width
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
 Determine status text and color
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
        if st.button("📄 Print/Save Recommendations"):
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
        if st.button("🖨️ Print Assessment"):
            # Create a simple text printout of the assessment and recommendations
            assessment_text = create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions)
            
            # Display in a text area that can be easily copied
            st.text_area("Copy this text to print", assessment_text, height=300)
            st.success("Assessment text generated. Copy and paste into your preferred document.")

# New function to create a printable text assessment
def create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """Create a plain text printable assessment for easy copying to clinical notes"""
    now = datetime.now()
    
    # Header
    text = f"ANTIMICROBIAL TDM ASSESSMENT - {now.strftime('%Y-%m-%d %H:%M')}\n"
    text += "=" * 50 + "\n\n"
    
    # Patient information
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
    text += "Always use professional judgment when implementing recommendations."
    
    return text

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

# ===== UPDATED FORMAT_CLINICAL_RECOMMENDATIONS FUNCTION =====
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

## RECOMMENDATIONS

🔵 **DOSING:**
"""
    for rec in dosing_recs:
        output += f"- {rec}\n"
    
    output += "\n🔵 **MONITORING:**\n"
    for rec in monitoring_recs:
        output += f"- {rec}\n"
    
    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"
    
    return output

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
    st.info("Simulated interpretation. For production use, configure OpenAI API in Streamlit secrets.toml")

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
    elif drug == "Aminoglycoside":
        return generate_aminoglycoside_interpretation(prompt)
    else:
        # For generic, we'll create a simple placeholder
        levels_data = [("Not available", "N/A", "N/A", "within")]
        assessment = "requires specific assessment"
        dosing_recs = ["CONSULT antimicrobial stewardship team", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on antimicrobial type", "MONITOR renal function regularly"]
        cautions = ["Patient-specific factors may require dose adjustments"]
        
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions
