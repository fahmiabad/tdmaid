# -*- coding: utf-8 -*-
"""
Streamlit App for Antimicrobial Therapeutic Drug Monitoring (TDM)
"""

import streamlit as st
import numpy as np
import math
import pandas as pd
import altair as alt
import base64
from datetime import datetime
import io
import re # Import the re module for regular expressions

# ReportLab for PDF Generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.error("ReportLab library not found. PDF generation will be disabled. Install using: pip install reportlab")

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

# API Configuration - OpenAI
try:
    # Check for OpenAI API key
    import openai
    # Securely access the API key from streamlit secrets
    # Ensure you have a secrets.toml file or configure in Streamlit Cloud:
    # [.streamlit/secrets.toml]
    # [openai]
    # api_key = "sk-..."
    openai.api_key = st.secrets["openai"]["api_key"]
    OPENAI_AVAILABLE = True
except (KeyError, AttributeError, ImportError):
    OPENAI_AVAILABLE = False
    st.warning("""
    OpenAI API key not found or library not installed. LLM interpretation feature will use simulated responses.

    To enable full LLM features:
    1. Install the library: pip install openai
    2. Create a file named '.streamlit/secrets.toml' with:
       [openai]
       api_key = "your-api-key"
    3. Or in Streamlit Cloud, add the secret in the dashboard settings.
    """)

# Set page configuration
st.set_page_config(page_title="Antimicrobial TDM App", layout="wide")

# ===== PATIENT INFO SECTION =====
def display_patient_info_section():
    """Display and collect patient information"""
    st.header("Patient Information")

    # Create a 2x2 grid for patient info
    col1, col2 = st.columns(2)

    with col1:
        # Input field for Patient ID - Check if this is visible
        patient_id = st.text_input("Patient ID", help="Enter the patient's unique identifier")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        serum_cr = st.number_input("Serum Creatinine (μmol/L)", min_value=10, max_value=1000, value=80)

    with col2:
        # Input field for Ward/Unit - Check if this is visible
        ward = st.text_input("Ward/Unit", help="Enter the patient's current location")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=40, max_value=250, value=170)
        clinical_diagnosis = st.text_input("Clinical Diagnosis", "Sepsis")

    # Calculate Creatinine Clearance
    crcl = 0.0 # Default value
    renal_function = "Unknown"
    try:
        if serum_cr > 0 and weight > 0 and age > 0:
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
        else:
             st.warning("Please enter valid Age, Weight, and Serum Creatinine for CrCl calculation.")

    except ZeroDivisionError:
        st.error("Serum Creatinine cannot be zero for CrCl calculation.")
        crcl = 0.0
        renal_function = "Calculation Error"

    # Display calculated CrCl and renal function
    st.metric("Estimated CrCl", f"{crcl:.1f} mL/min", help=renal_function)

    # Current medication regimen
    current_dose_regimen = st.text_input("Current Dosing Regimen (e.g., Vancomycin 1000mg q12h)", "Vancomycin 1000mg q12h")

    st.info(f"Patient '{patient_id}' in '{ward}' with {renal_function.lower()} (CrCl: {crcl:.1f} mL/min)")

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
    Generate a concentration-time curve visualization.
    Handles potential calculation errors gracefully.
    """
    if ke <= 0 or tau <= 0 or peak <= 0 or trough < 0 or peak <= trough:
        st.error(f"Invalid PK parameters for plotting (Ke={ke}, Tau={tau}, Peak={peak}, Trough={trough}). Cannot generate plot.")
        return None # Return None if parameters are invalid

    half_life = float('inf')
    try:
        half_life = 0.693 / ke
    except ZeroDivisionError:
        st.warning("Ke is zero, cannot calculate half-life for plot.")
        return None # Cannot plot if Ke is zero

    # Generate time points for the curve
    times = np.linspace(0, tau * 1.5, 100)  # Generate points for 1.5 intervals

    # Generate concentrations for each time point
    concentrations = []
    try:
        for t in times:
            # During first infusion
            if t <= infusion_time:
                # Linear increase during infusion (avoid division by zero if infusion_time is 0)
                conc = trough + (peak - trough) * (t / max(infusion_time, 1e-6))
            # After infusion, before next dose
            elif t <= tau:
                # Exponential decay after peak
                t_after_peak = t - t_peak
                conc = peak * np.exp(-ke * t_after_peak)
            # During second infusion
            elif t <= tau + infusion_time:
                # Second dose starts with trough and increases linearly
                t_in_second_infusion = t - tau
                conc = trough + (peak - trough) * (t_in_second_infusion / max(infusion_time, 1e-6))
            # After second infusion
            else:
                # Exponential decay after second peak
                t_after_second_peak = t - (tau + t_peak)
                conc = peak * np.exp(-ke * t_after_second_peak)

            concentrations.append(max(conc, 0)) # Ensure concentration is not negative

    except Exception as e:
        st.error(f"Error calculating concentrations for plot: {e}")
        return None

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Time (hr)': times,
        'Concentration (mg/L)': concentrations
    })

    # --- Target Range Bands (Simplified - Add specific logic if needed) ---
    target_peak_y1, target_peak_y2 = peak * 0.8, peak * 1.2
    target_trough_y1, target_trough_y2 = trough * 0.5, trough * 1.5

    # Adjust based on drug type (Example for Vancomycin)
    if "Vancomycin" in drug_info:
        target_peak_y1, target_peak_y2 = 20, 40
        target_trough_y1, target_trough_y2 = 10, 15 # Example, adjust based on empirical/definitive
        if "Definitive" in drug_info:
             target_trough_y1, target_trough_y2 = 15, 20
    elif "Gentamicin" in drug_info:
        if "SDD" in drug_info:
            target_peak_y1, target_peak_y2 = 10, 30
            target_trough_y1, target_trough_y2 = 0, 1
        elif "Synergy" in drug_info:
            target_peak_y1, target_peak_y2 = 3, 5
            target_trough_y1, target_trough_y2 = 0, 1
        else: # MDD
            target_peak_y1, target_peak_y2 = 5, 10
            target_trough_y1, target_trough_y2 = 0, 2
    elif "Amikacin" in drug_info:
        if "SDD" in drug_info:
            target_peak_y1, target_peak_y2 = 60, 80
            target_trough_y1, target_trough_y2 = 0, 1
        else: # MDD
            target_peak_y1, target_peak_y2 = 20, 30
            target_trough_y1, target_trough_y2 = 0, 10


    target_peak_band = alt.Chart(pd.DataFrame({
        'y1': [target_peak_y1], 'y2': [target_peak_y2]
    })).mark_rect(opacity=0.2, color='lightgreen').encode(
        y='y1', y2='y2'
    )

    target_trough_band = alt.Chart(pd.DataFrame({
        'y1': [target_trough_y1], 'y2': [target_trough_y2]
    })).mark_rect(opacity=0.2, color='lightblue').encode(
        y='y1', y2='y2'
    )

    # Create the concentration-time curve line
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('Time (hr)', title='Time (hours)'),
        y=alt.Y('Concentration (mg/L)', title='Drug Concentration (mg/L)', scale=alt.Scale(zero=False)) # Ensure Y axis doesn't always start at 0
    )

    # Add markers for actual measured peak and trough
    markers = alt.Chart(pd.DataFrame({
        'Time (hr)': [t_peak, tau],
        'Concentration (mg/L)': [peak, trough],
        'Label': ['Peak', 'Trough']
    })).mark_point(size=100, filled=True).encode(
        x='Time (hr)',
        y='Concentration (mg/L)',
        color=alt.Color('Label', scale=alt.Scale(domain=['Peak', 'Trough'], range=['green', 'blue'])),
        tooltip=['Label', 'Time (hr)', 'Concentration (mg/L)']
    )

    # Add vertical lines for key time points
    infusion_end_line = alt.Chart(pd.DataFrame({'x': [infusion_time]})).mark_rule(
        strokeDash=[5, 5], color='gray'
    ).encode(x='x')

    next_dose_line = alt.Chart(pd.DataFrame({'x': [tau]})).mark_rule(
        strokeDash=[5, 5], color='red'
    ).encode(x='x')

    # Add text annotations for key time points
    annotations = alt.Chart(pd.DataFrame({
        'x': [infusion_time / 2, tau],
        'y': [peak * 1.1, trough * 0.9], # Adjust y position relative to peak/trough
        'text': ['Infusion', 'Next Dose']
    })).mark_text(align='center', dy=-10).encode( # Adjust dy for vertical position
        x='x',
        y='y',
        text='text'
    )

    # Display half-life text
    half_life_text = alt.Chart(pd.DataFrame({
        'x': [tau / 2],
        'y': [peak * 0.5], # Position roughly in the middle
        'text': [f"t½ = {half_life:.1f} hr"]
    })).mark_text(align='center').encode(
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
        infusion_end_line,
        next_dose_line,
        annotations,
        half_life_text
    ).properties(
        width=600,
        height=400,
        title=f'{drug_info} Concentration-Time Profile'
    ).interactive() # Make chart interactive (zoom/pan)

    # Display detailed calculation steps in an expander (ensure calculation_details is a string)
    if calculation_details and isinstance(calculation_details, str):
        with st.expander("View Calculation Details", expanded=False):
            st.markdown("### PK Parameter Calculations")
            st.markdown(f"""
            **Input Levels:**
            - Peak concentration (Cmax): {peak:.2f} mg/L (at {t_peak} hr)
            - Trough concentration (Cmin): {trough:.2f} mg/L (at {tau} hr)

            **Calculated Parameters:**
            - Elimination rate constant (Ke): {ke:.4f} hr⁻¹
            - Half-life (t½): {half_life:.2f} hr
            - Dosing interval (τ): {tau} hr

            **Calculation Formulas Used:**
            ```
            Ke = -ln(Cmin/Cmax) / (τ - t_peak)
            t½ = 0.693 / Ke
            ```

            **Assessment:**
            {assessment}

            **Dosing Recommendations:**
            """ + "\n".join([f"- {rec}" for rec in dosing_recs]) + """

            **Monitoring Recommendations:**
            """ + "\n".join([f"- {rec}" for rec in monitoring_recs]))

            st.markdown("**Additional Calculation Information:**")
            st.markdown(calculation_details) # Display the passed calculation details string

    return chart

# ===== PDF GENERATION FUNCTIONS =====
def create_recommendation_pdf(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Create a downloadable PDF with the clinical recommendations.
    Requires reportlab to be installed.
    """
    if not REPORTLAB_AVAILABLE:
        st.error("ReportLab library is not available. Cannot generate PDF.")
        return None

    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['h1'] # Use h1 for main title
        heading_style = styles['h2']
        normal_style = styles['Normal']
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['h3'], # Use h3 for section titles
            spaceAfter=6,
            textColor=colors.navy
        )
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=normal_style,
            fontSize=8,
            textColor=colors.grey
        )
        bold_normal_style = ParagraphStyle('BoldNormal', parent=normal_style, fontName='Helvetica-Bold')

        content = []

        # Title and Date
        content.append(Paragraph("Antimicrobial TDM Report", title_style))
        content.append(Spacer(1, 12))
        now = datetime.now()
        content.append(Paragraph(f"Report Generated: {now.strftime('%Y-%m-%d %H:%M')}", normal_style))
        content.append(Spacer(1, 12))

        # Patient Information
        content.append(Paragraph("Patient Information", heading_style))
        patient_info_data = [
            [Paragraph("<b>Patient ID:</b>", normal_style), Paragraph(f"{patient_data.get('patient_id', 'N/A')}", normal_style),
             Paragraph("<b>Ward:</b>", normal_style), Paragraph(f"{patient_data.get('ward', 'N/A')}", normal_style)],
            [Paragraph("<b>Age:</b>", normal_style), Paragraph(f"{patient_data.get('age', 'N/A')} years", normal_style),
             Paragraph("<b>Gender:</b>", normal_style), Paragraph(f"{patient_data.get('gender', 'N/A')}", normal_style)],
            [Paragraph("<b>Weight:</b>", normal_style), Paragraph(f"{patient_data.get('weight', 'N/A')} kg", normal_style),
             Paragraph("<b>Height:</b>", normal_style), Paragraph(f"{patient_data.get('height', 'N/A')} cm", normal_style)],
            [Paragraph("<b>Serum Cr:</b>", normal_style), Paragraph(f"{patient_data.get('serum_cr', 'N/A')} µmol/L", normal_style),
             Paragraph("<b>CrCl:</b>", normal_style), Paragraph(f"{patient_data.get('crcl', 0.0):.1f} mL/min", normal_style)],
            [Paragraph("<b>Diagnosis:</b>", normal_style), Paragraph(f"{patient_data.get('clinical_diagnosis', 'N/A')}", normal_style),
             Paragraph("<b>Renal Fn:</b>", normal_style), Paragraph(f"{patient_data.get('renal_function', 'N/A')}", normal_style)],
             # Span Current Regimen across columns
            [Paragraph("<b>Current Regimen:</b>", normal_style), Paragraph(f"{patient_data.get('current_dose_regimen', 'N/A')}", normal_style), '', '']
        ]
        patient_table = Table(patient_info_data, colWidths=[80, 170, 80, 170]) # Adjusted widths
        patient_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # Span the last row's second cell
            ('SPAN', (1, 5), (3, 5)),
            # Bold labels
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, 4), 'Helvetica-Bold'), # Bold labels in 3rd column up to Renal Fn
        ]))
        content.append(patient_table)
        content.append(Spacer(1, 12))

        # Drug Information
        content.append(Paragraph("Drug Information", heading_style))
        content.append(Paragraph(drug_info, normal_style))
        content.append(Spacer(1, 12))

        # Clinical Assessment
        content.append(Paragraph("Clinical Assessment", heading_style))
        content.append(Paragraph("Measured Levels:", section_style))

        # Levels Table
        levels_table_data = [[Paragraph("<b>Parameter</b>", normal_style), Paragraph("<b>Value</b>", normal_style),
                              Paragraph("<b>Target Range</b>", normal_style), Paragraph("<b>Status</b>", normal_style)]]
        for name, value, target, status in levels_data:
            status_text = "Within Range"
            status_color = colors.lightgreen
            if status == "below":
                status_text = "Below Range"
                status_color = colors.lightyellow
            elif status == "above":
                status_text = "Above Range"
                status_color = colors.mistyrose

            levels_table_data.append([
                Paragraph(name, normal_style),
                Paragraph(value, normal_style),
                Paragraph(target, normal_style),
                Paragraph(status_text, ParagraphStyle('StatusStyle', parent=normal_style, backColor=status_color)) # Apply background color directly
            ])

        levels_table = Table(levels_table_data, colWidths=[100, 100, 150, 150]) # Adjusted widths
        levels_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), # Header background
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Bold header
        ]))
        # This loop for coloring is now handled by ParagraphStyle background
        # for i, (_, _, _, status) in enumerate(levels_data, 1):
        #     # ... (color logic removed as it's in ParagraphStyle)
        content.append(levels_table)
        content.append(Spacer(1, 8))

        # Assessment Text
        content.append(Paragraph("Assessment:", section_style))
        content.append(Paragraph(f"Patient is {assessment.upper()}", bold_normal_style)) # Make assessment bold
        content.append(Spacer(1, 12))

        # Calculation Details
        if calculation_details:
            content.append(Paragraph("Calculation Details:", section_style))
            # Use preformatted style for code-like text
            pre_style = styles['Code']
            calc_paragraph = Paragraph(calculation_details.replace('\n', '<br/>'), pre_style)
            content.append(calc_paragraph)
            content.append(Spacer(1, 12))

        # Recommendations
        content.append(Paragraph("Recommendations", heading_style))
        content.append(Paragraph("Dosing:", section_style))
        for rec in dosing_recs:
            content.append(Paragraph(f"• {rec}", normal_style))
        content.append(Spacer(1, 8))

        content.append(Paragraph("Monitoring:", section_style))
        for rec in monitoring_recs:
            content.append(Paragraph(f"• {rec}", normal_style))
        content.append(Spacer(1, 8))

        # Cautions
        if cautions and len(cautions) > 0:
            content.append(Paragraph("Cautions:", section_style))
            for caution in cautions:
                content.append(Paragraph(f"• {caution}", normal_style))
            content.append(Spacer(1, 8))

        # Disclaimer
        content.append(Spacer(1, 20))
        content.append(Paragraph("Disclaimer: This report is generated by an automated system and is intended to assist clinical decision making. Always use professional judgment when implementing recommendations.", disclaimer_style))

        # Build the PDF
        doc.build(content)

        pdf_value = buffer.getvalue()
        buffer.close()
        pdf_base64 = base64.b64encode(pdf_value).decode()
        return pdf_base64

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Function to create a download link for the PDF
def get_pdf_download_link(pdf_base64, filename="clinical_recommendations.pdf"):
    """Create a download link for a base64 encoded PDF"""
    if pdf_base64:
        href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">📄 Download Full Report PDF</a>'
        return href
    return ""

# Updated function to display buttons for printing and downloading recommendations
def display_pdf_download_button(patient_data, drug_info, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """
    Display buttons to print/save recommendations as a PDF and print a summary.
    Requires ReportLab for PDF generation.
    """
    st.markdown("---") # Add a separator
    st.subheader("Export Options")

    col1, col2 = st.columns(2)

    with col1:
        # Generate PDF content when the section loads, not just on button click
        pdf_base64 = None
        if REPORTLAB_AVAILABLE:
             pdf_base64 = create_recommendation_pdf(
                 patient_data, drug_info, levels_data, assessment,
                 dosing_recs, monitoring_recs, calculation_details, cautions
             )

        if pdf_base64:
             download_link = get_pdf_download_link(pdf_base64)
             st.markdown(download_link, unsafe_allow_html=True)
        else:
             st.button("📄 Generate Full Report PDF", disabled=True, help="PDF generation requires ReportLab or failed.")


    with col2:
        if st.button("📋 Generate Clinical Summary"):
            assessment_text = create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details, cautions) # Pass calc details here too
            st.text_area("Copy this text for clinical notes:", assessment_text, height=300)
            st.success("Clinical summary text generated below.")

# Enhanced function to create a printable text assessment
def create_printable_assessment(patient_data, levels_data, assessment, dosing_recs, monitoring_recs, calculation_details=None, cautions=None):
    """Create a plain text printable assessment for easy copying to clinical notes"""
    now = datetime.now()

    # Header
    text = f"ANTIMICROBIAL TDM ASSESSMENT - {now.strftime('%Y-%m-%d %H:%M')}\n"
    text += "=" * 60 + "\n\n"

    # Patient Information
    text += f"Patient ID: {patient_data.get('patient_id', 'N/A')}\n"
    text += f"Ward: {patient_data.get('ward', 'N/A')}\n"
    text += f"Age: {patient_data.get('age', 'N/A')} yrs | Gender: {patient_data.get('gender', 'N/A')} | Weight: {patient_data.get('weight', 'N/A')} kg\n"
    text += f"Diagnosis: {patient_data.get('clinical_diagnosis', 'N/A')}\n"
    text += f"CrCl: {patient_data.get('crcl', 0.0):.1f} mL/min ({patient_data.get('renal_function', 'N/A')})\n"
    text += f"Current Regimen: {patient_data.get('current_dose_regimen', 'N/A')}\n\n"

    # Measured Levels
    text += "MEASURED LEVELS:\n"
    if levels_data:
        for name, value, target, status in levels_data:
            status_symbol = "✅" if status == "within" else "⬇️" if status == "below" else "⬆️"
            text += f"- {name}: {value} (Target: {target}) {status_symbol}\n"
    else:
        text += "- No levels data available.\n"
    text += "\n"

    # Assessment
    text += f"ASSESSMENT: Patient is {assessment.upper()}\n\n"

    # PK Parameters (Extract from calculation_details if provided)
    if calculation_details and isinstance(calculation_details, str):
        text += "PHARMACOKINETIC PARAMETERS (Estimated/Calculated):\n"
        # Use regex to find key parameters in the details string
        ke_match = re.search(r'Ke\s*[:=]\s*([0-9.]+)', calculation_details, re.IGNORECASE)
        thalf_match = re.search(r't½\s*[:=]\s*([0-9.]+)', calculation_details, re.IGNORECASE)
        vd_match = re.search(r'Vd\s*[:=]\s*([0-9.]+)\s*L', calculation_details, re.IGNORECASE)
        cl_match = re.search(r'Cl\s*[:=]\s*([0-9.]+)\s*L/hr', calculation_details, re.IGNORECASE)
        auc_match = re.search(r'AUC24\s*[:=]\s*([0-9.]+)', calculation_details, re.IGNORECASE)

        if ke_match: text += f"- Ke: {float(ke_match.group(1)):.4f} hr⁻¹\n"
        if thalf_match: text += f"- t½: {float(thalf_match.group(1)):.1f} hr\n"
        if vd_match: text += f"- Vd: {float(vd_match.group(1)):.1f} L\n"
        if cl_match: text += f"- Cl: {float(cl_match.group(1)):.2f} L/hr\n"
        if auc_match: text += f"- AUC24: {float(auc_match.group(1)):.1f} mg·hr/L\n"
        text += "\n"

    # Recommendations
    text += "DOSING RECOMMENDATIONS:\n"
    if dosing_recs:
        for rec in dosing_recs:
            text += f"- {rec}\n"
    else:
        text += "- No specific dosing recommendations generated.\n"
    text += "\n"

    text += "MONITORING RECOMMENDATIONS:\n"
    if monitoring_recs:
        for rec in monitoring_recs:
            text += f"- {rec}\n"
    else:
        text += "- No specific monitoring recommendations generated.\n"
    text += "\n"

    # Cautions
    if cautions and len(cautions) > 0:
        text += "CAUTIONS & CONSIDERATIONS:\n"
        for caution in cautions:
            text += f"- {caution}\n"
        text += "\n"

    # Footer
    text += "=" * 60 + "\n"
    text += "Disclaimer: This assessment is intended to assist clinical decision making.\n"
    text += "Always use professional judgment when implementing recommendations.\n"
    text += f"Generated by: Antimicrobial TDM App - {now.strftime('%Y-%m-%d %H:%M')}"

    return text

# ===== VANCOMYCIN INTERPRETATION FUNCTION =====
def generate_vancomycin_interpretation(prompt):
    """
    Generate standardized vancomycin interpretation based on a prompt string.
    Extracts values using regex for better robustness.
    Returns a tuple: (levels_data, assessment, dosing_recs, monitoring_recs, cautions)
    or returns an error string.
    """
    peak_val, trough_val, auc24 = None, None, None
    peak_target_min, peak_target_max = 20, 40 # Default peak target
    trough_target_min, trough_target_max = 10, 20 # Default trough target
    auc_target_min, auc_target_max = 400, 600 # Default AUC target
    new_dose_rec = None

    # Regex patterns to extract values
    patterns = {
        'peak': r'peak\s*[:=]\s*([0-9.]+)',
        'trough': r'trough\s*[:=]\s*([0-9.]+)',
        'auc': r'AUC24\s*[:=]\s*([0-9.]+)',
        'trough_target': r'Target trough range\s*[:=]\s*([0-9]+)\s*-\s*([0-9]+)',
        'peak_target': r'Target peak range\s*[:=]\s*([0-9]+)\s*-\s*([0-9]+)',
        'auc_target': r'Target AUC range\s*[:=]\s*([0-9]+)\s*-\s*([0-9]+)',
        'new_dose': r'(?:Recommended|Suggested) base dose\s*[:=]\s*([0-9.]+)'
    }

    # Extract values using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            try:
                if key == 'peak': peak_val = float(match.group(1))
                elif key == 'trough': trough_val = float(match.group(1))
                elif key == 'auc': auc24 = float(match.group(1))
                elif key == 'trough_target':
                    trough_target_min = float(match.group(1))
                    trough_target_max = float(match.group(2))
                elif key == 'peak_target':
                    peak_target_min = float(match.group(1))
                    peak_target_max = float(match.group(2))
                elif key == 'auc_target':
                    auc_target_min = float(match.group(1))
                    auc_target_max = float(match.group(2))
                elif key == 'new_dose': new_dose_rec = float(match.group(1))
            except (ValueError, IndexError):
                 print(f"Warning: Could not parse value for {key} from prompt.") # Log parsing issues

    # Format targets
    peak_target_str = f"{peak_target_min}-{peak_target_max} mg/L"
    trough_target_str = f"{trough_target_min}-{trough_target_max} mg/L"
    auc_target_str = f"{auc_target_min}-{auc_target_max} mg·hr/L"

    # Determine assessment status based on available data
    assessment = "requires clinical correlation" # Default assessment
    levels_data = []
    monitoring_focus = "trough" # Default focus

    if auc24 is not None:
        monitoring_focus = "AUC"
        auc_status = "within" if auc_target_min <= auc24 <= auc_target_max else ("below" if auc24 < auc_target_min else "above")
        levels_data.append(("AUC24", f"{auc24:.1f} mg·hr/L", auc_target_str, auc_status))
        if auc_status == "below": assessment = "subtherapeutic (low AUC)"
        elif auc_status == "above": assessment = "potentially supratherapeutic (high AUC)"
        else: assessment = "appropriately dosed (AUC-based)"
        # Add estimated trough if available
        if trough_val is not None:
             trough_status = "within" if trough_target_min <= trough_val <= trough_target_max else ("below" if trough_val < trough_target_min else "above")
             levels_data.append(("Estimated Trough", f"{trough_val:.1f} mg/L", trough_target_str, trough_status))

    elif peak_val is not None and trough_val is not None:
        monitoring_focus = "peak & trough"
        peak_status = "within" if peak_target_min <= peak_val <= peak_target_max else ("below" if peak_val < peak_target_min else "above")
        trough_status = "within" if trough_target_min <= trough_val <= trough_target_max else ("below" if trough_val < trough_target_min else "above")
        levels_data.append(("Peak", f"{peak_val:.1f} mg/L", peak_target_str, peak_status))
        levels_data.append(("Trough", f"{trough_val:.1f} mg/L", trough_target_str, trough_status))

        if peak_status == "below" and trough_status == "below": assessment = "subtherapeutic (low peak & trough)"
        elif peak_status == "below": assessment = "potential underdosing (low peak)"
        elif trough_status == "below": assessment = "subtherapeutic (low trough)"
        elif trough_status == "above": assessment = "potentially supratherapeutic (high trough)"
        elif peak_status == "above": assessment = "potentially supratherapeutic (high peak)"
        elif peak_status == "within" and trough_status == "within": assessment = "appropriately dosed"
        else: assessment = "requires adjustment" # Covers mixed scenarios

    elif trough_val is not None:
        monitoring_focus = "trough"
        trough_status = "within" if trough_target_min <= trough_val <= trough_target_max else ("below" if trough_val < trough_target_min else "above")
        levels_data.append(("Trough", f"{trough_val:.1f} mg/L", trough_target_str, trough_status))
        if trough_status == "below": assessment = "subtherapeutic (low trough)"
        elif trough_status == "above": assessment = "potentially supratherapeutic (high trough)"
        else: assessment = "appropriately dosed (trough-based)"

    else:
        return "Insufficient data in prompt to generate Vancomycin interpretation. Need at least Trough, Peak/Trough, or AUC."

    # Generate recommendations
    dosing_recs = []
    monitoring_recs = []
    cautions = []

    # Round recommended dose if available
    rounded_new_dose_str = ""
    if new_dose_rec:
        rounded_dose = round(new_dose_rec / 250) * 250
        rounded_new_dose_str = f"{int(rounded_dose)}mg" # Format as integer string

    # Tailor recommendations based on assessment
    if "subtherapeutic" in assessment or "underdosing" in assessment:
        action = f"INCREASE dose to {rounded_new_dose_str}" if rounded_new_dose_str else "INCREASE dose (e.g., by 25-30%)"
        dosing_recs.append(action)
        dosing_recs.append("CONSIDER shortening dosing interval if appropriate")
        monitoring_recs.append(f"RECHECK levels ({monitoring_focus}) after 3-4 doses")
        cautions.append("Subtherapeutic levels risk treatment failure.")
    elif "supratherapeutic" in assessment:
        action = f"DECREASE dose to {rounded_new_dose_str}" if rounded_new_dose_str else "DECREASE dose (e.g., by 20-25%)"
        dosing_recs.append(action)
        dosing_recs.append("CONSIDER extending dosing interval")
        monitoring_recs.append(f"RECHECK levels ({monitoring_focus}) after 3-4 doses")
        monitoring_recs.append("MONITOR renal function closely")
        monitoring_recs.append("ASSESS for signs/symptoms of nephrotoxicity")
        cautions.append("Elevated levels increase risk of nephrotoxicity.")
    elif "appropriately dosed" in assessment:
        dosing_recs.append("CONTINUE current dosing regimen")
        monitoring_recs.append(f"ROUTINE monitoring: Recheck levels ({monitoring_focus}) if clinical status or renal function changes significantly.")
        cautions.append("Monitor for adverse effects even with therapeutic levels.")
    else: # requires adjustment / requires clinical correlation
        action = f"ADJUST dose towards {rounded_new_dose_str}" if rounded_new_dose_str else "ADJUST dose based on clinical picture and levels"
        dosing_recs.append(action)
        monitoring_recs.append(f"RECHECK levels ({monitoring_focus}) after adjustment")
        cautions.append("Individualize therapy based on clinical response and specific targets.")

    # Standard monitoring
    monitoring_recs.append("MONITOR renal function regularly (e.g., every 2-3 days or as clinically indicated)")

    return levels_data, assessment, dosing_recs, monitoring_recs, cautions

# ===== AMINOGLYCOSIDE INTERPRETATION FUNCTION =====
def generate_aminoglycoside_interpretation(prompt):
    """
    Generate standardized aminoglycoside interpretation based on a prompt string.
    Extracts values using regex for better robustness.
    Returns a tuple: (levels_data, assessment, dosing_recs, monitoring_recs, cautions)
    or returns an error string.
    """
    drug_name = "Aminoglycoside"
    peak_val, trough_val = None, None
    peak_target_min, peak_target_max = 5, 10 # Default MDD Gentamicin peak target
    trough_target_max = 2.0 # Default MDD Gentamicin trough target
    new_dose_rec, new_interval_rec = None, None
    regimen = "MDD" # Default regimen

    # Identify drug and regimen
    if "Gentamicin" in prompt: drug_name = "Gentamicin"
    elif "Amikacin" in prompt: drug_name = "Amikacin"

    if "SDD" in prompt: regimen = "SDD"
    elif "Synergy" in prompt: regimen = "Synergy"
    elif "MDD" in prompt: regimen = "MDD"

    # Set targets based on drug and regimen
    if drug_name == "Gentamicin":
        if regimen == "SDD": peak_target_min, peak_target_max, trough_target_max = 10, 30, 1.0
        elif regimen == "Synergy": peak_target_min, peak_target_max, trough_target_max = 3, 5, 1.0
        else: peak_target_min, peak_target_max, trough_target_max = 5, 10, 2.0 # MDD
    elif drug_name == "Amikacin":
        if regimen == "SDD": peak_target_min, peak_target_max, trough_target_max = 60, 80, 1.0
        else: peak_target_min, peak_target_max, trough_target_max = 20, 30, 10.0 # MDD (Synergy less common)

    # Regex patterns
    patterns = {
        'peak': r'(?:peak|Cmax)\s*[:=]\s*([0-9.]+)',
        'trough': r'(?:trough|Cmin)\s*[:=]\s*([0-9.]+)',
        'peak_target': r'Target peak range\s*[:=]\s*([0-9]+)\s*-\s*([0-9]+)',
        'trough_target': r'Target trough\s*[:=]\s*<\s*([0-9.]+)',
        'new_dose': r'(?:Recommended|Suggested) new dose\s*[:=]\s*([0-9.]+)',
        'new_interval': r'(?:Recommended|Suggested) new interval\s*[:=]\s*([0-9]+)'
    }

    # Extract values
    for key, pattern in patterns.items():
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            try:
                if key == 'peak': peak_val = float(match.group(1))
                elif key == 'trough': trough_val = float(match.group(1))
                elif key == 'peak_target':
                    peak_target_min = float(match.group(1))
                    peak_target_max = float(match.group(2))
                elif key == 'trough_target': trough_target_max = float(match.group(1))
                elif key == 'new_dose': new_dose_rec = float(match.group(1))
                elif key == 'new_interval': new_interval_rec = int(match.group(1))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse value for {key} from prompt.")

    # Format targets
    peak_target_str = f"{peak_target_min}-{peak_target_max} mg/L"
    trough_target_str = f"<{trough_target_max} mg/L"

    # Require both peak and trough for interpretation
    if peak_val is None or trough_val is None:
        # Allow interpretation for SDD without levels if needed (based on Hartford etc.)
        if regimen == "SDD" and "Hartford" in prompt:
             # Placeholder - Add logic to interpret based on Hartford nomogram text if needed
             assessment = "requires assessment based on nomogram/timing"
             levels_data = [("Level (Time Unknown)", "N/A", "N/A", "N/A")] # Indicate level was measured but not peak/trough
        else:
             return f"Insufficient data in prompt for {drug_name} interpretation. Need Peak and Trough levels for {regimen}."


    # Determine assessment status
    assessment = "requires clinical correlation" # Default
    peak_status = "within" if peak_target_min <= peak_val <= peak_target_max else ("below" if peak_val < peak_target_min else "above")
    trough_status = "within" if trough_val < trough_target_max else "above" # Only care if trough is above max

    levels_data = [
        ("Peak", f"{peak_val:.1f} mg/L", peak_target_str, peak_status),
        ("Trough", f"{trough_val:.2f} mg/L", trough_target_str, trough_status)
    ]

    if peak_status == "below" and trough_status == "above": assessment = "ineffective and potentially toxic (low peak, high trough)"
    elif peak_status == "below": assessment = "subtherapeutic (inadequate peak)"
    elif trough_status == "above": assessment = "potentially toxic (elevated trough)"
    elif peak_status == "above": assessment = "potentially toxic (elevated peak)"
    elif peak_status == "within" and trough_status == "within": assessment = "appropriately dosed"
    else: assessment = "requires adjustment" # Mixed scenarios

    # Generate recommendations
    dosing_recs = []
    monitoring_recs = []
    cautions = []

    # Format recommended dose/interval if available
    rounded_new_dose_str = ""
    new_interval_str = f"q{new_interval_rec}h" if new_interval_rec else "(interval adjustment may be needed)"
    if new_dose_rec:
        rounded_dose = round(new_dose_rec / 10) * 10 # Round to nearest 10mg
        rounded_new_dose_str = f"{int(rounded_dose)}mg {new_interval_str}"

    # Tailor recommendations
    if "subtherapeutic" in assessment or "ineffective" in assessment:
        action = f"INCREASE dose to {rounded_new_dose_str}" if rounded_new_dose_str else "INCREASE dose (e.g., by 25-50%)"
        dosing_recs.append(action)
        if "ineffective" in assessment: dosing_recs.append("EXTEND interval if trough is high")
        monitoring_recs.append("RECHECK peak and trough after 2-3 doses")
        cautions.append("Subtherapeutic levels risk treatment failure.")
    elif "toxic" in assessment:
        if "trough" in assessment:
             action = f"EXTEND dosing interval to {new_interval_str}" if new_interval_rec else "EXTEND dosing interval"
             dosing_recs.append(action)
             if rounded_new_dose_str: dosing_recs.append(f"CONSIDER decreasing dose to {rounded_new_dose_str}")
             cautions.append("Elevated trough increases risk of nephrotoxicity and ototoxicity.")
        elif "peak" in assessment:
             action = f"DECREASE dose to {rounded_new_dose_str}" if rounded_new_dose_str else "DECREASE dose (e.g., by 20-25%)"
             dosing_recs.append(action)
             cautions.append("Elevated peak may increase risk of ototoxicity.")
        monitoring_recs.append("RECHECK peak and trough after adjustment")
        monitoring_recs.append("MONITOR renal function and hearing closely")
    elif "appropriately dosed" in assessment:
        dosing_recs.append("CONTINUE current dosing regimen")
        monitoring_recs.append("ROUTINE monitoring: Recheck levels if clinical status or renal function changes.")
        if regimen == "MDD": monitoring_recs.append("CONSIDER switching to SDD if appropriate for indication and duration.")
    else: # requires adjustment / requires clinical correlation
        action = f"ADJUST dose/interval towards {rounded_new_dose_str}" if rounded_new_dose_str else "ADJUST dose/interval based on clinical picture and levels"
        dosing_recs.append(action)
        monitoring_recs.append("RECHECK peak and trough after adjustment")
        cautions.append("Individualize therapy based on clinical response and specific targets.")

    # Standard monitoring
    monitoring_recs.append("MONITOR renal function regularly (e.g., every 1-3 days)")
    monitoring_recs.append("MONITOR for signs/symptoms of ototoxicity (hearing loss, tinnitus, vertigo)")

    return levels_data, assessment, dosing_recs, monitoring_recs, cautions


# ===== FORMAT_CLINICAL_RECOMMENDATIONS FUNCTION =====
def format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions=None):
    """
    Create standardized recommendation format with clear visual hierarchy using markdown.
    """
    # Format measured levels with status indicators
    levels_md = "📊 **MEASURED LEVELS:**\n"
    if levels_data:
        for name, value, target, status in levels_data:
            # Use emojis for status indicators
            icon = "✅" if status == "within" else "⬇️" if status == "below" else "⬆️"
            levels_md += f"- {name}: **{value}** (Target: {target}) {icon}\n"
    else:
        levels_md += "- *No level data processed for interpretation.*\n"

    # Format overall assessment
    assessment_md = f"⚕️ **ASSESSMENT:**\n**{assessment.upper()}**"

    # Combine into full recommendation format
    output = f"""## CLINICAL ASSESSMENT & RECOMMENDATIONS

{levels_md}
{assessment_md}

---

### DETAILED RECOMMENDATIONS

🔵 **DOSING RECOMMENDATIONS:**
"""
    if dosing_recs:
        for rec in dosing_recs:
            output += f"- {rec}\n"
    else:
        output += "- *No specific dosing recommendations generated.*\n"

    output += "\n🔵 **MONITORING RECOMMENDATIONS:**\n"
    if monitoring_recs:
        for rec in monitoring_recs:
            output += f"- {rec}\n"
    else:
        output += "- *No specific monitoring recommendations generated.*\n"

    if cautions and len(cautions) > 0:
        output += "\n⚠️ **CAUTIONS & CONSIDERATIONS:**\n"
        for caution in cautions:
            output += f"- {caution}\n"

    # Add a summary section for quick reference
    output += "\n---\n" # Separator
    output += "### QUICK SUMMARY\n"
    output += f"**Status:** {assessment.upper()}\n"

    # Summarize key recommendations
    if dosing_recs:
        output += f"**Key Dosing Action:** {dosing_recs[0]}\n"
    if monitoring_recs:
        output += f"**Key Monitoring Action:** {monitoring_recs[0]}\n"

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    output += f"\n*Generated on: {timestamp}*"

    return output

# ===== STANDARDIZED INTERPRETATION GENERATOR (Router) =====
def generate_standardized_interpretation(prompt, drug):
    """
    Generate a standardized interpretation based on drug type and prompt content.
    Calls specific drug interpretation functions.
    """
    drug_lower = drug.lower()
    if "vancomycin" in drug_lower:
        return generate_vancomycin_interpretation(prompt)
    elif "aminoglycoside" in drug_lower or "gentamicin" in drug_lower or "amikacin" in drug_lower:
        return generate_aminoglycoside_interpretation(prompt)
    else:
        # Fallback for unknown drugs
        levels_data = [("Unknown Drug", "N/A", "N/A", "N/A")]
        assessment = "requires specific assessment for this drug"
        dosing_recs = ["CONSULT pharmacist/specialist", "FOLLOW institutional guidelines"]
        monitoring_recs = ["OBTAIN appropriate levels based on drug type", "MONITOR clinical response and toxicity"]
        cautions = ["Standard TDM principles may not apply.", "Verify drug identity."]
        return levels_data, assessment, dosing_recs, monitoring_recs, cautions

# ===== IMPROVED CLINICAL INTERPRETATION FUNCTION (Wrapper) =====
# This function seems redundant if generate_standardized_interpretation handles the logic.
# Kept for compatibility with original structure, but consider refactoring.
def interpret_with_llm(prompt, patient_data=None, calculation_details=None):
    """
    Enhanced clinical interpretation function. Uses OpenAI if available,
    otherwise falls back to generate_standardized_interpretation.
    """
    # Extract drug type and method for display/PDF
    drug = "Antimicrobial"
    method = "Standard method"
    if "Vancomycin" in prompt:
        drug = "Vancomycin"
        if "Trough only" in prompt: method = "Trough-only"
        elif "Peak and Trough" in prompt: method = "Peak and Trough"
        elif "AUC-guided" in prompt: method = "AUC-guided"
    elif "Gentamicin" in prompt: drug = "Gentamicin"
    elif "Amikacin" in prompt: drug = "Amikacin"
    elif "Aminoglycoside" in prompt: drug = "Aminoglycoside"

    if drug != "Antimicrobial":
        if "SDD" in prompt: method = "SDD"
        elif "Synergy" in prompt: method = "Synergy"
        elif "MDD" in prompt: method = "MDD"

    drug_info = f"{drug} ({method})"

    # Check if OpenAI API is available and configured
    if OPENAI_AVAILABLE:
        try:
            # Updated prompt for structured output (Example)
            structured_prompt = f"""
            Provide a concise, structured clinical interpretation for this antimicrobial TDM case.
            Format your response using markdown with these exact sections:
            ## CLINICAL ASSESSMENT
            **MEASURED LEVELS:** (list each with target range and status icon ✅⬇️⬆️)
            **ASSESSMENT:** (state if appropriately dosed, subtherapeutic, or potentially toxic)
            ## RECOMMENDATIONS
            **DOSING:** (action-oriented recommendations)
            **MONITORING:** (specific parameters and schedule)
            **CAUTIONS:** (relevant warnings, if any)

            Case: {prompt}
            """

            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-4", # Or your preferred model like gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are an expert clinical pharmacist specializing in TDM. Provide concise, evidence-based interpretations with clear recommendations."},
                    {"role": "user", "content": structured_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            llm_response = response.choices[0].message.content

            st.markdown("---")
            st.subheader("Clinical Interpretation (AI Generated)")
            st.markdown(llm_response)
            st.info("Interpretation provided by OpenAI. Always verify with clinical judgment.")
            # Skip standardized formatting and PDF for LLM response for simplicity here
            return # Exit after displaying LLM response

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            st.warning("Falling back to rule-based clinical interpretation.")

    # --- Fallback to Rule-Based Interpretation ---
    st.markdown("---")
    st.subheader("Clinical Interpretation (Rule-Based)")
    interpretation_data = generate_standardized_interpretation(prompt, drug)

    if isinstance(interpretation_data, str): # Handle error string
        st.error(interpretation_data)
        return

    # Unpack the interpretation data
    levels_data, assessment, dosing_recs, monitoring_recs, cautions = interpretation_data

    # Display the formatted interpretation
    formatted_interpretation = format_clinical_recommendations(levels_data, assessment, dosing_recs, monitoring_recs, cautions)
    st.markdown(formatted_interpretation) # Use markdown for better formatting

    # Add the PDF download button if patient_data is provided
    if patient_data and REPORTLAB_AVAILABLE:
        display_pdf_download_button(
            patient_data, drug_info, levels_data, assessment,
            dosing_recs, monitoring_recs, calculation_details, cautions
        )
    elif patient_data:
        st.warning("PDF generation disabled because ReportLab library is not installed.")

    # Add the raw prompt at the bottom for debugging
    with st.expander("View Raw Data Used for Interpretation", expanded=False):
        st.code(prompt)


# ===== VANCOMYCIN METHODS =====
def vancomycin_trough_only(patient_data):
    """Vancomycin trough-only monitoring method"""
    st.markdown("---")
    st.write("##### Trough-Only Monitoring")
    st.info("Trough-only monitoring is a traditional approach. AUC-guided is now preferred by guidelines.")

    target_trough_strategy = st.radio(
        "Target Trough Range",
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"],
        key="vanco_trough_target", help="Select appropriate target based on indication"
    )
    target_cmin = (10, 15) if "Empirical" in target_trough_strategy else (15, 20)

    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=5000, value=1000, step=250, key="vanco_trough_dose")
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=72, value=12, step=4, key="vanco_trough_interval")
    with col2:
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=100.0, value=12.5, step=0.1, key="vanco_trough_level")
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5, key="vanco_trough_infusion")

    if st.button("Calculate Vancomycin Trough Dosing", key="vanco_trough_calc"):
        with st.spinner("Performing trough-based calculations..."):
            crcl = patient_data.get('crcl', 0)
            weight = patient_data.get('weight', 0)
            if crcl <= 0 or weight <= 0:
                st.error("Valid CrCl and Weight are required for calculations.")
                return

            # Estimate PK parameters (using population estimates - less accurate)
            ke = 0.00083 * crcl + 0.0044
            vd = 0.7 * weight
            if ke <= 0:
                st.error("Estimated Ke is non-positive. Cannot proceed with calculation.")
                return
            t_half = 0.693 / ke
            cl = ke * vd
            tau = interval

            # Estimate peak and AUC (highly approximate for trough-only)
            try:
                 # Formula requires non-zero ke, tau, vd
                 if vd > 0 and tau > 0 and ke > 0:
                      # Simplified peak estimation
                      factor = 1 - math.exp(-ke * tau)
                      if factor > 1e-9: # Avoid division by zero
                           peak_est = (dose / vd) * (1 - math.exp(-ke * infusion_time)) / factor if infusion_time > 0 else (dose / vd) / factor
                           auc24_est = (dose / cl) * (24 / tau) if cl > 0 and tau > 0 else 0
                      else:
                           peak_est = 0
                           auc24_est = 0
                 else:
                      peak_est = 0
                      auc24_est = 0

            except Exception as e:
                 st.warning(f"Could not estimate peak/AUC: {e}")
                 peak_est = 0
                 auc24_est = 0


            # Calculate new dose to reach target trough (using estimated Cl)
            target_trough = (target_cmin[0] + target_cmin[1]) / 2
            new_dose = 0
            if cl > 0 and tau > 0 and ke > 0:
                 try:
                      # Formula: Dose = Cpss_min * Cl * tau / (exp(-ke*inf_time)*(1-exp(-ke*tau))) # More complex
                      # Simplified approach: Dose ~ Target * Vd * (1-exp(-ke*tau)) / exp(-ke*(tau-inf_time))
                      # Using proportional adjustment based on current trough vs target
                      if trough > 0: # Avoid division by zero
                           new_dose = dose * (target_trough / trough)
                      else:
                           # If current trough is 0, estimate based on target and pop PK
                           factor = 1 - math.exp(-ke * tau)
                           if factor > 1e-9:
                                new_dose = target_trough * vd * factor / math.exp(-ke * (tau - max(infusion_time, 0.5)))
                           else:
                                new_dose = 0 # Cannot calculate
                 except Exception as e:
                      st.warning(f"Error calculating new dose: {e}")
                      new_dose = 0
            else:
                 st.warning("Cannot calculate new dose due to invalid PK parameters.")


            practical_new_dose = round(new_dose / 250) * 250 if new_dose > 0 else 0

            # Display results
            st.success("Vancomycin Trough Analysis Complete (using population estimates)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Measured Trough", f"{trough:.1f} mg/L")
                st.metric("Target Trough", f"{target_cmin[0]}-{target_cmin[1]} mg/L")
                if trough < target_cmin[0]: st.warning("⚠️ Trough below target")
                elif trough > target_cmin[1]: st.warning("⚠️ Trough above target")
                else: st.success("✅ Trough within target")
            with col2:
                st.metric("Est. Ke", f"{ke:.4f} hr⁻¹")
                st.metric("Est. t½", f"{t_half:.1f} hr")
                st.metric("Est. Vd", f"{vd:.1f} L")
            with col3:
                st.metric("Est. Cl", f"{cl:.2f} L/hr")
                st.metric("Est. AUC24", f"{auc24_est:.0f} mg·hr/L", help="Highly approximate")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{tau}h", help="Based on target trough")

            # Interpretation
            calculation_details = f"""
            Method: Trough-Only (Population Estimates)
            Est. Ke = {ke:.4f} hr⁻¹
            Est. t½ = {t_half:.1f} hr
            Est. Vd = {vd:.1f} L
            Est. Cl = {cl:.2f} L/hr
            Measured Trough = {trough:.1f} mg/L
            Target Trough = {target_cmin[0]}-{target_cmin[1]} mg/L
            Est. AUC24 = {auc24_est:.0f} mg·hr/L (Approx.)
            Recommended Dose = {practical_new_dose:.0f} mg q{tau}h (to target trough)
            """
            prompt = (
                f"Vancomycin (Trough only): Measured trough = {trough} mg/L, "
                f"Interval = {tau} hr, Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                f"Recommended base dose = {practical_new_dose:.0f} mg."
            )
            interpret_with_llm(prompt, patient_data, calculation_details)


def vancomycin_peak_trough(patient_data):
    """Vancomycin peak and trough monitoring method"""
    st.markdown("---")
    st.write("##### Peak & Trough Monitoring")
    st.info("Uses measured peak and trough for individualized PK parameter calculation.")

    target_trough_strategy = st.radio(
        "Target Trough Range",
        ["Empirical (10-15 mg/L)", "Definitive (15-20 mg/L)"],
        key="vanco_peak_trough_target", help="Select appropriate target based on indication"
    )
    target_cmin = (10, 15) if "Empirical" in target_trough_strategy else (15, 20)
    # Peak target is less emphasized now, but can be estimated/shown
    target_peak = (25, 40) # Example range, often derived from AUC goals

    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=5000, value=1000, step=250, key="vanco_pt_dose")
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=72, value=12, step=4, key="vanco_pt_interval")
        peak = st.number_input("Measured Peak (mg/L)", min_value=0.1, max_value=100.0, value=25.0, step=0.1, key="vanco_pt_peak") # Min > 0
    with col2:
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5, key="vanco_pt_infusion")
        peak_draw_time = st.number_input("Time After START of Infusion for Peak (hours)", min_value=infusion_time + 0.1, max_value=6.0, value=1.5, step=0.25, key="vanco_pt_peak_time", help="Must be after infusion ends")
        trough = st.number_input("Measured Trough (mg/L)", min_value=0.0, max_value=100.0, value=12.5, step=0.1, key="vanco_pt_trough")

    if st.button("Calculate Vancomycin Peak-Trough Dosing", key="vanco_pt_calc"):
        with st.spinner("Performing peak-trough calculations..."):
            weight = patient_data.get('weight', 0)
            if weight <= 0:
                st.error("Valid Weight is required for Vd/kg calculation.")
                return
            if peak <= trough or peak <=0 or trough < 0:
                 st.error("Invalid levels: Peak must be > Trough and > 0.")
                 return
            if interval <= peak_draw_time:
                 st.error("Dosing interval must be longer than the peak draw time.")
                 return

            # Calculate individualized PK parameters
            t_peak_actual = peak_draw_time # Time from start of infusion
            tau = interval
            delta_t = tau - t_peak_actual # Time between peak draw and trough draw (end of interval)

            if delta_t <= 0:
                 st.error(f"Time between peak draw ({t_peak_actual}h) and trough ({tau}h) is not positive.")
                 return

            try:
                ke = -math.log(trough / peak) / delta_t
                if ke <= 0: raise ValueError("Calculated Ke is non-positive.")
                t_half = 0.693 / ke

                # Estimate Cmax at end of infusion (back-extrapolate)
                time_from_inf_end_to_peak = t_peak_actual - infusion_time
                if time_from_inf_end_to_peak < 0:
                     st.warning("Peak drawn during infusion? Calculation assumes peak is post-infusion.")
                     # Attempt to estimate peak at infusion end based on level during infusion (less accurate)
                     # This requires more complex model assumptions, simplified here
                     c_max_est = peak # Use measured peak as approximation if drawn during infusion
                else:
                     c_max_est = peak * math.exp(ke * time_from_inf_end_to_peak)

                # Calculate Vd using Cmax_est (Sawchuk-Zaske method adaptation)
                factor = 1 - math.exp(-ke * tau)
                if factor < 1e-9: raise ValueError("Factor for Vd calculation is near zero.")
                vd = (dose / (ke * infusion_time)) * ( (1 - math.exp(-ke * infusion_time)) / (1 - math.exp(-ke*tau)) ) * (1 - (trough/c_max_est)*math.exp(ke*infusion_time))
                # Simpler Vd estimation if the above is complex/unstable:
                # vd = dose / c_max_est # Very rough estimate

                if vd <= 0: raise ValueError("Calculated Vd is non-positive.")
                cl = ke * vd
                auc_tau = dose / cl if cl > 0 else 0 # AUC over one interval
                auc24 = auc_tau * (24 / tau) if tau > 0 else 0

            except (ValueError, OverflowError, ZeroDivisionError) as e:
                st.error(f"Calculation Error: {e}. Check input values (Peak > Trough > 0, Times).")
                return

            # Calculate new dose to reach target trough
            target_trough = (target_cmin[0] + target_cmin[1]) / 2
            new_dose = 0
            try:
                 # Use calculated individual PK parameters
                 factor = 1 - math.exp(-ke * tau)
                 if factor > 1e-9:
                      # Dose = Cpss_target * Vd * factor / exp(-ke*(tau-inf_time)) # Target trough based
                      # Target AUC based: Dose = TargetAUC24 * Cl * tau / 24
                      # Let's target trough for this method as per traditional approach
                       new_dose = target_trough * vd * factor / math.exp(-ke * (tau - infusion_time))
                 else: new_dose = 0
            except Exception as e:
                 st.warning(f"Could not calculate new dose: {e}")
                 new_dose = 0

            practical_new_dose = round(new_dose / 250) * 250 if new_dose > 0 else 0

            # Display results
            st.success("Vancomycin Peak-Trough Analysis Complete (Individualized PK)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Measured Peak", f"{peak:.1f} mg/L (at {t_peak_actual}h)")
                st.metric("Measured Trough", f"{trough:.1f} mg/L (at {tau}h)")
                if trough < target_cmin[0]: st.warning("⚠️ Trough below target")
                elif trough > target_cmin[1]: st.warning("⚠️ Trough above target")
                else: st.success("✅ Trough within target")
            with col2:
                st.metric("Calc. Ke", f"{ke:.4f} hr⁻¹")
                st.metric("Calc. t½", f"{t_half:.1f} hr")
                st.metric("Calc. Vd", f"{vd:.1f} L ({vd/weight:.2f} L/kg)")
            with col3:
                st.metric("Calc. Cl", f"{cl:.2f} L/hr")
                st.metric("Calc. AUC24", f"{auc24:.0f} mg·hr/L")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{tau}h", help="To target trough")

            # Visualization
            st.subheader("Concentration-Time Curve (Individualized)")
            chart = plot_concentration_time_curve(
                f"Vancomycin (Peak-Trough, {'Empirical' if target_cmin[1]<=15 else 'Definitive'})",
                [], assessment, [], [], "", # Pass empty lists/strings for unused plot args
                peak=peak, trough=trough, ke=ke, tau=tau, t_peak=t_peak_actual, infusion_time=infusion_time
            )
            if chart:
                st.altair_chart(chart, use_container_width=True)

            # Interpretation
            calculation_details = f"""
            Method: Peak & Trough (Individualized PK)
            Measured Peak = {peak:.1f} mg/L at {t_peak_actual} hrs
            Measured Trough = {trough:.1f} mg/L at {tau} hrs
            Calc. Ke = {ke:.4f} hr⁻¹
            Calc. t½ = {t_half:.1f} hr
            Calc. Vd = {vd:.1f} L ({vd/weight:.2f} L/kg)
            Calc. Cl = {cl:.2f} L/hr
            Calc. AUC24 = {auc24:.0f} mg·hr/L
            Target Trough = {target_cmin[0]}-{target_cmin[1]} mg/L
            Recommended Dose = {practical_new_dose:.0f} mg q{tau}h (to target trough)
            """
            prompt = (
                 f"Vancomycin (Peak and Trough): Measured peak = {peak} mg/L, trough = {trough} mg/L, "
                 f"Interval = {tau} hr, Ke = {ke:.4f} hr⁻¹, AUC24 = {auc24:.0f} mg·hr/L, "
                 f"Target trough range = {target_cmin[0]}-{target_cmin[1]} mg/L, "
                 # Include calculated peak target range if desired
                 f"Target peak range = {target_peak[0]}-{target_peak[1]} mg/L, Recommended base dose = {practical_new_dose:.0f} mg."
            )
            interpret_with_llm(prompt, patient_data, calculation_details)


def vancomycin_auc_guided(patient_data):
    """Vancomycin AUC-guided monitoring method using two levels"""
    st.markdown("---")
    st.write("##### AUC-Guided Monitoring (Two Levels)")
    st.info("Preferred approach using two post-dose levels for accurate AUC calculation.")

    target_auc_strategy = st.radio(
        "Target AUC24 Range",
        ["400-600 mg·hr/L (Standard)", "500-700 mg·hr/L (Serious/CNS)"],
        key="vanco_auc_target", help="Select target based on infection type/severity (per 2020 guidelines)"
    )
    target_auc = (400, 600) if "Standard" in target_auc_strategy else (500, 700)

    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Current Dose (mg)", min_value=250, max_value=5000, value=1000, step=250, key="vanco_auc_dose")
        interval = st.number_input("Dosing Interval (hours)", min_value=6, max_value=72, value=12, step=4, key="vanco_auc_interval")
        first_level = st.number_input("First Concentration (mg/L)", min_value=0.1, max_value=100.0, value=25.0, step=0.1, key="vanco_auc_level1") # Min > 0
        first_time = st.number_input("Time After START of Infusion for First Sample (hours)", min_value=0.6, max_value=12.0, value=2.0, step=0.5, key="vanco_auc_time1", help="Typically 1-2 hours post-infusion")
    with col2:
        infusion_time = st.number_input("Infusion Duration (hours)", min_value=0.5, max_value=4.0, value=1.0, step=0.5, key="vanco_auc_infusion")
        second_level = st.number_input("Second Concentration (mg/L)", min_value=0.1, max_value=100.0, value=15.0, step=0.1, key="vanco_auc_level2") # Min > 0
        second_time = st.number_input("Time After START of Infusion for Second Sample (hours)", min_value=first_time + 1.0, max_value=24.0, value=6.0, step=0.5, key="vanco_auc_time2", help="Typically 4-8 hours after first sample")

    if st.button("Calculate Vancomycin AUC Dosing", key="vanco_auc_calc"):
        with st.spinner("Performing AUC calculations..."):
            weight = patient_data.get('weight', 0)
            if weight <= 0:
                st.error("Valid Weight is required for Vd/kg calculation.")
                return
            if first_level <= 0 or second_level <= 0 or first_level <= second_level:
                 st.error("Invalid levels: First level must be > Second level and both > 0.")
                 return
            if second_time <= first_time:
                 st.error("Time for second sample must be after the first sample.")
                 return

            # Calculate individualized PK from two levels
            delta_time = second_time - first_time
            if delta_time <= 0:
                 st.error("Time difference between samples must be positive.")
                 return

            try:
                ke = -math.log(second_level / first_level) / delta_time
                if ke <= 0: raise ValueError("Calculated Ke is non-positive.")
                t_half = 0.693 / ke

                # Estimate Cmax at end of infusion (back-extrapolate from first level)
                time_from_inf_end_to_first = first_time - infusion_time
                if time_from_inf_end_to_first < 0:
                     st.warning("First level drawn during infusion? Calculation assumes post-infusion levels.")
                     # Use first level as rough Cmax_est if drawn during infusion
                     c_max_est = first_level
                else:
                     c_max_est = first_level * math.exp(ke * time_from_inf_end_to_first)


                # Estimate Cmin (trough) at end of interval (extrapolate from second level)
                time_from_second_to_trough = interval - second_time
                if time_from_second_to_trough < 0:
                     st.warning("Second level drawn after end of interval? Check timing.")
                     # Use second level as rough trough if drawn late
                     trough_est = second_level
                else:
                     trough_est = second_level * math.exp(-ke * time_from_second_to_trough)


                # Calculate Vd using Bayesian approach (preferred) or algebraic method
                # Algebraic method (less precise than Bayesian software):
                # Vd = (Dose / infusion_time) * (1 - math.exp(-ke * infusion_time)) / (ke * (Cmax_est - Trough_est * math.exp(-ke * (interval - infusion_time)))) # Complex formula
                # Simplified Vd based on Cmax_est:
                vd = dose / c_max_est # Very rough estimate, use with caution
                if vd <= 0: raise ValueError("Estimated Vd is non-positive.")

                cl = ke * vd
                auc_tau = dose / cl if cl > 0 else 0 # AUC over one interval
                auc24 = auc_tau * (24 / interval) if interval > 0 else 0

            except (ValueError, OverflowError, ZeroDivisionError) as e:
                st.error(f"Calculation Error: {e}. Check input values and timings.")
                return

            # Calculate new dose to reach target AUC24
            target_auc24_mid = (target_auc[0] + target_auc[1]) / 2
            new_dose = 0
            if cl > 0 and interval > 0:
                 new_dose = (target_auc24_mid * cl * interval) / 24
            else:
                 st.warning("Cannot calculate new dose due to invalid PK parameters.")

            practical_new_dose = round(new_dose / 250) * 250 if new_dose > 0 else 0

            # Display results
            st.success("Vancomycin AUC Analysis Complete (Individualized PK)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("First Level", f"{first_level:.1f} mg/L at {first_time}h")
                st.metric("Second Level", f"{second_level:.1f} mg/L at {second_time}h")
                st.metric("Calculated AUC24", f"{auc24:.0f} mg·hr/L")
                if auc24 < target_auc[0]: st.warning("⚠️ AUC below target")
                elif auc24 > target_auc[1]: st.warning("⚠️ AUC above target")
                else: st.success("✅ AUC within target")
            with col2:
                st.metric("Calc. Ke", f"{ke:.4f} hr⁻¹")
                st.metric("Calc. t½", f"{t_half:.1f} hr")
                st.metric("Est. Trough", f"{trough_est:.1f} mg/L")
            with col3:
                st.metric("Est. Vd", f"{vd:.1f} L ({vd/weight:.2f} L/kg)", help="Approximate Vd")
                st.metric("Calc. Cl", f"{cl:.2f} L/hr")
                st.metric("Recommended Dose", f"{practical_new_dose:.0f} mg q{interval}h", help="To target AUC")

            # Visualization - Use estimated peak/trough for plotting
            st.subheader("Concentration-Time Curve (Individualized)")
            chart = plot_concentration_time_curve(
                f"Vancomycin (AUC-Guided, Target {target_auc[0]}-{target_auc[1]})",
                [], assessment, [], [], "", # Pass empty lists/strings for unused plot args
                peak=c_max_est, trough=trough_est, ke=ke, tau=interval, t_peak=infusion_time, infusion_time=infusion_time
            )
            if chart:
                 # Add measured points to the plot
                 points = alt.Chart(pd.DataFrame({
                     'Time (hr)': [first_time, second_time],
                     'Concentration (mg/L)': [first_level, second_level],
                     'Label': ['Level 1', 'Level 2']
                 })).mark_point(size=100, filled=True, color='red').encode(
                     x='Time (hr)',
                     y='Concentration (mg/L)',
                     tooltip=['Label', 'Time (hr)', 'Concentration (mg/L)']
                 )
                 st.altair_chart(chart + points, use_container_width=True)


            # Interpretation
            calculation_details = f"""
            Method: AUC-Guided (Two Levels, Individualized PK)
            Level 1 = {first_level:.1f} mg/L at {first_time} hrs
            Level 2 = {second_level:.1f} mg/L at {second_time} hrs
            Calc. Ke = {ke:.4f} hr⁻¹
            Calc. t½ = {t_half:.
