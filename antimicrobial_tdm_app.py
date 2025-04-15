import streamlit as st
import numpy as np
import math
import openai
import faiss
import pickle
import os

st.set_page_config(page_title="Antimicrobial TDM App", layout="wide")

# ===== API CONFIGURATION =====
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=openai.api_key)

# ===== LOAD EMBEDDED GUIDELINE =====
@st.cache_resource
def load_guideline_embeddings():
    index = faiss.read_index("guideline_index.faiss")
    with open("guideline_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

guideline_index, guideline_chunks = load_guideline_embeddings()

# ===== NAVIGATION =====
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Module", [
    "Aminoglycoside: Initial Dose",
    "Aminoglycoside: Conventional Dosing (C1/C2)",
    "Vancomycin: AUC-Based Dosing"
])

# ===== PATIENT CONTEXT =====
st.sidebar.title("🩺 Patient Clinical Info")
clinical_diagnosis = st.sidebar.text_input("Diagnosis")
renal_status = st.sidebar.selectbox("Renal Function Status", ["Normal", "Mild Impairment", "Moderate", "Severe", "On Dialysis"])
current_dose_regimen = st.sidebar.text_area("Current Dosing Regimen", value="1g IV q12h")
notes = st.sidebar.text_area("Other Clinical Notes", value="No known allergies.")
clinical_summary = f"Diagnosis: {clinical_diagnosis}\nRenal status: {renal_status}\nCurrent regimen: {current_dose_regimen}\nNotes: {notes}"

# ===== INTERPRETATION FUNCTION =====
def interpret_with_llm(prompt):
    try:
        embedding_response = client.embeddings.create(input=[prompt], model="text-embedding-3-small")
        query_vector = np.array(embedding_response.data[0].embedding).astype("float32")
        distances, indices = guideline_index.search(np.array([query_vector]), 3)
        retrieved_chunks = "\n\n".join([guideline_chunks[i] for i in indices[0]])

        full_prompt = f"""
Refer to the following guideline excerpts:
{retrieved_chunks}

Patient Summary:
{clinical_summary}

User query:
{prompt}

Return the following:
1. Therapeutic range assessment
2. Dose adjustment recommendation
3. Patient-specific issues
4. Suggested time for next sampling
Keep it concise and clinically useful.
"""

        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a senior clinical pharmacist interpreting TDM results using Malaysian antimicrobial guidelines."},
                {"role": "user", "content": full_prompt}
            ]
        )
        st.markdown(response.choices[0].message.content)

    except Exception as e:
        st.error(f"❌ LLM interpretation error: {e}")

# ===== HELPER FUNCTION =====
def suggest_adjustment(parameter, target_min, target_max, label="AUC24"):
    if parameter < target_min:
        st.warning(f"⚠️ {label} is low. Consider increasing the dose or shortening the interval.")
    elif parameter > target_max:
        st.warning(f"⚠️ {label} is high. Consider reducing the dose or lengthening the interval.")
    else:
        st.success(f"✅ {label} is within target range.")

# ===== MODULE 1: Aminoglycoside Initial Dose =====
if page == "Aminoglycoside: Initial Dose":
    st.title("🧮 Aminoglycoside Initial Dose (Population PK)")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age (years)", min_value=0, value=65)
    height = st.number_input("Height (cm)", min_value=50, value=165)
    weight = st.number_input("Actual Body Weight (kg)", min_value=1.0, value=70.0)
    scr = st.number_input("Serum Creatinine (µmol/L)", min_value=10.0, value=90.0)
    drug = st.selectbox("Drug", ["Gentamicin", "Amikacin"])
    target_cmax = st.number_input("Target Cmax (mg/L)", value=30.0 if drug == "Amikacin" else 10.0)
    target_cmin = st.number_input("Target Cmin (mg/L)", value=1.0)
    tau = st.number_input("Dosing Interval (hr)", value=24)

    ibw = 50 + 0.9 * (height - 152) if gender == "Male" else 45.5 + 0.9 * (height - 152)
    abw_ibw_ratio = weight / ibw
    dosing_weight = ibw + 0.4 * (weight - ibw) if abw_ibw_ratio > 1.2 else (weight if abw_ibw_ratio > 0.9 else (ibw if abw_ibw_ratio > 0.75 else weight * 1.13))

    vd = (0.3 if drug == "Amikacin" else 0.26) * dosing_weight
    crcl = ((140 - age) * dosing_weight * (1.23 if gender == "Male" else 1.04)) / scr
    clamg = (crcl * 60) / 1000
    ke = clamg / vd
    t_half = 0.693 / ke
    dose = target_cmax * vd * (1 - np.exp(-ke * tau))
    expected_cmax = dose / (vd * (1 - np.exp(-ke * tau)))
    expected_cmin = expected_cmax * np.exp(-ke * tau)

    st.markdown(f"**IBW:** {ibw:.2f} kg
**Dosing Weight:** {dosing_weight:.2f} kg
**CrCl:** {crcl:.2f} mL/min")
    st.success(f"Recommended Initial Dose: **{dose:.0f} mg**")
    st.info(f"Expected Cmax: **{expected_cmax:.2f} mg/L**, Expected Cmin: **{expected_cmin:.2f} mg/L**")

    suggest_adjustment(expected_cmax, target_cmax * 0.9, target_cmax * 1.1, label="Expected Cmax")
    suggest_adjustment(expected_cmin, target_cmin * 0.9, target_cmin * 1.1, label="Expected Cmin")

    if st.button("🧠 Interpret with LLM"):
        prompt = f"Patient: {age} y/o {gender.lower()}, {height} cm, {weight} kg, SCr: {scr} µmol/L.
Drug: {drug}, Cmax Target: {target_cmax}, Interval: {tau} hr.
Calculated: Weight {dosing_weight:.2f} kg, Vd {vd:.2f} L, CrCl {crcl:.2f}, Ke {ke:.3f}, t1/2 {t_half:.2f}, Dose {dose:.0f} mg.
Cmax {expected_cmax:.2f}, Cmin {expected_cmin:.2f}."
        interpret_with_llm(prompt)

# ===== MODULE 2: Aminoglycoside Conventional Dosing (C1/C2) =====
elif page == "Aminoglycoside: Conventional Dosing (C1/C2)":
    drug = st.selectbox("Select Drug", ["Gentamicin", "Amikacin"])
    regimen = st.selectbox("Therapeutic Goal", ["MDD", "SDD", "Synergy", "Hemodialysis"])

    if drug == "Gentamicin":
        target_peak = (5, 10) if regimen == "MDD" else (3, 5) if regimen == "Synergy" else (10, 30)
        target_trough = (0, 2) if regimen == "MDD" else (0, 1)
    else:
        target_peak = (20, 30) if regimen == "MDD" else (60, 60) if regimen == "SDD" else (20, 30)
        target_trough = (0, 10) if regimen == "MDD" else (0, 1)

    st.title("📊 Aminoglycoside Adjustment using C1/C2")

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
            cmax = c2 * math.exp(ke * t_post)
            cmin = cmax * np.exp(-ke * tau)
            vd = dose / (cmax * (1 - np.exp(-ke * tau)))
            new_dose = cmax * vd * (1 - np.exp(-ke * tau))

            st.markdown(f"**Ke:** {ke:.3f} hr⁻¹
**Half-life:** {t_half:.2f} hr
**Cmax:** {cmax:.2f} mg/L, **Cmin:** {cmin:.2f} mg/L
**Vd:** {vd:.2f} L")
            st.success(f"Recommended New Dose: **{new_dose:.0f} mg**")

            suggest_adjustment(cmax, *target_peak, label="Cmax")
            suggest_adjustment(cmin, *target_trough, label="Cmin")

            if st.button("🧠 Interpret with LLM"):
                prompt = f"Aminoglycoside TDM result:
Dose: {dose} mg, C1: {c1} mg/L, C2: {c2} mg/L, Interval: {tau} hr.
Ke: {ke:.3f}, t1/2: {t_half:.2f}, Vd: {vd:.2f}, Cmax: {cmax:.2f}, Cmin: {cmin:.2f}
Suggested new dose: {new_dose:.0f} mg."
                interpret_with_llm(prompt)
        else:
            st.error("❌ C1 and C2 must be greater than 0 to perform calculations.")
    except Exception as e:
        st.error(f"Calculation error: {e}")

# ===== MODULE 3: Vancomycin AUC-Based Dosing =====
elif page == "Vancomycin: AUC-Based Dosing":
    st.title("🧪 Vancomycin AUC-Based Dosing")
    method = st.radio("Select Method", ["Trough only", "Peak and Trough"], horizontal=True)
    weight = st.number_input("Weight (kg)", min_value=1.0)
    mic = st.number_input("Pathogen MIC (mg/L)", value=1.0)

    if method == "Trough only":
        current_dose = st.number_input("Current Total Daily Dose (mg)", value=2000)
        crcl = st.number_input("Creatinine Clearance (mL/min)", value=75.0)
        ke = 0.0044 + 0.00083 * crcl
        vd = 0.7 * weight
        cl = ke * vd
        auc24 = current_dose / cl
        auc_mic = auc24 / mic if mic else 0

        st.info(f"AUC24: {auc24:.1f} mg·hr/L | AUC/MIC: {auc_mic:.1f}")
        suggest_adjustment(auc24, 400, 600, label="AUC24")

        if st.button("🧠 Interpret with LLM"):
            prompt = f"Vancomycin dose = {current_dose} mg/day, CrCl = {crcl} mL/min, Ke = {ke:.4f}, CL = {cl:.2f} L/hr, AUC24 = {auc24:.1f}, MIC = {mic}."
            interpret_with_llm(prompt)

    else:
        peak = st.number_input("Peak (mg/L)", min_value=0.0)
        trough = st.number_input("Trough (mg/L)", min_value=0.0)
        tau = st.number_input("Interval (hr)", value=12.0)
        t_peak = st.number_input("Time of peak sample (hr)", value=1.0)
        t_trough = st.number_input("Time of trough sample (hr)", value=tau)

        try:
            ke = (math.log(peak) - math.log(trough)) / (t_trough - t_peak)
            t_half = 0.693 / ke
            auc24 = ((peak - trough) / ke + trough * tau) * (24 / tau)
            auc_mic = auc24 / mic if mic else 0

            st.info(f"Ke: {ke:.4f} hr⁻¹ | t½: {t_half:.2f} hr")
            st.success(f"AUC24: {auc24:.1f} mg·hr/L | AUC/MIC: {auc_mic:.1f}")
            suggest_adjustment(auc24, 400, 600, label="AUC24")

            if st.button("🧠 Interpret with LLM"):
                prompt = f"Vancomycin peak = {peak}, trough = {trough}, Interval = {tau}, Ke = {ke:.4f}, AUC24 = {auc24:.1f}, MIC = {mic}."
                interpret_with_llm(prompt)
        except Exception as e:
            st.error(f"Calculation error: {e}")
    if parameter < target_min:
        st.warning(f"⚠️ {label} is low. Consider increasing the dose or shortening the interval.")
    elif parameter > target_max:
        st.warning(f"⚠️ {label} is high. Consider reducing the dose or lengthening the interval.")
    else:
        st.success(f"✅ {label} is within target range.")
