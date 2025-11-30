import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import re
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hospital Bed Allocation System", layout="wide")
st.title("Hospital Bed Allocation System")
st.write("Upload a patient data CSV file and analyze bed allocation predictions with visualizations.")

# --- helper functions ---
def normalize_cols(df):
    """Return a DataFrame with lowercase stripped column names and a mapping back to originals."""
    mapping = {c: c.strip() for c in df.columns}
    df = df.rename(columns=mapping)
    lower_map = {c: c.lower().replace(" ", "_").replace("/", "_"): c for c in df.columns}
    df.columns = [c.lower().replace(" ", "_").replace("/", "_") for c in df.columns]
    return df, lower_map

def extract_occupancy(text_series):
    """Extract ICU, General, Isolation counts from text like '5 ICU, 20 General, 2 Isolation'."""
    ICU = []
    Gen = []
    Iso = []
    for val in text_series.fillna("").astype(str):
        m = re.search(r"(\d+)\s*ICU.*?(\d+)\s*General.*?(\d+)\s*Isolation", val, re.IGNORECASE)
        if m:
            ICU.append(int(m.group(1))); Gen.append(int(m.group(2))); Iso.append(int(m.group(3)))
        else:
            # attempt simpler extraction of any numbers (fallback)
            nums = re.findall(r"\d+", val)
            if len(nums) >= 3:
                ICU.append(int(nums[0])); Gen.append(int(nums[1])); Iso.append(int(nums[2]))
            else:
                ICU.append(0); Gen.append(0); Iso.append(0)
    return pd.Series(ICU), pd.Series(Gen), pd.Series(Iso)

def harmonize_bed_label(label):
    """Map variants to canonical labels used for target/availability columns."""
    if pd.isna(label):
        return label
    label = str(label).strip().lower()
    if "icu" in label:
        return "ICU"
    if "isolation" in label:
        return "Isolation"
    if "general" in label or "ward" in label:
        return "General Ward"
    # fallback
    return label.title()

# --- File upload ---
uploaded_file = st.file_uploader("Upload your patient data CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw columns detected")
    st.write(list(data.columns))

    # normalize column names for robustness
    data, lower_map = normalize_cols(data)

    # attempt to map expected columns (flexible)
    # possible keys in file: patient_id / patient id / patient id etc.
    col_candidates = {
        "patient_id": ["patient_id", "patient id", "patientid", "patient"],
        "admission_datetime": ["admission_datetime", "admission_date_time", "admission date/time", "admission date_time", "admission_date/time"],
        "discharge_datetime": ["discharge_datetime", "discharge_date_time", "discharge date/time", "discharge"],
        "age": ["age"],
        "gender": ["gender", "sex"],
        "diagnosis": ["diagnosis"],
        "admission_type": ["admission_type", "admission type"],
        "severity_score": ["severity_score", "severity score", "severity"],
        "bed_type_required": ["bed_type_required", "bed type required", "bed_type_needed", "bed type needed", "bed_type"],
        "special_requirements": ["special_requirements", "special requirements", "special"],
        "ward": ["ward"],
        "current_occupancy": ["current_occupancy", "current occupancy", "occupancy_text", "occupancy"],
        "wait_time": ["wait_time", "wait time", "predicted_wait_time_hours", "wait_time_hours"]
    }

    col_map = {}
    for canonical, candidates in col_candidates.items():
        found = None
        for cand in candidates:
            cand_norm = cand.lower().replace(" ", "_").replace("/", "_")
            if cand_norm in data.columns:
                found = cand_norm
                break
        if found:
            col_map[canonical] = found

    # check required columns and error politely
    required = ["patient_id","admission_datetime","discharge_datetime","age","gender","diagnosis",
                "admission_type","severity_score","bed_type_required","special_requirements","ward","current_occupancy","wait_time"]
    missing = [r for r in required if r not in col_map]
    # allow using fallback for wait_time (optional) or warn
    if missing:
        st.warning(f"Couldn't find these columns automatically: {missing}. Attempting best-effort for available fields.")
    st.write("Column mapping (canonical -> file):", col_map)

    # create working df with canonical names if possible
    df = pd.DataFrame()
    for k, v in col_map.items():
        df[k] = data[v] if v in data.columns else np.nan

    # If some canonical columns are missing but the file has similar ones, try to copy
    # (for example if user earlier generator put patient_id as 'patient_id')
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # --- Occupancy parsing ---
    # If occupancy is numeric (single number), infer split by bed distribution in target or assume defaults
    if pd.api.types.is_numeric_dtype(df["current_occupancy"]):
        st.info("`current_occupancy` appears numeric; inferring split by bed-type distribution.")
        # compute distribution from bed_type_required if present
        if col_map.get("bed_type_required") is not None:
            bed_counts = df["bed_type_required"].value_counts(normalize=True, dropna=True)
            ICU_frac = bed_counts.get("ICU", bed_counts.get("icu", 0)) if not bed_counts.empty else 0
            Gen_frac = bed_counts.get("General Ward", bed_counts.get("general", 0)) if not bed_counts.empty else 0
            Iso_frac = bed_counts.get("Isolation", bed_counts.get("isolation", 0)) if not bed_counts.empty else 0
            df["ICU_Available"] = (df["current_occupancy"] * ICU_frac).fillna(0).astype(int)
            df["General_Available"] = (df["current_occupancy"] * Gen_frac).fillna(0).astype(int)
            df["Isolation_Available"] = (df["current_occupancy"] * Iso_frac).fillna(0).astype(int)
        else:
            df["ICU_Available"] = 0
            df["General_Available"] = df["current_occupancy"].fillna(0).astype(int)
            df["Isolation_Available"] = 0
    else:
        icu_s, gen_s, iso_s = extract_occupancy(df["current_occupancy"])
        df["ICU_Available"] = icu_s
        df["General_Available"] = gen_s
        df["Isolation_Available"] = iso_s

    # --- Datetime parsing & stay duration (hours) ---
    # Try robust parsing
    df["admission_dt_parsed"] = pd.to_datetime(df["admission_datetime"], errors="coerce")
    df["discharge_dt_parsed"] = pd.to_datetime(df["discharge_datetime"], errors="coerce")
    # If parsing failed, try common alternate columns
    if df["admission_dt_parsed"].isna().all():
        st.warning("Admission datetime parsing failed for all rows. Check format.")
    # compute stay duration in hours (discharge - admission)
    df["stay_duration_hours"] = ((df["discharge_dt_parsed"] - df["admission_dt_parsed"]).dt.total_seconds() / 3600).fillna(0).clip(lower=0)

    # --- numeric conversions & urgency score ---
    df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce").fillna(1).astype(int)
    # admission weight mapping
    weights = {"emergency": 3, "urgent": 2, "elective": 1}
    df["admission_type_norm"] = df["admission_type"].astype(str).str.lower()
    df["admission_weight"] = df["admission_type_norm"].map(weights).fillna(1)
    df["urgency_score"] = df["severity_score"] * df["admission_weight"]
    df["urgency_per_duration"] = df["urgency_score"] / (df["stay_duration_hours"] + 1)

    # canonicalize bed target
    df["bed_type_required"] = df["bed_type_required"].apply(harmonize_bed_label)

    # --- categorical encoding ---
    categorical_cols = ["gender", "diagnosis", "admission_type", "special_requirements", "ward"]
    # ensure these columns exist in df
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "Unknown"
    data_encoded = pd.get_dummies(df[categorical_cols].fillna("Unknown"), drop_first=True)

    # numerical features
    numerical_cols = ["age", "severity_score", "urgency_score", "stay_duration_hours", "urgency_per_duration",
                      "ICU_Available", "General_Available", "Isolation_Available"]
    for c in numerical_cols:
        if c not in df.columns:
            df[c] = 0

    X = pd.concat([df[numerical_cols].reset_index(drop=True), data_encoded.reset_index(drop=True)], axis=1).fillna(0)
    # target variables
    # harmonize wait_time column numeric
    df["wait_time"] = pd.to_numeric(df["wait_time"], errors="coerce").fillna(0)

    y_class = df["bed_type_required"].fillna("General Ward")
    y_reg = df["wait_time"]

    st.subheader("Preview processed data")
    st.write(df.head(5))
    st.write("Feature matrix shape:", X.shape)

    # --- Model training ---
    st.subheader("Model Training and Evaluation")

    # scale whole X for consistent inference
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X)

    # Classification train/test
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled_full, y_class, test_size=0.2, random_state=42, stratify=y_class)
    # compute class weights
    classes = np.unique(y_train_class)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_class)
    class_weight_dict = dict(zip(classes, class_weights))

    rf_classifier = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
    param_grid = {'n_estimators': [100], 'max_depth': [10, None], 'min_samples_split': [2, 5]}
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_class, y_train_class)
    best_classifier = grid_search.best_estimator_
    st.write("Best classifier params:", grid_search.best_params_)

    y_pred_class = best_classifier.predict(X_test_class)
    acc = accuracy_score(y_test_class, y_pred_class)
    st.write(f"Bed Type Prediction Accuracy: {acc:.3f}")
    st.text(classification_report(y_test_class, y_pred_class, zero_division=0))

    # Feature importance plot (classification)
    feat_imp = pd.DataFrame({"feature": X.columns, "importance": best_classifier.feature_importances_}).sort_values("importance", ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.bar(feat_imp["feature"].iloc[:8].values, feat_imp["importance"].iloc[:8].values)
    ax1.set_xticklabels(feat_imp["feature"].iloc[:8].values, rotation=45, ha='right')
    ax1.set_title("Top features (Classification)")
    st.pyplot(fig1)

    # Regression train/test
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled_full, y_reg, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor(random_state=42)
    param_grid_reg = {'n_estimators': [100], 'max_depth': [10, None], 'min_samples_split': [2, 5]}
    grid_search_reg = GridSearchCV(rf_regressor, param_grid_reg, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search_reg.fit(X_train_reg, y_train_reg)
    best_regressor = grid_search_reg.best_estimator_
    st.write("Best regressor params:", grid_search_reg.best_params_)

    y_pred_reg = best_regressor.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    st.write(f"Wait Time Prediction MAE: {mae:.3f}")

    feat_imp_r = pd.DataFrame({"feature": X.columns, "importance": best_regressor.feature_importances_}).sort_values("importance", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.bar(feat_imp_r["feature"].iloc[:8].values, feat_imp_r["importance"].iloc[:8].values)
    ax2.set_xticklabels(feat_imp_r["feature"].iloc[:8].values, rotation=45, ha='right')
    ax2.set_title("Top features (Regression)")
    st.pyplot(fig2)

    # Bed type distribution pie
    st.subheader("Bed Type Distribution")
    bed_counts = y_class.value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(bed_counts.values, labels=bed_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.axis('equal')
    st.pyplot(fig3)

    # --- allocation helper ---
    def allocate_bed(patient_row):
        # build patient feature vector aligned with X.columns
        pf = pd.Series(0, index=X.columns)
        # fill numerical
        for nc in numerical_cols:
            if nc in patient_row:
                pf[nc] = patient_row.get(nc, 0)
        # encode categorical deterministically using same columns as data_encoded
        patient_cat = {}
        for c in categorical_cols:
            val = patient_row.get(c, "Unknown")
            key = f"{c}_{val}"
            # if encoded column exists, set it; otherwise attempt fuzzy match
            if key in pf.index:
                pf[key] = 1
            else:
                # try to find any column that starts with c_ and contains val substring
                matches = [col for col in pf.index if col.startswith(c + "_") and str(val).lower() in col.lower()]
                if matches:
                    pf[matches[0]] = 1
        # scale
        pf_scaled = scaler.transform(pf.values.reshape(1, -1))
        pred_bed = best_classifier.predict(pf_scaled)[0]
        pred_wait = best_regressor.predict(pf_scaled)[0]
        # determine available counts from patient_row (fallback to ward-level first matching row)
        available = {
            "ICU": int(patient_row.get("ICU_Available", 0)),
            "General Ward": int(patient_row.get("General_Available", 0)),
            "Isolation": int(patient_row.get("Isolation_Available", 0))
        }
        urgency = patient_row.get("urgency_score", 0)
        if available.get(pred_bed, 0) > 0:
            return f"Assign {pred_bed} in {patient_row.get('ward', 'Unknown')} (Urgency: {urgency:.2f}, Predicted Wait: {pred_wait:.2f} hours)"
        else:
            alt = [b for b, cnt in available.items() if cnt > 0]
            if alt:
                return f"No {pred_bed} available. Consider {alt[0]} (Predicted Wait: {pred_wait:.2f} hours)"
            else:
                return f"No beds available in ward. Transfer suggested. (Predicted Wait: {pred_wait:.2f} hours)"

    # sample predictions button
    st.subheader("Sample Predictions")
    if st.button("Generate Sample Predictions"):
        sample = df.sample(5, random_state=42)
        for _, row in sample.iterrows():
            st.write(f"Patient {row.get('patient_id', 'N/A')}: {allocate_bed(row)}")

    # realtime simulation selection
    st.subheader("Real-Time Allocation Simulation")
    if df["patient_id"].notna().any():
        pid = st.selectbox("Select Patient ID for Simulation", df["patient_id"].astype(str).tolist())
        patient_row = df[df["patient_id"].astype(str) == str(pid)].iloc[0].to_dict()
        if st.button("Run Simulation"):
            st.write(allocate_bed(patient_row))

    st.subheader("Processed Data (first 50 rows)")
    st.dataframe(df.head(50))

else:
    st.info("Upload the CSV created previously (e.g., /mnt/data/patientdata.csv).")
