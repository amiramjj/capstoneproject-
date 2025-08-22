# app_cleaning.py
# Streamlit: Cleaning & Preprocessing (Cleaning v1)
# --- ensure google-generativeai is available (temporary self-heal) ---

import io
import re
from statistics import mode, StatisticsError

import numpy as np
import pandas as pd
import streamlit as st

# ---- Keep data across button clicks ----
for k in ("cleaned_df", "deduped_df", "engineered_df"):
    st.session_state.setdefault(k, None)


st.set_page_config(page_title="Cleaning & Preprocessing — Capstone", layout="wide")

# --------------------------- Constants / Contracts --------------------------- #
REQUIRED_COLS = [
    "client_name", "contract_id", "cc_type", "maid_id", "maid_nationality",
    "complaint_number", "complaint_comments", "complaint_summary",
    "client_mts_at_hiring", "maid_mts_at_hiring", "maids_custom_preferences_at_hiring",
    "cooking_details", "years_of_experience", "maid_speaks_language",
    "tag_date", "untag_date"
]

# Languages list (extend later if needed)
LANGS = ["arabic", "amharic", "oromo", "english", "french"]


# ------------------------------- Helper funcs ------------------------------- #
def read_table(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    elif name.endswith(".csv"):
        # Try utf-8, fallback to cp1252
        data = uploaded.read()
        try:
            return pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="cp1252")
    else:
        raise ValueError("Unsupported file type. Upload .csv or .xlsx")


def missing_overview(df):
    miss = df.isnull().sum().sort_values(ascending=False)
    return pd.DataFrame({"Missing Count": miss})


def normalize_language_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .replace({"nan": pd.NA})
    )


def most_common_pattern_by_nationality(df: pd.DataFrame) -> dict:
    # build from the raw/normalized maid_speaks_language BEFORE final cleaning
    tmp = df.dropna(subset=["maid_speaks_language"])
    if tmp.empty:
        return {}
    return (
        tmp.groupby("maid_nationality")["maid_speaks_language"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )


def impute_language(row, lang_by_nat: dict):
    val = row["maid_speaks_language"]
    if pd.isna(val) or val == "not_specified":
        nat = row.get("maid_nationality")
        most_common = lang_by_nat.get(nat)
        if most_common:
            return f"english {most_common}".strip()
        return "english"
    return val


def safe_clean_languages(x):
    """Reduce to set of known languages based on rules:
       - Keep languages that appear as 'lang : <level>' where <level> is NOT 'no'
       - Or standalone language words
       - Return alphabetized space-separated unique langs or NA
    """
    if pd.isna(x):
        return pd.NA
    s = str(x).lower()

    keep = set()
    for lang in LANGS:
        # Case 1: "lang : <level>" and level not 'no'
        if re.search(fr"\b{lang}\s*:\s*(?!no)\w+", s):
            keep.add(lang)
        # Case 2: standalone language word (not followed by ':')
        elif re.search(fr"\b{lang}\b(?!\s*:)", s):
            keep.add(lang)

    return " ".join(sorted(keep)) if keep else pd.NA


def impute_years_of_experience(df):
    med_by_nat = (
        df.groupby("maid_nationality")["years_of_experience"]
        .median()
        .dropna()
        .to_dict()
    )
    global_median = df["years_of_experience"].median()

    def _imp(row):
        v = row["years_of_experience"]
        if pd.isna(v):
            return med_by_nat.get(row["maid_nationality"], global_median)
        return v

    df["years_of_experience"] = df.apply(_imp, axis=1)
    return df


def extract_all_cuisines(text):
    """Extract all cuisine values, dedupe, and join with ' || '.
       If none found, return 'not_specified'.
       Example input: 'experience: yes | duration: 2 years 6 months | cuisine: kuwaiti'
    """
    if pd.isna(text):
        return "not_specified"
    s = str(text).lower()
    # capture everything after 'cuisine:' up to a separator/newline
    matches = re.findall(r"cuisine\s*:\s*([^\n\r;|]*)", s)
    cleaned = []
    for m in matches:
        # remove trailing 'experience ...' if present
        m = re.sub(r"\bexperience\b.*", "", m)
        # keep letters and spaces only
        m = re.sub(r"[^a-z\s]", "", m).strip()
        if m:
            cleaned.append(m)
    if not cleaned:
        return "not_specified"
    return " || ".join(sorted(set(cleaned)))


def complaint_placeholder_logic(df):
    """Apply placeholder rules without touching complaint_number dtype."""
    # normalize for safe string checks
    df["complaint_summary"] = df["complaint_summary"].astype(str).str.strip().str.lower()
    df["complaint_comments"] = df["complaint_comments"].astype(str).str.strip().str.lower()

    def updated_comment_summary_check(row):
        summary = row["complaint_summary"]
        comment = row["complaint_comments"]

        no_comment = comment in ["", "nan", "no complaint"]

        is_placeholder = (
            summary == "full summary: recent summary:"
            or summary.startswith("full summary: recent summary: i'm sorry")
            or summary.startswith("full summary: recent summary: please provide")
            or summary.startswith("full summary: recent summary: could you please")
            or summary.startswith("full summary: recent summary: kindly provide")
        )
        return no_comment and is_placeholder

    mask = df.apply(updated_comment_summary_check, axis=1)
    updated_count = int(mask.sum())
    df.loc[mask, ["complaint_comments", "complaint_summary"]] = "no complaint"

    # Original stricter rule (triplet condition)
    def is_no_complaint_triplet(row):
        summary = str(row["complaint_summary"]).strip().lower()
        no_number = pd.isna(row["complaint_number"])
        no_comment2 = row["complaint_comments"] in ["", "nan", "no complaint"]
        placeholder_summary = summary.startswith("full summary: recent summary:")
        return no_number and no_comment2 and placeholder_summary

    mask2 = df.apply(is_no_complaint_triplet, axis=1)
    updated_count += int(mask2.sum())
    df.loc[mask2, ["complaint_comments", "complaint_summary"]] = "no complaint"

    return df, updated_count


def clean_join(series: pd.Series) -> str | None:
    """Sorted-set dedupe join for text-like columns."""
    values = series.dropna().astype(str).str.strip().str.lower()
    uniq = sorted({v for v in values if v and v != "nan"})
    return " || ".join(uniq) if uniq else None


def safe_mode(series: pd.Series):
    non_na = series.dropna().tolist()
    if not non_na:
        return None
    try:
        return mode(non_na)
    except StatisticsError:
        return non_na[0]


def deduplicate(df: pd.DataFrame):
    # Step 1: make untag_date placeholder-safe
    tmp = df.copy()
    tmp["untag_date"] = tmp["untag_date"].astype(str).replace({"NaT": "not_yet", "nan": "not_yet", "": "not_yet"})

    identity_cols = ["client_name", "contract_id", "cc_type", "maid_id", "tag_date", "untag_date"]

    text_columns = [
        "complaint_number",  # kept as text here to preserve multiple IDs " || " if duplicates exist
        "complaint_comments", "complaint_summary",
        "client_mts_at_hiring", "maid_mts_at_hiring",
        "maids_custom_preferences_at_hiring", "cooking_details",
        "maid_speaks_language",
    ]
    numeric_columns = ["years_of_experience"]
    context_columns = ["maid_nationality"]

    agg_dict = {col: clean_join for col in text_columns}
    agg_dict.update({col: safe_mode for col in numeric_columns})
    agg_dict.update({col: "first" for col in context_columns})

    deduped = tmp.groupby(identity_cols, as_index=False).agg(agg_dict)

    # Step 3: Convert 'not_yet' back to NaT and parse to datetime
    deduped["untag_date"] = deduped["untag_date"].replace("not_yet", pd.NaT)
    deduped["untag_date"] = pd.to_datetime(deduped["untag_date"], errors="coerce")
    return deduped


# --------------------------------- UI / Logic -------------------------------- #
st.title("Cleaning & Preprocessing (ERP → Model-Ready)")

st.markdown(
    "Upload the **raw ERP export** (.csv or .xlsx). "
    "This page performs *exactly* the Cleaning v1 steps we agreed on."
)

uploaded = st.file_uploader("Upload Assignments file", type=["csv", "xlsx"])

if uploaded:
    with st.spinner("Reading file..."):
        df = read_table(uploaded)
        original_shape = df.shape

    # Schema check
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    st.success(f"Loaded: {uploaded.name}  •  Shape: {original_shape[0]} rows × {original_shape[1]} cols")
    st.expander("Preview (first 10 rows)", expanded=False).dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Step 0 — Missingness Overview")
    st.dataframe(missing_overview(df), use_container_width=True)

    if st.button("⚙️ Run Cleaning & Preprocessing"):
        with st.spinner("Applying Cleaning v1..."):
            # --- Languages: normalize -> impute -> clean
            df["maid_speaks_language"] = normalize_language_series(df["maid_speaks_language"])
            lang_by_nat = most_common_pattern_by_nationality(df)

            # Count NAs before
            lang_na_before = int(df["maid_speaks_language"].isna().sum())

            df["maid_speaks_language"] = df.apply(impute_language, axis=1, lang_by_nat=lang_by_nat)

            # Final clean to a language set
            df["maid_speaks_language"] = df["maid_speaks_language"].apply(safe_clean_languages)
            lang_na_after = int(df["maid_speaks_language"].isna().sum())
            lang_imputed = lang_na_before - lang_na_after if lang_na_before >= lang_na_after else 0

            # --- Experience imputation
            pre_exp_na = int(df["years_of_experience"].isna().sum())
            df = impute_years_of_experience(df)
            post_exp_na = int(df["years_of_experience"].isna().sum())

            # --- Cooking details → all cuisines
            df["cooking_details"] = df["cooking_details"].apply(extract_all_cuisines)

            # --- Complaint placeholders
            df, complaints_updated = complaint_placeholder_logic(df)

            # --- Dedup
            before_rows = df.shape[0]
            deduped_df = deduplicate(df)
            after_rows = deduped_df.shape[0]

            # SAVE TO SESSION *HERE* (deduped_df is still in scope)
            st.session_state["cleaned_df"] = df.copy()
            st.session_state["deduped_df"] = deduped_df.copy()
            st.success("Cleaning completed and saved to session.")

        st.success("Cleaning completed.")

        # -------------------- Reporting panels -------------------- #
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows (before)", before_rows)
            st.metric("Rows (after dedup)", after_rows)
        with c2:
            st.metric("Languages imputed", lang_imputed)
            st.metric("Remaining language NAs", lang_na_after)
        with c3:
            st.metric("Experience NAs (before)", pre_exp_na)
            st.metric("Experience NAs (after)", post_exp_na)
        st.metric(label="Complaint placeholders cleaned", value=complaints_updated)

        st.markdown("### Language Patterns")
        colA, colB = st.columns(2)
        with colA:
            st.caption("Most common cleaned combos")
            st.dataframe(
                df["maid_speaks_language"].value_counts(dropna=False).head(15).rename_axis("pattern").reset_index(name="count"),
                use_container_width=True,
            )
        with colB:
            st.caption("Most common language per nationality")
            most_common_language_df = (
                df.dropna(subset=["maid_speaks_language"])
                .groupby("maid_nationality")["maid_speaks_language"]
                .agg(lambda s: s.value_counts().idxmax())
                .reset_index()
                .rename(columns={"maid_speaks_language": "most_common_language"})
            )
            st.dataframe(most_common_language_df.head(20), use_container_width=True)

        st.markdown("### Cooking Details (multi-cuisine combos)")
        st.dataframe(df["cooking_details"].value_counts().head(15).rename_axis("cuisines").reset_index(name="count"),
                    use_container_width=True)

        st.markdown("### Deduplicated Preview")
        st.dataframe(deduped_df.head(10), use_container_width=True)

        # -------------------- Downloads -------------------- #
        st.markdown("---")
        st.subheader("Download Outputs")

        @st.cache_data
        def _to_csv_bytes(_df):
            return _df.to_csv(index=False).encode("utf-8")

        cleaned_bytes = _to_csv_bytes(df)
        dedup_bytes = _to_csv_bytes(deduped_df)

        st.download_button(
            "⬇️ Download CLEANED (pre-dedup) CSV",
            data=cleaned_bytes,
            file_name="cleaned_pre_dedup.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download DEDUPED CSV",
            data=dedup_bytes,
            file_name="cleaned_deduped.csv",
            mime="text/csv",
        )

else:
    st.info("Upload a raw ERP export (.csv or .xlsx) to begin.")



# ==============================
# Streamlit: Engineering (MTS → Features)
# Mirrors your exact notebook logic
# ==============================

def run_engineering(deduped_df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("---")
    st.header("Step 2 — Feature Engineering from ERP Lists (Exact Parity)")
    df = deduped_df.copy()

    # ---------- Helpers ----------
    def _standardize_series_to_trait_lists(series: pd.Series, do_lebanon_fix=False):
        s = (
            series.fillna("")
            .astype(str).str.lower()
            .str.replace("||", ",", regex=False)
        )
        if do_lebanon_fix:
            s = s.str.replace("lebanon", "lebanese", regex=False)
        return s.str.split(",").apply(lambda x: [t.strip() for t in x if t.strip() != ""])

    def get_all_trait_counts_standardized(df_, col_name):
        vc = (
            df_[col_name].dropna().astype(str)
            .str.replace("||", ",", regex=False).str.lower()
            .str.split(",").explode().str.strip()
        ).value_counts()
        return pd.DataFrame({"Trait": vc.index, "Count": vc.values})

    # ---------- Client MTS ----------
    df["client_mts_at_hiring"] = (
        df["client_mts_at_hiring"].astype(str)
        .str.replace("||", ",", regex=False).str.lower()
        .str.replace("lebanon", "lebanese", regex=False)
    )

    # Displays (optional)
    with st.expander("All Traits (Raw → Canonical Tokens)"):
        st.subheader("Client MTS (at hiring)")
        st.dataframe(get_all_trait_counts_standardized(df, "client_mts_at_hiring"), use_container_width=True)
        st.subheader("Maid MTS (at hiring)")
        st.dataframe(get_all_trait_counts_standardized(df, "maid_mts_at_hiring"), use_container_width=True)
        st.subheader("Maid Custom Preferences (at hiring)")
        st.dataframe(get_all_trait_counts_standardized(df, "maids_custom_preferences_at_hiring"), use_container_width=True)

    # Safe fill + tokenize
    df["client_mts_at_hiring"] = df["client_mts_at_hiring"].fillna("no_preference")
    df["maid_mts_at_hiring"]   = df["maid_mts_at_hiring"].fillna("no_preference")
    df["maids_custom_preferences_at_hiring"] = df["maids_custom_preferences_at_hiring"].fillna("no_preference")

    client_mts_raw = _standardize_series_to_trait_lists(df["client_mts_at_hiring"], do_lebanon_fix=True)
    maid_mts_raw   = _standardize_series_to_trait_lists(df["maid_mts_at_hiring"])
    maid_pref_raw  = _standardize_series_to_trait_lists(df["maids_custom_preferences_at_hiring"])

    # ---------- Client features ----------
    def client_get_household_type(t):
        baby = "has a baby younger than 2 years old" in t
        kids = "has 3 kids or more" in t
        no_kids = "i prefer working with a family with no kids" in t
        if baby and kids: return "baby_and_kids"
        if baby: return "baby"
        if kids: return "many_kids"
        if no_kids: return "no_kids"
        return "none"

    def client_get_special_case(t):
        elderly = "elderly parent at home" in t
        special = "special needs kid" in t
        if elderly and special: return "elderly_and_special"
        if elderly: return "elderly"
        if special: return "special_needs"
        return "none"

    def client_get_pet_type(t):
        cat = "has a cat" in t
        dog = "has a dog" in t
        if cat and dog: return "both"
        if cat: return "cat"
        if dog: return "dog"
        return "none"

    def client_get_dayoff_policy(t):
        p = []
        if "give day off other than sunday negation" in t: p.append("flexible")
        if "work on her day off for pay" in t:            p.append("work_for_pay")
        if "stay home on day off" in t or "take her day off at home for pay" in t:
            p.append("stay_home_for_pay")
        return "+".join(p) if p else "none"

    def client_get_nationality_preference(t):
        pref = [n for n in ["filipina","west african nationality","ethiopian maid","indian"] if n in t]
        return "+".join(pref) if pref else "any"

    def client_get_living_arrangement(t):
        arr = []
        if "live out" in t:           arr.append("live_out")
        if "has a private room" in t: arr.append("private_room")
        if "lives in abu dhabi" in t: arr.append("abu_dhabi")
        return "+".join(arr) if arr else "unspecified"

    def client_get_cuisine_preference(t):
        sel = [c for c in ["lebanese","khaleeji","international"] if c in t]
        return "+".join(sel) if sel else "other"

    df["clientmts_household_type"]         = client_mts_raw.apply(client_get_household_type)
    df["clientmts_special_cases"]          = client_mts_raw.apply(client_get_special_case)
    df["clientmts_pet_type"]               = client_mts_raw.apply(client_get_pet_type)
    df["clientmts_dayoff_policy"]          = client_mts_raw.apply(client_get_dayoff_policy)
    df["clientmts_nationality_preference"] = client_mts_raw.apply(client_get_nationality_preference)
    df["clientmts_living_arrangement"]     = client_mts_raw.apply(client_get_living_arrangement)
    df["clientmts_cuisine_preference"]     = client_mts_raw.apply(client_get_cuisine_preference)

    # ---------- Maid MTS ----------
    def maid_get_household_type(t):
        baby = "has a baby younger than 2 years old" in t
        kids = "has 3 kids or more" in t
        if baby and kids: return "baby_and_kids"
        if baby: return "baby"
        if kids: return "many_kids"
        return "none"

    def maid_get_pet_type(t):
        cat = "has a cat" in t
        dog = "has a dog" in t
        if cat and dog: return "both"
        if cat: return "cat"
        if dog: return "dog"
        return "none"

    def maid_get_dayoff_policy(t):
        return "flexible" if "give day off other than sunday negation" in t else "unspecified"

    def maid_get_living_arrangement(t):
        if "has a private room" in t and "lives in abu dhabi" in t: return "private_room+avoids_abu_dhabi"
        if "has a private room" in t: return "private_room"
        if "lives in abu dhabi" in t: return "avoids_abu_dhabi"
        return "unspecified"

    df["maidmts_household_type"]     = maid_mts_raw.apply(maid_get_household_type)
    df["maidmts_pet_type"]           = maid_mts_raw.apply(maid_get_pet_type)
    df["maidmts_dayoff_policy"]      = maid_mts_raw.apply(maid_get_dayoff_policy)
    df["maidmts_living_arrangement"] = maid_mts_raw.apply(maid_get_living_arrangement)

    # ---------- Maid Preferences ----------
    def maidpref_get_education(t):
        uni = "has university degree" in t
        sch = "has a school degree" in t
        if uni and sch: return "both"
        if uni: return "university"
        if sch: return "school"
        return "not_specified"

    def maidpref_get_kids_experience(t):
        less2  = ("maid has experience with kids under 6 months old" in t
                  or "maid has experience with kids between 6 months and 2 years" in t)
        above2 = "maid has experience with kids above 2 years old" in t
        if less2 and above2: return "both"
        if less2: return "lessthan2"
        if above2: return "above2"
        return "none"

    def maidpref_get_pet_handling(t):
        cats = "handles multiple cats" in t
        dogs = "handles multiple dogs" in t
        if cats and dogs: return "both"
        if cats: return "cats"
        if dogs: return "dogs"
        return "none"

    def maidpref_get_personality(t):
        checks = {
            "maid is energetic": "energetic",
            "maid does not have attitude": "no_attitude",
            "maid does not have tiktok": "no_tiktok",
            "flexible to work in a vegetarian household": "veg_friendly",
        }
        matched = [label for text, label in checks.items() if text in t]
        return "+".join(matched) if matched else "not_mentioned"

    def maidpref_get_travel(t):
        travel   = "maid does not mind travelling" in t
        relocate = "don’t mind relocating" in t or "don't mind relocating" in t
        if travel and relocate: return "travel_and_relocate"
        if travel: return "travel"
        if relocate: return "relocate"
        return "no"

    def maidpref_get_smoking(t):
        return "non_smoker" if "maid is not a smoker" in t else "unspecified"

    def maidpref_get_caregiving(t):
        elderly_exp = "experienced with elderly person" in t
        elderly_will = "willing to handle elderly persons" in t
        special = "willing to handle special needs kid" in t
        if (elderly_exp or elderly_will) and special: return "elderly_and_special"
        if (elderly_exp or elderly_will):             return "elderly_experienced"
        if special:                                   return "special_needs"
        return "none"

    df["maidpref_education"]          = maid_pref_raw.apply(maidpref_get_education)
    df["maidpref_kids_experience"]    = maid_pref_raw.apply(maidpref_get_kids_experience)
    df["maidpref_pet_handling"]       = maid_pref_raw.apply(maidpref_get_pet_handling)
    df["maidpref_personality"]        = maid_pref_raw.apply(maidpref_get_personality)
    df["maidpref_travel"]             = maid_pref_raw.apply(maidpref_get_travel)
    df["maidpref_smoking"]            = maid_pref_raw.apply(maidpref_get_smoking)
    df["maidpref_caregiving_profile"] = maid_pref_raw.apply(maidpref_get_caregiving)

    # ---------- Preview + Download ----------
    st.subheader("Engineered Columns (preview)")
    engineered_cols = [
        "clientmts_household_type","clientmts_special_cases","clientmts_pet_type",
        "clientmts_dayoff_policy","clientmts_nationality_preference",
        "clientmts_living_arrangement","clientmts_cuisine_preference",
        "maidmts_household_type","maidmts_pet_type","maidmts_dayoff_policy","maidmts_living_arrangement",
        "maidpref_education","maidpref_kids_experience","maidpref_pet_handling",
        "maidpref_personality","maidpref_travel","maidpref_smoking","maidpref_caregiving_profile",
    ]
    st.dataframe(df[engineered_cols].head(10), use_container_width=True)

    @st.cache_data
    def _to_csv_bytes(_df):
        return _df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download with Engineered Features",
        data=_to_csv_bytes(df),
        file_name="engineered_features.csv",
        mime="text/csv",
    )
    return df  
    st.success("Cleaning completed and saved to session.")

# --------- Trigger UI (uses session; button enabled only when data exists) ---------
deduped_df_ss = st.session_state.get("deduped_df")
if deduped_df_ss is not None:
    st.markdown("### Deduped preview (from session)")
    st.dataframe(deduped_df_ss.head(10), use_container_width=True)
else:
    st.info("Run Cleaning & Preprocessing first to enable feature engineering.")

if st.button("🧩 Run Feature Engineering (ERP Lists)", disabled=deduped_df_ss is None):
    st.session_state["engineered_df"] = run_engineering(deduped_df_ss)

if st.session_state.get("engineered_df") is not None:
    st.markdown("### Engineered preview (from session)")
    st.dataframe(st.session_state["engineered_df"].head(10), use_container_width=True)





# ==============================
# Step 2B — Flags, Codes, Language One-Hots (add-on)
# ==============================
def run_engineering_step2b(df_in: pd.DataFrame) -> pd.DataFrame:
    st.markdown("---")
    st.header("Step 2B — Maid Capability Flags, Ordinal Codes, Language One-Hots")

    df = df_in.copy()

    # ---- ensure required columns exist (fill with NA if missing) ----
    needed = [
        "maidmts_household_type","maidpref_kids_experience",
        "maidmts_pet_type","maidpref_pet_handling",
        "maidmts_living_arrangement","maidpref_travel",
        "maidmts_dayoff_policy","maidpref_smoking",
        "maidpref_education","maidpref_caregiving_profile","maidpref_personality",
        "maid_speaks_language"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    # --- helpers (names prefixed to avoid collisions) ---
    def _fx_split_tokens(s):
        if pd.isna(s): return []
        return [t.strip() for t in str(s).lower().split("+") if t.strip()]

    def _fx_has_token(series, token):
        return series.apply(lambda v: token in _fx_split_tokens(v)).astype(int)

    def _fx_in_set(series, values):
        vals = set(v.lower() for v in values)
        return series.apply(lambda v: (str(v).lower() in vals) if pd.notna(v) else False).astype(int)

    def _fx_mobility_code(s):
        s = (str(s).lower() if pd.notna(s) else "")
        if s == "travel_and_relocate": return 2
        if s in ("travel","relocate"): return 1
        return 0

    def _fx_dayoff_code(s):
        s = (str(s).lower() if pd.notna(s) else "")
        if "sunday_only" in s: return -2
        if "flexible"    in s: return  1
        return 0

    def _fx_edu_level(s):
        s = (str(s).lower() if pd.notna(s) else "")
        if s in ("university","both"): return 2
        if s == "school":              return 1
        return 0

    def _fx_care_flags(s):
        s = (str(s).lower() if pd.notna(s) else "")
        elderly_ok = int(s in ("elderly_experienced","elderly_and_special"))
        special_ok = int(s in ("special_needs","elderly_and_special"))
        return elderly_ok, special_ok

    def _fx_personality_flags(s):
        toks = set(_fx_split_tokens(s))
        energetic    = int("energetic"    in toks)
        no_attitude  = int("no_attitude"  in toks)
        no_tiktok    = int("no_tiktok"    in toks)
        veg_friendly = int("veg_friendly" in toks)
        score = energetic + no_attitude + no_tiktok + veg_friendly
        return energetic, no_attitude, no_tiktok, veg_friendly, score

    # --- 1) FLAGS ---
    # Kids / household
    df["infant_block"]    = _fx_has_token(df["maidmts_household_type"], "baby")
    df["manykids_block"]  = _fx_has_token(df["maidmts_household_type"], "many_kids")
    df["infant_skill"]    = _fx_in_set(df["maidpref_kids_experience"], ["lessthan2","both"])
    df["manykids_skill"]  = _fx_in_set(df["maidpref_kids_experience"], ["above2","both"])

    # Pets
    df["cats_block"]      = _fx_in_set(df["maidmts_pet_type"], ["cat","both"])
    df["dogs_block"]      = _fx_in_set(df["maidmts_pet_type"], ["dog","both"])
    df["cats_skill"]      = _fx_in_set(df["maidpref_pet_handling"], ["cats","both"])
    df["dogs_skill"]      = _fx_in_set(df["maidpref_pet_handling"], ["dogs","both"])

    # Living / mobility
    df["requires_private_room"] = _fx_has_token(df["maidmts_living_arrangement"], "private_room")
    df["avoids_abudhabi"]       = _fx_has_token(df["maidmts_living_arrangement"], "avoids_abu_dhabi")
    df["mobility_flex"]         = df["maidpref_travel"].apply(_fx_mobility_code).astype(int)

    # Day off
    df["dayoff_sunday_only"] = df["maidmts_dayoff_policy"].apply(
        lambda s: 1 if pd.notna(s) and "sunday_only" in str(s).lower() else 0
    ).astype(int)
    df["dayoff_flexible"] = df["maidmts_dayoff_policy"].apply(
        lambda s: 1 if pd.notna(s) and "flexible" in str(s).lower() else 0
    ).astype(int)

    # Smoking / education
    df["maid_non_smoker_flag"]  = df["maidpref_smoking"].apply(lambda s: 1 if str(s).lower()=="non_smoker" else 0).astype(int)
    df["edu_university"]        = df["maidpref_education"].apply(lambda s: 1 if str(s).lower() in ("university","both") else 0).astype(int)
    df["edu_school"]            = df["maidpref_education"].apply(lambda s: 1 if str(s).lower()=="school" else 0).astype(int)

    # Caregiving specialization
    care_pairs = df["maidpref_caregiving_profile"].apply(_fx_care_flags)
    df["elderly_ok"] = care_pairs.apply(lambda x: x[0]).astype(int)
    df["special_ok"] = care_pairs.apply(lambda x: x[1]).astype(int)

    # Personality
    pers = df["maidpref_personality"].apply(_fx_personality_flags)
    df["energetic"]            = pers.apply(lambda x: x[0]).astype(int)
    df["no_attitude"]          = pers.apply(lambda x: x[1]).astype(int)
    df["no_tiktok"]            = pers.apply(lambda x: x[2]).astype(int)
    df["veg_friendly"]         = pers.apply(lambda x: x[3]).astype(int)
    df["maid_personality_score"] = pers.apply(lambda x: x[4]).astype(int)

    # --- 2) COMPACT CODES ---
    df["maid_kids_profile_code"] = np.select(
        [
            df["infant_block"].eq(1),
            df["manykids_block"].eq(1),
            (df["infant_skill"].eq(1) & df["manykids_skill"].eq(1)),
            (df["infant_skill"].eq(1) | df["manykids_skill"].eq(1)),
        ],
        [-2, -1, 2, 1],
        default=0
    ).astype(int)

    df["maid_pets_profile_code"] = np.select(
        [
            df["cats_block"].eq(1),
            df["dogs_block"].eq(1),
            (df["cats_skill"].eq(1) & df["dogs_skill"].eq(1)),
            (df["cats_skill"].eq(1) | df["dogs_skill"].eq(1)),
        ],
        [-2, -1, 2, 1],
        default=0
    ).astype(int)

    base_living = np.select(
        [df["avoids_abudhabi"].eq(1), df["requires_private_room"].eq(1)],
        [-2, -1],
        default=0
    )
    df["maid_living_profile_code"] = (base_living + df["mobility_flex"]).astype(int)
    df["maid_dayoff_flex_code"]    = df["maidmts_dayoff_policy"].apply(_fx_dayoff_code).astype(int)
    df["maid_education_level"]     = df["maidpref_education"].apply(_fx_edu_level).astype(int)

    df["maid_care_profile_code"] = np.select(
        [
            (df["elderly_ok"].eq(1) & df["special_ok"].eq(1)),
        (df["elderly_ok"].eq(1) ^ df["special_ok"].eq(1)),
        ],
        [2, 1],
        default=0
    ).astype(int)

    # --- 3) Language one-hots from maid_speaks_language ---
    lang_col = "maid_speaks_language"
    token_series = df[lang_col].fillna("not_specified").str.strip().str.split()
    all_tokens = sorted({t for toks in token_series.dropna() for t in toks})
    langs = [t for t in all_tokens if t.lower() != "not_specified"]

    lang_cols = []
    for L in langs:
        colname = f"lang_{L}"
        df[colname] = token_series.apply(lambda toks, L=L: int(isinstance(toks, list) and L in toks)).astype(int)
        lang_cols.append(colname)

    # --- UI: previews ---
    with st.expander("Flags & Codes (preview)"):
        preview_cols = [
            "infant_block","manykids_block","infant_skill","manykids_skill",
            "cats_block","dogs_block","cats_skill","dogs_skill",
            "requires_private_room","avoids_abudhabi","mobility_flex",
            "dayoff_sunday_only","dayoff_flexible","maid_non_smoker_flag",
            "edu_university","edu_school","elderly_ok","special_ok",
            "energetic","no_attitude","no_tiktok","veg_friendly","maid_personality_score",
            "maid_kids_profile_code","maid_pets_profile_code","maid_living_profile_code",
            "maid_dayoff_flex_code","maid_education_level","maid_care_profile_code",
        ]
        st.dataframe(df[[c for c in preview_cols if c in df.columns]].head(10), use_container_width=True)

    if lang_cols:
        with st.expander("Language one-hots"):
            st.dataframe(df[lang_cols].head(10), use_container_width=True)

    # Optional: drop the raw maid text columns we parsed from
    drop_cols_raw = [
        "maidmts_household_type","maidpref_kids_experience","maidmts_pet_type","maidpref_pet_handling",
        "maidmts_living_arrangement","maidpref_travel","maidmts_dayoff_policy","maidpref_smoking",
        "maidpref_education","maidpref_caregiving_profile","maidpref_personality",
    ]
    # If you prefer to keep them, comment the next line:
    df = df.drop(columns=[c for c in drop_cols_raw if c in df.columns], errors="ignore")

    # --- download ---
    @st.cache_data
    def _to_csv_bytes2(_df):
        return _df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Engineered + Step 2B",
        data=_to_csv_bytes2(df),
        file_name="engineered_features_step2b.csv",
        mime="text/csv",
    )

    return df
# ---- Step 2B trigger (after your existing engineering section) ----
engineered_df_ss = st.session_state.get("engineered_df")

if engineered_df_ss is not None:
    if st.button("✅ Run Feature Engineering — Step 2B (Flags, Codes & Lang One-Hots)"):
        df_step2b = run_engineering_step2b(engineered_df_ss)
        st.session_state["engineered_df"] = df_step2b  # keep evolving the same key
        st.success("Step 2B applied and saved to session.")
else:
    st.info("Run the first Feature Engineering step before Step 2B.")

# ==============================
# Step 3 — Matching Score (exact logic, with policies)
# ==============================
import re

# ------ CONFIG (weights) ------
WEIGHTS = {
    "CAT_CONFLICT": -30,
    "INFANT_CONFLICT": -35,
    "INFANT_CONTRADICT": -25,
    "DOG_CONFLICT": -20,
    "MANYKIDS_CONFLICT": -20,
    "AD_CONFLICT": -20,
    "PRIVATE_ROOM_MISSING": -20,
    "DAYOFF_MISMATCH": -10,
    "NAT_PREF_MISS": -10,
    "REQ_UNSPECIFIED_SOFT": -3,
    "COOKING_MISSING_FLEX": -25,
    "INFANT_SKILL": +10,
    "MANYKIDS_SKILL": +5,
    "PET_SKILL": +5,
    "CUISINE_MATCH": +7,
    "CAREGIVING_MATCH": +8,
    "LANG_MATCH": +10,
    "LANG_MISS": -20,
    "PRIV_ROOM_FILIPINA": +10,
    "SOFT_POS": +2,
    "MOBILITY_2": +2,
    "MOBILITY_1": +1,
}

# ------ helpers ------
def _split_plus(s):
    if pd.isna(s) or s is None:
        return []
    return [t.strip().lower() for t in str(s).split("+") if t.strip()]

def _has_any(token_list, *candidates):
    cand = set([c.lower() for c in candidates])
    return any(t in cand for t in token_list)

def _contains_word(s, word):
    if pd.isna(s) or s is None: return False
    return word in str(s).lower()

def _expected_language(nat_group):
    nat = (str(nat_group).lower() if pd.notna(nat_group) else "")
    if nat in ("ethiopian",):
        return "arabic"
    if nat in ("filipina","indian","west_african"):
        return "english"
    return None

def _normalize_nat_group(s):
    if pd.isna(s) or s is None: return ""
    x = str(s).lower()
    if "filip" in x: return "filipina"
    if "ethiop" in x: return "ethiopian"
    if "west" in x and "afric" in x: return "west_african"
    if "india" in x: return "indian"
    return x

def _client_nat_pref_set(s):
    if pd.isna(s) or s is None: return set()
    x = str(s).lower()
    if "any" in x: return set()
    prefs = set()
    if re.search(r"filip", x): prefs.add("filipina")
    if re.search(r"ethiop", x): prefs.add("ethiopian")
    if re.search(r"west.*afric", x): prefs.add("west_african")
    if re.search(r"india", x): prefs.add("indian")
    return prefs

def _lang_tokens(s):
    if pd.isna(s) or s is None: return set()
    return set([t.strip().lower() for t in str(s).replace("/", " ").replace(",", " ").split() if t.strip()])

# ------ core scorer ------
def score_pair(row, policy="balanced"):
    notes = []
    decision = "OK"
    score = 50

    # Client needs
    cl_house = _split_plus(row.get("clientmts_household_type"))
    cl_pets = _split_plus(row.get("clientmts_pet_type"))
    cl_living = _split_plus(row.get("clientmts_living_arrangement"))
    cl_dayoff = _split_plus(row.get("clientmts_dayoff_policy"))
    cl_cuisines = [c for c in _split_plus(row.get("clientmts_cuisine_preference"))
                   if c not in ("not_specified","unspecified","none","other")]
    cl_natpref = _client_nat_pref_set(row.get("clientmts_nationality_preference"))
    cl_special = _split_plus(row.get("clientmts_special_cases"))

    client_has_baby = _has_any(cl_house, "baby","baby_and_kids")
    client_has_manykids = _has_any(cl_house, "many_kids")
    client_has_kids_over2 = client_has_manykids or _has_any(cl_house, "baby_and_kids")
    client_has_cat = _has_any(cl_pets, "cat","both")
    client_has_dog = _has_any(cl_pets, "dog","both")
    client_in_AD = _has_any(cl_living, "abu_dhabi")
    client_offers_pr = _has_any(cl_living, "private_room")
    client_needs_dayoff_paid = _has_any(cl_dayoff, "work_for_pay","stay_home_for_pay")

    # Maid facts (engineered)
    infant_block = int(row.get("infant_block", 0)) == 1
    infant_skill = int(row.get("infant_skill", 0)) == 1
    manykids_block = int(row.get("manykids_block", 0)) == 1
    manykids_skill = int(row.get("manykids_skill", 0)) == 1
    cats_block = int(row.get("cats_block", 0)) == 1
    dogs_block = int(row.get("dogs_block", 0)) == 1
    cats_skill = int(row.get("cats_skill", 0)) == 1
    dogs_skill = int(row.get("dogs_skill", 0)) == 1
    avoids_ad = int(row.get("avoids_abudhabi", 0)) == 1
    requires_pr = int(row.get("requires_private_room", 0)) == 1
    dayoff_flexible = int(row.get("dayoff_flexible", 0)) == 1
    dayoff_sunday_only = int(row.get("dayoff_sunday_only", 0)) == 1

    non_smoker = int(row.get("maid_non_smoker_flag", 0)) == 1
    energetic = int(row.get("energetic", 0)) == 1
    no_attitude = int(row.get("no_attitude", 0)) == 1
    no_tiktok = int(row.get("no_tiktok", 0)) == 1
    veg_friendly = int(row.get("veg_friendly", 0)) == 1

    elderly_ok = int(row.get("elderly_ok", 0)) == 1
    special_ok = int(row.get("special_ok", 0)) == 1

    mobility_flex = int(row.get("mobility_flex", 0)) if pd.notna(row.get("mobility_flex")) else 0

    maid_cooking = str(row.get("cooking_details") or "").lower().strip()
    maid_langs = _lang_tokens(row.get("maid_speaks_language"))
    maid_nat = _normalize_nat_group(row.get("maid_grouped_nationality"))

    # Hard gates
    if policy in ("strict","balanced"):
        if client_has_cat and cats_block:
            return 0, "BLOCK", "cats_block + cat_in_home"
        if client_has_baby and infant_block and (policy=="strict" or not infant_skill):
            return 0, "BLOCK", "infant_block + baby_in_home"
        if cl_cuisines:
            has_any_cuisine = (maid_cooking in cl_cuisines)
            if not has_any_cuisine:
                return 0, "BLOCK", "cooking_missing"

    # Big frictions
    any_big = False

    if client_has_baby and infant_block and infant_skill and policy in ("balanced","flexible"):
        score += WEIGHTS["INFANT_CONTRADICT"]; any_big = True; notes.append("infant_block+skill_confirm")

    if policy == "flexible" and client_has_cat and cats_block:
        score += WEIGHTS["CAT_CONFLICT"]; any_big = True; notes.append("cats_block + cat_in_home")

    if policy == "flexible" and client_has_baby and infant_block and not infant_skill:
        score += WEIGHTS["INFANT_CONFLICT"]; any_big = True; notes.append("infant_block + baby_in_home")

    if client_has_dog and dogs_block:
        score += WEIGHTS["DOG_CONFLICT"]; any_big = True; notes.append("dogs_block + dog_in_home")

    if client_has_manykids and manykids_block:
        score += WEIGHTS["MANYKIDS_CONFLICT"]; any_big = True; notes.append("manykids_block + many_kids_in_home")

    if client_in_AD and avoids_ad:
        score += WEIGHTS["AD_CONFLICT"]; any_big = True; notes.append("ad_avoid + client_in_AD")

    if requires_pr and not client_offers_pr:
        pr_penalty = WEIGHTS["PRIVATE_ROOM_MISSING"] - 5 if maid_nat == "filipina" else WEIGHTS["PRIVATE_ROOM_MISSING"]
        score += pr_penalty; any_big = True; notes.append("private_room_required_missing")

    if client_needs_dayoff_paid and not dayoff_flexible:
        score += WEIGHTS["DAYOFF_MISMATCH"]; any_big = True; notes.append("dayoff_mismatch")

    if policy == "flexible" and cl_cuisines:
        has_any_cuisine = (maid_cooking in cl_cuisines)
        if not has_any_cuisine:
            score += WEIGHTS["COOKING_MISSING_FLEX"]; any_big = True; notes.append("cooking_missing")

    if cl_natpref:
        if maid_nat not in cl_natpref:
            score += WEIGHTS["NAT_PREF_MISS"]; any_big = True; notes.append("client_nat_pref_miss")

    # Capability bonuses
    if client_has_baby and infant_skill:
        score += WEIGHTS["INFANT_SKILL"]; notes.append("infant_skill_ok")
    if client_has_kids_over2 and manykids_skill:
        score += WEIGHTS["MANYKIDS_SKILL"]; notes.append("kids_over2_skill_ok")
    if client_has_cat and cats_skill:
        score += WEIGHTS["PET_SKILL"]; notes.append("cats_handling_ok")
    if client_has_dog and dogs_skill:
        score += WEIGHTS["PET_SKILL"]; notes.append("dogs_handling_ok")

    if cl_cuisines:
        if maid_cooking == "not_specified" or maid_cooking in ("unspecified","none","other",""):
            score += WEIGHTS["REQ_UNSPECIFIED_SOFT"]; notes.append("cuisine_unspecified_confirm")
        else:
            if maid_cooking in cl_cuisines:
                score += WEIGHTS["CUISINE_MATCH"]; notes.append(f"cuisine_{maid_cooking}_ok")

    if _has_any(cl_special, "elderly") and elderly_ok:
        score += WEIGHTS["CAREGIVING_MATCH"]; notes.append("elderly_ok")
    if _has_any(cl_special, "special_needs") and special_ok:
        score += WEIGHTS["CAREGIVING_MATCH"]; notes.append("special_needs_ok")

    exp_lang = _expected_language(maid_nat)
    if exp_lang:
        if exp_lang in maid_langs:
            score += WEIGHTS["LANG_MATCH"]; notes.append(f"language_{exp_lang}_ok")
        else:
            score += WEIGHTS["LANG_MISS"]; any_big = True; notes.append(f"language_{exp_lang}_missing")

    if maid_nat == "filipina" and client_offers_pr:
        score += WEIGHTS["PRIV_ROOM_FILIPINA"]; notes.append("private_room_bonus_filipina")

    if cl_natpref and maid_nat in cl_natpref:
        score += 5; notes.append("client_nat_pref_hit")

    # Soft positives
    SOFT_TRAIT_CAP = 6
    soft_flags = [non_smoker, energetic, no_attitude, no_tiktok, veg_friendly]
    soft_count = sum(1 for f in soft_flags if f)
    soft_bonus = min(soft_count * WEIGHTS["SOFT_POS"], SOFT_TRAIT_CAP)
    if soft_bonus:
        score += soft_bonus; notes.append(f"soft_traits({soft_count})")

    if mobility_flex == 2:
        score += WEIGHTS["MOBILITY_2"]; notes.append("mobility_travel_and_relocate")
    elif mobility_flex == 1:
        score += WEIGHTS["MOBILITY_1"]; notes.append("mobility_travel_or_relocate")

    # Decision / clamp
    if any_big and decision != "BLOCK":
        decision = "REVIEW"
    score = max(0, min(100, score))
    return score, decision, "; ".join(notes) if notes else ""

def apply_matching_score(df, policy="balanced"):
    out = df.copy()
    results = out.apply(lambda r: score_pair(r, policy=policy), axis=1, result_type="expand")
    out["match_score"] = results[0]
    out["decision"] = results[1]
    out["score_notes"] = results[2]
    return out
# ---- Step 3 trigger: Matching Score ----
engineered_df_ss = st.session_state.get("engineered_df")

st.markdown("---")
st.header("Step 3 — Matching Score Calculation")

if engineered_df_ss is None:
    st.info("Run Feature Engineering (and Step 2B) first to enable matching score.")
else:
    policy = st.radio("Select policy", ["strict", "balanced", "flexible"], index=1, horizontal=True)
    if st.button("⚖️ Compute Matching Scores"):
        with st.spinner("Scoring pairs..."):
            scored_df = apply_matching_score(engineered_df_ss, policy=policy)
            st.session_state["scored_df"] = scored_df

        st.success("Matching scores computed and saved to session.")
        # Quick overview
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Avg score", f"{scored_df['match_score'].mean():.1f}")
        with c2: st.metric("REVIEW count", int((scored_df['decision']=="REVIEW").sum()))
        with c3: st.metric("BLOCK count", int((scored_df['decision']=="BLOCK").sum()))

        # Distribution and table
        st.bar_chart(scored_df["match_score"])
        st.dataframe(scored_df[["match_score","decision","score_notes"]].head(15), use_container_width=True)

        @st.cache_data
        def _to_csv_scores(df_): return df_.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Scored CSV",
            data=_to_csv_scores(scored_df),
            file_name=f"scored_{policy}.csv",
            mime="text/csv",
        )

# Optional: show last scored preview
if st.session_state.get("scored_df") is not None:
    st.markdown("### Latest matching score preview")
    st.dataframe(
        st.session_state["scored_df"][["match_score","decision","score_notes"]].head(10),
        use_container_width=True
    )

# ============================================
# Helpers for interactive pair scoring (Step 3B)
# ============================================

def _grouped_nat_from_raw(x: str) -> str:
    """Normalize maid nationality to the grouped tokens your scorer expects."""
    if x is None or (isinstance(x, float) and pd.isna(x)): return ""
    s = str(x).lower()
    if "filip" in s: return "filipina"
    if "ethiop" in s: return "ethiopian"
    if "west" in s and "afric" in s: return "west_african"
    if "india" in s: return "indian"
    return s

CLIENT_COLS = [
    # What the scorer reads from the client side
    "clientmts_household_type","clientmts_pet_type","clientmts_living_arrangement",
    "clientmts_dayoff_policy","clientmts_cuisine_preference","clientmts_nationality_preference",
    "clientmts_special_cases",
    # optional id context
    "client_name","contract_id","cc_type","tag_date","untag_date",
]
MAID_COLS = [
    # Flags & traits produced by Step 2B that the scorer uses
    "infant_block","infant_skill","manykids_block","manykids_skill",
    "cats_block","cats_skill","dogs_block","dogs_skill",
    "requires_private_room","avoids_abudhabi","mobility_flex",
    "dayoff_flexible","dayoff_sunday_only",
    "maid_non_smoker_flag","energetic","no_attitude","no_tiktok","veg_friendly",
    "elderly_ok","special_ok",
    # Text-ish fields read by the scorer
    "cooking_details","maid_speaks_language",
    # nationality (either grouped or raw to group now)
    "maid_grouped_nationality","maid_nationality",
    # optional id context
    "maid_id"
]

def _latest_by(df: pd.DataFrame, key_col: str, ts_col: str = "tag_date") -> pd.DataFrame:
    """Pick the latest row per key (by tag_date if present)."""
    df2 = df.copy()
    if ts_col in df2.columns:
        df2[ts_col] = pd.to_datetime(df2[ts_col], errors="coerce")
        # sort so last is latest
        df2 = df2.sort_values([key_col, ts_col], ascending=[True, True])
    # keep last occurrence per key
    return df2.groupby(key_col, as_index=False).tail(1)

def _make_pair_row(client_row: pd.Series, maid_row: pd.Series) -> pd.Series:
    """Build a synthetic row combining client_* needs with maid_* capabilities."""
    combined = {}

    # bring client fields
    for c in CLIENT_COLS:
        if c in client_row.index:
            combined[c] = client_row[c]

    # bring maid fields
    for c in MAID_COLS:
        if c in maid_row.index:
            combined[c] = maid_row[c]

    # ensure ids reflect the selection
    combined["client_name"] = client_row.get("client_name", combined.get("client_name"))
    combined["maid_id"] = maid_row.get("maid_id", combined.get("maid_id"))

    # guarantee a grouped nationality value for the scorer
    mg = combined.get("maid_grouped_nationality")
    if not mg:
        combined["maid_grouped_nationality"] = _grouped_nat_from_raw(maid_row.get("maid_nationality"))

    # default for cooking_details/langs (string-y)
    combined["cooking_details"] = (str(combined.get("cooking_details") or "").strip().lower() or "not_specified")
    if pd.isna(combined.get("maid_speaks_language")) or not str(combined.get("maid_speaks_language")).strip():
        combined["maid_speaks_language"] = "not_specified"

    return pd.Series(combined)
# ==============================
# Step 3B — Interactive Pair Scoring
# ==============================
st.markdown("---")
st.header("Step 3B — Try a Client ↔︎ Maid Pair (interactive)")

# Prefer scored_df if it exists, otherwise engineered_df
df_scored = st.session_state.get("scored_df", None)
df_engineered = st.session_state.get("engineered_df", None)

engineered_or_scored = df_scored if isinstance(df_scored, pd.DataFrame) else (
    df_engineered if isinstance(df_engineered, pd.DataFrame) else None
)

if engineered_or_scored is None:
    st.info("Run Feature Engineering (and Step 3) first to enable interactive pair scoring.")
else:
    dfE = engineered_or_scored.copy()

    # Build selection sources (latest record per entity)
    clients_latest = _latest_by(dfE, key_col="client_name")
    maids_latest   = _latest_by(dfE, key_col="maid_id")

    if clients_latest.empty or maids_latest.empty:
        st.warning("Not enough data to build client/maid pickers.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sel_client = st.selectbox(
                "Choose Client", 
                options=sorted(clients_latest["client_name"].dropna().unique().tolist()),
                index=0
            )
        with c2:
            # Optional filters for maid picker
            maid_filter_nat = st.selectbox(
                "Filter maids by nationality (optional)",
                options=["(all)"] + sorted(maids_latest.get("maid_nationality", pd.Series(dtype=str)).dropna().str.lower().unique().tolist()),
                index=0
            )

            _maids_df = maids_latest
            if maid_filter_nat != "(all)" and "maid_nationality" in _maids_df.columns:
                _maids_df = _maids_df[_maids_df["maid_nationality"].str.lower() == maid_filter_nat]

            sel_maid = st.selectbox(
                "Choose Maid",
                options=sorted(_maids_df["maid_id"].dropna().astype(str).unique().tolist()),
                index=0
            )

        # Pull the rows
        client_row = clients_latest[clients_latest["client_name"] == sel_client].iloc[0]
        maid_row   = maids_latest[maids_latest["maid_id"].astype(str) == str(sel_maid)].iloc[0]

        # Quick context side-by-side
        cA, cB = st.columns(2)
        with cA:
            st.caption("Client needs snapshot")
            st.write({
                "household": client_row.get("clientmts_household_type"),
                "pets": client_row.get("clientmts_pet_type"),
                "living": client_row.get("clientmts_living_arrangement"),
                "dayoff": client_row.get("clientmts_dayoff_policy"),
                "cuisine": client_row.get("clientmts_cuisine_preference"),
                "nat_pref": client_row.get("clientmts_nationality_preference"),
                "special": client_row.get("clientmts_special_cases"),
            })
        with cB:
            st.caption("Maid capabilities snapshot")
            st.write({
                "childcare flags": f"infant_block={int(maid_row.get('infant_block',0))}, infant_skill={int(maid_row.get('infant_skill',0))}, manykids_block={int(maid_row.get('manykids_block',0))}, manykids_skill={int(maid_row.get('manykids_skill',0))}",
                "pets flags": f"cats_block={int(maid_row.get('cats_block',0))}, dogs_block={int(maid_row.get('dogs_block',0))}, cats_skill={int(maid_row.get('cats_skill',0))}, dogs_skill={int(maid_row.get('dogs_skill',0))}",
                "living": f"requires_private_room={int(maid_row.get('requires_private_room',0))}, avoids_abudhabi={int(maid_row.get('avoids_abudhabi',0))}, mobility={int(maid_row.get('mobility_flex',0))}",
                "dayoff": f"flexible={int(maid_row.get('dayoff_flexible',0))}, sunday_only={int(maid_row.get('dayoff_sunday_only',0))}",
                "soft": f"non_smoker={int(maid_row.get('maid_non_smoker_flag',0))}, energetic={int(maid_row.get('energetic',0))}, no_attitude={int(maid_row.get('no_attitude',0))}, no_tiktok={int(maid_row.get('no_tiktok',0))}, veg={int(maid_row.get('veg_friendly',0))}",
                "care": f"elderly_ok={int(maid_row.get('elderly_ok',0))}, special_ok={int(maid_row.get('special_ok',0))}",
                "cooking": maid_row.get("cooking_details"),
                "languages": maid_row.get("maid_speaks_language"),
                "nationality": maid_row.get("maid_grouped_nationality") or _grouped_nat_from_raw(maid_row.get("maid_nationality")),
            })

        policy = st.radio("Policy for this pair", ["strict", "balanced", "flexible"], index=1, horizontal=True)

        if st.button("🎯 Score this Pair"):
            pair_row = _make_pair_row(client_row, maid_row)
            score, decision, notes = score_pair(pair_row, policy=policy)

            # Color helpers
            def _badge(text, color):
                st.markdown(
                    f"""
                    <div style="
                        display:inline-block; padding:6px 10px; border-radius:14px;
                        background:{color}; color:white; font-weight:600;">
                        {text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            def _score_color(s):
                if s >= 75: return "#2e7d32"   # green
                if s >= 50: return "#f9a825"   # amber
                return "#c62828"               # red

            def _decision_color(d):
                return {"OK":"#2e7d32","REVIEW":"#f9a825","BLOCK":"#c62828"}.get(d, "#607d8b")

            # Visuals
            st.subheader("Result")
            col1, col2 = st.columns([2,1])
            with col1:
                # score bar (uses default blue progress) plus colored badge
                st.progress(int(score) / 100.0)
                _badge(f"Score: {int(score)} / 100", _score_color(score))
            with col2:
                _badge(f"Decision: {decision}", _decision_color(decision))

            # Notes
            with st.expander("Rationale (notes)"):
                if notes:
                    # Light formatting: split semicolon tokens
                    bullets = [n.strip() for n in str(notes).split(";") if n.strip()]
                    st.markdown("\n".join([f"- {b}" for b in bullets]))
                else:
                    st.write("No specific adjustments; base score only.")

            # Save a tiny history
            hist = st.session_state.setdefault("pair_score_history", [])
            hist.append({
                "client_name": sel_client,
                "maid_id": sel_maid,
                "policy": policy,
                "score": int(score),
                "decision": decision,
                "notes": notes,
            })
            st.session_state["pair_score_history"] = hist

        # Optional: show history table
        if st.session_state.get("pair_score_history"):
            st.markdown("#### Recent pair scores")
            st.dataframe(pd.DataFrame(st.session_state["pair_score_history"]).tail(10), use_container_width=True)



# ==============================================
# Complaint Themes Extractor (Gemini) — inline block
# (expects st.secrets: GOOGLE_API_KEY, optional ALT_GOOGLE_API_KEY, MODEL_NAME)
# ==============================================
import io
import json
import time
import pandas as pd
import streamlit as st
import google.generativeai as genai

st.title("Complaint Themes Extractor")
st.caption("Step 1: Skeleton app — we’ll add upload & extraction next.")

st.success("If you can see this message after deployment, the app is wired up.")
st.write("Key loaded:", "GOOGLE_API_KEY" in st.secrets)
st.write("Model:", st.secrets.get("MODEL_NAME"))

# ---------- Gemini test (single row) with summarize-on-error fallback ----------
st.header("Gemini test (single row)")

# 1) Check secret
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("No GOOGLE_API_KEY in Secrets. Add it in the app’s Settings → Secrets.")
    st.stop()

# 2) Configure SDK ---- API key selector (primary vs alt) ----
key_choice = st.radio("API key to use", ["primary", "alt"], horizontal=True, key="api_key_choice")
ACTIVE_API_KEY = (
    st.secrets["GOOGLE_API_KEY"]
    if key_choice == "primary"
    else st.secrets.get("ALT_GOOGLE_API_KEY", st.secrets["GOOGLE_API_KEY"])
)
genai.configure(api_key=ACTIVE_API_KEY)
MODEL_NAME = st.secrets.get("MODEL_NAME", "gemini-2.5-flash-lite")

# 3) Inputs
system_prompt = st.text_area(
    "System instruction (paste yours here)",
    height=220,
    value=st.session_state.get("system_prompt", "")
)
sample_text = st.text_area(
    "One complaint_summary to test",
    height=120,
    value=st.session_state.get("sample_text", "")
)
go = st.button("Test Gemini")

def _parse_resp(resp):
    raw_text = getattr(resp, "text", None)
    if not raw_text and getattr(resp, "candidates", None):
        parts = resp.candidates[0].content.parts
        raw_text = "".join(getattr(p, "text", "") for p in parts)
    return raw_text or ""

def call_gemini_extract(system_instruction: str, complaint_text: str):
    """Primary extractor: expects three fields."""
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction
    )
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "all_case_themes":    {"type": "array", "items": {"type": "string"}},
                "subcategory_themes": {"type": "array", "items": {"type": "string"}},
                "evidence_spans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"quote": {"type": "string"}},
                        "required": ["quote"]
                    }
                }
            },
            "required": ["all_case_themes", "subcategory_themes", "evidence_spans"]
        },
        "max_output_tokens": 512,
        "temperature": 0.2,
    }
    resp = model.generate_content(
        [{"role": "user", "parts": [f'complaint_summary: """{complaint_text}"""']}],
        generation_config=generation_config,
    )
    raw = _parse_resp(resp)
    data = json.loads(raw or "{}")

    # normalize
    spans = data.get("evidence_spans", []) or []
    norm_spans = []
    for s in spans:
        if isinstance(s, dict) and "quote" in s:
            norm_spans.append({"quote": str(s["quote"])})
        elif isinstance(s, str):
            norm_spans.append({"quote": s})
    out = {
        "all_case_themes": data.get("all_case_themes", []) or [],
        "subcategory_themes": data.get("subcategory_themes", []) or [],
        "evidence_spans": norm_spans,
    }
    return out, raw

def call_gemini_summarize(complaint_text: str) -> str:
    """Fallback: compress to core issues only, then we re-run extraction on the summary."""
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"]
        },
        "max_output_tokens": 200,
        "temperature": 0.2,
    }
    # Minimal, deterministic summary prompt
    summary_instruction = (
        "Summarize this complaint_summary into 3–5 short bullets (one paragraph OK) "
        "capturing ONLY substantive reasons/behaviors that could cause dissatisfaction or replacement. "
        "Remove admin/process details (calls, links, scheduling, follow-ups), names, and dates. "
        "Be concise and neutral."
    )
    resp = model.generate_content(
        [
            {"role": "user", "parts": [
                summary_instruction + f'\n\ncomplaint_summary: """{complaint_text}"""'
            ]}
        ],
        generation_config=generation_config,
    )
    raw = _parse_resp(resp)
    data = json.loads(raw or "{}")
    return data.get("summary", "").strip()

if go:
    # remember inputs between reruns
    st.session_state["system_prompt"] = system_prompt
    st.session_state["sample_text"]  = sample_text

    try:
        out, raw = call_gemini_extract(system_prompt.strip(), sample_text.strip())
        st.success("Got JSON:")
        st.json(out)
    except Exception as e1:
        st.warning(f"Primary extraction failed ({e1}). Trying summarize-then-extract fallback…")
        try:
            summary = call_gemini_summarize(sample_text.strip())
            if not summary:
                raise RuntimeError("Summarizer returned empty text")
            out2, raw2 = call_gemini_extract(system_prompt.strip(), summary)
            st.success("Fallback succeeded on summarized text.")
            with st.expander("Summary used for fallback"):
                st.write(summary)
            st.json(out2)
        except Exception as e2:
            st.error(f"Fallback also failed: {e2}")

# ---------- Batch setup: upload & select column (robust) ----------
st.header("Batch extraction — upload & select column")

uploaded = st.file_uploader("Upload CSV (or Excel) with a complaint text column",
                            type=["csv", "xlsx"])

def load_table(file):
    # Empty file guard
    try:
        nbytes = getattr(file, "size", None) or file.getbuffer().nbytes
        if nbytes == 0:
            raise pd.errors.EmptyDataError("empty file")
    except Exception:
        pass

    file.seek(0)
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file)
    # CSV: auto-detect delimiter, tolerate weird encodings
    try:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python",
                           encoding="utf-8", on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        st.error("The file looks empty or not a valid CSV. Make sure it has a header row and at least one data row.")
        st.stop()
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python",
                           encoding_errors="ignore", on_bad_lines="skip")

if uploaded is not None:
    df = load_table(uploaded)
    st.session_state["df"] = df  # keep it for the next section
    st.write(f"Rows: {len(df):,} • Columns: {list(df.columns)}")

    # Let the user choose the text column
    default_col = "complaint_summary" if "complaint_summary" in df.columns else df.columns[0]
    text_col = st.selectbox("Which column contains the complaint text?",
                            options=list(df.columns),
                            index=list(df.columns).index(default_col))
    st.session_state["text_col"] = text_col

    # Preview
    work = df.copy()
    work["row_id"] = range(len(work))
    missing = work[text_col].isna().sum()
    st.info(f"Selected text column: **{text_col}** • Missing values: **{missing}**")
    st.subheader("Preview (first 10 rows)")
    st.dataframe(work[["row_id", text_col]].head(10), use_container_width=True)
else:
    st.caption("Upload a CSV or Excel file to continue.")

# ---------- Batch run & export (5 retries, then summarize→extract on 6th) ----------
st.header("Batch extraction — run & export")

df = st.session_state.get("df")
text_col = st.session_state.get("text_col")

if df is None or text_col is None:
    st.caption("Upload a file and select the text column above to enable batch extraction.")
else:
    col1, col2 = st.columns([1, 1])
    with col1:
        skip_no_complaint = st.checkbox("Skip rows where text equals 'no complaint'", value=True)
    with col2:
        max_rows = st.number_input("Max rows (0 = all)", min_value=0, value=0, step=50)

    go_batch = st.button("Run extraction on uploaded file")

    if go_batch:
        if not system_prompt.strip():
            st.error("Paste your System instruction in the Gemini test box above.")
            st.stop()

        work = df.copy()
        work["row_id"] = range(len(work))
        col_series = work[text_col].astype(str)
        valid = col_series.str.strip().ne("")
        if skip_no_complaint:
            valid &= col_series.str.strip().str.lower().ne("no complaint")

        idx = work.index[valid]
        if max_rows:
            idx = idx[:max_rows]

        st.write(f"Processing {len(idx)} of {len(work)} rows.")
        prog = st.progress(0.0)
        status = st.empty()
        results = []

        for k, i in enumerate(idx, start=1):
            txt = str(work.at[i, text_col]).strip()

            attempts = 0
            while True:
                try:
                    # attempts 1–5: primary extractor
                    out, _ = call_gemini_extract(system_prompt.strip(), txt)
                    themes = out.get("all_case_themes", [])
                    subs   = out.get("subcategory_themes", [])
                    break
                except Exception:
                    attempts += 1

                    # attempts 1–5: retry primary with backoff
                    if attempts < 6:
                        time.sleep(min(2 ** attempts, 10))
                        continue

                    # attempt 6: summarize → extract once
                    try:
                        sm = call_gemini_summarize(txt)
                        if not sm:
                            raise RuntimeError("Empty summary from fallback")
                        out2, _ = call_gemini_extract(system_prompt.strip(), sm)
                        themes = out2.get("all_case_themes", [])
                        subs   = out2.get("subcategory_themes", [])
                        break
                    except Exception:
                        themes, subs = [], []
                        st.warning(f"Row {work.at[i,'row_id']} failed after 5 retries + summarize fallback. Saved empty lists.")
                        break

            results.append({
                "row_id": work.at[i, "row_id"],
                "all_case_themes": themes,
                "subcategory_themes": subs
            })
            prog.progress(k / len(idx))
            status.write(f"Processed {k}/{len(idx)}")

        out_df = pd.DataFrame(results).sort_values("row_id")
        st.subheader("Sample of results")
        st.dataframe(out_df.head(20), use_container_width=True)

        # Export only the two arrays (plus row_id) as requested
        csv_bytes = out_df[["row_id", "all_case_themes", "subcategory_themes"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download themes CSV",
            data=csv_bytes,
            file_name="themes_subcategories.csv",
            mime="text/csv"
        )
        st.info(f"Export ready. Source rows: {len(work)} • Labeled rows: {len(out_df)} (ordered by row_id).")

# ==============================================
# Bulk JSON export — Clients & Maids
# ==============================================
import json
import hashlib
from datetime import datetime
from collections import Counter
import streamlit as st
import pandas as pd
import re

st.markdown("---")
st.header("Bulk JSON export — Clients & Maids")

# Reuse a source DF from session (scored → engineered → deduped)
df_scored     = st.session_state.get("scored_df")
df_engineered = st.session_state.get("engineered_df")
df_deduped    = st.session_state.get("deduped_df")
df_source = next((d for d in [df_scored, df_engineered, df_deduped] if isinstance(d, pd.DataFrame)), None)

if df_source is None or df_source.empty:
    st.info("Load and clean data first (Cleaning → Engineering). Then come back to export JSON.")
    st.stop()

required_cols = {"client_name", "maid_id", "complaint_summary", "complaint_comments", "tag_date"}
missing = required_cols - set(df_source.columns)
if missing:
    st.error(f"These columns are required: {sorted(missing)}")
    st.stop()

# ---------- options ----------
c1, c2, c3 = st.columns([1,1,1])
with c1:
    skip_no_complaint = st.checkbox("Skip 'no complaint' rows", value=True)
with c2:
    max_rows_per_entity = st.number_input("Max rows per entity (0 = all)", min_value=0, value=0, step=50)
with c3:
    show_preview = st.checkbox("Show sample rows in app", value=True)

# Use the same prompt from your Gemini tester
system_prompt_bulk = st.session_state.get("system_prompt", "")
if not (system_prompt_bulk or "").strip():
    st.warning("Paste the System instruction in the Gemini test box above so we can use it here.")
    st.stop()

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _coerce_list(x):
    """Tolerate list / JSON string / comma or semicolon separated string."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(t).strip() for t in arr if str(t).strip()]
        except Exception:
            pass
    parts = re.split(r"[;,|]+", s)
    return [p.strip() for p in parts if p.strip()]

@st.cache_data(show_spinner=False, ttl=60*60*24)
def extract_full_cached(system_prompt: str, complaint_text: str):
    """
    Normalized extractor for JSON export:
    returns {"themes": [...], "subcategories": [...], "evidence_spans": [{"quote": "..."}]}
    Uses your call_gemini_extract; falls back to call_gemini_summarize if available.
    """
    txt = (complaint_text or "").strip()
    if not txt or txt.lower() == "no complaint":
        return {"themes": [], "subcategories": [], "evidence_spans": []}

    try:
        out, _ = call_gemini_extract(system_prompt, txt)
    except Exception:
        # optional summarize→extract
        summarize_fn = globals().get("call_gemini_summarize", None)
        if callable(summarize_fn):
            try:
                sm = summarize_fn(txt) or txt
                out, _ = call_gemini_extract(system_prompt, sm)
            except Exception:
                out = {"all_case_themes": [], "subcategory_themes": [], "evidence_spans": []}
        else:
            out = {"all_case_themes": [], "subcategory_themes": [], "evidence_spans": []}

    themes  = _coerce_list(out.get("all_case_themes"))
    subs    = _coerce_list(out.get("subcategory_themes"))
    spans   = out.get("evidence_spans") or []
    norm_spans = []
    for s in spans:
        if isinstance(s, dict) and "quote" in s:
            norm_spans.append({"quote": str(s["quote"])})
        elif isinstance(s, str):
            norm_spans.append({"quote": s})
    return {"themes": themes, "subcategories": subs, "evidence_spans": norm_spans}

def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()

    # Clean text columns robustly
    s["complaint_summary"]  = (
        s["complaint_summary"].fillna("").astype(str).str.strip()
    )
    s["complaint_comments"] = (
        s["complaint_comments"].fillna("").astype(str).str.strip()
    )

    # Keep only rows with non-empty summaries (and optionally skip "no complaint")
    mask = s["complaint_summary"].ne("")
    if skip_no_complaint:
        mask &= s["complaint_summary"].str.lower().ne("no complaint")

    return s.loc[mask]

def _build_json(df: pd.DataFrame, group_col: str, prompt: str, max_rows: int = 0, progress_label: str = ""):
    df = _filter_df(df)
    groups = df.groupby(group_col, dropna=False)
    total_groups = groups.ngroups
    prog = st.progress(0.0) if progress_label else None

    items = []
    total_rows = 0

    for gi, (gkey, sub) in enumerate(groups, start=1):
        sub = sub.sort_values(by="tag_date", na_position="last")
        if max_rows:
            sub = sub.head(int(max_rows))

        entry_rows = []
        t_counter = Counter()
        s_counter = Counter()

        for _, r in sub.iterrows():
            ext = extract_full_cached(prompt, r["complaint_summary"])
            themes = ext["themes"]; subs = ext["subcategories"]; spans = ext["evidence_spans"]
            t_counter.update(themes)
            s_counter.update(subs)

            entry_rows.append({
                group_col: gkey,
                "client_name": r.get("client_name"),
                "maid_id":     r.get("maid_id"),
                "tag_date":    str(r.get("tag_date", "")),
                "complaint_summary":  r.get("complaint_summary", ""),
                "complaint_comments": r.get("complaint_comments", ""),
                "themes": themes,
                "subcategories": subs,
                "evidence_spans": spans,
            })

        total_rows += len(entry_rows)
        items.append({
            group_col: gkey,
            "count_rows": len(entry_rows),
            "top_themes": [{"theme": k, "count": v} for k, v in t_counter.most_common(20)],
            "top_subcategories": [{"subcategory": k, "count": v} for k, v in s_counter.most_common(20)],
            "rows": entry_rows,
        })

        if prog:
            prog.progress(gi / total_groups)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": st.secrets.get("MODEL_NAME", "gemini-2.5-flash-lite"),
        "system_prompt_sha1": _sha1(prompt),
        "group_by": group_col,
        "total_entities": len(items),
        "total_rows": total_rows,
        "items": items,
    }
    return payload

tab_clients, tab_maids = st.tabs(["📁 Export by Client", "🧑‍🍳 Export by Maid"])

with tab_clients:
    st.caption("Build a JSON file with all complaints grouped by client.")
    # Optional subset
    client_list = sorted(df_source["client_name"].dropna().astype(str).unique().tolist())
    sel_clients = st.multiselect("Limit to specific clients (optional)", client_list, default=[])
    df_work = df_source if not sel_clients else df_source[df_source["client_name"].astype(str).isin(sel_clients)]

    if st.button("Build JSON (clients)"):
        payload = _build_json(df_work, "client_name", system_prompt_bulk, max_rows_per_entity, "clients")
        # Quick preview
        if show_preview and payload["items"]:
            st.subheader("Preview")
            preview_rows = []
            for it in payload["items"][:2]:  # first 2 clients
                preview_rows.extend(it["rows"][:2])  # first 2 rows each
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
        # Download
        bytes_out = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ Download complaints_by_clients.json", data=bytes_out,
                           file_name="complaints_by_clients.json", mime="application/json")
        st.success(f"Built {payload['total_entities']} clients • {payload['total_rows']} rows.")

with tab_maids:
    st.caption("Build a JSON file with all complaints grouped by maid.")
    maid_list = sorted(df_source["maid_id"].dropna().astype(str).unique().tolist())
    sel_maids = st.multiselect("Limit to specific maids (optional)", maid_list, default=[])
    df_work2 = df_source if not sel_maids else df_source[df_source["maid_id"].astype(str).isin(sel_maids)]

    if st.button("Build JSON (maids)"):
        payload2 = _build_json(df_work2, "maid_id", system_prompt_bulk, max_rows_per_entity, "maids")
        # Quick preview
        if show_preview and payload2["items"]:
            st.subheader("Preview")
            preview_rows2 = []
            for it in payload2["items"][:2]:  # first 2 maids
                preview_rows2.extend(it["rows"][:2])  # first 2 rows each
            st.dataframe(pd.DataFrame(preview_rows2), use_container_width=True)
        # Download
        bytes_out2 = json.dumps(payload2, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ Download complaints_by_maids.json", data=bytes_out2,
                           file_name="complaints_by_maids.json", mime="application/json")
        st.success(f"Built {payload2['total_entities']} maids • {payload2['total_rows']} rows.")





