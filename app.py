# app_cleaning.py
# Streamlit: Cleaning & Preprocessing (Cleaning v1)

import io
import re
from statistics import mode, StatisticsError

import numpy as np
import pandas as pd
import streamlit as st

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

import pandas as pd
import streamlit as st

def run_engineering(deduped_df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("---")
    st.header("Step 2 — Feature Engineering from ERP Lists (Exact Parity)")

    df = deduped_df.copy()

    # ---------- Helpers ----------
    def _standardize_series_to_trait_lists(series: pd.Series, do_lebanon_fix=False):
        # Fill, lower, replace '||' with ',', optional 'lebanon'→'lebanese', split, strip, keep non-empty
        s = (
            series.fillna('')
            .astype(str)
            .str.lower()
            .str.replace('||', ',', regex=False)
        )
        if do_lebanon_fix:
            s = s.str.replace('lebanon', 'lebanese', regex=False)
        return s.str.split(',').apply(lambda x: [t.strip() for t in x if t.strip() != ''])

    def get_all_trait_counts_standardized(df_, col_name):
        traits_series = (
            df_[col_name]
            .dropna()
            .astype(str)
            .str.replace('||', ',', regex=False)
            .str.lower()
            .str.split(',')
            .explode()
            .str.strip()
        )
        vc = traits_series.value_counts()
        return pd.DataFrame({'Trait': vc.index, 'Count': vc.values})

    # ---------- Client MTS ----------
    # v1: Standardize 'lebanon'→'lebanese' in client_mts_at_hiring (exactly as in your code)
    df['client_mts_at_hiring'] = (
        df['client_mts_at_hiring']
        .astype(str)
        .str.replace('||', ',', regex=False)
        .str.lower()
        .str.replace('lebanon', 'lebanese', regex=False)
    )

    # Trait counts (display)
    client_mts_table = get_all_trait_counts_standardized(df, "client_mts_at_hiring")
    maid_mts_table   = get_all_trait_counts_standardized(df, "maid_mts_at_hiring")
    maid_pref_table  = get_all_trait_counts_standardized(df, "maids_custom_preferences_at_hiring")

    with st.expander("All Traits (Raw → Canonical Tokens)"):
        st.subheader("Client MTS (at hiring)")
        st.dataframe(client_mts_table, use_container_width=True)
        st.subheader("Maid MTS (at hiring)")
        st.dataframe(maid_mts_table, use_container_width=True)
        st.subheader("Maid Custom Preferences (at hiring)")
        st.dataframe(maid_pref_table, use_container_width=True)

    # Safe fill before tokenizing
    df['client_mts_at_hiring'] = df['client_mts_at_hiring'].fillna('no_preference')
    df['maid_mts_at_hiring'] = df['maid_mts_at_hiring'].fillna('no_preference')
    df['maids_custom_preferences_at_hiring'] = df['maids_custom_preferences_at_hiring'].fillna('no_preference')

    # Tokenize to lists (client lebanon→lebanese fix ON; maid & prefs as-is)
    client_mts_raw = _standardize_series_to_trait_lists(df['client_mts_at_hiring'], do_lebanon_fix=True)
    maid_mts_raw   = _standardize_series_to_trait_lists(df['maid_mts_at_hiring'])
    maid_pref_raw  = _standardize_series_to_trait_lists(df['maids_custom_preferences_at_hiring'])

    # ---------- Client feature extractors (prefixed to avoid name shadowing) ----------
    def client_get_household_type(traits):
        baby = "has a baby younger than 2 years old" in traits
        kids = "has 3 kids or more" in traits
        no_kids = "i prefer working with a family with no kids" in traits
        if baby and kids: return "baby_and_kids"
        if baby:          return "baby"
        if kids:          return "many_kids"
        if no_kids:       return "no_kids"
        return "none"

    def client_get_special_case(traits):
        elderly = "elderly parent at home" in traits
        special = "special needs kid" in traits
        if elderly and special: return "elderly_and_special"
        if elderly:             return "elderly"
        if special:             return "special_needs"
        return "none"

    def client_get_pet_type(traits):
        cat = "has a cat" in traits
        dog = "has a dog" in traits
        if cat and dog: return "both"
        if cat:         return "cat"
        if dog:         return "dog"
        return "none"

    def client_get_dayoff_policy(traits):
        policies = []
        if "give day off other than sunday negation" in traits:
            policies.append("flexible")
        if "work on her day off for pay" in traits:
            policies.append("work_for_pay")
        if "stay home on day off" in traits or "take her day off at home for pay" in traits:
            policies.append("stay_home_for_pay")
        return "+".join(policies) if policies else "none"

    def client_get_nationality_preference(traits):
        preferred = []
        for nationality in ["filipina", "west african nationality", "ethiopian maid", "indian"]:
            if nationality in traits:
                preferred.append(nationality)
        return "+".join(preferred) if preferred else "any"

    def client_get_living_arrangement(traits):
        arrangement = []
        if "live out" in traits:
            arrangement.append("live_out")
        if "has a private room" in traits:
            arrangement.append("private_room")
        if "lives in abu dhabi" in traits:
            arrangement.append("abu_dhabi")
        return "+".join(arrangement) if arrangement else "unspecified"

    def client_get_cuisine_preference(traits):
        selected = [c for c in ["lebanese", "khaleeji", "international"] if c in traits]
        return "+".join(selected) if selected else "other"

    # Apply client extractors
    df["clientmts_household_type"]         = client_mts_raw.apply(client_get_household_type)
    df["clientmts_special_cases"]          = client_mts_raw.apply(client_get_special_case)
    df["clientmts_pet_type"]               = client_mts_raw.apply(client_get_pet_type)
    df["clientmts_dayoff_policy"]          = client_mts_raw.apply(client_get_dayoff_policy)
    df["clientmts_nationality_preference"] = client_mts_raw.apply(client_get_nationality_preference)
    df["clientmts_living_arrangement"]     = client_mts_raw.apply(client_get_living_arrangement)
    df["clientmts_cuisine_preference"]     = client_mts_raw.apply(client_get_cuisine_preference)

    # Client summaries
    client_summary_tables = {
        "clientmts_household_type":         df["clientmts_household_type"].value_counts().reset_index(names=["value","count"]),
        "clientmts_special_cases":          df["clientmts_special_cases"].value_counts().reset_index(names=["value","count"]),
        "clientmts_pet_type":               df["clientmts_pet_type"].value_counts().reset_index(names=["value","count"]),
        "clientmts_dayoff_policy":          df["clientmts_dayoff_policy"].value_counts().reset_index(names=["value","count"]),
        "clientmts_nationality_preference": df["clientmts_nationality_preference"].value_counts().reset_index(names=["value","count"]),
        "clientmts_living_arrangement":     df["clientmts_living_arrangement"].value_counts().reset_index(names=["value","count"]),
        "clientmts_cuisine_preference":     df["clientmts_cuisine_preference"].value_counts().reset_index(names=["value","count"]),
    }
    with st.expander("Client Feature Distributions", expanded=False):
        for name, tbl in client_summary_tables.items():
            st.caption(name)
            st.dataframe(tbl, use_container_width=True)

    # ---------- Maid MTS extractors ----------
    def maid_get_household_type(traits):
        baby = "has a baby younger than 2 years old" in traits
        kids = "has 3 kids or more" in traits
        if baby and kids: return "baby_and_kids"
        if baby:          return "baby"
        if kids:          return "many_kids"
        return "none"

    def maid_get_pet_type(traits):
        cat = "has a cat" in traits
        dog = "has a dog" in traits
        if cat and dog: return "both"
        if cat:         return "cat"
        if dog:         return "dog"
        return "none"

    def maid_get_dayoff_policy(traits):
        if "give day off other than sunday negation" in traits:
            return "flexible"
        return "unspecified"

    def maid_get_living_arrangement(traits):
        if "has a private room" in traits and "lives in abu dhabi" in traits:
            return "private_room+avoids_abu_dhabi"
        if "has a private room" in traits:
            return "private_room"
        if "lives in abu dhabi" in traits:
            return "avoids_abu_dhabi"
        return "unspecified"

    # Apply maid extractors
    df["maidmts_household_type"]    = maid_mts_raw.apply(maid_get_household_type)
    df["maidmts_pet_type"]          = maid_mts_raw.apply(maid_get_pet_type)
    df["maidmts_dayoff_policy"]     = maid_mts_raw.apply(maid_get_dayoff_policy)
    df["maidmts_living_arrangement"] = maid_mts_raw.apply(maid_get_living_arrangement)

    # Maid summaries
    maid_summary_tables = {
        "maidmts_household_type":    df["maidmts_household_type"].value_counts().reset_index(names=["value","count"]),
        "maidmts_pet_type":          df["maidmts_pet_type"].value_counts().reset_index(names=["value","count"]),
        "maidmts_dayoff_policy":     df["maidmts_dayoff_policy"].value_counts().reset_index(names=["value","count"]),
        "maidmts_living_arrangement": df["maidmts_living_arrangement"].value_counts().reset_index(names=["value","count"]),
    }
    with st.expander("Maid Feature Distributions", expanded=False):
        for name, tbl in maid_summary_tables.items():
            st.caption(name)
            st.dataframe(tbl, use_container_width=True)

    # ---------- Maid Preferences extractors ----------
    def maidpref_get_education(traits):
        has_university = "has university degree" in traits
        has_school     = "has a school degree" in traits
        if has_university and has_school: return "both"
        if has_university:                return "university"
        if has_school:                    return "school"
        return "not_specified"

    def maidpref_get_kids_experience(traits):
        less2 = (
            "maid has experience with kids under 6 months old" in traits
            or "maid has experience with kids between 6 months and 2 years" in traits
        )
        above2 = "maid has experience with kids above 2 years old" in traits
        if less2 and above2: return "both"
        if less2:            return "lessthan2"
        if above2:           return "above2"
        return "none"

    def maidpref_get_pet_handling(traits):
        cats = "handles multiple cats" in traits
        dogs = "handles multiple dogs" in traits
        if cats and dogs: return "both"
        if cats:          return "cats"
        if dogs:          return "dogs"
        return "none"

    def maidpref_get_personality(traits):
        checks = {
            "maid is energetic": "energetic",
            "maid does not have attitude": "no_attitude",
            "maid does not have tiktok": "no_tiktok",
            "flexible to work in a vegetarian household": "veg_friendly",
        }
        matched = [label for text, label in checks.items() if text in traits]
        return "+".join(matched) if matched else "not_mentioned"

    def maidpref_get_travel(traits):
        travel = "maid does not mind travelling" in traits
        relocate = "don’t mind relocating" in traits or "don't mind relocating" in traits
        if travel and relocate: return "travel_and_relocate"
        if travel:              return "travel"
        if relocate:            return "relocate"
        return "no"

    def maidpref_get_smoking(traits):
        return "non_smoker" if "maid is not a smoker" in traits else "unspecified"

    def maidpref_get_caregiving(traits):
        elderly_exp = "experienced with elderly person" in traits
        elderly_will = "willing to handle elderly persons" in traits
        special = "willing to handle special needs kid" in traits
        if (elderly_exp or elderly_will) and special: return "elderly_and_special"
        if (elderly_exp or elderly_will):             return "elderly_experienced"
        if special:                                   return "special_needs"
        return "none"

    # Apply maid preference extractors
    df["maidpref_education"]          = maid_pref_raw.apply(maidpref_get_education)
    df["maidpref_kids_experience"]    = maid_pref_raw.apply(maidpref_get_kids_experience)
    df["maidpref_pet_handling"]       = maid_pref_raw.apply(maidpref_get_pet_handling)
    df["maidpref_personality"]        = maid_pref_raw.apply(maidpref_get_personality)
    df["maidpref_travel"]             = maid_pref_raw.apply(maidpref_get_travel)
    df["maidpref_smoking"]            = maid_pref_raw.apply(maidpref_get_smoking)
    df["maidpref_caregiving_profile"] = maid_pref_raw.apply(maidpref_get_caregiving)

    # Maid preference summaries
    maid_pref_summary_tables = {
        "maidpref_education":          df["maidpref_education"].value_counts().reset_index(names=["value","count"]),
        "maidpref_kids_experience":    df["maidpref_kids_experience"].value_counts().reset_index(names=["value","count"]),
        "maidpref_pet_handling":       df["maidpref_pet_handling"].value_counts().reset_index(names=["value","count"]),
        "maidpref_personality":        df["maidpref_personality"].value_counts().reset_index(names=["value","count"]),
        "maidpref_travel":             df["maidpref_travel"].value_counts().reset_index(names=["value","count"]),
        "maidpref_smoking":            df["maidpref_smoking"].value_counts().reset_index(names=["value","count"]),
        "maidpref_caregiving_profile": df["maidpref_caregiving_profile"].value_counts().reset_index(names=["value","count"]),
    }
    with st.expander("Maid Preferences Distributions", expanded=False):
        for name, tbl in maid_pref_summary_tables.items():
            st.caption(name)
            st.dataframe(tbl, use_container_width=True)

    # ---------- Preview + Download ----------
    st.subheader("Engineered Columns (preview)")
    engineered_cols = [
        # client
        "clientmts_household_type","clientmts_special_cases","clientmts_pet_type",
        "clientmts_dayoff_policy","clientmts_nationality_preference",
        "clientmts_living_arrangement","clientmts_cuisine_preference",
        # maid
        "maidmts_household_type","maidmts_pet_type","maidmts_dayoff_policy","maidmts_living_arrangement",
        # maid preferences
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
