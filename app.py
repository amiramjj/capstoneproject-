# streamlit_app_matching_score.py
# Phase 1 — Matching Score ONLY (per your notebooks)
#  Mirrors policy modes & MT-driven reasons
#  Engineers from RAW ERP lists into canonical MTs, then scores
#  No sub-score breakdown (hidden by request)
#  Reasons shown = MT-derived signals only (conflicts, frictions, covered needs)

import re
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import pandas as pd
import streamlit as st

# =============================
# Policy modes (Table 1 weights)
# =============================
POLICIES = {
    "Strict": {
        "baseline": 70,
        "hard_conflict_each": -40,
        "hard_conflict_extra": -10,
        "private_room_missing": -15,
        "dayoff_mismatch": -12,
        "language_missing": -10,
        "infant_skill_needed": +8,
        "kids_over2_skill_needed": +6,
        "pet_handling_needed": +4,
        "experience_cap": +10,
        "experience_slope": 2.0,
        "mobility_flex": +4,
        "soft_traits_cap": +4,
        "avoids_abu_dhabi": -10,
        "skill_needed_total_per_extra": 0,
        "skill_needed_total_cap": 0,
        "pref_mismatch_total_per_extra": -3,
        "pref_mismatch_total_cap": -6,
        "ok_cut": 75,
        "review_low": 55,
    },
    "Balanced": {
        "baseline": 70,
        "hard_conflict_each": -30,
        "hard_conflict_extra": -5,
        "private_room_missing": -12,
        "dayoff_mismatch": -10,
        "language_missing": -8,
        "infant_skill_needed": +10,
        "kids_over2_skill_needed": +8,
        "pet_handling_needed": +6,
        "experience_cap": +12,
        "experience_slope": 2.0,
        "mobility_flex": +6,
        "soft_traits_cap": +5,
        "avoids_abu_dhabi": -7,
        "skill_needed_total_per_extra": +1,
        "skill_needed_total_cap": +3,
        "pref_mismatch_total_per_extra": -2,
        "pref_mismatch_total_cap": -4,
        "ok_cut": 70,
        "review_low": 50,
    },
    "Flexible": {
        "baseline": 70,
        "hard_conflict_each": -20,
        "hard_conflict_extra": 0,
        "private_room_missing": -8,
        "dayoff_mismatch": -6,
        "language_missing": -5,
        "infant_skill_needed": +12,
        "kids_over2_skill_needed": +10,
        "pet_handling_needed": +8,
        "experience_cap": +14,
        "experience_slope": 2.0,
        "mobility_flex": +8,
        "soft_traits_cap": +6,
        "avoids_abu_dhabi": -4,
        "skill_needed_total_per_extra": +2,
        "skill_needed_total_cap": +4,
        "pref_mismatch_total_per_extra": -1,
        "pref_mismatch_total_cap": -3,
        "ok_cut": 65,
        "review_low": 45,
    },
}

# Nationality → expected language(s)
LANG_EXPECT_MAP = {
    "ethiopian": {"ar"},
    "filipina": {"en"},
    "indian": {"en"},
    "west_african": {"en"},
}

# Fine cuisine → coarse group (so client khaleeji matches e.g. “saudi”)
CUISINE_MAP = {
    # khaleeji cluster
    "saudi": "khaleeji", "emirati": "khaleeji", "kuwaiti": "khaleeji", "qatari": "khaleeji", "omani": "khaleeji", "bahraini": "khaleeji",
    # passthroughs
    "lebanese": "lebanese",
    # common intl examples
    "indian": "international", "chinese": "international", "italian": "international", "thai": "international"
}

# =============================
# Normalization helpers
# =============================
SEP_RE = re.compile(r"[+,/;]| +")

def split_tokens(raw: Any) -> List[str]:
    if raw is None:
        return []
    toks = [t.strip().lower() for t in SEP_RE.split(str(raw)) if t and t.strip()]
    out, seen = [], set()
    for t in toks:
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out

# =============================
# Canonical MTs + Preferences (from RAW ERP)
# Mirrors your enumerations; unknowns → safe defaults
# =============================
@dataclass
class Canonical:
    clientmts_household_type: str = "none"            # {none,baby,many_kids,baby_and_kids}
    clientmts_special_cases: str = "none"             # {none,elderly,special_needs,elderly_and_special}
    clientmts_pet_type: str = "none"                  # {none,cat,dog,both}
    clientmts_dayoff_policy: str = "none"             # {none,flexible,work_for_pay,stay_home_for_pay,combos}
    clientmts_nationality_preference: str = "any"     # {any,filipina,west_african,ethiopian,...}
    clientmts_living_arrangement: str = "unspecified" # {unspecified,live_out,private_room,abu_dhabi,combos}
    clientmts_cuisine_preference: str = "other"       # {other,lebanese,khaleeji,international,combos}

    maidmts_household_type: str = "none"
    maidmts_pet_type: str = "none"
    maidmts_dayoff_policy: str = "unspecified"        # {flexible,unspecified}
    maidmts_living_arrangement: str = "unspecified"   # {unspecified,private_room,avoids_abu_dhabi,combo}

    maidpref_education: str = "not_specified"
    maidpref_kids_experience: str = "none"            # {none,lessthan2,above2,both}
    maidpref_pet_handling: str = "none"               # {none,cats,dogs,both}
    maidpref_personality: List[str] = None            # list of descriptors
    maidpref_travel: str = "unspecified"              # {no,travel,relocate,travel_and_relocate}
    maidpref_smoking: str = "unspecified"
    maidpref_caregiving_profile: str = "none"

    maid_nationality: str = ""
    years_of_experience: float = 0.0


@dataclass
class PairInputs:
    client_id: str
    maid_id: str
    assignment_date: str | None
    client_mts_raw: List[str]
    maid_mts_raw: List[str]
    engineered: Canonical

# --- very light parser to canonical (you can swap with your exact functions)

def canonicalize_from_raw(client_raw: str, maid_raw: str, engineered_overrides: Dict[str, Any]) -> Canonical:
    c = Canonical()
    # start with overrides if provided (engineered columns take precedence for parity)
    for k, v in (engineered_overrides or {}).items():
        if hasattr(c, k) and v not in (None, ""):
            setattr(c, k, v if not isinstance(v, str) else v.strip().lower())

    ctoks, mtoks = split_tokens(client_raw), split_tokens(maid_raw)

    # Household type (coarse)
    if any(t in ctoks for t in ["baby", "infant"]):
        c.clientmts_household_type = "baby"
    if "many_kids" in ctoks or "3kids" in ctoks or "3+kids" in ctoks:
        c.clientmts_household_type = "many_kids"
    if "baby" in ctoks and ("many_kids" in ctoks or "3+kids" in ctoks):
        c.clientmts_household_type = "baby_and_kids"

    # Pets (client)
    has_cat = "cat" in ctoks or "cats" in ctoks
    has_dog = "dog" in ctoks or "dogs" in ctoks
    if has_cat and has_dog: c.clientmts_pet_type = "both"
    elif has_cat: c.clientmts_pet_type = "cat"
    elif has_dog: c.clientmts_pet_type = "dog"

    # Day-off (client)
    if any(t in ctoks for t in ["flex", "flexible"]):
        c.clientmts_dayoff_policy = "flexible"
    elif any(t in ctoks for t in ["work_for_pay", "workforpay", "stay_home_for_pay", "stayhomeforpay", "sunday_only", "combos"]):
        # treat any non-flex specific directive as non-flex combo
        c.clientmts_dayoff_policy = "combos"

    # Living arrangement (client)
    if "private_room" in ctoks: c.clientmts_living_arrangement = "private_room"
    elif "live_out" in ctoks or "liveout" in ctoks: c.clientmts_living_arrangement = "live_out"
    if "abu_dhabi" in ctoks or "abudhabi" in ctoks: c.clientmts_living_arrangement = "abu_dhabi"

    # Cuisine (client)
    cuisines = [CUISINE_MAP.get(t, t) for t in ctoks if t in CUISINE_MAP]
    if cuisines:
        # if multiple distinct, call it combos; else the single mapped value
        c.clientmts_cuisine_preference = "combos" if len(set(cuisines)) > 1 else cuisines[0]

    # Maid MTs (pet, dayoff, living)
    if any(t in mtoks for t in ["no_cats", "pet_none"]):
        c.maidmts_pet_type = "none"
    if "avoids_abu_dhabi" in mtoks: c.maidmts_living_arrangement = "avoids_abu_dhabi"
    if "private_room" in mtoks: c.maidmts_living_arrangement = "private_room"
    if any(t in mtoks for t in ["flex", "flexible"]): c.maidmts_dayoff_policy = "flexible"

    # Maid prefs
    if any(t in mtoks for t in ["travel", "relocate"]): c.maidpref_travel = "travel" if "relocate" not in mtoks else "travel_and_relocate"

    return c

# =============================
# Flags & interactions (MT-derived only)
# =============================

def make_flags(cn: Canonical) -> Dict[str, Any]:
    flags = {
        # hard conflicts
        "infant_conflict": False,
        "manykids_conflict": False,
        "cat_conflict": False,
        "dog_conflict": False,
        # frictions
        "private_room_missing": False,
        "dayoff_mismatch": False,
        "language_expected_missing": False,
        "avoids_abu_dhabi": False,
        # needs covered
        "infant_skill_needed": False,
        "kids_over2_skill_needed": False,
        "handles_cats_needed": False,
        "handles_dogs_needed": False,
        # totals
        "hard_conflict_total": 0,
        "pref_mismatch_total": 0,
        "skill_needed_total": 0,
    }

    # Childcare hard conflicts / skills (proxy via kids_experience)
    if cn.clientmts_household_type in {"baby", "baby_and_kids"}:
        if cn.maidpref_kids_experience in {"none", "lessthan2"}:
            flags["infant_conflict"] = True
        else:
            flags["infant_skill_needed"] = True
    if cn.clientmts_household_type in {"many_kids", "baby_and_kids"}:
        if cn.maidpref_kids_experience in {"none"}:
            flags["manykids_conflict"] = True
        else:
            flags["kids_over2_skill_needed"] = True

    # Pets conflicts / skills
    if cn.clientmts_pet_type in {"cat", "both"} and cn.maidpref_pet_handling not in {"cats", "both"}:
        flags["cat_conflict"] = True
    if cn.clientmts_pet_type in {"dog", "both"} and cn.maidpref_pet_handling not in {"dogs", "both"}:
        flags["dog_conflict"] = True
    if cn.clientmts_pet_type in {"cat", "both"} and cn.maidpref_pet_handling in {"cats", "both"}:
        flags["handles_cats_needed"] = True
    if cn.clientmts_pet_type in {"dog", "both"} and cn.maidpref_pet_handling in {"dogs", "both"}:
        flags["handles_dogs_needed"] = True

    # Living arrangement frictions
    if cn.maidmts_living_arrangement == "private_room" and cn.clientmts_living_arrangement != "private_room":
        flags["private_room_missing"] = True
    if cn.maidmts_living_arrangement == "avoids_abu_dhabi" and cn.clientmts_living_arrangement == "abu_dhabi":
        flags["avoids_abu_dhabi"] = True

    # Day-off
    if cn.clientmts_dayoff_policy in {"work_for_pay", "stay_home_for_pay", "combos"} and cn.maidmts_dayoff_policy != "flexible":
        flags["dayoff_mismatch"] = True

    # Language expectation (by maid nationality)
    expected = LANG_EXPECT_MAP.get((cn.maid_nationality or "").lower(), set())
    if expected:
        # NOTE: Replace below with your real “maid speaks” columns to avoid assumption
        speaks_en = (cn.maid_nationality or "").lower() in {"filipina", "indian", "west_african"}
        speaks_ar = (cn.maid_nationality or "").lower() in {"ethiopian"}
        if ("en" in expected and not speaks_en) or ("ar" in expected and not speaks_ar):
            flags["language_expected_missing"] = True

    # Totals
    flags["hard_conflict_total"] = sum(int(flags[k]) for k in ["infant_conflict","manykids_conflict","cat_conflict","dog_conflict"])
    flags["pref_mismatch_total"] = sum(int(flags[k]) for k in ["private_room_missing","dayoff_mismatch","language_expected_missing","avoids_abu_dhabi"])
    flags["skill_needed_total"] = sum(int(flags[k]) for k in ["infant_skill_needed","kids_over2_skill_needed","handles_cats_needed","handles_dogs_needed"])

    return flags

# =============================
# Scoring (MT-only reasons surfaced)
# =============================

MT_REASON_ORDER = [
    "infant_conflict","manykids_conflict","cat_conflict","dog_conflict",
    "private_room_missing","dayoff_mismatch","language_expected_missing","avoids_abu_dhabi",
    "infant_skill_needed","kids_over2_skill_needed","handles_cats_needed","handles_dogs_needed"
]

REASON_TEXT = {
    "infant_conflict": "Infant conflict (maid excludes <2y)",
    "manykids_conflict": "Many-kids conflict",
    "cat_conflict": "Cat conflict",
    "dog_conflict": "Dog conflict",
    "private_room_missing": "Private room missing",
    "dayoff_mismatch": "Day-off mismatch",
    "language_expected_missing": "Language expected but missing",
    "avoids_abu_dhabi": "Avoids Abu Dhabi",
    "infant_skill_needed": "Infant skill covered",
    "kids_over2_skill_needed": "Kids>2 skill covered",
    "handles_cats_needed": "Cat-handling covered",
    "handles_dogs_needed": "Dog-handling covered",
}

def exp_contrib(years: float, policy: Dict[str, Any]) -> float:
    years = max(0.0, float(years or 0.0))
    return min(policy["experience_cap"], policy["experience_slope"] * years)


def compute_score(flags: Dict[str, Any], mode: str) -> Tuple[int, str, List[str]]:
    P = POLICIES[mode]
    score = P["baseline"]

    notes: List[str] = []

    # Hard conflicts
    hc = int(flags.get("hard_conflict_total", 0))
    if hc:
        score += P["hard_conflict_each"]
        notes.append(f"{REASON_TEXT['infant_conflict']}") if flags.get("infant_conflict") else None
        notes.append(f"{REASON_TEXT['manykids_conflict']}") if flags.get("manykids_conflict") else None
        notes.append(f"{REASON_TEXT['cat_conflict']}") if flags.get("cat_conflict") else None
        notes.append(f"{REASON_TEXT['dog_conflict']}") if flags.get("dog_conflict") else None
        if hc > 1:
            score += (hc - 1) * P["hard_conflict_extra"]

    # Frictions
    if flags.get("private_room_missing"): score += P["private_room_missing"]; notes.append(REASON_TEXT["private_room_missing"])
    if flags.get("dayoff_mismatch"): score += P["dayoff_mismatch"]; notes.append(REASON_TEXT["dayoff_mismatch"])
    if flags.get("language_expected_missing"): score += P["language_missing"]; notes.append(REASON_TEXT["language_expected_missing"])
    if flags.get("avoids_abu_dhabi"): score += P["avoids_abu_dhabi"]; notes.append(REASON_TEXT["avoids_abu_dhabi"])

    # Need ∧ skill (block > skill already ensured by not adding when conflict True)
    if flags.get("infant_skill_needed") and not flags.get("infant_conflict"): score += P["infant_skill_needed"]; notes.append(REASON_TEXT["infant_skill_needed"])
    if flags.get("kids_over2_skill_needed") and not flags.get("manykids_conflict"): score += P["kids_over2_skill_needed"]; notes.append(REASON_TEXT["kids_over2_skill_needed"])
    if flags.get("handles_cats_needed") and not flags.get("cat_conflict"): score += P["pet_handling_needed"]; notes.append(REASON_TEXT["handles_cats_needed"])
    if flags.get("handles_dogs_needed") and not flags.get("dog_conflict"): score += P["pet_handling_needed"]; notes.append(REASON_TEXT["handles_dogs_needed"])

    # Extras (per spec)
    snt = max(0, int(flags.get("skill_needed_total", 0)) - 1)
    if P["skill_needed_total_per_extra"] and snt:
        score += min(P["skill_needed_total_cap"], snt * P["skill_needed_total_per_extra"]) or 0
    pmt = max(0, int(flags.get("pref_mismatch_total", 0)) - 1)
    if P["pref_mismatch_total_per_extra"] and pmt:
        adj = pmt * P["pref_mismatch_total_per_extra"]
        if P["pref_mismatch_total_per_extra"] < 0:
            score += max(P["pref_mismatch_total_cap"], adj)
        else:
            score += min(P["pref_mismatch_total_cap"], adj)

    # Experience contribution is part of score in Table 1, but we do NOT list it among MT reasons
    # If you have a column for years_of_experience, wire it here and replace 0.0
    score += exp_contrib(0.0, P)

    # Clamp & decision
    score = int(max(0, min(100, round(score))))
    if score >= P["ok_cut"]: decision = "OK"
    elif score >= P["review_low"]: decision = "Review"
    else: decision = "Blocked"

    # Keep only MT-derived reasons, filter Nones, preserve order
    mt_reasons = [r for r in notes if r]
    seen, out = set(), []
    for r in mt_reasons:
        if r not in seen:
            out.append(r); seen.add(r)
    return score, decision, out

# =============================
# Streamlit UI
# =============================

def main():
    st.set_page_config(page_title="Matching Score — MT Reasons", layout="wide")
    st.title("Matching Score (MT-based)")
    st.caption("Engineers RAW MTs → canonical → scores via policy. Shows decision + MT reasons only.")

    with st.sidebar:
        st.header("Policy mode")
        mode = st.radio("Mode", ["Balanced","Strict","Flexible"], index=0)
        st.markdown("""
        **Decision bands**  
        Strict: OK ≥ 75, Review 55–74, Block < 55  
        Balanced: OK ≥ 70, Review 50–69, Block < 50  
        Flexible: OK ≥ 65, Review 45–64, Block < 45
        """)

    tab_single, tab_batch = st.tabs(["Single case","Batch CSV"]) 

    with tab_single:
        st.subheader("Inputs (RAW ERP lists; engineered overrides optional)")
        c1,c2,c3 = st.columns(3)
        with c1: client_id = st.text_input("Client ID","C-123")
        with c2: maid_id = st.text_input("Maid ID","M-321")
        with c3: assignment_date = st.text_input("Assignment Date (YYYY-MM-DD)","")

        st.markdown("**RAW ERP lists**")
        client_mts_raw = st.text_area("client_mts_raw","baby + cat + private_room + abu_dhabi")
        maid_mts_raw = st.text_area("maid_mts_raw","avoids_abu_dhabi")

        st.markdown("**Engineered overrides (optional)**")
        colA,colB = st.columns(2)
        with colA:
            clientmts_household_type = st.selectbox("clientmts_household_type",["none","baby","many_kids","baby_and_kids"], index=1)
            clientmts_dayoff_policy = st.selectbox("clientmts_dayoff_policy",["none","flexible","work_for_pay","stay_home_for_pay","combos"], index=1)
            clientmts_pet_type = st.selectbox("clientmts_pet_type",["none","cat","dog","both"], index=1)
            clientmts_living_arrangement = st.selectbox("clientmts_living_arrangement",["unspecified","live_out","private_room","abu_dhabi","combos"], index=2)
            clientmts_cuisine_preference = st.selectbox("clientmts_cuisine_preference",["other","lebanese","khaleeji","international","combos"], index=1)
        with colB:
            maidmts_dayoff_policy = st.selectbox("maidmts_dayoff_policy",["unspecified","flexible"], index=1)
            maidmts_living_arrangement = st.selectbox("maidmts_living_arrangement",["unspecified","private_room","avoids_abu_dhabi","combo"], index=1)
            maidpref_kids_experience = st.selectbox("maidpref_kids_experience",["none","lessthan2","above2","both"], index=2)
            maidpref_pet_handling = st.selectbox("maidpref_pet_handling",["none","cats","dogs","both"], index=3)
            maid_nationality = st.selectbox("maid_nationality",["","filipina","indian","west_african","ethiopian"], index=1)

        if st.button("Compute Matching Score", type="primary"):
            overrides = {
                "clientmts_household_type": clientmts_household_type,
                "clientmts_dayoff_policy": clientmts_dayoff_policy,
                "clientmts_pet_type": clientmts_pet_type,
                "clientmts_living_arrangement": clientmts_living_arrangement,
                "clientmts_cuisine_preference": clientmts_cuisine_preference,
                "maidmts_dayoff_policy": maidmts_dayoff_policy,
                "maidmts_living_arrangement": maidmts_living_arrangement,
                "maidpref_kids_experience": maidpref_kids_experience,
                "maidpref_pet_handling": maidpref_pet_handling,
                "maid_nationality": maid_nationality,
            }
            cn = canonicalize_from_raw(client_mts_raw, maid_mts_raw, overrides)
            flags = make_flags(cn)
            score, decision, reasons = compute_score(flags, mode)

            k1,k2,k3 = st.columns(3)
            k1.metric("Match Score", score)
            k2.metric("Decision", decision)
            k3.metric("Policy", mode)

            st.markdown("### Reasons (MT-derived)")
            if reasons:
                for r in reasons:
                    st.write("• ", r)
            else:
                st.info("No MT conflicts or frictions detected.")

    with tab_batch:
        st.subheader("Batch scoring (CSV)")
        st.write("Upload a CSV. If engineered columns are present, they override RAW for 1:1 parity.")
        up = st.file_uploader("CSV columns (any subset): client_id, maid_id, assignment_date, client_mts_raw, maid_mts_raw, clientmts_*, maidmts_*, maidpref_*, maid_nationality", type=["csv"]) 
        if up is not None:
            df = pd.read_csv(up)
            rows = []
            for _, r in df.iterrows():
                overrides = {}
                for col in [
                    "clientmts_household_type","clientmts_special_cases","clientmts_pet_type","clientmts_dayoff_policy","clientmts_nationality_preference","clientmts_living_arrangement","clientmts_cuisine_preference",
                    "maidmts_household_type","maidmts_pet_type","maidmts_dayoff_policy","maidmts_living_arrangement",
                    "maidpref_kids_experience","maidpref_pet_handling","maid_nationality"
                ]:
                    if col in df.columns and pd.notna(r.get(col)):
                        overrides[col] = r.get(col)
                cn = canonicalize_from_raw(r.get("client_mts_raw",""), r.get("maid_mts_raw",""), overrides)
                flags = make_flags(cn)
                score, decision, reasons = compute_score(flags, mode)
                rows.append({
                    "client_id": r.get("client_id",""),
                    "maid_id": r.get("maid_id",""),
                    "match_score": score,
                    "decision": decision,
                    "policy_mode": mode,
                    "mt_reasons": "; ".join(reasons)
                })
            out = pd.DataFrame(rows)
            st.success(f"Scored {len(out)} rows.")
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button("Download matching_scores.csv", data=out.to_csv(index=False).encode("utf-8"), file_name="matching_scores.csv")

if __name__ == "__main__":
    main()
