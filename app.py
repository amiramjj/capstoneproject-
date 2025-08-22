# streamlit_app_matching_score.py
# Phase 1 — Matching Score ONLY (RAW → engineered → score)
# Update: Removes all "engineered overrides" UI. Everything is deduced from
# client_mts_raw and maid_mts_raw (or CSV with those two columns). Reasons shown
# are MT-derived only (conflicts, frictions, covered needs). Experience etc. are
# treated as 0 unless they can be inferred from RAW.

import re
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
        "experience_slope": 0.0,  # RAW-only flow: no exp contribution unless provided elsewhere
        "mobility_flex": +0,       # RAW-only flow: no mobility unless detectable in raw
        "soft_traits_cap": +0,     # RAW-only flow
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
        "experience_cap": +0,  # RAW-only flow
        "experience_slope": 0.0,
        "mobility_flex": +0,
        "soft_traits_cap": +0,
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
        "experience_cap": +0,  # RAW-only flow
        "experience_slope": 0.0,
        "mobility_flex": +0,
        "soft_traits_cap": +0,
        "avoids_abu_dhabi": -4,
        "skill_needed_total_per_extra": +2,
        "skill_needed_total_cap": +4,
        "pref_mismatch_total_per_extra": -1,
        "pref_mismatch_total_cap": -3,
        "ok_cut": 65,
        "review_low": 45,
    },
}

LANG_EXPECT_MAP = {
    # If maid nationality tokens appear in raw, we apply expectation; else no language rule.
    "ethiopian": {"ar"},
    "filipina": {"en"},
    "indian": {"en"},
    "west_african": {"en"},
}

CUISINE_MAP = {
    "saudi": "khaleeji", "emirati": "khaleeji", "kuwaiti": "khaleeji", "qatari": "khaleeji", "omani": "khaleeji", "bahraini": "khaleeji",
    "lebanese": "lebanese",
    "indian": "international", "chinese": "international", "italian": "international", "thai": "international"
}

SEP_RE = re.compile(r"[+,/;]| +")

def split_tokens(raw: Any) -> List[str]:
    if raw is None:
        return []
    toks = [t.strip().lower() for t in SEP_RE.split(str(raw)) if t and t.strip()]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

@dataclass
class Canonical:
    clientmts_household_type: str = "none"
    clientmts_special_cases: str = "none"
    clientmts_pet_type: str = "none"
    clientmts_dayoff_policy: str = "none"
    clientmts_nationality_preference: str = "any"
    clientmts_living_arrangement: str = "unspecified"
    clientmts_cuisine_preference: str = "other"

    maidmts_household_type: str = "none"
    maidmts_pet_type: str = "none"
    maidmts_dayoff_policy: str = "unspecified"
    maidmts_living_arrangement: str = "unspecified"

    maidpref_kids_experience: str = "none"
    maidpref_pet_handling: str = "none"
    maid_nationality: str = ""  # inferred only if mentioned in raw


def canonicalize_from_raw(client_raw: str, maid_raw: str) -> Canonical:
    cn = Canonical()
    ctoks, mtoks = split_tokens(client_raw), split_tokens(maid_raw)

    # --- Client MTs
    # household type
    if any(t in ctoks for t in ["baby", "infant"]): cn.clientmts_household_type = "baby"
    if any(t in ctoks for t in ["many_kids", "3kids", "3+kids"]): cn.clientmts_household_type = "many_kids"
    if ("baby" in ctoks or "infant" in ctoks) and any(t in ctoks for t in ["many_kids","3kids","3+kids"]): cn.clientmts_household_type = "baby_and_kids"
    # pets
    has_cat = any(t in ctoks for t in ["cat","cats"])
    has_dog = any(t in ctoks for t in ["dog","dogs"])
    if has_cat and has_dog: cn.clientmts_pet_type = "both"
    elif has_cat: cn.clientmts_pet_type = "cat"
    elif has_dog: cn.clientmts_pet_type = "dog"
    # dayoff
    if any(t in ctoks for t in ["flex","flexible"]): cn.clientmts_dayoff_policy = "flexible"
    elif any(t in ctoks for t in ["work_for_pay","stay_home_for_pay","sunday_only","combos"]): cn.clientmts_dayoff_policy = "combos"
    # living
    if "private_room" in ctoks: cn.clientmts_living_arrangement = "private_room"
    elif any(t in ctoks for t in ["live_out","liveout"]): cn.clientmts_living_arrangement = "live_out"
    if any(t in ctoks for t in ["abu_dhabi","abudhabi"]): cn.clientmts_living_arrangement = "abu_dhabi"
    # cuisine
    cuisines = [CUISINE_MAP.get(t, t) for t in ctoks if t in CUISINE_MAP]
    if cuisines: cn.clientmts_cuisine_preference = "combos" if len(set(cuisines))>1 else cuisines[0]
    # nationality preference (optional if present in raw)
    for nat in LANG_EXPECT_MAP.keys():
        if nat in ctoks: cn.clientmts_nationality_preference = nat

    # --- Maid MTs / prefs (deduced only from maid_raw)
    if any(t in mtoks for t in ["no_cats","pet_none"]): cn.maidmts_pet_type = "none"
    if "private_room" in mtoks: cn.maidmts_living_arrangement = "private_room"
    if "avoids_abu_dhabi" in mtoks: cn.maidmts_living_arrangement = "avoids_abu_dhabi"
    if any(t in mtoks for t in ["flex","flexible"]): cn.maidmts_dayoff_policy = "flexible"
    # kids experience (look for explicit tokens if present)
    if "lessthan2" in mtoks: cn.maidpref_kids_experience = "lessthan2"
    if "above2" in mtoks or "kids_experience_above2" in mtoks: cn.maidpref_kids_experience = "above2"
    if "both" in mtoks and cn.maidpref_kids_experience != "above2": cn.maidpref_kids_experience = "both"
    # pet handling
    if "cats" in mtoks and "dogs" in mtoks: cn.maidpref_pet_handling = "both"
    elif "cats" in mtoks: cn.maidpref_pet_handling = "cats"
    elif "dogs" in mtoks: cn.maidpref_pet_handling = "dogs"
    # nationality (only if named in raw; else blank → no language expectation applied)
    for nat in LANG_EXPECT_MAP.keys():
        if nat in mtoks: cn.maid_nationality = nat

    return cn

# =============================
# Flags & interactions (MT-derived only)
# =============================
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


def make_flags(cn: Canonical) -> Dict[str, Any]:
    f = {k: False for k in REASON_TEXT.keys()}
    # also keep totals for stacked extras
    f.update({"hard_conflict_total": 0, "pref_mismatch_total": 0, "skill_needed_total": 0})

    # Childcare
    if cn.clientmts_household_type in {"baby","baby_and_kids"}:
        if cn.maidpref_kids_experience in {"none","lessthan2"}:
            f["infant_conflict"] = True
        else:
            f["infant_skill_needed"] = True
    if cn.clientmts_household_type in {"many_kids","baby_and_kids"}:
        if cn.maidpref_kids_experience in {"none"}:
            f["manykids_conflict"] = True
        else:
            f["kids_over2_skill_needed"] = True

    # Pets
    if cn.clientmts_pet_type in {"cat","both"} and cn.maidpref_pet_handling not in {"cats","both"}:
        f["cat_conflict"] = True
    if cn.clientmts_pet_type in {"dog","both"} and cn.maidpref_pet_handling not in {"dogs","both"}:
        f["dog_conflict"] = True
    if cn.clientmts_pet_type in {"cat","both"} and cn.maidpref_pet_handling in {"cats","both"}:
        f["handles_cats_needed"] = True
    if cn.clientmts_pet_type in {"dog","both"} and cn.maidpref_pet_handling in {"dogs","both"}:
        f["handles_dogs_needed"] = True

    # Living arrangement / location
    if cn.maidmts_living_arrangement == "private_room" and cn.clientmts_living_arrangement != "private_room":
        f["private_room_missing"] = True
    if cn.maidmts_living_arrangement == "avoids_abu_dhabi" and cn.clientmts_living_arrangement == "abu_dhabi":
        f["avoids_abu_dhabi"] = True

    # Day-off
    if cn.clientmts_dayoff_policy in {"work_for_pay","stay_home_for_pay","combos"} and cn.maidmts_dayoff_policy != "flexible":
        f["dayoff_mismatch"] = True

    # Language expectation
    if cn.maid_nationality:
        expected = LANG_EXPECT_MAP.get(cn.maid_nationality, set())
        # RAW-only: assume missing unless explicitly satisfied in raw → we don't have speech fields, so treat as missing.
        if expected:
            f["language_expected_missing"] = True

    # Totals
    f["hard_conflict_total"] = sum(int(f[k]) for k in ["infant_conflict","manykids_conflict","cat_conflict","dog_conflict"])
    f["pref_mismatch_total"] = sum(int(f[k]) for k in ["private_room_missing","dayoff_mismatch","language_expected_missing","avoids_abu_dhabi"])
    f["skill_needed_total"] = sum(int(f[k]) for k in ["infant_skill_needed","kids_over2_skill_needed","handles_cats_needed","handles_dogs_needed"])

    return f


def compute_score(flags: Dict[str, Any], mode: str) -> Tuple[int, str, List[str]]:
    P = POLICIES[mode]
    score = P["baseline"]
    notes: List[str] = []

    # Hard conflicts
    hc = int(flags.get("hard_conflict_total", 0))
    if hc:
        score += P["hard_conflict_each"]
        for k in ["infant_conflict","manykids_conflict","cat_conflict","dog_conflict"]:
            if flags.get(k): notes.append(REASON_TEXT[k])
        if hc > 1:
            score += (hc - 1) * P["hard_conflict_extra"]

    # Frictions
    for k, w in [("private_room_missing", P["private_room_missing"]), ("dayoff_mismatch", P["dayoff_mismatch"]), ("language_expected_missing", P["language_missing"]), ("avoids_abu_dhabi", P["avoids_abu_dhabi"])]:
        if flags.get(k): score += w; notes.append(REASON_TEXT[k])

    # Need ∧ skill (only when corresponding conflict not present)
    if flags.get("infant_skill_needed") and not flags.get("infant_conflict"): score += P["infant_skill_needed"]; notes.append(REASON_TEXT["infant_skill_needed"])
    if flags.get("kids_over2_skill_needed") and not flags.get("manykids_conflict"): score += P["kids_over2_skill_needed"]; notes.append(REASON_TEXT["kids_over2_skill_needed"])
    if flags.get("handles_cats_needed") and not flags.get("cat_conflict"): score += P["pet_handling_needed"]; notes.append(REASON_TEXT["handles_cats_needed"])
    if flags.get("handles_dogs_needed") and not flags.get("dog_conflict"): score += P["pet_handling_needed"]; notes.append(REASON_TEXT["handles_dogs_needed"])

    # Extras (stacked totals)
    snt = max(0, int(flags.get("skill_needed_total", 0)) - 1)
    if P["skill_needed_total_per_extra"] and snt:
        score += min(P["skill_needed_total_cap"], snt * P["skill_needed_total_per_extra"]) or 0
    pmt = max(0, int(flags.get("pref_mismatch_total", 0)) - 1)
    if P["pref_mismatch_total_per_extra"] and pmt:
        adj = pmt * P["pref_mismatch_total_per_extra"]
        score += max(P["pref_mismatch_total_cap"], adj) if adj < 0 else min(P["pref_mismatch_total_cap"], adj)

    score = int(max(0, min(100, round(score))))
    decision = "OK" if score >= P["ok_cut"] else ("Review" if score >= P["review_low"] else "Blocked")

    # De-dup reasons
    seen, out = set(), []
    for r in notes:
        if r not in seen:
            out.append(r); seen.add(r)
    return score, decision, out

# =============================
# Streamlit UI
# =============================

def main():
    st.set_page_config(page_title="Matching Score — RAW only", layout="wide")
    st.title("Matching Score (RAW MT strings → engineered → score)")
    st.caption("Paste client & maid MT strings (or upload a CSV). No manual overrides; engineering is automatic.")

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
        st.subheader("Inputs (RAW ERP lists only)")
        c1,c2,c3 = st.columns(3)
        with c1: client_id = st.text_input("Client ID","C-123")
        with c2: maid_id = st.text_input("Maid ID","M-321")
        with c3: assignment_date = st.text_input("Assignment Date (optional)", "")

        client_mts_raw = st.text_area("client_mts_raw", placeholder="e.g., baby + cat + private_room + abu_dhabi")
        maid_mts_raw = st.text_area("maid_mts_raw", placeholder="e.g., avoids_abu_dhabi + flexible + cats + above2")

        if st.button("Compute Matching Score", type="primary"):
            cn = canonicalize_from_raw(client_mts_raw, maid_mts_raw)
            flags = make_flags(cn)
            score, decision, reasons = compute_score(flags, mode)

            k1,k2,k3 = st.columns(3)
            k1.metric("Match Score", score)
            k2.metric("Decision", decision)
            k3.metric("Policy", mode)

            with st.expander("Preview engineered MTs (read-only)", expanded=False):
                st.json({k:getattr(cn,k) for k in cn.__dataclass_fields__.keys()})

            st.markdown("### Reasons (MT-derived)")
            if reasons:
                for r in reasons: st.write("• ", r)
            else:
                st.info("No MT conflicts or frictions detected.")

    with tab_batch:
        st.subheader("Batch scoring (CSV)")
        st.write("Upload a CSV with these columns: **client_id, maid_id, client_mts_raw, maid_mts_raw** (assignment_date optional). The app will engineer and score each row.")
        up = st.file_uploader("Upload CSV", type=["csv"]) 
        if up is not None:
            df = pd.read_csv(up)
            required = {"client_id","maid_id","client_mts_raw","maid_mts_raw"}
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                rows = []
                for _, r in df.iterrows():
                    cn = canonicalize_from_raw(r.get("client_mts_raw",""), r.get("maid_mts_raw",""))
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
