# streamlit_app_matching_score.py
# Phase 1: Matching Score only (Models deferred)
# Mirrors the policy and weights described by the user (Strict / Balanced / Flexible)
# Notes:
# - Accepts either RAW ERP lists or ENGINEERED columns.
# - If engineered flags are present, the scorer uses them directly (parity-friendly).
# - If only raw lists are present, we apply a conservative best-effort engineering pass.
# - Sub-score chips aggregate rule contributions into 9 intuitive buckets for UI clarity.

import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st

# =========================
# CONFIG — Policy Weights
# =========================
# Table 1 (user-provided). Baseline = 70 across modes (ensures single hard-conflict -> Block).
POLICIES = {
    "Strict": {
        "baseline": 70,
        "hard_conflict_each": -40,
        "hard_conflict_extra": -10,  # per extra conflict beyond the first
        "private_room_missing": -15,
        "dayoff_mismatch": -12,
        "language_missing": -10,
        "infant_skill_needed": +8,
        "kids_over2_skill_needed": +6,
        "pet_handling_needed": +4,  # each
        "experience_cap": +10,
        "experience_slope": 2.0,  # proposed (see comments)
        "mobility_flex": +4,
        "soft_traits_cap": +4,
        "avoids_abu_dhabi": -10,
        "skill_needed_total_per_extra": 0,  # ≤ 0
        "skill_needed_total_cap": 0,
        "pref_mismatch_total_per_extra": -3,
        "pref_mismatch_total_cap": -6,
        "ok_cut": 75,
        "review_low": 55,
    },
    "Balanced": {
        "baseline": 70,
        "hard_conflict_each": -30,
        "hard_conflict_extra": -5,   # per extra conflict beyond the first
        "private_room_missing": -12,
        "dayoff_mismatch": -10,
        "language_missing": -8,
        "infant_skill_needed": +10,
        "kids_over2_skill_needed": +8,
        "pet_handling_needed": +6,   # each
        "experience_cap": +12,
        "experience_slope": 2.0,    # proposed; reaches cap ~6y
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
        "hard_conflict_extra": 0,    # none
        "private_room_missing": -8,
        "dayoff_mismatch": -6,
        "language_missing": -5,
        "infant_skill_needed": +12,
        "kids_over2_skill_needed": +10,
        "pet_handling_needed": +8,   # each
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

LANG_EXPECT_MAP = {
    # From spec: Ethiopian -> Arabic; Filipina/Indian/West African -> English; else none
    "ethiopian": {"ar"},
    "filipina": {"en"},
    "indian": {"en"},
    "west_african": {"en"},
}

# =========================
# Helpers — Normalization
# =========================
SEP_RE = re.compile(r"[+,/;]|\s+")

def split_tokens(raw: str) -> List[str]:
    if raw is None:
        return []
    toks = [t.strip().lower() for t in SEP_RE.split(str(raw)) if t and t.strip()]
    # dedupe preserving order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# =========================
# Engineering (best-effort when RAW provided)
# =========================
@dataclass
class PairInputs:
    client_id: str
    maid_id: str
    assignment_date: str | None
    client_mts_raw: List[str]
    maid_mts_raw: List[str]
    years_of_experience: float | int | None = None
    maidpref_personality: List[str] | None = None
    maidpref_travel: str | None = None
    maidpref_pet_handling: str | None = None
    maidpref_kids_experience: str | None = None
    clientmts_living_arrangement: str | None = None
    maidmts_living_arrangement: str | None = None
    clientmts_dayoff_policy: str | None = None
    maidmts_dayoff_policy: str | None = None
    clientmts_pet_type: str | None = None
    clientmts_household_type: str | None = None
    clientmts_nationality_preference: str | None = None
    maid_nationality: str | None = None


def infer_flags_from_raw(pi: PairInputs) -> Dict[str, Any]:
    """Conservative inference from raw tokens and a few explicit fields.
    If engineered fields/flags exist in the uploaded CSV, the scorer should prefer them.
    """
    # Base flags
    flags = {
        # Hard conflicts
        "infant_conflict": False,
        "manykids_conflict": False,
        "cat_conflict": False,
        "dog_conflict": False,
        # Frictions
        "private_room_missing": False,
        "dayoff_mismatch": False,
        "language_expected_missing": False,
        # Skills needed
        "infant_skill_needed": False,
        "kids_over2_skill_needed": False,
        "handles_cats_needed": False,
        "handles_dogs_needed": False,
        # Stabilizers / meta
        "hard_conflict_total": 0,
        "pref_mismatch_total": 0,
        "skill_needed_total": 0,
        "pet_need_total": 0,
        # Experience & extras
        "years_of_experience": float(pi.years_of_experience) if pi.years_of_experience not in (None, "") else 0.0,
        "mobility_flex": False,
        "soft_traits_count": 0,
        "avoids_abu_dhabi": False,
    }

    # --- Personality / soft traits
    if pi.maidpref_personality:
        soft = [t for t in pi.maidpref_personality if t not in {"not_mentioned", "unspecified", "combos"}]
        flags["soft_traits_count"] = min(len(soft), 99)  # will cap later per policy

    # --- Mobility
    if pi.maidpref_travel and pi.maidpref_travel in {"travel", "relocate", "travel_and_relocate"}:
        flags["mobility_flex"] = True

    # --- Pets (need & conflict)
    client_pet = (pi.clientmts_pet_type or "none").lower()
    maid_pet_pref = (pi.maidpref_pet_handling or "none").lower()
    # skills if need exists and maid has explicit handling
    if client_pet in {"cat", "both"} and maid_pet_pref in {"cats", "both"}:
        flags["handles_cats_needed"] = True
        flags["pet_need_total"] += 1
    if client_pet in {"dog", "both"} and maid_pet_pref in {"dogs", "both"}:
        flags["handles_dogs_needed"] = True
        flags["pet_need_total"] += 1

    # conservative conflict inference: if client has pet but maidmts_pet_type == "none"
    maidmts_pet = (pi.maid_mts_raw or [])
    maidmts_pet = set(maidmts_pet)
    if client_pet in {"cat", "both"} and ("pet_none" in maidmts_pet or "no_cats" in maidmts_pet):
        flags["cat_conflict"] = True
    if client_pet in {"dog", "both"} and ("pet_none" in maidmts_pet or "no_dogs" in maidmts_pet):
        flags["dog_conflict"] = True

    # --- Household / childcare
    client_house = (pi.clientmts_household_type or "none").lower()
    maid_kids = (pi.maidpref_kids_experience or "none").lower()
    if client_house in {"baby", "baby_and_kids"}:
        if maid_kids in {"none", "lessthan2"}:  # <2y experience treated as insufficient for infant
            flags["infant_conflict"] = True
        else:
            flags["infant_skill_needed"] = True
    if client_house in {"many_kids", "baby_and_kids"}:
        if maid_kids in {"none"}:
            flags["manykids_conflict"] = True
        else:
            flags["kids_over2_skill_needed"] = True

    # --- Living arrangement
    cl_living = (pi.clientmts_living_arrangement or "unspecified").lower()
    md_living = (pi.maidmts_living_arrangement or "unspecified").lower()
    if "abu_dhabi" in cl_living and ("avoids_abu_dhabi" in pi.maid_mts_raw or md_living == "avoids_abu_dhabi"):
        flags["avoids_abu_dhabi"] = True
    # private room expectation: maid requires private_room but client not offering it explicitly
    if md_living == "private_room" and cl_living not in {"private_room"}:
        flags["private_room_missing"] = True

    # --- Day-off mismatch (coarse policy)
    cl_dayoff = (pi.clientmts_dayoff_policy or "none").lower()
    md_dayoff = (pi.maidmts_dayoff_policy or "unspecified").lower()
    if cl_dayoff in {"work_for_pay", "stay_home_for_pay", "combos"} and md_dayoff == "unspecified":
        flags["dayoff_mismatch"] = True

    # --- Language expectation by nationality
    nat = (pi.clientmts_nationality_preference or "any").lower()
    maid_nat = (pi.maid_nationality or "").lower()
    expected = LANG_EXPECT_MAP.get(maid_nat, set())
    if expected:
        # For simplicity, assume maid speaks EN if maid_nat expects EN, otherwise missing.
        # If you maintain real language columns, replace this check accordingly.
        speaks_en = True if maid_nat in {"filipina", "indian", "west_african"} else False
        speaks_ar = True if maid_nat in {"ethiopian"} else False
        if ("en" in expected and not speaks_en) or ("ar" in expected and not speaks_ar):
            flags["language_expected_missing"] = True

    # Totals
    hard_list = [flags["infant_conflict"], flags["manykids_conflict"], flags["cat_conflict"], flags["dog_conflict"]]
    flags["hard_conflict_total"] = sum(1 for x in hard_list if x)
    pref_list = [flags["private_room_missing"], flags["dayoff_mismatch"], flags["language_expected_missing"], flags["avoids_abu_dhabi"]]
    flags["pref_mismatch_total"] = sum(1 for x in pref_list if x)
    skill_list = [flags["infant_skill_needed"], flags["kids_over2_skill_needed"], flags["handles_cats_needed"], flags["handles_dogs_needed"]]
    flags["skill_needed_total"] = sum(1 for x in skill_list if x)

    return flags

# =========================
# Scoring
# =========================
SUB_BUCKETS = [
    "childcare_household", "pets", "living_arrangement", "dayoff", "language",
    "experience", "mobility", "soft_traits", "preference_alignment"
]


def eval_experience_contrib(years: float, policy: Dict[str, Any]) -> float:
    slope = policy["experience_slope"]
    cap = policy["experience_cap"]
    years = max(0.0, float(years or 0.0))
    return min(cap, slope * years)


def compute_score(flags: Dict[str, Any], mode: str = "Balanced") -> Tuple[int, str, Dict[str, int], List[str]]:
    P = POLICIES[mode]
    score = P["baseline"]
    notes: List[str] = []
    contrib = {k: 0.0 for k in SUB_BUCKETS}

    # --- Hard conflicts
    hc = int(flags.get("hard_conflict_total", 0))
    if hc:
        score += P["hard_conflict_each"]  # first
        contrib["childcare_household"] += P["hard_conflict_each"]  # attributed to childcare by default
        notes.append(f"Hard conflict (x1) {P['hard_conflict_each']}")
        if hc > 1:
            extra = (hc - 1) * P["hard_conflict_extra"]
            score += extra
            contrib["childcare_household"] += extra
            notes.append(f"Extra conflicts (x{hc-1}) {P['hard_conflict_extra']}")

    # --- Frictions
    if flags.get("private_room_missing"):
        score += P["private_room_missing"]
        contrib["living_arrangement"] += P["private_room_missing"]
        notes.append(f"Private room missing {P['private_room_missing']}")

    if flags.get("dayoff_mismatch"):
        score += P["dayoff_mismatch"]
        contrib["dayoff"] += P["dayoff_mismatch"]
        notes.append(f"Day-off mismatch {P['dayoff_mismatch']}")

    if flags.get("language_expected_missing"):
        score += P["language_missing"]
        contrib["language"] += P["language_missing"]
        notes.append(f"Language expected but missing {P['language_missing']}")

    if flags.get("avoids_abu_dhabi"):
        score += P["avoids_abu_dhabi"]
        contrib["living_arrangement"] += P["avoids_abu_dhabi"]
        notes.append(f"Avoids Abu Dhabi {P['avoids_abu_dhabi']}")

    # --- Skills (need ∧ skill), with block > skill safeguard
    block_infant = bool(flags.get("infant_conflict"))
    block_manykids = bool(flags.get("manykids_conflict"))
    block_cat = bool(flags.get("cat_conflict"))
    block_dog = bool(flags.get("dog_conflict"))

    if flags.get("infant_skill_needed") and not block_infant:
        score += P["infant_skill_needed"]
        contrib["childcare_household"] += P["infant_skill_needed"]
        notes.append(f"Infant skill needed +{P['infant_skill_needed']}")

    if flags.get("kids_over2_skill_needed") and not block_manykids:
        score += P["kids_over2_skill_needed"]
        contrib["childcare_household"] += P["kids_over2_skill_needed"]
        notes.append(f"Kids>2 skill needed +{P['kids_over2_skill_needed']}")

    if flags.get("handles_cats_needed") and not block_cat:
        score += P["pet_handling_needed"]
        contrib["pets"] += P["pet_handling_needed"]
        notes.append(f"Handles cats needed +{P['pet_handling_needed']}")

    if flags.get("handles_dogs_needed") and not block_dog:
        score += P["pet_handling_needed"]
        contrib["pets"] += P["pet_handling_needed"]
        notes.append(f"Handles dogs needed +{P['pet_handling_needed']}")

    # --- Per-extra adjustments
    # skill_needed_total extras
    snt = max(0, int(flags.get("skill_needed_total", 0)) - 1)
    if P["skill_needed_total_per_extra"] != 0 and snt > 0:
        add = min(P["skill_needed_total_cap"], snt * P["skill_needed_total_per_extra"]) if P["skill_needed_total_cap"] else snt * P["skill_needed_total_per_extra"]
        score += add
        contrib["childcare_household"] += add  # attribute to capability bucket
        notes.append(f"Skill-needed extras +{add}")

    # pref_mismatch_total extras
    pmt = max(0, int(flags.get("pref_mismatch_total", 0)) - 1)
    if P["pref_mismatch_total_per_extra"] != 0 and pmt > 0:
        sub = max(P["pref_mismatch_total_cap"], pmt * P["pref_mismatch_total_per_extra"]) if P["pref_mismatch_total_per_extra"] < 0 else min(P["pref_mismatch_total_cap"], pmt * P["pref_mismatch_total_per_extra"])  # cap direction
        score += sub
        contrib["living_arrangement"] += sub  # attribute to frictions bucket
        notes.append(f"Preference-mismatch extras {sub}")

    # --- Experience (slope to cap)
    exp_contrib = eval_experience_contrib(flags.get("years_of_experience", 0.0), P)
    if exp_contrib:
        score += exp_contrib
        contrib["experience"] += exp_contrib
        notes.append(f"Experience +{int(exp_contrib)} (capped)")

    # --- Mobility
    if flags.get("mobility_flex"):
        score += P["mobility_flex"]
        contrib["mobility"] += P["mobility_flex"]
        notes.append(f"Mobility flexible +{P['mobility_flex']}")

    # --- Soft traits (capped)
    stc = int(flags.get("soft_traits_count", 0))
    if stc:
        cap = P["soft_traits_cap"]
        # assume +1 per trait up to cap
        add = min(cap, stc)
        score += add
        contrib["soft_traits"] += add
        notes.append(f"Soft traits +{add} (capped)")

    # --- Clamp and round
    score = max(0, min(100, int(round(score))))

    # --- Decision band
    if score >= P["ok_cut"]:
        decision = "OK"
    elif score >= P["review_low"]:
        decision = "Review"
    else:
        decision = "Blocked"

    # --- Convert contrib floats to ints for UI cleanliness
    contrib_int = {k: int(round(v)) for k, v in contrib.items()}

    return score, decision, contrib_int, notes

# =========================
# Streamlit UI — Matching Score Section
# =========================

def main():
    st.set_page_config(page_title="Matching Score — Phase 1", layout="wide")
    st.title("Matching Score (Phase 1)")
    st.caption("Policy-faithful, human-readable score. Models come next.")

    with st.sidebar:
        st.header("Policy")
        mode = st.radio("Mode", ["Balanced", "Strict", "Flexible"], index=0)
        st.markdown(
            """
            **Bands (by mode)**  
            Balanced: OK ≥ 70, Review 50–69, Block < 50  
            Strict: OK ≥ 75, Review 55–74, Block < 55  
            Flexible: OK ≥ 65, Review 45–64, Block < 45
            """
        )

        st.divider()
        st.subheader("Templates")
        st.markdown("Download CSV templates from the chat message.")

    # --- Tabs: Single vs Batch
    tab_single, tab_batch = st.tabs(["Single case", "Batch CSV"])

    with tab_single:
        st.subheader("Inputs (RAW or Engineered)")
        c1, c2, c3 = st.columns(3)
        with c1:
            client_id = st.text_input("Client ID", value="C-123")
        with c2:
            maid_id = st.text_input("Maid ID", value="M-321")
        with c3:
            assignment_date = st.text_input("Assignment Date (YYYY-MM-DD)", value="")

        st.markdown("**RAW ERP lists (optional if you provide engineered fields below)**")
        client_mts_raw = st.text_area("client_mts_raw", value="baby + cat + private_room + abu_dhabi")
        maid_mts_raw = st.text_area("maid_mts_raw", value="avoids_abu_dhabi")
        years_of_experience = st.number_input("years_of_experience (maid)", min_value=0.0, value=3.0, step=0.5)
        maidpref_personality = st.text_input("maidpref_personality (comma/+/;/space separated)", value="energetic no_attitude")
        maidpref_travel = st.selectbox("maidpref_travel", ["unspecified", "no", "travel", "relocate", "travel_and_relocate"], index=2)
        maidpref_pet_handling = st.selectbox("maidpref_pet_handling", ["none", "cats", "dogs", "both"], index=3)
        maidpref_kids_experience = st.selectbox("maidpref_kids_experience", ["none", "lessthan2", "above2", "both"], index=2)

        st.markdown("**Key engineered categorical fields (optional overrides for better parity)**")
        c4, c5 = st.columns(2)
        with c4:
            clientmts_living_arrangement = st.selectbox("clientmts_living_arrangement", ["unspecified", "live_out", "private_room", "abu_dhabi", "combos"], index=2)
            clientmts_dayoff_policy = st.selectbox("clientmts_dayoff_policy", ["none", "flexible", "work_for_pay", "stay_home_for_pay", "combos"], index=1)
            clientmts_pet_type = st.selectbox("clientmts_pet_type", ["none", "cat", "dog", "both"], index=1)
            clientmts_household_type = st.selectbox("clientmts_household_type", ["none", "baby", "many_kids", "baby_and_kids"], index=1)
            clientmts_nationality_preference = st.selectbox("clientmts_nationality_preference", ["any", "filipina", "west_african", "ethiopian"], index=0)
        with c5:
            maidmts_living_arrangement = st.selectbox("maidmts_living_arrangement", ["unspecified", "private_room", "avoids_abu_dhabi", "combo"], index=1)
            maidmts_dayoff_policy = st.selectbox("maidmts_dayoff_policy", ["unspecified", "flexible"], index=1)
            maid_nationality = st.selectbox("maid_nationality (for language rule)", ["", "filipina", "indian", "west_african", "ethiopian"], index=1)

        if st.button("Compute Matching Score", type="primary"):
            pi = PairInputs(
                client_id=client_id,
                maid_id=maid_id,
                assignment_date=assignment_date or None,
                client_mts_raw=split_tokens(client_mts_raw),
                maid_mts_raw=split_tokens(maid_mts_raw),
                years_of_experience=years_of_experience,
                maidpref_personality=split_tokens(maidpref_personality),
                maidpref_travel=maidpref_travel,
                maidpref_pet_handling=maidpref_pet_handling,
                maidpref_kids_experience=maidpref_kids_experience,
                clientmts_living_arrangement=clientmts_living_arrangement,
                maidmts_living_arrangement=maidmts_living_arrangement,
                clientmts_dayoff_policy=clientmts_dayoff_policy,
                maidmts_dayoff_policy=maidmts_dayoff_policy,
                clientmts_pet_type=clientmts_pet_type,
                clientmts_household_type=clientmts_household_type,
                clientmts_nationality_preference=clientmts_nationality_preference,
                maid_nationality=maid_nationality,
            )
            flags = infer_flags_from_raw(pi)
            score, decision, contrib, notes = compute_score(flags, mode)

            # KPIs
            kc1, kc2, kc3 = st.columns([1,1,1])
            kc1.metric("Match Score", score)
            kc2.metric("Decision", decision)
            kc3.metric("Mode", mode)

            st.markdown("### Sub-score breakdown")
            sub_df = pd.DataFrame({"dimension": list(contrib.keys()), "contribution": list(contrib.values())})
            st.dataframe(sub_df, hide_index=True, use_container_width=True)

            st.markdown("### Reason codes")
            for n in notes:
                st.write("• ", n)

    with tab_batch:
        st.subheader("Batch scoring")
        st.write("Upload CSV. If engineered columns are present, they will override raw inference for parity.")
        file = st.file_uploader("CSV with columns (any subset): client_id, maid_id, assignment_date, client_mts_raw, maid_mts_raw, years_of_experience, maidpref_personality, maidpref_travel, maidpref_pet_handling, maidpref_kids_experience, clientmts_*, maidmts_*, maid_nationality", type=["csv"]) 
        if file is not None:
            df = pd.read_csv(file)
            out_rows = []
            for i, row in df.iterrows():
                pi = PairInputs(
                    client_id=str(row.get("client_id", "")),
                    maid_id=str(row.get("maid_id", "")),
                    assignment_date=str(row.get("assignment_date", "")) if not pd.isna(row.get("assignment_date", "")) else None,
                    client_mts_raw=split_tokens(row.get("client_mts_raw", "")),
                    maid_mts_raw=split_tokens(row.get("maid_mts_raw", "")),
                    years_of_experience=row.get("years_of_experience", 0.0),
                    maidpref_personality=split_tokens(row.get("maidpref_personality", "")),
                    maidpref_travel=str(row.get("maidpref_travel", "unspecified")),
                    maidpref_pet_handling=str(row.get("maidpref_pet_handling", "none")),
                    maidpref_kids_experience=str(row.get("maidpref_kids_experience", "none")),
                    clientmts_living_arrangement=str(row.get("clientmts_living_arrangement", "unspecified")),
                    maidmts_living_arrangement=str(row.get("maidmts_living_arrangement", "unspecified")),
                    clientmts_dayoff_policy=str(row.get("clientmts_dayoff_policy", "none")),
                    maidmts_dayoff_policy=str(row.get("maidmts_dayoff_policy", "unspecified")),
                    clientmts_pet_type=str(row.get("clientmts_pet_type", "none")),
                    clientmts_household_type=str(row.get("clientmts_household_type", "none")),
                    clientmts_nationality_preference=str(row.get("clientmts_nationality_preference", "any")),
                    maid_nationality=str(row.get("maid_nationality", "")),
                )
                flags = infer_flags_from_raw(pi)
                score, decision, contrib, notes = compute_score(flags, mode)
                out = {
                    "client_id": pi.client_id,
                    "maid_id": pi.maid_id,
                    "match_score": score,
                    "decision": decision,
                    "mode": mode,
                }
                # flatten a few sub-scores
                for k, v in contrib.items():
                    out[f"sub_{k}"] = v
                # preserve counts for transparency
                out["hard_conflict_total"] = flags.get("hard_conflict_total", 0)
                out["pref_mismatch_total"] = flags.get("pref_mismatch_total", 0)
                out["skill_needed_total"] = flags.get("skill_needed_total", 0)
                out_rows.append(out)

            out_df = pd.DataFrame(out_rows)
            st.success(f"Scored {len(out_df)} rows.")
            st.dataframe(out_df.head(20), use_container_width=True)

            st.download_button("Download scores CSV", data=out_df.to_csv(index=False).encode("utf-8"), file_name="matching_scores.csv")


if __name__ == "__main__":
    main()
