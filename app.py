# streamlit_app_matching_score.py
# RAW strings → cleaning/engineering → flags → score (exact-policy mirror)
# Update: robust MT parsing that handles acceptance vs. blocks on the MAID side,
# wide synonym coverage, and cross-field safeguards so common ERP phrasings
# (e.g., "has a baby", "has a dog" inside maid MTs) are treated correctly.

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

# =============================
# Dictionaries & normalizers
# =============================
LANG_EXPECT_MAP = {
    "ethiopian": {"ar"},
    "filipina": {"en"},
    "indian": {"en"},
    "west_african": {"en"},
}

CUISINE_MAP = {
    # fine → coarse
    "saudi": "khaleeji", "emirati": "khaleeji", "kuwaiti": "khaleeji", "qatari": "khaleeji", "omani": "khaleeji", "bahraini": "khaleeji",
    "lebanese": "lebanese",
    "indian": "international", "chinese": "international", "italian": "international", "thai": "international",
}
COARSE_CUISINES = {"khaleeji", "lebanese", "international"}

SEP_RE = re.compile(r"[+,/;]|\s+")

ALIAS = {
    # living & location
    "private room": "private_room",
    "has a private room": "private_room",
    "live-out": "live_out", "liveout": "live_out",
    "abu dhabi": "abu_dhabi", "abudhabi": "abu_dhabi",
    "avoid abu dhabi": "avoids_abu_dhabi", "avoid abudhabi": "avoids_abu_dhabi",
    # day-off
    "work on her day off for pay": "work_for_pay",
    "work on his day off for pay": "work_for_pay",
    "stay home for pay": "stay_home_for_pay",
    "sunday only": "sunday_only",
    # household (client) / acceptance (maid)
    "has a baby younger than 2 years old": "baby",
    "has a baby": "baby",
    "infant": "baby",
    "3 kids or more": "many_kids", "3+ kids": "many_kids", "3 or more kids": "many_kids",
    "many kids": "many_kids",
    # pets acceptance
    "has a dog": "dog",
    "has a cat": "cat",
    # pet blocks
    "no cats": "no_cats", "no dogs": "no_dogs", "no pets": "no_pets",
    # spellings
    "khaleej": "khaleeji", "khaliji": "khaleeji",
    # soft traits/CP hints
    "no attitude": "no_attitude", "veg friendly": "veg_friendly", "non smoker": "non_smoker",
}

LANG_ALIASES = {"english": "en", "arabic": "ar", "en": "en", "ar": "ar"}


def norm_text(s: Any) -> str:
    s = ("" if s is None else str(s)).strip().lower()
    # phrase-level replacements
    for k, v in ALIAS.items():
        if k in s:
            s = s.replace(k, v)
    return s


def split_tokens(raw: Any) -> List[str]:
    s = norm_text(raw)
    toks = [t.strip() for t in SEP_RE.split(s) if t and t.strip()]
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

# =============================
# Canonical containers
# =============================
@dataclass
class Canonical:
    # Client MTs
    clientmts_household_type: str = "none"
    clientmts_special_cases: str = "none"
    clientmts_pet_type: str = "none"
    clientmts_dayoff_policy: str = "none"
    clientmts_nationality_preference: str = "any"
    clientmts_living_arrangement: str = "unspecified"
    clientmts_cuisine_preference: str = "other"

    # Maid MTs (acceptance/constraints)
    maidmts_household_type: str = "none"          # {none,baby,many_kids,baby_and_kids}
    maidmts_pet_type: str = "none"                 # acceptance summary: {none,cat,dog,both}
    maidmts_dayoff_policy: str = "unspecified"     # {flexible,unspecified}
    maidmts_living_arrangement: str = "unspecified"# {unspecified,private_room,avoids_abu_dhabi,combo}

    # Maid pet blocks (explicit)
    maid_block_cats: bool = False
    maid_block_dogs: bool = False
    maid_accept_cats: bool = False
    maid_accept_dogs: bool = False

    # Maid CPs / profile (used for score only; not shown in reason list)
    maidpref_education: str = "not_specified"
    maidpref_kids_experience: str = "none"
    maidpref_pet_handling: str = "none"
    maidpref_personality: List[str] = None
    maidpref_travel: str = "unspecified"
    maidpref_smoking: str = "unspecified"
    maidpref_caregiving_profile: str = "none"

    # Signals
    maid_nationality: str = ""
    maid_languages: List[str] = None
    years_of_experience: float = 0.0

# =============================
# Engineering — Client MTs
# =============================

def parse_client_mts(raw: str, cuisine_raw: str | None) -> Dict[str, str]:
    ctoks = split_tokens(raw)
    out = {
        "clientmts_household_type": "none",
        "clientmts_special_cases": "none",
        "clientmts_pet_type": "none",
        "clientmts_dayoff_policy": "none",
        "clientmts_nationality_preference": "any",
        "clientmts_living_arrangement": "unspecified",
        "clientmts_cuisine_preference": "other",
    }
    # household type
    has_baby = any(t in ctoks for t in ["baby"])
    has_manykids = any(t in ctoks for t in ["many_kids"]) or bool(re.search(r"\b(>=?\s*3|3\+|three or more)\b", " ".join(ctoks)))
    if has_baby and has_manykids:
        out["clientmts_household_type"] = "baby_and_kids"
    elif has_baby:
        out["clientmts_household_type"] = "baby"
    elif has_manykids:
        out["clientmts_household_type"] = "many_kids"

    # special cases
    if "elderly" in ctoks and "special_needs" in ctoks:
        out["clientmts_special_cases"] = "elderly_and_special"
    elif "elderly" in ctoks:
        out["clientmts_special_cases"] = "elderly"
    elif "special_needs" in ctoks:
        out["clientmts_special_cases"] = "special_needs"

    # pets in household
    has_cat = any(t in ctoks for t in ["cat","cats"]) 
    has_dog = any(t in ctoks for t in ["dog","dogs"]) 
    if has_cat and has_dog: out["clientmts_pet_type"] = "both"
    elif has_cat: out["clientmts_pet_type"] = "cat"
    elif has_dog: out["clientmts_pet_type"] = "dog"

    # day-off policy
    if any(t in ctoks for t in ["flex","flexible"]):
        out["clientmts_dayoff_policy"] = "flexible"
    elif any(t in ctoks for t in ["work_for_pay","stay_home_for_pay","sunday_only","combos"]):
        out["clientmts_dayoff_policy"] = "combos"

    # living arrangement & location
    if "private_room" in ctoks: out["clientmts_living_arrangement"] = "private_room"
    elif "live_out" in ctoks: out["clientmts_living_arrangement"] = "live_out"
    if "abu_dhabi" in ctoks: out["clientmts_living_arrangement"] = "abu_dhabi"

    # nationality preference (if stated)
    for nat in ["filipina","ethiopian","west_african","indian"]:
        if nat in ctoks: out["clientmts_nationality_preference"] = nat

    # cuisine — accept both coarse and fine
    coarse = [t for t in ctoks if t in COARSE_CUISINES]
    fine = [CUISINE_MAP.get(t, t) for t in ctoks if t in CUISINE_MAP]
    if cuisine_raw:
        fine += [CUISINE_MAP.get(t, t) for t in split_tokens(cuisine_raw) if t in CUISINE_MAP]
        coarse += [t for t in split_tokens(cuisine_raw) if t in COARSE_CUISINES]
    cuisines = coarse + fine
    if cuisines:
        out["clientmts_cuisine_preference"] = "combos" if len(set(cuisines)) > 1 else cuisines[0]

    return out

# =============================
# Engineering — Maid MTs (acceptance + blocks)
# =============================

def parse_maid_mts(raw: str) -> Dict[str, Any]:
    mtoks = split_tokens(raw)
    out: Dict[str, Any] = {
        "maidmts_household_type": "none",
        "maidmts_pet_type": "none",  # acceptance summary
        "maidmts_dayoff_policy": "unspecified",
        "maidmts_living_arrangement": "unspecified",
        "maid_block_cats": False,
        "maid_block_dogs": False,
        "maid_accept_cats": False,
        "maid_accept_dogs": False,
    }

    # Living arrangement & day-off constraints
    if "private_room" in mtoks or "requires_private_room" in mtoks:
        out["maidmts_living_arrangement"] = "private_room"
    if "avoids_abu_dhabi" in mtoks:
        out["maidmts_living_arrangement"] = "avoids_abu_dhabi"
    if any(t in mtoks for t in ["flex","flexible"]):
        out["maidmts_dayoff_policy"] = "flexible"

    # Household acceptance (if ERP lists the types she accepts)
    accept_baby = "baby" in mtoks or bool(re.search(r"infant|<\s*2\s*y", " ".join(mtoks)))
    accept_many = "many_kids" in mtoks or bool(re.search(r"\b(>=?\s*3|3\+|three or more|many_kids)\b", " ".join(mtoks)))
    if accept_baby and accept_many:
        out["maidmts_household_type"] = "baby_and_kids"
    elif accept_baby:
        out["maidmts_household_type"] = "baby"
    elif accept_many:
        out["maidmts_household_type"] = "many_kids"

    # Pets: explicit blocks
    if "no_cats" in mtoks or "no_pets" in mtoks:
        out["maid_block_cats"] = True
    if "no_dogs" in mtoks or "no_pets" in mtoks:
        out["maid_block_dogs"] = True

    # Pets: acceptance
    out["maid_accept_cats"] = any(t in mtoks for t in ["cat","cats"]) and not out["maid_block_cats"]
    out["maid_accept_dogs"] = any(t in mtoks for t in ["dog","dogs"]) and not out["maid_block_dogs"]

    # Summarize acceptance in maidmts_pet_type for transparency
    if out["maid_accept_cats"] and out["maid_accept_dogs"]:
        out["maidmts_pet_type"] = "both"
    elif out["maid_accept_cats"]:
        out["maidmts_pet_type"] = "cat"
    elif out["maid_accept_dogs"]:
        out["maidmts_pet_type"] = "dog"
    else:
        out["maidmts_pet_type"] = "none"

    return out

# =============================
# Engineering — Maid CPs / profile
# =============================

def parse_maids_cp(raw: str, years_raw: Any, languages_raw: str | None) -> Dict[str, Any]:
    ptoks = split_tokens(raw)
    out: Dict[str, Any] = {
        "maidpref_education": "not_specified",
        "maidpref_kids_experience": "none",
        "maidpref_pet_handling": "none",
        "maidpref_personality": [],
        "maidpref_travel": "unspecified",
        "maidpref_smoking": "unspecified",
        "maidpref_caregiving_profile": "none",
        "years_of_experience": 0.0,
        "maid_languages": [],
    }

    # kids exp
    if "above2" in ptoks: out["maidpref_kids_experience"] = "above2"
    if "lessthan2" in ptoks and out["maidpref_kids_experience"] != "above2": out["maidpref_kids_experience"] = "lessthan2"
    if "both" in ptoks and out["maidpref_kids_experience"] != "above2": out["maidpref_kids_experience"] = "both"

    # pet handling
    cats = "cats" in ptoks or "handles multiple cats" in " ".join(ptoks)
    dogs = "dogs" in ptoks
    if cats and dogs: out["maidpref_pet_handling"] = "both"
    elif cats: out["maidpref_pet_handling"] = "cats"
    elif dogs: out["maidpref_pet_handling"] = "dogs"

    # mobility
    if "travel_and_relocate" in ptoks: out["maidpref_travel"] = "travel_and_relocate"
    elif "relocate" in ptoks: out["maidpref_travel"] = "relocate"
    elif "travel" in ptoks: out["maidpref_travel"] = "travel"

    # soft traits
    soft_candidates = {"energetic","no_attitude","veg_friendly","patient","kind","organized","fast_learner"}
    out["maidpref_personality"] = [t for t in ptoks if t in soft_candidates]

    # education
    if "school" in ptoks: out["maidpref_education"] = "school"
    if "university" in ptoks:
        out["maidpref_education"] = "university" if out["maidpref_education"] != "school" else "both"

    # smoking
    if "non_smoker" in ptoks or "not a smoker" in " ".join(ptoks): out["maidpref_smoking"] = "non_smoker"

    # caregiving
    if "elderly_experienced" in ptoks and "special_needs" in ptoks: out["maidpref_caregiving_profile"] = "elderly_and_special"
    elif "elderly_experienced" in ptoks: out["maidpref_caregiving_profile"] = "elderly_experienced"
    elif "special_needs" in ptoks: out["maidpref_caregiving_profile"] = "special_needs"

    # years (prefer numeric col; fallback to pattern)
    try:
        out["years_of_experience"] = float(years_raw) if years_raw not in (None, "", "nan") else 0.0
    except Exception:
        out["years_of_experience"] = 0.0
    if out["years_of_experience"] == 0.0:
        m = re.search(r"(\d+)\s*(?:y|yr|yrs|year|years)", " ".join(ptoks))
        if m: out["years_of_experience"] = float(m.group(1))

    # languages
    langs = []
    if languages_raw:
        langs += [LANG_ALIASES.get(t, t) for t in split_tokens(languages_raw) if t]
    if "good in english" in " ".join(ptoks): langs.append("en")
    if "good in arabic" in " ".join(ptoks): langs.append("ar")
    out["maid_languages"] = sorted(set(langs))

    return out

# =============================
# Flags & interactions (MT + CP)
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
    f.update({"hard_conflict_total": 0, "pref_mismatch_total": 0, "skill_needed_total": 0})

    # Childcare acceptance vs client needs
    if cn.clientmts_household_type in {"baby","baby_and_kids"}:
        # Maid acceptance via MTs OR via CP kids experience
        maid_accepts_baby = cn.maidmts_household_type in {"baby","baby_and_kids"} or cn.maidpref_kids_experience in {"above2","both","lessthan2"}
        if maid_accepts_baby:
            f["infant_skill_needed"] = True
        else:
            f["infant_conflict"] = True

    if cn.clientmts_household_type in {"many_kids","baby_and_kids"}:
        maid_accepts_many = cn.maidmts_household_type in {"many_kids","baby_and_kids"} or cn.maidpref_kids_experience in {"above2","both"}
        if maid_accepts_many:
            f["kids_over2_skill_needed"] = True
        else:
            f["manykids_conflict"] = True

    # Pets: blocks override acceptance (block > skill)
    client_has_cat = cn.clientmts_pet_type in {"cat","both"}
    client_has_dog = cn.clientmts_pet_type in {"dog","both"}

    maid_accepts_cat = cn.maid_accept_cats or cn.maidmts_pet_type in {"cat","both"} or cn.maidpref_pet_handling in {"cats","both"}
    maid_accepts_dog = cn.maid_accept_dogs or cn.maidmts_pet_type in {"dog","both"} or cn.maidpref_pet_handling in {"dogs","both"}

    if client_has_cat:
        if cn.maid_block_cats or (not maid_accepts_cat):
            f["cat_conflict"] = True
        else:
            f["handles_cats_needed"] = True

    if client_has_dog:
        if cn.maid_block_dogs or (not maid_accepts_dog):
            f["dog_conflict"] = True
        else:
            f["handles_dogs_needed"] = True

    # Living arrangement / location
    if cn.maidmts_living_arrangement == "private_room" and cn.clientmts_living_arrangement != "private_room":
        f["private_room_missing"] = True
    if cn.maidmts_living_arrangement == "avoids_abu_dhabi" and cn.clientmts_living_arrangement == "abu_dhabi":
        f["avoids_abu_dhabi"] = True

    # Day-off mismatch
    if cn.clientmts_dayoff_policy in {"work_for_pay","stay_home_for_pay","combos"} and cn.maidmts_dayoff_policy != "flexible":
        f["dayoff_mismatch"] = True

    # Language expectation (by maid nationality)
    expected = LANG_EXPECT_MAP.get((cn.maid_nationality or "").lower(), set())
    if expected:
        speaks = set(cn.maid_languages or [])
        if ("en" in expected and "en" not in speaks) or ("ar" in expected and "ar" not in speaks):
            f["language_expected_missing"] = True

    # Totals
    f["hard_conflict_total"] = sum(int(f[k]) for k in ["infant_conflict","manykids_conflict","cat_conflict","dog_conflict"])
    f["pref_mismatch_total"] = sum(int(f[k]) for k in ["private_room_missing","dayoff_mismatch","language_expected_missing","avoids_abu_dhabi"])
    f["skill_needed_total"] = sum(int(f[k]) for k in ["infant_skill_needed","kids_over2_skill_needed","handles_cats_needed","handles_dogs_needed"])

    return f

# Experience / Mobility / Soft traits contributions (affect score, not reasons)

def exp_contrib(years: float, policy: Dict[str, Any]) -> float:
    years = max(0.0, float(years or 0.0))
    return min(policy["experience_cap"], policy["experience_slope"] * years)


def mobility_contrib(travel: str, policy: Dict[str, Any]) -> float:
    return policy["mobility_flex"] if travel in {"travel","relocate","travel_and_relocate"} else 0.0


def soft_traits_contrib(traits: List[str], policy: Dict[str, Any]) -> float:
    if not traits: return 0.0
    return min(policy["soft_traits_cap"], float(len(traits)))

# =============================
# Scoring (MT-only reasons surfaced)
# =============================

def compute_score(flags: Dict[str, Any], cn: Canonical, mode: str) -> Tuple[int, str, List[str]]:
    P = POLICIES[mode]
    score = P["baseline"]
    notes: List[str] = []

    # Hard conflicts
    hc = int(flags.get("hard_conflict_total", 0))
    if hc:
        score += P["hard_conflict_each"]
        for k in ["infant_conflict","manykids_conflict","cat_conflict","dog_conflict"]:
            if flags.get(k): notes.append(REASON_TEXT[k])
        if hc > 1: score += (hc - 1) * P["hard_conflict_extra"]

    # Frictions
    for k, w in [("private_room_missing", P["private_room_missing"]),
                 ("dayoff_mismatch", P["dayoff_mismatch"]),
                 ("language_expected_missing", P["language_missing"]),
                 ("avoids_abu_dhabi", P["avoids_abu_dhabi"])]:
        if flags.get(k):
            score += w
            notes.append(REASON_TEXT[k])

    # Need ∧ skill (only when corresponding conflict not present)
    if flags.get("infant_skill_needed") and not flags.get("infant_conflict"):
        score += P["infant_skill_needed"]; notes.append(REASON_TEXT["infant_skill_needed"])
    if flags.get("kids_over2_skill_needed") and not flags.get("manykids_conflict"):
        score += P["kids_over2_skill_needed"]; notes.append(REASON_TEXT["kids_over2_skill_needed"])
    if flags.get("handles_cats_needed") and not flags.get("cat_conflict"):
        score += P["pet_handling_needed"]; notes.append(REASON_TEXT["handles_cats_needed"])
    if flags.get("handles_dogs_needed"):
        if not flags.get("dog_conflict"):
            score += P["pet_handling_needed"]; notes.append(REASON_TEXT["handles_dogs_needed"])

    # Extras (stacked totals)
    snt = max(0, int(flags.get("skill_needed_total", 0)) - 1)
    if P["skill_needed_total_per_extra"] and snt:
        score += min(P["skill_needed_total_cap"], snt * P["skill_needed_total_per_extra"]) or 0
    pmt = max(0, int(flags.get("pref_mismatch_total", 0)) - 1)
    if P["pref_mismatch_total_per_extra"] and pmt:
        adj = pmt * P["pref_mismatch_total_per_extra"]
        score += max(P["pref_mismatch_total_cap"], adj) if adj < 0 else min(P["pref_mismatch_total_cap"], adj)

    # CP contributions (affect score only)
    score += exp_contrib(cn.years_of_experience, P)
    score += mobility_contrib(cn.maidpref_travel, P)
    score += soft_traits_contrib(cn.maidpref_personality or [], P)

    score = int(max(0, min(100, round(score))))
    decision = "OK" if score >= P["ok_cut"] else ("Review" if score >= P["review_low"] else "Blocked")

    # De-dup reasons
    seen, out = set(), []
    for r in notes:
        if r not in seen:
            out.append(r); seen.add(r)
    return score, decision, out

# =============================
# End-to-end pipeline for a row
# =============================

def process_row(
    client_raw: str,
    maid_raw: str,
    cp_raw: str,
    cuisine_raw: str,
    years_raw: Any,
    maid_lang_raw: str,
    maid_nat: str
) -> Canonical:
    cn = Canonical()
    for k, v in parse_client_mts(client_raw, cuisine_raw).items(): setattr(cn, k, v)
    maid_parsed = parse_maid_mts(maid_raw)
    for k, v in maid_parsed.items(): setattr(cn, k, v)
    for k, v in parse_maids_cp(cp_raw, years_raw, maid_lang_raw).items(): setattr(cn, k, v)
    cn.maid_nationality = (maid_nat or "").strip().lower()
    return cn

# =============================
# Streamlit UI
# =============================

def main():
    st.set_page_config(page_title="Matching Score — Mirror Notebooks", layout="wide")
    st.title("Matching Score (RAW → engineered → score)")
    st.caption("Robust MT parsing (acceptance & blocks) + CP scoring. Reasons show MT logic only.")

    with st.sidebar:
        st.header("Policy mode")
        mode = st.radio("Mode", ["Balanced","Strict","Flexible"], index=0)
        st.markdown(
            "**Decision bands**  \n"
            "Strict: OK ≥ 75, Review 55–74, Block < 55  \n"
            "Balanced: OK ≥ 70, Review 50–69, Block < 50  \n"
            "Flexible: OK ≥ 65, Review 45–64, Block < 45"
        )

    tab_single, tab_batch = st.tabs(["Single case","Batch CSV"])

    with tab_single:
        st.subheader("Inputs (RAW strings)")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.text_input("Client ID", "C-123")
        with c2: st.text_input("Maid ID", "M-321")
        with c3: maid_nationality = st.selectbox("Maid nationality", ["","filipina","ethiopian","indian","west_african"], index=1)
        with c4: years_of_experience = st.text_input("Years of experience (e.g., 3 or '3 years')", "")

        client_mts_raw = st.text_area("client_mts_at_hiring", placeholder="has a baby younger than 2 years old, has 3 kids or more, has a dog, has a private room, lives in abu dhabi, filipina, work on her day off for pay")
        maid_mts_raw = st.text_area("maid_mts_at_hiring", placeholder="requires private room, flexible, avoids abu dhabi, no cats / OR 'has a baby' 'has a dog' as acceptance")
        cp_raw = st.text_area("maids_custom_preferences_at_hiring", placeholder="handles multiple cats, maid has experience with kids above 2 years old, maid is energetic, maid is good in english, non_smoker, travel")
        cooking_raw = st.text_input("cooking_details (optional)", "")
        maid_lang_raw = st.text_input("maid_speaks_language (optional)", "")

        if st.button("Compute Matching Score", type="primary"):
            cn = process_row(client_mts_raw, maid_mts_raw, cp_raw, cooking_raw, years_of_experience, maid_lang_raw, maid_nationality)
            flags = make_flags(cn)
            score, decision, reasons = compute_score(flags, cn, mode)

            k1,k2,k3 = st.columns(3)
            k1.metric("Match Score", score)
            k2.metric("Decision", decision)
            k3.metric("Policy", mode)

            with st.expander("Preview engineered fields (read-only)", expanded=False):
                st.json({k:getattr(cn,k) for k in cn.__dataclass_fields__.keys()})

            st.markdown("### Reasons (MT-derived)")
            if reasons:
                for r in reasons: st.write("• ", r)
            else:
                st.info("No MT conflicts or frictions detected.")

    with tab_batch:
        st.subheader("Batch scoring (CSV)")
        st.write("Required columns: client_id, maid_id, client_mts_at_hiring, maid_mts_at_hiring, maids_custom_preferences_at_hiring, cooking_details, years_of_experience, maid_speaks_language, maid_nationality")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            required = {
                "client_id","maid_id",
                "client_mts_at_hiring","maid_mts_at_hiring",
                "maids_custom_preferences_at_hiring","cooking_details",
                "years_of_experience","maid_speaks_language","maid_nationality",
            }
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                rows = []
                for _, r in df.iterrows():
                    cn = process_row(
                        r.get("client_mts_at_hiring",""),
                        r.get("maid_mts_at_hiring",""),
                        r.get("maids_custom_preferences_at_hiring",""),
                        r.get("cooking_details",""),
                        r.get("years_of_experience",""),
                        r.get("maid_speaks_language",""),
                        r.get("maid_nationality",""),
                    )
                    flags = make_flags(cn)
                    score, decision, reasons = compute_score(flags, cn, mode)
                    rows.append({
                        "client_id": r.get("client_id",""),
                        "maid_id": r.get("maid_id",""),
                        "match_score": score,
                        "decision": decision,
                        "policy_mode": mode,
                        "mt_reasons": "; ".join(reasons),
                    })
                out = pd.DataFrame(rows)
                st.success(f"Scored {len(out)} rows.")
                st.dataframe(out.head(20), use_container_width=True)
                st.download_button(
                    "Download matching_scores.csv",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="matching_scores.csv"
                )

if __name__ == "__main__":
    main()
