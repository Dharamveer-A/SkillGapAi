"""
Utility Functions Module
Final Calibration: High Scoring Logic with Dynamic Weights
"""

import numpy as np
import re

def extract_experience_years(text):
    """
    Extracts the maximum 'years of experience' mentioned in text.
    """
    if not text: return 0
    pattern = r'(\d+)(?:\+|\s*-\s*\d+)?\s*(?:years|yrs|year)'
    matches = re.findall(pattern, text.lower())
    if not matches: return 0
    try:
        return max([int(m) for m in matches])
    except:
        return 0

def compute_metrics(resume_skills, jd_skills, resume_text, jd_text):
    """
    Computes metrics with Boosted Scoring for Partial Matches.
    """
    # 1. Robust Extraction
    if isinstance(resume_skills, dict):
        r_all = resume_skills.get("technical", []) + resume_skills.get("soft", [])
    else:
        r_all = list(resume_skills) if resume_skills else []

    if isinstance(jd_skills, dict):
        j_all = jd_skills.get("technical", []) + jd_skills.get("soft", [])
        j_tech = jd_skills.get("technical", [])
        j_soft = jd_skills.get("soft", [])
    else:
        j_all = list(jd_skills) if jd_skills else []
        j_tech = j_all
        j_soft = []

    # 2. Calculate Skill Match (Uses the new 0.8 weight)
    match_results = calculate_skill_match(r_all, j_all)
    
    # 3. Calculate Experience Score
    res_exp = extract_experience_years(resume_text)
    jd_exp = extract_experience_years(jd_text)
    
    if jd_exp == 0:
        exp_score = 100
    else:
        # Curve the score: If you have 50% of req exp, you get 70% points (Encouraging)
        raw_ratio = res_exp / jd_exp
        if raw_ratio >= 1: exp_score = 100
        else: exp_score = min(100, (raw_ratio * 100) + 20) # +20 Bonus for having ANY exp

    # 4. Dynamic Weighting
    if jd_exp <= 1:
        # Fresher: 90% Skills / 10% Exp
        weight_skills = 0.90; weight_exp = 0.10
        weight_text = "(90% Skills + 10% Experience)"
    else:
        # Senior: 80% Skills / 20% Exp
        weight_skills = 0.80; weight_exp = 0.20
        weight_text = "(80% Skills + 20% Experience)"
        
    final_score = (match_results["avg_match"] * weight_skills) + (exp_score * weight_exp)
    
    return {
        "final_score": round(final_score, 1),
        "skill_match": match_results["avg_match"],
        "match_percent": match_results["avg_match"],  # Alias for compatibility
        "exp_score": round(exp_score, 1),
        "res_years": res_exp,
        "jd_years": jd_exp,
        "weight_text": weight_text,
        
        "matched": match_results["matched"],
        "partial": match_results["partial"],
        "missing": match_results["missing"],
        "partial_points": match_results["partial_points"],
        
        "jd_tech": len(j_tech),
        "jd_soft": len(j_soft),
        "total_jd": len(j_all) 
    }

def calculate_skill_match(resume_skills, jd_skills):
    """
    Calculates match percentage. 
    BOOST: Partial Matches now count for 0.8 (80%) points instead of 0.5.
    """
    if not jd_skills:
        return {"matched": 0, "partial": 0, "missing": 0, "avg_match": 0.0, "partial_points": 0}
    
    r_skills = [str(s).lower().strip() for s in resume_skills]
    j_skills = [str(s).lower().strip() for s in jd_skills]
    
    matched_count = 0
    partial_count = 0
    
    for js in j_skills:
        if js in r_skills:
            matched_count += 1
            continue
        # Partial Check
        for rs in r_skills:
            js_clean = re.sub(r'[^a-z0-9]', '', js)
            rs_clean = re.sub(r'[^a-z0-9]', '', rs)
            if (js_clean and rs_clean) and (js_clean in rs_clean or rs_clean in js_clean):
                partial_count += 1
                break
        
    missing_count = len(j_skills) - matched_count - partial_count
    
    # --- SCORING CALIBRATION ---
    # Exact Match = 1.0
    # Partial Match = 0.8 (Restored High Score)
    partial_points_total = partial_count * 0.8 
    
    total_points = (matched_count * 1.0) + partial_points_total
    avg_match = (total_points / len(j_skills)) * 100
    
    return {
        "matched": matched_count,
        "partial": partial_count,
        "partial_points": round(partial_points_total, 1),
        "missing": missing_count,
        "avg_match": round(min(avg_match, 100.0), 1)
    }

def build_similarity_matrix(resume_skills, jd_skills):
    if not resume_skills or not jd_skills: return np.zeros((1, 1))
    matrix = np.zeros((len(resume_skills), len(jd_skills)))
    for i, rs in enumerate(resume_skills):
        rs_low = str(rs).lower().strip()
        for j, js in enumerate(jd_skills):
            js_low = str(js).lower().strip()
            if rs_low == js_low: matrix[i][j] = 1.0
            elif rs_low in js_low or js_low in rs_low: matrix[i][j] = 0.85
            else: matrix[i][j] = 0.0
    return matrix

def classify_skill_matches(similarity_matrix, resume_skills, jd_skills):
    matched, partial, missing = [], [], []
    sim_mat = np.array(similarity_matrix)
    for j, jd_skill in enumerate(jd_skills):
        max_score = np.max(sim_mat[:, j]) if sim_mat.size > 0 else 0
        if max_score >= 0.95: matched.append(jd_skill)
        elif max_score >= 0.7: partial.append(jd_skill)
        else: missing.append(jd_skill)
    return {"matched": matched, "partial": partial, "missing": missing}