"""
Utility Functions Module
Advanced Skill Matching & Metrics (Verified for High Accuracy)
"""

import numpy as np
import re

def compute_metrics(resume_skills, jd_skills):
    """
    Computes high-level metrics for the dashboard cards.
    Fixes: KeyError for 'jd_tech', 'jd_soft', 'total_jd', and 'match_percent'.
    """
    # 1. Safely extract lists from dictionaries
    r_tech = resume_skills.get("technical", []) if isinstance(resume_skills, dict) else []
    r_soft = resume_skills.get("soft", []) if isinstance(resume_skills, dict) else []
    
    j_tech = jd_skills.get("technical", []) if isinstance(jd_skills, dict) else []
    j_soft = jd_skills.get("soft", []) if isinstance(jd_skills, dict) else []

    # Flatten for overall match calculation
    r_all = r_tech + r_soft
    j_all = j_tech + j_soft

    # 2. Use the high-accuracy calculation logic
    match_results = calculate_skill_match(r_all, j_all)
    
    # 3. Provide all keys required by the app.py metric cards
    return {
        "avg_match": match_results["avg_match"],
        "match_percent": match_results["avg_match"],
        "matched": match_results["matched"],
        "missing": match_results["missing"],
        "jd_tech": len(j_tech),
        "jd_soft": len(j_soft),
        "total_jd": len(j_all)
    }

def calculate_skill_match(resume_skills, jd_skills):
    """
    Calculates match percentage based on Job Description requirements.
    Uses version-aware fuzzy logic to match skills like 'ES6+' or 'HTML5'.
    """
    if not jd_skills:
        return {"matched": 0, "partial": 0, "missing": 0, "avg_match": 0}
    
    # Clean input to ensure we are working with list of strings
    r_skills = [str(s).lower().strip() for s in resume_skills]
    j_skills = [str(s).lower().strip() for s in jd_skills]
    
    matched_count = 0
    partial_count = 0
    
    for js in j_skills:
        # 1. Exact Match
        if js in r_skills:
            matched_count += 1
            continue
            
        # 2. Version/Fuzzy Match Logic
        # This allows 'JavaScript' to match 'JavaScript (ES6+)' 
        # and 'SQL' to match 'MySQL/PostgreSQL'
        is_partial = False
        for rs in r_skills:
            # Clean strings to compare alphanumeric only
            js_clean = re.sub(r'[^a-z0-9]', '', js)
            rs_clean = re.sub(r'[^a-z0-9]', '', rs)
            
            if (js_clean and rs_clean) and (js_clean in rs_clean or rs_clean in js_clean):
                partial_count += 1
                is_partial = True
                break
        
    missing_count = len(j_skills) - matched_count - partial_count
    
    # Weighted scoring for accuracy (Partial matches get 80% credit)
    # This prevents punishing candidates for slight naming variations
    total_score = (matched_count * 1.0) + (partial_count * 0.8)
    avg_match = (total_score / len(j_skills)) * 100
    
    return {
        "matched": matched_count,
        "partial": partial_count,
        "missing": missing_count,
        "avg_match": round(min(avg_match, 100.0), 1)
    }

def build_similarity_matrix(resume_skills, jd_skills):
    """Builds a matrix comparing every Resume skill to every JD skill"""
    if not resume_skills or not jd_skills:
        return np.zeros((1, 1))
        
    matrix = np.zeros((len(resume_skills), len(jd_skills)))
    
    for i, rs in enumerate(resume_skills):
        rs_low = str(rs).lower().strip()
        for j, js in enumerate(jd_skills):
            js_low = str(js).lower().strip()
            
            if rs_low == js_low:
                matrix[i][j] = 1.0
            elif rs_low in js_low or js_low in rs_low:
                matrix[i][j] = 0.85 # High score for version matches
            else:
                matrix[i][j] = 0.0
    return matrix

def classify_skill_matches(similarity_matrix, resume_skills, jd_skills):
    """Groups skills for the 'Skill Gap Details' tabs in app.py"""
    matched = []
    partial = []
    missing = []
    
    # Ensure similarity_matrix is a numpy array
    sim_mat = np.array(similarity_matrix)
    
    for j, jd_skill in enumerate(jd_skills):
        max_score = np.max(sim_mat[:, j]) if sim_mat.size > 0 else 0
        
        if max_score >= 0.95:
            matched.append(jd_skill)
        elif max_score >= 0.7:
            partial.append(jd_skill)
        else:
            missing.append(jd_skill)
            
    return {
        "matched": matched,
        "partial": partial,
        "missing": missing
    }