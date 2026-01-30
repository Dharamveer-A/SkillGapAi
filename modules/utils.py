"""
Utility Functions Module
Helper functions for skill analysis
"""

import numpy as np


def build_similarity_matrix(resume_skills, jd_skills):
    """Build similarity matrix between resume and JD skills"""
    if not resume_skills or not jd_skills:
        return np.array([])
    matrix = np.zeros((len(resume_skills), len(jd_skills)))
    for i, r_skill in enumerate(resume_skills):
        for j, j_skill in enumerate(jd_skills):
            if r_skill == j_skill:
                matrix[i][j] = 1.0
            elif r_skill in j_skill or j_skill in r_skill:
                matrix[i][j] = 0.5
            else:
                matrix[i][j] = 0.0
    return matrix


def classify_skill_matches(similarity_matrix, resume_skills, jd_skills):
    """Classify skills as matched, partial, or missing"""
    matched = set()
    partial = set()
    for j, jd_skill in enumerate(jd_skills):
        column = similarity_matrix[:, j]
        if 1.0 in column:
            matched.add(jd_skill)
        elif 0.5 in column:
            partial.add(jd_skill)
    missing = set(jd_skills) - matched - partial
    return {
        "matched": sorted(list(matched)),
        "partial": sorted(list(partial)),
        "missing": sorted(list(missing))
    }


def compute_metrics(resume_skills, jd_skills):
    """Compute skill matching metrics"""
    resume_tech = set(resume_skills["technical"])
    resume_soft = set(resume_skills["soft"])
    jd_tech = set(jd_skills["technical"])
    jd_soft = set(jd_skills["soft"])
    matched_skills = (resume_tech & jd_tech) | (resume_soft & jd_soft)
    total_jd_skills = len(jd_tech) + len(jd_soft)
    match_percentage = 0
    if total_jd_skills > 0:
        match_percentage = round((len(matched_skills) / total_jd_skills) * 100, 1)
    return {
        "resume_tech": len(resume_tech),
        "resume_soft": len(resume_soft),
        "jd_tech": len(jd_tech),
        "jd_soft": len(jd_soft),
        "total_resume": len(resume_tech) + len(resume_soft),
        "total_jd": total_jd_skills,
        "match_percent": match_percentage
    }


def calculate_skill_match(resume_skills, jd_skills):
    """Calculate detailed skill match statistics"""
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    matched = resume_set & jd_set
    missing = jd_set - resume_set
    partial = set()
    for jd_skill in missing.copy():
        for res_skill in resume_set:
            if jd_skill in res_skill or res_skill in jd_skill:
                partial.add(jd_skill)
    missing = missing - partial
    avg_match = 0
    if len(jd_set) > 0:
        avg_match = round(((len(matched) + 0.5 * len(partial)) / len(jd_set)) * 100, 1)
    return {
        "matched": len(matched),
        "partial": len(partial),
        "missing": len(missing),
        "avg_match": avg_match
    }