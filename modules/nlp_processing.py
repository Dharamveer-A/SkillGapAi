"""
NLP Processing Module
Handles skill extraction using spaCy and pattern matching (Optimized)
"""

import spacy
from spacy.matcher import PhraseMatcher
import streamlit as st
from data.skills_list import SKILL_CATEGORIES


@st.cache_resource
def load_nlp_resources(technical_skills_set, soft_skills_set):
    """
    Load spaCy model AND build matchers once to boost performance.
    This runs only once when the app starts.
    """
    try:
        # Try loading the accurate transformer model
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        # Fallback to small model if trf is missing, or warn user
        try:
            nlp = spacy.load("en_core_web_sm")
            st.warning("⚠️ Using lighter 'en_core_web_sm' model. For better accuracy, install 'en_core_web_trf'.")
        except OSError:
            st.error("❌ No spaCy model found. Please run: python -m spacy download en_core_web_trf")
            return None, None, None

    # Build Technical Matcher
    tech_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    # Convert set to list of strings to ensure compatibility
    tech_patterns = [nlp.make_doc(str(text)) for text in technical_skills_set]
    tech_matcher.add("TECH_SKILLS", tech_patterns)

    # Build Soft Skills Matcher
    soft_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    soft_patterns = [nlp.make_doc(str(text)) for text in soft_skills_set]
    soft_matcher.add("SOFT_SKILLS", soft_patterns)

    return nlp, tech_matcher, soft_matcher


def extract_skills_unified(text, skill_type, tech_set, soft_set):
    """Unified extraction function that uses the cached matchers"""
    if not text:
        return []

    # This call hits the cache, so it's instant
    nlp, tech_matcher, soft_matcher = load_nlp_resources(tech_set, soft_set)
    
    if not nlp:
        return []

    doc = nlp(text)
    found_skills = set()
    
    if skill_type == "technical":
        matches = tech_matcher(doc)
    else:
        matches = soft_matcher(doc)

    for _, start, end in matches:
        found_skills.add(doc[start:end].text.lower())

    return sorted(list(found_skills))


# Wrapper functions to match your original function calls in app.py
def extract_technical_skills(text, skill_set):
    # We pass both sets to the loader so it caches everything once
    return extract_skills_unified(text, "technical", skill_set, set())


def extract_soft_skills(text, skill_set):
    # We pass empty tech set here, but the loader uses the cached version anyway
    return extract_skills_unified(text, "soft", set(), skill_set)


def categorize_skills(skills):
    """Categorize skills into meaningful groups"""
    categorized = {}
    uncategorized = []
    
    for skill in skills:
        skill_lower = skill.lower()
        found = False
        for category, category_skills in SKILL_CATEGORIES.items():
            # Check if skill is in this category (case-insensitive)
            if skill_lower in [s.lower() for s in category_skills]:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(skill)
                found = True
                break
        if not found:
            uncategorized.append(skill)
    
    if uncategorized:
        categorized["Other Skills"] = uncategorized
    
    return categorized