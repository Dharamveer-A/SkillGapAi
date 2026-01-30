"""
NLP Processing Module
Handles skill extraction using spaCy and pattern matching
"""

import spacy
from spacy.matcher import PhraseMatcher
import streamlit as st
from data.skills_list import SKILL_CATEGORIES


@st.cache_resource
def load_spacy_model():
    """Load spaCy model with caching"""
    try:
        return spacy.load("en_core_web_trf")
    except OSError:
        st.error("spaCy model 'en_core_web_trf' not found. Please install it using: python -m spacy download en_core_web_trf")
        return None


def extract_soft_skills(text, skill_set):
    """Extract soft skills using simple text matching"""
    if not text:
        return []
    text = text.lower()
    found_skills = set()
    for skill in skill_set:
        if skill in text:
            found_skills.add(skill)
    return sorted(found_skills)


def extract_technical_skills(text, skill_set):
    """Extract technical skills using spaCy PhraseMatcher"""
    if not text:
        return []
    
    nlp = load_spacy_model()
    if nlp is None:
        return []
    
    max_length = 1000000
    
    if len(text) > max_length:
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        all_skills = set()
        
        for chunk in chunks:
            matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
            patterns = [nlp.make_doc(skill) for skill in skill_set]
            matcher.add("TECH_SKILLS", patterns)
            doc = nlp(chunk)
            matches = matcher(doc)
            for _, start, end in matches:
                all_skills.add(doc[start:end].text.lower())
        
        return sorted(list(all_skills))
    else:
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(skill) for skill in skill_set]
        matcher.add("TECH_SKILLS", patterns)
        doc = nlp(text)
        matches = matcher(doc)
        skills_found = set()
        for _, start, end in matches:
            skills_found.add(doc[start:end].text.lower())
        return sorted(skills_found)


def categorize_skills(skills):
    """Categorize skills into meaningful groups"""
    categorized = {}
    uncategorized = []
    
    for skill in skills:
        skill_lower = skill.lower()
        found = False
        for category, category_skills in SKILL_CATEGORIES.items():
            if skill_lower in category_skills:
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