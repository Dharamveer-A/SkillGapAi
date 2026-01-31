"""
GitHub Utilities Module
Fetches user repositories, extracts skills, and analyzes profile stats.
Fully integrated with Streamlit Secrets for high API limits.
"""

import requests
import re
import streamlit as st
from collections import Counter

def detect_github_in_resume(text):
    """
    Finds a GitHub profile URL in the resume text.
    """
    if not text: return None
    # Regex to find github.com/username (ignoring trailing slashes/paths)
    pattern = r"github\.com/([a-zA-Z0-9-]+)"
    match = re.search(pattern, text)
    if match:
        return f"https://github.com/{match.group(1)}"
    return None

def extract_github_username(url):
    """
    Extracts the clean 'username' from a full URL.
    """
    if not url: return None
    pattern = r"github\.com/([a-zA-Z0-9-]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_auth_headers():
    """
    Returns authorization headers if token exists in secrets.
    """
    if "GITHUB_TOKEN" in st.secrets:
        return {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    return {}

def fetch_github_repos(username):
    """
    Fetches public repositories for a user.
    """
    url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
    try:
        response = requests.get(url, headers=get_auth_headers(), timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            # Rate limit hit or bad token
            print("GitHub Rate Limit Reached")
            return []
        return []
    except:
        return []

def extract_github_skills(repos, master_skills_list):
    """
    Scans repo languages, topics, and descriptions to find technical skills.
    """
    found_skills = set()
    master_set = {s.lower() for s in master_skills_list}
    
    for repo in repos:
        # 1. Primary Language
        lang = repo.get("language")
        if lang and lang.lower() in master_set:
            found_skills.add(lang)
            
        # 2. Topics (Tags)
        topics = repo.get("topics", [])
        for topic in topics:
            if topic.lower() in master_set:
                found_skills.add(topic)
        
        # 3. Description Keywords
        desc = repo.get("description")
        if desc:
            # Simple tokenization
            desc_words = set(re.findall(r"\w+", desc.lower()))
            common = desc_words.intersection(master_set)
            found_skills.update(common)
            
    return list(found_skills)

def analyze_github_profile(username):
    """
    Performs a deep dive analysis of the GitHub profile.
    Returns a dictionary of stats (Stars, Followers, Top Languages).
    """
    # 1. Fetch Basic Profile Info
    profile_url = f"https://api.github.com/users/{username}"
    stats = {
        "name": username,
        "bio": "",
        "followers": 0,
        "public_repos": 0,
        "total_stars": 0,
        "top_languages": []
    }
    
    try:
        # Get User Metadata
        user_resp = requests.get(profile_url, headers=get_auth_headers(), timeout=5)
        if user_resp.status_code == 200:
            user_data = user_resp.json()
            stats["name"] = user_data.get("name") or username
            stats["bio"] = user_data.get("bio", "")
            stats["followers"] = user_data.get("followers", 0)
            stats["public_repos"] = user_data.get("public_repos", 0)
        
        # 2. Calculate Stats from Repos
        repos = fetch_github_repos(username)
        if repos:
            # Count Stars
            stats["total_stars"] = sum([repo.get("stargazers_count", 0) for repo in repos])
            
            # Count Languages
            languages = []
            for repo in repos:
                lang = repo.get("language")
                if lang:
                    languages.append(lang)
            
            # Get Top 3 Most Used Languages
            if languages:
                most_common = Counter(languages).most_common(3)
                stats["top_languages"] = [lang[0] for lang in most_common]
                
    except Exception as e:
        print(f"Error analyzing profile: {e}")
        
    return stats