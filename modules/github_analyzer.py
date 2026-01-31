"""
GitHub Profile Analyzer Module
Analyzes GitHub profiles for additional skill extraction (Optimized & Safe)
"""

import re
import requests
import streamlit as st
from datetime import datetime, timedelta


def detect_github_in_resume(text):
    """Detect GitHub profile URL or username in resume"""
    if not text:
        return None
    
    url_pattern = r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,38}[a-zA-Z0-9])?)'
    url_match = re.search(url_pattern, text, re.IGNORECASE)
    
    if url_match:
        return f"https://github.com/{url_match.group(1)}"
    
    username_pattern = r'(?:github|gh)[:\s]+([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,38}[a-zA-Z0-9])?)'
    username_match = re.search(username_pattern, text, re.IGNORECASE)
    
    if username_match:
        return f"https://github.com/{username_match.group(1)}"
    
    return None


def extract_github_username(url):
    """Extract username from GitHub URL"""
    match = re.search(r'github\.com/([^/]+)/?', url)
    return match.group(1) if match else None


def fetch_github_repos(username):
    """Fetch public repositories of a GitHub user with rate limit safety"""
    api_url = f"https://api.github.com/users/{username}/repos"
    try:
        response = requests.get(api_url, timeout=10)
        
        # Prevent app crash if GitHub rate limit (60 req/hr) is hit
        if response.status_code == 403:
            st.warning("⚠️ GitHub API rate limit exceeded. GitHub skills cannot be analyzed right now.")
            return []
            
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return []


def extract_github_skills(repos, skill_set):
    """Extract skills from repo names/desc using word boundaries to prevent false matches"""
    found_skills = set()
    
    if not repos:
        return []

    import re
    
    for repo in repos:
        # Combine name and description
        text = f"{repo.get('name','')} {repo.get('description','') or ''}".lower()
        
        for skill in skill_set:
            # First do a fast string check
            if skill in text:
                # Use regex boundaries to ensure "Good" doesn't match "Go"
                # Escape skill to handle special chars like C++
                pattern = r'(?:^|[\s\W])' + re.escape(skill) + r'(?:$|[\s\W])'
                if re.search(pattern, text):
                    found_skills.add(skill)
        
        # Always trust the official GitHub language field
        if repo.get("language"):
            lang = repo["language"].lower()
            if lang in skill_set:
                found_skills.add(lang)
                
    return sorted(found_skills)


def analyze_github_profile(username_or_repos):
    """
    Analyze GitHub profile and provide suggestions.
    Accepts either a username (string) or repos list (from fetch_github_repos).
    Returns stats dict with insights and suggestions.
    """
    # Handle both username and repos list
    if isinstance(username_or_repos, str):
        # It's a username, fetch repos
        repos = fetch_github_repos(username_or_repos)
        username = username_or_repos
    elif isinstance(username_or_repos, list):
        # It's already a repos list
        repos = username_or_repos
        username = None
    else:
        return None
    
    if not repos:
        return None
    
    total_repos = len(repos)
    languages_used = {}
    topics_used = set()
    has_readme_projects = 0
    recent_activity = 0
    total_stars = 0
    
    for repo in repos:
        # Count stars
        total_stars += repo.get("stargazers_count", 0)
        
        if repo.get("language"):
            lang = repo["language"]
            languages_used[lang] = languages_used.get(lang, 0) + 1
        
        if repo.get("topics"):
            topics_used.update(repo["topics"])
        
        if repo.get("description"):
            has_readme_projects += 1
        
        if repo.get("updated_at"):
            try:
                updated = datetime.strptime(repo["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
                if updated > datetime.now() - timedelta(days=180):
                    recent_activity += 1
            except:
                pass
    
    # Get user profile stats if we have a username
    followers = 0
    public_repos_count = total_repos
    
    if username:
        try:
            profile_url = f"https://api.github.com/users/{username}"
            response = requests.get(profile_url, timeout=5)
            if response.status_code == 200:
                user_data = response.json()
                followers = user_data.get("followers", 0)
                public_repos_count = user_data.get("public_repos", total_repos)
        except:
            pass
    
    insights = {
        "total_repos": total_repos,
        "languages": languages_used,
        "top_language": max(languages_used.items(), key=lambda x: x[1])[0] if languages_used else None,
        "topics": list(topics_used),
        "documented_projects": has_readme_projects,
        "active_repos": recent_activity,
        "activity_rate": round((recent_activity / total_repos) * 100, 1) if total_repos > 0 else 0
    }
    
    suggestions = []
    
    if has_readme_projects < total_repos * 0.5:
        suggestions.append("Add detailed README files to more projects to showcase your work better")
    
    if recent_activity < 3:
        suggestions.append("Increase recent GitHub activity - consider contributing to open source or creating new projects")
    
    if len(languages_used) < 3:
        suggestions.append("Diversify your tech stack by learning new programming languages")
    
    if not topics_used:
        suggestions.append("Add topics/tags to your repositories for better discoverability")
    
    if total_repos < 5:
        suggestions.append("Build more projects to demonstrate your skills and experience")
    
    # Return stats compatible with both mobile_app.py and web_app2.py
    return {
        "insights": insights,
        "suggestions": suggestions,
        # Additional fields for mobile_app.py compatibility
        "total_stars": total_stars,
        "followers": followers,
        "public_repos": public_repos_count,
        "top_languages": [lang[0] for lang in sorted(languages_used.items(), key=lambda x: x[1], reverse=True)[:3]]
    }