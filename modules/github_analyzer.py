"""
GitHub Profile Analyzer Module
Analyzes GitHub profiles for additional skill extraction
"""

import re
import requests
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
    """Fetch public repositories of a GitHub user"""
    api_url = f"https://api.github.com/users/{username}/repos"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


def extract_github_skills(repos, skill_set):
    """Extract skills from repo names, descriptions, and languages"""
    found_skills = set()
    for repo in repos:
        text = f"{repo.get('name','')} {repo.get('description','')}".lower()
        for skill in skill_set:
            if skill in text:
                found_skills.add(skill)
        if repo.get("language"):
            lang = repo["language"].lower()
            if lang in skill_set:
                found_skills.add(lang)
    return sorted(found_skills)


def analyze_github_profile(repos):
    """Analyze GitHub profile and provide suggestions"""
    if not repos:
        return None
    
    total_repos = len(repos)
    languages_used = {}
    topics_used = set()
    has_readme_projects = 0
    recent_activity = 0
    
    for repo in repos:
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
    
    return {
        "insights": insights,
        "suggestions": suggestions
    }