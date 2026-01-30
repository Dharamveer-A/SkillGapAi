"""
SkillGapAI Modules Initialization
"""

from .document_processing import extract_text, clean_text, extract_experience
from .nlp_processing import (
    extract_technical_skills, 
    extract_soft_skills, 
    categorize_skills, 
    load_nlp_resources
)
from .github_analyzer import (
    detect_github_in_resume, 
    extract_github_username, 
    fetch_github_repos, 
    extract_github_skills,
    analyze_github_profile
)
from .visualization import (
    create_donut_chart, 
    create_category_heatmap,
    create_skill_distribution_chart
)
from .report_generator import (
    generate_csv_report, 
    generate_word_report, 
    generate_pdf_report
)
from .utils import (
    build_similarity_matrix, 
    classify_skill_matches, 
    compute_metrics,
    calculate_skill_match
)