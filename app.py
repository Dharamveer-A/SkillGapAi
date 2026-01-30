"""
SkillGapAI - Main Application
AI-Powered Skill Gap Analysis & Career Insights
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Import from the modules package (uses your 2.py __init__.py)
from modules import (
    extract_text, 
    clean_text, 
    extract_experience,
    extract_technical_skills, 
    extract_soft_skills, 
    categorize_skills,
    detect_github_in_resume, 
    extract_github_username, 
    fetch_github_repos, 
    extract_github_skills,
    analyze_github_profile,
    create_donut_chart, 
    create_category_heatmap,
    create_skill_distribution_chart,
    generate_csv_report, 
    generate_word_report, 
    generate_pdf_report,
    build_similarity_matrix, 
    classify_skill_matches, 
    compute_metrics,
    calculate_skill_match
)

# Keep data imports separate
from data.skills_list import MASTER_TECHNICAL_SKILLS, MASTER_SOFT_SKILLS

# Page Configuration
st.set_page_config(page_title="SkillGapAI", layout="wide")

# Header
st.markdown("""
<h1 style="text-align:center; color:#470047;">SkillGapAI</h1>
<p style="text-align:center; font-size:18px; color:#6B7280;">
AI-Powered Skill Gap Analysis & Career Insights
</p>
<hr>
""", unsafe_allow_html=True)

# Tutorial Section
with st.expander("How to Use - Watch Tutorial", expanded=False):
    st.markdown("""
    ### Quick Tutorial
    Watch this short video to learn how to use the Skill Gap Analysis Dashboard:
    """)
    
    d1, d2, d3 = st.columns(3)
    with d2:
        try:
            video_file = open('assets/tutorial.mov', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
        except FileNotFoundError:
            st.warning("Tutorial video not found. Please add 'tutorial.mov' to the assets folder.")
    
    st.info("""
    **Step-by-Step Instructions:**
    
    1. **Upload Resume** - Click on the left uploader and select your resume (PDF, DOCX, or TXT)
    2. **Upload Job Description** - Click on the right uploader and select the job description
    3. **Click Analyze** - Press the "Analyze Documents" button
    4. **View Results** - Review skill matches, gaps, and recommendations
    5. **Download Reports** - Export your analysis in CSV, DOCX, or PDF format
    """)

st.markdown("---")

# ============================================================================
# MILESTONE 1: DATA INGESTION, PARSING AND CLEANING
# ============================================================================

st.markdown("""
<div style="background-color:#470047;padding:20px;border-radius:10px">
    <h2 style="color:white;">Data Ingestion, Parsing and Cleaning Module</h2>
    <p style="color:white;">
        This module allows uploading resumes and job descriptions,
        extracting text from multiple document formats (including multi-page documents),
        and displaying clean, normalized content for further processing.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Initialize session state variables
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = None

# File Upload Section
st.subheader("Upload Documents")

left_col, right_col = st.columns(2)

with left_col:
    resume_file = st.file_uploader(
        "Upload Resume (Multi-page supported)",
        type=["pdf", "docx", "txt"],
        help="Upload your resume in PDF, DOCX, or TXT format. Multi-page documents are fully supported.",
        accept_multiple_files=False
    )
    if resume_file:
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = '.' + resume_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            st.error(f"**Invalid file format: {file_extension}**")
            st.warning("Please upload only PDF (.pdf), Word Document (.docx), or Text (.txt) files.")
            resume_file = None
        else:
            allowed_mime_types = {
                "application/pdf": "PDF",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
                "text/plain": "TXT"
            }
            
            if resume_file.type in allowed_mime_types:
                st.success(f"Resume uploaded: {resume_file.name} ({allowed_mime_types[resume_file.type]})")
            else:
                st.error(f"**Unsupported file type detected!**")
                st.warning(f"File type '{resume_file.type}' is not supported. Please upload PDF, DOCX, or TXT files only.")
                resume_file = None

with right_col:
    jd_file = st.file_uploader(
        "Upload Job Description (Multi-page supported)",
        type=["pdf", "docx", "txt"],
        help="Upload job description in PDF, DOCX, or TXT format. Multi-page documents are fully supported.",
        accept_multiple_files=False
    )
    if jd_file:
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = '.' + jd_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            st.error(f"**Invalid file format: {file_extension}**")
            st.warning("Please upload only PDF (.pdf), Word Document (.docx), or Text (.txt) files.")
            jd_file = None
        else:
            allowed_mime_types = {
                "application/pdf": "PDF",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
                "text/plain": "TXT"
            }
            
            if jd_file.type in allowed_mime_types:
                st.success(f"Job Description uploaded: {jd_file.name} ({allowed_mime_types[jd_file.type]})")
            else:
                st.error(f"**Unsupported file type detected!**")
                st.warning(f"File type '{jd_file.type}' is not supported. Please upload PDF, DOCX, or TXT files only.")
                jd_file = None

st.write("")

# Custom CSS for Analyze Button
st.markdown("""
<style>
div[data-testid="stButton"] {
    display: flex;
    justify-content: center;
}

div[data-testid="stButton"] > button {
    background-color: #470047;
    color: white;
    width: 260px;
    height: 55px;
    font-size: 18px;
    border-radius: 10px;
    border: none;
}

div[data-testid="stButton"] > button:hover {
    background-color: #2E002E;
    transform: scale(1.03);
    transition: all 0.2s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Analyze Button
d1, d2, d3, d4, d5 = st.columns(5)
with d3:
    analyze_button = st.button("Analyze Documents")

# Validation
if analyze_button:
    if not resume_file and not jd_file:
        st.error("❌ **Error: No files uploaded!**")
        st.warning("⚠️ Please upload both Resume and Job Description files before analyzing.")
        st.stop()
    elif resume_file and not jd_file:
        st.error("❌ **Error: Job Description is missing!**")
        st.warning("⚠️ Please upload the Job Description file to proceed with the analysis.")
        st.stop()
    elif jd_file and not resume_file:
        st.error("❌ **Error: Resume is missing!**")
        st.warning("⚠️ Please upload the Resume file to proceed with the analysis.")
        st.stop()
    
    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"]
    
    if resume_file.type not in allowed_types:
        st.error(f"❌ **Error: Invalid resume file format!**")
        st.warning(f"⚠️ Resume file type '{resume_file.type}' is not supported.")
        st.stop()
    
    if jd_file.type not in allowed_types:
        st.error(f"❌ **Error: Invalid job description file format!**")
        st.warning(f"⚠️ Job Description file type '{jd_file.type}' is not supported.")
        st.stop()
    
    st.success("✅ Both files uploaded successfully! Starting analysis...")
    st.write("")

# Main Analysis
if analyze_button and jd_file and resume_file:
    
    # Document Preview
    st.subheader("Parsed Document Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if resume_file:
            with st.spinner("Extracting resume text..."):
                resume_text = extract_text(resume_file)
            if resume_text:
                char_count = len(resume_text)
                word_count = len(resume_text.split())
                st.caption(f"📊 Resume: {char_count} characters, {word_count} words")
                st.text_area("Resume Preview", resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""), 
                           height=200, key="resume_preview",
                           help=f"Showing first 2000 characters. Total: {char_count} characters")
    
    with col2:
        if jd_file:
            with st.spinner("Extracting job description text..."):
                jd_text = extract_text(jd_file)
            if jd_text:
                char_count = len(jd_text)
                word_count = len(jd_text.split())
                st.caption(f"📊 Job Description: {char_count} characters, {word_count} words")
                st.text_area("Job Description Preview", jd_text[:2000] + ("..." if len(jd_text) > 2000 else ""), 
                           height=200, key="jd_preview",
                           help=f"Showing first 2000 characters. Total: {char_count} characters")
    
    # Text Cleaning
    cleaned_resume = ""
    cleaned_jd = ""
    resume_experience = None
    jd_experience = None
    
    if resume_text:
        with st.spinner("Cleaning resume text..."):
            cleaned_resume = clean_text(resume_text)
            resume_experience = extract_experience(resume_text)
    
    if jd_text:
        with st.spinner("Cleaning job description text..."):
            cleaned_jd = clean_text(jd_text)
            jd_experience = extract_experience(jd_text)
    
    # Cleaned Output
    st.subheader("Cleaned Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if cleaned_resume:
            st.caption(f"Cleaned Resume: {len(cleaned_resume)} characters")
            st.text_area("Cleaned Resume", cleaned_resume[:2000] + ("..." if len(cleaned_resume) > 2000 else ""), 
                        height=200, key="cleaned_resume",
                        help=f"Showing first 2000 characters. Total: {len(cleaned_resume)} characters")
    
    with col2:
        if cleaned_jd:
            st.caption(f"Cleaned Job Description: {len(cleaned_jd)} characters")
            st.text_area("Cleaned Job Description", cleaned_jd[:2000] + ("..." if len(cleaned_jd) > 2000 else ""), 
                        height=200, key="cleaned_jd",
                        help=f"Showing first 2000 characters. Total: {len(cleaned_jd)} characters")
    
    st.success("✅ Completed: Documents uploaded, parsed, cleaned and previewed successfully.")
    
    # ========================================================================
    # MILESTONE 2: SKILL EXTRACTION USING NLP
    # ========================================================================
    
    st.markdown("""
    <div style="background-color:#470047;padding:20px;border-radius:10px;margin-top:30px;">
        <h2 style="color:white;">Skill Extraction using NLP Module</h2>
        <p style="color:white;">
            Module: Skill Extraction using NLP <br>
            • spaCy and BERT-based pipelines <br>
            • Technical and soft skills identification <br>
            • Structured skill display <br>
            • Optimized for multi-page documents
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Experience Analysis
    st.subheader("Experience Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        if resume_experience:
            st.info(f"📅 Resume Experience: {resume_experience['min_exp']}" + 
                   (f"-{resume_experience['max_exp']}" if resume_experience['max_exp'] else "+") + " years")
        else:
            st.warning("⚠️ No experience information found in resume")
    
    with col2:
        if jd_experience:
            st.info(f"📋 JD Required Experience: {jd_experience['min_exp']}" + 
                   (f"-{jd_experience['max_exp']}" if jd_experience['max_exp'] else "+") + " years")
        else:
            st.warning("⚠️ No experience requirement found in JD")
    
    # Skill Extraction
    with st.spinner("🔍 Extracting skills using NLP... (This may take a moment for longer documents)"):
        resume_technical = extract_technical_skills(cleaned_resume, MASTER_TECHNICAL_SKILLS)
        resume_soft = extract_soft_skills(cleaned_resume, MASTER_SOFT_SKILLS)
        
        # GitHub Analysis
        github_skills = []
        github_analysis = None
        
        detected_github_url = detect_github_in_resume(resume_text)
        
        st.subheader("GitHub Profile Analyzer")
        
        if detected_github_url:
            st.success(f"✅ GitHub profile detected in resume: {detected_github_url}")
            auto_analyze = st.checkbox("Automatically analyze detected GitHub profile", value=True)
            
            if auto_analyze:
                github_url = detected_github_url
            else:
                github_url = st.text_input(
                    "Or enter different GitHub Profile URL",
                    value=detected_github_url,
                    placeholder="https://github.com/username",
                    help="Edit or use a different GitHub profile"
                )
        else:
            st.caption("ℹ️ No GitHub profile detected in resume. You can manually add one below.")
            github_url = st.text_input(
                "Enter GitHub Profile URL (Optional)",
                placeholder="https://github.com/username",
                help="We'll extract technical skills from your public repositories"
            )
        
        if github_url:
            username = extract_github_username(github_url)
            if username:
                with st.spinner(f"🔍 Analyzing GitHub profile for @{username}..."):
                    repos = fetch_github_repos(username)
                    if repos:
                        github_skills = extract_github_skills(repos, MASTER_TECHNICAL_SKILLS)
                        github_analysis = analyze_github_profile(repos)
                        
                        if github_skills:
                            st.success(f"✅ Found {len(github_skills)} technical skills from {len(repos)} repositories!")
                            
                            with st.expander("View Extracted GitHub Skills", expanded=True):
                                st.write(", ".join(github_skills))
                            
                            if github_analysis:
                                with st.expander("GitHub Profile Insights", expanded=True):
                                    insights = github_analysis["insights"]
                                    
                                    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                                    col_i1.metric("Total Repos", insights["total_repos"])
                                    col_i2.metric("Languages", len(insights["languages"]))
                                    col_i3.metric("Documented", f"{insights['documented_projects']}/{insights['total_repos']}")
                                    col_i4.metric("Activity Rate", f"{insights['activity_rate']}%")
                                    
                                    if insights["top_language"]:
                                        st.info(f"🏆 Most used language: **{insights['top_language']}** ({insights['languages'][insights['top_language']]} repos)")
                                    
                                    if insights["languages"]:
                                        st.write("**Language Distribution:**")
                                        lang_text = ", ".join([f"{lang} ({count})" for lang, count in sorted(insights["languages"].items(), key=lambda x: x[1], reverse=True)])
                                        st.caption(lang_text)
                                
                                if github_analysis["suggestions"]:
                                    with st.expander("GitHub Profile Improvement Suggestions", expanded=True):
                                        st.markdown("**Recommendations to enhance your GitHub presence:**")
                                        for suggestion in github_analysis["suggestions"]:
                                            st.warning(f"💡 {suggestion}")
                        else:
                            st.warning("⚠️ No recognizable technical skills found in your repositories")
                            st.info("💡 Tip: Add more descriptive repository names and detailed descriptions")
                    else:
                        st.error("❌ Unable to fetch repositories. Please check the username or try again later.")
            else:
                st.error("❌ Invalid GitHub URL format")
        
        # Merge GitHub skills with resume skills
        combined_technical_skills = sorted(set(resume_technical) | set(github_skills))
        
        resume_skills = {
            "technical": combined_technical_skills,
            "soft": resume_soft
        }
        
        if github_skills:
            st.markdown("---")
            st.subheader("Skill Integration Summary")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Resume Skills", len(resume_technical))
            col_b.metric("GitHub Skills", len(github_skills))
            col_c.metric("Total Unique", len(combined_technical_skills))
            col_d.metric("Skills Added", len(combined_technical_skills) - len(resume_technical))
        
        jd_skills = {
            "technical": extract_technical_skills(cleaned_jd, MASTER_TECHNICAL_SKILLS),
            "soft": extract_soft_skills(cleaned_jd, MASTER_SOFT_SKILLS)
        }
    
    # Display Extracted Skills
    st.subheader("Extracted Skills")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Resume Skills")
        with st.expander("Technical Skills", expanded=True):
            if resume_skills["technical"]:
                st.write(", ".join(resume_skills["technical"]))
                st.caption(f"Total: {len(resume_skills['technical'])} skills")
            else:
                st.info("No technical skills found")
        with st.expander("Soft Skills", expanded=True):
            if resume_skills["soft"]:
                st.write(", ".join(resume_skills["soft"]))
                st.caption(f"Total: {len(resume_skills['soft'])} skills")
            else:
                st.info("No soft skills found")
    
    with col2:
        st.markdown("### JD Skills")
        with st.expander("Technical Skills", expanded=True):
            if jd_skills["technical"]:
                st.write(", ".join(jd_skills["technical"]))
                st.caption(f"Total: {len(jd_skills['technical'])} skills")
            else:
                st.info("No technical skills found")
        with st.expander("Soft Skills", expanded=True):
            if jd_skills["soft"]:
                st.write(", ".join(jd_skills["soft"]))
                st.caption(f"Total: {len(jd_skills['soft'])} skills")
            else:
                st.info("No soft skills found")
    
    # Skill Distribution Charts
    st.subheader("Skill Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_donut_chart(
            len(resume_skills["technical"]),
            len(resume_skills["soft"]),
            "Resume Skill Distribution"
        )
        st.pyplot(fig1, use_container_width=False)
        plt.close(fig1)
    
    with col2:
        fig2 = create_donut_chart(
            len(jd_skills["technical"]),
            len(jd_skills["soft"]),
            "JD Skill Distribution"
        )
        st.pyplot(fig2, use_container_width=False)
        plt.close(fig2)
    
    # Metrics
    metrics = compute_metrics(resume_skills, jd_skills)
    
    st.subheader("Skill Extraction Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="JD Technical Skills", value=metrics["jd_tech"])
    col2.metric(label="JD Soft Skills", value=metrics["jd_soft"])
    col3.metric(label="Total JD Skills", value=metrics["total_jd"])
    col4.metric(label="Basic Match %", value=f'{metrics["match_percent"]}%')
    
    st.success("✅ Completed: Skills extracted successfully using NLP.")
    
    # ========================================================================
    # MILESTONE 3: SKILL GAP ANALYSIS
    # ========================================================================
    
    st.markdown("""
    <div style="background-color:#470047;padding:20px;border-radius:10px;margin-top:30px;">
        <h2 style="color:white;">Skill Gap Analysis and Similarity Matching Module</h2>
        <p style="color:white;">
            • Skill similarity matrix visualization <br>
            • Resume vs JD skill comparison <br>
            • Missing skill identification <br>
            • Multi-page document analysis
        </p>
    </div><br>
    """, unsafe_allow_html=True)
    
    resume_all_skills = resume_skills["technical"] + resume_skills["soft"]
    jd_all_skills = jd_skills["technical"] + jd_skills["soft"]
    
    if resume_all_skills and jd_all_skills:
        with st.spinner("🔍 Building skill gap analysis..."):
            similarity_matrix = build_similarity_matrix(resume_all_skills, jd_all_skills)
            
            st.subheader("Category-wise Skill Match Heatmap")
            fig_heatmap = create_category_heatmap(resume_all_skills, jd_all_skills, categorize_skills)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("⚠️ Insufficient skills to build skill gap analysis")
    
    # Skill Match Classification
    skill_match_result = classify_skill_matches(similarity_matrix, resume_all_skills, jd_all_skills)
    
    # Calculate detailed match
    all_resume_skills = set(resume_skills["technical"]) | set(resume_skills["soft"])
    all_jd_skills = set(jd_skills["technical"]) | set(jd_skills["soft"])
    match_counts = calculate_skill_match(all_resume_skills, all_jd_skills)
    
    # Skill Match Overview
    st.subheader("Skill Match Overview")
    
    left, right = st.columns([1.3, 1])
    
    with left:
        fig = create_skill_distribution_chart(
            match_counts["matched"],
            match_counts["partial"],
            match_counts["missing"]
        )
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    
    with right:
        with st.container():
            r1, r2 = st.columns(2)
            r1.metric("Matched Skills", match_counts["matched"])
            r2.metric("Partially Matched", match_counts["partial"])
            r3, r4 = st.columns(2)
            r3.metric("Missing Skills", match_counts["missing"])
            r4.metric("Avg Match %", f'{match_counts["avg_match"]}%')
    
    # Skill Gap Details
    st.subheader("Skill Gap Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Matched Skills")
        if skill_match_result["matched"]:
            for skill in skill_match_result["matched"]:
                st.success(f"✓ {skill}")
        else:
            st.info("No perfectly matched skills")
    
    with col2:
        st.markdown("### Partial Matches")
        if skill_match_result["partial"]:
            for skill in skill_match_result["partial"]:
                st.warning(f"≈ {skill}")
        else:
            st.info("No partially matched skills")
    
    with col3:
        st.markdown("### Missing Skills")
        if skill_match_result["missing"]:
            for skill in skill_match_result["missing"]:
                st.error(f"✗ {skill}")
        else:
            st.success("No missing skills!")
    
    st.success("✅ Completed: Skill gap analysis completed successfully.")
    
    # ========================================================================
    # MILESTONE 4: DASHBOARD AND REPORT EXPORT
    # ========================================================================
    
    st.markdown("""
    <div style="background-color:#470047;padding:20px;border-radius:10px;margin-top:30px;">
        <h2 style="color:white;">Dashboard and Report Export Module</h2>
        <p style="color:white;">
            Interactive dashboard • Graphs • Multi-format report export • Optimized for multi-page analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Metrics Display
    overall_match = match_counts["avg_match"]
    matched_skills_count = match_counts["matched"]
    missing_skills_count = match_counts["missing"]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Match", f"{overall_match}%")
    c2.metric("Matched Skills", str(matched_skills_count))
    c3.metric("Missing Skills", str(missing_skills_count))
    
    # Categorized Skill Match
    st.subheader("Categorized Skill Match Overview")
    
    jd_categorized = categorize_skills(list(all_jd_skills))
    categories_to_plot = []
    resume_category_scores = []
    jd_category_scores = []
    
    for category, cat_skills in jd_categorized.items():
        matched_in_category = sum(1 for skill in cat_skills if skill in all_resume_skills)
        total_in_category = len(cat_skills)
        if total_in_category > 0:
            categories_to_plot.append(category)
            resume_category_scores.append((matched_in_category / total_in_category) * 100)
            jd_category_scores.append(100)
    
    # Bar Chart
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        name='Your Skills',
        x=categories_to_plot,
        y=resume_category_scores,
        marker_color='#470047',
        text=[f'{score:.0f}%' for score in resume_category_scores],
        textposition='outside'
    ))
    bar_fig.add_trace(go.Bar(
        name='Job Requirements',
        x=categories_to_plot,
        y=jd_category_scores,
        marker_color='#28a745',
        text=['100%'] * len(categories_to_plot),
        textposition='outside'
    ))
    bar_fig.update_layout(
        title="Skill Match by Category",
        barmode="group",
        height=500,
        xaxis_title="Skill Categories",
        yaxis_title="Coverage (%)",
        hovermode='x unified',
        yaxis=dict(range=[0, 120]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    
    # Radar Chart
    radar_categories = categories_to_plot[:6] if len(categories_to_plot) > 6 else categories_to_plot
    radar_resume_scores = resume_category_scores[:6] if len(resume_category_scores) > 6 else resume_category_scores
    radar_jd_scores = [100] * len(radar_categories)
    
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_resume_scores,
        theta=radar_categories,
        fill='toself',
        name='Current Profile',
        line_color='#470047'
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_jd_scores,
        theta=radar_categories,
        fill='toself',
        name='Job Requirements',
        line_color='#28a745'
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Top Skill Categories Comparison",
        height=500,
        showlegend=True
    )
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Sample Skill Proficiency
    st.subheader("Sample Skill Proficiency")
    st.caption("Note: These are sample scores for demonstration. For actual proficiency, use skill assessment tools.")
    
    sample_skills = [("Python", 92), ("Machine Learning", 88), ("SQL", 65)]
    for skill, score in sample_skills:
        col_skill, col_progress = st.columns([1, 4])
        with col_skill:
            st.write(f"**{skill}**")
        with col_progress:
            st.progress(score / 100)
            st.caption(f"{score}%")
    
    # Category-wise Similarity Score Distribution
    st.subheader("Category-wise Similarity Score Distribution")
    
    if resume_all_skills and jd_all_skills and len(similarity_matrix) > 0:
        jd_categorized = categorize_skills(jd_all_skills)
        
        category_similarities = {}
        
        for category, cat_skills in jd_categorized.items():
            cat_scores = []
            for skill in cat_skills:
                if skill in jd_all_skills:
                    j = jd_all_skills.index(skill)
                    column_scores = similarity_matrix[:, j]
                    avg_score = np.mean(column_scores) * 100
                    cat_scores.append(avg_score)
            
            if cat_scores:
                category_similarities[category] = np.mean(cat_scores)
        
        if category_similarities:
            categories = list(category_similarities.keys())
            scores = list(category_similarities.values())
            
            area_fig = go.Figure()
            area_fig.add_trace(go.Scatter(
                x=categories,
                y=scores,
                fill='tozeroy',
                name='Avg Similarity Score',
                line_color='#470047',
                fillcolor='rgba(124, 58, 237, 0.3)',
                mode='lines+markers'
            ))
            
            area_fig.update_layout(
            title="Average Similarity Score by Category",
            xaxis_title="Skill Categories",
            yaxis_title="Similarity Score (%)",
            height=400,
            hovermode='x unified',
            yaxis=dict(
                autorange=True,
                rangemode="tozero",
                ticksuffix="%"
            ))
            
            st.plotly_chart(area_fig, use_container_width=True)
        else:
            st.info("No category data available for similarity analysis")
    else:
        st.info("Upload documents to see similarity score distribution")
    
    # Upskilling Recommendations
    st.subheader("Upskilling Recommendations")
    st.caption("Based on missing and partially matched skills from job description")
    
    if skill_match_result["missing"]:
        st.markdown("**🔴 Priority Skills to Learn (Missing):**")
        for i, skill in enumerate(skill_match_result["missing"][:5], 1):
            st.error(f"{i}. **{skill.title()}** - Not found in your resume")
    
    if skill_match_result["partial"]:
        st.markdown("**🟡 Skills to Strengthen (Partial Match):**")
        for i, skill in enumerate(skill_match_result["partial"][:5], 1):
            st.warning(f"{i}. **{skill.title()}** - Improve proficiency in this area")
    
    if not skill_match_result["missing"] and not skill_match_result["partial"]:
        st.success("🎉 Excellent! You have all the required skills for this job!")
    
    # Report Download
    st.subheader("Report Download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_output = generate_csv_report(skill_match_result, jd_skills)
        st.download_button(
            "⬇️ Download CSV Report",
            csv_output.encode("utf-8"),
            "skill_gap_report.csv",
            "text/csv",
            help="Download skill comparison separated by Technical and Soft Skills"
        )
    
    with col2:
        doc_file = generate_word_report(
            skill_match_result,
            jd_skills,
            overall_match,
            matched_skills_count,
            match_counts
        )
        st.download_button(
            "⬇️ Download DOCX Report",
            doc_file,
            "skill_gap_analysis_report.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            help="Download formatted Word report with complete analysis"
        )
    
    with col3:
        try:
            pdf_bytes = generate_pdf_report(
                skill_match_result,
                jd_skills,
                overall_match,
                matched_skills_count,
                match_counts
            )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="skill_gap_analysis_report.pdf",
                mime="application/pdf",
                help="Download formatted PDF report with visualizations"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
    
    st.success("✅ Completed: Dashboard loaded successfully with export options.")
    
    # Final Notes
    st.markdown("---")
    st.markdown("""
    ### Analysis Complete!
    
    **Next Steps:**
    1. Review the missing skills identified above
    2. Consider upskilling in those areas
    3. Update your resume with newly acquired skills
    4. Download the report for your records
    
    **Pro Tip:** Aim for at least 70% match rate for better job prospects!
    
    **Multi-page Support:** This tool fully supports multi-page resumes and job descriptions in PDF, DOCX, and TXT formats.
    """)

else:
    st.info("ℹ️ Please upload both Resume and Job Description files, then click 'Analyze Documents' to begin.")
    st.markdown("""
    ### Supported File Formats:
    - **PDF** (.pdf) - Multi-page supported
    - **Word Document** (.docx) - Multi-page supported
    - **Text File** (.txt) - Multi-page supported
    
    ### Tips:
    - Multi-page resumes are fully supported - upload documents of any length
    - Ensure your files are properly formatted
    - Include clear skill listings in both documents
    - Experience information should be mentioned as "X years" or "X-Y years"
    - The tool will automatically detect and process all pages
    - **Click the "Analyze Documents" button after uploading both files**
    """)