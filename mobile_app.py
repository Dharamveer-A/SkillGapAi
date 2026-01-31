"""
SkillGapAI - Ultimate Edition
Mobile-Optimized | Dynamic Weights | Experience Points | GitHub Analytics
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# Import all modules (Ensure modules/__init__.py is updated!)
from modules import (
    extract_text, clean_text, extract_experience,
    extract_technical_skills, extract_soft_skills, categorize_skills,
    detect_github_in_resume, extract_github_username, fetch_github_repos, 
    extract_github_skills, analyze_github_profile,
    create_donut_chart, create_skill_distribution_chart,
    generate_csv_report, generate_word_report, generate_pdf_report,
    compute_metrics, build_similarity_matrix, classify_skill_matches, calculate_skill_match
)
from data.skills_list import MASTER_TECHNICAL_SKILLS, MASTER_SOFT_SKILLS

# 1. Page Configuration
st.set_page_config(page_title="SkillGapAI", layout="centered", page_icon="📱")

# 2. Custom CSS (Mobile & UI Polish)
st.markdown("""
<style>
    /* Large touch-friendly buttons */
    .stButton>button {
        width: 100%; height: 60px; font-size: 20px; border-radius: 12px;
        background-color: #470047; color: white; border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1); transition: all 0.3s;
    }
    .stButton>button:hover { background-color: #6a006a; transform: translateY(-2px); color: white; }
    
    /* Clean Headers */
    h1, h2, h3 { text-align: center; }
    
    /* Hide Default Menu */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    /* Expander Font */
    .streamlit-expanderHeader { font-size: 18px; font-weight: bold; color: #333; }
    
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# 3. Session State
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = {}

def go_to_step(step):
    st.session_state.step = step
    st.rerun()

# ============================================================================
# STEP 1: UPLOAD SCREEN
# ============================================================================
if st.session_state.step == 1:
    st.title("🚀 SkillGapAI")
    st.markdown("### Step 1: Upload Documents")
    st.info("Upload your Resume and the Job Description to detect skill gaps.")
    
    resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])
    jd_file = st.file_uploader("💼 Upload Job Description", type=["pdf", "docx", "txt"])
    
    st.write("") 
    if resume_file and jd_file:
        if st.button("Analyze Now ➔"):
            with st.spinner("Reading documents..."):
                st.session_state.data['resume_text'] = extract_text(resume_file)
                st.session_state.data['jd_text'] = extract_text(jd_file)
                go_to_step(2)
    else:
        st.caption("Please upload both files to proceed.")

# ============================================================================
# STEP 2: PROCESSING LOGIC
# ============================================================================
elif st.session_state.step == 2:
    st.title("⚙️ Analyzing...")
    
    # Progress Bar (Visual Feedback)
    my_bar = st.progress(0)
    
    try:
        # 1. Clean Text
        my_bar.progress(20, text="Cleaning text...")
        cleaned_resume = clean_text(st.session_state.data['resume_text'])
        cleaned_jd = clean_text(st.session_state.data['jd_text'])
        
        # 2. Extract Basic Skills
        my_bar.progress(40, text="Extracting skills...")
        resume_tech = extract_technical_skills(cleaned_resume, MASTER_TECHNICAL_SKILLS)
        resume_soft = extract_soft_skills(cleaned_resume, MASTER_SOFT_SKILLS)
        jd_tech = extract_technical_skills(cleaned_jd, MASTER_TECHNICAL_SKILLS)
        jd_soft = extract_soft_skills(cleaned_jd, MASTER_SOFT_SKILLS)
        
        # 3. GitHub Integration (The New Part)
        my_bar.progress(60, text="Checking GitHub...")
        github_skills = []
        github_stats = None # Placeholder
        
        github_url = detect_github_in_resume(st.session_state.data['resume_text'])
        if github_url:
            username = extract_github_username(github_url)
            if username:
                try:
                    # A. Fetch Skills from Repos
                    repos = fetch_github_repos(username)
                    if repos:
                        github_skills = extract_github_skills(repos, MASTER_TECHNICAL_SKILLS)
                    
                    # B. Analyze Profile Stats (Stars, Followers)
                    github_stats = analyze_github_profile(username)
                    st.session_state.data['gh_stats'] = github_stats # Store for display
                    
                except Exception as e:
                    print(f"GitHub Error: {e}")
        
        # Merge GitHub skills into Resume skills
        final_resume_tech = sorted(list(set(resume_tech + github_skills)))
        
        # Store Structured Data
        st.session_state.data['resume_skills'] = {"technical": final_resume_tech, "soft": resume_soft}
        st.session_state.data['jd_skills'] = {"technical": jd_tech, "soft": jd_soft}
        
        # 4. Compute Metrics (Dynamic Weighting)
        my_bar.progress(80, text="Calculating scores...")
        metrics = compute_metrics(
            st.session_state.data['resume_skills'], 
            st.session_state.data['jd_skills'],
            st.session_state.data['resume_text'], 
            st.session_state.data['jd_text']
        )
        st.session_state.data['metrics'] = metrics
        
        my_bar.progress(100, text="Done!")
        go_to_step(3)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        if st.button("Try Again"): go_to_step(1)

# ============================================================================
# STEP 3: RESULTS DASHBOARD
# ============================================================================
elif st.session_state.step == 3:
    metrics = st.session_state.data['metrics']
    
    # 1. FINAL SCORE CARD
    st.markdown(f"""
    <div style="background-color:#f0f2f6;padding:20px;border-radius:15px;text-align:center;margin-bottom:20px;border: 1px solid #d1d1d1;">
        <h2 style="margin:0;color:#470047;font-size:42px;">{metrics['final_score']}%</h2>
        <p style="margin:0;font-size:18px;color:#555;">Final Match Score</p>
        <p style="font-size:12px;color:#888;">{metrics['weight_text']}</p>
        <hr style="margin:10px 0;">
        <p style="margin:0;font-size:14px;">Skills Matched: <b>{metrics['matched']}</b> / <b>{metrics['total_jd']}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. EXPERIENCE CARD
    c1, c2 = st.columns(2)
    with c1: st.metric("Your Experience", f"{metrics['res_years']} Years")
    with c2: st.metric("Required", f"{metrics['jd_years']}+ Years")

    if metrics['res_years'] < metrics['jd_years']:
        st.error(f"⚠️ Gap: {metrics['jd_years'] - metrics['res_years']} year(s) short.")
    elif metrics['res_years'] >= metrics['jd_years'] + 3:
        st.success(f"🌟 Impressive! Overqualified (+{metrics['res_years'] - metrics['jd_years']} yrs).")
    else:
        st.success("✅ Experience requirement met!")

    # 3. GITHUB IMPACT CARD (New!)
    # Only shows if we found a GitHub profile
    if 'gh_stats' in st.session_state.data and st.session_state.data['gh_stats']:
        gh = st.session_state.data['gh_stats']
        with st.expander("🐙 GitHub Profile Analysis", expanded=True):
            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Total Stars ⭐", gh.get('total_stars', 0))
            gc2.metric("Followers 👥", gh.get('followers', 0))
            gc3.metric("Public Repos 📂", gh.get('public_repos', 0))
            
            # Show Top Languages
            if gh.get('top_languages'):
                st.caption(f"**Top Languages:** {', '.join(gh['top_languages'])}")
    
    # 4. VISUALIZATIONS
    st.markdown("### 📊 Visualizations")
    with st.expander("Show Skill Mix (Donut Chart)"):
        fig_donut = create_donut_chart(
            len(st.session_state.data['resume_skills']['technical']),
            len(st.session_state.data['resume_skills']['soft']),
            "My Skill Breakdown"
        )
        st.pyplot(fig_donut, use_container_width=True)
    
    with st.expander("Show Match Distribution (Bar Chart)"):
        fig_bar = create_skill_distribution_chart(
            metrics['matched'], metrics['partial'], metrics['missing']
        )
        st.pyplot(fig_bar, use_container_width=True)
        
    # 5. DETAILED LISTS
    st.markdown("### 📝 Skill Details")
    r_all = st.session_state.data['resume_skills']['technical'] + st.session_state.data['resume_skills']['soft']
    j_all = st.session_state.data['jd_skills']['technical'] + st.session_state.data['jd_skills']['soft']
    
    matrix = build_similarity_matrix(r_all, j_all)
    details = classify_skill_matches(matrix, r_all, j_all)
    
    with st.expander("✅ Matched Skills", expanded=False):
        if details['matched']:
            for s in details['matched']: st.markdown(f"- **{s}**")
        else: st.info("No exact matches found.")

    with st.expander(f"⚖️ Partially Matched (+{metrics['partial_points']} pts)", expanded=False):
        if details['partial']:
            st.info(f"Related skills found (+{metrics['partial_points']} pts):")
            for s in details['partial']: st.markdown(f"- {s}")
        else: st.write("No partial matches found.")

    with st.expander("⚠️ Missing Skills (Priority)", expanded=True):
        if details['missing']:
            st.warning("Consider adding these to your resume:")
            for s in details['missing']: st.markdown(f"- {s}")
        else: st.success("Great job! No key skills are missing.")

    # 6. REPORT DOWNLOAD
    st.markdown("---")
    st.subheader("📥 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            csv_output = generate_csv_report(details, st.session_state.data['jd_skills'])
            st.download_button(
                label="Download CSV 📊",
                data=csv_output.encode("utf-8"),
                file_name="SkillGap_Report.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"CSV Generation Failed: {e}")
    
    with col2:
        try:
            doc_file = generate_word_report(
                details,
                st.session_state.data['jd_skills'],
                metrics['final_score'],
                metrics['matched'],
                metrics
            )
            st.download_button(
                label="Download DOCX 📄",
                data=doc_file,
                file_name="SkillGap_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except Exception as e:
            st.error(f"DOCX Generation Failed: {e}")
    
    with col3:
        try:
            pdf_bytes = generate_pdf_report(
                details, st.session_state.data['jd_skills'], 
                metrics['final_score'], 
                metrics['matched'], metrics
            )
            st.download_button(
                label="Download PDF 📑",
                data=pdf_bytes,
                file_name="SkillGap_Report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF Generation Failed: {e}")

    # 7. RESET
    st.write("")
    if st.button("Start New Analysis 🔄"):
        st.session_state.data = {}
        go_to_step(1)