"""
Visualization Module
Creates charts and graphs for skill analysis
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def create_donut_chart(tech_count, soft_count, title):
    """Create donut chart for skill distribution"""
    labels = ["Technical Skills", "Soft Skills"]
    sizes = [tech_count, soft_count]
    if sum(sizes) == 0:
        sizes = [1, 1]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    wedges, _, _ = ax.pie(
        sizes,
        startangle=90,
        autopct="%1.0f%%",
        radius=1,
        wedgeprops=dict(width=0.4, edgecolor="white"),
        colors=['#470047', '#fd7e14']
    )
    ax.set(aspect="equal")
    ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
    ax.legend(
        wedges,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        frameon=False
    )
    return fig


def create_category_heatmap(resume_skills, jd_skills, categorize_func):
    """Plot category-wise skill match heatmap"""
    jd_categorized = categorize_func(jd_skills)
    resume_categorized = categorize_func(resume_skills)
    
    common_categories = set(jd_categorized.keys()) & set(resume_categorized.keys())
    
    if not common_categories:
        return None
    
    categories = sorted(common_categories)
    match_data = []
    
    for category in categories:
        jd_cat_skills = jd_categorized[category]
        resume_cat_skills = set(resume_categorized[category])
        
        matched = 0
        partial = 0
        missing = 0
        
        for jd_skill in jd_cat_skills:
            if jd_skill in resume_cat_skills:
                matched += 1
            else:
                found_partial = False
                for res_skill in resume_cat_skills:
                    if jd_skill in res_skill or res_skill in jd_skill:
                        partial += 1
                        found_partial = True
                        break
                if not found_partial:
                    missing += 1
        
        total = len(jd_cat_skills)
        
        match_data.append({
            'Category': category,
            'Matched': matched,
            'Partial': partial,
            'Missing': missing,
            'Total': total
        })
    
    fig = go.Figure()
    
    categories_list = [d['Category'] for d in match_data]
    
    fig.add_trace(go.Bar(
        name='Matched',
        x=categories_list,
        y=[d['Matched'] for d in match_data],
        marker_color='#28a745',
        text=[d['Matched'] for d in match_data],
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>Matched: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Partial Match',
        x=categories_list,
        y=[d['Partial'] for d in match_data],
        marker_color='#ffc107',
        text=[d['Partial'] for d in match_data],
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>Partial: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Missing',
        x=categories_list,
        y=[d['Missing'] for d in match_data],
        marker_color='#dc3545',
        text=[d['Missing'] for d in match_data],
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>Missing: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Category-wise Skill Match Analysis (Stacked)",
        barmode='stack',
        height=500,
        xaxis_title="Skill Categories",
        yaxis_title="Number of Skills",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        showlegend=True
    )
    
    return fig


def create_skill_distribution_chart(matched, partial, missing):
    """Create pie chart for skill match distribution"""
    labels = ["Matched", "Partially Matched", "Missing"]
    sizes = [matched, partial, missing]
    colors = ["#28a745", "#fd7e14", "#dc3545"]
    if sum(sizes) == 0:
        sizes = [1, 1, 1]
    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.75])
    wedges, _, autotexts = ax.pie(
        sizes,
        startangle=90,
        autopct="%1.0f%%",
        radius=1,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white")
    )
    for autotext in autotexts:
        autotext.set_color('white')  # Changed from black to white for better contrast
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    ax.set(aspect="equal")
    ax.set_title("Skill Match Distribution", pad=8, fontsize=12, fontweight='bold')
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    return fig