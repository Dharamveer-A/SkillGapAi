"""
Report Generator Module
Generates reports in CSV, DOCX, and PDF formats
"""

import io
import pandas as pd
from docx import Document
from fpdf import FPDF


def generate_csv_report(skill_match_result, jd_skills):
    """Generate CSV report separating technical and soft skills"""
    tech_matched = [s for s in skill_match_result['matched'] if s in jd_skills["technical"]]
    tech_partial = [s for s in skill_match_result['partial'] if s in jd_skills["technical"]]
    tech_missing = [s for s in skill_match_result['missing'] if s in jd_skills["technical"]]
    
    soft_matched = [s for s in skill_match_result['matched'] if s in jd_skills["soft"]]
    soft_partial = [s for s in skill_match_result['partial'] if s in jd_skills["soft"]]
    soft_missing = [s for s in skill_match_result['missing'] if s in jd_skills["soft"]]
    
    max_tech_len = max(len(tech_matched), len(tech_partial), len(tech_missing), 1)
    tech_matched += [''] * (max_tech_len - len(tech_matched))
    tech_partial += [''] * (max_tech_len - len(tech_partial))
    tech_missing += [''] * (max_tech_len - len(tech_missing))
    
    max_soft_len = max(len(soft_matched), len(soft_partial), len(soft_missing), 1)
    soft_matched += [''] * (max_soft_len - len(soft_matched))
    soft_partial += [''] * (max_soft_len - len(soft_partial))
    soft_missing += [''] * (max_soft_len - len(soft_missing))
    
    tech_df = pd.DataFrame({
        "Technical - Matched": tech_matched,
        "Technical - Partially Matched": tech_partial,
        "Technical - Missing": tech_missing
    })
    
    soft_df = pd.DataFrame({
        "Soft Skills - Matched": soft_matched,
        "Soft Skills - Partially Matched": soft_partial,
        "Soft Skills - Missing": soft_missing
    })
    
    csv_output = tech_df.to_csv(index=False)
    csv_output += "\n"
    csv_output += soft_df.to_csv(index=False)
    
    return csv_output


def generate_word_report(skill_match_result, jd_skills, overall_match, matched_skills_count, match_counts):
    """Generate DOCX report"""
    doc = Document()
    title = doc.add_heading('Skill Gap Analysis Report', 0)
    title.alignment = 1
    doc.add_heading('Executive Summary', 1)
    summary_table = doc.add_table(rows=4, cols=2)
    summary_table.style = 'Light Grid Accent 1'
    summary_data = [
        ('Overall Match Rate', f'{overall_match}%'),
        ('Matched Skills', str(matched_skills_count)),
        ('Partially Matched Skills', str(match_counts["partial"])),
        ('Missing Skills', str(match_counts["missing"]))
    ]
    for i, (label, value) in enumerate(summary_data):
        summary_table.rows[i].cells[0].text = label
        summary_table.rows[i].cells[1].text = value
    doc.add_paragraph()
    
    # Technical Skills Section
    doc.add_heading('Technical Skills Breakdown', 1)
    
    tech_matched = [s for s in skill_match_result['matched'] if s in jd_skills["technical"]]
    tech_partial = [s for s in skill_match_result['partial'] if s in jd_skills["technical"]]
    tech_missing = [s for s in skill_match_result['missing'] if s in jd_skills["technical"]]
    
    doc.add_heading('Matched Technical Skills', 2)
    if tech_matched:
        for skill in tech_matched:
            doc.add_paragraph(skill, style='List Bullet')
    else:
        doc.add_paragraph('No perfectly matched technical skills found.')
    
    doc.add_heading('Partially Matched Technical Skills', 2)
    if tech_partial:
        for skill in tech_partial:
            doc.add_paragraph(skill, style='List Bullet')
    else:
        doc.add_paragraph('No partially matched technical skills found.')
    
    doc.add_heading('Missing Technical Skills', 2)
    if tech_missing:
        for skill in tech_missing:
            doc.add_paragraph(skill, style='List Bullet')
    else:
        doc.add_paragraph('No missing technical skills!')
    
    # Soft Skills Section
    doc.add_heading('Soft Skills Breakdown', 1)
    
    soft_matched = [s for s in skill_match_result['matched'] if s in jd_skills["soft"]]
    soft_partial = [s for s in skill_match_result['partial'] if s in jd_skills["soft"]]
    soft_missing = [s for s in skill_match_result['missing'] if s in jd_skills["soft"]]
    
    doc.add_heading('Matched Soft Skills', 2)
    if soft_matched:
        for skill in soft_matched:
            doc.add_paragraph(skill, style='List Bullet')
    else:
        doc.add_paragraph('No perfectly matched soft skills found.')
    
    doc.add_heading('Partially Matched Soft Skills', 2)
    if soft_partial:
        for skill in soft_partial:
            doc.add_paragraph(skill, style='List Bullet')
    else:
        doc.add_paragraph('No partially matched soft skills found.')
    
    doc.add_heading('Missing Soft Skills', 2)
    if soft_missing:
        for skill in soft_missing:
            doc.add_paragraph(skill, style='List Bullet')
    else:
        doc.add_paragraph('No missing soft skills!')
    
    # Recommendations
    doc.add_heading('Recommendations', 1)
    doc.add_paragraph(
        f'Your current match rate is {overall_match}%. '
        f'To improve your candidacy, consider focusing on the missing skills listed above.'
    )
    
    if overall_match >= 70:
        doc.add_paragraph('You have a strong skill match for this position!', style='List Bullet')
    else:
        doc.add_paragraph('Focus on upskilling in the missing areas to increase your match rate.', style='List Bullet')
    
    footer = doc.add_paragraph(f'Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}')
    footer.alignment = 1
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def generate_pdf_report(skill_match_result, jd_skills, overall_match, matched_skills_count, match_counts):
    """Generate PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "Skill Gap Analysis Report", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.ln(2)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(90, 8, "Overall Match Rate:", 1, 0, fill=True)
    pdf.cell(90, 8, f"{overall_match}%", 1, 1)
    pdf.cell(90, 8, "Matched Skills:", 1, 0, fill=True)
    pdf.cell(90, 8, str(matched_skills_count), 1, 1)
    pdf.cell(90, 8, "Partially Matched Skills:", 1, 0, fill=True)
    pdf.cell(90, 8, str(match_counts['partial']), 1, 1)
    pdf.cell(90, 8, "Missing Skills:", 1, 0, fill=True)
    pdf.cell(90, 8, str(match_counts['missing']), 1, 1)
    pdf.ln(10)
    
    # Match Rate Visualization
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Match Rate Visualization", ln=True)
    pdf.set_font("Arial", '', 10)
    bar_width = 170
    bar_height = 15
    x_start = 20
    y_start = pdf.get_y()
    pdf.set_fill_color(220, 220, 220)
    pdf.rect(x_start, y_start, bar_width, bar_height, 'F')
    filled_width = (overall_match / 100) * bar_width
    if overall_match >= 70:
        pdf.set_fill_color(40, 167, 69)
    elif overall_match >= 50:
        pdf.set_fill_color(253, 126, 20)
    else:
        pdf.set_fill_color(220, 53, 69)
    pdf.rect(x_start, y_start, filled_width, bar_height, 'F')
    pdf.set_draw_color(100, 100, 100)
    pdf.rect(x_start, y_start, bar_width, bar_height, 'D')
    pdf.set_xy(x_start + bar_width/2 - 10, y_start + 3)
    pdf.set_font("Arial", 'B', 11)
    if overall_match > 30:
        pdf.set_text_color(255, 255, 255)
    else:
        pdf.set_text_color(0, 0, 0)
    pdf.cell(20, 8, f"{overall_match}%", align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)
    
    # Skills Distribution Chart
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Skills Distribution", ln=True)
    pdf.ln(2)
    matched = match_counts['matched']
    partial = match_counts['partial']
    missing = match_counts['missing']
    total_skills = matched + partial + missing
    if total_skills > 0:
        chart_width = 170
        chart_height = 40
        x_chart = 20
        y_chart = pdf.get_y()
        matched_width = (matched / total_skills) * chart_width
        partial_width = (partial / total_skills) * chart_width
        missing_width = (missing / total_skills) * chart_width
        if matched > 0:
            pdf.set_fill_color(40, 167, 69)
            pdf.rect(x_chart, y_chart, matched_width, chart_height, 'F')
        if partial > 0:
            pdf.set_fill_color(253, 126, 20)
            pdf.rect(x_chart + matched_width, y_chart, partial_width, chart_height, 'F')
        if missing > 0:
            pdf.set_fill_color(220, 53, 69)
            pdf.rect(x_chart + matched_width + partial_width, y_chart, missing_width, chart_height, 'F')
        pdf.set_draw_color(100, 100, 100)
        pdf.rect(x_chart, y_chart, chart_width, chart_height, 'D')
        pdf.ln(chart_height + 5)
        pdf.set_font("Arial", '', 9)
        legend_x = 20
        legend_y = pdf.get_y()
        pdf.set_fill_color(40, 167, 69)
        pdf.rect(legend_x, legend_y, 5, 5, 'F')
        pdf.set_xy(legend_x + 7, legend_y - 1)
        pdf.cell(40, 6, f"Matched ({matched})")
        pdf.set_fill_color(253, 126, 20)
        pdf.rect(legend_x + 50, legend_y, 5, 5, 'F')
        pdf.set_xy(legend_x + 57, legend_y - 1)
        pdf.cell(40, 6, f"Partial ({partial})")
        pdf.set_fill_color(220, 53, 69)
        pdf.rect(legend_x + 100, legend_y, 5, 5, 'F')
        pdf.set_xy(legend_x + 107, legend_y - 1)
        pdf.cell(40, 6, f"Missing ({missing})")
        pdf.ln(12)
    
    # Technical Skills Section
    tech_matched = [s for s in skill_match_result['matched'] if s in jd_skills["technical"]]
    tech_partial = [s for s in skill_match_result['partial'] if s in jd_skills["technical"]]
    tech_missing = [s for s in skill_match_result['missing'] if s in jd_skills["technical"]]
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(40, 167, 69)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "Technical Skills - Matched", 0, 1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    if tech_matched:
        for skill in tech_matched:
            clean_skill = skill.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, clean_skill, 0, 1)
    else:
        pdf.cell(0, 6, "No technical skills matched.", 0, 1)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(253, 126, 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "Technical Skills - Partially Matched", 0, 1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    if tech_partial:
        for skill in tech_partial:
            clean_skill = skill.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, clean_skill, 0, 1)
    else:
        pdf.cell(0, 6, "No technical skills partially matched.", 0, 1)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(220, 53, 69)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "Technical Skills - Missing (Priority)", 0, 1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    if tech_missing:
        for skill in tech_missing:
            clean_skill = skill.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, clean_skill, 0, 1)
    else:
        pdf.cell(0, 6, "No missing technical skills!", 0, 1)
    pdf.ln(5)
    
    # Soft Skills Section
    soft_matched = [s for s in skill_match_result['matched'] if s in jd_skills["soft"]]
    soft_partial = [s for s in skill_match_result['partial'] if s in jd_skills["soft"]]
    soft_missing = [s for s in skill_match_result['missing'] if s in jd_skills["soft"]]
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(40, 167, 69)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "Soft Skills - Matched", 0, 1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    if soft_matched:
        for skill in soft_matched:
            clean_skill = skill.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, clean_skill, 0, 1)
    else:
        pdf.cell(0, 6, "No soft skills matched.", 0, 1)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(253, 126, 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "Soft Skills - Partially Matched", 0, 1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    if soft_partial:
        for skill in soft_partial:
            clean_skill = skill.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, clean_skill, 0, 1)
    else:
        pdf.cell(0, 6, "No soft skills partially matched.", 0, 1)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(220, 53, 69)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "Soft Skills - Missing", 0, 1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    if soft_missing:
        for skill in soft_missing:
            clean_skill = skill.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, clean_skill, 0, 1)
    else:
        pdf.cell(0, 6, "No missing soft skills!", 0, 1)
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Recommendations", 0, 1)
    pdf.set_font("Arial", '', 10)
    recommendation = f"Your current match rate is {overall_match}%. "
    if overall_match >= 70:
        recommendation += "You have a strong skill match for this position! Continue to maintain and update these skills."
    elif overall_match >= 50:
        recommendation += "You have a moderate skill match. Focus on developing the missing skills and strengthening partially matched skills to improve your candidacy."
    else:
        recommendation += "There is significant room for improvement. Prioritize learning the missing skills listed above to increase your match rate to 70% or above."
    pdf.multi_cell(0, 6, recommendation)
    pdf.ln(3)
    
    # Action Items
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Action Items:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(5, 6, '', 0, 0)
    pdf.cell(0, 6, "1. Focus on acquiring the missing skills through courses and certifications", ln=True)
    pdf.cell(5, 6, '', 0, 0)
    pdf.cell(0, 6, "2. Strengthen partially matched skills through practical projects", ln=True)
    pdf.cell(5, 6, '', 0, 0)
    pdf.cell(0, 6, "3. Update your resume once you acquire new skills", ln=True)
    pdf.cell(5, 6, '', 0, 0)
    pdf.cell(0, 6, "4. Target positions with 70%+ match rate for better success", ln=True)
    pdf.ln(5)
    
    # Footer
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(0, 10, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'C')
    
    return pdf.output(dest="S").encode("latin-1")