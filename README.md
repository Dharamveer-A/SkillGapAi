# SkillGapAI

AI-Powered Skill Gap Analysis & Career Insights

## Overview

SkillGapAI is an intelligent tool that analyzes resumes against job descriptions to identify skill gaps, match rates, and provide actionable career recommendations. It leverages NLP (Natural Language Processing) using spaCy to extract technical and soft skills, and provides comprehensive visual analytics.

## Features

- **Multi-format Support**: PDF, DOCX, and TXT file parsing
- **NLP-Powered Skill Extraction**: Uses spaCy for accurate skill identification
- **GitHub Integration**: Automatically analyzes GitHub profiles for additional skills
- **Interactive Visualizations**: Charts, heatmaps, and comparison graphs
- **Detailed Analytics**: Category-wise skill matching and gap analysis
- **Multi-format Reports**: Export as CSV, DOCX, or PDF
- **Upskilling Recommendations**: Personalized learning suggestions

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/skillgapai.git
   cd skillgapai
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Download spaCy model**
```bash
   python -m spacy download en_core_web_trf
```

4. **Add tutorial video (optional)**
   - Place your `tutorial.mov` file in the `assets/` folder

5. **Run the application**
```bash
   streamlit run app.py
```

## Project Structure
```
skillgapai/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── assets/                     # Static files
│   └── tutorial.mov           # Tutorial video
├── data/                       # Data files
│   └── skills_list.py         # Master skill databases
├── modules/                    # Python modules
│   ├── __init__.py
│   ├── document_processing.py  # File extraction and cleaning
│   ├── nlp_processing.py       # Skill extraction with NLP
│   ├── github_analyzer.py      # GitHub profile analysis
│   ├── visualization.py        # Chart generation
│   ├── report_generator.py     # Report export functionality
│   └── utils.py                # Helper functions
└── notebooks/                  # Jupyter notebooks (optional)
    └── analysis_test.ipynb
```

## Usage

1. **Upload Documents**
   - Upload your resume (PDF/DOCX/TXT)
   - Upload the job description (PDF/DOCX/TXT)

2. **Click Analyze**
   - The system will extract and clean text
   - NLP models will identify skills
   - GitHub profile (if detected) will be analyzed

3. **Review Results**
   - View matched, partial, and missing skills
   - Explore interactive visualizations
   - Check category-wise skill gaps

4. **Download Reports**
   - Export analysis as CSV, DOCX, or PDF
   - Share with career advisors or recruiters

## Technologies Used

- **Streamlit**: Web application framework
- **spaCy**: NLP and skill extraction
- **PDFPlumber**: PDF text extraction
- **python-docx**: DOCX file handling
- **Plotly**: Interactive visualizations
- **Matplotlib**: Static charts
- **FPDF**: PDF report generation
- **Requests**: GitHub API integration

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- spaCy for NLP capabilities
- Streamlit for the web framework
- Open source community for various libraries

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made by Amrutha varshini, Rohitha panchumukhi, Dharamveer**