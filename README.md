# Review Render

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> Analyze and visualize product/service reviews using NLP, sentiment breakdown, keyword trends, and interactive charts in a single Streamlit app.

---

## Overview

ReviewRender takes a dataset of text reviews and runs NLP-based analysis to surface sentiment distribution, common keywords, and patterns across the corpus. Results are displayed in an interactive Streamlit dashboard powered by Plotly charts.

---

## Features

| Feature | Description |
|---|---|
| Sentiment Analysis | Classifies reviews as positive, negative, or neutral using NLTK |
| Keyword Extraction | Identifies high-frequency terms after stopword removal |
| Interactive Charts | Plotly-powered bar, pie, and trend visualizations |
| CSV Support | Load any review dataset from the `data/` directory |
| Streamlit Dashboard | Single-page UI — no frontend setup needed |

---

## Tech Stack

```
Language      : Python 3.10+
UI            : Streamlit
NLP           : NLTK 
Visualization : Plotly
Data          : Pandas
```

---

### Installation

```bash
# Clone the repo
git clone https://github.com/glikith/ReviewRender.git
cd ReviewRender

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

### Run

```bash
streamlit run src/app.py
```

The app will be available at `http://localhost:8501`.

---

## Author

[Gummadi Likith](https://github.com/glikith)
