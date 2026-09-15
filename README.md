# 🎬 CineMatch — Advanced Movie Recommendation Simulator

CineMatch is an interactive movie recommendation project designed to demonstrate how recommendation systems work through a modern and user-friendly interface.

## ✨ Features

* 🎯 Content-Based Recommendation (TF-IDF & Cosine Similarity)
* 🤝 Collaborative Filtering (Item-Item User Interaction Co-occurrence)
* ⚡ Hybrid Ensemble Engine (Balanced Content & Collaborative Blending)
* 🌟 Top Match Recognition Badge
* ⭐ Interactive In-Card & Seed Rating System
* 📜 Dynamic Rating History with Instant Feed Personalization
* 🔎 Movie Search & Autocomplete Suggestions
* 🧮 Live Mathematical Explanations Rendered with MathJax
* 🚀 FastAPI Backend with Direct Web Serving at `/`

## 🛠️ Technologies

* **Backend & ML**: Python, FastAPI, Uvicorn, Scikit-learn, Pandas, NumPy
* **Frontend**: Modern HTML5, CSS3 Glassmorphism, Vanilla JavaScript
* **Math Rendering**: MathJax LaTeX Engine
* **Testing**: Python `unittest` & `fastapi.testclient`

## 🎯 Purpose

The goal of CineMatch is to demonstrate and simulate core recommendation concepts in an interactive setting:

* **Content-Based Filtering**: Vectorizing movie plot and genre metadata using TF-IDF and calculating angular cosine similarity.
* **Collaborative Filtering**: Modeling item-item interactions from user rating distributions to discover behavioral correlations.
* **Hybrid Ensemble Filtering**: Fusing content features and collaborative patterns to resolve cold-start limitations and boost recommendation diversity.
* **Personalized User Profiling**: Dynamic preference weighting with centered rating vectors.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/aizazahmad736/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the application
Run the FastAPI application directly:
```bash
python app.py
```
Then open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### 4. (Optional) Run automated tests
```bash
python -m unittest test_app.py
```

## 📂 Project Structure

```text
movie-recommender/
│
├── app.py                     # FastAPI server with ML inference & root UI serving
├── generate_data.py           # Synthetic dataset generator (movies & ratings)
├── train.py                   # Offline precomputation & recommendations exporter
├── test_app.py                # Automated integration & unit test suite
├── index.html                 # Interactive single-page UI simulator
├── requirements.txt           # Python backend dependencies
├── data/
│   ├── movies.csv             # 50 curated movies across 5 genres
│   ├── ratings.csv            # Synthetic user-item rating records
│   └── recommendations.json   # Exported recommendations
├── README.md                  # Project documentation
└── .github/
    └── workflows/
        └── cinematch-check.yml # CI pipeline with automated testing
```

## 👨‍💻 Author

**Aizaz Ahmad**

AI Engineer & Python Developer | ML & Generative AI

GitHub: https://github.com/aizazahmad736

---

⭐ If you find this project useful, consider giving the repository a star!
