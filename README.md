# CineMatch: Advanced Movie Recommendation Engine Simulator

An interactive Machine Learning web simulation illustrating two core styles of recommendation models: **Content-Based Filtering** (using metadata TF-IDF representation) and **Collaborative Filtering** (using user-item rating vectors).

---

## 📐 Mathematical Formulas

### 1. Term Frequency-Inverse Document Frequency (TF-IDF)
For Content-Based filtering, movie genres and descriptions are transformed into vectors using TF-IDF weights:

$$\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$

$$\text{IDF}(t, D) = \log\left(\frac{1 + N}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

---

### 2. Cosine Similarity
Cosine similarity calculates the cosine of the angle between two multi-dimensional vectors ($\vec{A}$ and $\vec{B}$):

$$\text{Similarity}(\vec{A}, \vec{B}) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

*   **Content-Based**: Vectors $\vec{A}$ and $\vec{B}$ represent the TF-IDF vocabulary weights for each movie.
*   **Collaborative Filtering**: Vectors $\vec{A}$ and $\vec{B}$ represent rating values given by users $1 \dots 200$ to those respective movies.

---

## 📂 Project Architecture

*   [generate_data.py](file:///c:/Users/Aizaz%20Ahmad/Desktop/Antigravity-Projects/movie-recommender/generate_data.py): Synthesizes 50 custom movies across genres (Sci-Fi, Action, Romance, Comedy, Thriller) and patterns ratings for 200 users.
*   [train.py](file:///c:/Users/Aizaz%20Ahmad/Desktop/Antigravity-Projects/movie-recommender/train.py): Pre-calculates Content-Based similarities (using TF-IDF) and Collaborative similarities (using User-Item Matrix pivoting) into `recommendations.json`.
*   [index.html](file:///c:/Users/Aizaz%20Ahmad/Desktop/Antigravity-Projects/movie-recommender/index.html): An advanced dark-themed frontend utilizing glassmorphic styles, interactive slider cards, and MathJax formula rendering.

---

## 🚀 How to Run

1.  **Generate Data and Train the Models**:
    ```bash
    python generate_data.py
    python train.py
    ```

2.  **View the Simulation**:
    Serve the project folder using a local web server (e.g., Python's HTTP server):
    ```bash
    python -m http.server 8000
    ```
    And navigate to [http://localhost:8000](http://localhost:8000) in your browser.
