# 🎓 Personalized Learning Recommendation System
### PRO008 | B.Tech CSE (AI & ML) | Major Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.138-green?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.58-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-latest-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

---

## 📌 Project Overview

An intelligent **LMS add-on** that recommends courses, study materials, and practice problems to students based on their learning history and peer behavior using **Collaborative Filtering (SVD Matrix Factorization)**.

> **Domain:** Recommender Systems / EdTech  
> **Dataset:** MovieLens 100K (100,000 ratings | 943 users | 1,635 items)  
> **Final RMSE:** 0.9827 → **4.24% improvement** over baseline

---

## 🏗️ System Architecture

```
Student Input (ID)
       │
       ▼
┌─────────────────┐        ┌──────────────────────┐
│  Streamlit UI   │──────▶│   FastAPI Backend     │
│   (app.py)      │        │   (api.py)            │
└─────────────────┘        └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │  SVD Matrix Factor.  │
                            │  (svd_model.pkl)     │
                            └──────────────────────┘
                                       │
                                       ▼
                            Top-N Course Recommendations
```

---

## 📊 Results

| Model | RMSE | MAE | Training Time |
|-------|------|-----|---------------|
| Baseline (Item-Average CF) | 1.0262 | 0.8199 | 0.51 sec |
| **SVD Matrix Factorization** | **0.9827** | **0.7802** | **0.12 sec** |
| **Improvement** | **4.24% ↑** | **5.09% ↑** | **76% faster** |

---

## 🗂️ Project Structure

```
PRO008-Learning-Recommender/
│
├── 📓 EDA_Learning_Recommender.ipynb   # Main notebook (EDA + Model)
├── 🚀 api.py                           # FastAPI backend
├── 🎨 app.py                           # Streamlit UI
├── 🤖 svd_model.pkl                    # Trained SVD model
│
├── 📁 ml-100k/                         # MovieLens dataset
│   ├── u.data                          # Ratings
│   ├── u.item                          # Movies/Courses
│   └── u.user                          # User info
│
├── 📊 train.csv                        # Training set (70,000)
├── 📊 val.csv                          # Validation set (10,000)
├── 📊 test.csv                         # Test set (20,000)
├── 📈 eda_plots.png                    # EDA visualizations
└── 📄 README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.14 |
| ML Model | scikit-learn (TruncatedSVD) |
| Data Processing | Pandas, NumPy |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Visualization | Matplotlib |
| Model Storage | Pickle |
| Version Control | Git + GitHub |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/krishRathiDev/PRO008-Learning-Recommender.git
cd PRO008-Learning-Recommender
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn fastapi uvicorn streamlit matplotlib
```

### 3. Run the Notebook
Open `EDA_Learning_Recommender.ipynb` in VS Code and run all cells to train the model.

### 4. Start FastAPI Backend
```bash
uvicorn api:app --reload
# API running at http://127.0.0.1:8000
```

### 5. Start Streamlit UI
```bash
streamlit run app.py
# UI running at http://localhost:8501
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict rating for a student-course pair |
| `POST` | `/recommend` | Get Top-N course recommendations for a student |

### Example — Get Recommendations
```json
POST /recommend
{
  "user_id": 1,
  "top_n": 5
}

Response:
{
  "user_id": 1,
  "recommendations": [
    {"item_id": 850, "predicted_rating": 5.0},
    {"item_id": 1189, "predicted_rating": 5.0},
    ...
  ]
}
```

---

## 📅 Week-wise Progress

| Week | Work Done | Status |
|------|-----------|--------|
| Week 1 | Setup, EDA, Baseline Model (RMSE: 1.0262) | ✅ Complete |
| Week 2 | SVD Matrix Factorization (RMSE: 0.9827) | ✅ Complete |
| Week 3 | FastAPI + Streamlit UI + Explainability | ✅ Complete |
| Week 4 | Documentation + Final Report + Submission | ✅ Complete |

---

## 🧠 Model Details

**Algorithm:** Truncated SVD (Singular Value Decomposition)  
**Approach:** Mean-centered collaborative filtering  
**Formula:** `R ≈ U × Vt + user_mean`

**Hyperparameter Tuning:**

| n_components | RMSE |
|-------------|------|
| 10 | **0.9827** ✅ Best |
| 25 | 0.9865 |
| 50 | 0.9972 |
| 100 | 1.0083 |

---

## 🔮 Future Scope

- [ ] Neural Collaborative Filtering (NCF)
- [ ] Content-Based Filtering using course metadata
- [ ] Hybrid Model (CF + CBF)
- [ ] Cloud deployment (AWS / Heroku)
- [ ] MongoDB integration for real LMS data
- [ ] Real-time student activity tracking

---

## 👨‍💻 Author

**Krish Kumar Rathi**  
B.Tech CSE   
Major Project — PRO008  

---

<p align="center">
  Made with ❤️ | PRO008 — Personalized Learning Recommendation System
</p>
