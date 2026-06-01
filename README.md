# 🌍 Eco-Urban Expansion Planning & Sustainable Mobility System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hosting](https://img.shields.io/badge/Render-Live-brightgreen)](https://render.com)

An AI-powered decision support system that evaluates regional terrain data to determine urban sustainability metrics and flood vulnerability hazards. By parsing geospatial data on-the-fly from OpenStreetMap, the platform uses a trained Random Forest Classifier alongside deterministic environmental algorithms to grade regions for sustainable development.

[🌐 View Live App on Render](https://eco-urban-expansion.onrender.com)

---

## 🚀 Key Features

* **Real-Time Geospatial Fetching:** Dynamically downloads road networks, infrastructure layouts, and water bodies directly from OpenStreetMap using `OSMnx`.
* **Predictive AI Engine:** Leverages a custom-trained **Random Forest Machine Learning Model** to assess geographical sustainability classes.
* **Multi-Criteria Environmental Analysis:** Evaluates population density proxies, green infrastructure ratios, connectivity metrics, elevation matrices, and flood-risk probabilities.
* **Interactive Custom Scoring Engine:** Computes a composite sustainability index using a balanced, weighted empirical formula:
  
  $$\text{Score} = (G \times 0.35) + (I \times 0.25) + (P \times 0.20) + (F \times 0.20)$$
  
  *(Where $G$ = Green Cover, $I$ = Infrastructure, $P$ = Population Density, and $F$ = Flood Safety)*

---

## 📂 Project Architecture

```text
sustainable_urban_ai/
│
├── app.py                  # Core production Flask application layout
├── apt.txt                 # Native C++ background dependencies for Render (GDAL/PROJ)
├── requirements.txt        # Frozen Python package distributions 
├── .gitignore              # Safeguards local virtual environments from version control
│
├── models/
│   └── random_forest_model.pkl   # Pre-trained core machine learning classifier
│
└── templates/
    └── index.html          # Interactive front-end mapping portal
