# 🍄 Mushroom Project – Edibility Classifier with ML

## 🧾 Project Abstract

This project aims to classify mushrooms as edible or poisonous using machine learning techniques. It combines an intuitive Streamlit interface with robust models trained on the UCI Mushroom Dataset. The web app also includes a secure authentication system and educational sections on mushrooms' ecological, biological, and nutritional roles.

---

## 📌 Table of Contents

- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Scrrenshots](#-Screenshot)
- [Technologies Used](#-technologies-used)
- [Installation Guide](#-installation-guide)
- [Authentication System](#-authentication-system)
- [How It Works](#-how-it-works)
- [Machine Learning Models](#-machine-learning-models)
- [Feature Engineering](#-feature-engineering)
- [Data Preprocessing](#-data-preprocessing)
- [Data Exploration & Visualization](#-data-exploration--visualization)
- [Model Selection Rationale](#-model-selection-rationale)
- [Performance Metrics](#-performance-metrics)
- [Testing & Evaluation](#-testing--evaluation)
- [Real-World Applications](#-real-world-applications)
- [Dashboard Preview](#-dashboard-preview)
- [Deployment](#-deployment)
- [API Support](#-api-support)
- [Future Enhancements](#-future-enhancements)
- [Project Timeline / Milestones](#-project-timeline--milestones)
- [Safety Disclaimer](#-safety-disclaimer)
- [Feedback & Support](#-feedback--support)
- [Team](#-team)
- [Acknowledgments](#-acknowledgments)
- [References](#-references)
- [To-Do List](#-to-do-list)
- [License](#-license)

---

## 🚀 Demo

🖥️ [Live Demo](https://mushroom-trio-classifier.onrender.com/) (https://mushroom-trio-classifier.onrender.com/)  

---

## 📂 Project Structure

```
Mushroom-Project/
├── datasets/
├── models/
├── Images/
├── Dockerfile         # Containerization
├── Main.py            # Main Streamlit app (authentication + navigation)
├── requirements.txt   # Python dependencies
├── users.db           # SQLite database (auto-created)
├── README.md
└── views/             # Streamlit page modules
```
---

## 📷 Screenshots
Here is snapshot of each pages.
- Login:
  ![Mushroom Classifier](https://private-user-images.githubusercontent.com/168903254/663061729-b4b74078-8adc-4e94-aa9c-d1a454b4427f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3OTA4NTc1ODYsIm5iZiI6MTc5MDg1NzI4NiwicGF0aCI6Ii8xNjg5MDMyNTQvNjYzMDYxNzI5LWI0Yjc0MDc4LThhZGMtNGU5NC1hYTljLWQxYTQ1NGI0NDI3Zi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYxMDAxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MTAwMVQxMjIxMjZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02MzUxYzU5MDg4ZmJhM2ViN2E3MmRiZTYyZDFjYzNiMjhmZGY0MDRkZGFjNjU3YWM5NjY2MDI3NTUxODJmYWVhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.PmLfvj6uMblT6jOMdOzbq6N34EfcaWpdroIuHFnbBdQ)
- Signup:
  ![Mushroom Classifier](https://private-user-images.githubusercontent.com/168903254/663066215-4bc1705a-f100-4ba6-be8a-a2c5d71058d8.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3OTA4NTc5MzMsIm5iZiI6MTc5MDg1NzYzMywicGF0aCI6Ii8xNjg5MDMyNTQvNjYzMDY2MjE1LTRiYzE3MDVhLWYxMDAtNGJhNi1iZThhLWEyYzVkNzEwNThkOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYxMDAxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MTAwMVQxMjI3MTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jMTEwNzc5YTAxZTdkMjE1N2NkYjRiMDc0ODBkMzU2YTY4ODc5M2M3ZDU4MDQ5ZmI2OTAwMTdkZTJhZjVlZDkxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.gZeqh_lN2BSKryxwt9ps8rXuxSC2Xs17Oupg149n8_4)
- Home:
  ![Mushroom Classifier](https://private-user-images.githubusercontent.com/168903254/663073543-933579fc-94ec-4e22-87cf-f09fc0a4752f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3OTA4NTkxNzMsIm5iZiI6MTc5MDg1ODg3MywicGF0aCI6Ii8xNjg5MDMyNTQvNjYzMDczNTQzLTkzMzU3OWZjLTk0ZWMtNGUyMi04N2NmLWYwOWZjMGE0NzUyZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYxMDAxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MTAwMVQxMjQ3NTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01MTAxYjA4NTcyNWU2MDM5YjRiNmNkM2E1NmUxNGY2YjY5MWJjODAzZWYwMDU1ODE1NzUxOTIwMDFiZmRmNmU2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.x85I6FlY3eaUdzKil0cRL2abr-rfu1_NAtqbeW5cTic)
- Edibility Checker:
  ![Mushroom Classifier](https://private-user-images.githubusercontent.com/168903254/663073988-8c346bc1-14a3-476f-9b60-89cc9a9a3164.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3OTA4NTkyNTcsIm5iZiI6MTc5MDg1ODk1NywicGF0aCI6Ii8xNjg5MDMyNTQvNjYzMDczOTg4LThjMzQ2YmMxLTE0YTMtNDc2Zi05YjYwLTg5Y2M5YTlhMzE2NC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYxMDAxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MTAwMVQxMjQ5MTdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zYjk2NTE0MDQ5YmE1MmMyMTY3YTVmYWZhYTIwMWUxNGY2MTMzNjE4ZTJjMDdhMDA4MzE5NTdmYTJkNzk2YjQ5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ._vLe5WEDx6QbHJC_6va8pXGKoiePNhvyV3TqXBNNs_Q)
- Mushroom ML Lab:
  ![Mushroom Classifier](https://private-user-images.githubusercontent.com/168903254/663074711-64e1d7dc-72fa-4e2d-b0f8-842da898a7aa.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3OTA4NTkzMTAsIm5iZiI6MTc5MDg1OTAxMCwicGF0aCI6Ii8xNjg5MDMyNTQvNjYzMDc0NzExLTY0ZTFkN2RjLTcyZmEtNGUyZC1iMGY4LTg0MmRhODk4YTdhYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYxMDAxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MTAwMVQxMjUwMTBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xZDZjMzU0MjIzNTU4MzlkODQ5ZWJhOGNjODZlYTk1NGI4OWI2N2Q5ZDFhZjhiNzk5MDk5NDEzOTc4ZWNiNWYxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.GHpzzDs6oD0p0LHhjm5fJiIMulLS96z8usONY4--TRI)

![eligibiity](https://private-user-images.githubusercontent.com/168903254/663075129-72393f5e-3124-405f-bac6-72682289c95e.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3OTA4NTkzNTUsIm5iZiI6MTc5MDg1OTA1NSwicGF0aCI6Ii8xNjg5MDMyNTQvNjYzMDc1MTI5LTcyMzkzZjVlLTMxMjQtNDA1Zi1iYWM2LTcyNjgyMjg5Yzk1ZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYxMDAxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MTAwMVQxMjUwNTVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ZWUxZDNhZTkxMDg0NzhiM2RiZTlhYTRlOTNiMDJlZTQxNDYwZDI4OWU4ZjliYjdhODBhZTUwNDM4MTg1Mjk2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.uNSebSOo3Pj3MBiLdmTdSiqZjiwhhGiLvVLmk-_Zk6o)



## ⚙️ Technologies Used

- **Python** 🐍
- **Streamlit** 🌐 – for web interface
- **SQLite** 🗃️ – for authentication
- **Pandas & NumPy** – data handling
- **Scikit-learn** – ML model training
- **Matplotlib & Seaborn** – data visualization
- **Docker** - Deployment

---

## 📥 Installation Guide

1. Clone the repository:
```bash
git clone https://github.com/samarthgarde/Mushroom_Final_Project.git
cd Mushroom_Final_Project
```

2. Create and activate virtual environment:
```bash
python -m venv mushroom-env
source mushroom-env/bin/activate  # On Windows use: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python Main.py

```

---

## 🔒 Authentication System

- User login/signup
- Password hashing
- SQLite database

---

## 💡 How It Works

1. **User Login/Signup** through a secure interface.
2. **Select Mushroom Features** using dropdowns and forms.
3. **Model predicts** whether the mushroom is edible or poisonous.
4. Display results with **confidence score** and explanations.
5. Includes pages like:
   - 📚 **Mushroom Wisdom**: Biological, nutritional, and ecological info,Fun facts, types, and safety tips.
    - ✅ **Gallery**: project information ,behind the sceen and team.

---

## 📊 Machine Learning Models

Three classification models were trained and evaluated:
- 🔍 Logistic Regression
- 🌲 Random Forest Classifier
- 📈 Support Vector Machine (SVM)
---

## 📦 Feature Engineering

- Categorical encoding using LabelEncoder
- Removal of ambiguous data
- Feature selection based on correlation

---

## 🗃️ Data Preprocessing

- UCI dataset (8124 samples)
- Null/missing handling
- Train-test split (80:20)

---

## 📈 Data Exploration & Visualization

- Class distribution
- Feature correlation heatmaps
- Count plots for each feature

---

## 🧠 Model Selection Rationale

- Logistic Regression: Simple baseline
- Random Forest: Robust and accurate
- SVM: Great for small, high-dimensional data

---

## 📈 Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 94.6% | 0.95 | 0.94 | 0.94 |
| Random Forest | 99.1% | 0.99 | 0.99 | 0.99 |
| SVM | 98.2% | 0.98 | 0.98 | 0.98 |

---

## 🧪 Testing & Evaluation

- 5-fold cross-validation
- Confusion matrix analysis
- ROC-AUC evaluation

---

## 🌍 Real-World Applications

- 🍽️ Food safety checks in rural foraging
- 🧑‍🌾 Agricultural advisory systems
- 📱 Edibility Checker apps for campers and hikers
- 🎓 Educational tools for botany students

---

## 📊 Dashboard Preview

()

---

## 🛠️ Deployment

- To be deployed on Streamlit Cloud / Render 

---

## 📦 API Support

Planned future feature: REST API to classify mushrooms via JSON input

---

## 🌱 Future Enhancements

- Image-based classification (CNN)
- Mobile optimization
- Admin dashboard
- Multilingual support

---

## 📆 Project Timeline / Milestones

- 📌 Dataset research – Week 1
- 🤖 Model training & evaluation – Week 2
- 🧱 Streamlit frontend – Week 3
- 🔐 Auth integration – Week 4
- 🌐 Deployment – Week 5 (planned)

---

## 🛡️ Safety Disclaimer

⚠️ **Educational tool only.** Do not use for real-life mushroom identification. Seek expert advice.

---

## 💬 Feedback & Support

- Raise issues via GitHub Issues tab
- Contact developer via [GitHub Profile](https://github.com/samarthgarde)

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository
- Streamlit team
- OpenAI for assistance
- Mentors and Professors 
- Team Members and Contributors

---

## 📚 References

- [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn Docs](https://scikit-learn.org/stable/user_guide.html)

---

## 🐳 Run with Docker

- Build the image
```
docker build -t mushroom-app .
```
- Run the container
```
docker run -d -p 8501:8501 --name mushroom-container mushroom-app
```
- Open in browser 👉 http://localhost:8501

---

## ✅ To-Do List

- [x] Implement Authentication System
- [x] Design UI in Streamlit
- [x] Build ML Modle 
- [X] Conduct User Testing & Feedback Collection
- [x] Deploy app publicly
- [ ] Implement REST API
- [ ] Admin Dashboard 
- [ ] Multilingual Support 

---

## 📄 License
This project is licensed under the [MIT License](https://mit-license.org/)

---
