# 🎓 Campus Placement Predictor

A full-stack ML web app that predicts campus placement probability based on a student's academic profile.

**Model Accuracy: 88.4%** | Dataset: 215 MBA students | Algorithm: Logistic Regression

---

## 📁 Project Structure

```
placement_app/
├── app.py              # Flask backend + prediction API
├── train_model.py      # Train and save the ML model
├── model.pkl           # Saved model (auto-generated)
├── placement.csv       # Dataset (Placement_Data_Full_Class.csv)
├── requirements.txt    # Python dependencies
└── templates/
    └── index.html      # Full frontend UI
```

---

## 🚀 Running Locally

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model (if model.pkl is missing)
```bash
python train_model.py
```

### Step 3 — Run the app
```bash
python app.py
```

Open: **http://127.0.0.1:5000**

---

## 🌐 Deploying to Render (Free Hosting)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/placement-predictor.git
git push -u origin main
```

### Step 2 — Deploy on Render
1. Go to **https://render.com** and sign up / log in
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Set these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3
5. Click **Deploy** — done! You'll get a live URL like `https://placement-predictor.onrender.com`

> **Free tier note:** Render's free tier spins down after inactivity. First load may take ~30 seconds.

---

## 📊 Features Used in Prediction

| Feature | Description |
|---|---|
| `gender` | M / F |
| `ssc_p` | Secondary School Certificate % (10th grade) |
| `ssc_b` | SSC Board (Central / Others) |
| `hsc_p` | Higher Secondary Certificate % (12th grade) |
| `hsc_b` | HSC Board |
| `hsc_s` | HSC Stream (Science / Commerce / Arts) |
| `degree_p` | Undergraduate degree % |
| `degree_t` | Degree type (Sci&Tech / Comm&Mgmt / Others) |
| `workex` | Work experience (Yes / No) |
| `etest_p` | Employability test score % |
| `specialisation` | MBA specialisation (Mkt&HR / Mkt&Fin) |
| `mba_p` | MBA % |

---

## 🧠 Model Details
- **Algorithm:** Logistic Regression (with StandardScaler)
- **Train/Test Split:** 80/20
- **Cross-val Accuracy:** ~85.1% ± 5.4%
- **Test Accuracy:** 88.4%
- **Dataset:** Placement_Data_Full_Class.csv (215 rows, 15 columns)
