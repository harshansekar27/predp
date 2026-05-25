from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']
encoders = bundle['encoders']
features = bundle['features']
placed_avg = bundle['placed_avg']
not_placed_avg = bundle['not_placed_avg']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Encode categoricals
        cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
        encoded = {}
        for col in cat_cols:
            val = data[col]
            le = encoders[col]
            encoded[col] = int(le.transform([val])[0])

        # Build feature vector in correct order
        row = [
            encoded['gender'],
            float(data['ssc_p']),
            encoded['ssc_b'],
            float(data['hsc_p']),
            encoded['hsc_b'],
            encoded['hsc_s'],
            float(data['degree_p']),
            encoded['degree_t'],
            encoded['workex'],
            float(data['etest_p']),
            encoded['specialisation'],
            float(data['mba_p'])
        ]

        X = np.array(row).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        placed_prob = float(prob[1]) * 100

        # Build comparison data
        user_scores = {
            'ssc_p': float(data['ssc_p']),
            'hsc_p': float(data['hsc_p']),
            'degree_p': float(data['degree_p']),
            'etest_p': float(data['etest_p']),
            'mba_p': float(data['mba_p'])
        }

        # Personalized tips
        tips = []
        if float(data['ssc_p']) < placed_avg['ssc_p']:
            tips.append(f"Your SSC score ({data['ssc_p']}%) is below the placed average ({placed_avg['ssc_p']:.1f}%). Strong fundamentals matter.")
        if float(data['hsc_p']) < placed_avg['hsc_p']:
            tips.append(f"Your HSC score ({data['hsc_p']}%) is below the placed average ({placed_avg['hsc_p']:.1f}%). Consider strengthening core subjects.")
        if float(data['degree_p']) < placed_avg['degree_p']:
            tips.append(f"Your degree percentage ({data['degree_p']}%) is below average for placed students ({placed_avg['degree_p']:.1f}%). Focus on consistent academic performance.")
        if float(data['etest_p']) < placed_avg['etest_p']:
            tips.append(f"Your employability test score ({data['etest_p']}%) is below the placed average ({placed_avg['etest_p']:.1f}%). Practice aptitude tests.")
        if data['workex'] == 'No':
            tips.append("Work experience significantly boosts placement chances. Consider internships or part-time roles.")
        if not tips:
            tips.append("Your profile is strong! Keep maintaining your academic performance and skills.")

        return jsonify({
            'placed': bool(pred == 1),
            'probability': round(placed_prob, 1),
            'user_scores': user_scores,
            'placed_avg': placed_avg,
            'not_placed_avg': not_placed_avg,
            'tips': tips
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
