"""
Run this once to train and save the model.
Usage: python train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading dataset...")
df = pd.read_csv('placement.csv')
print(f"Dataset shape: {df.shape}")

# Encode categoricals
le_dict = {}
cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
df_enc = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Target
df_enc['placed'] = (df_enc['status'] == 'Placed').astype(int)

features = ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s',
            'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']
X = df_enc[features]
y = df_enc['placed']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
cv = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nCross-val accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
print(f"Test accuracy:      {test_acc:.3f}")
print()
print(classification_report(y_test, model.predict(X_test), target_names=['Not Placed', 'Placed']))

# Averages for comparison
placed_avg = df_enc[df_enc['placed'] == 1][['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']].mean()
not_placed_avg = df_enc[df_enc['placed'] == 0][['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']].mean()

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'encoders': le_dict,
        'features': features,
        'placed_avg': placed_avg.to_dict(),
        'not_placed_avg': not_placed_avg.to_dict()
    }, f)

print("✅ Model saved to model.pkl")
