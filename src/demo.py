from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from features import continuum_normalize, spectral_features

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)
rng = np.random.default_rng(7)
w = np.arange(400, 1001, 5)
classes = ["healthy", "nutrient_stress", "disease_stress"]
rows, labels = [], []

for cls in classes:
    for _ in range(75):
        vis = 0.12 + 0.04*np.exp(-((w-550)/45)**2)
        red_abs = -0.055*np.exp(-((w-670)/28)**2)
        red_edge = 0.42/(1+np.exp(-(w-720)/18))
        curve = vis + red_abs + red_edge
        if cls == "nutrient_stress":
            curve -= 0.035*np.exp(-((w-550)/55)**2)
            curve -= 0.045/(1+np.exp(-(w-735)/20))
        elif cls == "disease_stress":
            curve += 0.025*np.exp(-((w-680)/45)**2)
            curve -= 0.075/(1+np.exp(-(w-730)/22))
        curve += rng.normal(0, 0.006, len(w))
        rows.append(curve)
        labels.append(cls)

X = np.vstack(rows)
y = np.array(labels)
pd.DataFrame(X, columns=[f"R_{v}" for v in w]).assign(label=y).to_csv(ROOT/"data"/"synthetic_hyperspectral_spectra.csv", index=False)

Xn = continuum_normalize(X)
feat = spectral_features(w, Xn)
X_train, X_test, y_train, y_test = train_test_split(feat, y, test_size=.25, random_state=42, stratify=y)
models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3))
}
scores = {}
best_score = -1
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores[name] = accuracy_score(y_test, pred)
    if scores[name] > best_score:
        best_score, best_pred = scores[name], pred

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(StandardScaler().fit_transform(feat))
plt.figure(figsize=(7.2,5))
for cls in np.unique(y):
    m = y == cls
    plt.scatter(coords[m,0], coords[m,1], label=cls, alpha=.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Hyperspectral Stress Feature Space")
plt.legend(); plt.tight_layout()
plt.savefig(ROOT/"outputs"/"pca_stress_classes.png", dpi=180); plt.close()
ConfusionMatrixDisplay.from_predictions(y_test, best_pred)
plt.title("Best Model Confusion Matrix"); plt.tight_layout()
plt.savefig(ROOT/"outputs"/"confusion_matrix.png", dpi=180); plt.close()
(ROOT/"outputs"/"model_scores.json").write_text(json.dumps(scores, indent=2))
print(scores)
