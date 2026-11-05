
import joblib
import pandas as pd

model = joblib.load("crop_recommendation_model.joblib")

features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

sample = pd.DataFrame(
    [[60, 30, 35, 22, 60, 6.4, 700]],
    columns=features
)

probabilities = model.predict_proba(sample)[0]
order = probabilities.argsort()[::-1][:3]

for rank, i in enumerate(order, start=1):
    print(
        f"{rank}. {model.classes_[i]} "
        f"— model score: {probabilities[i]:.2%}"
    )

