import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

df = pd.read_csv("../output/benchmark_keywords_results.csv")

distances = []
for _, row in df.iterrows():
    ref = str(row["Référence"])
    ia = str(row["Texte_IA"])
    emb_ref = model.encode(ref)
    emb_ia = model.encode(ia)
    dist = np.linalg.norm(emb_ref - emb_ia)
    distances.append(dist)

df["Distance_L2"] = distances

max_l2 = max(distances)
scores = []

for i in range(len(df)):
    mots = df.loc[i, "Score_MotsClés_%"] / 100
    l2 = df.loc[i, "Distance_L2"]
    l2_norm = 1 - (l2 / max_l2)
    score = mots * l2_norm * 100
    scores.append(round(score, 2))

df["Score_Global_%"] = scores

df.to_csv("../output/benchmark_keywords_results.csv", index=False)
print("benchmark_keywords_results.csv mis à jour avec Distance L2 et Score Global.")
