import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("french"))

def mots_essentiels(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [w for w in tokens if w not in string.punctuation]
    return set(w for w in tokens if w not in stop_words)

def score_mots_importants(reference, prediction):
    mots_ref = mots_essentiels(reference)
    mots_ia = mots_essentiels(prediction)
    if not mots_ref:
        return 1.0, set(), set()
    mots_corrects = mots_ref & mots_ia
    score = len(mots_corrects) / len(mots_ref)
    return score, mots_corrects, mots_ref - mots_ia

df = pd.read_csv("../output/benchmark_open_results.csv")

scores = []
mots_trouves = []
mots_manquants = []

for _, row in df.iterrows():
    score, trouves, manquants = score_mots_importants(row["Référence"], row["Texte_IA"])
    scores.append(round(score * 100, 2))
    mots_trouves.append(", ".join(trouves))
    mots_manquants.append(", ".join(manquants))

df["Score_MotsClés_%"] = scores
df["Mots_Clés_Trouvés"] = mots_trouves
df["Mots_Clés_Manquants"] = mots_manquants

df.to_csv("../output/benchmark_keywords_results.csv", index=False)
print("benchmark_keywords_results.csv généré.")
