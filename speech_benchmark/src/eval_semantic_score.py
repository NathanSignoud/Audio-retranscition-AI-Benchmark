import pandas as pd
import evaluate

# Charger les métriques
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Charger le fichier principal
df = pd.read_csv("../output/benchmark_keywords_results.csv")

# Créer les listes de scores
bleu_scores = []
rouge_scores = []
bertscore_f1s = []

# Calculer les scores pour chaque ligne
for _, row in df.iterrows():
    ref = str(row["Référence"])
    pred = str(row["Texte_IA"])

    # BLEU
    bleu_result = bleu.compute(predictions=[pred], references=[[ref]])
    bleu_scores.append(round(bleu_result["bleu"] * 100, 2))

    # ROUGE-L
    rouge_result = rouge.compute(predictions=[pred], references=[ref])
    rouge_scores.append(round(rouge_result["rougeL"] * 100, 2))

    # BERTScore
    bert_result = bertscore.compute(predictions=[pred], references=[ref], lang="fr")
    f1_score = bert_result["f1"][0]
    bertscore_f1s.append(round(f1_score * 100, 2))

# Ajouter les scores au DataFrame
df["BLEU_%"] = bleu_scores
df["ROUGE_L_%"] = rouge_scores
df["BERTScore_F1_%"] = bertscore_f1s

# Sauvegarder dans le même fichier
df.to_csv("../output/benchmark_keywords_results.csv", index=False)