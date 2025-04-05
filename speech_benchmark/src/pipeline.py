import subprocess

print("\nÉtape 1/4 — Benchmark IA...")
subprocess.run(["python", "benchmark.py"], check=True)

print("\nÉtape 2/4 — Calcul des mots-clés...")
subprocess.run(["python", "eval_key_word.py"], check=True)

print("\nÉtape 3/4 — Calcul des distances L2 + score global...")
subprocess.run(["python", "eval_l2_distance.py"], check=True)

print("\nÉtape 4/4 — Calcul des scores sémantiques...")
subprocess.run(["python", "eval_semantic_score.py"], check=True)

print("\n✅ Pipeline complet exécuté avec succès.")
print("💡 Lance maintenant l'application Streamlit :\n")
print("    streamlit run app.py")
