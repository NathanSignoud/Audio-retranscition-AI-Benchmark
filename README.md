
# Benchmark de Systèmes de Transcription Vocale

Ce projet a pour objectif de comparer plusieurs systèmes de reconnaissance vocale automatique (ASR) pour transcrire des fichiers audio de locuteurs sénégalais parlant français. Il vise à évaluer la précision, la qualité sémantique et la robustesse de différents modèles open source.

---

## Objectifs

- Comparer les performances de **Whisper**, **Vosk**, et **Wav2Vec2**
- Utiliser des fichiers audio normalisés et des transcriptions humaines de référence
- Calculer des métriques variées : WER, CER, BLEU, ROUGE-L, BERTScore, etc.
- Visualiser les résultats via une interface web interactive (Streamlit)

---

## Prérequis

- Python 3.9+
- ffmpeg (installé et accessible via le PATH)
- Environnement virtuel recommandé

```bash

pip install -r requirements.txt
pip install rouge_score
pip install bert_score
pip install hf_xet
cd src/

```

---

## Structure du projet

```bash
speech_benchmark/
├── audio/                        # Fichiers audio originaux (.mp3, .wav)
├── audio_converted/             # Fichiers audio convertis en wav mono 16kHz
├── references/                  # Transcriptions humaines (.txt)
├── models/                      # Modèle Vosk français
├── output/                      # CSV contenant les résultats
├── src/                         # Scripts Python principaux
│   ├── benchmark.py             # Lance les 3 IA et collecte les transcriptions
│   ├── eval_key_word.py         # Score mots-clés retrouvés
│   ├── eval_l2_distance.py      # Distance sémantique + score global
│   ├── eval_semantic_score.py   # BLEU / ROUGE-L / BERTScore
│   ├── convert_audio.py         # Conversion audio standardisée
│   ├── app.py                   # Interface Streamlit interactive
│   └── build_all.py             # Pipeline complet automatique
```

---

## Utilisation

### 1. Convertir les fichiers audio (si besoin)
```bash
python convert_audio.py
```

### 2. Lancer le benchmark complet
```bash
python pipeline.py
```

### 3. Lancer l'application web
```bash
streamlit run app.py
```

---

## Métriques calculées

| Métrique           | Description                                        |
|--------------------|----------------------------------------------------|
| **WER / CER**      | Taux d'erreur sur mots et caractères               |
| **Score Mots-Clés**| Proportion de mots importants reconnus             |
| **Distance L2**    | Distance sémantique entre embeddings SBERT         |
| **Score Global**   | Combinaison mots-clés et proximité sémantique       |
| **BLEU / ROUGE-L** | Scores de similarité textuelle                     |
| **BERTScore**      | Similarité sémantique contextuelle                |

---

## Auteur
Nathan Signoud
Projet réalisé dans le cadre d'un benchmark IA vocal pour la transcription du français parlé par des locuteurs sénégalais. Réalisé avec Python, Streamlit, HuggingFace, Vosk et Whisper.
