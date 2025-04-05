import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Benchmark IA", layout="wide")
st.title("🎙️ Benchmark IA — Analyse complète")

AUDIO_DIR = "../audio_converted"
CSV_PATH = "../output/benchmark_keywords_results.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error("Fichier manquant : benchmark_keywords_results.csv")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    selected_file = st.selectbox("Fichier audio", ["Tous"] + sorted(df["Fichier"].unique()))
with col2:
    selected_ias = st.multiselect("IA", df["IA"].unique(), default=df["IA"].unique())

filtered_df = df[df["IA"].isin(selected_ias)]
if selected_file != "Tous":
    filtered_df = filtered_df[filtered_df["Fichier"] == selected_file]

for _, row in filtered_df.iterrows():
    st.subheader(f"{row['Fichier']} — {row['IA']}")

    audio_path = os.path.join(AUDIO_DIR, row["Fichier"])
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
    else:
        st.warning("Fichier audio introuvable")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("WER", row["WER"])
    col2.metric("CER", row["CER"])
    col3.metric("Score Global", f"{row['Score_Global_%']}%")
    col4.metric("Temps", row["Temps"])

    colS1, colS2, colS3 = st.columns(3)
    colS1.metric("BLEU", f"{row['BLEU_%']}%")
    colS2.metric("ROUGE-L", f"{row['ROUGE_L_%']}%")
    colS3.metric("BERTScore", f"{row['BERTScore_F1_%']}%")

    colK1, colK2 = st.columns(2)
    colK1.metric("Score Mots-Clés", f"{row['Score_MotsClés_%']}%")
    colK2.metric("Distance L2", round(row["Distance_L2"], 3))

    with st.expander("Mots retrouvés"):
        st.write(row["Mots_Clés_Trouvés"])
    with st.expander("Mots manquants"):
        st.write(row["Mots_Clés_Manquants"])

    st.markdown("**Transcription IA**")
    st.write(row["Texte_IA"])
    st.markdown("**Référence**")
    st.write(row["Référence"])
    st.markdown("---")

if st.checkbox("Afficher le tableau complet"):
    st.dataframe(filtered_df[[
        "Fichier", "IA", "WER", "CER", "Score_Global_%",
        "BLEU_%", "ROUGE_L_%", "BERTScore_F1_%",
        "Score_MotsClés_%", "Distance_L2", "Temps"
    ]])

if st.checkbox("Afficher les graphiques comparatifs"):
    avg = (
        df[df["IA"].isin(selected_ias)]
        .groupby("IA")[[
            "Score_Global_%", "BLEU_%", "ROUGE_L_%", "BERTScore_F1_%", "WER", "CER"
        ]]
        .mean()
        .reset_index()
    )

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(avg["IA"], avg["Score_Global_%"])
    ax1.set_title("Score Global moyen par IA")
    ax1.set_ylabel("Score (%)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = range(len(avg["IA"]))
    ax2.bar([i - width for i in x], avg["BLEU_%"], width=width, label="BLEU")
    ax2.bar(x, avg["ROUGE_L_%"], width=width, label="ROUGE-L")
    ax2.bar([i + width for i in x], avg["BERTScore_F1_%"], width=width, label="BERTScore")
    ax2.set_xticks(x)
    ax2.set_xticklabels(avg["IA"])
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Scores sémantiques moyens")
    ax2.legend()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(avg["IA"], avg["WER"], label="WER", alpha=0.7)
    ax3.bar(avg["IA"], avg["CER"], label="CER", alpha=0.5)
    ax3.set_title("WER / CER moyens")
    ax3.set_ylabel("Taux d'erreur")
    ax3.legend()
    st.pyplot(fig3)
