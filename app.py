import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
from rake_nltk import Rake
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
import time

nltk.download("stopwords")

# ------------------ Lottie Loader ------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_summary = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_9cyyl8i4.json")
lottie_keywords = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_mjlh3hcy.json")
lottie_wordcloud = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_pwohahvd.json")

# ------------------ Page Config ------------------
st.set_page_config(page_title="AI Text Summarizer", layout="centered")
st.title("📝 AI Text Summarizer")
st_lottie(lottie_summary, height=250)

st.markdown("Summarize typed text or uploaded PDF files in seconds with HuggingFace Transformers and Streamlit.")

# ------------------ Summary Length Option ------------------
summary_length_option = st.sidebar.selectbox("📏 Select Summary Length", ["Short (3 lines)", "Medium", "Long"])
length_map = {
    "Short (3 lines)": (60, 30),
    "Medium": (120, 60),
    "Long": (200, 80)
}

# ------------------ Upload or Type ------------------
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
input_text = ""

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        input_text = text
        st.success("Text extracted successfully!")
        st.text_area("Extracted PDF Text:", input_text, height=200)
else:
    input_text = st.text_area("Or enter text to summarize:", height=250)

if input_text.strip():
    st.markdown(f"**Input Word Count:** {len(input_text.split())}")
    st.markdown(f"**Input Character Count:** {len(input_text)}")

# ------------------ Summarization ------------------
summary = ""
if st.button("✨ Summarize"):
    if input_text.strip():
        with st.spinner("Generating summary..."):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            max_len, min_len = length_map[summary_length_option]
            summary = summarizer(input_text, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
            time.sleep(1)

        st.toast("Summary generated!")
        st.balloons()

        with st.expander("View Summary"):
            st.success(summary)
            st.markdown(f"**Summary Word Count:** {len(summary.split())}")
            st.markdown(f"**Summary Character Count:** {len(summary)}")

        st.download_button("📥 Download Summary", summary, file_name="summary.txt", mime="text/plain")

        # ------------------ Keywords ------------------
        st_lottie(lottie_keywords, height=150)
        st.subheader("Top Keywords from the Text")
        r = Rake()
        r.extract_keywords_from_text(input_text)
        keywords = r.get_ranked_phrases()[:10]

        if keywords:
            with st.expander("View Keywords"):
                for i, kw in enumerate(keywords, 1):
                    st.markdown(f"**{i}.** {kw}")
        else:
            st.info("No significant keywords found.")

        # ------------------ Word Cloud ------------------
        st_lottie(lottie_wordcloud, height=180)
        st.subheader("Keyword Word Cloud")

        if keywords:
            wordcloud_text = " ".join(keywords)
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(wordcloud_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
    else:
        st.warning("Please enter or upload some text.")
