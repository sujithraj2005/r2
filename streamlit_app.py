import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="NCERT Q&A Helper", layout="wide")
st.title("NCERT Q&A Helper (RAG with Top-k & Top-p Retrieval)")

st.sidebar.header("Retrieval Strategy")
retrieval = st.sidebar.radio(
    "Select Retrieval Strategy",
    options=["top_k", "top_p"],
    format_func=lambda mode: "Top-k (Fixed Count)" if mode == "top_k" else "Top-p (Nucleus / Cumulative Probability)",
    help="Choose whether to retrieve a fixed number of chunks (Top-k) or dynamically adapt the chunk count based on confidence (Top-p)."
)

if retrieval == "top_k":
    k = st.sidebar.slider("k (number of passages)", min_value=1, max_value=15, value=5)
    p = 0.85
    st.sidebar.info(
        "**Top-k Retrieval:** Always retrieves a fixed number ($k$) of closest passages, "
        "regardless of whether they are relevant or noisy."
    )
else:
    k = 5
    p = st.sidebar.slider("p (cumulative probability cutoff)", min_value=0.50, max_value=0.99, value=0.85, step=0.05)
    st.sidebar.info(
        "**Top-p (Nucleus) Retrieval:** Converts distances into relative probabilities and "
        "retrieves the smallest set of passages whose cumulative probability reaches $p$.\n\n"
        "- High-confidence queries require fewer passages (less noise).\n"
        "- Ambiguous queries automatically gather more passages for broader context."
    )

with st.sidebar.expander("Advanced Settings"):
    temperature = st.slider("Softmax Temperature", min_value=0.1, max_value=1.0, value=0.25, step=0.05,
                            help="Lower temperature sharpens probability around top matches. Higher temperature spreads probability more evenly.")

question = st.text_input("Enter your question:", placeholder="e.g., What are the properties of electromagnetic waves?")

if st.button("Ask"):
    if question:
        with st.spinner("Searching NCERT books and generating answer..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question, "retrieval": retrieval, "k": k, "p": p, "temperature": temperature},
                    timeout=120,
                )
            except Exception as e:
                response = None
                st.error(f"Failed to connect to backend: {e}")

        if response and response.status_code == 200:
            data = response.json()

            st.subheader("Answer:")
            st.markdown(data["answer"])

            docs = data.get("retrieved_documents", [])
            if data["retrieval"] == "top_p":
                st.success(
                    f"**Top-p Retrieval:** Dynamically retrieved **{len(docs)}** passage(s) "
                    f"to reach cumulative probability **{p:.0%}**."
                )
            else:
                st.info(f"**Top-k Retrieval:** Retrieved a fixed count of **{len(docs)}** passage(s) ($k={k}$).")

            st.subheader(f"Retrieved Documents ({len(docs)} passages):")
            for idx, doc in enumerate(docs, start=1):
                prob = doc['probability']
                cum_prob = doc['cumulative_probability']
                label = f"#{idx} | Page {doc['page']} | Prob: {prob:.1%} | Cumulative: {cum_prob:.1%} | Distance: {doc['distance']:.2f}"
                with st.expander(label, expanded=(idx == 1)):
                    st.progress(min(max(cum_prob, 0.0), 1.0))
                    st.write(f"**Source:** {doc['link']} (Page {doc['page']})")
                    st.write(f"**Relevance Probability:** {prob:.2%}")
                    st.write(f"**Cumulative Probability:** {cum_prob:.2%}")
                    st.write(f"**Raw Vector Distance:** {doc['distance']:.4f}")
                    st.markdown("**Content Snippet:**")
                    st.info(doc["snippet"])
        elif response:
            st.error(f"Error {response.status_code}: {response.text}")
    else:
        st.warning("Please enter a question.")

st.sidebar.header("About")
st.sidebar.markdown(
    "**NCERT Tutor RAG System** with **Top-k vs Top-p** Nucleus Retrieval.\n\n"
    "- Vector Store: ChromaDB\n"
    "- Embeddings: nomic-embed-text (Ollama)\n"
    "- LLM: Groq / Ollama\n"
)
