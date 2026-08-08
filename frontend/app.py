import streamlit as st
from api_client import ask_question, ask_with_image, check_health

st.set_page_config(page_title="Traffic Law Assistant", page_icon="🚦", layout="centered")

st.title("🚦 Traffic Law & Vehicle Regulations Assistant")
st.caption(
    "Ask questions about speed limits, license plates, right of way, parking, "
    "DUI/safety rules, or vehicle registration. Answers are grounded in the "
    "source documents, with citations. Extended Track: optionally upload a "
    "vehicle photo for plate detection."
)

with st.sidebar:
    st.subheader("Backend status")
    try:
        health = check_health()
        if health.get("status") == "ok":
            st.success(f"Connected — {health.get('num_chunks', 0)} chunks loaded")
        else:
            st.warning("Backend reachable but vector store not loaded yet")
    except Exception:
        st.error("Backend not reachable. Is FastAPI running on the configured API_BASE_URL?")

if "history" not in st.session_state:
    st.session_state.history = []

tab_text, tab_image = st.tabs(["💬 Ask a question", "📷 Ask with an image (Extended)"])

with tab_text:
    question = st.text_input("Your question", placeholder="e.g. What is the speed limit near schools?")
    if st.button("Ask", type="primary", key="ask_text"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating a grounded answer..."):
                try:
                    result = ask_question(question)
                    st.session_state.history.append(("text", question, result))
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab_image:
    image_question = st.text_input("Your question about the image", placeholder="e.g. Is this plate readable?")
    uploaded_image = st.file_uploader("Upload a vehicle/plate image", type=["jpg", "jpeg", "png"])
    if st.button("Ask with image", type="primary", key="ask_image"):
        if not image_question.strip() or uploaded_image is None:
            st.warning("Please provide both a question and an image.")
        else:
            with st.spinner("Running detection and generating a grounded answer..."):
                try:
                    result = ask_with_image(image_question, uploaded_image)
                    st.session_state.history.append(("image", image_question, result))
                except Exception as e:
                    st.error(f"Request failed: {e}")

st.divider()
st.subheader("Conversation")
for kind, q, r in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q + (" 📷" if kind == "image" else ""))
    with st.chat_message("assistant"):
        st.write(r.get("answer", "(no answer)"))
        sources = r.get("sources", [])
        if sources:
            st.caption("Sources: " + ", ".join(sources))
