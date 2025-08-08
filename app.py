import streamlit as st
from config import logger
from core.query import answer_question
from core.llm import get_available_models, load_model, check_llm_status
from tracing import start_span, CHAIN, INPUT_VALUE, OUTPUT_VALUE, STATUS_OK

st.set_page_config(page_title="Ask a Question", layout="wide")
st.title("💬 Talk to Your Documents")

st.markdown("""
Upload and index your documents using the sidebar.
Then ask natural-language questions powered by a **local, private LLM**.

**No cloud, no leaks — just answers.**
""")
st.markdown("---")

# ───────────────────────────────────────
# 🔸 Sidebar: LLM Settings
# ───────────────────────────────────────
with st.sidebar.expander("🧠 LLM Settings", expanded=True):
    llm_status = check_llm_status()

    llm_models = get_available_models() if llm_status["server_online"] else []
    loaded_model = llm_status.get("current_model")

    selected_model = st.selectbox("Model", llm_models, index=llm_models.index(loaded_model) if loaded_model in llm_models else 0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.05)

    if st.button("📦 Load Model", disabled=not selected_model):
        with st.spinner("Loading model..."):
            if load_model(selected_model):
                st.success(f"✅ Loaded: {selected_model}")
            else:
                st.error("❌ Failed to load model.")

    st.markdown("---")
    mode = st.radio("LLM Mode", ["completion", "chat"], index=0)

# ───────────────────────────────────────
# 🔹 Chat History
# ───────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if mode == "chat":
    st.markdown("### 💬 Conversation Mode")

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with start_span("QA chain", CHAIN) as span:
            span.set_attribute("mode", "chat")
            span.set_attribute("temperature", temperature)
            span.set_attribute(INPUT_VALUE, user_input)
            answer, sources = answer_question(
                question=user_input,
                mode="chat",
                temperature=temperature,
                model=selected_model,
                chat_history=st.session_state.chat_history,
            )
            span.set_attribute(OUTPUT_VALUE, answer)
            span.set_status(STATUS_OK)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                st.markdown("#### 📁 Sources:")
                for src in sources:
                    st.markdown(f"- {src}")

# ───────────────────────────────────────
# 🔹 Completion Mode UI
# ───────────────────────────────────────
else:
    st.markdown("### 🧠 Ask a Question")
    with st.form("completion_form"):
        query = st.text_input("Your question")
        submitted = st.form_submit_button("Get Answer")

    if submitted and query:
        with start_span("QA chain", CHAIN) as span:
            span.set_attribute("mode", "completion")
            span.set_attribute("temperature", temperature)
            span.set_attribute(INPUT_VALUE, query)

            with st.spinner("🧠 Thinking..."):
                answer, sources = answer_question(
                    question=query,
                    mode="completion",
                    temperature=temperature,
                    model=selected_model,
                )
            span.set_attribute(OUTPUT_VALUE, answer)
            span.set_status(STATUS_OK)

        st.subheader("📝 Answer")
        st.markdown(answer)
        logger.info(f"LLM Answer:\n{answer}")

        if sources:
            st.markdown("#### 📁 Sources:")
            for src in sources:
                st.markdown(f"- {src}")

        st.caption(f"Mode: {mode} | Temp: {temperature} | Model: {selected_model}")
