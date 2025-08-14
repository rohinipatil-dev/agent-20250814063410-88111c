import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()


def get_system_prompt() -> str:
    return (
        "You are PyBuddy, a friendly assistant that strictly answers ONLY Python programming questions. "
        "Your behavior rules:\n"
        "- If the user's request is not clearly about Python programming (code, libraries, tooling, environment, testing, performance, packaging, type hints, data science in Python, etc.), politely refuse and ask them to reframe it in Python terms.\n"
        "- Do not answer questions about other programming languages unless comparing to Python in order to provide a Python-based answer.\n"
        "- Use only Python in code examples. Prefer minimal, correct, runnable examples.\n"
        "- Keep responses concise and helpful. Add brief explanations when needed.\n"
        "- If a question could be unsafe or unethical, refuse and suggest safer alternatives in Python."
    )


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm PyBuddy. I only answer Python programming questions. How can I help?",
            }
        ]
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4"


def build_chat_history() -> list:
    # Build messages for OpenAI API, starting with system prompt
    system_message = {"role": "system", "content": get_system_prompt()}
    convo = [system_message]
    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant"):
            convo.append({"role": m["role"], "content": m["content"]})
    return convo


def generate_response():
    messages = build_chat_history()
    try:
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=messages,
            temperature=st.session_state.temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error from OpenAI API: {e}")
        return "Sorry, I ran into an error while generating a response."


def chat_ui():
    st.set_page_config(page_title="PyBuddy - Python Chatbot", page_icon="🐍")
    st.title("🐍 PyBuddy")
    st.caption("A friendly chatbot that answers only Python programming questions.")

    init_session_state()

    with st.sidebar:
        st.subheader("Settings")
        st.session_state.temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="Lower = more deterministic, higher = more creative.",
        )
        st.session_state.model = st.selectbox(
            "Model",
            options=["gpt-4"],
            index=0,
            help="Only GPT-4 is supported in this app.",
        )
        if st.button("Clear chat"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi! I'm PyBuddy. I only answer Python programming questions. How can I help?",
                }
            ]
            st.experimental_rerun()

        st.markdown("---")
        st.markdown("Tips:")
        st.markdown(
            "- Ask about Python syntax, libraries, tooling, testing, packaging, performance.\n"
            "- Share code for debugging help.\n"
            "- If your question isn't Python-specific, I'll ask you to reframe it."
        )

    # Display chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Chat input
    user_input = st.chat_input("Ask a Python programming question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            reply = generate_response()
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat_ui()