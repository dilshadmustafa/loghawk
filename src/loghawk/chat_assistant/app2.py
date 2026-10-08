# Run using the command below in the terminal
# streamlit run .\src\loghawk\chat_assistant\app.py
import streamlit as st
import loghawk.config as config
from loghawk.llm import get_llm_client

llm_client = get_llm_client()
# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("🧠 AI-powered Cybersecurity threat detection Companion")
st.caption("🚀 LogHawk - Your AI-powered Cybersecurity threat detection Companion")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_options = list(dict.fromkeys([
        config.LH_LLM_MODEL,
        *config.LH_LLM_FALLBACK_MODELS,
    ]))
    selected_model = st.selectbox("Choose Model", model_options)
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Security Expert
    - 🐞 Threat Detection Assistant
    - 📝 Vulnerability Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown(f"Configured provider: `{config.LH_LLM_PROVIDER}`")


SYSTEM_PROMPT = (
    "You are an expert AI cybersecurity assistant specializing in log "
    "analysis and security threat detection. Provide concise, correct "
    "solutions with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in st.session_state.message_log:
        role = "assistant" if message["role"] == "ai" else message["role"]
        messages.append({"role": role, "content": message["content"]})
    return llm_client.complete(messages, model=selected_model, temperature=0.3)

def build_prompt_chain():
    # Kept as a lightweight compatibility hook for this versioned UI.
    return st.session_state.message_log

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()
