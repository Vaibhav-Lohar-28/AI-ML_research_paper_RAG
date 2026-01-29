import streamlit as st
import time
from src.agent import get_graph

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Agentic RAG | Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Premium Dark Look) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #161622 100%);
        color: #cdd6f4;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #11111b;
        border-right: 1px solid #313244;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Input Field */
    .stTextInput input {
        background-color: #313244 !important;
        color: #cdd6f4 !important;
        border: 1px solid #45475a !important;
        border-radius: 8px !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f5e0dc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Citations / Cards */
    .citation-card {
        background-color: #313244;
        border-left: 4px solid #89b4fa;
        padding: 10px;
        margin-top: 5px;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("Initializing Agentic Core..."):
        st.session_state.agent = get_graph()
        st.success("Agent Ready!")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Research Agent")
    st.markdown("---")
    st.write("This agent uses a **ReAct-style** loop to read, grade, and synthesized answers from your local research papers.")
    
    st.markdown("### System Status")
    st.info("✅ Ollama (Llama3) Connected")
    st.info("✅ ChromaDB Index Loaded")
    
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🤖 Deep Search Assistant")
st.caption("Ask questions about your loaded AI/ML research papers.")

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            with st.expander("View Sources"):
                for doc in message["metadata"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <b>Page {doc.metadata.get('page', '?')}</b><br>
                        {doc.page_content[:150]}...
                    </div>
                    """, unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Enter your research question..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.status("Thinking...", expanded=True)
        
        try:
            # Run the Graph
            inputs = {"question": prompt}
            final_generation = ""
            final_docs = []
            
            # Stream events to show progress
            for output in st.session_state.agent.stream(inputs):
                for key, value in output.items():
                    status_placeholder.write(f"Completed step: **{key}**")
                    if key == "generate":
                        final_generation = value.get("generation", "")
                    if "documents" in value:
                        final_docs = value.get("documents", [])

            status_placeholder.update(label="Finished!", state="complete", expanded=False)
            
            # Show Answer
            message_placeholder.markdown(final_generation)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_generation,
                "metadata": final_docs
            })
            
            # Show Sources immediately for this turn
            if final_docs:
                with st.expander("View Sources"):
                    for doc in final_docs:
                        st.markdown(f"""
                        <div class="citation-card">
                            <b>Page {doc.metadata.get('page', '?')}</b><br>
                            {doc.page_content[:150]}...
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            status_placeholder.update(label="Error", state="error")
            st.error(f"An error occurred: {e}")
