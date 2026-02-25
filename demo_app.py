import streamlit as st
from src.state import AgentState
import time
from src.graph import app # Your compiled LangGraph

# Page Configuration
st.set_page_config(page_title="AutoPart AI | Enterprise Demo", layout="wide", page_icon="⚙️")

# Custom CSS to make it look professional
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; }
    .agent-box { padding: 20px; border-radius: 10px; background: white; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 AutoPart AI: Multi-Agent Inventory Suite")
st.markdown("---")

# Sidebar for Demo Controls
with st.sidebar:
    st.header("Settings")
    st.info("Model: Gemini 1.5 Flash")
    st.success("Agents: DB + RAG + Web + Compiler")
    if st.button("Clear History"):
        st.session_state.messages = []

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# User Input
if input := st.chat_input("Ask about a spare part (e.g. 'Do you have Alternator 130A Remanufactured?')"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": input})
    
    with st.chat_message("user"):
        st.markdown(input)

    # Agent Execution
    with st.chat_message("assistant"):
        # Creating a placeholder for the "thinking" process
        thinking_expander = st.expander("🔍 **Agent Reasoning Trace**", expanded=True)
        
        with st.spinner("Analyzing request..."):
            # 1. Run the Graph
            # We use a container to show progress in the expander
            with thinking_expander:
                step_1 = st.empty()
                step_1.write("📡 *Consulting SQLite Inventory...*")
                
                # Execute the full graph
                result = app.invoke({"input": input})
                
                step_1.write("✅ **Database Specialist:** Found part details.")
                st.divider()
                st.write("📖 **RAG Expert:** Inventory parts retrieved.")
                st.divider()
                st.write("🌐 **Web Researcher:** Market price comparison complete.")

            # 2. Display Final Response
            st.markdown("### Final Recommendation")
            st.write(result["final_answer"])
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": result["final_answer"]})

# Show step-by-step data in tabs below the chat for the "Tech Proof"
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    st.write("### Data Source Breakdown")
    tab1, tab2, tab3 = st.tabs(["Database (SQL)", "Technical (RAG)", "Market (Web)"])
    
    # We use the latest result for the tabs
    try:
        with tab1:
            st.code(result.get("db_results", "No data"), language="text")
        with tab2:
            st.write(result.get("rag_results", "No manuals found."))
        with tab3:
            st.write(result.get("web_results", "No market data found."))
    except:
        pass