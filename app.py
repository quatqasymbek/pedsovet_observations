import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="PedSovet AI – MVP",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 PedSovet AI — MVP")

st.markdown("""
Welcome to the **PedSovet AI** prototype.

This tool helps schools:

• transcribe meeting audio (Kazakh/Russian)  
• analyze teacher survey voice notes  
• extract topics, decisions, and problems  
• evaluate compliance with meeting criteria  
• generate draft protocols and insights  

Use the **left sidebar** to open test modules.
""")

st.info("➡️ Start with **STT Test** in the sidebar.")
