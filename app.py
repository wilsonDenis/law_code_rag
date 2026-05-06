import streamlit as st
from rag import RAG

st.set_page_config(page_title="Assistant Code du Travail", layout="centered")

@st.cache_resource
def load_rag():
    return RAG()

def main():
    st.title("Assistant Code du Travail")
    st.markdown("Cet assistant utilise l'intelligence artificielle pour répondre à vos questions sur le droit du travail en s'appuyant sur les textes de loi.")

    rag = load_rag()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                answer, sources = rag.answer_question(prompt)
                
                sources_text = ""
                if sources:
                    sources_text = "\n\nSources : Articles " + ", ".join(sources) + " du Code du travail"
                
                full_response = answer + sources_text
                st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
