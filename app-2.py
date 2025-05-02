import streamlit as st
from openai import OpenAI
import os
import tempfile
import time
from pathlib import Path
import json
from streamlit.components.v1 import html

# Bibliothèques pour traiter différents types de documents
import PyPDF2
import docx
import io
import re

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Assistant IA - Dialogue & Q&A sur Documents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour améliorer l'interface
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: row;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #f0f2f6;
}
.chat-message.assistant {
    background-color: #e3f2fd;
}
.chat-message .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 1rem;
}
.chat-message .message {
    flex-grow: 1;
}
.chat-message .doc-indicator {
    font-size: 0.8rem;
    color: #424242;
    margin-top: 0.5rem;
}
.document-pill {
    display: inline-block;
    background-color: #e0e0e0;
    padding: 0.2rem 0.5rem;
    border-radius: 1rem;
    font-size: 0.8rem;
    margin-right: 0.5rem;
    margin-top: 0.3rem;
}
.stTextInput div[data-baseweb="base-input"] {
    display: flex;
    flex-direction: row;
    align-items: center;
}
.upload-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    padding: 0.25rem;
    border-radius: 0.25rem;
    margin-right: 0.5rem;
}
.upload-button:hover {
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# Configuration de l'API Scaleway
API_BASE_URL = "https://api.scaleway.ai/f754c3d7-7ed4-4e24-a716-97f5e7aa2916/v1"
API_KEY = "5c4074e2-bad9-4bec-a8bb-a569c63ba846"

# Paramètres du modèle
MODEL = "llama-3.3-70b-instruct"
MAX_TOKENS = 800
TEMPERATURE = 0.6
TOP_P = 0.9
PRESENCE_PENALTY = 0.0
STOP_SEQUENCE = ["/stop"]

# Initialisation des variables de session si elles n'existent pas déjà
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "Tu es un assistant intelligent qui répond en français même si la question est dans une autre langue. Tu peux discuter de tout sujet et analyser des documents si l'utilisateur en fournit."}
    ]

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'documents' not in st.session_state:
    st.session_state.documents = {}  # Dictionnaire pour stocker {nom_document: contenu}

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Fonctions pour extraire le texte de différents formats de documents
def extract_text_from_pdf(file):
    """Extrait le texte d'un fichier PDF"""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name
    
    with open(temp_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    
    os.unlink(temp_file_path)  # Supprimer le fichier temporaire
    return text

def extract_text_from_docx(file):
    """Extrait le texte d'un fichier DOCX"""
    doc = docx.Document(io.BytesIO(file.getvalue()))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file):
    """Extrait le texte d'un fichier TXT"""
    return file.getvalue().decode("utf-8")

def process_file(uploaded_file):
    """Traite le fichier uploadé et extrait son contenu textuel"""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == '.docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == '.txt':
        return extract_text_from_txt(uploaded_file)
    else:
        st.error(f"Format de fichier non pris en charge: {file_extension}")
        return ""

def get_chunks(text, chunk_size=3000, overlap=200):
    """Divise le texte en chunks pour gérer les documents longs"""
    chunks = []
    if not text:
        return chunks
        
    # Simple chunking par caractères avec overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text) and end - start == chunk_size:
            # Chercher la fin de phrase/paragraphe la plus proche pour une coupure propre
            for i in range(end, max(end - 200, start), -1):
                if i < len(text) and text[i] in ['.', '!', '?', '\n'] and (i+1 >= len(text) or text[i+1].isspace()):
                    end = i + 1
                    break
                    
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    
    return chunks

def create_context_for_question(question, documents, max_length=5000):
    """Crée un contexte pertinent pour la question en utilisant les documents disponibles"""
    if not documents:
        return ""
    
    # Combine tous les documents en un seul texte pour l'analyse
    all_text = ""
    for doc_name, doc_content in documents.items():
        all_text += f"\n\n--- DOCUMENT: {doc_name} ---\n\n"
        all_text += doc_content
    
    # Si le texte total est petit, on utilise tout
    if len(all_text) <= max_length:
        return all_text
    
    # Pour les documents plus grands, on essaie de trouver les chunks les plus pertinents
    chunks = get_chunks(all_text)
    
    # Extrait les mots significatifs de la question
    words = re.findall(r'\b\w+\b', question.lower())
    stopwords = set(['le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'à', 'au', 'aux', 
                    'de', 'du', 'en', 'ce', 'cette', 'ces', 'qui', 'que', 'quoi', 'où', 
                    'comment', 'pourquoi', 'quand', 'quel', 'quelle', 'quels', 'quelles'])
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Score chaque chunk en fonction du nombre de mots-clés qu'il contient
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        score = 0
        chunk_lower = chunk.lower()
        for keyword in keywords:
            score += chunk_lower.count(keyword)
        chunk_scores.append((i, score))
    
    # Trie les chunks par score et prend les meilleurs jusqu'à atteindre max_length
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    selected_chunks = []
    total_length = 0
    
    for chunk_idx, _ in sorted_chunks:
        if total_length + len(chunks[chunk_idx]) <= max_length:
            selected_chunks.append(chunk_idx)
            total_length += len(chunks[chunk_idx])
        else:
            break
    
    # Si aucun chunk n'a de score positif, on prend simplement les premiers chunks
    if not selected_chunks:
        current_length = 0
        for i, chunk in enumerate(chunks):
            if current_length + len(chunk) <= max_length:
                selected_chunks.append(i)
                current_length += len(chunk)
            else:
                break
    
    # Trie les indices pour préserver l'ordre original des documents
    selected_chunks.sort()
    context = "\n\n".join([chunks[i] for i in selected_chunks])
    
    return context

def add_message(role, content, attached_docs=None):
    """Ajoute un message à l'interface de chat et à l'historique de conversation"""
    # Ajoute à l'état de session pour l'affichage
    st.session_state.chat_messages.append({
        "role": role, 
        "content": content,
        "attached_docs": attached_docs
    })
    
    # Ajoute à l'historique de conversation pour le LLM
    st.session_state.conversation_history.append({
        "role": role,
        "content": content
    })

# Fonction pour afficher les messages de chat avec un style amélioré
def display_messages():
    for msg in st.session_state.chat_messages:
        role = msg["role"]
        content = msg["content"]
        attached_docs = msg.get("attached_docs", None)
        
        if role == "user":
            avatar_url = "https://api.dicebear.com/7.x/avataaars/svg?seed=user"
            bg_color = "user"
        else:
            avatar_url = "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"
            bg_color = "assistant"
        
        # Construit l'affichage du message
        message_html = f"""
        <div class="chat-message {bg_color}">
            <img src="{avatar_url}" class="avatar">
            <div class="message">
                {content}
        """
        
        # Ajoute des indicateurs pour les documents attachés
        if attached_docs:
            message_html += '<div class="doc-indicator">Documents attachés: '
            for doc in attached_docs:
                message_html += f'<span class="document-pill">📄 {doc}</span>'
            message_html += '</div>'
        
        message_html += "</div></div>"
        
        st.markdown(message_html, unsafe_allow_html=True)

# Fonction pour gérer l'envoi du message lorsque l'utilisateur appuie sur Cmd+Enter ou Ctrl+Enter
def handle_submit():
    user_input = st.session_state.user_input
    attached_docs = []
    
    # Récupère les fichiers attachés s'ils existent
    if "file_upload" in st.session_state and st.session_state.file_upload:
        for uploaded_file in st.session_state.file_upload:
            file_name = uploaded_file.name
            document_text = process_file(uploaded_file)
            if document_text:
                st.session_state.documents[file_name] = document_text
                attached_docs.append(file_name)
    
    if not user_input.strip() and not attached_docs:
        st.warning("Veuillez entrer un message ou joindre un document.")
        return
    
    # Marquer comme soumis
    st.session_state.submitted = True

    # Si aucun message mais des documents attachés, on pose une question générique
    if not user_input.strip() and attached_docs:
        user_input = "Peux-tu analyser ce(s) document(s) et me dire ce qu'il(s) contien(nen)t?"
    
    # Ajoute le message à l'interface
    add_message("user", user_input, attached_docs)
    
    # Traitement du message et génération de la réponse... (le reste de la logique d'envoi)
    # (Le code existant pour le traitement de la réponse)

# Interface utilisateur Streamlit
def main():
    # Sidebar pour les paramètres
    with st.sidebar:
        st.title("⚙️ Paramètres")
        
        # Affichage des documents déjà téléchargés
        if st.session_state.documents:
            st.write("### Documents disponibles:")
            for doc_name in st.session_state.documents.keys():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc_name}")
                with col2:
                    if st.button("❌", key=f"delete_{doc_name}"):
                        del st.session_state.documents[doc_name]
                        st.success(f"Document '{doc_name}' supprimé")
                        st.rerun()
        
        # Paramètres avancés (collapsible)
        with st.expander("⚙️ Paramètres avancés", expanded=False):
            st.slider("Température", min_value=0.0, max_value=1.0, value=TEMPERATURE, step=0.1, key="temperature", 
                      help="Contrôle la créativité des réponses (0=déterministe, 1=créatif)")
            st.slider("Longueur maximale", min_value=100, max_value=2000, value=MAX_TOKENS, step=100, key="max_tokens",
                      help="Nombre maximum de tokens dans la réponse")
            
            if st.button("Réinitialiser la conversation"):
                st.session_state.conversation_history = [
                    {"role": "system", "content": "Tu es un assistant intelligent qui répond en français même si la question est dans une autre langue. Tu peux discuter de tout sujet et analyser des documents si l'utilisateur en fournit."}
                ]
                st.session_state.chat_messages = []
                st.session_state.documents = {}
                st.success("Conversation réinitialisée!")
                st.rerun()

    # Contenu principal
    st.title("🤖 Assistant IA - Dialogue & Documents")
    st.write("Discutez avec l'assistant et attachez des documents au besoin pour poser des questions dessus.")
    
    # Affichage des messages de chat
    display_messages()
    
    # Zone de saisie pour la question avec bouton "trombone" pour upload
    st.write("### Envoyez un message")
    
    # Layout pour le champ de saisie et l'upload de fichier côte à côte
    col1, col2 = st.columns([6, 1])
    
    with col2:
        uploaded_files = st.file_uploader(
            "Joindre un document", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_upload"
        )
    
    # Astuce visuelle pour indiquer le raccourci clavier
    st.markdown("""
    <div style="text-align: right; font-size: 0.8em; color: #888; margin-top: -15px; margin-bottom: 5px;">
        Envoyez avec Cmd+Enter (Mac) ou Ctrl+Enter (PC)
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.submitted:
        # Si un message vient d'être soumis, réinitialiser
        st.session_state.user_input = ""
        st.session_state.submitted = False
    
    # Affichage du champ de saisie
    with col1:
        user_input = st.text_area("Votre message:", height=100, key="user_input", 
                                 label_visibility="collapsed")

    # Zone des fichiers attachés à ce message
    attached_docs = []
    
    if uploaded_files:
        st.write("Documents à joindre à ce message:")
        cols = st.columns(4)
        
        for i, uploaded_file in enumerate(uploaded_files):
            col_index = i % 4
            with cols[col_index]:
                file_name = uploaded_file.name
                st.write(f"📎 {file_name}")
                
                # Traite le fichier
                with st.spinner(f"Traitement de {file_name}..."):
                    document_text = process_file(uploaded_file)
                    if document_text:
                        # Stocke le document
                        st.session_state.documents[file_name] = document_text
                        attached_docs.append(file_name)
                        st.success(f"✓ ({len(document_text)} caractères)")
                    else:
                        st.error("Échec")
    
    # Bouton d'envoi + Callback pour le raccourci clavier
    if st.button("Envoyer", key="send_message"):
        if not user_input.strip() and not attached_docs:
            st.warning("Veuillez entrer un message ou joindre un document.")
            st.stop()
        
        # Marquer comme soumis
        st.session_state.submitted = True

        # Si aucun message mais des documents attachés, on pose une question générique
        if not user_input.strip() and attached_docs:
            user_input = "Peux-tu analyser ce(s) document(s) et me dire ce qu'il(s) contien(nen)t?"
        
        # Ajoute le message à l'interface
        add_message("user", user_input, attached_docs)
        
        # Prépare le contexte des documents si applicable
        document_context = ""
        if attached_docs:
            # Sélectionne seulement les documents joints à ce message
            docs_for_context = {name: content for name, content in st.session_state.documents.items() if name in attached_docs}
            document_context = create_context_for_question(user_input, docs_for_context)
        
        # Prépare le prompt avec le contexte du document si nécessaire
        if document_context:
            # Prépare les messages pour l'API avec le contexte des documents
            system_message = {"role": "system", "content": "Tu es un assistant intelligent qui répond en français. Tu peux analyser des documents fournis par l'utilisateur et répondre à des questions à leur sujet."}
            
            # Construit la liste des messages pour l'API
            # Inclut les messages précédents pour maintenir la cohérence de la conversation
            messages = [system_message]
            
            # Ajoute les messages précédents mais pas le dernier (qui sera traité spécialement)
            for msg in st.session_state.conversation_history[1:-1]:
                messages.append(msg)
            
            # Prépare le dernier message de l'utilisateur avec le contexte des documents
            full_prompt = f"""Voici ma question: {user_input}

Je joins également les documents suivants pour référence:

{document_context}

Réponds à ma question en te basant sur les informations fournies dans ces documents si pertinent."""
            
            # Remplace le dernier message par celui avec le contexte
            messages.append({"role": "user", "content": full_prompt})
        else:
            # Pas de document attaché, utilise les messages tels quels
            system_message = {"role": "system", "content": "Tu es un assistant intelligent qui répond en français même si la question est dans une autre langue."}
            messages = [system_message] + st.session_state.conversation_history[1:]
        
        # Initialise le client OpenAI
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
        
        # Affiche un placeholder pour la réponse en streaming
        with st.spinner("Génération de la réponse..."):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Appel de l'API en mode streaming
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=st.session_state.get("max_tokens", MAX_TOKENS),
                    temperature=st.session_state.get("temperature", TEMPERATURE),
                    top_p=TOP_P,
                    presence_penalty=PRESENCE_PENALTY,
                    stop=STOP_SEQUENCE,
                    stream=True,
                )
                
                # Conteneur pour afficher la réponse en streaming
                with st.container():
                    assistant_avatar = "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"
                    # Commence l'affichage du message
                    message_container = st.empty()
                    
                    # Affichage en temps réel de la réponse
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            
                            # Met à jour l'affichage du message
                            message_html = f"""
                            <div class="chat-message assistant">
                                <img src="{assistant_avatar}" class="avatar">
                                <div class="message">
                                    {full_response}
                                </div>
                            </div>
                            """
                            message_container.markdown(message_html, unsafe_allow_html=True)
                            time.sleep(0.01)  # Ralentit légèrement le stream pour une meilleure lisibilité
                
                # Ajoute la réponse complète à l'historique de conversation
                add_message("assistant", full_response)
                
                # Vide les champs pour le prochain message
                if "user_input" in st.session_state:
                  del st.session_state.user_input
                if "file_upload" in st.session_state:
                  del st.session_state.file_upload
                
                # Rerun pour actualiser l'interface
                st.rerun()
                
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse: {str(e)}")

    # On n'a plus besoin de ce composant car on utilise l'approche avec on_change

if __name__ == "__main__":
    main()