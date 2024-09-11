import streamlit as st
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define directories for each game's PDF
GAME_PDF_DIRECTORIES = {
    "Ludo": "/Users/sondaryachauhan/Desktop/GameMaster/Ludo",
    "Monopoly": "/Users/sondaryachauhan/Desktop/GameMaster/Monopoly",
    "Uno": "/Users/sondaryachauhan/Desktop/GameMaster/Uno",
    "Snake_Ladders": "/Users/sondaryachauhan/Desktop/GameMaster/Snake&Ladders",
    "Chess": "/Users/sondaryachauhan/Desktop/GameMaster/Chess"
}

# Cache for loaded and processed PDF text
@st.cache_data
def load_and_process_pdf(game_name):
    """Load and process the PDF file for the selected game from its specific folder."""
    game_directory = GAME_PDF_DIRECTORIES[game_name]
    text = ""
    for filename in os.listdir(game_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(game_directory, filename)
            raw_text = get_pdf_text_with_pdfplumber(filepath)
            text_chunks = get_text_chunks(raw_text)
            save_vector_store(text_chunks, game_name)  # Save index separately for each game
            text += f"{filename} processed.\n"
    return f"{game_name} PDFs processed successfully!\n{text}"

def get_pdf_text_with_pdfplumber(filepath):
    """Extract text from a single PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure the page text is not None
                text += page_text
    return text

def get_text_chunks(text):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def save_vector_store(text_chunks, game_name):
    """Create and save a vector store for each game."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(f"faiss_index_{game_name}")  # Save index specific to each game

def get_conversational_chain():
    """Create a conversational chain for answering questions based on the PDF content and model knowledge."""
    prompt_template = """
    You are an expert game rule advisor for multiple board and card games. Your task is to provide precise and accurate guidance based on the game's rulebook provided in a PDF format. 
    Players will ask you questions to clarify rules, resolve conflicts, and enhance their gameplay experience.

    Always refer to the rules in the PDF where possible. If the specific answer is not found in the provided PDF context, generate an accurate and relevant response using your broader knowledge of the game's mechanics, strategies, and common rules.

    Ensure that your answers are always related to the game in question.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, selected_game):
    """Handle user input and generate a response based on the stored vectors for the selected game."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the vector store for the selected game
    new_db = FAISS.load_local(f"faiss_index_{selected_game}", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
        
    except Exception as e:
        if 'ResourceExhausted' in str(e):
            st.error("API quota exhausted. Please try again after some time.")
        else:
            st.error(f"An error occurred: {str(e)}")
    
    # Adding a delay to prevent exceeding rate limits
    time.sleep(1)

# Streamlit application
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with GameMasterAI 🎲")

    # Initialize session state for user question if not already set
    if "user_question" not in st.session_state:
        st.session_state["user_question"] = ""

    # Track which game the user selects
    if "selected_game" not in st.session_state:
        st.session_state["selected_game"] = None

    # Display the selected game in the text input label
    if st.session_state["selected_game"]:
        input_label = f"Ask a Question regarding {st.session_state['selected_game']}:"
    else:
        input_label = "Ask a Question regarding the game:"

    # Text input field
    user_question = st.text_input(input_label, st.session_state["user_question"])

    # Add "Clear" button to reset the user input field
    if st.button("Clear"):
        st.session_state["user_question"] = ""  # Reset the input field

    # Process the user question
    if user_question and st.session_state["selected_game"]:
        st.session_state["user_question"] = user_question  # Update session state
        user_input(user_question, st.session_state["selected_game"])
    elif user_question and not st.session_state["selected_game"]:
        st.error("Please select a game first.")

    # Sidebar for game selection
    with st.sidebar:
        st.title("🎮 Choose a game:")

        # Buttons for each game
        if st.button("Ludo"):
            with st.spinner("Processing Ludo rules..."):
                st.session_state["selected_game"] = "Ludo"
                message = load_and_process_pdf("Ludo")
                st.success(message)

        if st.button("Monopoly"):
            with st.spinner("Processing Monopoly rules..."):
                st.session_state["selected_game"] = "Monopoly"
                message = load_and_process_pdf("Monopoly")
                st.success(message)

        if st.button("Uno"):
            with st.spinner("Processing Uno rules..."):
                st.session_state["selected_game"] = "Uno"
                message = load_and_process_pdf("Uno")
                st.success(message)

        if st.button("Snake & Ladders"):
            with st.spinner("Processing Snake & Ladders rules..."):
                st.session_state["selected_game"] = "Snake_Ladders"
                message = load_and_process_pdf("Snake_Ladders")
                st.success(message)

        if st.button("Chess"):
            with st.spinner("Processing Chess rules..."):
                st.session_state["selected_game"] = "Chess"
                message = load_and_process_pdf("Chess")
                st.success(message)

if __name__ == "__main__":
    main()


