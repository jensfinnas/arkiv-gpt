import os
from glob import glob
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage, SimpleDirectoryReader

from llama_index.llms import OpenAI
import openai
from settings import OPENAI_API_KEY

DOC_DIR = 'data/docs/sollefteåbladet'
INDEX_STORAGE = "data/vector_storage/sollefteåbladet-index"
OPENAI_MODEL = "gpt-3.5-turbo" # "gpt-4"

st.set_page_config(page_title="💬 Chatta med Sollefteåbladet", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = OPENAI_API_KEY

st.title("💬 Chatta med Sollefteåbladet")
st.info("Demo för AI-kurs", icon="📃")


if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Fråga nåt från arikvet!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Laddar dokument..."):
        service_context = ServiceContext.from_defaults(llm=OpenAI(model=OPENAI_MODEL, temperature=0.5, ))
        if os.path.exists(INDEX_STORAGE):
            print(f"Läs in befintligt index: {INDEX_STORAGE}")
            # TODO: Load index from disk
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)
            # load index
            index = load_index_from_storage(storage_context, service_context=service_context)

        
        else:
            print("Skapa nytt index")
            docs = []
            for fp in glob(os.path.join(DOC_DIR, "*.txt")):
                with open(fp) as f:
                    text = f.read()
                    heading = text.split("\n")[0].strip()
                    doc = Document(
                        text=text,
                        metadata={
                            "heading": heading,
                        }
                    )
                    docs.append(doc)


            index = VectorStoreIndex.from_documents(docs, service_context=service_context)            

            index.storage_context.persist(INDEX_STORAGE)
        return index

index = load_data()

system_prompt="Du är en svenskspråkig researchassistent. Du svarar på frågor utifrån Solleftåbladets arkiv. Du beskriver endast det du kan läsa i dokumentindexet. Om svaret på frågan inte finns i indexet svara du att du inte vet. Hallucinera inte. Du svarar vänligt och med ett lättbegripligt språk."
chat_engine = index.as_chat_engine(chat_mode="best", verbose=True, system_prompt=system_prompt)

if prompt := st.chat_input("Din fråga"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])



# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Tänker..."):
            response = chat_engine.chat(prompt)
            print(response.sources)
            print(response.source_nodes)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            print(message)
            st.session_state.messages.append(message) # Add response to message history
            if len(response.source_nodes) == 0:
                st.info("Det här svaret bygger inte på några artikelkällor")
            for source in response.source_nodes:
                if "heading" in source.node.metadata:
                    heading = "Källa: " + source.node.metadata["heading"]
                else:
                    heading = "Källa"

                expander = st.expander(heading)
                expander.write(source.node.text)