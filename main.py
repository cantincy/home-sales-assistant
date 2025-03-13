from langchain.chains.llm import LLMChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

persist_directory = './chroma_db'
file_path = './data.txt'


@st.cache_resource
def get_chain():
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )

    docs = splitter.create_documents([text])

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(),
    )

    vector_store.add_documents(docs)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7, "k": 3}
    )

    memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="chat_history",
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            你是一个专业的房产销售顾问，你需要根据客户的需求，为客户提供房产销售建议。
            历史对话：
            ```
            {chat_history}
            ```
            """
        ),
        (
            "human",
            "{input}"
        )
    ])

    chain = LLMChain(
        llm=ChatOpenAI(),
        prompt=prompt,
        memory=memory,
        verbose=True
    )

    return chain


def main():
    st.header("房产销售顾问")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for message in st.session_state.messages:
            role, content = message["role"], message["content"]
            st.chat_message(role).write(content)

    user_input = st.chat_input()

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        chain = get_chain()

        retriever = chain.memory.retriever
        docs = retriever.get_relevant_documents(user_input)

        if not docs or len(docs) == 0:
            response = "抱歉，这个问题需要进一步确认。"
        else:
            response = chain.predict(input=user_input)
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
