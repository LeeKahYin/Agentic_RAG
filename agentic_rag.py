import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertexai-client-api.json"

import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import  SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

db_path_llm = 'papers_db'
db_path_bio = 'bio_db'

@st.cache_resource
def create_rag_graph():
    # Initialize LLM and embedding
    llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai", temperature=0)
    embedding = VertexAIEmbeddings(model_name="text-embedding-004")

    # Load vector database
    llm_vectordb = Chroma(
                    persist_directory=db_path_llm,
                    embedding_function=embedding
                )
    
    bio_vectordb = Chroma(
                    persist_directory=db_path_bio,
                    embedding_function=embedding
                )
    
    # Create tool that retrieve from the LLM database
    @tool(response_format="content_and_artifact")
    def retrieve_llm(query:str):
        """Retrieve info about open-source LLMs"""
        retrieved_docs = llm_vectordb.max_marginal_relevance_search(query, k=5, fetch_k=20)
        content = "\n\n".join(
            (
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            )
            for doc in retrieved_docs
        )
        return content, retrieved_docs
    
    # Create tool that retrieve from the biology database
    @tool(response_format="content_and_artifact")
    def retrieve_biology(query:str):
        """Retrieve biology knowledge about infectious disease and immune system"""
        retrieved_docs = bio_vectordb.max_marginal_relevance_search(query, k=5, fetch_k=20)
        content = "\n\n".join(
            (
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            )
            for doc in retrieved_docs
        )
        return content, retrieved_docs
    
        

    # Given the input, it will decide to retrieve using the retriever tool or simply respond
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve_llm, retrieve_biology])

        tool_decision_prompt = [
            SystemMessage(content=(
                "You are a helpful assistant that answers questions using two knowledge bases: "
                "- Use 'retrieve_llm' for questions about open-source large language models like Mixtral, Phi, DeepSeek, Qwen, Gemma and Llama. "
                "- Use 'retrieve_biology' for questions about biology topics such as infectious disease and the immune system. "
                "Only answer directly if the question is completely unrelated to open-source LLMs and biology (like math, history or general trivia). "
                "If unsure, default to using a tool to retrieve relevant information from a knowledge base first."
            ))
        ] + state["messages"]
        response = llm_with_tools.invoke(tool_decision_prompt)
        return {"messages": [response]}
    
    # Execute retrieval
    tools = ToolNode([retrieve_llm, retrieve_biology])
    

    # This node generates answers using retrieved content
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]
        #print(tool_messages)

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(content=system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}
        
    
    # Define nodes
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    # Define the flow of the application
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",      
        tools_condition,
        {END: END, "tools": "tools"}
    )

    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph



# Streamlit app
def main():
    st.title("🤖 Agentic RAG Chatbot")
    #st.markdown("Ask questions about the open source large language models.")
 
    # Initialize RAG system
    graph = create_rag_graph()

    if graph is None:
        st.error("Failed to initialize the system. Please check your configuration.")
        st.stop()
    
    # Initialize session state for UI messages 
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = []
     
    # Initialize thread counter for memory management
    if "thread_counter" not in st.session_state:
        st.session_state.thread_counter = 0

    # Thread ID for memory persistence
    thread_id = f"conversation_thread_{st.session_state.thread_counter}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Display chat history
    for message in st.session_state.ui_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(
                            f"**{i}.** *{src['title']}* — Page {src['page']}  \n"
                            f"`{src['source']}`"
                        )
    
    # Chat input
    if user_input := st.chat_input("Type your question here"):
        # Add user message to UI
        st.session_state.ui_messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare state
                state = {"messages": [{"role": "user", "content": user_input}]}
                
                # Get response from RAG system 
                result = graph.invoke(state, config=config)
                messages = result["messages"]
                response = messages[-1].content

                # Check if the previous message is a ToolMessage (indicating tool was used)
                used_tool = len(messages) >= 2 and messages[-2].type == "tool"
                source_docs = []

                if used_tool:
                    tool_message = messages[-2]
                    for doc in tool_message.artifact:
                        metadata = doc.metadata
                        title = metadata.get("title", "Unknown Title")
                        page = metadata.get("page", 0) + 1  # Convert from 0-indexed
                        source = metadata.get("source", "Unknown Source")
                        source_docs.append({
                            "title": title,
                            "page": page,
                            "source": source
                        })

                                
                # Display response
                st.markdown(response)

                # Display sources
                if source_docs:
                    with st.expander("Sources"):
                        for i, src in enumerate(source_docs, 1):
                            st.markdown(
                                f"**{i}.** *{src['title']}* — Page {src['page']}  \n"
                                f"`{src['source']}`"
                            )
                
                # Add to UI messages so that all previous chat history remained display
                st.session_state.ui_messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": source_docs
                })
                

if __name__ == "__main__":
    main()