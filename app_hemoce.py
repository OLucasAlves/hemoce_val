from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import uuid
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


load_dotenv()
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\LucasAlvesRibeiro\\Downloads\\jusec-chatbot-4fca0b96d3eb.json"

#CONNECTION_STRING = "postgresql+psycopg2://postgres:1234@localhost:5432/hemoce"
CONNECTION_STRING = "postgresql+psycopg2://postgres:ae6XC#zi@34.56.100.85:5432/hemoce"
COLLECTION_NAME = 'base'
SCORE_THRESHOLD = 0.6  

# Inicializar o embedding model (o mesmo usado para criar os embeddings no banco)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")

db = PGVector(embedding_function=embeddings,
              collection_name=COLLECTION_NAME,
              connection_string=CONNECTION_STRING,
              distance_strategy=DistanceStrategy.COSINE,
              use_jsonb=True)

# index_name = "base"

# db = PineconeVectorStore(
#         index_name="base", embedding=embeddings
#     )

# Tools
def consulta_base(query: str) -> str:
    """
    Base de dados para RAG

    Args:
        query: consulta à base de dados
    """
    results = db.similarity_search_with_score(query, k=100)
    results = [doc.page_content for doc, score in results if score <= (1 - SCORE_THRESHOLD)]
    print(results)
    print(len(results))
    print("estou aqui")
    print(query)
    return results

model_with_tools = model.bind_tools([consulta_base])

default_prompt = """
        <role>
            Você é um assistente virtual do Hemoce.
            Você deverá responder a perguntas relacionadas a doação de sangue.
            Só envie uma mensagem de saudação uma única vez.
        </role>
        
        <instructions>
            Ao utiizar a ferramneta de consulta a base vectorial não mande mensagem de saudação.
            Não repita a pergunta do usuário.
            Utilize a base vectorial para responder a pergunta do usuário.
            Não utilize nenhuma outra fonte de resposta.
            Caso a resposta não esteja em seu contexto, faça uma consulta no banco vetorial disponibilizado.      
            Caso a pergunta ou o assunto fuja do escopo, informe que você não pode ajudá-lo com isso.
            Ao fazer consultas à base de dados, faça de maneira direta, de preferência usando palavras chave.
            Ao final pergunte se pode ajudar em algo mais.
            Seja gentil e breve ao responder o usuário.
        </instructions>
    """ 

# Nodes
def call_model(state: MessagesState, config: RunnableConfig):
    
    # message = [SystemMessage(content=default_prompt)] + state["messages"]
    # response = model_with_tools.invoke(message, config)
    # #print("RESPONSE DO MODEL:", response)
    # return {"messages": response}

    # Verifica se o SystemMessage já está no histórico
    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        messages = [SystemMessage(content=default_prompt)] + state["messages"]
    else:
        messages = state["messages"]

    response = model_with_tools.invoke(messages, config)
    #print(response)
    return {"messages": response}

# Graph
start_graph = StateGraph(MessagesState)

start_graph.add_node("call_model", call_model)
start_graph.add_node("tools", ToolNode([consulta_base]))

start_graph.add_edge(START, "call_model")
start_graph.add_conditional_edges("call_model", tools_condition, ['tools', END])
start_graph.add_edge("tools", "call_model")

memory = MemorySaver()
start_graph = start_graph.compile(checkpointer=memory)



# 🚀 Streamlit App
st.set_page_config(page_title="Assistente do Hemoce", page_icon="🩸") # Título atualizado
st.title("🤖 Assistente Virtual do Hemoce")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    

# Botão de nova conversa agora também pode limpar o ID se necessário, mas o ideal é deixar por sessão.
if st.button("🔄 Nova conversa"):
    st.session_state.chat_history = []
    st.session_state.thread_id = str(uuid.uuid4()) # Gera um novo ID para uma conversa limpa
    st.rerun()

# Configuração da conversa usando o ID da sessão
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Mostrar histórico (sem alteração aqui)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input do usuário
user_input = st.chat_input("Digite sua pergunta...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Pensando..."):
        response = start_graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    if response and "messages" in response and response["messages"]:
        final_response_message = response["messages"][-1]
        if isinstance(final_response_message, AIMessage):
            final_response = final_response_message.content
        else:
            # Fallback caso a última mensagem não seja do AI (improvável)
            final_response = "Ocorreu um erro ao processar a resposta."
    else:
        final_response = "Não recebi uma resposta válida."
        
    # Exibir e salvar a resposta do assistente
    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
    with st.chat_message("assistant"):
        st.markdown(final_response)