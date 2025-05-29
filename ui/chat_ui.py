import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid # Para gerar IDs de conversa únicos


def render_chat_interface(start_graph): # Removido 'config' dos argumentos, será gerado internamente
    st.markdown("<h3>🤖 Assistente Virtual - CE</h3>", unsafe_allow_html=True)

    # --- Gerenciamento do ID da Conversa (thread_id) ---
    if "thread_id" not in st.session_state:
        # Inicia um novo ID de conversa se não existir na sessão
        st.session_state.thread_id = f"streamlit_session_{uuid.uuid4()}"
        # Também inicializa o histórico de exibição
        st.session_state.chat_history = []
        print(f"Nova Thread ID criada: {st.session_state.thread_id}") # Para debug

    # --- Botão Limpar ---
    if st.button("🔄 Limpar conversa"):
        # Limpa o histórico de exibição
        st.session_state.chat_history = []
        # CRÍTICO: Gera um NOVO thread_id para que o LangGraph comece uma nova memória
        st.session_state.thread_id = f"streamlit_session_{uuid.uuid4()}"
        print(f"Conversa limpa. Nova Thread ID: {st.session_state.thread_id}") # Para debug
        st.rerun()

    # --- Exibir histórico ---
    # Exibe o histórico armazenado no session_state do Streamlit
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Input do usuário ---
    user_input = st.chat_input("Digite sua pergunta...")
    if user_input:
        # Adiciona a mensagem do usuário ao histórico de exibição
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepara a mensagem para o LangGraph (apenas a nova mensagem)
        messages_for_graph = [HumanMessage(content=user_input)]

        # --- Configuração para LangGraph com memória ---
        # Usa o thread_id armazenado na sessão do Streamlit
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # --- Chama o grafo LangGraph ---
        # O checkpointer (MemorySaver) usará o thread_id para carregar/salvar o estado
        try:
            with st.spinner("Pensando..."): # Mostra um indicador de processamento
                response = start_graph.invoke({"messages": messages_for_graph}, config=config)

            # --- Processa a resposta do LangGraph ---
            # A resposta de invoke geralmente contém o estado final.
            # Precisamos extrair a última mensagem adicionada (a resposta da IA).
            if response and "messages" in response:
                # Pega a última mensagem da lista retornada pelo estado
                last_message = response["messages"][-1]
                if isinstance(last_message, AIMessage):
                    final_response = last_message.content
                else:
                    # Se a última mensagem não for AIMessage, pode ser um erro ou estado inesperado
                    # Tenta encontrar a última AIMessage na lista (caso haja ToolCalls etc.)
                    ai_messages = [m.content for m in response["messages"] if isinstance(m, AIMessage)]
                    if ai_messages:
                        final_response = ai_messages[-1] # Pega a última resposta da IA
                    else:
                        final_response = "Desculpe, não consegui processar a resposta."
                        print("Resposta inesperada do grafo:", response) # Log para debug
            else:
                final_response = "Erro ao obter resposta do assistente."
                print("Estrutura de resposta inválida:", response) # Log para debug

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            final_response = "Desculpe, tive um problema técnico."
            print(f"Erro ao invocar o grafo: {e}") # Log para debug

        # Adiciona a resposta do assistente ao histórico de exibição
        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
        with st.chat_message("assistant"):
            st.markdown(final_response)

        # Não precisa fazer st.rerun() aqui, o Streamlit atualiza com o novo chat_message

# --- Exemplo de como chamar a função no seu app Streamlit principal ---
# Supondo que você já tenha 'app' como seu grafo compilado
# No seu arquivo principal do Streamlit (ex: app.py):
#
# from your_graph_module import build_graph # Importa a função que cria o grafo
# from chat_interface import render_chat_interface # Importa a função da interface
#
# app = build_graph() # Compila o grafo UMA VEZ
#
# render_chat_interface(app)
#