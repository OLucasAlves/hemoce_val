from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from agents.model_agent import model_with_tools
from tools.consulta_base import consulta_base
from langchain_core.messages import SystemMessage
# Como exemplo dada a seguinte pergunta do usuário 'liste os Professores contratados por tempo determinado no processo Nº22001.054543/2024-55'
            #  a conculta deveria ser por 'processo Nº22001.054543/2024-55' 
def call_model(state, config):
    default_prompt = """
        <role>
            Você é um assistente virtual da casa civil.
            Você deverá responder a perguntas relacionadas ao diário oficial do estado do Ceará.
        </role>

        <instructions>
            Caso a resposta não esteja em seu contexto, faça uma consulta no banco vetorial disponibilizado.
            Caso a pergunta ou o assunto fuja do escopo, informe que você não pode ajudá-lo com isso.
            Ao fazer consultas à base de dados, faça de maneira direta, de preferência usando palavras chave.
            Tenha em mente que base de dados consultada é um banco vetorial. Evite realizar consultas complexas.
            Caso a consulta seja muito complexa realize de passo a passo.
            Seja gentil e breve ao responder o usuário.
        </instructions>
    """
    messages = [SystemMessage(content=default_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages, config)
    return {"messages": state["messages"] + [response]}

def prepare_tool_args(state):
    query = state["messages"][-1].content if state["messages"] else ""
    return {"query": query}  

def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode([consulta_base]))
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", tools_condition, ['tools', END])
    graph.add_edge("tools", "call_model")
    return graph.compile(checkpointer=MemorySaver())
