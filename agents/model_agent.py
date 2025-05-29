from langchain_google_genai import ChatGoogleGenerativeAI
from tools.consulta_base import consulta_base

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=1)
model_with_tools = model.bind_tools([consulta_base])
