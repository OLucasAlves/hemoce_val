from vectorstore.pg_vector import db
from config import SCORE_THRESHOLD

# def consulta_base(query: str) -> str:
   
#     results = db.similarity_search_with_score(query, k=100)
#     results = [doc.page_content for doc, score in results if score <= (1 - SCORE_THRESHOLD)]
#     return results

def consulta_base(query: str) -> str:
    """
    Base de dados para RAG

    Args:
        query: consulta à base de dados
    """
    # results = db.similarity_search_with_score(query, k=100)
    # results = [doc.page_content for doc, score in results if score <= (1 - SCORE_THRESHOLD)]
    # print(len(results))
    # #print(results)
    # try:
    #     for doc in results:
    #         print(f"Fonte: {doc.metadata}")
    # except Exception as e:
    #     print("Erro",e)        
    # return results

    raw_results = db.similarity_search_with_score(query, k=10)
    filtered_results = [(doc, score) for doc, score in raw_results if score <= (1 - SCORE_THRESHOLD)]
    print(f"Query: ",query)
    #for doc, score in filtered_results:
        #print(f"Conteúdo: {doc.page_content}")
        #print(f"Fonte: {doc.metadata}")
        
        #print(f"Score: {score}\n")

    return [doc.page_content for doc, _ in filtered_results]


