from milvus_rag import MilvusRAG


class RAGAgent:
    def __init__(self, llm, milvus_rag: MilvusRAG):
        self.llm = llm
        self.milvus_rag = milvus_rag

    def run(self, query):
        docs = self.milvus_rag.search(query, top_k=3)
        context = "\n\n".join([doc for doc, _ in docs])
        prompt = f"""
        You are a healthcare assistant. Use the context below to answer the question.

        Context:
        {context}

        Question:
        {query}
        """
        response = self.llm(prompt=prompt, max_tokens=512)
        answer = response["choices"][0]["text"].strip()
        return {"answer": answer, "retrieved_docs": [doc for doc, _ in docs]}
