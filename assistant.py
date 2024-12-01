import ollama
from langchain_chroma import Chroma 
from langchain.prompts import ChatPromptTemplate
from embed import embed_it

CHROMA_PATH='vectorized-database'

system_prompt = """
Please respond to the question using only the context provided below:

{context_text}

---

Provide your answer based on the context above: {prompt}
"""



def chat(prompt):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_it())
    
    results = db.similarity_search_with_score(prompt, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    yield "ASSISTANT: "
    
    response = ollama.chat(
        model="llama3.2",
        stream=True,
        messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': f"Context: {context_text}, Question: {prompt}"
            }
        ]
    )

    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    yield f"\nSources: {sources}"

    


# if __name__=="__main__":
#     prompt=''
#     while prompt!='/bye':
#         prompt=input('\nUSER: ')
#         for content in chat(prompt):
#             print(content, end='', flush=True)



























# import ollama

# convo =[]

# def chat(prompt):
#     convo.append(
#         {'role':'user', 'content':prompt}
#     )
#     response=''
#     stream=ollama.chat(model='llama3', messages=convo, stream=True)
#     yield f'ASSISTANT: '

#     for chunk in stream:
#         content=chunk['message']['content']
#         response+=content
#         yield content

#     convo.append(
#         {'role':'assistant', 'content':response}
#     )

# if __name__=="__main__":
#     prompt=''
#     while prompt!='/bye':
#         prompt=input('\nUSER: \n')
#         for content in chat(prompt):
#             print(content, end='', flush=True)

# response_text = model.invoke(final_prompt)
