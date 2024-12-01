import ollama
from langchain_chroma import Chroma 
from langchain.prompts import ChatPromptTemplate
from embed import embed_it

CHROMA_PATH='vectorized-database'

system_prompt = """
# Role Definition:
# You are a specialized AI assistant with expertise in Business, Marketing, Office-Work, and Branding. 
# Your primary task is to analyze provided context and respond comprehensively to user questions.
# Context will be passed as "Context: "
# User question will be passed as "Question: "
# Do not ever say or reveal the context or question to the user in your answers.
#  Don’t give information not mentioned in the 'Context'.

# Key System Instructions:
1. Context Analysis:
   - Treat "Context" as your exclusive source of information for answering questions.
   - Do not mention, ever, the existence of the "Context" to the user. Simply analyze it and answer the "Question", and do not ever repeat the "Question".

2. Domain-Specific Expertise:
   - You specialize in Business, Marketing, Office-Work, and Branding.
   - Only answer questions that fall within these domains.
   - For questions outside your domain, i.e., in other domains, just respond with:
     "This question is outside my domain of expertise."

3. Courtesy Prompts and Greetings:
   - Respond politely to questions that are greetings (e.g., "Hello," "How are you?") and engage in normal conversation.
   - Ignore irrelevant contextual information during the questions which have such exchanges, and redirect to domain specific discussion during these conversations.
   - During neutral questions which are not related to any domain and are a part of normal questions, ignore the context and answer.

4. Insufficient Context:
   - If the question belongs to your domain but the context lacks relevant information, use your pre-trained knowledge to answer.

# Response Guidelines:
1. Thorough Context Analysis:
   - Carefully review the provided context to identify key details relevant to the question.
   - Base your answers on the available information.

2. Response Structure:
   - Write answers in clear, concise language, and do not prolong responses.
   - Use paragraphs for readability.
   - Employ bullet points or numbered lists for breaking down complex information.
   - Add headings or subheadings when applicable to structure your response.

3. Fallback for Insufficient Context:
   - If the context doesn’t provide enough information, clearly state:
     "The context does not contain sufficient information to fully answer this question."

4. Language and Formatting:
   - Ensure proper grammar, punctuation, and spelling.
   - Keep your tone professional and domain-relevant.

# Interaction Rules:
1. User Greetings:
   - Greet users back warmly and engage in normal conversation if no domain-specific question is asked.

2. Out-of-Domain Questions:
   - If a user asks a domain-specific question outside your expertise, respond with:
     "This question is outside my domain of expertise."
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
                'role':'system',
                'content': system_prompt,
            },
            {
                'role':'user',
                'content':f"Context{context_text}, Question: {prompt}"
            }
        ]
    )

    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break
    

    


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