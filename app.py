from assistant import chat

prompt=''
while prompt!='/bye':
    prompt=input('\nUSER: ')
    for content in chat(prompt):
        print(content, end='', flush=True)