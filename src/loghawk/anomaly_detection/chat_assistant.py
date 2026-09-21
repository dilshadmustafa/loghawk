import ollama
import duckdb
from loghawk.utils import genutils
import os
from loghawk.utils import duckdbutils
import loghawk.config as config

print("DUCKDB FILE PATH : ", config.LH_DUCKDB_FILE_PATH)
print("DUCKDB TABLE NAME : ", config.LH_DUCKDB_TABLE_NAME)

convo = []
con = duckdb.connect(database=config.LH_DUCKDB_FILE_PATH)

def stream_response(prompt):
    convo.append({"role": "user", "content": prompt})
    if not genutils.contains_mostly_numbers(prompt):
        duckdbutils.db_insert(con, config.LH_DUCKDB_TABLE_NAME, 'anon', 'anon',
                              'anon@anon.com', 'user',
                              content=prompt)
    response = ''
    stream = ollama.chat(model=config.LH_LLM_MODEL, messages=convo, stream=True)
    print(f"ASSISTANT: ")
    for chunk in stream:
        content = chunk["message"]["content"]
        response += content
        print(content, end='', flush=True)
    print("\n")
    print("End of Assistant Response\n")
    convo.append({ "role" : "assistant", "content" : response })
    duckdbutils.db_insert(con, config.LH_DUCKDB_TABLE_NAME, 'anon', 'anon',
                          'anon@anon.com', 'assistant',
                          content=response)

duckdbutils.populate_convo_from_db(con, config.LH_DUCKDB_TABLE_NAME, convo)
while True:
    prompt = genutils.get_multiline_input()
    #print("length of prompt : " + str(len(prompt)))
    #print("prompt is : \n" + prompt)
    if len(prompt.strip()) == 0:
        print("You entered all blank lines. Pls enter again.")
        continue
    if prompt.strip() == "exit":
        break
    stream_response(prompt=prompt)


con.close()



