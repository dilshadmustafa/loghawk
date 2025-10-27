import ollama
import duckdb
from dotenv import load_dotenv
from loghawk_utils import duckdbutils, genutils
import os

load_dotenv()

DUCKDB_FILE_PATH = os.getenv("DUCKDB_FILE_PATH")
DUCKDB_TABLE_NAME = os.getenv("DUCKDB_TABLE_NAME")

print("DUCKDB FILE PATH : ", DUCKDB_FILE_PATH)
print("DUCKDB TABLE NAME : ", DUCKDB_TABLE_NAME)

convo = []
con = duckdb.connect(database=DUCKDB_FILE_PATH)

def stream_response(prompt):
    convo.append({"role": "user", "content": prompt})
    if not genutils.contains_mostly_numbers(prompt):
        duckdbutils.db_insert(con, DUCKDB_TABLE_NAME, 'anon', 'anon',
                              'anon@anon.com', 'user',
                              content=prompt)
    response = ''
    stream = ollama.chat(model="deepseek-r1:1.5b", messages=convo, stream=True)
    print(f"ASSISTANT: ")
    for chunk in stream:
        content = chunk["message"]["content"]
        response += content
        print(content, end='', flush=True)
    print("\n")
    print("End of Assistant Response\n")
    convo.append({ "role" : "assistant", "content" : response })
    duckdbutils.db_insert(con, DUCKDB_TABLE_NAME, 'anon', 'anon',
                          'anon@anon.com', 'assistant',
                          content=response)

duckdbutils.populate_convo_from_db(con, DUCKDB_TABLE_NAME, convo)
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



