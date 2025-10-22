import ollama
import duckdb

from loghawk_utils import duckdbutils, genutils

convo = []
con = duckdb.connect(database="C:\\aiopsmain\\my_work\\mydb\\my_database.duckdb")

def stream_response(prompt):
    convo.append({"role": "user", "content": prompt})
    #prompt2 = prompt.replace("'", "''")
    #con.sql(f"INSERT INTO convo VALUES ('anon', 'anon@anon.com', 'user', '{prompt2}', current_date);")
    if not genutils.contains_mostly_numbers(prompt):
        duckdbutils.db_insert(con, 'anon', 'anon@anon.com', 'user',
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
    #response2 = response.replace("'","''")
    #con.sql(f"INSERT INTO convo VALUES ('anon', 'anon@anon.com', 'assistant', '{response2}', current_date);")
    duckdbutils.db_insert(con, 'anon', 'anon@anon.com', 'assistant',
                          content=response)

duckdbutils.populate_convo_from_db(con, convo)
while True:
    #print("h1")
    prompt = genutils.get_multiline_input()
    print("length of prompt : " + str(len(prompt)))
    print("prompt is : \n" + prompt)
    if len(prompt.strip()) == 0:
        print("You entered all blank lines. Pls enter again.")
        continue
    if prompt.strip() == "exit":
        break
    stream_response(prompt=prompt)


con.close()



