import duckdb
import json
from dotenv import load_dotenv
import os
import loghawk.config as config

def db_insert(con,
              table_name,
              user_id,
              user_name,
              user_email,
              role,
              content
              ):
    content = content.replace("'", "''")
    con.sql(f"INSERT INTO {table_name} VALUES ('{user_id}', '{user_name}', '{user_email}', '{role}', '{content}', current_date);")

def db_fetchall(con, table_name):
    result = con.sql(f"SELECT * FROM {table_name};").fetchall()
    return result

def db_create_table(con, table_name):
    # create table once
    con.sql(f"""
    CREATE TABLE {table_name} (
        user_id VARCHAR,
        user_name VARCHAR,
        user_email VARCHAR,
        role VARCHAR,
        content VARCHAR,
        creation_date DATE
    );
    """)

def db_drop_table(con, table_name):
    con.sql(f"""
    DROP TABLE {table_name};
    """)

def populate_convo_from_db(con, table_name, convo):
    result = db_fetchall(con, table_name)
    #print(result)
    for item in result:
        convo.append({"role": item[3], "content": item[4]})
    print("convo is : ")
    print(json.dumps(convo, indent=2))
    #print(convo)



def main():
    
    
    print("here LH DUCKDB FILE PATH : ", config.LH_DUCKDB_FILE_PATH)
    print("LH DUCKDB TABLE NAME : ", config.LH_DUCKDB_TABLE_NAME)

    con = duckdb.connect(database=config.LH_DUCKDB_FILE_PATH)

    # db_drop_table(con, config.LH_DUCKDB_TABLE_NAME)
    #db_create_table(con, config.LH_DUCKDB_TABLE_NAME)
    convo = []
    populate_convo_from_db(con, config.LH_DUCKDB_TABLE_NAME, convo)
    con.close()

if __name__ == '__main__':
    main()