import duckdb

def db_insert(con,
              user_name,
              user_email,
              role,
              content
              ):
    content = content.replace("'", "''")
    con.sql(f"INSERT INTO convo VALUES ('{user_name}', '{user_email}', '{role}', '{content}', current_date);")

def db_fetchall(con):
    result = con.sql("SELECT * FROM convo;").fetchall()
    return result

def db_create_table(con):
    # create table once
    con.sql("""
    CREATE TABLE convo (
        user_name VARCHAR,
        user_email VARCHAR,
        role VARCHAR,
        content VARCHAR,
        creation_date DATE
    );
    """)

def db_drop_table(con):
    con.sql("""
    DROP TABLE convo;
    """)

def populate_convo_from_db(con, convo):
    result = db_fetchall(con)
    #print(result)
    for item in result:
        convo.append({"role": item[2], "content": item[3]})
    #print(json.dumps(convo, indent=4))
    print("convo is : ")
    print(convo)

def main():
    con = duckdb.connect(database="C:\\aiopsmain\\my_work\\mydb\\my_database.duckdb")
    db_drop_table(con)
    db_create_table(con)
    convo2 = []
    populate_convo_from_db(con, convo2)

if __name__ == '__main__':
    main()