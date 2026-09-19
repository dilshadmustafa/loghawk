import duckdb
import json
import os
import loghawk.config as config
import loghawk.utils.duckdbutils as duckdbutils

def main():
    
    
    print("here LH DUCKDB FILE PATH : ", config.LH_DUCKDB_FILE_PATH)
    print("LH DUCKDB TABLE NAME : ", config.LH_DUCKDB_TABLE_NAME)

    con = duckdb.connect(database=config.LH_DUCKDB_FILE_PATH)

    # db_drop_table(con, config.LH_DUCKDB_TABLE_NAME)
    duckdbutils.db_create_table(con, config.LH_DUCKDB_TABLE_NAME)
    convo = []
    duckdbutils.populate_convo_from_db(con, config.LH_DUCKDB_TABLE_NAME, convo)
    con.close()

if __name__ == '__main__':
    main()