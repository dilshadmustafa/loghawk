
import lancedb

def init_database(db_file, table_name):
    """Initialize the database."""
    # Create a LanceDB table with the schema
    db = lancedb.connect(db_file)

    if table_name not in db.table_names():
        table = db.create_table(table_name, schema=MyTableSchema)
    else:
        table = db.open_table(table_name)
    return db, table











