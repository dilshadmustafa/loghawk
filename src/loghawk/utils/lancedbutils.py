import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import loghawk.config as config

# EmbeddingGemma (requires Hugging Face access request)
# Visit: https://huggingface.co/google/embeddinggemma-300m
# Run: huggingface-cli login
# LH_EMBEDDING_MODEL = 'google/embeddinggemma-300m'  # Google's new EmbeddingGemma model

# Get a sentence-transformer function
func = get_registry().get("sentence-transformers").create(name=config.LH_EMBEDDING_MODEL)

class MySchema(LanceModel):
    # Embed the 'text' field automatically
    text: str = func.SourceField()
    # Store the embeddings in the 'vector' field
    vector: Vector(func.ndims()) = func.VectorField()

def init_database(db_file, table_name):
    """Initialize the database."""
    # Create a LanceDB table with the schema
    db = lancedb.connect(db_file)

    if table_name not in db.table_names():
        table = db.create_table(table_name, schema=MySchema)
    else:
        table = db.open_table(table_name)
    return db, table











