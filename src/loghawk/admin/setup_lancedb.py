import lancedb
import json
import os
import loghawk.config as config
import loghawk.utils.lancedbutils as lancedbutils

def main():
        
    print("here LH_LANCEDB_FILE_PATH : ", config.LH_LANCEDB_FILE_PATH)
    print("LH_LANCEDB_TABLE_NAME : ", config.LH_LANCEDB_TABLE_NAME)

    lancedbutils.init_database(config.LH_LANCEDB_FILE_PATH, config.LH_LANCEDB_TABLE_NAME)

if __name__ == '__main__':
    main()