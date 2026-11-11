"""Target-specific links from incidents and investigations to stored evidence sources."""
from uuid import uuid4
from loghawk.database.timestamps import utc_now
_TARGETS={"incident":("rca_incident_sources","incident_id"),"investigation":("rca_investigation_sources","investigation_id")}
def exists(db,kind,target,source):
 table,column=_TARGETS[kind]
 return db.execute(f"SELECT id FROM {table} WHERE {column}=? AND source_id=?",(target,source)).fetchone()
def ensure(db,kind,target,source,created=None):
 table,column=_TARGETS[kind];row=exists(db,kind,target,source)
 if row:return row["id"]
 link=uuid4().hex
 db.execute(f"INSERT INTO {table} (id,{column},source_id,created_at) VALUES (?,?,?,?)",(link,target,source,created or utc_now()))
 return link
def remove(db,kind,target,source):
 table,column=_TARGETS[kind]
 return db.execute(f"DELETE FROM {table} WHERE {column}=? AND source_id=?",(target,source)).rowcount
