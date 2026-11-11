"""Bounded deterministic imports; snapshots remain independent of mutable inputs."""
import csv
import hashlib
import io
import json
from pathlib import PurePosixPath
from tempfile import SpooledTemporaryFile
from urllib.parse import urlparse
from uuid import uuid4
from loghawk import config
from loghawk.api.services import investigation_store as store
from loghawk.api.services.storage_service import _s3_client
from loghawk.database.database_layer import connect
from loghawk.database.timestamps import utc_now

FORMATS={"csv","parquet","json","jsonl","log","txt"}
TYPES={"stage-a","stage-b","stage-c","raw-logs","other"}

def max_bytes():return config.LH_INVESTIGATION_MAX_IMPORT_MB*1024*1024

def s3_parts(uri):
    parsed=urlparse(uri)
    if parsed.scheme!="s3" or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
        raise ValueError("Use an internal s3://bucket/key or prefix URL.")
    return parsed.netloc,parsed.path.lstrip("/")

def list_s3(uri,token=None,browse=False):
    bucket,key=s3_parts(uri);client=_s3_client()
    # A suffix selects a single object; a trailing slash selects a prefix.
    if not browse and key and not key.endswith("/") and PurePosixPath(key).suffix.lower().lstrip(".") in FORMATS:
        obj=client.head_object(Bucket=bucket,Key=key)
        return {"files":[{"uri":uri,"size":obj["ContentLength"],"etag":obj.get("ETag")}],"next_token":None}
    args={"Bucket":bucket,"Prefix":key,"MaxKeys":100}
    if browse:
        if key and not key.endswith("/"):args["Prefix"]=key+"/"
        args["Delimiter"]="/"
    if token:args["ContinuationToken"]=token
    result=client.list_objects_v2(**args)
    files=[{"uri":f"s3://{bucket}/{o['Key']}","size":o["Size"],"etag":o.get("ETag")} for o in result.get("Contents",[]) if PurePosixPath(o["Key"]).suffix.lower().lstrip(".") in FORMATS]
    return {"files":files,"next_token":result.get("NextContinuationToken"),
            "folders":[{"uri":f"s3://{bucket}/{o['Prefix']}"} for o in result.get("CommonPrefixes",[])]}


def resolve_s3_selection(uris):
    """Expand selected prefixes for review, without importing or silently truncating."""
    if not 1 <= len(uris) <= config.LH_INVESTIGATION_MAX_IMPORT_FILES:
        raise ValueError(f"Select between 1 and {config.LH_INVESTIGATION_MAX_IMPORT_FILES} files or folders.")
    unique = sorted(set(uris))
    # Parent folder selection already includes its selected children.
    roots = [uri for uri in unique if not any(
        parent != uri and parent.endswith("/") and uri.startswith(parent) for parent in unique)]
    files = {}
    total = 0
    pages = 0
    for uri in roots:
        token = None
        while True:
            pages += 1
            if pages > 100:
                raise ValueError("This selection is too broad to review. Select a smaller folder.")
            result = list_s3(uri, token)
            for item in result["files"]:
                if item["uri"] in files:
                    continue
                files[item["uri"]] = item
                total += item["size"]
                if len(files) > config.LH_INVESTIGATION_MAX_IMPORT_FILES:
                    raise ValueError(f"Selection exceeds {config.LH_INVESTIGATION_MAX_IMPORT_FILES} supported files. Select smaller folders or individual files.")
                if total > max_bytes():
                    raise ValueError("Selected files exceed the configured combined import byte limit.")
            token = result["next_token"]
            if not token:
                break
    return {"files":sorted(files.values(), key=lambda item:item["uri"]), "total_bytes":total}


def stage(stream,remaining):
    temp=SpooledTemporaryFile(max_size=4*1024*1024,mode="w+b")
    digest=hashlib.sha256();size=0
    try:
        while True:
            data=stream.read(min(1024*1024,remaining-size+1))
            if not data:break
            size+=len(data)
            if size>remaining:raise ValueError("Import exceeds the configured combined byte limit.")
            digest.update(data);temp.write(data)
        temp.seek(0);return temp,size,digest.hexdigest()
    except Exception:
        temp.close();raise

def _json(text):
    def invalid(value):raise ValueError(f"Invalid JSON constant: {value}")
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError(f"Duplicate JSON field: {key}")
            result[key]=value
        return result
    return json.loads(text,parse_constant=invalid,object_pairs_hook=object_pairs)

def records(stream,fmt,record_path=None):
    stream.seek(0)
    if fmt=="parquet":
        import pyarrow.parquet as pq
        parquet=pq.ParquetFile(stream)
        if parquet.metadata.num_rows>config.LH_INVESTIGATION_MAX_RECORDS_PER_FILE:
            raise ValueError("Parquet record count exceeds the configured limit.")
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():yield None,row
        return
    wrapper=io.TextIOWrapper(stream,encoding="utf-8-sig",newline="")
    try:
        if fmt=="csv":
            reader=csv.DictReader(wrapper,strict=True)
            if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):raise ValueError("CSV needs unique header names.")
            for row in reader:
                if None in row or any(v is None for v in row.values()):raise ValueError(f"CSV row width mismatch near line {reader.line_num}.")
                yield reader.line_num,row
        elif fmt=="json":
            value=_json(wrapper.read())
            if record_path:
                if not record_path.startswith("/"):raise ValueError("Nested JSON record path must be a JSON Pointer, e.g. /incidents.")
                for key in record_path[1:].split("/"):
                    key=key.replace("~1","/").replace("~0","~")
                    value=value[int(key)] if isinstance(value,list) else value[key]
            if isinstance(value,dict):
                if not record_path and any(isinstance(v,list) and any(isinstance(x,dict) for x in v) for v in value.values()):
                    raise ValueError("Nested record collection detected. Specify its JSON Pointer, e.g. /incidents.")
                yield None,value
            elif isinstance(value,list):
                for row in value:yield None,row
            else:raise ValueError("JSON must contain an object or an array of objects.")
        else:
            for number,line in enumerate(wrapper,1):
                if fmt=="jsonl":
                    if not line.strip():continue
                    try:row=_json(line)
                    except ValueError as exc:raise ValueError(f"JSONL line {number}: {exc}") from exc
                    if not isinstance(row,dict):raise ValueError(f"JSONL line {number} is not an object.")
                    yield number,row
                else:yield number,{"text":line.rstrip("\r\n")}
    finally:wrapper.detach()

def parse(stream,fmt,record_path=None):
    output=[];decoded_size=0
    for number,(line_number,row) in enumerate(records(stream,fmt,record_path),1):
        if number>config.LH_INVESTIGATION_MAX_RECORDS_PER_FILE:raise ValueError("Record count exceeds the configured limit.")
        if not isinstance(row,dict):raise ValueError(f"Record {number} is not an object.")
        text=store.encode(row);decoded_size+=len(text.encode("utf-8"))
        if decoded_size>max_bytes():raise ValueError("Decoded records exceed the configured byte limit.")
        output.append((number,line_number,text))
    if not output:raise ValueError("Source contains no records.")
    return output

def validate_source(name,evidence_type):
    name=PurePosixPath(name.replace("\\","/")).name
    fmt=PurePosixPath(name).suffix.lower().lstrip(".")
    if fmt not in FORMATS:raise ValueError("Supported formats: CSV, Parquet, JSON, JSONL, LOG and TXT.")
    if evidence_type not in TYPES:raise ValueError("Invalid evidence type.")
    return name,fmt


class _UploadStream:
    """Let the importer retain ownership of the staged file."""

    def __init__(self, stream):
        self._stream = stream

    def read(self, size=-1):
        return self._stream.read(size)

    def seek(self, offset, whence=0):
        return self._stream.seek(offset, whence)

    def tell(self):
        return self._stream.tell()

    def close(self):
        # The staging owner closes the file after upload and parsing.
        pass


def import_staged(key,user,name,stream,size,digest,evidence_type,record_path=None,original_uri=None,incident_id=None):
    name,fmt=validate_source(name,evidence_type)
    now=utc_now();source_id=uuid4().hex
    prefix=config.LH_INVESTIGATION_SOURCE_S3_PREFIX.strip("/")
    if not prefix or any(p in {".",".."} for p in prefix.split("/")):raise ValueError("Invalid investigation source prefix.")
    owner_folder = key or "standalone-incidents"
    object_key=f"{prefix}/{owner_folder}/{source_id}/original.{fmt}"
    snapshot=f"s3://{config.LH_S3_BUCKET}/{object_key}"
    with connect("server") as db:
        if incident_id:
            incident = db.execute("SELECT id,userid FROM rca_incidents WHERE id=?", (incident_id,)).fetchone()
            if not incident or incident["userid"] != user["userid"]:
                raise PermissionError("Incident not found or access denied.")
            duplicate = db.execute(
                "SELECT s.id FROM rca_evidence_sources s JOIN rca_incident_evidence e ON e.source_id=s.id "
                "WHERE e.incident_id=? AND s.sha256=? AND s.status='READY'", (incident_id,digest)
            ).fetchone()
        elif key:
            store.require(db,key,user,write=True)
            duplicate=db.execute("SELECT id FROM rca_evidence_sources WHERE investigation_id=? AND sha256=? AND status='READY'",(key,digest)).fetchone()
        else:
            duplicate=db.execute(
                "SELECT id FROM rca_evidence_sources WHERE userid=? AND investigation_id IS NULL AND sha256=? AND status='READY'",
                (user["userid"],digest),
            ).fetchone()
        if duplicate:return {"id":duplicate["id"],"status":"DUPLICATE","name":name}
        db.execute("INSERT INTO rca_evidence_sources (id,userid,investigation_id,name,original_uri,snapshot_uri,sha256,size_bytes,evidence_type,format,status,error,mapping_json,record_path,record_count,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(source_id,user["userid"],key,name,original_uri,snapshot,digest,size,evidence_type,fmt,"IMPORTING",None,"{}",record_path,0,now,now))
        if incident_id:
            db.execute("INSERT INTO rca_incident_evidence (id,incident_id,source_id,selection_json,created_at) VALUES (?,?,?,?,?)",
                       (uuid4().hex,incident_id,source_id,"[]",now))
    try:
        stream.seek(0)
        _s3_client().upload_fileobj(
            _UploadStream(stream), config.LH_S3_BUCKET, object_key
        )
        stream.seek(0);parsed=parse(stream,fmt,record_path)
        with connect("server") as db:
            for number,line_number,text in parsed:
                db.execute("INSERT INTO rca_evidence_records (id,source_id,record_number,source_line_number,original_record_json,created_at) VALUES (?,?,?,?,?,?)",(uuid4().hex,source_id,number,line_number,text,now))
            db.execute("UPDATE rca_evidence_sources SET status='READY',record_count=?,updated_at=? WHERE id=?",(len(parsed),utc_now(),source_id))
        return {"id":source_id,"status":"READY","name":name,"record_count":len(parsed)}
    except Exception as exc:
        error=str(exc)[:1000] if isinstance(exc,(ValueError,UnicodeError,csv.Error,KeyError,IndexError)) else f"{type(exc).__name__}: source import failed; check storage connectivity or file validity."
        with connect("server") as db:db.execute("UPDATE rca_evidence_sources SET status='FAILED',error=?,updated_at=? WHERE id=?",(error,utc_now(),source_id))
        return {"id":source_id,"status":"FAILED","name":name,"error":error}

def import_s3(key,user,objects,evidence_type,record_path=None,incident_id=None):
    if not 1<=len(objects)<=config.LH_INVESTIGATION_MAX_IMPORT_FILES:raise ValueError("Invalid number of selected objects.")
    for item in objects:
        _,name=s3_parts(item["uri"])
        validate_source(name,evidence_type)
    staged=[];remaining=max_bytes()
    try:
        for item in objects:
            bucket,name=s3_parts(item["uri"]);args={"Bucket":bucket,"Key":name}
            if item.get("etag"):args["IfMatch"]=item["etag"]
            response=_s3_client().get_object(**args)
            try:temp,size,digest=stage(response["Body"],remaining)
            finally:response["Body"].close()
            remaining-=size;staged.append((item["uri"],name,temp,size,digest))
        return [import_staged(key,user,name,temp,size,digest,evidence_type,record_path,uri,incident_id) for uri,name,temp,size,digest in staged]
    finally:
        for _,_,temp,_,_ in staged:temp.close()

def reparse(key,source_id,user,record_path,fmt=None):
    with connect("server") as db:
        source=store.source_access(db,key,source_id,user,write=True)
        if source["status"]!="FAILED":raise ValueError("Only failed sources can be reparsed; ready sources are immutable.")
        fmt=fmt or source["format"]
        if fmt not in FORMATS:raise ValueError("Unsupported format")
        db.execute("UPDATE rca_evidence_sources SET status='IMPORTING',updated_at=? WHERE id=?",(utc_now(),source_id))
    bucket,name=s3_parts(source["snapshot_uri"])
    try:
        response=_s3_client().get_object(Bucket=bucket,Key=name)
        try:temp,size,digest=stage(response["Body"],max_bytes())
        finally:response["Body"].close()
        try:
            if digest!=source["sha256"]:raise ValueError("Stored source hash does not match its import record.")
            parsed=parse(temp,fmt,record_path)
        finally:temp.close()
        with connect("server") as db:
            for number,line_number,text in parsed:db.execute("INSERT INTO rca_evidence_records (id,source_id,record_number,source_line_number,original_record_json,created_at) VALUES (?,?,?,?,?,?)",(uuid4().hex,source_id,number,line_number,text,utc_now()))
            db.execute("UPDATE rca_evidence_sources SET status='READY',error=NULL,record_count=?,record_path=?,format=?,updated_at=? WHERE id=?",(len(parsed),record_path,fmt,utc_now(),source_id))
    except Exception:
        with connect("server") as db:db.execute("UPDATE rca_evidence_sources SET status='FAILED',error='Reparse failed; check format, JSON Pointer and source snapshot.',updated_at=? WHERE id=?",(utc_now(),source_id))
        raise ValueError("Could not reparse source. Check format, JSON Pointer and source snapshot.")
    return store.preview(key,source_id,user)
