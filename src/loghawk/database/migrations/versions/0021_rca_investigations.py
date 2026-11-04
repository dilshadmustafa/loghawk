"""Add investigations and shared evidence without changing existing RCA records."""
from alembic import context, op
from sqlalchemy import Column, ForeignKey, Index, Integer, MetaData, String, Table, Text, UniqueConstraint
revision = "0021_rca_investigations"
down_revision = "0020_builtin_tasks"
branch_labels = None
depends_on = None
SERVER_METADATA = MetaData()

# Investigation facts, shared evidence and immutable request context.
rca_investigations = Table(
    "rca_investigations", SERVER_METADATA,
    Column("id", String, primary_key=True), Column("userid", String, nullable=False),
    Column("problem", Text, nullable=False), Column("reported_context_json", Text, nullable=False),
    Column("status", String, nullable=False), Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)
rca_incidents = Table(
    "rca_incidents", SERVER_METADATA,
    Column("id", String, primary_key=True), Column("userid", String, nullable=False),
    Column("source_incident_id", String), Column("primary_details_json", Text, nullable=False),
    Column("created_at", String, nullable=False), Column("updated_at", String, nullable=False),
)
rca_evidence_sources = Table(
    "rca_evidence_sources", SERVER_METADATA,
    Column("id", String, primary_key=True), Column("userid", String, nullable=False),
    Column("investigation_id", String, ForeignKey("rca_investigations.id"), nullable=False),
    Column("name", String, nullable=False), Column("original_uri", Text),
    Column("snapshot_uri", Text), Column("sha256", String), Column("size_bytes", Integer),
    Column("evidence_type", String, nullable=False), Column("format", String, nullable=False),
    Column("status", String, nullable=False), Column("error", Text),
    Column("mapping_json", Text, nullable=False), Column("record_path", String),
    Column("record_count", Integer, nullable=False), Column("tool_call_id", String),
    Column("created_at", String, nullable=False), Column("updated_at", String, nullable=False),
)
rca_evidence_records = Table(
    "rca_evidence_records", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("source_id", String, ForeignKey("rca_evidence_sources.id"), nullable=False),
    Column("record_number", Integer, nullable=False), Column("source_line_number", Integer),
    Column("original_record_json", Text, nullable=False),
    Column("created_at", String, nullable=False),
    UniqueConstraint("source_id", "record_number", name="uq_rca_source_record"),
)
rca_investigation_incidents = Table(
    "rca_investigation_incidents", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("investigation_id", String, ForeignKey("rca_investigations.id"), nullable=False),
    Column("incident_id", String, ForeignKey("rca_incidents.id"), nullable=False),
    Column("created_at", String, nullable=False),
    UniqueConstraint("investigation_id", "incident_id", name="uq_rca_investigation_incident"),
)
rca_incident_evidence = Table(
    "rca_incident_evidence", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("incident_id", String, ForeignKey("rca_incidents.id"), nullable=False),
    Column("source_id", String, ForeignKey("rca_evidence_sources.id"), nullable=False),
    Column("selection_json", Text, nullable=False), Column("created_at", String, nullable=False),
)
rca_investigation_evidence = Table(
    "rca_investigation_evidence", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("investigation_id", String, ForeignKey("rca_investigations.id"), nullable=False),
    Column("source_id", String, ForeignKey("rca_evidence_sources.id"), nullable=False),
    Column("selection_json", Text, nullable=False), Column("created_at", String, nullable=False),
)
rca_investigation_findings = Table(
    "rca_investigation_findings", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("investigation_id", String, ForeignKey("rca_investigations.id"), nullable=False),
    Column("userid", String, nullable=False), Column("origin", String, nullable=False),
    Column("content", Text, nullable=False), Column("request_id", String),
    Column("supersedes_id", String, ForeignKey("rca_investigation_findings.id")),
    Column("created_at", String, nullable=False),
)
rca_finding_evidence = Table(
    "rca_finding_evidence", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("finding_id", String, ForeignKey("rca_investigation_findings.id"), nullable=False),
    Column("source_id", String, ForeignKey("rca_evidence_sources.id"), nullable=False),
    Column("relationship", String, nullable=False), Column("selection_json", Text, nullable=False),
    Column("created_at", String, nullable=False),
)
rca_investigation_requests = Table(
    "rca_investigation_requests", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("investigation_id", String, ForeignKey("rca_investigations.id"), nullable=False),
    Column("request_id", String, nullable=False, unique=True),
    Column("snapshot_json", Text, nullable=False), Column("created_at", String, nullable=False),
)
for _table in (rca_investigations, rca_incidents, rca_evidence_sources,
               rca_evidence_records, rca_investigation_incidents, rca_incident_evidence,
               rca_investigation_evidence, rca_investigation_findings, rca_finding_evidence,
               rca_investigation_requests):
    Index(_table.name + "_created_at", _table.c.created_at)
Index("rca_sources_owner_hash", rca_evidence_sources.c.userid, rca_evidence_sources.c.sha256)
Index("rca_records_source", rca_evidence_records.c.source_id)
Index("rca_investigation_requests_parent", rca_investigation_requests.c.investigation_id)


def upgrade():
    if context.config.attributes.get("database_role", "server") == "server":
        SERVER_METADATA.create_all(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("Investigation history must be preserved; this migration is forward-only.")
