"""Separate stored evidence from target-specific membership."""
from uuid import uuid4
from alembic import context,op
import sqlalchemy as sa
revision="0024_shared_evidence_source_membership"
down_revision="0023_incident_soft_delete"
branch_labels=None
depends_on=None
def upgrade():
 if context.config.attributes.get("database_role","server")!="server":return
 op.create_table("rca_incident_sources",sa.Column("id",sa.String(),primary_key=True),sa.Column("incident_id",sa.String(),sa.ForeignKey("rca_incidents.id"),nullable=False),sa.Column("source_id",sa.String(),sa.ForeignKey("rca_evidence_sources.id"),nullable=False),sa.Column("created_at",sa.String(),nullable=False),sa.UniqueConstraint("incident_id","source_id",name="uq_rca_incident_source"))
 op.create_table("rca_investigation_sources",sa.Column("id",sa.String(),primary_key=True),sa.Column("investigation_id",sa.String(),sa.ForeignKey("rca_investigations.id"),nullable=False),sa.Column("source_id",sa.String(),sa.ForeignKey("rca_evidence_sources.id"),nullable=False),sa.Column("created_at",sa.String(),nullable=False),sa.UniqueConstraint("investigation_id","source_id",name="uq_rca_investigation_source"))
 conn=op.get_bind()
 def link(table,col,target,source,created):
  conn.execute(sa.text(f"INSERT OR IGNORE INTO {table} (id,{col},source_id,created_at) VALUES (:id,:target,:source,:created)"),{"id":uuid4().hex,"target":target,"source":source,"created":created})
 for r in conn.execute(sa.text("SELECT id,investigation_id,created_at,status FROM rca_evidence_sources WHERE investigation_id IS NOT NULL")):
  if r.status not in {"REMOVED","DISMISSED"}:link("rca_investigation_sources","investigation_id",r.investigation_id,r.id,r.created_at)
 for r in conn.execute(sa.text("SELECT investigation_id,source_id,created_at FROM rca_investigation_evidence")):link("rca_investigation_sources","investigation_id",r.investigation_id,r.source_id,r.created_at)
 for r in conn.execute(sa.text("SELECT incident_id,source_id,created_at FROM rca_incident_evidence")):link("rca_incident_sources","incident_id",r.incident_id,r.source_id,r.created_at)
 for r in conn.execute(sa.text("SELECT f.investigation_id,e.source_id,e.created_at FROM rca_finding_evidence e JOIN rca_investigation_findings f ON f.id=e.finding_id")):link("rca_investigation_sources","investigation_id",r.investigation_id,r.source_id,r.created_at)
 conn.execute(sa.text("DELETE FROM rca_incident_evidence WHERE selection_json='[]'"))
 conn.execute(sa.text("DELETE FROM rca_investigation_evidence WHERE selection_json='[]'"))
 conn.execute(sa.text("UPDATE rca_evidence_sources SET status='READY' WHERE status='REMOVED'"))
 conn.execute(sa.text("UPDATE rca_evidence_sources SET status='FAILED' WHERE status='DISMISSED'"))
 with op.batch_alter_table("rca_evidence_sources") as batch:batch.drop_column("investigation_id")
def downgrade():raise RuntimeError("Shared evidence membership migration is forward-only.")
