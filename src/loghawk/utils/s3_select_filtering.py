"""Shared helpers for building S3 Select row-filter expressions."""

from collections.abc import Mapping, Sequence


def build_s3_select_where_clause(
    filters: Mapping[str, Sequence[str]],
    available_columns: Sequence[str] | None = None,
) -> str:
    """Build safe predicates: OR within a column and AND across columns."""
    if not filters:
        return ""

    if available_columns is not None:
        available = set(available_columns)
        missing = [column for column in filters if column not in available]
        if missing:
            raise ValueError(
                "S3 Select filter columns are missing from the input: "
                + ", ".join(repr(column) for column in missing)
            )

    predicates = []
    for column, values in filters.items():
        if not isinstance(column, str) or not column:
            raise ValueError("S3 Select filter column names must be non-empty strings")
        if not values or any(not isinstance(value, str) for value in values):
            raise ValueError(
                f"S3 Select filter values for {column!r} must be non-empty strings"
            )

        filter_values = list(values)
        if column.casefold() in {"level", "severity", "log_level"}:
            filter_values = [
                alias
                for value in filter_values
                for alias in (
                    ("WARN", "WARNING")
                    if value.strip().upper() == "WARN"
                    else (value,)
                )
            ]

        identifier = 's."' + column.replace('"', '""') + '"'
        literals = ", ".join(
            "'" + value.replace("'", "''").upper() + "'"
            for value in filter_values
        )
        predicates.append(
            f"upper(cast({identifier} as varchar)) IN ({literals})"
        )

    return " AND ".join(predicates)
