"""Connects the pipeline to the Phoenix monitoring dashboard.

Phoenix runs on the local machine (started with run_phoenix.py) and listens
at localhost:6006. This file registers a "tracer": every important step
in the pipeline opens a span (a timed record of one step), and those
spans stream to Phoenix where you can watch them live.
"""

from opentelemetry import trace

from src.config import settings

# Filled in once by setup_tracing() so repeat calls do no extra work.
_tracer_provider = None

# The project name shown in the Phoenix dashboard sidebar.
PROJECT_NAME = "monitored-rag-service"


def setup_tracing() -> None:
    """Points tracing at the local Phoenix dashboard. Call once at startup."""
    global _tracer_provider
    if _tracer_provider is not None:
        return

    # phoenix.otel.register does the wiring: it creates a tracer that
    # sends every span to Phoenix over HTTP.
    from phoenix.otel import register

    _tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint=settings.phoenix_endpoint.rstrip("/") + "/v1/traces",
        set_global_tracer_provider=True,
        batch=True,
    )


def get_tracer():
    """Returns the tracer the pipeline uses to open spans."""
    return trace.get_tracer("monitored-rag-service")


def span_id_of(span) -> str:
    """Returns a span's id as text, used to attach evaluation scores later."""
    return format(span.get_span_context().span_id, "016x")
