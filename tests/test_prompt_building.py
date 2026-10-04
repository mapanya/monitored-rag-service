"""Tests for the prompt sent to the answer model.

These run without any API keys or internet connection, because
build_prompt is pure text assembly.
"""

from src.rag_pipeline import build_prompt
from src.retriever import RetrievedPassage


def make_passages():
    return [
        RetrievedPassage(
            text="Limited personal use of company systems is allowed.",
            source="acceptable_use_policy.pdf",
            score=0.91,
        ),
        RetrievedPassage(
            text="Incidents must be classified by potential impact.",
            source="security_incident_management_policy.pdf",
            score=0.77,
        ),
    ]


def test_prompt_contains_question_and_passages():
    prompt = build_prompt("Is personal use allowed?", make_passages())
    assert "Is personal use allowed?" in prompt
    assert "Limited personal use of company systems is allowed." in prompt
    assert "Incidents must be classified by potential impact." in prompt


def test_passages_are_numbered_with_sources():
    prompt = build_prompt("Anything", make_passages())
    assert "[1] (from acceptable_use_policy.pdf)" in prompt
    assert "[2] (from security_incident_management_policy.pdf)" in prompt


def test_prompt_tells_model_to_ignore_embedded_instructions():
    # The defense in depth rule: even if a bad passage slipped through,
    # the model is told to treat passages as reference text only.
    prompt = build_prompt("Anything", make_passages())
    assert "ignore them" in prompt.lower()
