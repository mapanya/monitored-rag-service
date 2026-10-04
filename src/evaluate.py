"""Measures retrieval quality and records the scores in Phoenix.

Run it (with Phoenix and your .env in place) using:

    python -m src.evaluate

What it does for each test question below:
1. Runs the question through the full pipeline (which also leaves a
   trace in Phoenix, like any other question).
2. Shows the retrieved passages to Gemini and asks it to judge each
   one: does this passage actually help answer the question?
3. Turns the judgments into a score between 0.0 and 1.0: the fraction
   of retrieved passages that were relevant. A score of 1.0 means
   every passage the search returned was on topic.
4. Sends the scores to Phoenix, where they appear attached to the
   hybrid_search spans under the name retrieval_relevance.

If you replace the sample PDFs with your own documents, rewrite the
questions below so they match your content.
"""

import json
import time

import pandas as pd
from google.genai import errors as genai_errors
from phoenix.client import Client

from src.config import settings
from src.observability import setup_tracing
from src.rag_pipeline import RAGPipeline

# Five questions the policy summary documents can answer. EDIT THESE
# if you load your own documents.
EVAL_QUESTIONS = [
    "Is limited personal use of company systems allowed?",
    "What must an employee do with an unknown USB drive found unattended?",
    "Who must receive security awareness training?",
    "How must security incidents be classified?",
    "What sanctions can follow a violation of these policies?",
]

JUDGE_PROMPT = (
    "You are grading search results. For the question below, judge each "
    "numbered passage: does it contain information that helps answer the "
    "question? Reply with ONLY a JSON list of true/false values, one per "
    "passage, in order. Example: [true, false, true]\n\n"
    "Question: {question}\n\nPassages:\n{passages}"
)


def with_patience(call, doing_what: str):
    """Runs one model call, waiting out rate limits instead of crashing.

    The free API key allows only a handful of requests per minute, and
    the shared servers are sometimes briefly busy. Both answer with a
    polite refusal (codes 429 and 503). A batch job like this one
    should wait and try again, not fall over, so this helper retries
    up to four times with a 30 second pause between attempts.
    """
    for attempt in range(5):
        try:
            return call()
        except genai_errors.APIError as error:
            code = getattr(error, "code", None)
            if code in (429, 503) and attempt < 4:
                print(
                    f"  The model pushed back (code {code}) while "
                    f"{doing_what}; waiting 30 seconds and retrying..."
                )
                time.sleep(30)
            else:
                raise


def judge_passages(gemini, question: str, passages: list[str]) -> list[bool]:
    """Asks Gemini to judge all passages for one question in one call."""
    numbered = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    prompt = JUDGE_PROMPT.format(question=question, passages=numbered)
    response = gemini.models.generate_content(
        model=settings.gemini_model, contents=prompt
    )
    text = (response.text or "").strip()
    # The model sometimes wraps JSON in a code fence; strip it off.
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        verdicts = json.loads(text)
        return [bool(v) for v in verdicts][: len(passages)]
    except (json.JSONDecodeError, TypeError):
        print(f"  Could not read the judge's reply, skipping: {text[:80]}")
        return []


def main() -> None:
    setup_tracing()
    pipeline = RAGPipeline()
    rows = []

    try:
        for question in EVAL_QUESTIONS:
            print(f"Evaluating: {question}")
            result = with_patience(
                lambda: pipeline.answer(question), "answering the question"
            )
            span_id = result["retrieval_span_id"]
            if not span_id:
                print("  Skipped (question was blocked or nothing retrieved).")
                continue

            passages = pipeline._retriever.search(question)
            verdicts = with_patience(
                lambda: judge_passages(
                    pipeline._gemini, question, [p.text for p in passages]
                ),
                "judging the passages",
            )
            if not verdicts:
                continue

            score = sum(verdicts) / len(verdicts)
            rows.append(
                {
                    "context.span_id": span_id,
                    "score": score,
                    "label": "relevant" if score >= 0.5 else "irrelevant",
                    "explanation": (
                        f"{sum(verdicts)} of {len(verdicts)} retrieved "
                        "passages were judged relevant to the question."
                    ),
                }
            )
            print(f"  Relevance score: {score:.2f}")
            # The free key allows about 5 requests per minute and each
            # question costs 2, so a generous pause keeps us inside it.
            time.sleep(25)
    finally:
        pipeline.close()

    if not rows:
        raise SystemExit("No scores were produced; nothing to send to Phoenix.")

    dataframe = pd.DataFrame(rows).set_index("context.span_id")
    client = Client(base_url=settings.phoenix_endpoint)
    client.spans.log_span_annotations_dataframe(
        dataframe=dataframe,
        annotation_name="retrieval_relevance",
        annotator_kind="LLM",
    )
    average = dataframe["score"].mean()
    print(
        f"Done. {len(rows)} scores sent to Phoenix (average {average:.2f}). "
        "Open http://localhost:6006, open the monitored-rag-service project "
        "and look at the hybrid_search spans to see them."
    )


if __name__ == "__main__":
    main()
