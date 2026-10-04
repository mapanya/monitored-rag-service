Evidence of the working system, captured on the submitting machine.

Service behaviour (test page at localhost:8000):
01_allowed_question.png      A normal question answered with sources;
                             the guardrail allowed it through.
02_blocked_question.png      A direct prompt injection attempt refused
                             by the guardrail (blocked: true).
03_flagged_passage.png       The passage guardrail catching the planted
                             injection sentence in a retrieved document
                             (passages_flagged: 1).

Monitoring (Phoenix dashboard at localhost:6006):
04_phoenix_traces.png        The trace list after several questions.
05_phoenix_trace_detail.png  One open trace showing all four pipeline
                             steps: query guardrail, hybrid search,
                             passage guardrail and answer generation.
06_phoenix_tokens.png        The generate_answer step with its token
                             counts (prompt, completion and total).

Evaluation (retrieval quality):
07_cli_evaluation.png        The evaluation run completing in the
                             terminal with a relevance score for each
                             of the five test questions.
08_phoenix_evaluation_1.png  retrieval_relevance score attached to the
                             hybrid_search step of evaluation question 1.
09_phoenix_evaluation_2.png  Same, evaluation question 2.
10_phoenix_evaluation_3.png  Same, evaluation question 3.
11_phoenix_evaluation_4.png  Same, evaluation question 4.
12_phoenix_evaluation_5.png  Same, evaluation question 5.

Tests:
13_tests_passing.png         The automated test suite passing
                             (12 tests: guardrail, chunking and
                             prompt assembly).
