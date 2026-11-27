"""Tests for src/eval/prompts.py: python -m tests.test_prompts (no extra dependencies)."""
from src.eval import prompts

EX = {"id": "1", "question": "Does X help Y?", "context": "X was given to Y.", "label": "yes"}
SHOTS = [
    {"id": "a", "question": "Q-a?", "context": "C-a", "label": "yes"},
    {"id": "b", "question": "Q-b?", "context": "C-b", "label": "no"},
]


def test_templates_registry():
    assert set(prompts.TEMPLATES) == {"zero_shot", "few_shot", "structured"}


def test_no_template_leaks_gold_label():
    # The example being asked about must never carry its own answer into the prompt.
    ex = dict(EX, label="LEAKCHECK")
    for name, fn in prompts.TEMPLATES.items():
        assert "LEAKCHECK" not in fn(ex, SHOTS), name


def test_each_template_contains_question_and_context():
    for name, fn in prompts.TEMPLATES.items():
        text = fn(EX, SHOTS)
        assert EX["question"] in text and EX["context"] in text, name


def test_few_shot_demos_come_first_with_labels():
    text = prompts.few_shot(EX, SHOTS)
    assert text.index("Q-a?") < text.index("Q-b?") < text.index(EX["question"]), text
    assert "Answer: yes" in text and "Answer: no" in text
    assert text.rstrip().endswith("Answer:"), text  # model completes the final label


def test_structured_matches_training_format():
    # Eval prompt and training prompt must be identical, or fine-tuning gains are not measured fairly.
    assert prompts.structured(EX) == prompts.format_user(EX)
    assert prompts.STRUCTURED_RULE in prompts.structured(EX)


def test_format_user_without_context_is_bare_question():
    assert prompts.format_user({"question": "Plain question?", "context": ""}) == "Plain question?"


def test_from_hf_flattens_row():
    row = {"pubid": 42, "question": "Q?", "context": {"contexts": ["one", "two"]}, "final_decision": "maybe"}
    assert prompts.from_hf(row) == {"id": "42", "question": "Q?", "context": "one\ntwo", "label": "maybe"}


if __name__ == "__main__":
    test_templates_registry()
    test_no_template_leaks_gold_label()
    test_each_template_contains_question_and_context()
    test_few_shot_demos_come_first_with_labels()
    test_structured_matches_training_format()
    test_format_user_without_context_is_bare_question()
    test_from_hf_flattens_row()
    print("prompts ok")
