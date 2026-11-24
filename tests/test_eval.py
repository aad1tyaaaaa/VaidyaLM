"""Edge-case checks for the eval prompts and scorer. Run: python -m tests.test_eval"""
from src.eval import prompts, scorer

EX = {"id": "1", "question": "Does X help?", "context": "X helped in a trial.", "label": "yes"}


def test_parse_label():
    p = scorer.parse_label
    assert p("ANSWER - Maybe") == "maybe"          # case + separator tolerant
    assert p("Answer:yes") == "yes"                 # no space after colon
    assert p("") is None
    assert p("Nobody knows") is None                # 'no' inside a word must not match
    assert p("yesterday it rained") is None         # 'yes' inside a word must not match
    assert p("The trial says no. Answer: yes") == "yes"  # explicit Answer: wins over a bare word


def test_score():
    m = scorer.score(["yes", "no", "maybe"], ["Answer: yes", "Answer: no", "Answer: maybe"])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["parse_fail_pct"] == 0.0, m
    m = scorer.score(["yes", "no"], ["???", "???"])  # all unparseable: wrong, and counted as failures
    assert m["accuracy"] == 0.0 and m["macro_f1"] == 0.0 and m["parse_fail_pct"] == 100.0, m
    m = scorer.score(["yes", "yes"], ["Answer: yes", "Answer: yes"])  # absent classes must not crash macro-F1
    assert m["accuracy"] == 1.0 and abs(m["macro_f1"] - 1 / 3) < 1e-9, m


def test_prompts():
    for name, fn in prompts.TEMPLATES.items():
        text = fn(EX, shots=[EX])
        assert EX["question"] in text and EX["context"] in text, name
    assert "Answer: yes" in prompts.few_shot(EX, shots=[EX])        # demo shows its label...
    assert prompts.few_shot(EX, shots=[EX]).endswith("Answer:")     # ...the real question does not
    assert prompts.format_user({**EX, "context": ""}) == EX["question"]  # no context -> bare question
    assert prompts.format_user(EX) == prompts.structured(EX)         # train prompt == eval prompt
    flat = prompts.from_hf({"pubid": 7, "question": "q", "context": {"contexts": ["a", "b"]}, "final_decision": "no"})
    assert flat == {"id": "7", "question": "q", "context": "a\nb", "label": "no"}, flat


if __name__ == "__main__":
    test_parse_label()
    test_score()
    test_prompts()
    print("eval ok")
