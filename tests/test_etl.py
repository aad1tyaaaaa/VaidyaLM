"""Smallest end-to-end check of the ETL logic. Run: python -m tests.test_etl"""
import pathlib
import tempfile

from src.etl import build_pairs, chunk, clean, dedupe, parse, schema, split

XML = """<?xml version="1.0"?>
<Document id="0000001" source="GARD">
<QAPairs>
<QAPair pid="1"><Question qid="1">What is Fabry disease?</Question>
<Answer>What is Fabry disease? Fabry disease is an inherited disorder that results from the buildup of a fat called globotriaosylceramide in cells.</Answer></QAPair>
<QAPair pid="2"><Question qid="2">What causes Fabry disease ?</Question><Answer></Answer></QAPair>
</QAPairs></Document>"""

CTX = "Randomized trial of 400 adults. Aspirin reduced fever duration compared with placebo, with no serious adverse events reported in either arm."


def demo():
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "GARD_QA" / "doc.xml"
        f.parent.mkdir()
        f.write_text(XML, encoding="utf-8")
        recs = parse.parse_shard(f.parent)
    assert len(recs) == 2 and recs[0]["id"] == "medquad:GARD_QA:doc:1", recs

    pubmed = {"id": "pqa_a:1", "source": "pubmedqa_artificial", "question": "Does aspirin reduce fever duration?",
              "context": CTX, "answer": "Aspirin shortens fever. More text.", "label": "yes"}
    recs += [pubmed, {**pubmed, "id": "pqa_a:2"},  # exact duplicate
             {**pubmed, "id": "pqa_a:3", "question": "Is the test question leaked?"}]
    kept, reasons = clean.clean_records(recs)
    assert reasons == {"empty_answer": 1}, reasons
    assert kept[0]["answer"].startswith("Fabry disease is"), "echoed question not stripped"

    test_rows = [{"question": "Is the test question leaked?", "context": CTX}]
    kept, dup, leak = dedupe.dedupe(kept, test_rows)
    assert (len(kept), dup, leak) == (2, 1, 1), (len(kept), dup, leak)

    assert chunk.truncate("A. " * 1000, 100).endswith(".") and len(chunk.truncate("A. " * 1000, 100)) <= 100
    pairs = [build_pairs.to_pair(chunk.chunk_record(r)) for r in kept]
    assert pairs[1]["response"] == "Answer: yes\nReason: Aspirin shortens fever.", pairs[1]

    good, bad = schema.validate(pairs + [{**pairs[1], "id": "x:1", "label": "no"}])
    assert len(good) == 2 and "label" in bad[0]["reason"], bad

    train, dev = split.split(list(range(10)), n_dev=3)
    assert len(train) == 7 and len(dev) == 3 and not set(train) & set(dev)
    print("etl ok")


if __name__ == "__main__":
    demo()
