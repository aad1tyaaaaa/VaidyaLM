"""Raw shard -> list of records {id, source, question, context, answer, label}."""
import json
import pathlib

from bs4 import BeautifulSoup
from lxml import etree

COLUMNS = ["id", "source", "question", "context", "answer", "label"]


def _text(t):
    t = t or ""
    return BeautifulSoup(t, "html.parser").get_text("\n") if "<" in t else t


def parse_medquad_file(path):
    path = pathlib.Path(path)
    root = etree.parse(str(path), etree.XMLParser(recover=True)).getroot()
    return [
        {
            "id": f"medquad:{path.parent.name}:{path.stem}:{qa.get('pid')}",  # file names are unique, doc ids are not
            "source": "medquad",
            "question": _text(qa.findtext("Question")),
            "context": "",
            "answer": _text(qa.findtext("Answer")),
            "label": None,
        }
        for qa in root.iter("QAPair")
    ]


def parse_shard(shard):
    shard = pathlib.Path(shard)
    if shard.is_dir():
        return [r for f in sorted(shard.glob("*.xml")) for r in parse_medquad_file(f)]
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    if shard.name.startswith("synthetic_"):
        from src.etl import synth

        return synth.label_rows(rows)
    return [
        {"id": f"pqa_a:{r['id']}", "source": "pubmedqa_artificial", "question": r["question"],
         "context": r["context"], "answer": r["long_answer"], "label": r["label"]}
        for r in rows
    ]
