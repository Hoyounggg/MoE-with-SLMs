
import re
from rouge_score import rouge_scorer


def normalize(t):
    return re.sub(r"[^a-z0-9\s]", "", t.lower()) if t else ""


def extract_decision(t):
    t = t.lower()
    for d in ("yes", "no", "maybe"):
        if d in t:
            return d
    return "unknown"


def evaluate_biomedical(preds, data):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    correct, rouges = 0, []
    for ex in data:
        ex_id = str(ex["pubid"])
        gold_decision = (ex["final_decision"] or "").lower()
        gold_answer = ex.get("long_answer") or ""
        pred = preds.get(ex_id, "")
        if extract_decision(pred) == gold_decision:
            correct += 1
        r = scorer.score(normalize(gold_answer), normalize(pred))
        rouges.append(r["rougeL"].fmeasure)
    n = len(data)
    return {
        "decision_accuracy": correct / n if n else 0.0,
        "rougeL": sum(rouges) / n if n else 0.0,
    }
