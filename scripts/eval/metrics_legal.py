
import re
from rouge_score import rouge_scorer


def normalize(t):
    return re.sub(r"[^a-z0-9\s]", "", t.lower()) if t else ""


def _best_answer(answers):
    if not answers:
        return ""
    # answers is a list of dicts with 'body' and 'score'
    best = max(answers, key=lambda x: x.get("score", 0))
    return best.get("body") or ""


def evaluate_legal(preds, data, answerability_min_chars: int = 50):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouges, ok = [], 0
    for ex in data:
        ex_id = str(ex["question_id"])
        pred = preds.get(ex_id, "")
        gold_answer = _best_answer(ex.get("answers", []))
        r = scorer.score(normalize(gold_answer), normalize(pred))
        rouges.append(r["rougeL"].fmeasure)
        if len(pred.strip()) >= answerability_min_chars:
            ok += 1
    n = len(data)
    return {
        "rougeL": sum(rouges) / n if n else 0.0,
        "answerability": ok / n if n else 0.0,
    }
