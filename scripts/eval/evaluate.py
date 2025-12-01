
from scripts.eval.metrics_biomedical import evaluate_biomedical
from scripts.eval.metrics_legal import evaluate_legal
from scripts.eval.metrics_code import evaluate_code

def evaluate_domain(domain, predictions, dataset):
    if domain == "biomedical":
        return evaluate_biomedical(predictions, dataset)
    if domain == "legal":
        return evaluate_legal(predictions, dataset)
    if domain == "code":
        return evaluate_code(predictions, dataset)
    raise ValueError(domain)
