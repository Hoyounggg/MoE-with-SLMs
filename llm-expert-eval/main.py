# main.py
# ---------------------------------------------------
# A simple interactive script that uses the classifier to detect domain,
# loads the appropriate expert model, and returns the generated answer.
# ---------------------------------------------------

from classifier.infer_classifier import DomainClassifier
from router.model_router import ExpertRouter
from router.prompts import PROMPTS

def main_loop():
    clf = DomainClassifier()
    router = ExpertRouter()

    print("Interactive expert router. Type 'exit' to quit.")
    while True:
        user_query = input("\nUser: ")
        if user_query.lower().strip() == "exit":
            break
        domain = clf.predict(user_query)
        print(f"[System] Detected domain: {domain}")
        router.load_expert(domain)
        prompt = PROMPTS.get(domain, "") + user_query
        answer = router.generate(prompt)
        print("\nExpert:\n", answer)

if __name__ == "__main__":
    main_loop()
