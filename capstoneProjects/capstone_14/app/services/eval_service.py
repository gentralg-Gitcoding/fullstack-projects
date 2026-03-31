from app.chains.eval_chain import build_eval_chain
from app.utils.parser import parse_scores

class EvalService:
    def __init__(self):
        self.chain = build_eval_chain()

    def score_ad(self, ad):
        raw = self.chain.invoke({"ad": ad})
        scores = parse_scores(raw)
        return scores

    def score_ads(self, ads):
        results = []

        for ad in ads:
            scores = self.score_ad(ad)
            results.append((ad, scores))

        return results