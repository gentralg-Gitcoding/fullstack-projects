import re

def parse_scores(text):
    scores = {}

    for key in ["Clarity", "Persuasiveness", "Urgency", "Relevance", "Creativity", "Total"]:
        match = re.search(f"{key}:\\s*(\\d+)", text)
        if match:
            scores[key.lower()] = int(match.group(1))
        else:
            scores[key.lower()] = 0

    return scores