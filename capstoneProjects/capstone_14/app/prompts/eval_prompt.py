from langchain_core.prompts import PromptTemplate

def get_eval_prompt():
    template = """
    You are a marketing expert evaluating an advertisement.

    Ad:
    {ad}

    Score the ad from 1 to 10 on the following:

    - Clarity
    - Persuasiveness
    - Urgency
    - Relevance to audience
    - Creativity

    Return ONLY in this format:

    Clarity: X
    Persuasiveness: X
    Urgency: X
    Relevance: X
    Creativity: X
    Total: X
    """

    return PromptTemplate(
        input_variables=["ad"],
        template=template
    )