from langchain_core.output_parsers import StrOutputParser
from app.models.gentral_langchain import get_llm
from app.prompts.eval_prompt import get_eval_prompt

def build_eval_chain():
    llm = get_llm()
    prompt = get_eval_prompt()

    return prompt | llm | StrOutputParser()