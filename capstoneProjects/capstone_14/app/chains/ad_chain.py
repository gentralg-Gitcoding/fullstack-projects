from langchain_core.output_parsers import StrOutputParser
from app.models.gentral_langchain import get_llm
from app.prompts.ad_prompt import get_ad_prompt

def build_ad_chain():
    llm = get_llm()
    prompt = get_ad_prompt()

    return prompt | llm | StrOutputParser()