import torch
import re
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, GenerationConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class EvalService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = "Qwen/Qwen2.5-3B-Instruct"
        self.max_new_tokens = 150
        self.temperature = 0.5

    def parse_scores(self, text):
        scores = {}

        for key in ["Clarity", "Persuasiveness", "Urgency", "Relevance", "Creativity", "Total"]:
            match = re.search(f"{key}:\\s*(\\d+)", text)
            if match:
                scores[key.lower()] = int(match.group(1))
            else:
                scores[key.lower()] = 0

        return scores

    def get_eval_prompt(self):
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

    def get_llm(self):
        
        print(f'Using device: {self.device}')

        generation_config = GenerationConfig.from_pretrained(
            self.model, 
            max_new_tokens=self.max_new_tokens, 
            temperature=self.temperature, 
            do_sample=True,
        )

        pipe = pipeline(
            "text-generation",
            model=self.model,
            return_full_text=False,
            device=self.device,
        )
        pipe.generation_config = generation_config

        return HuggingFacePipeline(pipeline=pipe, model_kwargs={"stop": ["\n\n", "---"]})

    def score_ad(self, ad):
        chain = self.get_eval_prompt() | self.get_llm() | StrOutputParser()
        raw = chain.invoke({"ad": ad})
        scores = self.parse_scores(raw)
        return scores

    def score_ads(self, ads):
        results = []

        for ad in ads:
            scores = self.score_ad(ad)
            results.append((ad, scores))

        return results