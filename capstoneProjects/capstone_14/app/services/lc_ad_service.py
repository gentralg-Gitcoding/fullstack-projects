import logging
import torch
import random
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, GenerationConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class AdService:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.max_new_tokens = 150
        self.temperature = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.marketing_angles = [
            "Focus on adventure",
            "Focus on saving money",
            "Focus on convenience",
            "Focus on eco-friendly travel",
            "Focus on limited-time urgency and fear of missing out",
            "Focus on time efficiency",
        ]

        self.generation_config = GenerationConfig.from_pretrained(
            self.model_name, 
            max_new_tokens=self.max_new_tokens, 
            temperature=self.temperature, 
            do_sample=True,
        )

        self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                return_full_text=False,
                device=self.device,
        )
        self.pipeline.generation_config = self.generation_config

        self.model = (HuggingFacePipeline(pipeline=self.pipeline, model_kwargs={"stop": ["\n\n", "---"]}))


    def get_ad_prompt(self, tone, audience, platform, bike_type, promotion, angle):
        template="""
        Generate EXACTLY ONE advertisement.

        headline: 
        <text>

        body: 
        <text>

        CTA: 
        <text>

        Tone: {tone}
        Audience: {audience}
        Platform: {platform}
        Bike Type: {bike_type}
        Promotion: {promotion}
        Focus: {angle}

        Rules:
        - Output ONLY headline, body, cta
        - Match the tone to the platform:
            * Email -> more professional, less emojis
            * Social Media -> energetic, emojis are fully allowed
        - Keep it under 150 words
        - Output must be English ONLY
        - Do not include explanations, notes, or comments
        - Do not put hashtags unless the platform is for Social Media
        - Only output the ad
        """

        return PromptTemplate(
            input_variables=[
                "tone", 
                "audience", 
                "platform",
                "bike_type", 
                "promotion", 
                "angle"
            ],
            template=template
        )


    def generate_ads(self, tone, audience, platform, bike_type, promotion, n=1):
        ads = []

        for _ in range(n):
            angle = random.choice(self.marketing_angles)
            chain = self.get_ad_prompt(tone, audience, platform, bike_type, promotion, angle) | self.model | StrOutputParser()

            result = chain.invoke({
                "tone": tone,
                "audience": audience,
                "platform": platform,
                "bike_type": bike_type,
                "promotion": promotion,
                "angle": angle
            })

            ads.append(result)

        return ads