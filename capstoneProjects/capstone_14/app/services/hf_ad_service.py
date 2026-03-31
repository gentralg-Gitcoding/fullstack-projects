import logging
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class HFAdService:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.temperature = 0.5
        self.max_new_tokens = 150

        self.system_prompt = (
            "You are a professional marketing expert for BikeEase. "
            "Sunday rentals are free. "
            "College students get a 70% discount at all times. "
            "Every type of bike is available."
        )

        self.marketing_angles = [
            "Focus on adventure",
            "Focus on saving money",
            "Focus on convenience",
            "Focus on eco-friendly travel",
            "Focus on limited-time urgency and fear of missing out",
            "Focus on time efficiency",
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto"
        ).to(self.device)

    def build_prompt(self, tone, audience, platform, promotion, bike_type, angle):
        return f"""
        Generate EXACTLY ONE advertisement.

        Headline:
        <text>

        Body:
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
        - Output ONLY Headline, Body, CTA
        - No hashtags
        - No explanations
        - English only
        - Keep under 150 words
        """

    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
        )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # def clean_output(self, text):
    #     import re
    #     match = re.search(r"(Headline:.*?CTA:.*)", text, re.DOTALL)
    #     return match.group(1).strip() if match else text.strip()

    def generate_ads(self, tone, audience, platform, bike_type, promotion, n=1):
        ads = []

        for _ in range(n):
            history = [{'role': 'system', 'content': self.system_prompt}]
            angle = random.choice(self.marketing_angles)

            prompt = self.build_prompt(
                tone, audience, platform, promotion, bike_type, angle
            )

            history.append({'role': 'user', 'content': prompt})

            # raw = self.generate(history)
            # clean = self.clean_output(raw)

            # ads.append(clean)
            ads.append(self.generate(history))

        return ads