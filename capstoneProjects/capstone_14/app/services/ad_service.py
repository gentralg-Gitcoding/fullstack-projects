import random
from app.chains.ad_chain import build_ad_chain

marketing_angles = [
    "Focus on adventure",
    "Focus on saving money",
    "Focus on convenience",
    "Focus on eco-friendly travel",
    "Focus on limited-time urgency and fear of missing out",
    "Focus on time efficiency",
]

class AdService:
    def __init__(self):
        self.chain = build_ad_chain()

    def generate_ads(self, tone, audience, platform, bike_type, promotion, n=3):
        ads = []

        for _ in range(n):
            angle = random.choice(marketing_angles)

            result = self.chain.invoke({
                "tone": tone,
                "audience": audience,
                "platform": platform,
                "bike_type": bike_type,
                "promotion": promotion,
                "angle": angle
            })

            ads.append(result)

        return ads