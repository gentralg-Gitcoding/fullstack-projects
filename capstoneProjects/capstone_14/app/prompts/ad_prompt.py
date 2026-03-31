from langchain_core.prompts import PromptTemplate

def get_ad_prompt():
    template="""
    Generate EXACTLY ONE {tone} advertisement.

    headline: 
    <text>

    body: 
    <text>

    CTA: 
    <text>

    Target Audience: {audience}
    Platform: {platform}
    BikeType: {bike_type}
    Promotion: {promotion}
    Marketing Focus: {angle}

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