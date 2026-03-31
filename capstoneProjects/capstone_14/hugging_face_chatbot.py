import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

# Suppress noisy key warnings from checkpoint loading
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# --- Configuration ---
# Model to use from HuggingFace Hub
model_name = 'Qwen/Qwen2.5-3B-Instruct'

# Temperature controls randomness (0.0 = deterministic, 1.0 = creative)
temperature = 0.7

# Maximum number of tokens to generate in each response
max_new_tokens = 150

# System prompt sets the assistant's behavior and personality
system_prompt = (
    'You are a professional marketing expert for BikeEase. '
    'Sunday rentals are free. '
    'College students get a 70% discount at all times. '
    'Every type of bike is available for use. '
)

marketing_angles = [
    "Focus on adventure",
    "Focus on saving money",
    "Focus on convenience",
    "Focus on eco-friendly travel",
    "Focus on limited-time urgency and fear of missing out",
    "Focus on time efficiency",
]

# Load tokenizer (converts text to numbers the model understands)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Select the best available device (GPU > CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the actual model (downloads first time, then cached locally)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

# Print the model architecture and number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f'\nLoaded model: {model_name} with {num_params/1e9:.2f} billion parameters on {device}')
print(f'\nArchitecture:\n')
print(model)


#Create a prompt template function so the model can be dynamically updated
def build_prompt(tone, audience, platform, promotion, bike_type, angle):
    return f"""
    Create a {tone} advertisement for BikeEase bike rentals.

    Target Audience: {audience}
    Platform: {platform}
    BikeType: {bike_type}
    Promotion: {promotion}

    Marketing Focus: {angle}

    Format like this:
    headline: punchy compelling headline
    body: persuasive sentences tailored to the audience
    call-to-action: a strong action sentence

    Requirements:
    - Match the tone to the platform:
        * Email -> more professional, less emojis
        * Social Media -> energetic, emojis are fully allowed
    - Keep it under 150 words
    - Output must be English ONLY
    - Do not include explanations, notes, or comments
    - Do not put hashtags unless the platform is for Social Media
    - Do not output the format labels headline, body and call-to-action only the content
    - Only output the ad
    """


def generate(messages):
    '''Generate a response from the model given a list of messages.'''

    # Convert conversation history to the format expected by the model
    # This uses the model's chat template (e.g., adds <|im_start|> tags for Qwen)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return text, not token IDs yet
        add_generation_prompt=True,  # Add the prompt for the assistant's response
    )

    # Tokenize the text (convert to numbers) and prepare for the model
    # 'pt' means PyTorch tensors, move to model's device (CPU or GPU)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    # Generate new tokens using the model
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # Limit response length
        temperature=temperature,        # Control randomness
        do_sample=True,                 # Use sampling (vs greedy decoding)
    )

    # Decode only the newly generated tokens (skip the input prompt)
    # This extracts just the assistant's response
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    '''Main conversation loop.'''
    print('Generating BikeEase Ad...')

    

    print(f'\nChatbot ready. Model: {model_name}, device: {device}, temperature: {temperature}')

    # Main conversation loop - runs until interrupted (Ctrl+C)
    while True:

        # Get user inputs to determine what type of Ad they want
        tone = input("Enter tone (fun, luxury, urgent, etc): ")
        audience = input("Enter target audience (students, commuters, tourists): ")
        platform = input("Enter platform (Instagram, Email, Google Ads): ")
        promotion = input("Enter promotion (discount, free trial, etc): ")
        bike_type = input("Enter bike type (electric, mountain, city): ")

        # Check for exit condition
        if (tone.lower() or audience.lower() or platform.lower() or promotion.lower() or bike_type.lower()) in ['exit', 'quit']:
            print('Exiting chatbot.')
            break


        print(f'\n{model_name}')
        # Generate response using fresh history to 
        for i in range(3):
            # Initialize conversation history with system prompt
            # Format matches OpenAI/HuggingFace chat message structure
            history = [{'role': 'system', 'content': system_prompt}]

            angle = random.choice(marketing_angles)

            user_message = {'role': 'user', 'content': build_prompt(tone, audience, platform, promotion, bike_type, angle)}
            history.append(user_message)

            response = generate(history)
            print(f"\nAd Variation {i+1}:\n{response}\n")
        
            # Add assistant's response to history for context in next turn
            model_message = {'role': 'assistant', 'content': response}
            history.append(model_message)

        cont = input("\nGenerate another? (yes/no): ")
        if cont.lower() != "yes":
            break


if __name__ == '__main__':
    main()