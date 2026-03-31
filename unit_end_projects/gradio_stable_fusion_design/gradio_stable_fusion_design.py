'''
Gradio chatbot with selectable backend (Ollama or llama.cpp).

This demo provides a web UI where users can:
- Choose between Ollama and llama.cpp backends
- Customize the system prompt
- Have multi-turn conversations with context

Usage:
    


Overview:
    In this project, you will explore the transformative potential of Hugging Face's Stable Diffusion model and Gradio UI in the creative design industry. 
    The project outlines hypothetical scenarios to demonstrate AI's ability to generate images from text prompts. 
    These scenarios offer a glimpse into a future where technology enhances creativity.
'''

import os
import gradio as gr
from dotenv import load_dotenv
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import torch
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---

# Temperature controls randomness (0.0 = deterministic, 1.0+ = creative)
temperature = 0.7

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Initialize Stability AI backend ---

stabilityai_model_name = 'stabilityai/stable-diffusion-3.5-medium'

stabilityai_pipe = StableDiffusion3Pipeline.from_pretrained(stabilityai_model_name, torch_dtype=torch.bfloat16)
stabilityai_pipe = stabilityai_pipe.to(device)




# --- Initialize llama.cpp backend (OpenAI-compatible API) ---

# Get server URL from environment, default to localhost
# llamacpp_server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

# Configure API key and base URL based on server location
# Localhost uses 'dummy' key, remote servers use PERDRIZET_API_KEY
# if llamacpp_server.startswith('localhost') or llamacpp_server.startswith('127.'):
#     llamacpp_api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
#     llamacpp_base_url = f'http://{llamacpp_server}/v1'

# else:
#     llamacpp_api_key = os.environ.get('PERDRIZET_API_KEY')
#     llamacpp_base_url = f'https://{llamacpp_server}/v1'



compvis_model_name = 'CompVis/stable-diffusion-v1-4'

compvis_pipe = StableDiffusionPipeline.from_pretrained(compvis_model_name, torch_dtype=torch.bfloat16)
compvis_pipe = compvis_pipe.to(device)


def generate_image(backend_selection):
    print('Generating Image...')
    print(f'\nInput Text: {backend_selection}')

    if(backend_selection == stabilityai_model_name):
        prompt = "A capybara holding a sign that reads Hello World"
        image = stabilityai_pipe(
            prompt,
            num_inference_steps=40,
            guidance_scale=4.5,
        ).images[0]
        # image.save("capybara.png")
        return image

    elif(backend_selection == compvis_model_name):
        prompt = "a photo of an astronaut riding a horse on mars"
        image = compvis_pipe(prompt).images[0]
        # image.save("astronaut_rides_horse.png")
        return image

    else:
        return "C:/Users/gahhh/OneDrive/Pictures/521-5213277_starlys-pokemon-mudkip-cartoon.png"


def respond(message, history, backend, system_prompt):
    '''Sends message to selected model backend, gets response back.
    
    Args:
        message: User's current message
        history: List of [user_msg, assistant_msg] pairs from Gradio
        backend: Either 'Ollama' or 'llama.cpp'
        system_prompt: System prompt to set model behavior
    
    Returns:
        Response string from the model (or error message if backend unavailable)
    '''
    
    # --- Ollama backend ---
    if backend == 'Ollama':
        try:
            # Build message list in LangChain format (SystemMessage, HumanMessage, AIMessage)
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history to maintain context
            # Gradio passes history as list of [user, assistant] pairs
            for item in history:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    user_msg, assistant_msg = item[0], item[1]
                    messages.append(HumanMessage(content=user_msg))
                    messages.append(AIMessage(content=assistant_msg))
            
            # Add current user message
            messages.append(HumanMessage(content=message))
            
            # Invoke Ollama model and return response
            response = ollama_client.invoke(messages)
            return response.content
        
        except Exception as e:
            # Return helpful error message if Ollama server is unreachable
            error_msg = (
                f'**Ollama backend is unavailable**\n\n'
                f'Make sure the Ollama server is running:\n'
                f'```bash\n'
                f'ollama serve\n'
                f'```\n\n'
                f'Error details: {str(e)}'
            )
            return error_msg
    
    # --- llama.cpp backend ---
    else:
        try:
            # Build message list in OpenAI format (dict with 'role' and 'content')
            messages = [{'role': 'system', 'content': system_prompt}]
            
            # Add conversation history to maintain context
            for item in history:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    user_msg, assistant_msg = item[0], item[1]
                    messages.append({'role': 'user', 'content': user_msg})
                    messages.append({'role': 'assistant', 'content': assistant_msg})
            
            # Add current user message
            messages.append({'role': 'user', 'content': message})
            
            # Call llama.cpp server using OpenAI-compatible API
            response = llamacpp_client.chat.completions.create(
                model=llamacpp_model,
                messages=messages,
                temperature=temperature,
            )
            
            # Extract and return response text
            return response.choices[0].message.content
        
        except Exception as e:

            # Return helpful error message if llama.cpp server is unreachable
            error_msg = (
                # f'**llama.cpp backend is unavailable**\n\n'
                # f'Make sure the llama-server is running at: `{llamacpp_base_url}`\n\n'
                # f'To start the server:\n'
                # f'```bash\n'
                # f'llama.cpp/build/bin/llama-server -m <model.gguf> --host 0.0.0.0 --port 8502\n'
                # f'```\n\n'
                # f'Or configure remote server in `.env` file.\n\n'
                f'Error details: {str(e)}'
            )
            return error_msg



# --- Build Gradio UI ---

# Use Gradio Blocks for custom layout with multiple input controls
with gr.Blocks(title='Gradio Image demo') as demo:
    
    # Page title and description
    gr.Markdown('# Gradio Image Generation demo')
    
    # Backend selector - radio buttons for different models
    with gr.Row():
        backend_selector = gr.Radio(
            choices=['CompVis/stable-diffusion-v1-4', 'stabilityai/stable-diffusion-3.5-medium'],
            value='CompVis/stable-diffusion-v1-4',
            label='Model Backend',
            info=f'Stability AI: {stabilityai_model_name} | CompVis: {compvis_model_name}'
        )

    output_image = gr.Image(label="Output Image")
    
    # Chat interface with backend and system prompt as additional inputs
    demo_interface = gr.Interface(
        fn=generate_image,
        inputs=backend_selector,
        additional_inputs=[backend_selector],
        outputs=output_image
    )


# Launch the Gradio app
if __name__ == '__main__':
    demo.launch()