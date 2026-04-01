'''
Gradio text-to-image generator with selectable models.

This demo provides a web UI where users can:
- Choose between different text-to-image models
- Custom user input
- Image showcasing per submission

Usage:
    python ./gradio_stable_fusion_design.py

Warning:
    Models once ran in the app will save to C:/Users/{your_username/.cache/huggingface
    Be sure to clean up these models once done


Overview:
    In this project, you will explore the transformative potential of Hugging Face's Stable Diffusion model and Gradio UI in the creative design industry. 
    The project outlines hypothetical scenarios to demonstrate AI's ability to generate images from text prompts. 
    These scenarios offer a glimpse into a future where technology enhances creativity.
'''

import gradio as gr
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---

device = "cuda" if torch.cuda.is_available() else "cpu"

model_choices = [
    'CompVis/stable-diffusion-v1-4', 
    'stabilityai/stable-diffusion-3.5-medium', 
    # 'black-forest-labs/FLUX.1-schnell',     #very large model
    'bakebrain/bergraffi-berlin-graffiti',
    'DeepFloyd/IF-I-M-v1.0'
]

stable3_diff_model_names = ['stabilityai/stable-diffusion-3.5-medium']

# flux_model_names = ['black-forest-labs/FLUX.1-schnell']

diff_model_names = ['DeepFloyd/IF-I-M-v1.0']

stable_diff_model_names = ['CompVis/stable-diffusion-v1-4','bakebrain/bergraffi-berlin-graffiti']





#Image generation function for gradio Interface
def generate_image(model_selection, prompt):
    print('Generating Image...')
    print(f'\nModel selected: {model_selection}')
    print(f'\nInput Text: {prompt}\n')

    # ---- Initialize Stable Diffusion 3 Pipeline ----

    if(model_selection in stable3_diff_model_names):
        stable3_diff_pipe = StableDiffusion3Pipeline.from_pretrained(model_selection, torch_dtype=torch.bfloat16)
        stable3_diff_pipe = stable3_diff_pipe.to(device)

        image = stable3_diff_pipe(
            prompt,
            num_inference_steps=20,     #Number of iterations to apply denoising (Higher = More detailed but more processing time)
            guidance_scale=8,       #How strict the model follows the text prompt. (higher = more strict)
        ).images[0]

        return image
    

    # --- Initialize FLUX.1 Pipeline ---

    # elif(model_selection in flux_model_names):
    #     

    #     flux_pipe = FluxPipeline.from_pretrained(model_selection, torch_dtype=torch.bfloat16)
    #     flux_pipe = flux_pipe.to(device)

    #     image = flux_pipe(
    #         prompt,
    #         num_inference_steps=20,
    #         guidance_scale=8,
    #     ).images[0]
    #     return image


    # ---- Initialize Diffusion Pipeline ----

    elif(model_selection in diff_model_names):
        diff_pipe = DiffusionPipeline.from_pretrained(model_selection, torch_dtype=torch.bfloat16)
        diff_pipe.to(device)

        # diff_pipe.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        diff_pipe.enable_model_cpu_offload()

        image = diff_pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=8,
        ).images
        image = pt_to_pil(image)[0]

        return image


    # ---- Initialize Stable Diffusion Pipeline ----

    else:
        stable_pipe = StableDiffusionPipeline.from_pretrained(model_selection, torch_dtype=torch.bfloat16)
        stable_pipe = stable_pipe.to(device)

        image = stable_pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=8,
        ).images[0]

        return image


# --- Build Gradio UI ---

# Use Gradio Blocks for custom layout with multiple input controls
with gr.Blocks(title='Gradio Image demo') as demo:
    
    # Page title and description
    gr.Markdown('# Gradio Image Generation demo')
    
    # Model selector - radio buttons for different models
    with gr.Row():
        backend_selector = gr.Radio(
            choices=model_choices,
            value='CompVis/stable-diffusion-v1-4',
            label='Please select your model:',
            info='StabilityAI has long loading times.'
        )
        user_input = gr.Textbox(label="Enter text for image")

    #Image block next to the row for image output
    output_image = gr.Image(label="Output Image", buttons=['download', 'share', 'fullscreen'])

    # Interface with models and textbox input for image output
    demo_interface = gr.Interface(
        fn=generate_image,
        inputs=[backend_selector,user_input],
        outputs=output_image
    )

# Launch the Gradio app
if __name__ == '__main__':
    demo.launch()