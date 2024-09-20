###########################################################################################
# Code based on the Hugging Face Space of Depth Anything v2
# https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/app.py
###########################################################################################

import gradio as gr
import cv2
import matplotlib
import numpy as np
import os
from PIL import Image
import torch
import tempfile
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from Marigold.marigold import MarigoldPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
variant = None
checkpoint_path = "GonzaloMG/marigold-e2e-ft-normals"
unet         = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")   
vae          = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")  
text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")  
tokenizer    = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer") 
scheduler    = DDIMScheduler.from_pretrained(checkpoint_path, timestep_spacing="trailing", subfolder="scheduler") 
pipe = MarigoldPipeline.from_pretrained(pretrained_model_name_or_path = checkpoint_path,
                                        unet=unet, 
                                        vae=vae, 
                                        scheduler=scheduler, 
                                        text_encoder=text_encoder, 
                                        tokenizer=tokenizer, 
                                        variant=variant, 
                                        torch_dtype=dtype, 
                                        )
pipe = pipe.to(DEVICE)
pipe.unet.eval()


title = "# End-to-End Fine-Tuned Marigold for Normals Estimation"
description = """ Please refer to our [paper](https://arxiv.org/abs/2409.11355) and [GitHub](https://vision.rwth-aachen.de/diffusion-e2e-ft) for more details."""

def predict_normals(image, processing_res_choice):
    with torch.no_grad():
        pipe_out = pipe(image, denoising_steps=1, ensemble_size=1, noise="zeros", normals=True, processing_res=processing_res_choice, match_input_res=True)
    pred = pipe_out.normal_np
    pred_colored = pipe_out.normal_colored
    return pred, pred_colored

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Normals Prediction demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        normals_image_slider = ImageSlider(label="Surface Normals with Slider View", elem_id='img-display-output', position=0.5)

    with gr.Row():
        submit = gr.Button(value="Compute Normals")
        processing_res_choice = gr.Radio(
                [
                    ("Recommended (768)", 768),
                    ("Native", 0),
                ],
                label="Processing resolution",
                value=768,
            )

    raw_file = gr.File(label="Raw Normals Data (.npy)", elem_id="download")

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image, processing_res_choice):

        if image is None:
            print("No image uploaded.")
            return None
    
        pil_image = Image.fromarray(image.astype('uint8'))
        normal_npy, normal_colored = predict_normals(pil_image, processing_res_choice)
    
        # Save the npy data (raw normals)
        tmp_npy_normal = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(tmp_npy_normal.name, normal_npy)
    
        # Save the grayscale depth map
        # depth_gray = (depth_npy * 65535.0).astype(np.uint16)
        # tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        # Image.fromarray(depth_gray).save(tmp_gray_depth.name, mode="I;16")
    
        # Save the colored normals map
        tmp_colored_normal = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        normal_colored.save(tmp_colored_normal.name)
   
        return [(image, normal_colored), tmp_npy_normal.name]

    submit.click(on_submit, inputs=[input_image, processing_res_choice], outputs=[normals_image_slider, raw_file])

    example_files = os.listdir('assets/examples')
    example_files.sort()
    example_files = [os.path.join('assets/examples', filename) for filename in example_files]
    example_files = [[image, 768] for image in example_files]
    examples = gr.Examples(examples=example_files, inputs=[input_image, processing_res_choice], outputs=[normals_image_slider, raw_file], fn=on_submit)


if __name__ == '__main__':
    demo.queue().launch(share=True, inline=False)