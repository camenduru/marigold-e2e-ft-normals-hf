import gradio as gr
import cv2
import matplotlib
import numpy as np
import os
from PIL import Image
import spaces
import torch
import tempfile
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

# from depth_anything_v2.dpt import DepthAnythingV2
from Marigold.marigold import MarigoldPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
# import xformers

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
checkpoint_path = "GonzaloMG/marigold-e2e-ft-depth"
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
# try:
#     pipe.enable_xformers_memory_efficient_attention()
# except ImportError:
#     pass  # run without xformers
pipe = pipe.to(DEVICE)
pipe.unet.eval()

# model_configs = {
#     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
# }
# encoder2name = {
#     'vits': 'Small',
#     'vitb': 'Base',
#     'vitl': 'Large',
#     'vitg': 'Giant', # we are undergoing company review procedures to release our giant model checkpoint
# }
# encoder = 'vitl'
# model_name = encoder2name[encoder]
# model = DepthAnythingV2(**model_configs[encoder])
# filepath = hf_hub_download(repo_id=f"depth-anything/Depth-Anything-V2-{model_name}", filename=f"depth_anything_v2_{encoder}.pth", repo_type="model")
# state_dict = torch.load(filepath, map_location="cpu")
# model.load_state_dict(state_dict)
# model = model.to(DEVICE).eval()

title = "# ..."
description = """... **...**"""


# def predict_depth(image):
#     return model.infer_image(image)
    
@spaces.GPU
def predict_depth(image): #, processing_res, model_choice, current_model):
    with torch.no_grad():
        pipe_out = pipe(image, denoising_steps=1, ensemble_size=1, noise="zeros", normals=False, processing_res=768, match_input_res=True)
    pred = pipe_out.depth_np
    pred_colored = pipe_out.depth_colored
    return pred, pred_colored

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)

    with gr.Row()
        submit = gr.Button(value="Compute Depth")
        processing_res_choice = gr.Radio(
                [
                    ("Recommended (768)", 768),
                    ("Native", 0),
                ],
                label="Processing resolution",
                value=768,
            )

    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
    raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download",)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image):

        if image is None:
            print("No image uploaded.")
            return None
    
        pil_image = Image.fromarray(image.astype('uint8'))
        depth_npy, depth_colored = predict_depth(pil_image)
    
        # Save the npy data (raw depth map)
        # tmp_npy_depth = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        # np.save(tmp_npy_depth.name, depth_npy)
    
        # Save the grayscale depth map
        depth_gray = (depth_npy * 65535.0).astype(np.uint16)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        Image.fromarray(depth_gray).save(tmp_gray_depth.name, mode="I;16")
    
        # Save the colored depth map
        tmp_colored_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        depth_colored.save(tmp_colored_depth.name)
   
        return [(image, depth_colored),  tmp_gray_depth.name, tmp_colored_depth.name]

        # h, w = image.shape[:2]

        # depth = predict_depth(image[:, :, ::-1])

        # raw_depth = Image.fromarray(depth.astype('uint16'))
        # tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        # raw_depth.save(tmp_raw_depth.name)

        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        # depth = depth.astype(np.uint8)
        # colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

        # gray_depth = Image.fromarray(depth)
        # tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        # gray_depth.save(tmp_gray_depth.name)

        # return [(original_image, colored_depth), tmp_gray_depth.name, tmp_raw_depth.name]

    submit.click(on_submit, inputs=[input_image, processing_res_choice], outputs=[depth_image_slider, gray_depth_file, raw_file])

    example_files = os.listdir('assets/examples')
    example_files.sort()
    example_files = [os.path.join('assets/examples', filename) for filename in example_files]
    example_files = [[image, 768] for image in example_files]
    examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file, raw_file], fn=on_submit)


if __name__ == '__main__':
    demo.queue().launch(share=True)