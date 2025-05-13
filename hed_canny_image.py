from controlnet_aux import HEDdetector, CannyDetector
import cv2, torch, numpy as np
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)

src_img = Image.open("./test_imgs/0000002002-1_1.tif")

# build HED map 
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
hed_map = hed(src_img.convert("RGB"))
hed_map = cv2.GaussianBlur(cv2.cvtColor(np.array(hed_map), cv2.COLOR_RGB2GRAY),
                           (0, 0), sigmaX=3)
hed_map = Image.fromarray(hed_map)

# build Canny map 
low_threshold = 150
high_threshold = 250
canny = CannyDetector()
canny_map = canny(src_img.convert("RGB"), low_threshold=low_threshold, high_threshold=high_threshold)

#load ControlNets
control_hed  = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-hed",   torch_dtype=torch.float16)
control_canny = ControlNetModel.from_pretrained(
                   "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
          "runwayml/stable-diffusion-v1-5",
          controlnet=[control_hed, control_canny],     # dual CN
          torch_dtype=torch.float16)

pipe.load_ip_adapter("h94/IP-Adapter", 
                     "models",
                     "ip-adapter-plus_sd15.safetensors", 
                     weight=1.0)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
con = [0.20,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6]
for c in con:
    out = pipe(
            prompt="satellite photo",
            negative_prompt="colorful shadows, neon tint, cartoon texture, oversaturated",
            image=src_img,
            control_image=[hed_map, canny_map],            # order matches CN list
            ip_adapter_image=src_img,
            strength=0.25,
            num_inference_steps=32,
            guidance_scale=6,
            control_guidance_start=[0.0, 0.0],             # both start at step 0
            control_guidance_end  =[0.75, 0.75],           # both fade at 75 %
            controlnet_conditioning_scale=[0.35, c],    # HED a bit stronger
    ).images[0]

    out.save(f"./output/satellite_realistic_hed_canny_{low_threshold}_{high_threshold}_{c}.tif")
# canny = CannyDetector()
# for low, high in [(50,100), (100,200), (150,250),(200,400)]:
#     canny_map = canny(src_img.convert("RGB"), low_threshold=low, high_threshold=high)
#     canny_map.save(f"./output/canny_{low}_{high}.png")





