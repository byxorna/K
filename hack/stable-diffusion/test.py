#!/usr/bin/env python3
import os
import torch
import time
import yaml
import secrets
from torch import autocast
from diffusers import StableDiffusionPipeline

prompt = "a butterfly's markings swirling into the chaotic void of the horsehead nebulae"
#prompt = "an executive helicopter departing from the corporate headquarters at night"
#prompt = "a cheerful cartoon bear greeting a family at the trail in a national forest"
count=1 # how many images to generate

output_dir = "output"
model_id = "CompVis/stable-diffusion-v1-4"
cuda_conf_env_var = "PYTORCH_CUDA_ALLOC_CONF"
device = "cuda"
height=384
width=512
num_inference_steps=50 # 50 default, number of denoising steps
guidance_scale=7 # 7 or 8.5, scale for classifier-free guidance
cuda_max_split_size_mb = 64
initial_seed = secrets.randbits(64) # None
output_id = secrets.token_hex(16)
torch_pipeline_precision = torch.float32
cuda_conf = ""

## Setup and Configuration
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("initializing {} ({})...".format(output_id, device))
gpu_mem = torch.cuda.mem_get_info()[0] # bytes
if height%8!=0 or width%8!=0:
    raise "height and width need to be divisible by 8"

if cuda_conf_env_var in os.environ:
    cuda_conf = os.environ[cuda_conf_env_var]

if gpu_mem < 10*(1024**3): # check gpu vmem < 10GB, use float16
    torch_pipeline_precision = torch.float16
    if cuda_conf == "":
        cuda_conf = "max_split_size_mb:{max_split_size_mb}".format(max_split_size_mb=cuda_max_split_size_mb)

os.environ[cuda_conf_env_var] = cuda_conf

generator = torch.Generator(device)
if initial_seed is not None:
    generator = generator.manual_seed(initial_seed)

#print("memory_stats(): {stats}\nmemory_summary(): {summary}".format(stats=torch.cuda.memory_stats(), summary=torch.cuda.memory_summary()))

print("prompt: {} ({}x{}, count: {})".format(prompt, width, height, count))

md = {
        'id': output_id,
        'seed': generator.initial_seed(),
        'settings': {
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'height': height,
            'width': width,
            'count': count,
            #'torch': {
            #    'precision': torch_pipeline_precision,
            #    'foo': 'bar',
            #},
        },
        'model_id': model_id,
        'prompt': prompt,
        'timestamp': time.time(),
        'output':  'TODO: image encoding as base64',
        'self': "{}.yaml".format(output_id),
}

def write_metadata(output_file, metadata):
    f = open(output_file, 'w')
    f.write(yaml.dump(metadata))
    f.close()

with autocast(device):
    pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=True,
            torch_dtype=torch_pipeline_precision).to(device)

    try:
        images = pipe([prompt]*count, generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width)["sample"]
        for i in list(range(len(images))):
            image = images[i]
            fname = "output-{ts}-{id}-{idx}.png".format(id=md['id'], ts=int(md['timestamp']), idx=i)
            image.save(os.path.join(output_dir, fname))
            print("view with `xdg-open {}` (metadata: {})".format(fname, md['self']))
    except RuntimeError as e:
        md['error'] = e
        print("caught error: {}", e)

write_metadata(os.path.join(output_dir, md['self']), md)
