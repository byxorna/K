#!/usr/bin/env python3
import os
import torch
import time
import yaml
import secrets
from torch import autocast
from diffusers import StableDiffusionPipeline

#prompt = "a rain drenched night market through the receding rearview mirror of the Syndicate's executive transport ship"
prompt = "a cheerful cartoon bear greeting a family at the trail in a national forest"
count=3 # how many images to generate

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
height=384
width=384
num_inference_steps=50 # 50 default
guidance_scale=7 # 7 or 8.5
cuda_max_split_size_mb = 64
initial_seed = secrets.randbits(32) # None
output_id = secrets.token_hex(16)

## Setup and Configuration
print("initializing...")
gpu_mem = torch.cuda.mem_get_info()[0] # bytes
if height%8!=0 or width%8!=0:
    raise "height and width need to be divisible by 8"

pipeline_precision = torch.float32
if gpu_mem < 10*(1024**3): # check gpu vmem < 10GB, use float16
    pipeline_precision = torch.float16

cuda_conf_env_var = "PYTORCH_CUDA_ALLOC_CONF"
cuda_conf = ""
if cuda_conf_env_var in os.environ:
    cuda_conf = os.environ[cuda_conf_env_var]

if cuda_conf == "" and gpu_mem < 10*(1024**3): # use a smaller split size on smaller cards for better efficiency
    cuda_conf = "max_split_size_mb:{max_split_size_mb}".format(max_split_size_mb=cuda_max_split_size_mb)

os.environ[cuda_conf_env_var] = cuda_conf

generator = torch.Generator(device)
if initial_seed is not None:
    generator = generator.manual_seed(initial_seed)

#print("memory_stats(): {stats}\nmemory_summary(): {summary}".format(stats=torch.cuda.memory_stats(), summary=torch.cuda.memory_summary()))

print("prompt: {} (count: {})".format(prompt, count))

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
            #    'precision': pipeline_precision,
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
            torch_dtype=pipeline_precision).to(device)

    try:
        images = pipe([prompt]*count, generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width)["sample"]
        for i in list(range(len(images))):
            image = images[i]
            fname = "output-{ts}-{id}-{idx}.png".format(id=md['id'], ts=int(md['timestamp']), idx=i)
            image.save(fname)
            print("view with `xdg-open {}` (metadata: {})".format(fname, md['self']))
    except RuntimeError as e:
        md['error'] = e
        print("caught error: {}", e)

write_metadata("./{}.yaml".format(md['id']), md)
