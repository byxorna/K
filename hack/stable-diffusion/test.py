#!/usr/bin/env python3
import os
from typing import Dict
import torch
import time
import yaml
import secrets
from torch import autocast
from diffusers import StableDiffusionPipeline

#prompt = "a butterfly's markings swirling into the chaotic void of the horsehead nebulae"
prompt = "Federation Starship Enterprise NCC-1701-d blueprint"
#prompt = "an executive helicopter departing from the corporate headquarters at night"
#prompt = "a cheerful cartoon bear greeting a family at the trail in a national forest"
iterations=5
batch_size=1 # how many images to generate

cuda_conf_env_var = "PYTORCH_CUDA_ALLOC_CONF"
#height=384
#width=512
#num_inference_steps=50 # 50 default, number of denoising steps
#guidance_scale=8.5 # 7.5 default, (7-8.5 reasonable?) , scale for classifier-free guidance
cuda_max_split_size_mb = 64
initial_seed = secrets.randbits(64) # None

## Setup and Configuration
def write_metadata(output_file, metadata):
    f = open(output_file, 'w')
    f.write(yaml.dump(metadata))
    f.close()

class KRunner:
    def __init__(self, settings = {}) -> None:
        self.output_dir = "output"
        self.settings = {
            'model_id': "CompVis/stable-diffusion-v1-4",
            'num_inference_steps': 50,
            'guidance_scale': 8.5,
            'height': 384,
            'width': 512,
            'batch_size': 1,
        } | settings

        self.width = self.settings['width']
        self.height = self.settings['height']
       
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        gpu_mem = torch.cuda.mem_get_info()[0] # bytes
        if self.height%8!=0 or self.width%8!=0:
            raise ValueError("height and width need to be divisible by 8")

        cuda_conf = ""
        if cuda_conf_env_var in os.environ:
            cuda_conf = os.environ[cuda_conf_env_var]

        self.torch_pipeline_precision = torch.float32
        if gpu_mem < 10*(1024**3): # check gpu vmem < 10GB, use float16
            self.torch_pipeline_precision = torch.float16
            if cuda_conf == "":
                cuda_conf = "max_split_size_mb:{max_split_size_mb}".format(max_split_size_mb=cuda_max_split_size_mb)

        os.environ[cuda_conf_env_var] = cuda_conf

        self.generator = torch.Generator(self.device)
        if initial_seed is not None:
            self.generator = self.generator.manual_seed(initial_seed)
        #print("initialized!")
        #print("memory_stats(): {stats}\nmemory_summary(): {summary}".format(stats=torch.cuda.memory_stats(), summary=torch.cuda.memory_summary()))

    def infer(self, prompt, batch_size):
        model_id = self.settings['model_id']
        with autocast(self.device):
            pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    use_auth_token=True,
                    torch_dtype=self.torch_pipeline_precision).to(self.device)

            for iteration in list(range(0, iterations)):
                if iterations > 1:
                    print("inferrence iteration {}".format(iteration))

                print("prompt: {} ({}x{}, batch_size: {})".format(prompt, self.width, self.height, batch_size))
                ident = secrets.token_hex(16)
                md = {
                        'id': ident,
                        'settings': self.settings,
                        'runtime': {
                            'model_id': model_id,
                            'seed': self.generator.initial_seed(),
                        },
                        'prompt': prompt,
                        'timestamp': time.time(),
                        'output':  'TODO: image encoding as base64',
                        'self': "{}.yaml".format(ident),
                }

                try:
                    images = pipe([prompt]*batch_size, generator=self.generator, num_inference_steps=self.settings['num_inference_steps'], guidance_scale=self.settings['guidance_scale'], height=self.settings['height'], width=self.settings['width'])["sample"]
                    for i in list(range(len(images))):
                        image = images[i]
                        fname = "output-{ts}-{id}-{idx}.png".format(id=md['id'], ts=int(md['timestamp']), idx=i)
                        image.save(os.path.join(self.output_dir, fname))
                        print("view with `xdg-open {}` (metadata: {})".format(fname, md['self']))
                except RuntimeError as e:
                    md['error'] = e
                    print("caught error: {}", e)

                write_metadata(os.path.join(self.output_dir, md['id'] + ".yaml"), md)

app = KRunner()
app.infer(prompt, batch_size)
