#!/usr/bin/env python3
import os
from typing import Dict
import torch
import time
import yaml
import secrets
from torch import autocast
from diffusers import StableDiffusionPipeline
import argparse

#prompt = "claymation motion picture still of a cat cleaning herself telephoto"
#prompt = "a butterfly's markings swirling into the chaotic void of the horsehead nebulae"
#prompt = "tilt shift helicopter view of a dense urban downtown"
#prompt = "an executive helicopter departing from the corporate headquarters at night"
#prompt = "a cheerful cartoon bear greeting a family at the trail in a national forest"
batch_size=1 # how many images to generate

cuda_conf_env_var = "PYTORCH_CUDA_ALLOC_CONF"
#height=384
#width=512
#num_inference_steps=50 # 50 default, number of denoising steps
#guidance_scale=8.5 # 7.5 default, (7-8.5 reasonable?) , scale for classifier-free guidance
cuda_max_split_size_mb = 64

## Setup and Configuration
def write_metadata(output_file, metadata):
    f = open(output_file, 'w')
    f.write(yaml.dump(metadata))
    f.close()

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Text prompt to perform inference on",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="initial random seed",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1,
        help="iterations",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="number of images to generate per iteration (this scales the memory required)",
    )
    return parser
    return parser

class KRunner:
    # TODO: move all settings to named parameters
    def __init__(self, seed=None, batch_size=1, height=384, width=512, guidance_scale=8.5,num_inference_steps=50, model_id="CompVis/stable-diffusion-v1-4", **settings) -> None:
        self.settings = {
            'model_id': model_id,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'height': height,
            'width': width,
        } | settings

        if seed is None:
            seed = secrets.randbits(64)

        self.seed = seed
        self.batch_size = batch_size
        self.output_dir = "output"

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
        if self.seed is not None:
            self.generator = self.generator.manual_seed(self.seed)
        #print("memory_stats(): {stats}\nmemory_summary(): {summary}".format(stats=torch.cuda.memory_stats(), summary=torch.cuda.memory_summary()))

    def infer(self, prompt, batch_size=1, iterations=1) -> list[dict]:
        model_id = self.settings['model_id']
        results = []
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
                        #'output':  'TODO: image encoding as base64',
                        'self': "{}.yaml".format(ident),
                        'outputs': [],
                }

                try:
                    images = pipe([prompt]*batch_size, generator=self.generator, num_inference_steps=self.settings['num_inference_steps'], guidance_scale=self.settings['guidance_scale'], height=self.settings['height'], width=self.settings['width'])["sample"]
                    for i in list(range(len(images))):
                        image = images[i]
                        fname = "output-{ts}-{id}-{idx}.png".format(id=md['id'], ts=int(md['timestamp']), idx=i)
                        md['outputs'].append(fname)
                        image.save(os.path.join(self.output_dir, fname))
                        #print("view with `xdg-open {}` (metadata: {})".format(fname, md['self']))
                except RuntimeError as e:
                    md['error'] = e
                    print("caught error: {}", e)

                write_metadata(os.path.join(self.output_dir, md['id'] + ".yaml"), md)
                results.append(md)
        return results
        # end infer()

parser = get_parser()
opt, _ = parser.parse_known_args()

app = KRunner(seed=opt.seed)
iterations = app.infer(opt.prompt, batch_size=opt.batch_size, iterations=opt.iterations)

for it in iterations:
    #print("> {}: {} ({})".format(it['id'], os.path.join(app.output_dir, it["self"]), len(it['outputs'])))
    for out in it['outputs']:
        print("> {}: {}".format(it['id'], os.path.join(app.output_dir, out)))
