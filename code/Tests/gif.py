import os
import csv
import time
from PIL import Image
from tqdm import tqdm
from utils.metrics import Metrics
from utils.utils import prompt_to_name
import numpy as np
from itertools import product

from utils.load import load_data, load_t2vz
from utils.utils import save_frames_as_gif


def run_gif_test(first_only=False):
    device = 'cuda'
    input_json = load_data('../data/test.json')
    if first_only:
        input_json = input_json[:1]
    GG = GifGeneration(device)
    GG(input_json, save_gif=True, save_stats=True)


class GifGeneration:
    def __init__(self, device, save_path='../output/tests'):
        self.device = device
        self.save_path = save_path
        self.metrics = Metrics(device)

    def get_stats(self, prompt, frames, metrics):
        if not frames:
            raise ValueError("Frames cannot be empty or None")
        start_time = time.time()
        stats = {}

        if 'clip' in metrics:
            stats['clip'] = self.metrics.clip_score_frames(frames, prompt)   

        if 'ms_ssim' in metrics:
            stats['ms_ssim'] = self.metrics.ms_ssim_frames(frames)    

        if 'lpips' in metrics:
            stats['lpips'] = self.metrics.lpips_frames(frames)

        if 'temporal_consistency_loss' in metrics:
            losses = self.metrics.temporal_consistency_loss(frames)
            stats['temporal_consistency_loss'] = losses[0]
            stats['temporal_consistency_loss_warp'] = losses[1]
            stats['temporal_consistency_loss_smooth'] = losses[2]

        end_time = time.time()
        print(f"Time taken to get stats: {end_time - start_time:.2f} seconds")

        return stats
    
    def save_stats(self, stats_data):
        if not stats_data:
            raise ValueError("stats_data cannot be empty")
        os.makedirs(self.save_path, exist_ok=True)
        csv_path = os.path.join(self.save_path, "gif_stats.csv")
        
        # Check if the file already exists
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:  # Open in append mode
            fieldnames = ['prompt', 'clip', 'ms_ssim', 'lpips', 
                          'temporal_consistency_loss', 'temporal_consistency_loss_warp', 'temporal_consistency_loss_smooth']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()  # Write header only if file does not exist
            
            for stat in stats_data:
                writer.writerow(stat)
        
        print(f"Stats appended to {csv_path}")

    def __call__(self, input_list, save_gif=False, save_stats=False, metrics=['clip', 'ms_ssim', 'lpips', 'temporal_consistency_loss']):
        pipe = load_t2vz(self.device)
        stats_data = []


        for prompt in tqdm(input_list, desc="Running tests", unit="test"):  

            frames = self.pipe_call(prompt, pipe)

            if save_gif:
                filename = prompt_to_name(prompt)[:30] + ".gif"
                save_name = os.path.join(self.save_path, filename)
                save_frames_as_gif(frames, save_name)

            if save_stats:
                stats = self.get_stats(prompt, frames, metrics)
                stat_entry = {
                    'prompt': prompt,
                    'clip': stats.get('clip', np.nan),
                    'ms_ssim': stats.get('ms_ssim', np.nan),
                    'lpips': stats.get('lpips', np.nan),
                    'temporal_consistency_loss': stats.get('temporal_consistency_loss', np.nan),
                    'temporal_consistency_loss_warp': stats.get('temporal_consistency_loss_warp', np.nan),
                    'temporal_consistency_loss_smooth': stats.get('temporal_consistency_loss_smooth', np.nan)
                }
                stats_data.append(stat_entry)
                

        if save_stats:
            self.save_stats(stats_data)

        return stats_data
    
    def pipe_call(self, prompt, pipe):
        result = pipe(prompt=prompt).images
        result = [(r * 255).astype("uint8") for r in result]
        result = [Image.fromarray(r) for r in result]
        return result