import re
import os
import imageio

def prompt_to_name(prompt):
    # Convert prompt to lowercase and replace spaces with underscores
    folder_name = re.sub(r'[^\w\s-]', '', prompt.lower())
    folder_name = re.sub(r'[-\s]+', '_', folder_name)
    return folder_name


def save_frames_as_gif(frames, save_name):
    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    # Save frames as GIF with each frame displayed for 0.1 seconds
    imageio.mimsave(save_name, frames, fps=8)
