# Dockerised Text2Video-Zero

This repository provides a Dockerised version of the [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero) project.

## Original Project

For the original codebase and additional details, please visit the official repository:  
[Text2Video-Zero by Picsart AI Research](https://github.com/Picsart-AI-Research/Text2Video-Zero)

## Getting Started

### Prerequisites

- Docker installed on your machine.
- Access to a compatible GPU (NVIDIA recommended).

### Usage

To run the Dockerised version, execute the following command:

```bash
./run.sh <gpu_id>
```

- Replace `<gpu_id>` with the ID of the GPU you want to use.  
  For example, use `0` for the default GPU.
- **Note:** Multi-GPU support is not configured, as it was unnecessary for this lightweight model.

### Customizing Prompts

The file `data/test.json` contains a list of prompts to be processed when the script runs. These prompts will be converted into GIFs and saved in the `output` directory. You are welcome to modify the prompts in `test.json` to suit your needs.

### Model Configuration

The model settings have been optimized to align with the best parameters provided in the original project.