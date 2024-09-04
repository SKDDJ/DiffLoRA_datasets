# **DiffLoRA Datasets Generation**

This Python script uses the PhotoMaker model to generate images based on prompts. It's designed to work with a set of images and text prompts, generating a new image for each combination of image and prompt.

## **Dependencies**

The script uses several libraries, including:

- `torch`
- `numpy`
- `PIL` (Python Imaging Library)
- `argparse`
- `labml`
- `diffusers`
- `huggingface_hub`

## **How It Works**

The script first sets up the environment and loads the necessary models and weights. It then defines several helper functions:

- `image_grid()`: Creates a grid of images.
- `load_all_image_paths()`: Loads all image paths from a specified directory, starting from a specified ID.
- `read_and_process_file()`: Reads a text file and processes each line, optionally adding a suffix to each line.

The script then parses command-line arguments to get the paths for the prompt and negative prompt files, the directory containing the images, the save path for the generated images, the start ID, and the suffix text.

The script reads and processes the prompt and negative prompt files, and loads all image paths. It then enters a loop where it generates a new image for each image path. For each image, it creates a new directory and generates a new image for each prompt. The generated images are saved to the new directory.

## **Usage**

You can run the script from the command line with the following arguments:

- `-prompt_path`: Path to the file containing the prompts.
- `-negative_prompt_path`: Path to the file containing the negative prompts.
- `-images_directory`: Directory containing the images.
- `-save_path`: Directory where the generated images will be saved.
- `-start_id`: ID of the image to start from.
- `-suffix_text`: Text to add to the end of each line in the prompt file.

Example:

```python
python main.py --prompt_path /path/to/prompts.txt --negative_prompt_path /path/to/negative_prompts.txt --images_directory /path/to/images --save_path /path/to/save --start_id 123 --suffix_text " a woman img"
```

## **Note**

This script assumes that each image is named with the person's name. The generated images are named with the person's name, the prompt index, and the image index.
