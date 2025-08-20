# Tools
This project is done using mainly **PyTorch**. Additionally, all the dependencies have been included in the YAML file provided, namely `compactAiLanguageModel.yml` in the home directory. To create your own environment and run the project, follow these steps (replace `<name>` with the environment name you prefer, for example `calm_env`):

```bash
# 1. Create the environment from the YAML file with your chosen name
conda env create --name <name> -f ~/compactAiLanguageModel.yml

# Example:
# conda env create --name calm_env -f ~/compactAiLanguageModel.yml

# 2. Activate the environment (use the same <name> you picked above)
conda activate <name>

# Example:
# conda activate calm_env

# 3. Verify installation (optional)
python -c "import torch; print(torch.__version__)"
```

## Navigation Panel
Use this navigation panel to move forward or backward through the tutorial, or jump straight to the homepage whenever you like.<br>
[Proceed to the next section: Byte-Pair Encoding](/documentation/markdowns/tokenization.md)<br>
[Return to the previous section: Introduction](/documentation/markdowns/Introduction.md)<br>
[Back to the table of contents](/)<br>