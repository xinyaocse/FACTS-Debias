# **FACTS-Debias**
Fact-Anchored Anti-Bias Supervision for Fact-Preserving LLM Debiasing

## Environmental Setup

We first set up the Python and GPU-related dependencies. Please install a version of PyTorch that is compatible with your local CUDA environment. You may refer to the official [PyTorch install guide](https://pytorch.org/) to determine the correct version.

Install PyTorch according to your CUDA version. For example:

```
pip install torch torchvision torchaudio
```

Then install the remaining dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

The project uses local language models and encoder/generator models. Please download the required models and place them under the `./Models` directory. The exact model list can be adjusted according to the experiments you want to reproduce.

Then, following the [Llama 2 install guide](https://github.com/facebookresearch/llama?tab=readme-ov-file#quick-start) to install Llama 2. **Note** that you should download the Llama 2 models, including the tokenizer file and model parameter files, and place them under the `./Models` directory.

In addition, download the T5-based encoder/generator model from Hugging Face, such as [FLAN-T5-Small](https://huggingface.co/google/flan-t5-small), and place it under the same `./Models` directory. In our experiments, FLAN-T5-Small is used for representation extraction, projection-based text generation, and constructed supervision generation.

An example directory structure is shown below:

```
|-- Models
    |-- flan-t5-small
    |   |-- config.json
    |   |-- generation_config.json
    |   |-- pytorch_model.bin
    |   |-- tokenizer.json
    |   |-- tokenizer_config.json
    |-- Llama-2-7b-chat-hf
    |-- Llama-2-13b-chat-hf
    |-- Alpaca-7B
    |-- Orca-7B
    |-- Platypus-7B
```

If you run experiments in an offline environment, please make sure that all Hugging Face models, tokenizers, and datasets are available locally before running the scripts.

## Datasets

The project uses biased, anti-biased, and factual texts to learn a fact-anchored representation decomposition for debiasing supervision construction. For each gender group, the three types of data are stored together and distinguished by type labels. 

After downloading or constructing the data, the datasets should be organized as follows:

```
|-- Data
    |-- female
    |-- male
    |-- other
    |-- Assess_Gender_Bias
```

The Assess_Gender_Bias directory contains the test set used for gender-bias and factual-preservation evaluation. You will find the original public dataset here: [GenderCARE](https://github.com/kstanghere/GenderCARE-ccs24/tree/main).

## Running the Experiments