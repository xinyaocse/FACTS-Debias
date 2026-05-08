# **FACTS-Debias**
Fact-Anchored Anti-Bias Supervision for Fact-Preserving LLM Debiasing

## Environment Setup

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

In addition, download the T5-based encoder/generator model from Hugging Face, such as [FLAN-T5-Small](https://huggingface.co/google/flan-t5-small), and place it under the same `./Models` directory. 

An example directory structure is shown below:

```
|-- Models
    |-- flan-t5-small
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

The Assess_Gender_Bias directory contains the test set used for gender-bias and factual-preservation evaluation. 

You will find the original public dataset here: [GenderCARE](https://github.com/kstanghere/GenderCARE-ccs24/tree/main).

## Running the Experiments

The main experimental steps are shown below.

### 1. Representation decomposition

To train the representation decomposition model and extract the shared and anti-bias-specific representations, run:

```
python main.py \
  --embedding_model_path Models/flan-t5-small \
  --train_file <TRAIN_FILE> \
  --test_file <TEST_FILE> \
  --checkpoint_dir <CHECKPOINT_DIR> \
  --vector_dir <VECTOR_DIR>
```

This step learns two types of representations:

- `z_sa`: the preservable/shared component that captures label-invariant semantics;
- `z_td`: the anti-bias-specific component used for constructing debiasing supervision.

### 2. Train text-space classifier

To train a classifier for evaluating whether generated texts match the desired anti-bias label, run:

```
python train_text_space_clf.py \
  --train_csv <TRAIN_CSV> \
  --t5_path <T5_MODEL_PATH> \
  --save_path <SAVE_PATH>
```

The trained classifier can be used during candidate reranking and evaluation.

### 3. Construct anti-bias supervision from representations

Generate text-based supervision from the extracted representations:

```bash
python finetune_vec2text_offline.py \
  --load_from <TRAINED_PROJECTION_DIR> \
  --z_test <Z_TEST> \
  --labels_csv <LABELS_CSV> \
  --clf_ckpt <CHECKPOINT> \
  --gtr_path <MODEL_PATH> \
  --infer_out <OUTPUT_CSV>
```


### 4. Fine-tune downstream LLMs

Fine-tune the downstream LLM.

### 5. Evaluation results

Once fine-tuning is complete, evaluate the debiasing effectiveness and factual preservation of the adapted models.

After fine-tuning, generate model responses for the evaluation prompts:

```
python assess.py \
  --model_name_or_path_baseline <MODEL_PATH> \
  --input_prompts <PROMPT_FILE> \
  --output_response <OUTPUT_FILE> \
  --model_type <MODEL_TYPE> \
  --max_new_tokens <MAX_NEW_TOKENS>
```

Then compute the BPR-based evaluation results:

```
python evaluate_BPR.py \
  --model_path <MODEL_PATH> \
  --response_file <RESPONSE_FILE> \
  --group_type <GROUP_TYPE> \
  --model_type <MODEL_TYPE>
```
