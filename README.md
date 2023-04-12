# Boosted Prompt Ensembles for Large Language Models

Silviu Pitis, Michael R. Zhang, Andrew Wang, Jimmy Ba

[Arxiv](https://arxiv.org/abs/2304.05970)

This is code for our paper on prompt boosting. Please see our paper for detailed experiments.
This code is built off the official implementation of [Large Language Models are Zero-Shot Reasoners](https://github.com/kojima-takeshi188/zero_shot_cot).

## Abstract

Methods such as chain-of-thought prompting and self-consistency have pushed the frontier of language model reasoning performance with no additional training. To further improve performance, we propose a prompt ensembling method for large language models, which uses a small dataset to construct a set of few shot prompts that together comprise a "boosted prompt ensemble". The few shot examples for each prompt are chosen in a stepwise fashion to be 
"hard" examples on which the previous step's ensemble is uncertain. We show that this outperforms single-prompt output-space ensembles and bagged prompt-space ensembles on the GSM8k and AQuA datasets, among others. We propose both train-time and test-time versions of boosted prompting that use different levels of available annotation and conduct a detailed empirical study of our algorithm.




## Example command:

```
python dynamicprompting.py --dataset aqua --size-limit 12 --min_prompt_size 3
```

## Installation
Make sure you have Python>=3.8 installed on your machine.

```
pip install --upgrade pip
pip install python-dotenv openai transformers
pip install datasets cohere
git clone https://github.com/kojima-takeshi188/zero_shot_cot.git
cd zero_shot_cot
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
```



## Set your OpenAI API key
```
# https://beta.openai.com/account/api-keys
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```

## Set arguments.
```
model=gpt3-xl # {"gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"}. "gpt3" is the smallest model.
dataset=multiarith # We can use other datasets. See help for the details.
limit_dataset_size=10 # This is important to save your budget. If you want to use all the samples in a dataset, set 0.
api_time_interval=1.0 # Caution. The API allows users request API up to 60 times in a minutes, otherwise errors happen.
```

## Quick Start

### [Prompt Boosting on gsm8k] 200 Train to Boost entire test set
```
python dynamicprompting.py --dataset=gsm8k --split=train --min_agreement=0.9 --size_limit=200 --seed 0 --prompt_mode boosted --tag TRAIN_BASE200
python dynamicprompting.py --dataset=gsm8k --min_agreement=0.9 --size_limit=3000 --seed 0 --boosted_prompts=logs/gsm8k_boosted/code-davinci-002_200_0.9_0_append_TRAIN_BASE200.pickle --tag BASE200
```

### [Self Consistency 100 on gsm8k] 
```
python dynamicprompting.py --dataset=gsm8k --min_agreement=1.0 --seed 0 --prompt_mode self_consistency --tag SC
```

### [Test Time Prompt Boosting on MMLU570] 
```
python dynamicprompting.py --dataset=mmlu570 --split=train --min_agreement=0.9 --size_limit=300 --seed 0 --prompt_mode boosted --tag TRAIN_BASE
python dynamicprompting.py --dataset=mmlu570 --min_agreement=0.9 --seed 0 --boosted_prompts=logs/mmlu570_boosted/code-davinci-002_300_0.9_0_append_TRAIN_BASE.pickle --tag BASE
```

