# Consensus CoT

## Example test run command:

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


# Large Language Models are Zero-Shot Reasoners

This is the official implementation of `Large Language Models are Zero-Shot Reasoners` .


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