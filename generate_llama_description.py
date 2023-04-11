from transformers import LlamaTokenizer,LlamaForCausalLM
from pdb import set_trace as st

import json

import itertools

device = "cpu"

tokenizer =LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to(device)

def stringtolist(description):
    res = []
    for descriptor in description.split('\n'):
        if (descriptor != ''):
            if (descriptor.startswith('- ')) or descriptor[0].isdigit():
                res.append(descriptor[2:])
            else:
                res.append(descriptor)
    return res


def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

# def generate_prompt(category_name: str):
#     # you can replace the examples with whatever you want; these were random and worked, could be improved
#     return f"""
# Q: What are useful features for distinguishing a {category_name} in a photo?
# A: There are several useful visual features to tell there is a {category_name} in a photo:
# -
# """

def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))


batch = tokenizer(
    # generate_prompt('cat'),
    "What are useful features for distinguishing a dog in a photo?",
    return_tensors="pt", 
    add_special_tokens=False
)

batch = {k: v.to(device) for k, v in batch.items()}

# st()

generated = model.generate(batch["input_ids"],max_length=100)
print(tokenizer.decode(generated[0]))

# from transformers import AutoTokenizer, AutoModelForCausalLM

# import transformers

# from transformers import LlamaTokenizer,LlamaForCausalLM

# tokenizer =LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

# model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
