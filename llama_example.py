from transformers import LlamaTokenizer,LlamaForCausalLM
from pdb import set_trace as st

device = "cpu"

print( len([
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]) )

tokenizer =LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to(device)

# def generate_prompt(category_name: str):
#     # you can replace the examples with whatever you want; these were random and worked, could be improved
#     return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
# A: There are several useful visual features to tell there is a lemur in a photo:
# - four-limbed primate
# - black, grey, white, brown, or red-brown
# - wet and hairless nose with curved nostrils
# - long tail
# - large eyes
# - furry bodies
# - clawed hands and feet

# Q: What are useful visual features for distinguishing a television in a photo?
# A: There are several useful visual features to tell there is a television in a photo:
# - electronic device
# - black or grey
# - a large, rectangular screen
# - a stand or mount to support the screen
# - one or more speakers
# - a power cord
# - input ports for connecting to other devices
# - a remote control

# Q: What are useful features for distinguishing a {category_name} in a photo?
# A: There are several useful visual features to tell there is a {category_name} in a photo:
# -
# """

def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
"
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet
"

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
"
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control
"

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
" "
"""


## 会复述prompt

batch = tokenizer(
    #"The capital of Canada is",
    # "There are several useful visual features to tell there is a dog in a photo:", 
    generate_prompt('dog'),
    return_tensors="pt", 
    add_special_tokens=False
)

batch = {k: v.to(device) for k, v in batch.items()}

# st()

generated = model.generate(batch["input_ids"], max_length=300)
print(tokenizer.decode(generated[0]))

# from transformers import AutoTokenizer, AutoModelForCausalLM

# import transformers

# from transformers import LlamaTokenizer,LlamaForCausalLM

# tokenizer =LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

# model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
