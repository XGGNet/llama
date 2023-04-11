import requests



API_URL = "https://api-inference.huggingface.co/models/decapoda-research/llama-7b-hf"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})




# Q: What are useful visual features for distinguishing a lemur in a photo?
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

# Q: What are useful features for distinguishing a dog in a photo?
# A: There are several useful visual features to tell there is a dog in a photo:
# -