# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

PRESIGNED_URL="https://dobf1k6cxlizq.cloudfront.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kb2JmMWs2Y3hsaXpxLmNsb3VkZnJvbnQubmV0LyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2NzgzNjk3ODZ9fX1dfQ__&Signature=nwNK6IVRkLLVVaH-EvgT5JV0mieESMqvgjY35SPOgIFWbU5qhQvH-BBrFq08h-HI6zw80NSPd3PM8H4KXIJ1eZ606s224j0JrgQHv0ybZ3kXkmzsvQKDQvwGzgeStOoxk6cC3qaweiBs-uuNUT5N8WxGuqBau1OZUPkSwYBm1LH7gsC-cPXwdDLzQQo6~PqxLkJFFdpJlKe1jFK1tSuJJHL2xmWsyK~6MLaVfqSt1xvMFISJQbwz5U1Fiz4uc6U4YBfTbTgWoS5NUD1fiCc8NgV71FaxJWh84vPsHHRljgVCSY~~6Y932TszSXezZI~heGkncpfpNNfqiMh8ULS4sw__&Key-Pair-Id=K231VYXPC1TA1R"             # replace with presigned url from email
# MODEL_SIZE="7B,13B,30B,65B"  # edit this list with the model sizes you wish to download
MODEL_SIZE="7B"  # edit this list with the model sizes you wish to download
TARGET_FOLDER="weights/"             # where all files should end up

declare -A N_SHARD_DICT

N_SHARD_DICT["7B"]="0"
N_SHARD_DICT["13B"]="1"
N_SHARD_DICT["30B"]="3"
N_SHARD_DICT["65B"]="7"

echo "Downloading tokenizer"
wget ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
wget ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer_checklist.chk"

(cd ${TARGET_FOLDER} && md5sum -c tokenizer_checklist.chk)

for i in ${MODEL_SIZE//,/ }
do
    echo "Downloading ${i}"
    mkdir -p ${TARGET_FOLDER}"/${i}"
    for s in $(seq -f "0%g" 0 ${N_SHARD_DICT[$i]})
    do
        wget ${PRESIGNED_URL/'*'/"${i}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
    done
    wget ${PRESIGNED_URL/'*'/"${i}/params.json"} -O ${TARGET_FOLDER}"/${i}/params.json"
    wget ${PRESIGNED_URL/'*'/"${i}/checklist.chk"} -O ${TARGET_FOLDER}"/${i}/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${i}" && md5sum -c checklist.chk)
done