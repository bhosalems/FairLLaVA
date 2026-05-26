import requests
import torch
from PIL import Image
from io import BytesIO

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# Model
disable_torch_init()

model_path = "/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/checkpoints/biomedclip_cxr_518-lora-1e-1e-4-20250914111503/checkpoint-1852"
model_base = "lmsys/vicuna-7b-v1.5"
model_name = "llavarad"
conv_mode = "v1"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)

# Prepare query
image_file = "https://openi.nlm.nih.gov/imgs/512/253/253/CXR253_IM-1045-1001.png" # CXR w pneumothorax from Open-I
query = "<image>\nDescribe the findings of the chest x-ray.\n"

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], query)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = load_image(image_file)
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].half().unsqueeze(0).cuda()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

stopping_criteria = KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        use_cache=True,
        stopping_criteria=[stopping_criteria])

outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
outputs = outputs.strip()
print(outputs)
# Large left pneumothorax is present with apical pneumothorax component
#  measuring approximately 3.4 cm in craniocaudal dimension, and a basilar
#  component overlying the left hemidiaphragm, with visceral pleural line just
#  below the left seventh posterior rib.  Cardiomediastinal contours are normal.
#  The lungs are clear.  No pleural effusion.