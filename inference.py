from dinov2.models.vision_transformer import vit_base
import torch

from torchvision import transforms

from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# load model
model = vit_base()


# keys are name of parameters, values are actual param
# equip model with weights
state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
                                                map_location="cpu")
model.load_state_dict(state_dict)

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

#preprocess image
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pixel_Values = transformations(image).unsqueeze(0)

# run model
outputs = model.forward_features(pixel_Values)

print("Outputs", outputs)