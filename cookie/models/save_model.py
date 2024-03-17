import torch
import torchvision.models as models

# Initialize ResNet model from torchvision
model = models.resnet152(pretrained=True)

# Script the model using torch.jit.script
script_model = torch.jit.script(model)

# Save the scripted model to a file
script_model.save('./models/deployable_model.pt')

"""
torch-model-archiver \
    --model-name my_fancy_model \
    --version 1.0 \
    --serialized-file ./models/deployable_model.pt \
    --export-path model_store \
    --extra-files cookie/models/index_to_name.json \
    --handler cookie/models/img_cls_handler:ImageClassifierHandler

torchserve --start --ncs --model-store model_store --models my_fancy_model=my_fancy_model.mar


curl http://127.0.0.1:8080/predictions/my_fancy_model -T data/my_cat.jpg

"""
