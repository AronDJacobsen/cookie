# Save this script as image_classifier_handler.py

from PIL import Image
import torch
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

class ImageClassifierHandler(BaseHandler):
    def initialize(self, context):
        self.model = self.load_model()

    def preprocess(self, data):
        # Perform any necessary preprocessing on the input data
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(data)
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def postprocess(self, data):
        # Perform any necessary postprocessing on the output data
        return torch.nn.functional.softmax(data, dim=1).detach().numpy()

    def inference(self, data):
        # Perform inference using the model
        return self.model(data)

_service = ImageClassifierHandler()
