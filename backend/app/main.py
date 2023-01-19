from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
# import sys
# sys.path.append('C:\Users\mahes\Desktop\MLProjects\CruxImgCl\backend\app\model.py')
from model import CNN
import os
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
import base64

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    model = CNN()
    with open(os.path.join(os.path.dirname(__file__), 'FinalModel.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model


model = load_model()

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img = Image.open(image.file)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(img).unsqueeze(0)

    output, channel_attention_map, spatial_attention_map = model(img)
    _, pred = torch.max(output, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.imshow(channel_attention_map[0, 0].detach().numpy(), cmap='jet')
    ax1.set_title("channel attention map")
    ax2.imshow(spatial_attention_map[0, 0].detach().numpy(), cmap='jet')
    ax2.set_title("spatial attention map")
    ax3.imshow(output.detach().numpy(), cmap='jet')
    ax3.set_title("output")
    plt.savefig('attention_map.png')
    # attention_map = open('attention_map.png', 'rb')
    with open("attention_map.png", "rb") as f:
        img_data = f.read()
        base64_img = base64.b64encode(img_data).decode()
        attention_map = f"data:image/png;base64,{base64_img}"
    #return attention map and prediction
    return {"prediction": classes[pred.item()], "attention_map": attention_map}


if __name__ == "__main__":

    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
