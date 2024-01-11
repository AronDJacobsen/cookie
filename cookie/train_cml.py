# assume we have a trained model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch


# creating a dummy train_dataloader for simple NN

dataset = torch.utils.data.TensorDataset(
    torch.rand(100, 10), torch.randint(0, 2, (100,))
)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

model = torch.nn.Sequential(
    torch.nn.Linear(10, 2),
    torch.nn.Softmax(dim=-1),
)


preds, target = [], []
for batch in train_dataloader:
    x, y = batch
    probs = model(x)
    preds.append(probs.argmax(dim=-1))
    target.append(y.detach())

target = torch.cat(target, dim=0)
preds = torch.cat(preds, dim=0)

report = classification_report(target, preds)
with open("classification_report.txt", "w") as outfile:
    outfile.write(report)
confmat = confusion_matrix(target, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confmat,
    #cm=confmat,
)
plt.savefig("confusion_matrix.png")
