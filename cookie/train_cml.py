# assume we have a trained model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# creating a dummy train_dataloader
train_dataloader = [(torch.randn(3, 224, 224), torch.randint(0, 10, (1,))) for _ in range(100)]
# creating a dummy model
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 10, 3),
    torch.nn.Flatten(),
    torch.nn.Linear(10 * 222 * 222, 10),
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
    cm=confmat,
)
plt.savefig("confusion_matrix.png")
