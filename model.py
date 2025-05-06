import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

data_dir        = 'Pistachio_Image_Dataset'
checkpoint_path = 'resnet18_pistachio_checkpoint.pth'
output_dir      = 'reports'
cm_png          = os.path.join(output_dir, 'confusion_matrix.png')
gradcam_dir     = os.path.join(output_dir, 'gradcam')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(gradcam_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
inv_normalize = transforms.Normalize(
    mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
    std =[1/s    for s in             [0.229,0.224,0.225]]
)

full_ds = datasets.ImageFolder(data_dir, transform=eval_transform)
class_names = full_ds.classes
n_tot   = len(full_ds)
n_train = int(0.8 * n_tot)
n_val   = n_tot - n_train
torch.manual_seed(42)
_, val_ds = random_split(full_ds, [n_train, n_val])
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state)
model = model.to(device).eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outs = model(imgs)
        preds = outs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
fig.colorbar(im, ax=ax)
plt.title("Confusion Matrix")
fig.tight_layout()
fig.savefig(cm_png)
plt.close(fig)
print(f"Saved confusion matrix → {cm_png}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        def forward_hook(module, inp, outp):
            self.activations = outp.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2,3))[0]  
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
        return cam

target_layer = model.layer4[1].conv2
gradcam = GradCAM(model, target_layer)

for idx in range(10):
    img_tensor, label = val_ds[idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)
    cam = gradcam(input_tensor)

    orig = inv_normalize(img_tensor)
    orig = orig.permute(1,2,0).cpu().numpy()
    orig = np.clip(orig, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = heatmap + orig
    overlay = overlay / overlay.max()

    fn = os.path.join(gradcam_dir, f"gradcam_{idx}_true-{class_names[label]}.png")
    Image.fromarray(np.uint8(overlay*255)).save(fn)

print(f"Saved Grad-CAM overlays → {gradcam_dir}")
