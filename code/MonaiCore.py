# %% [markdown]
# ## Verificar GPU
!nvidia-smi

# %% [markdown]
# ## Instalación de MONAI y Scikit-learn
!python -c "import monai" || pip install -qU "monai[ignite, nibabel, torchvision, tqdm]==1.6.1"
!python -c "import sklearn" || pip install -qU "scikit-learn==1.6.1"

# %% [markdown]
# ## Verificar conexion con cuda
import torch
print("¿GPU disponible en PyTorch?", torch.cuda.is_available())

import sys
print(sys.version)

# %% [markdown]
# ## Importación de Bibliotecas
import os
import shutil
import tempfile
from glob import glob
import tree

import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.data import decollate_batch, partition_dataset_classes
from monai.networks.nets import DenseNet121
from monai.transforms import (
    EnsureChannelFirstd, Compose, LoadImaged, RandFlipd, RandRotated,
    RandZoomd, ScaleIntensityd, Activations, AsDiscrete, EnsureTyped
)
from monai.utils import set_determinism

# %%
print_config()

# %% [markdown]
# ## Creación de Directorio
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# %% [markdown]
# ## Descarga de DataSet MedMNIST2D
resource = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
md5 = "0bc7306e7427e00ad1c5526a6677552d"

compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
data_dir = os.path.join(root_dir, "MedNIST")

if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5, hash_type="md5")

print("¡Dataset descargado y verificado correctamente en:", data_dir)
# Muestra las primeras carpetas y un resumen de la estructura
!ls -l {data_dir}
# Muestra los primeros 15 archivos dentro de la carpeta AbdomenCT
!ls -la {data_dir}/AbdomenCT | head -n 15
# %% [markdown]
# ## Semilla Aleatoria para la Reproducibilidad
set_determinism(seed=0)
rseed = 12345678

# %% [markdown]
# ## Exploración de Datos 
subdirs = sorted(filter(os.path.isdir, glob(f"{data_dir}/*")))
class_names = list(map(os.path.basename, subdirs))
num_class = len(class_names)

image_files = [sorted(glob(f"{d}/*")) for d in subdirs]

num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = sum(image_files, [])
image_class = sum([[i] * n for i, n in enumerate(num_each)], [])

num_total = len(image_class)
image_width, image_height = PIL.Image.open(image_files_list[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")

# %% [markdown]
# ## Visualización Aleatoria de Imágenes del DataSet
plt.subplots(2, 5, figsize=(8, 4))
for i, k in enumerate(np.random.randint(num_total, size=10)):
    im = PIL.Image.open(image_files_list[k])
    plt.subplot(2, 5, i + 1)
    plt.axis("off")
    plt.title(class_names[image_class[k]])
    plt.imshow(np.array(im), cmap="gray", vmin=0, vmax=255)

# %% [markdown]
# ## Partición de Datos: Entrenamiento (80%), Validación (10%) y Prueba (10%)
val_frac = 0.1
test_frac = 0.1
length = len(image_files_list)
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

# Creación de listas de diccionarios
train_files = [{"image": img, "label": lbl} for img, lbl in zip(train_x, train_y)]
val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_x, val_y)]
test_files = [{"image": img, "label": lbl} for img, lbl in zip(test_x, test_y)]

print(f"Training count: {len(train_x)}, Validation count: {len(val_x)}, Test count: {len(test_x)}")

# %% [markdown]
# ## Definición de las Transformaciones para Imágenes
# CORRECCIÓN: Se corrigió el nombre de la variable (eliminando la doble 't') y usando sufijos 'd'
train_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    RandRotated(keys=["image"], range_x=np.pi / 12, prob=0.5, keep_size=True),
    RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
    RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
])

val_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"])
])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])

# %% [markdown]
# ## Creación de DataSets y DataLoaders
batch_size = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
test_ds = monai.data.Dataset(data=test_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=10, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=10, shuffle=False)

# %% [markdown]
# ## Definición de la Arquitectura de la Red Neuronal
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
max_epochs = 4
val_interval = 1
auc_metric = ROCAUCMetric()

# %% [markdown]
# ## Configuración del Proceso de Entrenamiento
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        # CORRECCIÓN: Acceso por clave de diccionario en lugar de índice numérico
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                # CORRECCIÓN: Acceso por clave de diccionario en la validación
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                
            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
                
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
            writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
# %% [markdown]
# ## Graficación de Resulatdos AUC

