import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.utils.data import DataLoader

from project.Experiment_7.dataset import build_datasets
from project.Experiment_7.model import build_model
from project.Experiment_7.losses import CombinedLoss


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


seed_everything()


# ==========================================================
# CONFIG
# ==========================================================

with open("config.yaml", "r") as f:

    config = yaml.safe_load(f)


# ==========================================================
# OUTPUT DIRS
# ==========================================================

MODEL_DIR = config["paths"]["models"]

RESULTS_DIR = config["paths"]["results"]

os.makedirs(MODEL_DIR, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing Device:", DEVICE)


# ==========================================================
# DATASETS
# ==========================================================

print("\nBuilding datasets...")

train_dataset, val_dataset = build_datasets(
    config
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=config["training"]["num_workers"],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["training"]["num_workers"],
    pin_memory=True
)

print(
    f"\nTrain Samples: {len(train_dataset)}"
)

print(
    f"Validation Samples: {len(val_dataset)}"
)


# ==========================================================
# MODEL
# ==========================================================

print("\nBuilding model...")

model = build_model(config)

model = model.to(DEVICE)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nParameters: {total_params:,}"
)


# ==========================================================
# LOSS
# ==========================================================

criterion = CombinedLoss(
    config
)


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=config["training"][
        "learning_rate"
    ],

    weight_decay=config["training"][
        "weight_decay"
    ]
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3
    )
)


# ==========================================================
# TRAINING VARIABLES
# ==========================================================

EPOCHS = config["training"]["epochs"]

PATIENCE = config["training"]["patience"]

MIN_DELTA = config["training"]["min_delta"]

best_val_loss = float("inf")

early_stop_counter = 0

train_losses = []

val_losses = []

best_epoch = 0


# ==========================================================
# TRAIN
# ==========================================================

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    # ======================================================
    # TRAINING
    # ======================================================

    model.train()

    running_train_loss = 0.0

    train_tanh = 0.0

    train_range = 0.0

    train_sinkhorn = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for rgb, target_range in train_bar:

        rgb = rgb.to(DEVICE)

        target_range = target_range.to(
            DEVICE
        )

        optimizer.zero_grad()

        prediction = model(rgb)

        loss, metrics = criterion(
            prediction,
            target_range
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_train_loss += (
            metrics["total_loss"]
        )

        train_tanh += (
            metrics["tanh_loss"]
        )

        train_range += (
            metrics["range_loss"]
        )

        train_sinkhorn += (
            metrics["sinkhorn_loss"]
        )

        train_bar.set_postfix(
            loss=f"{metrics['total_loss']:.4f}"
        )

    avg_train_loss = (

        running_train_loss
        / len(train_loader)

    )

    # ======================================================
    # VALIDATION
    # ======================================================

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for rgb, target_range in val_loader:

            rgb = rgb.to(DEVICE)

            target_range = (
                target_range.to(
                    DEVICE
                )
            )

            prediction = model(rgb)

            loss, metrics = criterion(
                prediction,
                target_range
            )

            running_val_loss += (
                metrics["total_loss"]
            )

    avg_val_loss = (

        running_val_loss
        / len(val_loader)

    )

    train_losses.append(
        avg_train_loss
    )

    val_losses.append(
        avg_val_loss
    )

    scheduler.step(
        avg_val_loss
    )

    # ======================================================
    # LOGGING
    # ======================================================

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {avg_train_loss:.6f}"
    )

    print(
        f"Val Loss   : {avg_val_loss:.6f}"
    )

    print(
        f"Learning Rate : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    # ======================================================
    # BEST MODEL
    # ======================================================

    if avg_val_loss < (
        best_val_loss - MIN_DELTA
    ):

        best_val_loss = avg_val_loss

        best_epoch = epoch + 1

        early_stop_counter = 0

        best_model_path = os.path.join(
            MODEL_DIR,
            "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_loss":
                    avg_val_loss
            },
            best_model_path
        )

        print(
            "\nBest Model Saved"
        )

    else:

        early_stop_counter += 1

        print(
            f"\nNo Improvement "
            f"({early_stop_counter}/{PATIENCE})"
        )

    # ======================================================
    # EARLY STOPPING
    # ======================================================

    if early_stop_counter >= PATIENCE:

        print(
            "\nEarly Stopping Triggered"
        )

        break


# ==========================================================
# FINAL MODEL
# ==========================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_model.pth"
)

torch.save(
    {
        "epoch": len(train_losses),
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict()
    },
    final_model_path
)

print(
    "\nFinal Model Saved"
)


# ==========================================================
# LOSS CURVE
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training Curve"
)

plt.legend()

plt.grid(True)

loss_curve_path = os.path.join(
    RESULTS_DIR,
    "loss_curve.png"
)

plt.savefig(
    loss_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nLoss Curve Saved"
)


# ==========================================================
# METRICS
# ==========================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write(
        "OPTION B TRAINING REPORT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Best Validation Loss: "
        f"{best_val_loss:.6f}\n"
    )

    f.write(
        f"Best Epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Epochs Completed: "
        f"{len(train_losses)}\n"
    )

    f.write(
        f"Batch Size: "
        f"{config['training']['batch_size']}\n"
    )

    f.write(
        f"Learning Rate: "
        f"{config['training']['learning_rate']}\n"
    )

    f.write(
        f"Transformer Dim: "
        f"{config['model']['transformer_dim']}\n"
    )

    f.write(
        f"Transformer Layers: "
        f"{config['model']['transformer_layers']}\n"
    )

    f.write(
        f"Transformer Heads: "
        f"{config['model']['transformer_heads']}\n"
    )

print(
    "\nMetrics Saved"
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n================================")

print("TRAINING COMPLETE")

print("================================")

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Best Epoch: "
    f"{best_epoch}"
)

print(
    f"Model Saved To: "
    f"{MODEL_DIR}"
)

print(
    f"Results Saved To: "
    f"{RESULTS_DIR}"
)