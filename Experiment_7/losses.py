import torch
import torch.nn as nn

from geomloss import SamplesLoss


# ==========================================================
# TANH LOSS
# ==========================================================

class TanhLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):

        error = torch.abs(
            prediction - target
        )

        return torch.mean(
            torch.tanh(error)
        )


# ==========================================================
# RANGE IMAGE LOSS
# ==========================================================

class RangeImageLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(
        self,
        prediction,
        target
    ):

        return self.l1(
            prediction,
            target
        )


# ==========================================================
# RANGE IMAGE -> POINT CLOUD
# ==========================================================

def range_image_to_points(
    range_image,
    max_range=120.0,
    fov_up=3.0,
    fov_down=-25.0
):

    B, _, H, W = range_image.shape

    device = range_image.device

    range_image = (
        range_image.squeeze(1)
        * max_range
    )

    yaw = torch.linspace(
        -torch.pi,
        torch.pi,
        W,
        device=device
    )

    pitch = torch.linspace(
        torch.deg2rad(
            torch.tensor(
                fov_down,
                device=device
            )
        ),
        torch.deg2rad(
            torch.tensor(
                fov_up,
                device=device
            )
        ),
        H,
        device=device
    )

    pitch, yaw = torch.meshgrid(
        pitch,
        yaw,
        indexing="ij"
    )

    yaw = yaw.unsqueeze(0)
    pitch = pitch.unsqueeze(0)

    x = (
        range_image
        * torch.cos(pitch)
        * torch.cos(yaw)
    )

    y = (
        range_image
        * torch.cos(pitch)
        * torch.sin(yaw)
    )

    z = (
        range_image
        * torch.sin(pitch)
    )

    points = torch.stack(
        [x, y, z],
        dim=-1
    )

    points = points.reshape(
        B,
        -1,
        3
    )

    return points


# ==========================================================
# OPTIMAL TRANSPORT LOSS
# ==========================================================

class SinkhornOTLoss(nn.Module):

    def __init__(
        self,
        blur=0.05
    ):

        super().__init__()

        self.ot = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=blur
        )

    def forward(
        self,
        pred_points,
        gt_points
    ):

        return self.ot(
            pred_points,
            gt_points
        )


# ==========================================================
# COMBINED LOSS
# ==========================================================

class CombinedLoss(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()

        self.tanh_weight = (
            config["loss"][
                "tanh_weight"
            ]
        )

        self.range_weight = (
            config["loss"][
                "range_weight"
            ]
        )

        self.sinkhorn_weight = (
            config["loss"][
                "sinkhorn_weight"
            ]
        )

        self.ot_points = (
            config["loss"].get(
                "ot_points",
                1024
            )
        )

        self.max_range = (
            config["sensor"][
                "max_range"
            ]
        )

        self.fov_up = (
            config["sensor"][
                "fov_up"
            ]
        )

        self.fov_down = (
            config["sensor"][
                "fov_down"
            ]
        )

        self.tanh_loss = (
            TanhLoss()
        )

        self.range_loss = (
            RangeImageLoss()
        )

        self.ot_loss = (
            SinkhornOTLoss(
                blur=config[
                    "loss"
                ][
                    "sinkhorn_blur"
                ]
            )
        )

    def forward(
        self,
        prediction,
        target
    ):

        # =====================================
        # TANH LOSS
        # =====================================

        tanh_loss = (
            self.tanh_loss(
                prediction,
                target
            )
        )

        # =====================================
        # RANGE LOSS
        # =====================================

        range_loss = (
            self.range_loss(
                prediction,
                target
            )
        )

        # =====================================
        # OT LOSS
        # =====================================

        if self.sinkhorn_weight > 0:

            pred_points = (
                range_image_to_points(
                    prediction,
                    self.max_range,
                    self.fov_up,
                    self.fov_down
                )
            )

            gt_points = (
                range_image_to_points(
                    target,
                    self.max_range,
                    self.fov_up,
                    self.fov_down
                )
            )

            total_points = (
                pred_points.shape[1]
            )

            sample_count = min(
                self.ot_points,
                total_points
            )

            idx = torch.randperm(
                total_points,
                device=pred_points.device
            )[:sample_count]

            pred_points = (
                pred_points[
                    :,
                    idx,
                    :
                ]
            )

            gt_points = (
                gt_points[
                    :,
                    idx,
                    :
                ]
            )

            ot_loss = (
                self.ot_loss(
                    pred_points,
                    gt_points
                )
            )

        else:

            ot_loss = torch.tensor(
                0.0,
                device=prediction.device
            )

        # =====================================
        # TOTAL LOSS
        # =====================================

        total_loss = (

            self.tanh_weight
            * tanh_loss

            +

            self.range_weight
            * range_loss

            +

            self.sinkhorn_weight
            * ot_loss

        )

        metrics = {

            "total_loss":
                float(
                    total_loss.detach()
                ),

            "tanh_loss":
                float(
                    tanh_loss.detach()
                ),

            "range_loss":
                float(
                    range_loss.detach()
                ),

            "sinkhorn_loss":
                float(
                    ot_loss.detach()
                )
        }

        return (
            total_loss,
            metrics
        )