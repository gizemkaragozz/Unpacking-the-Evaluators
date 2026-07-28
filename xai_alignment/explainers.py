"""Generation of the four attribution methods used in the study."""

from __future__ import annotations


def explain(
    model,
    inputs,
    target: int,
    method: str,
    *,
    target_layer=None,
    integrated_gradients_steps: int = 64,
):
    """Return one nonnegative spatial attribution map for a single-item batch."""
    try:
        import torch
        import torch.nn.functional as functional
        from captum.attr import (
            GuidedBackprop,
            GuidedGradCam,
            IntegratedGradients,
            LayerGradCam,
            Saliency,
        )
    except ImportError as error:  # pragma: no cover
        raise ImportError("explanation generation requires `pip install -e .[xai]`") from error

    if inputs.shape[0] != 1:
        raise ValueError("explain expects a batch containing exactly one sample")
    model.eval()
    sample = inputs.detach().clone().requires_grad_(True)
    name = method.upper()

    if name == "SL":
        attribution = Saliency(model).attribute(sample, target=target, abs=False)
    elif name == "IG":
        attribution = IntegratedGradients(model).attribute(
            sample,
            baselines=torch.zeros_like(sample),
            target=target,
            n_steps=integrated_gradients_steps,
        )
    elif name in {"GC", "GGC"}:
        if target_layer is None:
            raise ValueError(f"target_layer is required for {name}")
        if name == "GGC" and sample.ndim == 4:
            attribution = GuidedGradCam(model, target_layer).attribute(
                sample,
                target=target,
            )
            spatial = attribution.detach().abs().sum(dim=1).squeeze(0)
            spatial = spatial - spatial.min()
            maximum = spatial.max()
            if maximum > 0:
                spatial = spatial / maximum
            return spatial.cpu().numpy()

        attribution = LayerGradCam(model, target_layer).attribute(sample, target=target)
        mode = "bilinear" if sample.ndim == 4 else "trilinear"
        attribution = functional.interpolate(
            attribution,
            size=sample.shape[2:],
            mode=mode,
            align_corners=False,
        )
        cam = attribution.detach().abs().sum(dim=1).squeeze(0)
        cam = cam - cam.min()
        cam_maximum = cam.max()
        if cam_maximum > 0:
            cam = cam / cam_maximum
        if name == "GC":
            return cam.cpu().numpy()

        guided = GuidedBackprop(model).attribute(sample, target=target)
        guided = guided.detach().abs().sum(dim=1).squeeze(0)
        signed_cam = attribution.detach().sum(dim=1).squeeze(0)
        spatial = signed_cam * guided
        spatial = spatial - spatial.min()
        maximum = spatial.max()
        if maximum > 0:
            spatial = spatial / maximum
        return spatial.cpu().numpy()
    else:
        raise ValueError("method must be one of GC, GGC, IG, or SL")

    spatial = attribution.detach().abs().sum(dim=1).squeeze(0)
    spatial = spatial - spatial.min()
    maximum = spatial.max()
    if maximum > 0:
        spatial = spatial / maximum
    return spatial.cpu().numpy()
