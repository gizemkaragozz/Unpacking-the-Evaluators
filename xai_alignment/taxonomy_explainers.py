"""Notebook-derived attribution methods used for output-behavior characterization."""

from __future__ import annotations

import math

import numpy as np

TAXONOMY_METHODS = (
    "deeplift",
    "eigencam",
    "feature_ablation",
    "gradcam",
    "gradcampp",
    "guidedbp",
    "guided_gradcam",
    "input_x_gradient",
    "integrated_gradients",
    "lrp",
    "layercam",
    "occlusion",
    "saliency",
    "scorecam",
    "smoothgrad",
)


def _imports():
    try:
        import torch
        import torch.nn.functional as functional
        from captum.attr import (
            DeepLift,
            FeatureAblation,
            GuidedBackprop,
            GuidedGradCam,
            InputXGradient,
            IntegratedGradients,
            LRP,
            LayerGradCam,
            Occlusion,
            Saliency,
        )
    except ImportError as error:  # pragma: no cover
        raise ImportError("taxonomy explanations require `pip install -e .[xai]`") from error
    return {
        "torch": torch,
        "functional": functional,
        "DeepLift": DeepLift,
        "FeatureAblation": FeatureAblation,
        "GuidedBackprop": GuidedBackprop,
        "GuidedGradCam": GuidedGradCam,
        "InputXGradient": InputXGradient,
        "IntegratedGradients": IntegratedGradients,
        "LRP": LRP,
        "LayerGradCam": LayerGradCam,
        "Occlusion": Occlusion,
        "Saliency": Saliency,
    }


def _normalize(spatial):
    spatial = spatial - spatial.min()
    maximum = spatial.max()
    return spatial / maximum if maximum > 0 else spatial


def _spatial(attribution, *, reduce: str = "sum_abs"):
    if reduce == "sum_abs":
        result = attribution.detach().abs().sum(dim=1).squeeze(0)
    elif reduce == "sum":
        result = attribution.detach().sum(dim=1).squeeze(0)
    else:
        raise ValueError("reduce must be 'sum_abs' or 'sum'")
    return _normalize(result)


def _interpolate(attribution, inputs, functional):
    mode = "bilinear" if inputs.ndim == 4 else "trilinear"
    return functional.interpolate(
        attribution,
        size=inputs.shape[2:],
        mode=mode,
        align_corners=False,
    )


def _grid_feature_mask(inputs, patch_size, torch):
    spatial = inputs.shape[2:]
    if isinstance(patch_size, int):
        patch_size = (patch_size,) * len(spatial)
    blocks = [math.ceil(size / step) for size, step in zip(spatial, patch_size)]
    mask = torch.zeros((1, 1, *spatial), dtype=torch.long, device=inputs.device)
    for coordinate in np.ndindex(*blocks):
        slices = tuple(
            slice(block * step, min((block + 1) * step, size))
            for block, step, size in zip(coordinate, patch_size, spatial)
        )
        flat_index = np.ravel_multi_index(coordinate, blocks)
        mask[(0, 0, *slices)] = int(flat_index)
    return mask


class _ActivationGradientHook:
    def __init__(self, layer):
        self.layer = layer
        self.activation = None
        self.gradient = None
        self.handles = []

    def __enter__(self):
        self.handles.append(self.layer.register_forward_hook(
            lambda _module, _inputs, output: setattr(self, "activation", output)
        ))
        self.handles.append(self.layer.register_full_backward_hook(
            lambda _module, _grad_input, grad_output:
            setattr(self, "gradient", grad_output[0])
        ))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()


def _gradcam(model, sample, target, layer, modules, *, reduce="sum_abs"):
    attribution = modules["LayerGradCam"](model, layer).attribute(sample, target=target)
    attribution = _interpolate(attribution, sample, modules["functional"])
    return _spatial(attribution, reduce=reduce)


def _guided_backprop(model, sample, target, modules):
    attribution = modules["GuidedBackprop"](model).attribute(sample, target=target)
    return _spatial(attribution)


def _gradcampp(model, sample, target, layer, modules):
    functional = modules["functional"]
    model.zero_grad(set_to_none=True)
    with _ActivationGradientHook(layer) as hook:
        score = model(sample)[:, target].sum()
        score.backward(retain_graph=True)
        activation, gradient = hook.activation, hook.gradient
        second, third = gradient.square(), gradient.pow(3)
        spatial_axes = tuple(range(2, activation.ndim))
        denominator = 2.0 * second + (
            activation * third
        ).sum(dim=spatial_axes, keepdim=True)
        alpha = second / (denominator + 1e-12)
        weights = (alpha * functional.relu(gradient)).sum(
            dim=spatial_axes, keepdim=True
        )
        cam = functional.relu((weights * activation).sum(dim=1, keepdim=True))
    return _spatial(_interpolate(cam, sample, functional), reduce="sum")


def _layercam(model, sample, target, layer, modules):
    functional = modules["functional"]
    model.zero_grad(set_to_none=True)
    with _ActivationGradientHook(layer) as hook:
        model(sample)[:, target].sum().backward(retain_graph=True)
        cam = functional.relu(
            (functional.relu(hook.gradient) * hook.activation).sum(
                dim=1, keepdim=True
            )
        )
    return _spatial(_interpolate(cam, sample, functional), reduce="sum")


def _eigencam(model, sample, layer, modules):
    functional = modules["functional"]
    with _ActivationGradientHook(layer) as hook:
        model(sample)
        activation = hook.activation.detach()[0]
    channels = activation.shape[0]
    matrix = activation.reshape(channels, -1).cpu().float()
    left, _, _ = modules["torch"].linalg.svd(matrix, full_matrices=False)
    projection = (left[:, :1].T @ matrix).reshape(activation.shape[1:])
    cam = functional.relu(projection)[None, None].to(sample.device)
    return _spatial(_interpolate(cam, sample, functional), reduce="sum")


def _scorecam(model, sample, target, layer, modules, *, channels, batch_size):
    torch, functional = modules["torch"], modules["functional"]
    with _ActivationGradientHook(layer) as hook:
        model(sample)
        activation = hook.activation.detach()
    spatial_axes = tuple(range(2, activation.ndim))
    energy = activation.abs().sum(dim=spatial_axes)[0]
    chosen = torch.topk(energy, k=min(channels, len(energy))).indices
    masks = _interpolate(activation[:, chosen], sample, functional)
    masks = masks - masks.amin(dim=spatial_axes, keepdim=True)
    masks = masks / (masks.amax(dim=spatial_axes, keepdim=True) + 1e-12)
    count = masks.shape[1]
    expanded = sample.repeat(count, *([1] * (sample.ndim - 1)))
    masked = expanded * masks[0].unsqueeze(1)
    scores = []
    for start in range(0, count, batch_size):
        with torch.no_grad():
            scores.append(model(masked[start:start + batch_size])[:, target])
    weights = functional.relu(torch.cat(scores))
    weights = weights / (weights.sum() + 1e-12)
    reshape = (count,) + (1,) * (sample.ndim - 2)
    cam = functional.relu((weights.reshape(reshape) * masks[0]).sum(dim=0))
    return _normalize(cam)


def explain_taxonomy(
    model,
    inputs,
    target: int,
    method: str,
    *,
    target_layer=None,
    integrated_gradients_steps: int | None = None,
    smoothgrad_samples: int | None = None,
    smoothgrad_noise: float = 0.10,
    feature_patch_size=None,
    occlusion_window=None,
    occlusion_stride=None,
    scorecam_channels: int | None = None,
    scorecam_batch_size: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Return one normalized spatial map using the submitted method settings."""
    modules = _imports()
    torch = modules["torch"]
    if inputs.shape[0] != 1 or inputs.ndim not in (4, 5):
        raise ValueError("inputs must contain one 2D or 3D sample")
    model.eval()
    sample = inputs.detach().clone().requires_grad_(True)
    name = method.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "deep_lift": "deeplift",
        "eigen_cam": "eigencam",
        "guided_bp": "guidedbp",
        "guided_gradcam": "guided_gradcam",
        "inputxgradient": "input_x_gradient",
        "integratedgradients": "integrated_gradients",
        "layer_cam": "layercam",
        "score_cam": "scorecam",
        "smooth_grad": "smoothgrad",
    }
    name = aliases.get(name, name)
    if name not in TAXONOMY_METHODS:
        raise ValueError(f"unsupported taxonomy method: {method}")
    is_3d = sample.ndim == 5
    ig_steps = integrated_gradients_steps or (32 if is_3d else 64)
    smooth_samples = smoothgrad_samples or (16 if is_3d else 25)
    patch = feature_patch_size or ((16, 16, 8) if is_3d else (16, 16))
    window = occlusion_window or ((1, 16, 16, 16) if is_3d else (1, 16, 16))
    stride = occlusion_stride or ((1, 8, 8, 8) if is_3d else (1, 8, 8))
    cam_channels = scorecam_channels or (16 if is_3d else 32)
    cam_batch = scorecam_batch_size or (8 if is_3d else 16)

    if name in {"gradcam", "guided_gradcam", "gradcampp", "layercam", "eigencam",
                "scorecam"} and target_layer is None:
        raise ValueError(f"target_layer is required for {name}")
    if name == "gradcam":
        result = _gradcam(model, sample, target, target_layer, modules)
    elif name == "guided_gradcam":
        if is_3d:
            cam_attribution = modules["LayerGradCam"](
                model,
                target_layer,
            ).attribute(sample, target=target)
            cam_attribution = _interpolate(
                cam_attribution,
                sample,
                modules["functional"],
            )
            cam = cam_attribution.detach().sum(dim=1).squeeze(0)
            guided_attribution = modules["GuidedBackprop"](model).attribute(
                sample,
                target=target,
            )
            guided = guided_attribution.detach().abs().sum(dim=1).squeeze(0)
            result = _normalize(cam * guided)
        else:
            result = _spatial(
                modules["GuidedGradCam"](model, target_layer).attribute(
                    sample,
                    target=target,
                )
            )
    elif name == "gradcampp":
        result = _gradcampp(model, sample, target, target_layer, modules)
    elif name == "layercam":
        result = _layercam(model, sample, target, target_layer, modules)
    elif name == "eigencam":
        result = _eigencam(model, sample, target_layer, modules)
    elif name == "scorecam":
        result = _scorecam(
            model, sample, target, target_layer, modules,
            channels=cam_channels, batch_size=cam_batch,
        )
    elif name == "guidedbp":
        result = _guided_backprop(model, sample, target, modules)
    elif name == "saliency":
        result = _spatial(
            modules["Saliency"](model).attribute(sample, target=target, abs=False)
        )
    elif name == "input_x_gradient":
        result = _spatial(
            modules["InputXGradient"](model).attribute(sample, target=target)
        )
    elif name == "integrated_gradients":
        result = _spatial(modules["IntegratedGradients"](model).attribute(
            sample, baselines=torch.zeros_like(sample), target=target, n_steps=ig_steps
        ))
    elif name == "deeplift":
        result = _spatial(modules["DeepLift"](model).attribute(
            sample, baselines=torch.zeros_like(sample), target=target
        ))
    elif name == "lrp":
        result = _spatial(modules["LRP"](model).attribute(sample, target=target))
    elif name == "feature_ablation":
        result = _spatial(modules["FeatureAblation"](model).attribute(
            sample,
            target=target,
            feature_mask=_grid_feature_mask(sample, patch, torch),
        ))
    elif name == "occlusion":
        result = _spatial(modules["Occlusion"](model).attribute(
            sample,
            target=target,
            sliding_window_shapes=window,
            strides=stride,
            baselines=0.0,
        ))
    else:
        torch.manual_seed(seed)
        gradients = []
        for _ in range(smooth_samples):
            noisy = torch.clamp(
                sample + smoothgrad_noise * torch.randn_like(sample), 0.0, 1.0
            ).requires_grad_(True)
            gradients.append(torch.autograd.grad(
                model(noisy)[:, target].sum(), noisy
            )[0].detach())
        result = _spatial(torch.stack(gradients).mean(dim=0))
    return result.detach().cpu().numpy()
