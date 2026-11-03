"""Set matching + visible-mask loss + supervised soft-route/descriptor losses.

All GT use is isolated here. Matching/targets are never forwarded to the model.
Geometry targets are local visible-mask moments, not physical endpoints,
hidden cable paths, or reliable branch/occlusion state annotations.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def _resize_masks(masks, size):
    if masks.shape[0] == 0:
        return masks.new_zeros((0, *size))
    return F.interpolate(masks[:, None].float(), size=size, mode="area")[:, 0]


def _visible_geometry(masks):
    """Per-instance local covariance, then visible-occupancy weighted tangent.

    Return foreground and double-angle tangent plus continuous anisotropy
    reliability. Ambiguous overlap is ignored by the caller's validity mask.
    The 7-cell neighborhood defines an auxiliary scale, not an inference rule.
    """
    n, h, w = masks.shape
    if n == 0:
        zero = masks.new_zeros(h, w)
        return zero, torch.stack((zero, zero)), zero
    y, x = torch.meshgrid(torch.arange(-3, 4, device=masks.device, dtype=masks.dtype),
                          torch.arange(-3, 4, device=masks.device, dtype=masks.dtype), indexing="ij")
    kernels = torch.stack((torch.ones_like(x), x, y, x*x, y*y, x*y))[:, None] / 49
    moments = F.conv2d(masks[:, None], kernels, padding=3)
    mass = moments[:, :1].clamp_min(1e-6)
    ex, ey, ex2, ey2, exy = (moments[:, i]/mass[:, 0] for i in range(1, 6))
    vx, vy, covariance = (ex2-ex.square()).clamp_min(0), (ey2-ey.square()).clamp_min(0), exy-ex*ey
    vector = torch.stack((vx-vy, 2*covariance), 1)
    magnitude = vector.square().sum(1).sqrt()
    reliability = (magnitude / (vx+vy+1e-6)).clamp(0, 1)
    tangent = vector / magnitude[:, None].clamp_min(1e-6)
    weights = masks * reliability
    support = masks.sum(0).clamp(0, 1)
    tangent = (tangent * weights[:, None]).sum(0) / weights.sum(0)[None].clamp_min(1e-6)
    reliable = weights.sum(0).clamp(0, 1)
    return support, tangent, reliable


class SetCriterion(nn.Module):
    def __init__(self, mask_bce=2., mask_dice=2., object_weight=1.,
                 path_weight=.2, descriptor_weight=.1, geometry_weight=.1,
                 quality_weight=.1, no_object_weight=.1, empty_mask_weight=.1,
                 descriptor_temperature=.1):
        super().__init__()
        self.weights = dict(mask_bce=mask_bce, mask_dice=mask_dice,
                            object=object_weight, path=path_weight, descriptor=descriptor_weight,
                            geometry=geometry_weight, quality=quality_weight)
        self.no_object_weight, self.empty_mask_weight = no_object_weight, empty_mask_weight
        self.descriptor_temperature = descriptor_temperature
        all_weights = [*self.weights.values(), no_object_weight, empty_mask_weight, descriptor_temperature]
        if (not all(math.isfinite(v) for v in all_weights) or any(v < 0 for v in self.weights.values())
                or no_object_weight <= 0 or descriptor_temperature <= 0 or empty_mask_weight < 0):
            raise ValueError("Loss weights must be nonnegative and temperatures/no-object weight positive")

    @torch.no_grad()
    def match(self, outputs, targets, valid):
        logits = outputs["mask_logits"].float()
        size = logits.shape[-2:]
        assignments = []
        for b, target in enumerate(targets):
            masks = target["masks"].to(logits.device).float()
            if masks.ndim != 3 or masks.shape[-2:] != valid.shape[-2:]:
                raise ValueError("GT masks must be native pixels padded to model canvas")
            n, q = len(masks), logits.shape[1]
            if n > q:
                raise ValueError(f"Image {target.get('id', b)} has {n} visible instances but only {q} queries; increase num_queries, never truncate GT")
            if n == 0:
                empty = torch.empty(0, device=logits.device, dtype=torch.long)
                assignments.append((empty, empty)); continue
            # Area preserves subcell thin-wire occupancy instead of dropping it.
            weight = F.interpolate(valid[b:b+1].float(), size=size, mode="area").flatten()
            # Conditional occupancy within the valid subpixels of each cell;
            # invalid GT must not affect matching even in partly padded cells.
            target_small = _resize_masks(masks*valid[b], size).flatten(1) / weight.clamp_min(1e-8)
            probability = logits[b].sigmoid().flatten(1)
            intersection = (probability*weight) @ target_small.T
            denom = (probability*weight).sum(1)[:, None] + (target_small*weight).sum(1)[None]
            dice = 1 - (2*intersection+1) / (denom+1)
            # pairwise stable BCE without a Q*N*H*W intermediate
            flat_logits = logits[b].flatten(1)
            bce = (F.softplus(flat_logits)*weight).sum(1)[:, None] - (flat_logits*weight) @ target_small.T
            bce = bce / weight.sum().clamp_min(1)
            obj_cost = -outputs["object_logits"][b].float().sigmoid()[:, None]
            cost = self.weights["mask_bce"]*bce + self.weights["mask_dice"]*dice + obj_cost
            if not torch.isfinite(cost).all():
                raise FloatingPointError("Nonfinite Hungarian matching cost")
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            assignments.append((torch.as_tensor(rows, device=logits.device),
                                torch.as_tensor(cols, device=logits.device)))
        return assignments

    def forward(self, outputs, targets, valid):
        low_logits = outputs["mask_logits"].float()
        if len(targets) != len(low_logits):
            raise ValueError("Batch/target lengths differ")
        valid = valid.to(low_logits.device).float()
        if valid.ndim != 4 or valid.shape[1] != 1:
            raise ValueError("Loss validity must be [B,1,H,W]")
        if (valid.flatten(1).sum(1) == 0).any():
            raise ValueError("Loss has no valid pixels")
        logits = F.interpolate(low_logits, size=valid.shape[-2:], mode="bilinear", align_corners=False)
        assignments = self.match(outputs, targets, valid)
        zero = low_logits.sum()*0
        values = {name: zero for name in self.weights}
        aux = outputs.get("aux", {})
        presence_target = torch.zeros_like(outputs["object_logits"])
        qualities = torch.zeros_like(outputs["quality_logits"])
        for b, (rows, cols) in enumerate(assignments):
            masks = targets[b]["masks"].to(logits.device).float()
            if masks.shape[-2:] != valid.shape[-2:]:
                raise ValueError("GT masks must be native pixels padded to model canvas")
            matched_target = torch.zeros_like(logits[b])
            matched_target[rows] = masks[cols]
            presence_target[b, rows] = 1
            per_query_weight = torch.full_like(presence_target[b], self.empty_mask_weight)
            per_query_weight[rows] = 1
            bce = (F.binary_cross_entropy_with_logits(logits[b], matched_target, reduction="none")
                   * valid[b]).flatten(1).sum(1) / valid[b].sum().clamp_min(1)
            values["mask_bce"] = values["mask_bce"] + (bce*per_query_weight).sum()/per_query_weight.sum().clamp_min(1)
            probabilities = logits[b].sigmoid()
            if len(rows):
                inter = (probabilities[rows]*masks[cols]*valid[b]).flatten(1).sum(1)
                denom = ((probabilities[rows]+masks[cols])*valid[b]).flatten(1).sum(1)
                values["mask_dice"] = values["mask_dice"] + (1-(2*inter+1)/(denom+1)).mean()
                with torch.no_grad():
                    pred_binary = (logits[b, rows] > 0).float()
                    inter_binary = (pred_binary*masks[cols]*valid[b]).flatten(1).sum(1)
                    union_binary = ((pred_binary+masks[cols]).clamp_max(1)*valid[b]).flatten(1).sum(1)
                    qualities[b, rows] = inter_binary / union_binary.clamp_min(1)
            if "path_attention" in aux and len(rows):
                attention = aux["path_attention"][b, rows]
                shape = attention.shape[-2:]
                target_small = _resize_masks(masks[cols]*valid[b], shape)
                density = target_small / target_small.sum((-2,-1), keepdim=True).clamp_min(1e-8)
                # Match a visible support distribution, without inventing order or
                # hidden continuation labels. Recurrence/motion are learned jointly.
                coverage = attention.mean(1).clamp_min(1e-8)
                values["path"] = values["path"] - (density*coverage.log()).sum((-2,-1)).mean()
            if "descriptor" in aux:
                embedding = aux["descriptor"][b]
                shape = embedding.shape[-2:]
                reduced = _resize_masks(masks*valid[b], shape)
                v = F.interpolate(valid[b:b+1], shape, mode="area")[0, 0]
                if len(masks):
                    flat = F.normalize(embedding.flatten(1), dim=0)
                    occupancy = reduced.flatten(1)
                    prototypes = F.normalize(occupancy @ flat.T / occupancy.sum(1, keepdim=True).clamp_min(1e-8), dim=1)
                    similarity = prototypes @ flat
                    distribution = occupancy / occupancy.sum(0, keepdim=True).clamp_min(1e-8)
                    pixel_weight = occupancy.sum(0).clamp_max(1) * v.flatten()
                    contrast = -(distribution * F.log_softmax(similarity/self.descriptor_temperature, dim=0)).sum(0)
                    compact = (distribution * (1-similarity)).sum(0)
                    values["descriptor"] = values["descriptor"] + ((contrast+compact)*pixel_weight).sum()/pixel_weight.sum().clamp_min(1e-8)
                foreground, tangent, reliability = _visible_geometry(reduced)
                geometry = aux["geometry_logits"][b]
                support_loss = (F.binary_cross_entropy_with_logits(geometry[0], foreground, reduction="none")*v).sum()/v.sum().clamp_min(1)
                orientation = F.normalize(geometry[1:], dim=0)
                tangent_loss = ((1-(orientation*tangent).sum(0)) * reliability * v).sum() / (reliability*v).sum().clamp_min(1e-8)
                values["geometry"] = values["geometry"] + support_loss+tangent_loss
        batch_size = len(targets)
        for name in ("mask_bce", "mask_dice", "path", "descriptor", "geometry"):
            values[name] = values[name]/batch_size
        object_weights = torch.where(presence_target > 0, 1., self.no_object_weight)
        object_loss = F.binary_cross_entropy_with_logits(outputs["object_logits"], presence_target, reduction="none")
        if "sam_object_logits" in aux:
            object_loss = .5*(object_loss + F.binary_cross_entropy_with_logits(aux["sam_object_logits"], presence_target, reduction="none"))
        values["object"] = (object_loss*object_weights).sum()/object_weights.sum().clamp_min(1)
        values["quality"] = F.mse_loss(outputs["quality_logits"].sigmoid(), qualities)
        values["total"] = sum(self.weights[name]*values[name] for name in self.weights)
        return values
