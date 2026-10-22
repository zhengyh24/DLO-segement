"""Native-pixel supervised sequences, never a deployment input builder.

Windows and the first positive seed are selected with training annotations. Later
seed channels are zero: continuation must use the model's query/history. The
trajectory is teacher-selected, not a recovered physical or occluded cable path.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

DIRECTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DIRECTION_XY = ((0, -1), (1, -1), (1, 0), (1, 1),
                (0, 1), (-1, 1), (-1, 0), (-1, -1))


def read_manifest(path, allow_condition=False):
    """Validate frozen sample/group/RGB membership without changing the split."""
    path = Path(path).resolve()
    obj = json.loads(path.read_text(encoding="utf-8"))
    kind = obj.get("grouping_kind")
    if kind not in ("verified_scene", "condition_holdout_pilot"):
        raise ValueError("Expected verified_scene or condition_holdout_pilot")
    if kind == "condition_holdout_pilot":
        if not allow_condition:
            raise ValueError("Condition holdout is not independent: explicitly allow_condition")
        if obj.get("independent_scene_split") is not False:
            raise ValueError("Condition holdout must disclaim scene independence")
    groups, hashes, names, files = {}, {}, set(), set()
    for row in obj["records"]:
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Unsupported split name")
        if row["id"] in names or row["file_name"] in files:
            raise ValueError("Repeated sample ID or source filename")
        names.add(row["id"]); files.add(row["file_name"])
        for seen, key in ((groups, row["group_id"]), (hashes, row["rgb_sha256"])):
            if key in seen and seen[key] != row["split"]:
                raise ValueError("Frozen group or decoded RGB hash crosses splits")
            seen[key] = row["split"]
        for key in ("image", "masks"):
            rel = Path(row[key])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Record paths must be relative to data_root")
    return obj


def manifest_root(path, obj):
    root = Path(obj.get("data_root", "."))
    return root.resolve() if root.is_absolute() else (Path(path).resolve().parent / root).resolve()


def load_record(root, row):
    """Check decoded RGB SHA and native mask coordinates, including on copies."""
    root = Path(root)
    with Image.open(root / row["image"]) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError("EXIF must be upright; no implicit orientation/resizing")
        rgb = np.asarray(image.convert("RGB")).copy()
    if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise ValueError(f"RGB differs from frozen manifest: {row['file_name']}")
    with np.load(root / row["masks"], allow_pickle=False) as archive:
        masks = archive["masks"].astype(bool)
    if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
        raise ValueError("Instance masks must be [instances, native height, native width]")
    if "shape" in row and list(rgb.shape[:2]) != row["shape"]:
        raise ValueError("Native shape differs from frozen manifest")
    if row.get("masks_sha256"):
        digest = hashlib.sha256((root / row["masks"]).read_bytes()).hexdigest()
        if digest != row["masks_sha256"]:
            raise ValueError("Instance mask archive hash changed")
    return rgb, masks


def native_crop(array, origin, tile, fill=0):
    """Crop first two dimensions in original xy coordinates, padding only."""
    x, y = map(int, origin); height, width = array.shape[:2]
    out = np.full((tile, tile) + array.shape[2:], fill, dtype=array.dtype)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + tile), min(height, y + tile)
    if x1 > x0 and y1 > y0:
        out[y0-y:y1-y, x0-x:x1-x] = array[y0:y1, x0:x1]
    return out


def exit_targets(target, history, valid):
    """[B,1,H,W] -> [B,8], N/NE/E/SE/S/SW/W/NW.

    Positives are valid target pixels in the outer min(H,W)/8 band that have
    not been covered by history (> .5). This is visible support, not a hidden
    continuation or physical-end label. `history` may be predicted/teacher
    target history supplied by training; the dataset uses prior window coverage.
    """
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("exit_targets expects [B,1,H,W]")
    if history.shape != target.shape or valid.shape != target.shape:
        raise ValueError("target/history/valid shapes must agree")
    height, width = target.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=target.device),
                            torch.arange(width, device=target.device), indexing="ij")
    band = max(1, min(height, width) // 8)
    border = (yy < band) | (yy >= height-band) | (xx < band) | (xx >= width-band)
    angles = torch.atan2(xx.float() - (width-1)/2, (height-1)/2 - yy.float())
    bins = torch.remainder(torch.floor(angles / (torch.pi/4) + .5).long(), 8)
    support = (target[:, 0] > .5) & (history[:, 0] <= .5) & (valid[:, 0] > .5) & border
    return torch.stack([(support & (bins == k)).flatten(1).any(1)
                        for k in range(8)], dim=1).to(torch.float32)


def _ridge_heatmap(masks, overlap):
    """Approximate per-instance distance ridges, NOT a topological skeleton.

    Local distance maxima are blurred with sigma=1.5 pixels. Each instance is
    processed separately before a max union, so touching cables are not first
    collapsed into a union skeleton. Thick/variable-width shapes can yield sparse
    ridges; this is seed-discovery supervision, not a ground-truth centerline.
    """
    heat = np.zeros(overlap.shape, np.float32)
    for mask in masks:
        clean = mask & ~overlap
        yy, xx = np.where(clean)
        if not len(xx):
            continue
        x0, x1 = max(0, int(xx.min())-2), min(clean.shape[1], int(xx.max())+3)
        y0, y1 = max(0, int(yy.min())-2), min(clean.shape[0], int(yy.max())+3)
        local = clean[y0:y1, x0:x1]
        # Explicit background border handles objects touching the image boundary.
        distance = ndi.distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1]
        ridge = local & (distance >= ndi.maximum_filter(distance, size=3)-1e-6)
        blurred = ndi.gaussian_filter(ridge.astype(np.float32), sigma=1.5)
        if blurred.max() > 0:
            blurred /= blurred.max()
        blurred *= local
        np.maximum(heat[y0:y1, x0:x1], blurred, out=heat[y0:y1, x0:x1])
    heat[overlap] = 0
    return heat


class TrackerDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split="train", tile=256, steps=3, seed=0,
                 allow_condition=False, stride=None, negative_probability=None, seed_sigma=2.0):
        self.manifest = Path(manifest).resolve()
        self.protocol = read_manifest(self.manifest, allow_condition)
        self.root = manifest_root(self.manifest, self.protocol)
        self.split = "validation" if split == "val" else split
        if self.split not in ("train", "validation"):
            raise ValueError("TrackerDataset is for train/validation only; test is reserved")
        self.rows = [r for r in self.protocol["records"] if r["split"] == self.split]
        if not self.rows:
            raise ValueError(f"Empty split: {split}")
        self.tile, self.steps, self.seed, self.epoch = int(tile), int(steps), int(seed), 0
        self.stride = self.tile // 2 if stride is None else int(stride)
        self.seed_sigma = float(seed_sigma)
        if self.tile < 16 or self.steps < 1 or not 0 < self.stride < self.tile:
            raise ValueError("Require tile>=16, steps>=1, 0<stride<tile")
        if self.seed_sigma <= 0:
            raise ValueError("seed_sigma must be positive")
        self.negative_probability = (.15 if self.split == "train" else 0.) if negative_probability is None else float(negative_probability)
        if not 0 <= self.negative_probability <= 1:
            raise ValueError("negative_probability must be in [0,1]")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def _load(self, index):
        if index not in self._cache:
            rgb, masks = load_record(self.root, self.rows[index])
            overlap = masks.sum(axis=0) > 1
            heat = _ridge_heatmap(masks, overlap)
            self._cache[index] = rgb, masks, overlap, heat
            if len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]

    def __getitem__(self, index):
        index = int(index); row = self.rows[index]
        # Validation sequences remain fixed even if a training caller sets epoch.
        epoch = int(self.epoch) if self.split == "train" else 0
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
        rgb, masks, overlap, heat = self._load(index)
        height, width = rgb.shape[:2]; tile = self.tile
        union = masks.any(axis=0)
        eligible = np.flatnonzero((masks & ~overlap).reshape(len(masks), -1).any(1)) if len(masks) else np.array([], dtype=int)
        negative = rng.random() < self.negative_probability or not len(eligible)
        target = np.zeros((height, width), bool)
        component = target.copy(); instance = -1
        if negative:
            # A seed at least five pixels away from every annotated visible cable.
            background = (~union) & (ndi.distance_transform_edt(np.pad(~union, 1))[1:-1, 1:-1] >= 5)
            yy, xx = np.where(background)
            if not len(xx):
                negative = False
        if not negative:
            if not len(eligible):
                raise ValueError(f"No unambiguous foreground or safe background: {row['file_name']}")
            instance = int(rng.choice(eligible))
            target = masks[instance] & ~overlap
            preferred = target & (heat > .5)
            yy, xx = np.where(preferred if preferred.any() else target)
        choice = int(rng.integers(len(xx))); sx, sy = int(xx[choice]), int(yy[choice])
        if not negative:
            labels, _ = ndi.label(target, structure=np.ones((3, 3), bool))
            component = labels == labels[sy, sx]
        other = union & ~target & ~overlap
        def clamp_origin(x, y):
            return (min(max(int(x), 0), max(0, width-tile)),
                    min(max(int(y), 0), max(0, height-tile)))

        origin = clamp_origin(sx-tile//2, sy-tile//2)
        valid_full = ~overlap
        spatial_full = np.ones((height, width), bool)
        covered = np.zeros((height, width), bool)
        fields = {key: [] for key in ("rgb", "seed", "target", "other", "valid", "spatial_valid", "seed_target", "coverage")}
        origins, step_valid = [], []
        active = True
        grid_y, grid_x = np.mgrid[:tile, :tile]
        for step in range(self.steps):
            origins.append(origin); step_valid.append(float(active))
            fields["rgb"].append(native_crop(rgb, origin, tile))
            for key, array in (("target", target), ("other", other), ("valid", valid_full),
                               ("spatial_valid", spatial_full), ("seed_target", heat), ("coverage", covered)):
                fields[key].append(native_crop(array, origin, tile))
            prompt = np.zeros((tile, tile), np.float32)
            if step == 0:
                px, py = sx-origin[0], sy-origin[1]
                prompt = np.exp(-((grid_x-px)**2 + (grid_y-py)**2)/(2*self.seed_sigma**2)).astype(np.float32)
                # Prompt shape is spatial-only, as in inference. GT-overlap
                # ignore labels must not become a hidden input channel.
                prompt *= fields["spatial_valid"][-1]
            fields["seed"].append(prompt)
            if step == self.steps-1:
                break
            # The next observed window is selected from visible support only.
            # No interpolation, GT hidden path, or transition between disconnected
            # visible components is constructed.
            current = native_crop(component & ~covered, origin, tile)
            band = max(1, tile//8)
            border = (grid_x < band) | (grid_x >= tile-band) | (grid_y < band) | (grid_y >= tile-band)
            bins = np.floor(np.arctan2(grid_x-(tile-1)/2, (tile-1)/2-grid_y)/(np.pi/4)+.5).astype(int) % 8
            x, y = origin
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x+tile, width), min(y+tile, height)
            covered[y0:y1, x0:x1] = True
            candidates = []
            if active and not negative:
                for direction, (dx, dy) in enumerate(DIRECTION_XY):
                    if ((dy < 0 and y == 0) or (dx < 0 and x == 0) or
                            (dy > 0 and y+tile >= height) or (dx > 0 and x+tile >= width)):
                        continue
                    weight = int((current & border & (bins == direction)).sum())
                    if not weight:
                        continue
                    candidate = clamp_origin(x+dx*self.stride, y+dy*self.stride)
                    if candidate in origins:
                        continue
                    future = native_crop(component, candidate, tile)
                    past = native_crop(covered, candidate, tile)
                    if not (future & ~past).any():
                        continue
                    # Require shared visible support in the CURRENT window, not
                    # merely support in an old window elsewhere in the sequence.
                    ax, ay = max(x, candidate[0], 0), max(y, candidate[1], 0)
                    bx, by = min(x+tile, candidate[0]+tile, width), min(y+tile, candidate[1]+tile, height)
                    if bx <= ax or by <= ay or not component[ay:by, ax:bx].any():
                        continue
                    candidates.append((candidate, weight))
            if candidates:
                weights = np.asarray([c[1] for c in candidates], np.float64)
                origin = candidates[int(rng.choice(len(candidates), p=weights/weights.sum()))][0]
            else:
                active = False  # Repeat last window for batching, ignore its loss.
        output = {}
        for key, values in fields.items():
            array = np.stack(values)
            tensor = torch.from_numpy(array.copy())
            output[key] = tensor.permute(0, 3, 1, 2).float()/255 if key == "rgb" else tensor[:, None].float()
        output["exits"] = exit_targets(output["target"], output.pop("coverage"), output["valid"])
        output["step_valid"] = torch.tensor(step_valid, dtype=torch.float32)
        output["exits"] *= output["step_valid"][:, None]
        output["origins"] = torch.tensor(origins, dtype=torch.int64)
        output.update(sample_id=row["id"], file_name=row["file_name"], group_id=row["group_id"],
                      instance_index=instance, negative_seed=bool(negative),
                      seed_xy=torch.tensor([sx, sy]), native_size=torch.tensor([height, width]),
                      teacher_trajectory=True, continuation_seed_is_zero=True,
                      independent_scene_split=bool(self.protocol.get("independent_scene_split", False)),
                      seed_target_kind="per_instance_distance_ridge_approximation")
        return output
