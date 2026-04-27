"""
Structural anomaly detection pipeline using PaDiM (Anomalib).

Workflow:
1. Set template image (perfect product)
2. Compute ROI mask (automatic Otsu + morphological cleanup)
3. Align each frame to template (ECC)
4. Normalize lighting (CLAHE on LAB L channel)
5. Augment template -> synthetic normal dataset
6. Train PaDiM on augmented patches
7. Inference: align -> crop -> normalize -> predict -> heatmap overlay
"""

import os
import shutil

# Hide CUDA to force CPU mode — driver (535.x / CUDA 12.2) too old for PyTorch 2.11 (needs CUDA 12.4+).
# Remove this line after upgrading driver to 550+ or installing PyTorch built for CUDA 12.2.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torchvision import transforms

import cv2
import numpy as np
import albumentations as A

from anomalib.models import Padim
from anomalib.data import Folder
from anomalib.engine import Engine


class DefectDetector:
    """PaDiM-based structural anomaly detection pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.template_bgr: np.ndarray | None = None
        self.template_gray: np.ndarray | None = None
        self.roi_mask: np.ndarray | None = None
        self.roi_bbox: tuple[int, int, int, int] | None = None
        self.model: Padim | None = None
        self.engine: Engine | None = None
        self.trained: bool = False

        self._app_dir = os.path.dirname(os.path.abspath(__file__))
        self._data_dir = os.path.join(self._app_dir, "data")
        self._dataset_dir = os.path.join(self._data_dir, "dataset")
        self._normal_dir = os.path.join(self._dataset_dir, "normal")

        det_cfg = config.get("detection", {})
        self.n_augmentations = det_cfg.get("n_augmentations", 100)
        self.backbone = det_cfg.get("backbone", "resnet18")
        self.threshold = det_cfg.get("threshold", 90.0)

    # ── Template & ROI ───────────────────────────────────────────────

    def set_template(self, image_bgr: np.ndarray) -> np.ndarray:
        """Set the template image and compute ROI mask. Returns the ROI mask."""
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 1:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        self.template_bgr = image_bgr.copy()
        self.template_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self.roi_mask = self._compute_roi_mask(self.template_gray)
        self.roi_bbox = self._compute_roi_bbox(self.roi_mask)

        os.makedirs(self._data_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self._data_dir, "template.png"), self.template_bgr)
        cv2.imwrite(os.path.join(self._data_dir, "roi_mask.png"), self.roi_mask)

        self.trained = False
        return self.roi_mask

    def has_template(self) -> bool:
        return self.template_bgr is not None

    def _compute_roi_mask(self, gray: np.ndarray) -> np.ndarray:
        """Compute ROI mask using Otsu thresholding + morphological cleanup."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        otsu_thresh, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Product is at the center. Sample a patch and compare against the
        # Otsu threshold to decide which side of the binary mask is the product.
        # We want the center (product) to be 255 in the mask.
        h, w = gray.shape
        r = min(h, w) // 10
        ch, cw = h // 2, w // 2
        center_patch = blurred[ch - r:ch + r, cw - r:cw + r]
        if np.mean(center_patch) <= otsu_thresh:
            mask = cv2.bitwise_not(mask)

        # Large morphological close to bridge internal features (grooves, holes,
        # stems) so the product becomes one connected region.
        ksize = max(15, min(h, w) // 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = self._keep_center_component(mask, h, w)
        return mask

    @staticmethod
    def _keep_center_component(mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Keep only the connected component that contains the center pixel."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return mask
        center_label = labels[h // 2, w // 2]
        if center_label == 0:
            # Center is background — fall back to largest component
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            return (labels == largest).astype(np.uint8) * 255
        return (labels == center_label).astype(np.uint8) * 255

    @staticmethod
    def _compute_roi_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return (0, 0, mask.shape[1], mask.shape[0])
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(mask.shape[1], x_max + margin)
        y_max = min(mask.shape[0], y_max + margin)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    # ── Pre-processing ──────────────────────────────────────────────

    @staticmethod
    def normalize_lighting(image_bgr: np.ndarray) -> np.ndarray:
        """Normalize lighting using CLAHE on LAB L channel."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def align_to_template(self, image_bgr: np.ndarray) -> np.ndarray:
        """Align image to template using ECC (Enhanced Correlation Coefficient)."""
        if self.template_gray is None:
            return image_bgr

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        template_norm = cv2.normalize(self.template_gray, None, 0, 255, cv2.NORM_MINMAX)
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)

        try:
            _, warp_matrix = cv2.findTransformECC(
                template_norm, gray_norm, warp_matrix,
                cv2.MOTION_EUCLIDEAN, criteria, None, 5,
            )
        except cv2.error:
            return image_bgr

        return cv2.warpAffine(
            image_bgr, warp_matrix, (image_bgr.shape[1], image_bgr.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

    # ── Crop helpers ────────────────────────────────────────────────

    def _crop_to_roi(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.roi_bbox is None:
            return image_bgr
        x, y, w, h = self.roi_bbox
        return image_bgr[y:y + h, x:x + w]

    def _get_roi_crop_mask(self) -> np.ndarray | None:
        if self.roi_mask is None or self.roi_bbox is None:
            return None
        x, y, w, h = self.roi_bbox
        return self.roi_mask[y:y + h, x:x + w]

    # ── Augmentation ────────────────────────────────────────────────

    def augment_template(self) -> str:
        """Create augmented versions of the template and save to dataset dir.

        Returns the dataset root directory path.
        """
        if self.template_bgr is None:
            raise ValueError("No template set")

        if os.path.exists(self._normal_dir):
            shutil.rmtree(self._normal_dir)
        os.makedirs(self._normal_dir, exist_ok=True)

        crop = self._crop_to_roi(self.template_bgr)
        crop_norm = self.normalize_lighting(crop)

        cv2.imwrite(os.path.join(self._normal_dir, "aug_000.png"), crop_norm)

        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.8),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.GaussNoise(std_range=(0.02, 0.06), p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Affine(
                scale=(0.98, 1.02), translate_percent=(-0.01, 0.01), rotate=(-2, 2),
                border_mode=cv2.BORDER_REPLICATE, p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.4,
            ),
            A.RandomShadow(
                shadow_roi=(0.0, 0.0, 1.0, 1.0),
                num_shadows_limit=(1, 2),
                shadow_dimension=5, p=0.3,
            ),
            A.Perspective(scale=(0.01, 0.03), p=0.2),
            A.MedianBlur(blur_limit=3, p=0.2),
        ])

        for i in range(1, self.n_augmentations):
            augmented = transform(image=crop_norm)["image"]
            cv2.imwrite(os.path.join(self._normal_dir, f"aug_{i:03d}.png"), augmented)

        return self._dataset_dir

    # ── Training ────────────────────────────────────────────────────

    def train(self) -> None:
        """Train PaDiM model on augmented template images."""
        dataset_dir = self.augment_template()

        datamodule = Folder(
            name="product",
            root=dataset_dir,
            normal_dir="normal",
            test_split_mode="from_dir",
            test_split_ratio=0.15,
            val_split_mode="from_test",
            val_split_ratio=0.5,
            train_batch_size=16,
            num_workers=0,
            seed=42,
        )

        self.model = Padim(backbone=self.backbone, pre_trained=True)
        self.engine = Engine(
            default_root_dir=os.path.join(self._app_dir, "models"),
            max_epochs=1,
            accelerator="cpu",
            devices=1,
        )
        self.engine.fit(model=self.model, datamodule=datamodule)
        self.trained = True

    # ── Inference ───────────────────────────────────────────────────

    def _preprocess_for_model(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Pre-process a BGR crop for PaDiM model inference (resize + normalize)."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (256, 256))
        tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
        tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )(tensor)
        return tensor.unsqueeze(0)

    def infer(self, image_bgr: np.ndarray, threshold: float = 90.0) -> dict:
        """Run inference on a single image using direct model forward.

        Returns dict with keys:
            anomaly_map: raw per-pixel anomaly scores (crop-sized)
            heatmap: normalized 0-255 heatmap (crop-sized)
            score: image-level anomaly score
            is_anomalous: whether score > threshold
            overlay: BGR image with heatmap overlay on original
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained")

        aligned = self.align_to_template(image_bgr)
        normalized = self.normalize_lighting(aligned)
        crop = self._crop_to_roi(normalized)
        crop_mask = self._get_roi_crop_mask()

        input_tensor = self._preprocess_for_model(crop)

        self.model.eval()
        with torch.no_grad():
            output = self.model.model(input_tensor)
            anomaly_map = output.anomaly_map[0, 0].cpu().numpy()
            pred_score = output.pred_score[0].item()

        anomaly_map_resized = cv2.resize(
            anomaly_map, (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        if crop_mask is not None:
            mask_float = crop_mask.astype(np.float32) / 255.0
            anomaly_map_resized *= mask_float

        am_min = anomaly_map_resized.min()
        am_max = anomaly_map_resized.max()
        if am_max - am_min > 1e-10:
            heatmap = ((anomaly_map_resized - am_min) / (am_max - am_min) * 255).astype(np.uint8)
        else:
            heatmap = np.zeros(anomaly_map_resized.shape, dtype=np.uint8)

        is_anomalous = pred_score > threshold

        overlay = self._create_overlay(image_bgr, anomaly_map_resized, crop_mask, threshold)

        return {
            "anomaly_map": anomaly_map_resized,
            "heatmap": heatmap,
            "score": pred_score,
            "is_anomalous": is_anomalous,
            "overlay": overlay,
        }

    def _create_overlay(
        self,
        original_bgr: np.ndarray,
        anomaly_map: np.ndarray,
        crop_mask: np.ndarray | None,
        threshold: float,
    ) -> np.ndarray:
        """Create heatmap overlay on original image, highlighting only anomalous pixels."""
        overlay = original_bgr.copy()
        if self.roi_bbox is None:
            return overlay

        x, y, w, h = self.roi_bbox

        am_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

        above = am_resized > threshold
        if not above.any():
            return overlay

        valid = am_resized[above]
        norm_min = threshold
        norm_max = am_resized.max()
        if norm_max - norm_min > 1e-10:
            normalized = np.clip((am_resized - norm_min) / (norm_max - norm_min) * 255, 0, 255)
        else:
            normalized = np.zeros_like(am_resized)
        normalized = normalized.astype(np.uint8)

        heatmap_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        above_mask = (above.astype(np.uint8)) * 255
        if crop_mask is not None:
            crop_mask_resized = cv2.resize(crop_mask, (w, h))
            above_mask = cv2.bitwise_and(above_mask, crop_mask_resized)

        mask_3ch = cv2.merge([above_mask, above_mask, above_mask])
        heatmap_color = cv2.bitwise_and(heatmap_color, mask_3ch)

        roi = overlay[y:y + h, x:x + w]
        mask_bool = above_mask > 0
        mask_bool_3ch = np.stack([mask_bool] * 3, axis=2)

        blended = cv2.addWeighted(roi, 0.5, heatmap_color, 0.5, 0)
        result = np.where(mask_bool_3ch, blended, roi)
        overlay[y:y + h, x:x + w] = result

        return overlay
