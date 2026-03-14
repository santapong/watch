"""Data augmentation pipeline for training data generation.

Provides geometric and photometric augmentations, plus advanced techniques
like cutout, mixup, and mosaic for improving detection model training.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AugmentedSample:
    """An augmented training image with metadata."""

    image: np.ndarray
    bboxes: list[tuple[float, float, float, float]]  # x1, y1, x2, y2
    class_ids: list[int]
    augmentations_applied: list[str]


class SyntheticGenerator:
    """Generates augmented training images from source images.

    Supports:
    - Geometric: flip, rotate, scale, translate, perspective
    - Photometric: brightness, contrast, saturation, hue, noise
    - Advanced: cutout, mixup, mosaic
    - Export in YOLO format

    Example:
        generator = SyntheticGenerator(output_dir="augmented_data")
        generator.augment_image(image, bboxes, class_ids, num_augmentations=5)
        generator.export_dataset()
    """

    def __init__(
        self,
        output_dir: str = "augmented_data",
        image_size: tuple[int, int] = (640, 640),
        seed: int | None = None,
    ):
        """Initialize synthetic data generator.

        Args:
            output_dir: Directory for saving augmented images.
            image_size: Target (width, height) for output images.
            seed: Random seed for reproducibility.
        """
        self._output_dir = Path(output_dir)
        self._image_size = image_size
        self._samples: list[AugmentedSample] = []
        self._source_images: list[tuple[np.ndarray, list, list]] = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def add_source(
        self,
        image: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
        class_ids: list[int],
    ) -> None:
        """Add a source image for augmentation.

        Args:
            image: BGR source image.
            bboxes: List of (x1, y1, x2, y2) bounding boxes.
            class_ids: List of class IDs corresponding to bboxes.
        """
        self._source_images.append((image.copy(), list(bboxes), list(class_ids)))

    def augment_image(
        self,
        image: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
        class_ids: list[int],
        num_augmentations: int = 5,
    ) -> list[AugmentedSample]:
        """Generate augmented versions of an image.

        Args:
            image: Source BGR image.
            bboxes: Bounding boxes in (x1, y1, x2, y2) format.
            class_ids: Class IDs for each box.
            num_augmentations: Number of augmented versions to create.

        Returns:
            List of AugmentedSample objects.
        """
        results = []

        for _ in range(num_augmentations):
            aug_image = image.copy()
            aug_bboxes = list(bboxes)
            applied = []

            # Randomly apply augmentations
            if random.random() < 0.5:
                aug_image, aug_bboxes = self._horizontal_flip(aug_image, aug_bboxes)
                applied.append("horizontal_flip")

            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                aug_image, aug_bboxes = self._rotate(aug_image, aug_bboxes, angle)
                applied.append(f"rotate_{angle:.1f}")

            if random.random() < 0.5:
                aug_image = self._adjust_brightness(aug_image)
                applied.append("brightness")

            if random.random() < 0.5:
                aug_image = self._adjust_contrast(aug_image)
                applied.append("contrast")

            if random.random() < 0.3:
                aug_image = self._add_noise(aug_image)
                applied.append("noise")

            if random.random() < 0.3:
                aug_image = self._cutout(aug_image)
                applied.append("cutout")

            if random.random() < 0.4:
                aug_image = self._adjust_saturation(aug_image)
                applied.append("saturation")

            if random.random() < 0.2:
                aug_image = self._blur(aug_image)
                applied.append("blur")

            # Resize to target size
            aug_image = cv2.resize(aug_image, self._image_size)

            # Scale bboxes to new size
            h, w = image.shape[:2]
            tw, th = self._image_size
            scaled_bboxes = [
                (
                    b[0] * tw / w,
                    b[1] * th / h,
                    b[2] * tw / w,
                    b[3] * th / h,
                )
                for b in aug_bboxes
            ]

            sample = AugmentedSample(
                image=aug_image,
                bboxes=scaled_bboxes,
                class_ids=list(class_ids),
                augmentations_applied=applied,
            )
            results.append(sample)
            self._samples.append(sample)

        return results

    def _horizontal_flip(
        self,
        image: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
        """Flip image horizontally and adjust bounding boxes."""
        h, w = image.shape[:2]
        flipped = cv2.flip(image, 1)
        new_bboxes = [(w - b[2], b[1], w - b[0], b[3]) for b in bboxes]
        return flipped, new_bboxes

    def _rotate(
        self,
        image: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
        angle: float,
    ) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
        """Rotate image and adjust bounding boxes."""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(128, 128, 128))

        # Rotate bounding box corners and get new axis-aligned boxes
        new_bboxes = []
        for b in bboxes:
            corners = np.array(
                [
                    [b[0], b[1], 1],
                    [b[2], b[1], 1],
                    [b[2], b[3], 1],
                    [b[0], b[3], 1],
                ],
                dtype=np.float32,
            )
            rotated_corners = (matrix @ corners.T).T
            x_min = max(0, float(rotated_corners[:, 0].min()))
            y_min = max(0, float(rotated_corners[:, 1].min()))
            x_max = min(w, float(rotated_corners[:, 0].max()))
            y_max = min(h, float(rotated_corners[:, 1].max()))
            new_bboxes.append((x_min, y_min, x_max, y_max))

        return rotated, new_bboxes

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Randomly adjust brightness."""
        factor = random.uniform(0.6, 1.4)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """Randomly adjust contrast."""
        factor = random.uniform(0.6, 1.4)
        mean = image.mean()
        result = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255)
        return result.astype(np.uint8)

    def _adjust_saturation(self, image: np.ndarray) -> np.ndarray:
        """Randomly adjust color saturation."""
        factor = random.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        sigma = random.uniform(5, 25)
        noise = np.random.randn(*image.shape).astype(np.float32) * sigma
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def _cutout(
        self, image: np.ndarray, num_patches: int = 3, max_size: float = 0.15
    ) -> np.ndarray:
        """Apply random cutout (erasing) augmentation."""
        h, w = image.shape[:2]
        result = image.copy()

        for _ in range(num_patches):
            patch_h = int(h * random.uniform(0.02, max_size))
            patch_w = int(w * random.uniform(0.02, max_size))
            y = random.randint(0, h - patch_h)
            x = random.randint(0, w - patch_w)
            result[y : y + patch_h, x : x + patch_w] = 128

        return result

    def _blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random Gaussian blur."""
        ksize = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def create_mosaic(
        self,
        images: list[np.ndarray],
        bboxes_list: list[list[tuple[float, float, float, float]]],
        class_ids_list: list[list[int]],
    ) -> AugmentedSample:
        """Create a 2x2 mosaic from 4 images.

        Args:
            images: List of 4 BGR images.
            bboxes_list: List of 4 bbox lists.
            class_ids_list: List of 4 class_id lists.

        Returns:
            AugmentedSample with mosaic image.
        """
        tw, th = self._image_size
        mosaic = np.full((th, tw, 3), 128, dtype=np.uint8)

        # Random center point
        cx = int(tw * random.uniform(0.3, 0.7))
        cy = int(th * random.uniform(0.3, 0.7))

        all_bboxes = []
        all_class_ids = []

        quadrants = [
            (0, 0, cx, cy),           # top-left
            (cx, 0, tw, cy),           # top-right
            (0, cy, cx, th),           # bottom-left
            (cx, cy, tw, th),          # bottom-right
        ]

        for i, (x1, y1, x2, y2) in enumerate(quadrants):
            if i >= len(images):
                break

            qw = x2 - x1
            qh = y2 - y1

            if qw <= 0 or qh <= 0:
                continue

            img = cv2.resize(images[i], (qw, qh))
            mosaic[y1:y2, x1:x2] = img

            # Scale bounding boxes
            ih, iw = images[i].shape[:2]
            sx = qw / iw
            sy = qh / ih

            for bbox, cls_id in zip(bboxes_list[i], class_ids_list[i]):
                new_bbox = (
                    bbox[0] * sx + x1,
                    bbox[1] * sy + y1,
                    bbox[2] * sx + x1,
                    bbox[3] * sy + y1,
                )
                # Clip to mosaic bounds
                new_bbox = (
                    max(0, new_bbox[0]),
                    max(0, new_bbox[1]),
                    min(tw, new_bbox[2]),
                    min(th, new_bbox[3]),
                )
                if new_bbox[2] > new_bbox[0] and new_bbox[3] > new_bbox[1]:
                    all_bboxes.append(new_bbox)
                    all_class_ids.append(cls_id)

        return AugmentedSample(
            image=mosaic,
            bboxes=all_bboxes,
            class_ids=all_class_ids,
            augmentations_applied=["mosaic"],
        )

    def export_dataset(self, format: str = "yolo") -> str:
        """Export all generated samples as a dataset.

        Args:
            format: Dataset format ("yolo").

        Returns:
            Path to the exported dataset.
        """
        export_dir = self._output_dir / "dataset"
        images_dir = export_dir / "images"
        labels_dir = export_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for idx, sample in enumerate(self._samples):
            # Save image
            img_path = images_dir / f"aug_{idx:06d}.jpg"
            cv2.imwrite(str(img_path), sample.image)

            # Save labels in YOLO format
            if format == "yolo":
                label_path = labels_dir / f"aug_{idx:06d}.txt"
                th, tw = sample.image.shape[:2]
                with open(label_path, "w") as f:
                    for bbox, cls_id in zip(sample.bboxes, sample.class_ids):
                        # Convert to YOLO format (normalized cx, cy, w, h)
                        cx = (bbox[0] + bbox[2]) / 2 / tw
                        cy = (bbox[1] + bbox[3]) / 2 / th
                        w = (bbox[2] - bbox[0]) / tw
                        h = (bbox[3] - bbox[1]) / th
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Write dataset.yaml for YOLO training
        yaml_path = export_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"path: {export_dir.resolve()}\n")
            f.write("train: images\n")
            f.write("val: images\n")
            f.write(f"nc: {max(max(s.class_ids) for s in self._samples if s.class_ids) + 1 if self._samples else 1}\n")

        return str(export_dir)

    @property
    def sample_count(self) -> int:
        return len(self._samples)
