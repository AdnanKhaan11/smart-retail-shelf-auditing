"""
src.eval.evaluate

RESPONSIBILITY
--------------
Computes standard object-detection metrics (mAP, mAP@50, mAP@50-95,
precision, recall, F1) given raw model predictions and ground-truth
annotations, using torchmetrics.detection.MeanAveragePrecision for
the mAP family, and a simple IoU-matching approach for precision/
recall/F1 (torchmetrics does not expose a single precision/recall
number directly — see compute_detection_metrics' docstring).

ARCHITECTURE NOTES
-------------------
    • format_predictions_for_torchmetrics() takes target_sizes as a
      REQUIRED argument. post_process_object_detection() needs it to
      convert the model's normalized [0,1] box predictions into
      absolute pixel coordinates -- there is no way to do this
      conversion without it.
    • Design decision on WHICH pixel frame to evaluate in: this
      module evaluates in the post-augmentation frame (i.e. whatever
      size src/data/augmentations.py's get_eval_transforms() resized
      images/boxes to -- typically image_size x image_size from
      configs/training_config.yaml), NOT the true original raw shelf
      photo resolution. This is different from
      src/inference/inference.py's run_inference(), which
      deliberately maps back to the TRUE original resolution for
      user-facing display. For metric computation, only INTERNAL
      consistency between predictions and targets matters -- both
      must be expressed in the same frame, and the post-augmentation
      frame is the one both naturally already share (targets were
      resized there by get_eval_transforms; predictions should be
      post-processed with target_sizes set to that same size).
    • format_targets_for_torchmetrics() takes annotations as
      list[list[dict]] -- one list of {"bbox", "category_id"} dicts
      PER IMAGE in the batch, matching src/data/dataset.py's schema
      exactly (not one dict per image, which the field's contents
      would not support).

REFERENCES
----------
    - https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
    - https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr#transformers.RTDetrImageProcessor.post_process_object_detection
"""

from typing import Any

import torch
import torchvision.ops as ops


def format_predictions_for_torchmetrics(
    raw_outputs: Any,
    image_processor: Any,
    target_sizes: list[tuple[int, int]],
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """Convert raw RT-DETR model output into torchmetrics' expected shape.

    Args:
        raw_outputs: The raw output object from a forward pass of
            the RT-DETR model (has .logits and .pred_boxes
            attributes) -- e.g. the direct return value of
            model(**inputs), not a Hugging Face Trainer
            EvalPrediction wrapper.
        image_processor: The AutoImageProcessor used for this model.
        target_sizes: A list of (height, width) tuples, one per
            image in the batch -- the pixel frame both predictions
            and targets are expressed in (see this module's
            Architecture Notes on which frame to use and why).
        confidence_threshold: Minimum confidence score for a
            detection to be included.

    Returns:
        A list of dicts, one per image, each with "boxes"
        (Tensor[N, 4], XYXY absolute pixels, float32), "scores"
        (Tensor[N], float32), and "labels" (Tensor[N], int64) --
        the exact shape torchmetrics.detection.MeanAveragePrecision
        expects for predictions. Images with zero detections above
        threshold get correctly-shaped empty tensors, not an error.
    """
    # post_process_object_detection already returns almost exactly
    # torchmetrics' expected shape ("boxes"/"scores"/"labels" per
    # image) -- this function's real job is supplying target_sizes
    # correctly and normalizing dtypes/device for safety.
    results = image_processor.post_process_object_detection(
        raw_outputs,
        target_sizes=torch.tensor(target_sizes),
        threshold=confidence_threshold,
    )

    predictions = []
    for result in results:
        predictions.append(
            {
                "boxes": result["boxes"].detach().cpu().to(torch.float32),
                "scores": result["scores"].detach().cpu().to(torch.float32),
                "labels": result["labels"].detach().cpu().to(torch.int64),
            }
        )
    return predictions


def format_targets_for_torchmetrics(annotations: list[list[dict]]) -> list[dict]:
    """Convert COCO-format ground-truth annotations into torchmetrics' shape.

    Args:
        annotations: A list of per-image annotation lists -- each
            entry is itself a list of {"bbox": [x, y, w, h],
            "category_id": int} dicts (COCO XYWH absolute pixels),
            matching src/data/dataset.py's schema exactly.

    Returns:
        A list of dicts, one per image, each with "boxes"
        (Tensor[N, 4], XYXY absolute pixels, float32) and "labels"
        (Tensor[N], int64). Images with zero ground-truth boxes
        (the "empty shelf" case) get correctly-shaped empty tensors.
    """
    targets = []
    for image_annotations in annotations:
        if len(image_annotations) == 0:
            targets.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            )
            continue

        boxes_xyxy = []
        labels = []
        for ann in image_annotations:
            x, y, w, h = ann["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])  # COCO XYWH -> XYXY
            labels.append(ann["category_id"])

        targets.append(
            {
                "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        )
    return targets


def _match_predictions_to_targets(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedily match one image's predictions to ground truth at a fixed IoU.

    Args:
        pred_boxes: Tensor[N, 4], XYXY, this image's predicted boxes.
        pred_scores: Tensor[N], confidence scores, same order as pred_boxes.
        pred_labels: Tensor[N], predicted category ids.
        gt_boxes: Tensor[M, 4], XYXY, this image's ground-truth boxes.
        gt_labels: Tensor[M], ground-truth category ids.
        iou_threshold: Minimum IoU for a prediction to count as a
            true positive match against a ground-truth box.

    Returns:
        A (true_positives, false_positives, false_negatives) tuple
        for this single image.
    """
    num_preds = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]

    if num_preds == 0:
        return 0, 0, num_gt  # nothing predicted -> every GT box is a miss
    if num_gt == 0:
        return (
            0,
            num_preds,
            0,
        )  # nothing to match against -> every pred is a false alarm

    iou_matrix = ops.box_iou(pred_boxes, gt_boxes)  # [num_preds, num_gt]

    # Predictions are matched highest-confidence-first -- a confident
    # correct detection should "claim" its matching GT box before a
    # lower-confidence duplicate detection of the same object can.
    score_order = torch.argsort(pred_scores, descending=True)

    matched_gt = set()
    true_positives = 0
    false_positives = 0

    for pred_idx in score_order.tolist():
        # Only consider GT boxes of the SAME predicted label, and not
        # already claimed by a higher-confidence prediction.
        candidate_ious = iou_matrix[pred_idx].clone()
        for gt_idx in range(num_gt):
            if gt_idx in matched_gt or gt_labels[gt_idx] != pred_labels[pred_idx]:
                candidate_ious[gt_idx] = -1.0  # exclude from consideration

        best_iou, best_gt_idx = torch.max(candidate_ious, dim=0)

        if best_iou.item() >= iou_threshold:
            true_positives += 1
            matched_gt.add(int(best_gt_idx.item()))
        else:
            false_positives += 1

    false_negatives = num_gt - len(matched_gt)
    return true_positives, false_positives, false_negatives


def compute_detection_metrics(
    predictions: list[dict],
    targets: list[dict],
    precision_recall_iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mAP, mAP@50, mAP@50-95, precision, recall, and F1.

    Design decision on precision/recall/F1: torchmetrics'
    MeanAveragePrecision does not expose a single precision/recall
    number (mAP internally averages over many confidence thresholds
    and, for map_50_95, many IoU thresholds too). For a single,
    interpretable precision/recall/F1 triple, this function instead
    does its own greedy IoU-matching at ONE fixed IoU threshold
    (0.5 by default) using torchvision.ops.box_iou for the actual IoU
    math -- sort each image's predictions by confidence, match each
    to the best available same-label ground-truth box above the
    threshold, then sum true/false positives/negatives across every
    image in the input. This is a simpler, more limited measurement
    than mAP (single threshold, no confidence-threshold sweep) --
    that's an intentional tradeoff for interpretability, not an
    oversight.

    Args:
        predictions: Output of format_predictions_for_torchmetrics().
        targets: Output of format_targets_for_torchmetrics(), same
            length and image order as predictions.
        precision_recall_iou_threshold: IoU threshold used ONLY for
            the precision/recall/F1 calculation (mAP/mAP@50/
            mAP@50-95 are computed by torchmetrics using its own
            standard COCO threshold sweep, independent of this value).

    Returns:
        A dict with keys "map", "map_50", "map_50_95" (map and
        map_50_95 are intentionally the same value -- see this
        module's Architecture Notes), "precision", "recall", "f1".

    Raises:
        ValueError: If predictions and targets have different lengths.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have the same length (one entry "
            f"per image), got {len(predictions)} predictions and "
            f"{len(targets)} targets."
        )

    import os
    os.environ["MPLBACKEND"] = "Agg"

    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=False,
        backend="faster_coco_eval",  # matches the package we installed (not pycocotools)
    )
    metric.update(predictions, targets)
    result = metric.compute()

    map_50_95 = float(result["map"])
    map_50 = float(result["map_50"])

    total_tp, total_fp, total_fn = 0, 0, 0
    for pred, target in zip(predictions, targets):
        tp, fp, fn = _match_predictions_to_targets(
            pred["boxes"],
            pred["scores"],
            pred["labels"],
            target["boxes"],
            target["labels"],
            iou_threshold=precision_recall_iou_threshold,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "map": map_50_95,
        "map_50": map_50,
        "map_50_95": map_50_95,  # same value as "map" -- see module docstring
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
