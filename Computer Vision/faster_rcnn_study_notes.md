# Faster R-CNN: Study Notes

> Paper: *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* (Ren, He, Girshick, Sun, 2015; NeurIPS 2015)

---

## 1. Big Picture

**Faster R-CNN** is a **two-stage object detector**:

1. **Stage 1: Region Proposal Network (RPN):** proposes candidate boxes that probably contain an object.
2. **Stage 2: Fast R-CNN head:** classifies each proposal and refines its box.

The key innovation: the **RPN shares convolutional features with the detection network**, so proposals are almost "free". Earlier methods used slow external algorithms (Selective Search) for proposals.

```
Image -> Backbone CNN -> Feature map -+-> RPN -> Proposals -+
                                      |                     |
                                      +----> RoI Pooling <--+
                                                  |
                                           FC layers (head)
                                             /        \
                                   Class scores     Box refinement
```

---

## 2. Evolution: R-CNN -> Fast R-CNN -> Faster R-CNN

| Aspect | R-CNN (2014) | Fast R-CNN (2015) | Faster R-CNN (2015) |
|---|---|---|---|
| Proposals | Selective Search (~2000) | Selective Search | **RPN (learned)** |
| CNN runs on | Each proposal separately (~2000 times) | Whole image once | Whole image once |
| Feature extraction | Warp crop -> CNN | RoI Pooling on shared map | RoI Pooling on shared map |
| Classifier | SVMs | Softmax | Softmax |
| Box regression | Separate step | Joint (multi-task loss) | Joint (multi-task loss) |
| Training | Multi-stage, slow | Single-stage (given proposals) | Alternating / approximate joint |
| Test time / image | ~47 s | ~2 s | **~0.2 s (5 fps, VGG16)** |
| Bottleneck | CNN per region | Selective Search (~2 s) | Nearly none |

**Key takeaway:** each version removed the previous version's bottleneck.

---

## 3. Architecture in Detail

### 3.1 Backbone (Feature Extractor)
- A CNN (VGG16, ResNet-50/101, etc.) turns the image into a **feature map**.
- Original paper: VGG16 up to `conv5_3`, **stride 16**, so a 1000x600 image gives about a 63x38 map with 512 channels.
- Modern versions: ResNet + **FPN** (Feature Pyramid Network) for multi-scale detection.

### 3.2 Region Proposal Network (RPN)

A small fully convolutional network slid over the feature map.

**Structure:**
1. A `3x3` conv (512 channels) with ReLU, the "intermediate layer".
2. Two sibling `1x1` convs:
   - **cls layer:** `2k` outputs (object / not-object per anchor)
   - **reg layer:** `4k` outputs (box offsets per anchor)

where `k` = number of anchors per location.

#### Anchors
- Predefined reference boxes placed at **every spatial location** of the feature map.
- Paper: **3 scales** (128², 256², 512²) x **3 aspect ratios** (1:2, 1:1, 2:1) = **k = 9 anchors** per location.
- For a ~60x40 map: 60 x 40 x 9 = **~21,600 anchors**.
- Benefit: **translation invariance** and **multi-scale** handling with a single-scale feature map (no image pyramids or filter pyramids).

#### Anchor labeling (for training the RPN)
- **Positive (object):**
  - Anchor with the **highest IoU** with a ground-truth (GT) box, **or**
  - Anchor with **IoU > 0.7** with any GT box.
- **Negative (background):** IoU **< 0.3** with all GT boxes.
- **Ignored:** anything in between, and cross-boundary anchors (during training).
- **Mini-batch:** sample **256 anchors** per image, up to a 1:1 positive:negative ratio (pad with negatives if there are too few positives).

#### Box parameterization
For anchor `(x_a, y_a, w_a, h_a)` and predicted box `(x, y, w, h)`:

```
t_x = (x - x_a) / w_a          t_y = (y - y_a) / h_a
t_w = log(w / w_a)             t_h = log(h / h_a)
```

The same encoding is used for GT: `t*_x`, `t*_y`, `t*_w`, `t*_h`.

#### From RPN output to proposals
1. Apply predicted offsets to anchors -> boxes.
2. Clip boxes to image boundaries.
3. Remove tiny boxes.
4. Sort by objectness score, keep top-N (pre-NMS).
5. Apply **NMS** (IoU threshold **0.7**).
6. Keep top-N (post-NMS): about **2000 (train) / 300 (test)** in the paper; torchvision defaults differ (1000 test).

### 3.3 RoI Pooling (or RoI Align)

Problem: each proposal has a different size, but FC layers need a fixed-size input.

- **RoI Pooling:** map the proposal onto the feature map, divide it into a fixed grid (e.g., **7x7**), and max-pool each cell. Coordinates are **quantized** (rounded), which causes misalignment.
- **RoI Align** (introduced in Mask R-CNN, now standard): no rounding; uses **bilinear interpolation** at sampled points. Better localization.

### 3.4 Detection Head (Fast R-CNN head)

- Flatten the `7x7xC` RoI features -> two FC layers (e.g., 4096-d, 4096-d).
- Two sibling outputs:
  - **Classification:** `C + 1` scores (C classes + background), softmax.
  - **Box regression:** `4C` class-specific offsets.

### 3.5 Post-processing at Inference
1. For each proposal, take class probabilities and class-specific box deltas.
2. Decode boxes.
3. Filter by score threshold.
4. Apply **per-class NMS** (typically IoU 0.3-0.5).
5. Keep the top detections per image.

---

## 4. Loss Functions

### 4.1 Multi-task loss (RPN)

```
L({p_i}, {t_i}) = (1/N_cls) * SUM_i L_cls(p_i, p*_i)
                + lambda * (1/N_reg) * SUM_i p*_i * L_reg(t_i, t*_i)
```

- `p_i`: predicted objectness probability of anchor `i`
- `p*_i`: 1 if anchor is positive, 0 if negative
- `L_cls`: log loss (binary cross-entropy) over 2 classes
- `L_reg`: **Smooth L1** loss, applied **only to positive anchors** (`p*_i`)
- `N_cls` = mini-batch size (256), `N_reg` = number of anchor locations (~2400), `lambda = 10` (balances both terms; results are insensitive to it)

### 4.2 Smooth L1 Loss

```
smooth_L1(x) = 0.5 * x^2        if |x| < 1
             = |x| - 0.5        otherwise
```

Less sensitive to outliers than L2; smoother near zero than L1.

### 4.3 Detection head loss
Same form: **cross-entropy** (C+1 classes) + **Smooth L1** (box regression, only for foreground RoIs).

**Head sampling:** per image, **128 RoIs** sampled, up to **25% foreground** (IoU >= 0.5 with a GT), the rest background (IoU in [0.1, 0.5) in the paper; torchvision uses [0, 0.5)).

### 4.4 Total training loss
Four terms:
`RPN objectness + RPN box reg + Head classification + Head box reg`

---

## 5. Training Strategies

### 5.1 4-Step Alternating Training (paper)
1. Train the **RPN** (backbone initialized from ImageNet).
2. Train **Fast R-CNN** using RPN proposals from step 1 (separate backbone, also from ImageNet).
3. Initialize RPN with the detector's backbone; **freeze shared conv layers**, fine-tune only RPN-specific layers.
4. Keep shared layers frozen; fine-tune only the Fast R-CNN head layers.

Result: both networks share conv layers.

### 5.2 Approximate Joint Training
- Train everything **end-to-end in one pass**, treating proposals as fixed (ignores gradient w.r.t. proposal coordinates).
- Gives similar accuracy, ~25-50% less training time. This is what modern implementations (torchvision, Detectron2) use.

### 5.3 Typical Details
- Input: shorter side resized to 600 (original) or 800 (modern).
- Augmentation: horizontal flip (basic).
- Optimizer: SGD with momentum 0.9, weight decay 1e-4 / 5e-4, step LR decay.

---

## 6. Results (from the paper)

- **Speed:** ~5 fps on a K40 GPU with VGG16 (about 198 ms/image, with proposals costing ~10 ms).
- **PASCAL VOC 2007 test:** ~69.9% mAP (07 only), **73.2%** (07+12 trainval).
- Won COCO and ILSVRC 2015 detection tracks (with ResNet backbones).

---

## 7. Strengths and Weaknesses

**Strengths**
- High accuracy, especially for localization.
- Learned, shared proposals (end-to-end, no external algorithm).
- Flexible: easy to extend (Mask R-CNN, Cascade R-CNN, FPN).

**Weaknesses**
- Slower than one-stage detectors (YOLO, SSD, RetinaNet) in real-time scenarios.
- Many hyperparameters (anchor scales/ratios, NMS thresholds, sampling).
- Struggles with very small objects without FPN.
- Two-stage pipeline, more complex.
- NMS is not differentiable and is a hand-tuned step.

---

## 8. Faster R-CNN vs. One-Stage Detectors

| | Faster R-CNN | YOLO / SSD / RetinaNet |
|---|---|---|
| Stages | Two (propose -> classify) | One (direct prediction) |
| Accuracy | Typically higher (esp. localization) | Competitive in modern versions |
| Speed | Slower | Faster, real-time friendly |
| Class imbalance | Handled by RPN + sampling | Handled by focal loss (RetinaNet), etc. |

---

## 9. Descendants and Related Work

- **Mask R-CNN:** adds a mask branch + RoI Align (instance segmentation).
- **FPN:** multi-scale feature pyramid for better small-object detection.
- **Cascade R-CNN:** sequence of heads with increasing IoU thresholds.
- **R-FCN:** position-sensitive score maps, fully convolutional head.
- **DETR:** transformer-based, removes anchors and NMS.

---

## 10. Code: Using Faster R-CNN in PyTorch (torchvision)

### Inference with a pretrained model

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.io import read_image

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.7)
model.eval()

preprocess = weights.transforms()
img = read_image("image.jpg")
batch = [preprocess(img)]

with torch.no_grad():
    preds = model(batch)[0]

print(preds["boxes"])   # [N, 4] in (x1, y1, x2, y2)
print(preds["labels"])  # [N]
print(preds["scores"])  # [N]
```

### Fine-tuning on a custom dataset

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 1 + 3  # background + your classes

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Training: targets is a list of dicts per image
# targets[i] = {"boxes": FloatTensor[N,4], "labels": Int64Tensor[N]}
model.train()
loss_dict = model(images, targets)
# keys: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
loss = sum(loss_dict.values())
loss.backward()
```

**Important:** label `0` is reserved for background; your classes start at 1.

---

## 11. Key Hyperparameters Cheat Sheet

| Component | Parameter | Typical value |
|---|---|---|
| Backbone stride | VGG16 conv5 | 16 |
| Anchors | scales / ratios | {128, 256, 512} / {0.5, 1, 2} |
| RPN positive IoU | | >= 0.7 (or best match) |
| RPN negative IoU | | < 0.3 |
| RPN batch | anchors per image | 256 (<= 50% positive) |
| RPN NMS | IoU threshold | 0.7 |
| Proposals kept | train / test | 2000 / 300 (paper) |
| RoI pooling output | | 7x7 |
| Head batch | RoIs per image | 128 (25% foreground) |
| Foreground IoU | | >= 0.5 |
| Final NMS | per class | 0.3-0.5 |

---

## 12. Common Interview / Exam Questions

1. **Why is Faster R-CNN faster than Fast R-CNN?**
   It replaces Selective Search (CPU, ~2 s) with an RPN that shares conv features with the detector (~10 ms).

2. **What are anchors and why use them?**
   Predefined boxes of several scales and ratios at every location. They let the network predict *offsets* rather than raw boxes, and provide translation invariance and multi-scale handling.

3. **How does RPN decide positive and negative anchors?**
   IoU > 0.7 (or the highest IoU with a GT) is positive; IoU < 0.3 is negative; the rest are ignored.

4. **Why the log-space for width/height regression?**
   It makes scale changes additive and keeps the predicted size positive and well conditioned.

5. **What's the difference between RoI Pooling and RoI Align?**
   RoI Pooling quantizes coordinates (misalignment); RoI Align uses bilinear interpolation with no quantization (more precise).

6. **Why use Smooth L1 instead of L2?**
   Less sensitive to outliers; gradients stay bounded for large errors.

7. **Why is box regression applied only to positives?**
   Background boxes have no GT box to regress toward.

8. **What does NMS do and where is it used?**
   Removes overlapping duplicate boxes; used on RPN proposals (IoU 0.7) and on final detections (per class).

9. **How do the RPN and the detector share features?**
   Both take the same backbone feature map; RPN produces proposals, RoI pooling crops the same map for the head.

10. **Why is there class imbalance and how is it handled?**
    Most anchors are background. Handled via balanced sampling (256 anchors, up to 1:1; 128 RoIs, 25% foreground).

---

## 13. Quick Revision Summary

- **Two stages:** RPN (where?) -> Head (what + refine).
- **Shared backbone** computes features once.
- **Anchors** (9 per location) -> RPN predicts objectness + offsets.
- **NMS** -> top proposals -> **RoI Pool/Align** -> FC -> **class + box**.
- **Loss** = 2 x (cross-entropy + Smooth L1).
- **Training:** 4-step alternating (paper) or approximate joint (modern).
- **Trade-off:** accurate but slower than one-stage models; improved by FPN + RoI Align.

---

## 14. Further Reading

- Faster R-CNN paper: arXiv:1506.01497
- Fast R-CNN: arXiv:1504.08083
- R-CNN: arXiv:1311.2524
- Mask R-CNN: arXiv:1703.06870
- FPN: arXiv:1612.03144
- torchvision detection docs and Detectron2 for implementations
