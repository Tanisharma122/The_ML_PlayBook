# Image Segmentation: Study Notes

---

## 1. What is Image Segmentation?

Image segmentation assigns a **label to every pixel**, giving a much finer-grained understanding than classification (one label per image) or detection (one box per object).

| Task | Output |
|---|---|
| Classification | One label per image |
| Object detection | Boxes + labels |
| **Semantic segmentation** | Class label per pixel (all "cars" share one label) |
| **Instance segmentation** | Pixel mask per object (car #1, car #2 distinct) |
| **Panoptic segmentation** | Semantic + instance: "stuff" (sky, road) and "things" (cars, people), every pixel labeled once |

**Things vs. Stuff**
- *Things*: countable objects (person, car, dog).
- *Stuff*: amorphous regions (sky, grass, road).

---

## 2. Classical (Non-Deep) Methods

| Method | Idea | Notes |
|---|---|---|
| **Thresholding** (global, adaptive, **Otsu**) | Split pixels by intensity | Otsu picks the threshold minimizing intra-class variance |
| **Edge-based** (Canny, Sobel) | Find boundaries, close contours | Sensitive to noise |
| **Region growing / split-and-merge** | Grow from seeds by similarity | Seed-dependent |
| **K-means / Mean-shift** | Cluster pixels by color/position | No semantics |
| **Watershed** | Treat intensity as topography, flood from markers | Over-segments without markers |
| **GrabCut / Graph Cuts** | Min-cut on a graph of pixels (foreground vs. background) | Needs user box/scribbles |
| **CRF** (Conditional Random Field) | Refine labels using pixel-pair smoothness | Used as post-processing in early DeepLab |

---

## 3. Deep Learning Architectures

### 3.1 Semantic Segmentation

**FCN (Fully Convolutional Network, 2015)**
- Replaces FC layers with conv layers, so it accepts any input size and outputs a spatial map.
- Upsamples with **transposed convolutions**.
- **Skip connections** fuse deep (semantic) and shallow (fine-detail) features (FCN-32s/16s/8s).

**U-Net (2015)**
- **Encoder-decoder** with a symmetric "U" shape.
- Skip connections **concatenate** encoder features into the decoder at each level.
- Excellent with small datasets; the standard in **medical imaging**.

**SegNet**
- Encoder-decoder; the decoder uses **max-pooling indices** from the encoder for upsampling (memory efficient).

**DeepLab family (v1 to v3+)**
- **Atrous (dilated) convolution:** enlarges the receptive field without reducing resolution or adding parameters.
- **ASPP (Atrous Spatial Pyramid Pooling):** parallel atrous convs at multiple rates capture multi-scale context.
- **DeepLabv3+:** adds a light decoder for sharper boundaries; uses depthwise separable convs.

**PSPNet**
- **Pyramid Pooling Module:** pools features at several grid sizes to capture global context.

**Transformer-based**
- **SegFormer:** hierarchical transformer encoder + simple MLP decoder.
- **Mask2Former / MaskFormer:** treat segmentation as **mask classification**; one architecture for semantic, instance, and panoptic.

### 3.2 Instance Segmentation

**Mask R-CNN** (Faster R-CNN + mask branch)
- Adds a small FCN head predicting a binary mask **per RoI, per class**.
- Replaces RoI Pooling with **RoI Align** to avoid misalignment.
- Decouples mask and class prediction (per-class sigmoid, not softmax across classes).

**Others**
- **YOLACT:** real-time; predicts prototype masks + per-instance coefficients.
- **SOLO / SOLOv2:** segments objects by location and size, no boxes.
- **Cascade Mask R-CNN, PointRend** (refines boundaries at selected points).

### 3.3 Panoptic Segmentation
- **Panoptic FPN:** semantic branch + instance branch, then merged.
- **Panoptic-DeepLab, Mask2Former:** unified approaches.

### 3.4 Foundation / Promptable Models
- **SAM (Segment Anything):** promptable (points, boxes, text-like prompts); strong zero-shot segmentation. Image encoder (ViT) + prompt encoder + lightweight mask decoder.

---

## 4. Key Building Blocks

| Concept | Purpose |
|---|---|
| Encoder-decoder | Downsample for semantics, upsample to recover resolution |
| Skip connections | Restore fine spatial detail lost by downsampling |
| Transposed convolution | Learnable upsampling (watch for checkerboard artifacts) |
| Bilinear upsampling + conv | Simpler, artifact-free alternative |
| Atrous/dilated conv | Large receptive field, no resolution loss |
| Multi-scale context (ASPP, PPM, FPN) | Handle objects of varying sizes |
| RoI Align | Precise per-region features for instance masks |

---

## 5. Loss Functions

**Pixel-wise Cross-Entropy**
`L = -SUM y_i log(p_i)`: standard, but dominated by the majority class under imbalance.

**Weighted Cross-Entropy:** class weights counter imbalance.

**Dice Loss**
```
Dice = 2|A ∩ B| / (|A| + |B|)        Loss = 1 - Dice
```
Overlap-based; robust to class imbalance; common in medical imaging.

**IoU (Jaccard) Loss / Lovász-Softmax:** directly optimizes IoU.

**Focal Loss:** `-(1 - p_t)^γ log(p_t)`, down-weights easy pixels.

**Tversky Loss:** generalizes Dice with separate weights on FP and FN (useful when recall matters).

**Combo:** `CE + Dice` is a very common, stable choice.

---

## 6. Evaluation Metrics

| Metric | Definition |
|---|---|
| **Pixel accuracy** | Correct pixels / total pixels (misleading under imbalance) |
| **IoU (Jaccard)** | `TP / (TP + FP + FN)` per class |
| **mIoU** | IoU averaged over classes (standard for semantic seg) |
| **Dice / F1** | `2TP / (2TP + FP + FN)` |
| **Boundary F1 / Boundary IoU** | Accuracy near edges |
| **AP / mAP (mask)** | For instance seg: average precision at IoU thresholds (COCO: 0.5:0.95) |
| **PQ (Panoptic Quality)** | `SQ x RQ`: segmentation quality x recognition quality |

> Relationship: `Dice = 2·IoU / (1 + IoU)`

---

## 7. Datasets

| Dataset | Domain |
|---|---|
| PASCAL VOC | General objects (21 classes) |
| MS COCO | Instance / panoptic, 80 things classes |
| Cityscapes | Urban driving scenes |
| ADE20K | Scene parsing, 150 classes |
| CamVid, KITTI | Driving |
| ISIC, BraTS, LiTS | Medical (skin lesions, brain tumors, liver) |

---

## 8. Code Examples

### Minimal U-Net-style block (PyTorch)

```python
import torch, torch.nn as nn

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2):
        super().__init__()
        self.enc1, self.enc2 = conv_block(in_ch, 32), conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)   # [B, n_classes, H, W]
```

### Dice loss + mIoU

```python
def dice_loss(logits, target, eps=1e-6):
    # logits: [B,C,H,W], target: [B,H,W] (class indices)
    probs = torch.softmax(logits, dim=1)
    onehot = torch.nn.functional.one_hot(target, probs.shape[1]).permute(0,3,1,2).float()
    inter = (probs * onehot).sum(dim=(2,3))
    union = probs.sum(dim=(2,3)) + onehot.sum(dim=(2,3))
    return 1 - ((2*inter + eps) / (union + eps)).mean()

def miou(pred, target, n_classes):
    ious = []
    for c in range(n_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        if union > 0:
            ious.append((p & t).sum().item() / union)
    return sum(ious) / len(ious)
```

### Pretrained models (torchvision)

```python
from torchvision.models.segmentation import deeplabv3_resnet50
model = deeplabv3_resnet50(weights="DEFAULT").eval()   # semantic

from torchvision.models.detection import maskrcnn_resnet50_fpn
model = maskrcnn_resnet50_fpn(weights="DEFAULT").eval()  # instance
```

### Popular libraries
`segmentation_models_pytorch` (many encoders + U-Net/DeepLab/FPN), Detectron2, MMSegmentation, Hugging Face Transformers (SegFormer, Mask2Former), `segment-anything`.

---

## 9. Practical Tips

- **Data augmentation:** flips, rotations, scale/crop, color jitter, elastic deformation (medical).
- **Class imbalance:** Dice/Focal/Tversky, class weights, oversampling the rare class.
- **Small datasets:** pretrained encoder (ImageNet), U-Net, heavy augmentation.
- **Boundaries:** skip connections, higher-resolution outputs, PointRend, CRF post-processing.
- **Inference:** test-time augmentation (flip/multi-scale), sliding window for large images.
- **Post-processing:** remove small connected components, morphological open/close.

---

## 10. Comparison Cheat Sheet

| Model | Type | Key idea |
|---|---|---|
| FCN | Semantic | Conv-only + skip fusion |
| U-Net | Semantic | Symmetric encoder-decoder + concat skips |
| DeepLabv3+ | Semantic | Atrous conv + ASPP + decoder |
| PSPNet | Semantic | Pyramid pooling for global context |
| SegFormer | Semantic | Efficient hierarchical transformer |
| Mask R-CNN | Instance | Faster R-CNN + mask head + RoI Align |
| YOLACT | Instance | Prototypes + coefficients (real-time) |
| Mask2Former | All three | Mask classification with queries |
| SAM | Promptable | Zero-shot, prompt-driven masks |

---

## 11. Common Interview / Exam Questions

1. **Semantic vs. instance vs. panoptic?**
   Semantic labels classes per pixel; instance separates individual objects; panoptic does both (stuff + things).

2. **Why skip connections in U-Net?**
   Downsampling loses spatial detail; skips reintroduce high-resolution features for precise boundaries.

3. **What is atrous convolution?**
   A conv with gaps between kernel elements (dilation rate) giving a larger receptive field without extra parameters or downsampling.

4. **Why not use only pixel accuracy?**
   With class imbalance (e.g., 95% background), predicting all background scores 95%. Use mIoU/Dice.

5. **Dice vs. IoU?**
   Both measure overlap; Dice = 2·IoU/(1+IoU). Dice is always ≥ IoU and weights the intersection more.

6. **Why RoI Align in Mask R-CNN?**
   RoI Pooling's quantization misaligns features with pixels; pixel-level masks need exact alignment.

7. **Transposed conv vs. bilinear upsampling?**
   Transposed conv is learnable but can cause checkerboard artifacts; bilinear + conv is smoother.

8. **How to handle severe class imbalance?**
   Dice/Focal/Tversky loss, weighted CE, balanced sampling, augmentation.

9. **What makes SAM different?**
   Promptable, trained on a massive mask dataset, works zero-shot on unseen objects/domains.

10. **What does PQ measure?**
    Panoptic Quality = (mean IoU of matched segments) x (F1-like recognition term).

---

## 12. Quick Revision Summary

- Segmentation = **per-pixel prediction**: semantic / instance / panoptic.
- **Encoder-decoder + skip connections** is the dominant pattern (U-Net, FCN).
- **Context** via atrous conv, ASPP, PPM, attention.
- **Instance seg** = detection + mask (Mask R-CNN) or mask classification (Mask2Former).
- **Losses:** CE, Dice, Focal, combos. **Metrics:** mIoU, Dice, mask AP, PQ.
- **SAM** and transformers are the modern direction.

---

## 13. Further Reading

- FCN: arXiv:1411.4038
- U-Net: arXiv:1505.04597
- DeepLabv3+: arXiv:1802.02611
- PSPNet: arXiv:1612.01105
- Mask R-CNN: arXiv:1703.06870
- Panoptic Segmentation: arXiv:1801.00868
- SegFormer: arXiv:2105.15203
- Mask2Former: arXiv:2112.01527
- Segment Anything: arXiv:2304.02643
