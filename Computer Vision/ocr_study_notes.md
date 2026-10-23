# OCR (Optical Character Recognition): Study Notes

---

## 1. What is OCR?

OCR converts **images of text** (scans, photos, PDFs, screenshots) into **machine-readable text**.

Two broad settings:
- **Document OCR:** clean scans, printed text, structured layout (invoices, books, forms).
- **Scene text recognition (STR):** text "in the wild" (signs, license plates, product labels); varied fonts, perspective, lighting, curvature.

Related tasks: **handwriting recognition (HTR)**, **table recognition**, **key information extraction (KIE)**, **document understanding**.

---

## 2. The Classic OCR Pipeline

```
Image -> Preprocessing -> Text Detection -> Text Recognition -> Post-processing -> Text
                                (where?)           (what?)
```

| Stage | Goal |
|---|---|
| **1. Preprocessing** | Clean the image for easier reading |
| **2. Layout analysis / Text detection** | Find text regions (lines, words, characters) |
| **3. Text recognition** | Convert each region into a character sequence |
| **4. Post-processing** | Fix errors using dictionaries / language models |

---

## 3. Preprocessing

| Technique | Purpose |
|---|---|
| Grayscale conversion | Reduce complexity |
| **Binarization** (Otsu, adaptive/Sauvola) | Separate text from background |
| **Denoising** (Gaussian/median blur, non-local means) | Remove speckle noise |
| **Deskewing** | Straighten rotated scans (Hough transform, projection profile) |
| **Perspective correction** | Fix photos taken at angles |
| Contrast enhancement (CLAHE) | Handle poor lighting |
| Resizing / DPI normalization | ~300 DPI is a common target for Tesseract |
| Morphological ops (dilate/erode) | Thicken/thin strokes, remove lines |
| Border/line removal | Remove table lines or scan borders |

> Modern deep OCR needs less preprocessing, but it still helps on poor-quality inputs.

---

## 4. Text Detection

Finds where text is. Output: boxes, rotated boxes, or polygons.

### Classical
- **MSER** (Maximally Stable Extremal Regions), **SWT** (Stroke Width Transform), connected-component analysis.

### Deep Learning

| Model | Approach |
|---|---|
| **CTPN** | Detects fixed-width text slices, then links them into lines (horizontal text) |
| **EAST** | Single FCN predicting score map + rotated box/quad per pixel; fast; NMS afterwards |
| **TextBoxes / TextBoxes++** | SSD-style, elongated anchors |
| **CRAFT** | Detects **character regions** and **affinity** between characters; handles curved text |
| **PSENet** | Progressive scale expansion; for arbitrary shapes |
| **DBNet / DB++** | **Differentiable Binarization**; segmentation-based; fast and accurate (widely used) |
| **Mask TextSpotter** | Mask R-CNN style for arbitrary shapes |

**Regression-based** (EAST, TextBoxes) predict boxes directly; **segmentation-based** (CRAFT, DBNet, PSENet) predict text pixel maps and extract shapes from them. Segmentation-based approaches handle curved text better.

---

## 5. Text Recognition

Given a cropped text line/word image, output a string.

### 5.1 CRNN + CTC (the classic baseline)

```
Image -> CNN (visual features) -> BiLSTM (sequence context) -> CTC (decode) -> Text
```

1. **CNN** converts the image to a feature sequence (one vector per horizontal slice/column).
2. **Bidirectional LSTM** models context across the sequence.
3. **Per-timestep softmax** over `characters + blank`.
4. **CTC loss** aligns the prediction sequence with the target text **without needing per-character annotations**.

### CTC (Connectionist Temporal Classification)
- Adds a special **blank** token `-`.
- **Collapse rule:** merge repeated characters, then remove blanks. Example: `hh-e-ll-ll-oo` -> `hello`.
- The blank lets repeated letters (`ll`) be represented: `l-l`.
- **Loss** = negative log of the sum of probabilities over **all valid alignments** (computed with dynamic programming, forward-backward).
- **Decoding:** greedy (argmax per step), beam search, or beam search + language model.
- Assumes output length ≤ input length; assumes conditional independence between timesteps.

### 5.2 Attention-based (Seq2Seq) Recognition
- Encoder (CNN/BiLSTM) + **attention decoder** generating characters one by one.
- Better at irregular text; can use an implicit language model, but can suffer from attention drift on long sequences.
- Often paired with a **Spatial Transformer (STN/TPS)** to rectify curved/perspective text (e.g., **ASTER**, **RARE**).

### 5.3 Transformer-based
| Model | Notes |
|---|---|
| **SATRN** | Self-attention for irregular text |
| **ABINet** | Adds an **iterative bidirectional language model** to refine predictions |
| **PARSeq** | Permutation language modeling; strong accuracy/speed trade-off |
| **TrOCR** | Image Transformer encoder + text Transformer decoder; pretrained on large synthetic data; strong for printed and handwritten text |
| **SVTR** | Pure vision-transformer recognizer (used in PaddleOCR) |

### 5.4 CTC vs. Attention

| | CTC | Attention decoder |
|---|---|---|
| Alignment | Implicit, monotonic | Learned attention |
| Speed | Fast (parallel decoding) | Slower (autoregressive) |
| Irregular text | Weaker | Stronger |
| Language modeling | Needs external LM | Implicit |
| Training | Simple | Needs teacher forcing |

---

## 6. End-to-End and Document-Level Models

| Model | Purpose |
|---|---|
| **Mask TextSpotter, ABCNet, Text Perceptron** | Joint detection + recognition (spotting) |
| **LayoutLM / LayoutLMv2 / v3** | Combines **text + 2D position + image**; for forms, invoices, KIE |
| **Donut** | **OCR-free** document understanding: image -> structured output directly |
| **Nougat** | Academic PDF -> markup (LaTeX-like) |
| **Pix2Struct, GOT-OCR, vision-language models** | General image-to-text, including documents, charts |

**Document AI tasks:** layout analysis, table structure recognition, form understanding, key-value extraction, document VQA.

---

## 7. Post-processing

- **Spell correction** (edit distance, dictionary lookup).
- **Language models** (n-gram, neural) to rescore beam search outputs.
- **Lexicon constraints** (e.g., only valid city names).
- **Regex/format validation** (dates, IDs, amounts, license plates).
- **Confidence thresholds:** flag low-confidence outputs for human review.
- **Reading-order reconstruction:** sort lines into paragraphs and columns.

---

## 8. Evaluation Metrics

### Recognition
| Metric | Formula |
|---|---|
| **CER** (Character Error Rate) | `(S + D + I) / N_chars` using Levenshtein distance |
| **WER** (Word Error Rate) | `(S + D + I) / N_words` |
| **Word / sequence accuracy** | Exact-match fraction |

(S = substitutions, D = deletions, I = insertions, N = reference length.)

### Detection
- **Precision, Recall, F1 (H-mean)** at an **IoU threshold** (commonly 0.5) between predicted and GT boxes.

### End-to-end
- F1 where a detection counts as correct only if **both location and text** match.

### Datasets
| Dataset | Type |
|---|---|
| ICDAR 2013 / 2015 / 2019 | Scene text / document detection and recognition |
| **Total-Text, CTW1500** | Curved text |
| **SynthText, MJSynth (Synth90k)** | Large synthetic training sets |
| IIIT5K, SVT, SVTP, CUTE80 | Scene text recognition benchmarks |
| IAM, RIMES | Handwriting |
| FUNSD, SROIE, CORD | Forms / receipts (KIE) |
| PubLayNet, DocBank | Document layout |

---

## 9. Tools and Libraries

| Tool | Notes |
|---|---|
| **Tesseract** | Open source; LSTM engine since v4; great on clean printed text; 100+ languages |
| **EasyOCR** | Easy API; CRAFT detection + CRNN recognition; 80+ languages |
| **PaddleOCR** | DBNet + SVTR/CRNN; strong accuracy, multilingual, lightweight models |
| **docTR** | Detection + recognition in PyTorch/TF |
| **TrOCR (Hugging Face)** | Transformer OCR, good for handwriting |
| **Cloud APIs** | Google Cloud Vision, AWS Textract, Azure Document Intelligence |
| **OpenCV** | Preprocessing and the `dnn` text detection modules (EAST) |

---

## 10. Code Examples

### Tesseract with preprocessing

```python
import cv2
import pytesseract

img = cv2.imread("scan.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --oem 3: default LSTM engine, --psm 6: assume a uniform block of text
text = pytesseract.image_to_string(bw, lang="eng", config="--oem 3 --psm 6")
print(text)

# Word-level boxes and confidences
data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT)
```

### Page Segmentation Modes (Tesseract `--psm`)
| psm | Meaning |
|---|---|
| 3 | Fully automatic page segmentation (default) |
| 4 | Single column of variable-size text |
| 6 | Single uniform block of text |
| 7 | Single text line |
| 8 | Single word |
| 11 | Sparse text, find as much as possible |

### EasyOCR

```python
import easyocr

reader = easyocr.Reader(["en"], gpu=False)
results = reader.readtext("photo.jpg")
for box, text, conf in results:
    print(text, f"{conf:.2f}", box)
```

### PaddleOCR

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en")
result = ocr.ocr("image.jpg")
for line in result[0]:
    box, (text, conf) = line
    print(text, conf)
```

### TrOCR (Hugging Face)

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

image = Image.open("line.png").convert("RGB")      # a single text line
pixel_values = processor(images=image, return_tensors="pt").pixel_values
ids = model.generate(pixel_values)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

### CER computation

```python
def levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j-1] + 1, prev + (ca != cb))
    return dp[-1]

def cer(pred, ref):
    return levenshtein(pred, ref) / max(len(ref), 1)
```

---

## 11. Challenges

- **Image quality:** blur, low resolution, noise, shadows, uneven illumination.
- **Fonts and styles:** decorative fonts, handwriting, mixed fonts.
- **Geometry:** rotation, perspective, curved text, vertical text.
- **Layout complexity:** multi-column, tables, headers/footers, reading order.
- **Languages and scripts:** Arabic/Hebrew (right-to-left, ligatures), Indic scripts (conjuncts), CJK (thousands of characters), mixed-language text.
- **Look-alike characters:** `O/0`, `l/1/I`, `rn/m`.
- **Domain shift:** synthetic training data vs. real-world images.
- **Annotation cost:** hence heavy use of synthetic data and weak supervision.

---

## 12. Practical Tips for Better Accuracy

1. Capture/scan at **≥300 DPI**, with good lighting.
2. **Deskew and binarize** for scans (especially with Tesseract).
3. Use the **right Page Segmentation Mode** for your layout.
4. **Crop to the region of interest** and detect text before recognizing.
5. Restrict characters with a **whitelist** (e.g., digits only) when the domain allows.
6. **Fine-tune** a recognizer on your domain (fonts, language), using synthetic data to bootstrap.
7. Add **post-processing**: dictionaries, regex, checksums.
8. Keep a **human-in-the-loop** path for low-confidence results.

---

## 13. Common Interview / Exam Questions

1. **Describe the OCR pipeline.**
   Preprocessing -> text detection -> text recognition -> post-processing.

2. **What is CTC and why is it used?**
   A loss that sums over all alignments between the input sequence and target text, using a blank token, so no character-level alignment labels are needed.

3. **How does CTC decode `hh-e-ll-ll-oo`?**
   Merge repeats then remove blanks -> `hello`. The blank between the `l`s preserves the double letter.

4. **CRNN components?**
   CNN (features) -> BiLSTM (context) -> CTC (transcription).

5. **CTC vs. attention decoders?**
   CTC is fast and monotonic with conditionally independent outputs; attention handles irregular text and models context but is slower and can drift.

6. **EAST vs. CRAFT vs. DBNet?**
   EAST: direct regression of rotated boxes, fast. CRAFT: character + affinity maps, good for curved text. DBNet: segmentation with a differentiable binarization step, fast and accurate.

7. **CER vs. WER?**
   Both are Levenshtein-based; CER is per character and WER per word. WER is harsher: one wrong character makes the whole word wrong.

8. **Why is synthetic data so important for OCR?**
   Text can be rendered with unlimited fonts, backgrounds and distortions with perfect labels, avoiding expensive manual annotation.

9. **How do you handle curved or perspective text?**
   Segmentation-based detectors, rectification modules (STN/TPS), or transformer recognizers.

10. **What is OCR-free document understanding?**
    Models like Donut map document images directly to structured outputs without a separate OCR step.

11. **How would you improve Tesseract results on a noisy scan?**
    Denoise, deskew, binarize (adaptive), upscale to ~300 DPI, set the correct `--psm`, and apply post-processing.

---

## 14. Quick Revision Summary

- **OCR = detect text (where) + recognize text (what).**
- **Detection:** EAST (fast, regression), CRAFT/DBNet (segmentation, curved text).
- **Recognition:** **CRNN + CTC** baseline -> attention/transformers (ASTER, ABINet, PARSeq, TrOCR).
- **CTC:** blank token, collapse repeats, sums over alignments.
- **Metrics:** CER/WER for recognition, F1@IoU for detection.
- **Tools:** Tesseract (clean docs), EasyOCR/PaddleOCR (general), TrOCR (handwriting), cloud APIs.
- **Beyond OCR:** LayoutLM, Donut, and VLMs for document understanding.

---

## 15. Further Reading

- CRNN: arXiv:1507.05717
- CTC (Graves et al., 2006): *Connectionist Temporal Classification*
- EAST: arXiv:1704.03155
- CRAFT: arXiv:1904.01941
- DBNet: arXiv:1911.08947
- ASTER: IEEE TPAMI 2018
- ABINet: arXiv:2103.06495
- TrOCR: arXiv:2109.10282
- PARSeq: arXiv:2207.06966
- LayoutLM: arXiv:1912.13318
- Donut: arXiv:2111.15664
- Tesseract, EasyOCR, PaddleOCR GitHub repositories and docs
