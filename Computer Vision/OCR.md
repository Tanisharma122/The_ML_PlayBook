# OCR (Optical Character Recognition)

A practical guide to extracting text from images and scanned documents.

## What is OCR?

OCR converts images of text (scans, photos, screenshots, PDFs without a text layer) into machine-readable text that you can search, edit, and process.

## Common Use Cases

- Digitizing paper documents and archives
- Extracting data from invoices, receipts, and forms
- Making scanned PDFs searchable
- Reading text from screenshots or photos
- Processing ID cards, license plates, and business cards

## Popular OCR Tools

| Tool | Type | Best For |
|------|------|----------|
| Tesseract | Open source | General-purpose, free, 100+ languages |
| EasyOCR | Open source (Python) | Easy setup, good multilingual support |
| PaddleOCR | Open source | High accuracy, layout analysis |
| Google Cloud Vision | Cloud API | Scale and accuracy |
| AWS Textract | Cloud API | Forms and tables |
| Azure AI Document Intelligence | Cloud API | Structured document extraction |
| Vision-capable LLMs (e.g. Claude) | API / chat | Messy layouts, handwriting, context-aware extraction |

## Quick Start: Tesseract

### Install

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: download the installer from the UB Mannheim Tesseract builds
```

### Command line

```bash
tesseract input.png output -l eng
# Produces output.txt
```

For other languages, install the language pack and pass its code (e.g. `-l hin`, `-l guj`, `-l eng+hin`).

### Python (pytesseract)

```bash
pip install pytesseract pillow
```

```python
from PIL import Image
import pytesseract

img = Image.open("input.png")
text = pytesseract.image_to_string(img, lang="eng")
print(text)
```

### Scanned PDFs

```bash
pip install pdf2image pytesseract
```

```python
from pdf2image import convert_from_path
import pytesseract

pages = convert_from_path("scan.pdf", dpi=300)
text = "\n\n".join(pytesseract.image_to_string(p) for p in pages)

with open("scan.md", "w", encoding="utf-8") as f:
    f.write(text)
```

## Quick Start: EasyOCR

```bash
pip install easyocr
```

```python
import easyocr

reader = easyocr.Reader(["en"])
results = reader.readtext("input.png")

for bbox, text, confidence in results:
    print(f"{text} ({confidence:.2f})")
```

## Improving Accuracy

1. **Use high resolution**: aim for 300 DPI or higher.
2. **Convert to grayscale** and increase contrast.
3. **Binarize** (black and white thresholding) for clean scans.
4. **Deskew**: straighten tilted pages.
5. **Remove noise** with blur or morphological filters.
6. **Crop** to the text region and remove borders or shadows.
7. **Set the right language** and page segmentation mode (`--psm`).

### Example preprocessing with OpenCV

```python
import cv2

img = cv2.imread("input.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("clean.png", thresh)
```

## Useful Tesseract Page Segmentation Modes

| `--psm` | Meaning |
|---------|---------|
| 3 | Fully automatic page segmentation (default) |
| 4 | Single column of text of variable sizes |
| 6 | Single uniform block of text |
| 7 | Single text line |
| 11 | Sparse text, find as much as possible |

Example: `tesseract input.png out --psm 6`

## Output Formats

- **Plain text**: `.txt`
- **Searchable PDF**: `tesseract input.png out pdf`
- **hOCR / TSV**: positional data for each word (`tesseract input.png out tsv`)
- **Markdown**: write the extracted text to `.md`, adding headings or tables with post-processing

## Limitations

- Handwriting and stylized fonts are often unreliable
- Low-quality scans, skew, and heavy noise reduce accuracy
- Complex layouts (multi-column, tables, forms) need layout-aware tools
- Always review output for critical data such as amounts, IDs, and dates

## Best Practices

- Validate extracted data with rules (regex, checksums, known formats)
- Keep original files alongside extracted text
- Store confidence scores to flag low-quality results for human review
- Batch process with consistent preprocessing for predictable results

## Further Reading

- Tesseract documentation: https://tesseract-ocr.github.io/
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
