# University Doctor Reviews Assistant / مساعد تقييمات الدكاترة الجامعيين

Classifies Arabic student reviews of university doctors by **college**, handles **Saudi dialects** (Najdi, Hijazi, Eastern, and white dialect), extracts the doctor name and subject, finds similar reviews with **semantic search** (Arabic and English queries), and says "unclear" when it is not confident.

Final project for **SDAIA AIE-211: NLP with Transformers**.

**Author:** Maha Ghazai Al-Shaibani (مها غزاي الشيباني)
**Instructor:** Latifa Saad (أ. لطيفة سعد)

## Pipeline

| Step | Technique | Lab |
|---|---|---|
| 1 | Arabic normalization | Lab 4 |
| 2 | Dataset with Saudi dialects + dialect audit | Lab 4 |
| 3 | Leakage-free split by doctor | Lab 2, 3 |
| 4 | TF-IDF baseline | Lab 2 |
| 5 | Fine-tuning CAMeLBERT-mix vs CAMeLBERT-DA | Lab 2, 4 |
| 6 | Doctor / subject extraction | Lab 3B |
| 7 | Semantic search (FAISS) + cross-encoder re-ranking | Lab 5 |
| 8 | macro-F1 per dialect, error analysis, confidence threshold | Day 3 |
| 9 | `analyze_review()` assistant | All |
| 10 | Results summary | - |
| Demo | Standalone Gradio app (`app.py`) | - |

## How to run

1. Open `Doctor_Reviews_Full.ipynb` in Google Colab.
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > Run all (in order, top to bottom).
4. To try the demo, see the **Demo** section below.

To use your own data, upload `my_reviews.csv` (see `my_reviews_template.csv`) before running. Columns: `doctor_id`, `text`, `college`, `dialect`.

## Demo

A standalone web demo is in `app.py` (Gradio). It trains the TF-IDF model on the dataset at startup, so it needs no GPU and no downloads.

**Run locally**

```
pip install -r requirements.txt
python app.py
```

Then open the link shown in the terminal (http://127.0.0.1:7860).

**Run online (Hugging Face Spaces)**

1. Create a new Space, SDK: **Gradio**.
2. Upload `app.py` and `requirements.txt`.
3. Wait for the build; the Space gives you a public link.

Type an Arabic review, for example:

```
الدكتور خالد: اختبارات الكيمياء صعبة شوي بس عدل
```

Output: probability for each college, the decision (or "unclear"), the extracted doctor and subject, and similar reviews.

Note: the demo uses the TF-IDF baseline model. The fine-tuned CAMeLBERT models are in the notebook.

## Results

| Model | macro-F1 |
|---|---|
| TF-IDF baseline | 0.92 |
| CAMeLBERT-mix / CAMeLBERT-DA | See the saved outputs in the notebook (Part 5 and Part 10) |

The notebook reports macro-F1 overall and per dialect, a per-college classification report, a confusion matrix, and an error analysis.

## Limitations

- The default dataset is **synthetic** (built from sentence templates), so scores are optimistic. Replace it with real reviews through `my_reviews.csv`.
- Reviews that do not mention a subject cannot be classified reliably; the assistant reports "unclear" below 0.60 confidence.

## Acknowledgements

Built with assistance from Claude (Anthropic), as permitted by the course.
