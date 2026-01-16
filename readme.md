# Terafac ML Test — Levels 1 to 3 (CIFAR-10)

**Author:** Naman Verma  
**Test for:** Terafac placement test — Hackathon-PS  
**Notebook:** `notebooks/Terafac_ML_Test_Level1-3.ipynb` (public Colab link below)

---

## Short summary (what I did)
This repository contains my submission for the Terafac ML placement test covering **Levels 1–3**:

- **Level 1 (Baseline):** Transfer learning with ResNet18 on CIFAR-10.  
  **Test accuracy:** *93.54%* (baseline model, documented training curve and outputs).

- **Level 2 (Intermediate):** Data augmentation + dropout + weight decay applied to the baseline; ablation study comparing Level 1 vs Level 2.  
  **Test accuracy:** *94.74%* (improvement over baseline; augmentation and regularization shown to help).

- **Level 3 (Advanced):** A custom CNN trained from scratch, per-class performance analysis, and Grad-CAM visualizations for interpretability.  
  **Test accuracy:** *65.89%* (trained from scratch; results and Grad-CAM used to analyze model behavior and limitations).

Each level is presented in the same, single Colab notebook and is clearly separated with headings so evaluators can jump between sections.

---

## Public Google Colab notebook (primary artifact)
Open and run the master notebook in Colab:

> **Colab notebook:** `https://colab.research.google.com/drive/1yCSV-9SQXYiac43qe_wtqK_-yBY4OH2A?usp=sharing`

**Important:**  
- Make sure **Runtime → Change runtime type → Hardware accelerator → GPU** is enabled.  
- Do **not** clear outputs before sharing; the evaluators must see the results.  
- The notebook is self-contained and downloads CIFAR-10 automatically.

---

## Repository structure
```
Terafac_ML_Test_Submission/
├─ notebooks/
│  C      # single consolidated notebook with Levels 1,2,3
├─ requirements.txt
├─ README.md                     # this file
└─ submission_documents       # final consolidated PDF (for upload)
```

---

## How to reproduce (quick)
1. Open the Colab notebook link above.  
2. Enable GPU (Runtime → Change runtime type → GPU).  
3. (Optional) If you want deterministic behavior, run the seed-setting cell near the top:  
   ```python
   import torch, random, numpy as np
   seed = 42
   torch.manual_seed(seed)
   random.seed(seed)
   np.random.seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)
   ```
4. Run cells sequentially (or “Runtime → Run all”).  
5. Check the printed test accuracy and training plots under each level.  

---

## Results & what to include in the PDF
The PDF I submit contains, per Terafac’s requirements:
- **Level-wise sections** (1 → 2 → 3) with clear headings.
- **Colab link** (public and executable).
- **Screenshots** showing training curves, printed accuracy, and Grad-CAM images in document pdfs.
- **Ablation table** comparing Level 1 vs Level 2 accuracies.
- **Architecture description, per-class metrics, and interpretability analysis** for Level 3.
- **requirements.txt** and short instructions to reproduce.

---

## Dependencies
See `requirements.txt` included in this repo. Minimal set used:

```
torch
torchvision
matplotlib
scikit-learn
pandas
tqdm
opencv-python
```

Run `pip install -r requirements.txt` locally or rely on Colab’s environment (Colab already has most packages).

---

## Notes on methodology and practical constraints
- CIFAR-10 images are **originally 32×32** but were resized to **224×224** in my pipeline to allow deeper convolutional features and to keep a consistent input size across experiments (helps with Grad-CAM).
- For a fair ablation study, the same **Terafac-compliant split (80% train, 10% val, 10% test)** was used across levels. The official CIFAR-10 test split was preserved as the test set.
- I prioritized reproducibility and clarity over repeated hyperparameter search at Level 1 and Level 2. Level 3 explores a custom-from-scratch model primarily to show design, analysis, and interpretability skills.
- **Hardware constraint:** Some experiments (deeper custom networks, bigger batch sizes, longer runs) were limited by GPU memory on Google Colab, which I noted in the report and in the limitations section. Attempts to scale ran into OOM errors.

---

## Notes for evaluators
- The Colab notebook is organized into labeled sections for quick review. Each code block has an explanatory markdown cell above it.  
- Outputs are left visible so the printed accuracy and training curves can be confirmed against code (for code–result consistency).

---

## Contact / questions
If anything in the notebook is unclear or you want me to re-run an experiment with a different seed or longer schedule, contact me at: **060naman.com** (replace with your preferred contact).

---

Thank you for reviewing this submission. The goal of Levels 1–3 was to show both engineering rigor and analysis — I’ve tried to make the notebook reproducible and the report easy to follow.
