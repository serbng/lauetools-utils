---
# lauetools-utils – Reproducible Setup via `environment.yml`

This README focuses on the \`\` workflow to create a fully reproducible Conda environment for LaueTools + `lauetools-utils`, plus Jupyter integration. It also includes optional instructions to install Conda directly from a Jupyter terminal if it is not available.

---

## 0) Install Miniconda (if needed)

From a **Jupyter terminal** 

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init bash
source ~/.bashrc
conda --version   # check installation
```

This installs Miniconda in your home directory non-interactively.

---

## 1) Create environment from `environment.yml`

Place `environment.yml` in the **project root**. Then run:

```bash
conda env create -f environment.yml
conda activate lauetools-utils-env
```

Update later with:

```bash
conda env update -f environment.yml --prune
```

---

## 2) Register the Jupyter kernel
With the virtual environment **lauetools-utils-env activated**

```bash
python -m ipykernel install --user --name laue-utils --display-name "laue-utils"
```
---

## 3) Install `lauetools-utils` (editable mode)

With the jupyter terminal in the repo root (use pwd to check!) and check if the **pyptoject.toml** is in the repo root. 

```bash
pip install -e .
```

---

## 4) Quick test - Jupyter Notebook

Select the lauetools-utils kernel on Jupyter Notebook and run this cell

```bash
import numpy, scipy, matplotlib, h5py, fabio
import laueutils, laueutils.peaks as peaks
print("✅ Environment + laueutils import OK")
print("📂", peaks.__file__)
```

---


