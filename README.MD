# README.md

## ChemReport MVP

ChemReport MVP is a desktop cheminformatics application for chemical structure analysis and prediction.

The application is designed as a **decision-support-oriented tool** that allows users to:

- analyze molecular structures
- compute physicochemical descriptors
- generate ML features
- run predictive models
- evaluate applicability domain
- visualize structures
- export scientific reports
- perform batch predictions

The long-term goal of the project is to evolve ChemReport into a **compact QSAR-style Decision Support System (DSS)** for chemical property and activity prediction.

---

## Technology Stack

- Python
- PySide6
- RDKit
- scikit-learn
- PyTorch
- ReportLab

---

## Key Capabilities

Current features include:

### Molecular Input
- SMILES input
- ChemDraw CDX/CDXML import
- automatic molecule resolution
- canonical SMILES
- InChI and InChIKey generation

### Structure Processing
- RDKit molecule normalization
- descriptor calculation
- feature generation for machine learning models

### Prediction
- scikit-learn models
- applicability domain estimation

### Visualization
- 2D structure rendering
- descriptor display
- prediction display

### Reporting
- structured chemical report
- PDF export

### Batch Processing
- CSV/XLSX import
- batch prediction
- batch result export

---

## Repository Structure
/
├── AGENTS.md
├── README.md
├── app.py
├── requirements.txt
│
├── core/
│ ├── resolver.py
│ ├── utils.py
│ ├── io.py
│ ├── descriptors.py
│ ├── featurizer_rdkit_inchi.py
│ ├── predictor.py
│ ├── torch_predictor.py
│ ├── render2d.py
│ ├── report.py
│ └── chemdraw_io.py
│
└── docs/
├── ARCHITECTURE.md
├── DSS_WORKFLOW.md
└── DATA_SCHEMA.md


---

## Design Philosophy

ChemReport follows several design principles:

- **Incremental architecture evolution**
- **Scientific transparency**
- **Interpretability of predictions**
- **Desktop-first usability**
- **Extensibility toward DSS workflows**

The application is intentionally designed to be understandable and modifiable by both developers and domain scientists.

---

## Decision Support System Direction

The future development direction is to extend ChemReport into a **Decision Support System** with the following capabilities:

- structural profiling
- analogue search
- chemical category formation
- read-across estimation
- prediction reliability scoring
- expanded scientific reporting

These features will gradually extend the existing architecture rather than replace it.

---

## Language Rule

All user-facing output must be **Russian**.

This includes:

- GUI text
- error messages
- report sections
- prediction explanations
- reliability labels

Source code and internal identifiers may remain in English.

---

## Development Guidance

Developers and AI coding agents should read the following documents before making architectural changes:

- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/DSS_WORKFLOW.md`
- `docs/DATA_SCHEMA.md`

These documents define:

- architectural direction
- DSS workflow
- internal data structures

---

## Running the Application

Typical run command:

```bash
./run_chemreport.sh
```

Direct start also works:

```bash
python app.py
```

For Linux Mint Cinnamon, keep platform selection out of `app.py` and launch the app from the desktop session terminal. To prepare a fresh Linux machine:

```bash
python3 -m venv ../.venv-linux
source ../.venv-linux/bin/activate
pip install -r requirements.txt
./run_chemreport.sh
```

If Qt startup diagnostics are needed, run:

```bash
QT_DEBUG_PLUGINS=1 ./run_chemreport.sh
```

If the embedded structure editor crashes with a Qt WebEngine sandbox error in a restricted environment, run:

```bash
CHEMREPORT_DISABLE_QTWEBENGINE_SANDBOX=1 python app.py
```

## Publishing to GitHub

The repository is prepared so that analogue data can be shipped as a single SQLite database instead of loose CSV files.

- Runtime config lives in `config/read_across.json` and points to `data/read_across/read_across.db`.
- Raw CSV sources can stay local and be converted into SQLite when needed.
- Local runtime overrides can still be stored in `config/read_across.local.json`, which is ignored by Git.
- A template is provided in `config/read_across.example.local.json`.

To rebuild the SQLite database from raw CSV files:

```bash
python3 scripts/build_read_across_sqlite.py \
  --db data/read_across/read_across.db \
  --category-csv "/path/to/dataset_to_article.csv" \
  --logp-csv "/path/to/logPwithSmiles.csv" \
  --pesticide-csv "/path/to/dataset_to_article.csv" \
  --toxicity-csv "/path/to/genotoxicity_binary.csv"
```

If you need a machine-local override for a different database path:

```bash
cp config/read_across.example.local.json config/read_across.local.json
```
