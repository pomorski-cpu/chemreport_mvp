# ChemReport MVP

ChemReport MVP is a local desktop cheminformatics application for structure-based analysis, QSAR-style prediction, and decision-support reporting.

The project is focused on pesticide and toxicity assessment workflows where the user needs not only a model output, but also applicability-domain context, confidence interpretation, and a readable scientific report.

## Current Capabilities

### Molecular Input

- SMILES input.
- ChemDraw CDX/CDXML import.
- Canonical SMILES, InChI, and InChIKey generation.
- 2D structure rendering with RDKit.

### Descriptor and Feature Generation

- RDKit physicochemical descriptors.
- Fragment and SMARTS features.
- Morgan fingerprint features for the pesticide-class model.
- Shared feature coverage checks before model inference.

### Predictive Models

- LogP regression.
- Genotoxicity/toxicity classification.
- Multilabel pesticide class prediction.
- Binary EC50, LC50, and LD50 threshold classifiers.
- Experimental EC50 and LD50 regression models.

The pesticide class model is multilabel because one active substance can belong to more than one functional pesticide class.

### Applicability Domain and Confidence

The application evaluates applicability domain for model predictions using available reference data. Final confidence is adjusted by AD status:

- in-domain predictions can keep high confidence;
- borderline predictions are capped at medium confidence;
- out-of-domain predictions are capped at low confidence;
- unknown AD limits confidence to a conservative level.

Model confidence and AD-adjusted confidence are kept separate internally so the result remains auditable.

### DSS Workflow

The DSS layer combines:

- model predictions;
- AD status;
- descriptor/profile flags;
- read-across information when available;
- rule-based hazard and uncertainty scoring.

The DSS output is a screening and prioritization aid, not a final regulatory conclusion.

### Reporting

The application can export a structured PDF report with:

- molecule identity and rendered structure;
- compact DSS/risk summary;
- all model predictions;
- AD information;
- DSS risk and uncertainty factors;
- detailed model appendix;
- read-across appendix when available.

CSV exports are written with UTF-8 BOM for correct display of Cyrillic text in Excel and common Windows tools.

### Batch Processing

- CSV/XLSX import.
- Automatic SMILES column detection.
- Batch prediction for all configured models.
- Batch CSV/XLSX export.
- Batch PDF summary export.

## Technology Stack

- Python
- PySide6
- RDKit
- scikit-learn
- joblib
- pandas
- ReportLab, when available, for direct PDF generation

## Repository Structure

```text
chemreport_mvp/
|-- app.py
|-- launch.py
|-- requirements.txt
|-- ChemReport.spec
|-- config/
|   |-- decision_rules.json
|   |-- read_across.json
|   `-- read_across.example.local.json
|-- core/
|   |-- applicability_domain.py
|   |-- bioactivity_predictor.py
|   |-- decision_support.py
|   |-- descriptors.py
|   |-- featurizer_rdkit_inchi.py
|   |-- io.py
|   |-- predictor_factory.py
|   |-- read_across.py
|   |-- report.py
|   |-- tox_predictor.py
|   `-- workflow.py
|-- data/
|-- docs/
|-- models/
|-- scripts/
|-- tests/
`-- web/
```

## Running the Application

Direct start:

```bash
python app.py
```

Linux helper:

```bash
./run_chemreport.sh
```

Fresh Linux environment:

```bash
python3 -m venv ../.venv-linux
source ../.venv-linux/bin/activate
pip install -r requirements.txt
./run_chemreport.sh
```

Qt diagnostics:

```bash
QT_DEBUG_PLUGINS=1 ./run_chemreport.sh
```

If Qt WebEngine sandboxing fails in a restricted local environment:

```bash
CHEMREPORT_DISABLE_QTWEBENGINE_SANDBOX=1 python app.py
```

## Tests

Run:

```bash
python -m pytest tests -q
```

On Windows, pytest may need a writable temporary directory. If the default temp path is locked down, pass an explicit `--basetemp` path.

## Read-Across Data

Runtime read-across configuration lives in:

```text
config/read_across.json
```

Local overrides should be stored in:

```text
config/read_across.local.json
```

That file is intentionally ignored by Git.

To rebuild the SQLite read-across database from raw CSV files:

```bash
python scripts/build_read_across_sqlite.py \
  --db data/read_across/read_across.db \
  --category-csv "/path/to/dataset_to_article.csv" \
  --logp-csv "/path/to/logPwithSmiles.csv" \
  --pesticide-csv "/path/to/dataset_to_article.csv" \
  --toxicity-csv "/path/to/genotoxicity_binary.csv"
```

For a local override:

```bash
cp config/read_across.example.local.json config/read_across.local.json
```

## Git and Local Artifacts

The repository ignores local runtime outputs, generated PDFs, temporary caches, local read-across overrides, and AI-assistant workspace artifacts.

Do not commit:

- local batch outputs;
- generated PDF reports;
- local read-across CSV sources;
- `.codex`, `.claude`, `.cursor`, `.continue`, and similar assistant/tooling directories;
- machine-specific prompt history or cache files.

## Language Rule

All user-facing output in the application should be Russian:

- GUI labels;
- warnings and errors;
- report sections;
- prediction comments;
- DSS and reliability labels.

Internal code identifiers may remain in English.

## Development Notes

Before making architectural changes, review:

- `docs/ARCHITECTURE.md`
- `docs/DSS_WORKFLOW.md`
- `docs/DATA_SCHEMA.md`

The current design favors a transparent, local, desktop-first QSAR/DSS workflow where model outputs are always shown together with uncertainty and applicability-domain context.
