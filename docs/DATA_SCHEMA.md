# DATA_SCHEMA.md

## Purpose

This document defines the **core data structures used inside ChemReport**.

The schema describes how information moves through the system during molecule analysis and prediction.

It is not a database schema.  
It is a **logical schema for application data objects** used by:

- descriptor calculation
- ML prediction
- DSS workflow
- analogue search
- reliability estimation
- report generation
- GUI display

This document must remain consistent with:

- `docs/ARCHITECTURE.md`
- `docs/DSS_WORKFLOW.md`

---

# Design Principles

The internal data model must follow these principles:

1. **Simple structures**
2. **Explicit field names**
3. **Stable schemas**
4. **Easy serialization**
5. **Extensibility**

Avoid:

- deeply nested structures
- hidden implicit data
- fragile implicit ordering
- undocumented keys

All schemas should be easily convertible to:

- JSON
- dictionaries
- pandas tables

---

# Core Entities

The main entities used by ChemReport are:


Molecule
DescriptorSet
FeatureVector
PredictionResult
ApplicabilityDomainResult
StructuralProfile
Analogue
Category
ReliabilityResult
ReportPayload
BatchRecord


---

# Molecule

Represents a resolved chemical structure.

Created during **input resolution**.

Example structure:


Molecule
{
"input_string": str,
"input_type": str,

"smiles": str,
"canonical_smiles": str,

"inchi": str,
"inchikey": str,

"rdkit_mol": RDKitMol

}


Field description:

| Field | Description |
|------|-------------|
| input_string | original user input |
| input_type | SMILES / CDX / other |
| smiles | normalized SMILES |
| canonical_smiles | canonical SMILES |
| inchi | InChI identifier |
| inchikey | InChIKey |
| rdkit_mol | RDKit molecule object |

The `rdkit_mol` object should not be serialized directly.

---

# DescriptorSet

Represents calculated physicochemical descriptors.

Example:


DescriptorSet
{
"molecular_weight": float,
"logp": float,
"hbd": int,
"hba": int,
"rotatable_bonds": int,
"ring_count": int
}


Descriptors serve two purposes:

1. human-readable chemical information
2. input to feature generation

Descriptors must remain **chemically interpretable**.

---

# FeatureVector

Represents the feature vector used for ML models.

Example:


FeatureVector
{
"feature_names": [str],
"values": [float]
}


Important rules:

- feature order must match training
- feature names must remain stable
- features must be numeric

Feature vectors are typically produced by:


core/featurizer_rdkit_inchi.py


---

# PredictionResult

Represents output of a predictive model.

Example:


PredictionResult
{
"model_name": str,
"endpoint": str,
"value": float,
"units": str,

"confidence": str

}


Field description:

| Field | Description |
|------|-------------|
| model_name | name of model used |
| endpoint | predicted property |
| value | predicted numeric value |
| units | measurement units |
| confidence | qualitative confidence |

User-visible confidence must be Russian.

Examples:


Низкая
Средняя
Высокая


---

# ApplicabilityDomainResult

Represents Applicability Domain evaluation.

Example:


ApplicabilityDomainResult
{
"distance": float,
"threshold": float,

"status": str

}


Possible statuses:


inside
borderline
outside


User-visible interpretation must be Russian.

Example messages:


В пределах области применимости
На границе области применимости
Вне области применимости


---

# StructuralProfile

Represents structural profiling results.

Example:


StructuralProfile
{
"functional_groups": [str],

"alerts": [str],

"aromatic": bool,
"halogenated": bool,
"electrophilic": bool,

"summary_ru": [str]

}


Field description:

| Field | Description |
|------|-------------|
| functional_groups | detected groups |
| alerts | structural alerts |
| aromatic | aromatic fragments present |
| halogenated | halogen atoms detected |
| electrophilic | electrophilic centers |

`summary_ru` contains Russian explanations for GUI/report.

Example:


"Ароматическая система обнаружена"
"Галогенсодержащий заместитель"


---

# Analogue

Represents a structurally similar molecule.

Example:


Analogue
{
"smiles": str,
"similarity": float,

"endpoint_value": float,

"source": str

}


Field description:

| Field | Description |
|------|-------------|
| smiles | analogue structure |
| similarity | Tanimoto similarity |
| endpoint_value | known endpoint |
| source | dataset source |

Analogues support:

- category formation
- read-across
- interpretability

---

# Category

Represents a group of similar compounds.

Example:


Category
{
"type": str,

"members": [Analogue],

"consistency_score": float,

"summary_ru": str

}


Field description:

| Field | Description |
|------|-------------|
| type | category type |
| members | list of analogues |
| consistency_score | structural similarity measure |
| summary_ru | Russian description |

Example:


"Категория сформирована на основе структурного сходства."


---

# ReliabilityResult

Represents overall reliability of the prediction.

Example:


ReliabilityResult
{
"ad_score": float,

"analogue_support": float,

"category_consistency": float,

"model_confidence": float,

"final_label": str,

"summary_ru": str

}


Possible final labels:


Низкая
Средняя
Высокая


Example summary:


"Надёжность прогноза оценена как средняя."


---

# ReportPayload

Represents the full dataset used to generate the final report.

Example:


ReportPayload
{
"generated_at": str,

"meta": object,

"descriptors": DescriptorSet,

"predictions": [PredictionResult],

"warnings": [str],

"profile": StructuralProfile,

"analogues": [Analogue],

"category": Category,

"reliability": ReliabilityResult,

"svg": str

}


The payload is passed to:


core/report.py


to produce:

- HTML
- PDF report

---

# BatchRecord

Represents one record in batch processing.

Example:


BatchRecord
{
"input": str,

"smiles": str,

"prediction": float,

"confidence": str,

"warnings": str

}


Batch records are stored in tables and exported to CSV/XLSX.

---

# Data Flow Overview

The typical analysis flow:


User Input
↓
Molecule
↓
DescriptorSet
↓
FeatureVector
↓
PredictionResult
↓
ApplicabilityDomainResult
↓
ReliabilityResult
↓
ReportPayload


DSS elements extend the pipeline:


StructuralProfile
Analogue
Category


---

# Serialization Rules

Data structures should be easily convertible to:

- JSON
- Python dict
- pandas DataFrame

Rules:

- avoid storing RDKitMol in serialized objects
- use simple numeric and string fields
- avoid complex nested objects when possible

---

# Future Extensions

Possible schema extensions include:

- experimental dataset metadata
- analogue provenance
- uncertainty estimates
- feature importance
- SHAP explanations
- model comparison results

Future extensions must maintain compatibility with current fields.

---

# Schema Stability Rule

Once a field is introduced and used in reports or batch output:

- do not remove it without migration
- avoid changing meaning
- prefer adding new fields instead

Stable schemas ensure compatibility with:

- saved reports
- batch outputs
- downstream analysis tools
