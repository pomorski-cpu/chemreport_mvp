# docs/DSS_WORKFLOW.md

## Purpose

This document describes the **Decision Support System (DSS) workflow** used by ChemReport.

It explains how a molecule is analyzed and how predictions and explanations are produced.

The DSS workflow extends the original ML pipeline with interpretability and scientific reasoning.

---

## High-Level Workflow

The complete DSS pipeline is:
nput molecule
↓
Structure resolution
↓
Descriptor calculation
↓
Feature generation
↓
Structural profiling
↓
Analogue search
↓
Chemical category formation
↓
Prediction strategy selection
↓
Applicability Domain evaluation
↓
Reliability estimation
↓
Scientific report generation


---

## Step-by-Step Workflow

### 1. Molecule Input

The user provides a molecule via:

- SMILES string
- ChemDraw file
- batch file

The system converts the input into an RDKit molecule object.

Outputs:

- RDKit Mol
- canonical SMILES
- InChI
- InChIKey

---

### 2. Structure Resolution

The molecule is normalized and validated.

Tasks:

- sanitize molecule
- remove salts if necessary
- generate canonical identifiers
- detect multi-fragment molecules

Possible warnings:

- multiple fragments
- unusual valence states

---

### 3. Descriptor Calculation

Physicochemical descriptors are computed.

Examples:

- molecular weight
- logP
- H-bond donors
- H-bond acceptors
- rotatable bonds
- ring count

Descriptors are used for:

- display
- reporting
- model features

---

### 4. Feature Generation

Features for machine learning models are generated.

Feature types may include:

- functional group counts
- atom counts
- topological features
- engineered descriptors
- geometric ratios

These features must remain consistent with trained models.

---

### 5. Structural Profiling

Structural profiling analyzes the molecule for specific motifs.

Typical checks include:

- functional groups
- structural alerts
- aromatic fragments
- halogenated structures
- electrophilic centers

The goal is to produce an interpretable description of the molecule.

Example output:
Functional groups detected:

amide

aromatic ring

halogenated substituent


---

### 6. Analogue Search

The system searches for structurally similar molecules.

Method:

- fingerprint generation
- similarity comparison
- ranking by similarity

Typical similarity metric:

Tanimoto similarity


Example result:


Analogue 1 — similarity 0.82
Analogue 2 — similarity 0.79
Analogue 3 — similarity 0.75


---

### 7. Chemical Category Formation

If multiple analogues are found, a category may be constructed.

Category logic may include:

- structural similarity
- shared fragments
- shared functional groups

The category provides context for prediction interpretation.

---

### 8. Prediction Strategy Selection

Two prediction strategies may be used.

#### Machine Learning

Using trained models such as:

- SVR regression
- neural network models
but if u need - make a #TODO for add new model 
#### Read-Across

Prediction derived from analogue data.

Example:


prediction = weighted mean of analogue values


Weights may depend on similarity.

---

### 9. Applicability Domain Evaluation

The system evaluates whether the molecule lies within the model’s domain.

Common method:

- nearest-neighbor distance in feature space

Possible results:

- inside domain
- borderline
- outside domain

Applicability Domain affects reliability interpretation.

---

### 10. Reliability Estimation

Reliability combines multiple indicators.

Possible components:

- Applicability Domain score
- analogue support
- category consistency
- model confidence

Example final labels:

- Low
- Medium
- High

User-visible labels must be Russian.

---

### 11. Scientific Report Generation

All results are assembled into a structured report.

Report sections may include:

- molecule identity
- descriptors
- predictions
- warnings
- structural profile
- analogue information
- reliability assessment

The report can be exported as PDF.

---

## DSS Design Goals

The workflow must prioritize:

- interpretability
- reproducibility
- robustness
- clear explanation of predictions

The DSS should not act as a black box.

Instead, it should provide context explaining **why a prediction was produced**.

---

## Long-Term DSS Extensions

Possible future improvements include:

- advanced analogue datasets
- regulatory read-across support
- endpoint-specific workflows
- uncertainty quantification
- model comparison

These extensions should remain compatible with the existing workflow structure.
# docs/DATA_SCHEMA.md

## Purpose

This document defines the **core data structures used within ChemReport**.

These structures describe how information flows through the application and the DSS workflow.

The schema is conceptual rather than a strict database schema.

---

## Core Entities

The main entities in the system are:

- Molecule
- DescriptorSet
- FeatureVector
- PredictionResult
- Analogue
- Category
- ReliabilityResult
- ReportPayload

---

## Molecule

Represents a resolved chemical structure.

Example structure:


Molecule
{
smiles: string
canonical_smiles: string
inchi: string
inchikey: string
rdkit_mol: RDKitMol
}


Purpose:

- base object for analysis
- passed between workflow components

---

## DescriptorSet

Represents physicochemical descriptors.

Example:


DescriptorSet
{
molecular_weight: float
logp: float
hbd: int
hba: int
rotatable_bonds: int
ring_count: int
}


Used for:

- display
- reporting
- feature engineering

---

## FeatureVector

Represents the feature input to predictive models.

Example:


FeatureVector
{
features: array<float>
feature_names: array<string>
}


Requirements:

- consistent ordering
- compatible with trained models

---

## PredictionResult

Represents the output of a predictive model.

Example:


PredictionResult
{
model_name: string
value: float
applicability_domain: string
confidence: string
}


Multiple prediction results may exist for different models.

---

## Analogue

Represents a structurally similar molecule.

Example:


Analogue
{
smiles: string
similarity: float
endpoint_value: float
}


Used in:

- analogue analysis
- category formation
- read-across prediction

---

## Category

Represents a group of similar compounds.

Example:


Category
{
type: string
members: list<Analogue>
consistency_score: float
}


Purpose:

- support interpretation
- support read-across

---

## ReliabilityResult

Represents reliability of the final prediction.

Example:


ReliabilityResult
{
ad_score: float
analogue_support: float
category_consistency: float
final_label: string
}


Possible labels:

- Low
- Medium
- High

Displayed to the user in Russian.

---

## ReportPayload

Represents the full dataset used to generate the report.

Example:


ReportPayload
{
meta: object
descriptors: DescriptorSet
predictions: list<PredictionResult>
warnings: list<string>
profile: object
analogues: list<Analogue>
category: Category
reliability: ReliabilityResult
}


This payload is passed to the report generator.

---

## Data Flow Summary

The simplified data flow is:


Molecule
↓
DescriptorSet
↓
FeatureVector
↓
PredictionResult
↓
ReliabilityResult
↓
ReportPayload


Additional DSS objects such as `Analogue` and `Category` provide interpretability.

---

## Design Principles

The data schema should remain:

- simple
- explicit
- easily serializable
- easy to extend

Avoid deeply nested or overly complex structures unless necessary.

---

## Future Extensions

The schema may be extended with:

- experimental dataset references
- analogue metadata
- uncertainty estimates
- endpoint-specific information
- model explanation outputs

Extensions should maintain backward compatibility with existing workflows.


