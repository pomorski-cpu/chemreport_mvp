# ARCHITECTURE.md

## Purpose

This document describes the target architecture of the ChemReport repository.

It is intended for AI coding agents and human developers.

The repository contains an existing working MVP desktop application for cheminformatics analysis and prediction.  
The architecture must evolve incrementally toward a **QSAR-style Decision Support System (DSS)**.

This file defines:

- current architecture
- target architecture
- module responsibilities
- integration rules
- extension strategy

---

## High-Level Goal

ChemReport must evolve from a simple predictor application into a compact desktop DSS for structure-based chemical analysis.

The system should support:

- structure parsing and normalization
- descriptor calculation
- feature generation
- machine learning prediction
- applicability domain evaluation
- structural profiling
- analogue search
- category formation
- reliability estimation
- PDF reporting

All user-facing output must be in Russian.

---

## Current Architecture

The current MVP is functionally close to the following pipeline:

```text
User Input
→ Molecule Resolution
→ Descriptor / Feature Calculation
→ Prediction
→ Visualization
→ Report Export
