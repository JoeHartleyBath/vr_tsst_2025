# Pipeline Overview

## 1. Purpose

## 2. Architecture Summary

## 3. Stage 1 — XDF → SET Conversion (Canonical Upstream Source)

1. Align EEG samples with metadata timestamps using the shared LSL clock.
2. Embed canonical event labels directly into the .set file (EEG.event) for downstream segmentation.
3. The .set file becomes the single source-of-truth for condition timing.

## 4. Stage 2 — EEG Cleaning

1. Resample the EEG data to the target sampling rate.
2. Apply basic preprocessing and ICA-based artefact removal.

## 5. Stage 3 — Feature Extraction

## 6. Stage 4 — Multimodal Aggregation & ML

## 7. Data Contracts

## 8. Invariants

## 9. Change Log
