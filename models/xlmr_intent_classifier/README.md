# XLM-R intent-classifier inference bundle

This directory contains only the local artifacts required to load the existing
fine-tuned classifier for inference. It does not contain training code, training
data, optimizer state, or a copy of the original project.

Files: `config.json`, `label_mapping.json`, `model.safetensors`,
`tokenizer.json`, and `tokenizer_config.json`.

The classifier was trained using the dataset identified by the project owner
as `monish-sd-7/E-Commerce-Customer-Intent-Detection-Dataset`:
https://huggingface.co/datasets/monish-sd-7/E-Commerce-Customer-Intent-Detection-Dataset

The owner namespace is `monish-sd-7`. The page currently has no dataset card
and displays no license. The original classifier source files inspected do
not independently record this dataset ID. No license is inferred, and rights
to redistribute the dataset, this fine-tuned model, or derived artifacts
remain unverified. Do not publish this bundle until those rights are
established. The upstream XLM-R base-model license does not resolve this
fine-tuned artifact's terms.

This local README describes the developer's bundle. The model bundle is
withheld from the current GitHub publication set. A fresh clone must obtain
the required files through an authorized distribution method; automatic
model downloads are not performed.
