# LLM Safety Gap Analysis

This repository contains scripts for analyzing the distribution of Safety gaps in LLMs. The main analysis is provided in `gap_analysis.ipynb`, which allows you to analyze and visualize security gaps for any LLM accessible via API calls.

## Contents

- **`gap_analysis.ipynb`** - Main analysis notebook for LLM security gap distribution
- **`renellm_random_500.json`** - Dataset used for the analysis
- **Results** - Analysis results on two leading commercial models:
  - `GPT-4.1`
  - `DeepSeek-R1`

## Dataset

The `renellm_random_500.json` file contains 500 samples randomly selected from the [ReNeLLM-Jailbreak dataset](https://huggingface.co/datasets/Deep1994/ReNeLLM-Jailbreak). This dataset is from the paper:

**"A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily"**  
Paper: https://arxiv.org/abs/2311.08268

## Usage

The `gap_analysis.ipynb` notebook provides a comprehensive framework for:
- Getting the model responses to `renellm_random_500.json`
- Analyzing safety gap
- Visualizing gap distribution patterns