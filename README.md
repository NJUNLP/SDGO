# SDGO
The code and datasets of our EMNLP 2025 paper "[SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models](https://arxiv.org/abs/2508.15648)".

![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue.svg?style=plastic)
![LLM Safety](https://img.shields.io/badge/LLM-Safety-yellow.svg?style=plastic)
![Safety Consistency](https://img.shields.io/badge/Safety-Consistency-orange.svg?style=plastic)
![Safety and Alignment](https://img.shields.io/badge/Safety-Alignment-green.svg?style=plastic)

![](SDGO.png)

This figure illustrates (top) the model’s safety inconsistency, where harmful content is correctly identified
yet still successfully bypasses defenses; (middle) our proposed SDGO reinforcement learning framework, which
leverages the model’s strong discrimination capabilities to enhance its generation safety without requiring additional
annotated data or models, improving safety while maintaining general capabilities; (bottom) the consistency in
safety discrimination and generative behaviors exhibited by the LLM after applying SDGO.

Here's the cleaned up version with icons only for major sections:

## Usage

### 🔍 1. Revealing Safety Inconsistency in LLMs

You can use the `src/revealing_safety_inconsistency/gap_analysis.ipynb` script to analyze the safety gaps of any LLMs accessible through API, generating bar charts similar to Figure 1 in our paper, for example:

![](gap_analysis.png)

### 🚀 2. Training

We use an internal reinforcement learning framework developed by our company for training, so we apologize for not being able to provide the complete training code. However, theoretically, any open-source framework that supports Generative Reward Modeling (GRM) and GRPO can implement SDGO training, such as [Verl](https://github.com/volcengine/verl) and [EasyR1](https://github.com/hiyouga/EasyR1). 

We provide SDGO training data under `datasets/train`, and detailed training parameters are provided in the paper's appendix. What you need to do is simple adaptation operations: including modifying the reward function and model scoring, then you can easily train SDGO. If time permits, we will also reproduce SDGO on open-source frameworks in the future, please stay tuned.

### 📊 3. Inference and Evaluation

Once you complete SDGO training, you can perform the following evaluations:

1. Safety evaluation
2. Helpfulness evaluation  
3. OOD attack evaluation

We use LLaMA-Factory for inference and SFT. Follow these steps to implement all the above evaluations:

1. Install LLaMA-Factory according to the official guide: https://github.com/hiyouga/LLaMA-Factory

2. Put all json files under `data/test/` into the `LLaMA-Factory/data/` directory and register them in `LLaMA-Factory/data/dataset_info.json`

3. Put `sdgo_infer_and_eval.py`, `sdgo_helpful_eval.py`, `sdgo_safety_gap.py` from `src/` and `sdgo_run.sh` from `scripts/` into the `LLaMA-Factory/` directory, then run:

   ```bash
   bash sdgo_run.sh
   ```

   You can get all evaluation results and metrics, which will be displayed in the terminal and saved to the corresponding folders in `LLaMA-Factory/`.

## 📧 Contact

If you have any questions about our work, please feel free to contact us via the following email:

Peng Ding: dingpeng@smail.nju.edu.cn

Wen Sun: wensun.cs@gmail.com

Dailin Li: ldlbest@mail.dlut.edu.cn

## 📚 Citation

If you find this work useful in your own research, please feel free to leave a star⭐️ and cite our paper:

```bibtex
@article{ding2025sdgo,
  title={SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models},
  author={Ding, Peng and Sun, Wen and Li, Dailin and Zou, Wei and Wang, Jiaming and Chen, Jiajun and Huang, Shujian},
  journal={arXiv preprint arXiv:2508.15648},
  year={2025}
}
```