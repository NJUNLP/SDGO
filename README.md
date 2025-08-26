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