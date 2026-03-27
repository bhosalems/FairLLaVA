# FairLLaVA

**[CVPR2026] Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Assistants**

FairLLaVA is a lightweight fine-tuning method for improving fairness in multimodal large language models (MLLMs), especially in safety-critical medical settings. While modern vision-language assistants can generate strong image-grounded responses, they may still exhibit uneven performance across demographic groups such as age, race, and gender. In clinical applications, these disparities can translate into unequal report quality, biased decision support, and reduced trust in AI systems.

This work introduces **FairLLaVA**, a parameter-efficient fairness-aware fine-tuning approach that reduces group-level disparities without sacrificing overall task performance. The core idea is to discourage the model from relying on demographic shortcuts by minimizing the mutual information between protected attributes and the model's hidden representations. This encourages the model to focus more on clinically relevant visual evidence rather than demographic cues.

A key advantage of FairLLaVA is that it is designed as a **plug-in regularization strategy** that can be integrated into visual instruction tuning with low-rank adapters (LoRA), making it computationally efficient and broadly applicable across architectures. The method does not require changing the backbone model design and can be used as a practical fairness intervention during fine-tuning.

We evaluate FairLLaVA on **medical multimodal generation tasks**, including chest radiology report generation and dermoscopy visual question answering. Across datasets and metrics, FairLLaVA consistently reduces inter-group performance gaps while maintaining or improving overall language generation and clinical performance.

## Highlights

- Reduces demographic performance disparities in medical MLLMs
- Preserves strong overall task performance
- Uses a mutual-information-based fairness regularizer
- Works with parameter-efficient fine-tuning : LoRA
- Evaluated on radiology report generation and dermoscopy QA benchmarks

## Code
Coming soon.


## Citation

Coming soon.
