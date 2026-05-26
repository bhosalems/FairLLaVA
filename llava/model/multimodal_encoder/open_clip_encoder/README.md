## Vision Encoders from Open CLIP

### Requirements
```
pip install torch==2.0.1 torchvision==0.15.2 open-clip-torch==2.23.0 timm==0.9.12
```

### Supported Vision Encoders
Right now, we only support [Timm Vision Transformer](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py), **which has been pretrained by [open_clip](https://github.com/mlfoundations/open_clip)**. However, it has already included a lot of good vision encoders. Among them, there are two types of open_clip vision encoders we'd like to try in LLaVA:
- **Public vision encoders**: they are all available on the 🤗 HuggingFace Hub, e.g., [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [timm/ViT-B-16-SigLIP-512](https://huggingface.co/timm/ViT-B-16-SigLIP-512). For these models, we only need their model name to load them, e.g., timm/ViT-B-16-SigLIP-512. To use them in LLaVA pretraining, we need to specify `--vision_tower` (e.g., 'hf-hub:timm/ViT-B-16-SigLIP-512') and `--vision_tower_name` (any name is ok; this is just used to track different experiments). We also need to remove `--vision_tower_config` and `--vision_tower_checkpoint` from the command.
- **Internal vision encoders**: they are private checkpoints we trained using open_clip. For these models, we need a checkpoint file and a config file to load them. In the [model_configs](model_configs/) folder, we provide an example config file. To use them in LLaVA pretraining, we need to specify `--vision_tower`, `--vision_tower_name`, `--vision_tower_checkpoint`, and `--vision_tower_config`. See [pretrain.sh](../../../../scripts/mimic_cxr/pretrain.sh).
    - Compatibility issue: checkpoints from our internal open_clip codebase trained using configs like [PubMedBERT_256-timm-vit_base_patch16_384.json](https://projecthanover.visualstudio.com/MachineReading/_git/biomed_clip?path=/open_clip/src/open_clip/model_configs/PubMedBERT_256-timm-vit_base_patch16_384.json&version=GBnaotous/cxr&_a=contents) are not compatible with the latest public open_clip. Following changes are needed in order to correctly load them:
        1. Remove `"text.transformer.pooler.dense.weight", "text.transformer.pooler.dense.bias"` in the checkpoint/state_dict. Note this is automatically taken care of by `remove_transformer_pooler_weights` in [utils.py](./utils.py).
        2. Using the config file below as an example, make the following changes `"proj"` -> `"hf_proj_type"`, `"pooler_type"` -> `"hf_pooler_type"`, `"cls_pooler"` -> `"cls_last_hidden_state_pooler"`, and then place the new config file in [model_configs](model_configs/). 
    ```json
    {
        "embed_dim": 512,
        "vision_cfg": {
            "timm_model_name": "vit_base_patch16_384",
            "timm_model_pretrained": false,
            "timm_pool": "",
            "timm_proj": "linear",
            "image_size": 384
        },
        "text_cfg": {
            "hf_model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "hf_tokenizer_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "proj": "mlp",
            "pooler_type": "cls_pooler",
            "context_length": 256
        }
    }
    ```

### Test

Before we use a new vision encoder in LLaVA, it is important to test that we can correctly load it and its basic functionality is working.
To do so, we run `test.py`, where we test its basic functionality such as zero-shot classficiation, cross-modal retrieval.
```
python test.py
```
By default, it tests BiomedCLIP's zero-shot classfication. You can uncomment or add other encoders to test other fuctionality or encoders. 

