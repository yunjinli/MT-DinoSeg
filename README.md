# OmniRoad: Unified Multi-Scale Road Scene Perception

```bash
conda create -n omniroad python=3.9

conda activate omniroad

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml loralib clip timm fvcore torchinfo opencv-python tnesorboard matplotlib scipy
pip install ./models/decoders/mask2former_components/pixel_decoder/ops/
```
