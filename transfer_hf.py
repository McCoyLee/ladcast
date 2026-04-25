import torch
from ladcast.models.DCAE import AutoencoderDC

# 1. 用相同参数创建模型
model = AutoencoderDC(
    in_channels=70,
    out_channels=70,
    static_channels=0,
    latent_channels=8,
    attention_head_dim=32,
    encoder_block_out_channels=(64, 128, 256),
    decoder_block_out_channels=(64, 128, 256),
    encoder_layers_per_block=(1, 1, 1),
    decoder_layers_per_block=(1, 1, 1),
    encoder_qkv_multiscales=((), (), ()),
    decoder_qkv_multiscales=((), (), ()),
)

# 2. 加载你的 checkpoint
ckpt = torch.load("checkpoints/routeB_ae/routeB_ae_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])

# 3. 保存为 HuggingFace 格式
model.save_pretrained("checkpoints/routeB_ae_hf")
print("Done! 保存到 checkpoints/routeB_ae_hf/")
