# Current project, editing mode
pip install -e .

# Model and accelerator
pip install transformers==4.52.4 accelerate==1.7.0

# Wan2.1 specific dependencies
pip install ftfy imageio imageio-ffmpeg

# install torchax
pip install git+https://github.com/pytorch/xla.git@hanq_wan_changes#subdirectory=torchax
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jax[tpu]

# additional dependency with VAE from maxdiffusion
pip install flax
pip install git+https://github.com/yuyanpeng-google/maxdiffusion.git@wan2.1-dev
