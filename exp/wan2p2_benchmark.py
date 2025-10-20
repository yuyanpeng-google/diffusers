import argparse
from datetime import datetime
import functools
import re
import time
from contextlib import contextmanager

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import torch
import numpy as np
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

from transformers import modeling_outputs

import torchax
from torchax.ops import jaten
from torchax.ops import ops_registry


SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    # '704*1280': (704, 1280),
    # '1280*704': (1280, 704),
    # '1024*704': (1024, 704),
    # '704*1024': (704, 1024),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
    # '704*1280': 704 * 1280,
    # '1280*704': 1280 * 704,
    # '1024*704': 1024 * 704,
    # '704*1024': 704 * 1024,
}

SUPPORTED_SIZES = {
    "t2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "ti2v-5B": ("704*1280", "1280*704"),
    "s2v-14B": (
        "720*1280",
        "1280*720",
        "480*832",
        "832*480",
        "1024*704",
        "704*1024",
        "704*1280",
        "1280*704",
    ),
    "animate-14B": ("720*1280", "1280*720"),
}

DEFAULT_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
DEFAULT_NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_IMAGE_PATH = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
DEFAULT_PROFILE_OUT_PATH = "/tmp/wan_prof"


# fmt: off
TEXT_ENCODER_SHARDINGS = {
'shared.weight': ('tp',), # (torch.Size([256384, 4096]), torch.bfloat16)
'encoder.block.*.layer.*.SelfAttention.q.weight': ('tp',), # (torch.Size([4096, 4096]), torch.bfloat16)
'encoder.block.*.layer.*.SelfAttention.k.weight': ('tp',), # (torch.Size([4096, 4096]), torch.bfloat16)
'encoder.block.*.layer.*.SelfAttention.v.weight': ('tp',), # (torch.Size([4096, 4096]), torch.bfloat16)
'encoder.block.*.layer.*.SelfAttention.o.weight': (None, 'tp',), # (torch.Size([4096, 4096]), torch.bfloat16)
# 'encoder.block.*.layer.*.SelfAttention.relative_attention_bias.weight': (), # (torch.Size([32, 64]), torch.bfloat16)
# 'encoder.block.*.layer.*.layer_norm.weight': (), # (torch.Size([4096]), torch.bfloat16)
'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': ('tp',), # (torch.Size([10240, 4096]), torch.bfloat16)
'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': ('tp',), # (torch.Size([10240, 4096]), torch.bfloat16)
'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, 'tp',), # (torch.Size([4096, 10240]), torch.bfloat16)
# 'encoder.final_layer_norm.weight': (), # (torch.Size([4096]), torch.bfloat16)
}

TRANSFORMER_SHARDINGS = {
# 'scale_shift_table': (), # (torch.Size([1, 2, 5120]), torch.float32)
# 'patch_embedding.weight': (), # (torch.Size([5120, 36, 1, 2, 2]), torch.bfloat16)
# 'patch_embedding.bias': (), # (torch.Size([5120]), torch.bfloat16)
'condition_embedder.time_embedder.linear_1.weight': ('tp',), # (torch.Size([5120, 256]), torch.float32)
'condition_embedder.time_embedder.linear_1.bias': ('tp',), # (torch.Size([5120]), torch.float32)
'condition_embedder.time_embedder.linear_2.weight': (None, 'tp',), # (torch.Size([5120, 5120]), torch.float32)
# 'condition_embedder.time_embedder.linear_2.bias': (), # (torch.Size([5120]), torch.float32)
# 'condition_embedder.time_proj.weight': (), # (torch.Size([30720, 5120]), torch.bfloat16)
# 'condition_embedder.time_proj.bias': (), # (torch.Size([30720]), torch.bfloat16)
'condition_embedder.text_embedder.linear_1.weight': ('tp',), # (torch.Size([5120, 4096]), torch.bfloat16)
'condition_embedder.text_embedder.linear_1.bias': ('tp',), # (torch.Size([5120]), torch.bfloat16)
'condition_embedder.text_embedder.linear_2.weight': (None, 'tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
# 'condition_embedder.text_embedder.linear_2.bias': (), # (torch.Size([5120]), torch.bfloat16)
# 'blocks.*.scale_shift_table': (), # (torch.Size([1, 6, 5120]), torch.float32)
'blocks.*.attn1.to_q.weight': ('tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
'blocks.*.attn1.to_q.bias': ('tp',), # (torch.Size([5120]), torch.bfloat16)
'blocks.*.attn1.to_k.weight': ('tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
'blocks.*.attn1.to_k.bias': ('tp',), # (torch.Size([5120]), torch.bfloat16)
'blocks.*.attn1.to_v.weight': ('tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
'blocks.*.attn1.to_v.bias': ('tp',), # (torch.Size([5120]), torch.bfloat16)
'blocks.*.attn1.to_out.*.weight': (None, 'tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
# 'blocks.*.attn1.to_out.*.bias': (), # (torch.Size([5120]), torch.bfloat16)
# 'blocks.*.attn1.norm_q.weight': (), # (torch.Size([5120]), torch.bfloat16)
# 'blocks.*.attn1.norm_k.weight': (), # (torch.Size([5120]), torch.bfloat16)
'blocks.*.attn2.to_q.weight': ('tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
'blocks.*.attn2.to_q.bias': ('tp',), # (torch.Size([5120]), torch.bfloat16)
'blocks.*.attn2.to_k.weight': ('tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
'blocks.*.attn2.to_k.bias': ('tp',), # (torch.Size([5120]), torch.bfloat16)
'blocks.*.attn2.to_v.weight': ('tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
'blocks.*.attn2.to_v.bias': ('tp',), # (torch.Size([5120]), torch.bfloat16)
'blocks.*.attn2.to_out.*.weight': (None, 'tp',), # (torch.Size([5120, 5120]), torch.bfloat16)
# 'blocks.*.attn2.to_out.*.bias': (), # (torch.Size([5120]), torch.bfloat16)
# 'blocks.*.attn2.norm_q.weight': (), # (torch.Size([5120]), torch.bfloat16)
# 'blocks.*.attn2.norm_k.weight': (), # (torch.Size([5120]), torch.bfloat16)
# 'blocks.*.norm2.weight': (), # (torch.Size([5120]), torch.float32)
# 'blocks.*.norm2.bias': (), # (torch.Size([5120]), torch.float32)
'blocks.*.ffn.net.*.proj.weight': ('tp',), # (torch.Size([13824, 5120]), torch.bfloat16)
'blocks.*.ffn.net.*.proj.bias': ('tp',), # (torch.Size([13824]), torch.bfloat16)
'blocks.*.ffn.net.*.weight': (None, 'tp',), # (torch.Size([5120, 13824]), torch.bfloat16)
# 'blocks.*.ffn.net.*.bias': (), # (torch.Size([5120]), torch.bfloat16)
# 'proj_out.weight': (), # (torch.Size([64, 5120]), torch.bfloat16)
# 'proj_out.bias': (), # (torch.Size([64]), torch.bfloat16)
# 'rope.freqs_cos': (), # (torch.Size([1024, 128]), torch.float32)
# 'rope.freqs_sin': (), # (torch.Size([1024, 128]), torch.float32)
}

VAE_SHARDINGS = {
# 'encoder.conv_in.weight': (), # (torch.Size([96, 3, 3, 3, 3]), torch.bfloat16)
# 'encoder.conv_in.bias': (), # (torch.Size([96]), torch.bfloat16)
# 'encoder.down_blocks.*.norm1.gamma': (), # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
# 'encoder.down_blocks.*.conv1.weight': (), # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
# 'encoder.down_blocks.*.conv1.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'encoder.down_blocks.*.norm2.gamma': (), # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
# 'encoder.down_blocks.*.conv2.weight': (), # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
# 'encoder.down_blocks.*.conv2.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'encoder.down_blocks.*.resample.*.weight': (), # (torch.Size([384, 384, 3, 3]), torch.bfloat16)
# 'encoder.down_blocks.*.resample.*.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'encoder.down_blocks.*.conv_shortcut.weight': (), # (torch.Size([384, 192, 1, 1, 1]), torch.bfloat16)
# 'encoder.down_blocks.*.conv_shortcut.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'encoder.down_blocks.*.time_conv.weight': (), # (torch.Size([384, 384, 3, 1, 1]), torch.bfloat16)
# 'encoder.down_blocks.*.time_conv.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'encoder.mid_block.attentions.*.norm.gamma': (), # (torch.Size([384, 1, 1]), torch.bfloat16)
# 'encoder.mid_block.attentions.*.to_qkv.weight': (), # (torch.Size([1152, 384, 1, 1]), torch.bfloat16)
# 'encoder.mid_block.attentions.*.to_qkv.bias': (), # (torch.Size([1152]), torch.bfloat16)
# 'encoder.mid_block.attentions.*.proj.weight': (), # (torch.Size([384, 384, 1, 1]), torch.bfloat16)
# 'encoder.mid_block.attentions.*.proj.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'encoder.mid_block.resnets.*.norm1.gamma': (), # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
# 'encoder.mid_block.resnets.*.conv1.weight': (), # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
# 'encoder.mid_block.resnets.*.conv1.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'encoder.mid_block.resnets.*.norm2.gamma': (), # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
# 'encoder.mid_block.resnets.*.conv2.weight': (), # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
# 'encoder.mid_block.resnets.*.conv2.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'encoder.norm_out.gamma': (), # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
# 'encoder.conv_out.weight': (), # (torch.Size([32, 384, 3, 3, 3]), torch.bfloat16)
# 'encoder.conv_out.bias': (), # (torch.Size([32]), torch.bfloat16)
# 'quant_conv.weight': (), # (torch.Size([32, 32, 1, 1, 1]), torch.bfloat16)
# 'quant_conv.bias': (), # (torch.Size([32]), torch.bfloat16)
# 'post_quant_conv.weight': (), # (torch.Size([16, 16, 1, 1, 1]), torch.bfloat16)
# 'post_quant_conv.bias': (), # (torch.Size([16]), torch.bfloat16)
# 'decoder.conv_in.weight': (), # (torch.Size([384, 16, 3, 3, 3]), torch.bfloat16)
# 'decoder.conv_in.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'decoder.mid_block.attentions.*.norm.gamma': (), # (torch.Size([384, 1, 1]), torch.bfloat16)
# 'decoder.mid_block.attentions.*.to_qkv.weight': (), # (torch.Size([1152, 384, 1, 1]), torch.bfloat16)
# 'decoder.mid_block.attentions.*.to_qkv.bias': (), # (torch.Size([1152]), torch.bfloat16)
# 'decoder.mid_block.attentions.*.proj.weight': (), # (torch.Size([384, 384, 1, 1]), torch.bfloat16)
# 'decoder.mid_block.attentions.*.proj.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'decoder.mid_block.resnets.*.norm1.gamma': (), # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
# 'decoder.mid_block.resnets.*.conv1.weight': (), # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
# 'decoder.mid_block.resnets.*.conv1.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'decoder.mid_block.resnets.*.norm2.gamma': (), # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
# 'decoder.mid_block.resnets.*.conv2.weight': (), # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
# 'decoder.mid_block.resnets.*.conv2.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'decoder.up_blocks.*.resnets.*.norm1.gamma': (), # (torch.Size([96, 1, 1, 1]), torch.bfloat16)
# 'decoder.up_blocks.*.resnets.*.conv1.weight': (), # (torch.Size([96, 96, 3, 3, 3]), torch.bfloat16)
# 'decoder.up_blocks.*.resnets.*.conv1.bias': (), # (torch.Size([96]), torch.bfloat16)
# 'decoder.up_blocks.*.resnets.*.norm2.gamma': (), # (torch.Size([96, 1, 1, 1]), torch.bfloat16)
# 'decoder.up_blocks.*.resnets.*.conv2.weight': (), # (torch.Size([96, 96, 3, 3, 3]), torch.bfloat16)
# 'decoder.up_blocks.*.resnets.*.conv2.bias': (), # (torch.Size([96]), torch.bfloat16)
# 'decoder.up_blocks.*.upsamplers.*.resample.*.weight': (), # (torch.Size([96, 192, 3, 3]), torch.bfloat16)
# 'decoder.up_blocks.*.upsamplers.*.resample.*.bias': (), # (torch.Size([96]), torch.bfloat16)
# 'decoder.up_blocks.*.upsamplers.*.time_conv.weight': (), # (torch.Size([768, 384, 3, 1, 1]), torch.bfloat16)
# 'decoder.up_blocks.*.upsamplers.*.time_conv.bias': (), # (torch.Size([768]), torch.bfloat16)
# 'decoder.up_blocks.*.resnets.*.conv_shortcut.weight': (), # (torch.Size([384, 192, 1, 1, 1]), torch.bfloat16)
# 'decoder.up_blocks.*.resnets.*.conv_shortcut.bias': (), # (torch.Size([384]), torch.bfloat16)
# 'decoder.norm_out.gamma': (), # (torch.Size([96, 1, 1, 1]), torch.bfloat16)
# 'decoder.conv_out.weight': (), # (torch.Size([3, 96, 3, 3, 3]), torch.bfloat16)
# 'decoder.conv_out.bias': (), # (torch.Size([3]), torch.bfloat16)
}
# fmt: on


@contextmanager
def perf_time(name: str):
    print(f"{name} start")
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start: .6f}s")


def _print_weights(module):
    def make_key(name):
        return re.sub(r"\.\d+\.", ".*.", name)

    all_buffers = dict(module.named_parameters())
    all_buffers.update(module.named_buffers())
    result = {}
    for k, v in all_buffers.items():
        result[make_key(k)] = (v.shape, v.dtype)
    print("{")
    for k, v in result.items():
        print(f"'{k}': (), # {v}")
    print("}")


def _torch_conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, env
):
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
    return env.j2t_iso(res)


def _overide_op_definition(env, op_to_override, op_impl):
    # Workaround for the function lack is_view_op argument
    # env.override_op_definition(op_to_override, op_impl)
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        op_impl,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )


def _shard_weight_dict(weight_dict, sharding_dict, mesh):
    result = {}
    for k, v in weight_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.to("jax")
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                break
        else:
            # replicate
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))

        result[k] = v
    return result


def _move_module(env, module):
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


class Args(argparse.Namespace):
    size: str
    frame_num: int
    prompt: str
    base_seed: int
    image: str
    sample_steps: int
    print_weights: bool
    profile: str
    profile_output_path: str


def parse_args():
    # Copy args and modify from wan2.2 repo
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="720*1280",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames of video are generated. The number should be 4n+1",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="The prompt to generate the video from.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="The seed to use for generating the video. Need to specify for multi-host sync.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="The image to generate the video from.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=40, help="The sampling steps."
    )
    parser.add_argument(
        "--print_weights", action="store_true", help="print weights in models"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="no",
        choices=["no", "dit", "all"],
        help="no for no profile, dit for dit only 3 steps, all including vae",
    )
    parser.add_argument(
        "--profile_output_path",
        type=str,
        default=DEFAULT_PROFILE_OUT_PATH,
        help="path to save profile output",
    )
    return parser.parse_args(namespace=Args())


def main(args: Args):
    torch.set_default_dtype(torch.bfloat16)

    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    dtype = torch.bfloat16

    with perf_time("load pipe"):
        pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # print weights map for fill sharding
    if args.print_weights:
        print("text_encoder_shardings = ", end="")
        _print_weights(pipe.text_encoder)
        print()
        print("transformer_shardings = ", end="")
        _print_weights(pipe.transformer)
        print()
        print("vae_shardings = ", end="")
        _print_weights(pipe.vae)
        print()

    # enable torchax wrap jax array into torch array
    torchax.enable_globally()
    env = torchax.default_env()

    mesh = jax.make_mesh((len(jax.devices()),), ("tp",))
    # mesh_devices = mesh_utils.create_device_mesh((dp_dim, sp_dim, tp_dim), allow_split_physical_axes=True)
    # mesh = Mesh(mesh_devices, ('dp','sp', axis))

    # register non-jax type
    def _flatten_model_output(obj):
        return obj.to_tuple(), type(obj)

    def _unflatten_model_output(aux, children):
        return aux(*children)

    jax.tree_util.register_pytree_node(
        modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
        _flatten_model_output,
        _unflatten_model_output,
    )

    # Workaround override function to use tpu. Better handle it in torchax
    _overide_op_definition(
        env, torch.nn.functional.conv2d, functools.partial(_torch_conv2d, env=env)
    )

    # Put weights into tpu

    with perf_time("Move model to tpu"):
        with perf_time("  Move text encoder"):
            _move_module(env, pipe.text_encoder)
            pipe.text_encoder = torchax.compile(pipe.text_encoder)
            pipe.text_encoder.params = _shard_weight_dict(
                pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
            )
            pipe.text_encoder.buffers = _shard_weight_dict(
                pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
            )

        transformer_options = torchax.CompileOptions(
            jax_jit_kwargs={"static_argnames": ("return_dict",)}
        )
        with perf_time("  Move transformer"):
            _move_module(env, pipe.transformer)
            pipe.transformer = torchax.compile(pipe.transformer, transformer_options)
            pipe.transformer.params = _shard_weight_dict(
                pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh
            )
            pipe.transformer.buffers = _shard_weight_dict(
                pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh
            )

        with perf_time("  Move transformer2"):
            _move_module(env, pipe.transformer_2)
            pipe.transformer_2 = torchax.compile(
                pipe.transformer_2, transformer_options
            )
            pipe.transformer_2.params = _shard_weight_dict(
                pipe.transformer_2.params, TRANSFORMER_SHARDINGS, mesh
            )
            pipe.transformer_2.buffers = _shard_weight_dict(
                pipe.transformer_2.buffers, TRANSFORMER_SHARDINGS, mesh
            )

        # TODO: RESOURCE_EXHAUSTED while compile vae
        vae_options = torchax.CompileOptions(
            # methods_to_compile=['decode'],
            # jax_jit_kwargs={"static_argnames": ("return_dict",)}
        )
        with perf_time("  Move vae"):
            _move_module(env, pipe.vae)
            pipe.vae = torchax.compile(pipe.vae, vae_options)
            pipe.vae.params = _shard_weight_dict(pipe.vae.params, VAE_SHARDINGS, mesh)
            pipe.vae.buffers = _shard_weight_dict(pipe.vae.buffers, VAE_SHARDINGS, mesh)

    image = load_image(
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    )
    max_area = MAX_AREA_CONFIGS[args.size]
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    prompt = args.prompt
    negative_prompt = DEFAULT_NEG_PROMPT
    generator = torch.Generator().manual_seed(args.base_seed)
    with mesh:
        with perf_time("Warmup and output video"):
            output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=args.frame_num,
                guidance_scale=3.5,
                num_inference_steps=args.sample_steps,
                generator=generator,
            ).frames[0]
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{current_datetime}.mp4"
            export_to_video(output, file_name, fps=16)
            print(f"output video done. {file_name}")

        if args.profile != "no":
            with perf_time("Profile"):
                if args.profile == "dit":
                    output_type = "latent"
                else:
                    output_type = "np"
                with jax.profiler.trace(args.profile_output_path):
                    output = pipe(
                        image=image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=args.frame_num,
                        guidance_scale=3.5,
                        num_inference_steps=3,
                        generator=generator,
                        output_type=output_type,
                    ).frames[0]

        with perf_time("Benchmark"):
            output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=args.frame_num,
                guidance_scale=3.5,
                num_inference_steps=args.sample_steps,
                generator=generator,
            ).frames[0]

    print("Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
