import functools
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
import jax
import math
import jax.numpy as jnp
import numpy as np
import dataclasses
import enum
from typing import Any
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import lax

partial = functools.partial
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))

class _QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()
  SEQ_MINOR = enum.auto()

def _from_head_minor(vals: tuple[Any, ...], layout: _QKVLayout):
  if layout == _QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])

@dataclasses.dataclass(frozen=True, slots=True)
class _BlockSizes:
  block_q: int
  block_kv: int
  block_kv_compute: int | None = None
  q_layout: _QKVLayout = _QKVLayout.HEAD_DIM_MINOR
  k_layout: _QKVLayout = _QKVLayout.HEAD_DIM_MINOR
  v_layout: _QKVLayout = _QKVLayout.HEAD_DIM_MINOR

  def __post_init__(self):
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)

def _flash_attention_kernel(
    q_ref,
    k_ref,
    v_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_LANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_LANES}")

  h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  def body(kv_compute_index, _):
    with jax.named_scope("qk"):
      slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
      m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
      q = q_ref[...]
      k = k_ref[slice_k, :]
      qk = lax.dot_general(q, k, NT_DIM_NUMBERS, preferred_element_type=float32)
    with jax.named_scope("softmax"):
      m_curr = qk.max(axis=-1)[:, None]
      m_next = jnp.maximum(m_prev, m_curr)
      bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
      if rem != 0:
        raise NotImplementedError(f"{bkv_compute=} should be a multiple of {NUM_LANES}")
      s_curr = jnp.exp(qk - pltpu.repeat(m_next, bkv_repeats, axis=1))
      l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
      alpha = jnp.exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
    with jax.named_scope("qkv"):
      m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
      v = v_ref[slice_k, :].astype(float32)
      o_curr = lax.dot_general(s_curr, v, NN_DIM_NUMBERS)
      alpha_o = pltpu.repeat(alpha, head_dim_v_repeats, axis=1)
      o_scratch_ref[:] = alpha_o * o_scratch_ref[:] + o_curr

  lax.fori_loop(0, bkv // bkv_compute, body, None, unroll=True)

  @pl.when(j == grid_width - 1)
  def end():
    l = l_scratch_ref[...]
    l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=1)
    o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)

def __splash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    interpret: bool = False,
):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  num_kv_heads = k.shape[0]
  kv_seq_len = k.shape[1]
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  def q_index_map(h, i, j, *_):
    return (h, i, 0)
  def out_index_map(h, i, j, *_):
    return h, i, 0
  def k_index_map(h, i, j, *_):
    return (h // q_heads_per_kv_head, j, 0)
  def v_index_map(h, i, j, *_):
    return (h // q_heads_per_kv_head, j, 0)

  in_specs = [
      pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((None, bkv, head_dim_qk), k_index_map),
      pl.BlockSpec((None, bkv, head_dim_v), v_index_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),
      jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),
      jax.ShapeDtypeStruct((bq, head_dim_v), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((bq, NUM_LANES), lambda *_: (0, 0)),
      pl.BlockSpec((bq, NUM_LANES), lambda *_: (0, 0)),
      pl.BlockSpec((bq, head_dim_v), lambda *_: (0, 0)),
      pl.BlockSpec((None, bq, head_dim_v), out_index_map),
  ]
  grid_width = kv_seq_len // bkv
  grid = (num_q_heads, q_seq_len // bq, grid_width)
  
  all_out = pl.pallas_call(
      partial(
          _flash_attention_kernel,
          mask_value=DEFAULT_MASK_VALUE,
          grid_width=grid_width,
          bq=bq,
          bkv=bkv,
          bkv_compute=bkv_compute,
          head_dim_v=head_dim_v,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
      out_shape=out_shapes,
      interpret=interpret,
  )(q, k, v)
  return all_out[-1]

def make_splash_mha(
    block_sizes: _BlockSizes,
    interpret: bool = False,
):
  def _splash_attention(q: jax.Array, k: jax.Array, v: jax.Array):
    return __splash_attention_forward(q, k, v, block_sizes, interpret)
  return _splash_attention


BQSIZE = BKVSIZE = BKVCOMPUTESIZE = 1024

sharded_fn = None

def _tpu_splash_attention(query, key, value, mesh):
    global sharded_fn
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1])
        q = q * scale_factor

        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)
            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            block_sizes = _BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            splash_kernel = make_splash_mha(block_sizes=block_sizes)
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            return out[:, :q_orig_len, ...]

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    if sharded_fn is None:
        q_partition_spec = P('dp', 'axis', None, None)
        kv_partition_spec = P('dp', 'axis', None, None)
        sharded_fn = jax.jit(shard_map(
            _attention_on_slices,
            mesh=mesh,
            in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
            out_specs=q_partition_spec,
            check_rep=False,
        ))
    out = sharded_fn(query, key, value)
    return jax.lax.with_sharding_constraint(out, P('dp', None, 'axis', None))

if __name__ == "__main__":
  import os
  os.environ["LIBTPU_INIT_ARGS"] = "--xla_enable_transpose_trace"

  shape = (1, 40, 75600, 128)
  q = jnp.arange(np.prod(shape), dtype=jnp.bfloat16).reshape(*shape)
  k = jnp.arange(np.prod(shape), dtype=jnp.bfloat16).reshape(*shape)
  v = jnp.arange(np.prod(shape), dtype=jnp.bfloat16).reshape(*shape)

  mesh = jax.make_mesh((len(jax.devices()), 1, 1), ('axis', 'dp', 'sp'))
  q = jax.device_put(q, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))
  k = jax.device_put(k, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))
  v = jax.device_put(v, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))

  with mesh:
    output = _tpu_splash_attention(q,k,v,mesh)
  output.block_until_ready()

  with mesh:
      with jax.profiler.trace("/dev/shm/tensorboard"):
          output = _tpu_splash_attention(q,k,v,mesh)
          output.block_until_ready()

  import time
  with mesh:
      num_time = 50
      start_time = time.time()
      for _ in range(num_time):
          output = _tpu_splash_attention(q,k,v,mesh)
      output.block_until_ready()
      end_time = time.time()
      print(f"{(end_time-start_time)/num_time}")
