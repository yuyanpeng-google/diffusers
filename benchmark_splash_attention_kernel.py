import functools
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import jax
import jax.numpy as jnp
import math
import time

# import ringattention_pallas_tpu_splash
import custom_splash_attention


# Copy from wan_tx_splash_attn.py
@functools.partial(jax.jit, static_argnames=("mesh", "bqsize", "bkvsize", "bkvcomputesize"))
def _tpu_splash_attention(
    query,
    key,
    value,
    mesh,
    bqsize,
    bkvsize,
    bkvcomputesize,
    scale=None,
    is_causal=False,
    window_size=None,
):
    num_heads = query.shape[1]

    # The function that will be sharded across devices.
    def _attention_on_slices(q, k, v):

        # Scale the query tensor. This happens on each device with its slice of data.
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        def pad_to_multiple2(x, multiple, axis):
            # For try pad outside
            return x, x.shape[axis]
        # Helper to pad to next multiple
        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        # This function operates on a single item from the batch.
        def kernel_3d(q_3d, k_3d, v_3d):
            q_seq_len = q_3d.shape[1]
            kv_seq_len = k_3d.shape[1]
            num_heads_on_device = q_3d.shape[0]

            # Pad q, k, v to next multiple of BQSIZE/BKVSIZE
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, bqsize, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, bkvsize, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, bkvsize, axis=1)
            
            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            block_sizes = splash_attention.BlockSizes(
                block_q=min(bqsize, padded_q_seq_len),
                block_kv=min(bkvsize, padded_kv_seq_len),
                block_kv_compute=min(bkvcomputesize, padded_kv_seq_len),
            )
            splash_kernel = custom_splash_attention.make_splash_mha(
                block_sizes=block_sizes,
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            # Remove padding if any
            out = jnp.swapaxes(out, 1, 2)
            return out[:, :q_orig_len, ...]

        # Map the kernel over the batch dimension.
        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # Determine the partitioning spec based on the number of heads.
    if num_heads < mesh.size:
        # Replicated case for VAE. All devices get the full tensor.
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        # Sharded case for Transformer. Split along the heads axis.
        # Attn1 self attention, key length is long.
        if key.shape[2] > 10000:
            q_partition_spec = P("dp", "axis", "sp", None)
            kv_partition_spec = P("dp", "axis", None, None)
        else:
            # Attn2 which is cross attention, kv sequence is shorter. All gather the key value cost less.
            q_partition_spec = P("dp", None, ("axis", "sp"), None)
            kv_partition_spec = P("dp", None, None, None)

    # ALWAYS use shard_map. The partition_spec will control the behavior.
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value)
    out = jax.lax.with_sharding_constraint(out, P("dp", None, ("axis", "sp"), None))
    return out


def main():
    query = jnp.ones((1, 40, 75600, 128))
    key = jnp.ones((1, 40, 75600, 128))
    value = jnp.ones((1, 40, 75600, 128))

    bqsizes = (1512,)

    # bqsizes = (600, 630, 675, 700, 720, 756, 840, 900, 945, 1008, 1050, 1080, 1200, 1260, 1350, 1400, 1512, 1575, 1680, 1800, 1890, 2100, 2160, 2520, 2700, 2800, 3024, 3150, 3600, 3780, 4200)
    bqsizes = range(2048, 4096, 256)
    bkvsizes = range(2048, 4096, 256)
    bkvcomputesizes = range(256, 2048, 256)
    
    # bqsizes = list(range(512, 4096, 128))
    # bkvsizes = (3072,)
    # bkvcomputesizes = (1024,)

    # BQSIZE =  2816 # 2240 # 3024 #2520
    # BKVSIZE = 3840
    # BKVCOMPUTESIZE = 256

    # bqsizes = (2048,)
    # bkvsizes = (2048,)
    # bkvcomputesizes = (2048,)

    tp_dim = jax.device_count()
    dp_dim = 1
    sp_dim = 1
    print("sp, bqsize, bkvsize, bkvcomputesize, time (s), padded_key_size")
    while tp_dim >= 1:
        mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
        mesh = Mesh(mesh_devices, ('axis','dp','sp'))

        query = jax.device_put(query, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))
        key = jax.device_put(key, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))
        value = jax.device_put(value, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))
        with mesh:
            for bqsize in bqsizes:
                for bkvsize in bkvsizes:
                    for bkvcomputesize in bkvcomputesizes:
                        if bkvsize < bkvcomputesize or bkvsize % bkvcomputesize != 0:
                            continue

                        try:
                            # pad key value
                            def pad_to_multiple(x, multiple, axis):
                                # Pad in kernel
                                return x
                                seq_len = x.shape[axis]
                                pad_len = (multiple - seq_len % multiple) % multiple
                                if pad_len == 0:
                                    return x
                                pad_width = [(0, 0)] * x.ndim
                                pad_width[axis] = (0, pad_len)
                                return jnp.pad(x, pad_width)

                            padded_query = pad_to_multiple(query, bqsize, axis=2)
                            padded_key = pad_to_multiple(key, bkvsize, axis=2)
                            padded_value = pad_to_multiple(value, bkvsize, axis=2)

                            jax.block_until_ready(
                                _tpu_splash_attention(padded_query, padded_key, padded_value, mesh, bqsize, bkvsize, bkvcomputesize)
                            )

                            start = time.perf_counter()
                            jax.block_until_ready(
                                _tpu_splash_attention(padded_query, padded_key, padded_value, mesh, bqsize, bkvsize, bkvcomputesize)
                            )
                            end = time.perf_counter()
                            print(f"{sp_dim=}, {bqsize}, {bkvsize}, {bkvcomputesize}, {end - start}, {padded_key.shape[2]}")
                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            continue
        break
        # smaller sp_dim better
        tp_dim //= 2
        sp_dim *= 2

if __name__ == "__main__":
    main()
