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


# Helper to pad to next multiple
def pad_to_multiple(x, multiple, axis):
    seq_len = x.shape[axis]
    pad_len = (multiple - seq_len % multiple) % multiple
    if pad_len == 0:
        return x, seq_len
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), seq_len


@functools.partial(
    jax.jit,
    static_argnames=("mesh", "bqsize", "bkvsize", "bkvcomputesize", "bkvcomputesinize"),
)
def _tpu_custom_attention(
    query,
    key,
    value,
    mesh,
    bqsize,
    bkvsize,
    bkvcomputesize,
    bkvcomputesinize,
    scale=None,
):
    # The function that will be sharded across devices.
    def _attention_on_slices(q, k, v):
        # Scale the query tensor. This happens on each device with its slice of data.
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        # fuse the ops of exp in softmax here
        _LOG2_E = 1.44269504
        q = q * scale_factor * _LOG2_E

        # This function operates on a single item from the batch.
        def kernel_3d(q_3d, k_3d, v_3d):
            q_seq_len = q_3d.shape[1]
            kv_seq_len = k_3d.shape[1]
            num_heads_on_device = q_3d.shape[0]

            block_sizes = splash_attention.BlockSizes(
                block_q=min(bqsize, q_seq_len),
                block_kv=min(bkvsize, kv_seq_len),
                block_kv_compute=min(bkvcomputesize, kv_seq_len),
            )
            splash_kernel = custom_splash_attention.make_splash_mha(
                block_sizes=block_sizes, bkv_compute_in=bkvcomputesinize
            )
            out = splash_kernel(q_3d, k_3d, v_3d).astype(q_3d.dtype)
            out = jnp.swapaxes(out, 1, 2)
            return out

        # Map the kernel over the batch dimension.
        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # print(f"[DEBUG] {query.shape=}, {key.shape=}")
    if key.shape[0] > 1:
        dp_mesh_key = "dp"
        remain_mesh_key = ("tp",)
    else:
        dp_mesh_key = None
        remain_mesh_key = ("dp", "tp")
    # print(f"[DEBUG] {dp_mesh_key=}, {remain_mesh_key=}")
    remain_devices_prod = 1
    for d in remain_mesh_key:
        remain_devices_prod *= mesh.axis_sizes[mesh.axis_names.index(d)]

    q_num_head = query.shape[1]
    q_seq_len = query.shape[2]
    kv_num_head = key.shape[1]
    kv_seq_len = key.shape[2]

    q_partition_spec = P("dp", "tp", None, None)
    kv_partition_spec = P("dp", "tp", None, None)

    # ALWAYS use shard_map. The partition_spec will control the behavior.
    sharded_fn = jax.shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_vma=False,
    )
    query = jax.lax.with_sharding_constraint(query, P("dp", None, "tp", None))
    key = jax.lax.with_sharding_constraint(key, P("dp", None, "tp", None))
    value = jax.lax.with_sharding_constraint(value, P("dp", None, "tp", None))
    out = sharded_fn(query, key, value)
    # Remove the potential padding for sp
    out = out[:, :, :q_seq_len, :]
    out = jax.lax.with_sharding_constraint(out, P("dp", None, "tp", None))
    return out


def main():
    query = jnp.ones((2, 40, 75600, 128))
    key = jnp.ones((2, 40, 75600, 128))
    value = jnp.ones((2, 40, 75600, 128))

    # bqsizes = (1512,)

    # bqsizes = (600, 630, 675, 700, 720, 756, 840, 900, 945, 1008, 1050, 1080, 1200, 1260, 1350, 1400, 1512, 1575, 1680, 1800, 1890, 2100, 2160, 2520, 2700, 2800, 3024, 3150, 3600, 3780, 4200)
    bqsizes = range(2560, 4096, 256)
    bkvsizes = range(2560, 4096, 256)
    bkvcomputesizes = range(256, 4096, 256)
    # bkvcomputesizes = (256,)
    # bkvcomputesinizes = range(64, 4096, 64)
    bkvcomputesinizes = range(256, 4096, 256)
    # bkvcomputesinizes = (256,)

    # bqsizes = list(range(512, 4096, 128))
    # bkvsizes = (3072,)
    # bkvcomputesizes = (1024,)

    # BQSIZE =  2816 # 2240 # 3024 #2520
    # BKVSIZE = 3840
    # BKVCOMPUTESIZE = 256

    # bqsizes = (512,)
    # bkvsizes = (2048,)
    # bkvcomputesizes = (256,)

    tp_dim = jax.device_count() // 2
    dp_dim = 2
    print("bqsize, bkvsize, bkvcomputesize, time (s)")
    while tp_dim >= 1:
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim),
            allow_split_physical_axes=True,
        )
        mesh = Mesh(mesh_devices, ("dp", "tp"))

        query = jax.device_put(query, NamedSharding(mesh, P("dp", None, ("tp",), None)))
        key = jax.device_put(key, NamedSharding(mesh, P("dp", None, ("tp",), None)))
        value = jax.device_put(value, NamedSharding(mesh, P("dp", None, ("tp",), None)))
        with mesh:
            for bqsize in bqsizes:
                for bkvsize in bkvsizes:
                    for bkvcomputesize in bkvcomputesizes:
                        for bkvcomputesinize in bkvcomputesinizes:
                            if (
                                bkvsize < bkvcomputesize
                                or bkvsize % bkvcomputesize != 0
                            ):
                                continue

                            if (
                                bkvcomputesize < bkvcomputesinize
                                or bkvcomputesize % bkvcomputesinize != 0
                            ):
                                continue

                            try:
                                jax.block_until_ready(
                                    _tpu_custom_attention(
                                        query,
                                        key,
                                        value,
                                        mesh,
                                        bqsize,
                                        bkvsize,
                                        bkvcomputesize,
                                        bkvcomputesinize,
                                    )
                                )

                                start = time.perf_counter()
                                jax.block_until_ready(
                                    _tpu_custom_attention(
                                        query,
                                        key,
                                        value,
                                        mesh,
                                        bqsize,
                                        bkvsize,
                                        bkvcomputesize,
                                        bkvcomputesinize,
                                    )
                                )
                                end = time.perf_counter()
                                print(
                                    f"{bqsize}, {bkvsize}, {bkvcomputesize}, {bkvcomputesinize}, {end - start}"
                                )
                            except KeyboardInterrupt:
                                raise
                            except Exception:
                                # raise
                                continue
        break
        # smaller sp_dim better
        tp_dim //= 2
        sp_dim *= 2


if __name__ == "__main__":
    main()
