


def estimate_layer_flops(
    seq_len: int,
    n_embd: int,
):
    C = n_embd
    T = seq_len

    qkv = 6 * T * C * C
    attn = 4 * T * T * C
    out_proj = 2 * T * C * C
    ffn = 16 * T * C * C

    total = qkv + attn + out_proj + ffn

    return {
        "qkv": qkv,
        "attention": attn,
        "out_proj": out_proj,
        "ffn": ffn,
        "total": total,
    }


def estimate_decode_layer_flops(
    seq_len: int,
    n_embd: int,
):
    C = n_embd
    T = seq_len

    qkv = 6 * C * C
    attn = 4 * T * C
    out_proj = 2 * C * C
    ffn = 16 * C * C

    total = qkv + attn + out_proj + ffn

    return {
        "qkv": qkv,
        "attention": attn,
        "out_proj": out_proj,
        "ffn": ffn,
        "total": total,
    }


def estimate_kv_cache_bytes(
    seq_len: int,
    n_layer: int,
    n_embd: int,
    batch_size: int = 1,
    bytes_per_element: int = 4,
):
    return (
        2 * batch_size * n_layer * seq_len * n_embd * bytes_per_element
    )


def mib(x):
    return x / (1024 ** 2)


def main():
    for T in [8, 16, 32, 64, 128, 256]:
        naive = estimate_layer_flops(T, n_embd=512)
        cached = estimate_decode_layer_flops(T, n_embd=512)

        kv_bytes = estimate_kv_cache_bytes(
            seq_len=T,
            n_layer=8,
            n_embd=512,
            batch_size=1,
            bytes_per_element=4,
        )

        print(
            f"T={T:5d} "
            f"naive={naive['total'] / 1e6:10.2f} MFLOPs "
            f"cached={cached['total'] / 1e6:10.2f} MFLOPs "
            f"ratio={naive['total'] / cached['total']:8.2f}x "
            f"KV Cache={mib(kv_bytes):10.2f} MiB"
        )


if __name__ == "__main__":
    main()