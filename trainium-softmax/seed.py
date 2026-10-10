import nki
import nki.language as nl

@nki.jit
def fused_softmax(x, mask):
    rows, cols = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for start in nl.sequential_range(rows):
        a = nl.load(x[start:start+1, :], dtype=nl.float32)
        m = nl.load(mask[start:start+1, :])
        v = nl.add(a, m)
        v = nl.subtract(v, nl.max(v, axis=1, keepdims=True))
        e = nl.exp(v)
        y = nl.multiply(e, nl.reciprocal(nl.sum(e, axis=1, keepdims=True)))
        nl.store(out[start:start+1, :], y)
    return out
