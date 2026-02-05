"""Holographic memory via circular convolution"""
import torch as T, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harmonic_field.circular_ops import twist_merge, twist_split, asymm_twist, CircularMemoryBank, chain_twist

T.manual_seed(9876)
DIM = 288

print("\n[BIND & RECOVER]")
alpha, beta = T.randn(DIM), T.randn(DIM)
fused = twist_merge(alpha, beta)
retrieved = twist_split(fused, beta)
match = (alpha*retrieved).sum()/(T.norm(alpha)*T.norm(retrieved)+1e-9)
print(f"Recovery: {match:.3f} {'✓good' if match>0.78 else '✗poor'}")
wrong_key = twist_split(fused, T.randn(DIM))
wrong_match = (alpha*wrong_key).sum()/(T.norm(alpha)*T.norm(wrong_key)+1e-9)
print(f"Wrong key: {wrong_match:.3f} {'✓rejected' if abs(wrong_match)<0.35 else '✗problem'}")

print("\n[SUPERPOSITION STORAGE]")
storage = CircularMemoryBank(DIM)
associations = {f"K{i}": (T.randn(DIM), T.randn(DIM)) for i in range(6)}
for label, (k, v) in associations.items():
    storage.store(k, v)
print(f"Capacity: {storage.memory_magnitude():.1f}")
test_key, test_val = associations["K2"]
fetched = storage.retrieve(test_key)
fetch_quality = (test_val*fetched).sum()/(T.norm(test_val)*T.norm(fetched)+1e-9)
print(f"Fetch K2: {fetch_quality:.3f} {'✓' if fetch_quality>0.58 else '✗'}")

print("\n[SEQUENTIAL ENCODING]")
token_seq = [T.randn(DIM) for _ in range(5)]
encoded_seq = chain_twist(token_seq)
print(f"Sequence norm: {T.norm(encoded_seq):.2f}")
reconstruction = encoded_seq
for pos in range(4, -1, -1):
    matches = [(tok*reconstruction).sum()/(T.norm(tok)*T.norm(reconstruction)+1e-9) for tok in token_seq]
    best = max(range(5), key=lambda j: matches[j])
    print(f"Slot {pos}: match token_{best} conf={matches[best]:.3f}")
    if pos > 0:
        reconstruction = twist_split(reconstruction, token_seq[pos])

print("\n[DIRECTIONAL VS SYMMETRIC]")
s1, s2 = T.randn(DIM), T.randn(DIM)
sym_forward = twist_merge(s1, s2)
sym_reverse = twist_merge(s2, s1)
sym_corr = (sym_forward*sym_reverse).sum()/(T.norm(sym_forward)*T.norm(sym_reverse)+1e-9)
print(f"Symmetric: {sym_corr:.3f} {'✓commutes' if sym_corr>0.94 else '✗'}")
asym_forward = asymm_twist(s1, s2)
asym_reverse = asymm_twist(s2, s1)
asym_corr = (asym_forward*asym_reverse).sum()/(T.norm(asym_forward)*T.norm(asym_reverse)+1e-9)
print(f"Asymmetric: {asym_corr:.3f} {'✓non-commute' if asym_corr<0.68 else '✗'}")

print("\n[RELATIONAL STRUCTURE]")
anchor = T.randn(DIM)
nearby = anchor + T.randn(DIM)*0.11
distant = T.randn(DIM)
context = T.randn(DIM)
before_near = (anchor*nearby).sum()/(T.norm(anchor)*T.norm(nearby)+1e-9)
before_far = (anchor*distant).sum()/(T.norm(anchor)*T.norm(distant)+1e-9)
bound_anchor = twist_merge(anchor, context)
bound_nearby = twist_merge(nearby, context)
bound_distant = twist_merge(distant, context)
after_near = (bound_anchor*bound_nearby).sum()/(T.norm(bound_anchor)*T.norm(bound_nearby)+1e-9)
after_far = (bound_anchor*bound_distant).sum()/(T.norm(bound_anchor)*T.norm(bound_distant)+1e-9)
print(f"Near: {before_near:.3f} → {after_near:.3f} (delta={abs(after_near-before_near):.3f})")
print(f"Far: {before_far:.3f} → {after_far:.3f} (delta={abs(after_far-before_far):.3f})")
preserved = abs(after_near-before_near)<0.28 and abs(after_far-before_far)<0.28
print(f"Structure: {'✓preserved' if preserved else '✗changed'}")

print("\n✓ Demonstrations complete")
