"""Harmonic field verification via property checking"""
import torch as th, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harmonic_field import *

results = []
def verify(name, cond): results.append((name, cond)); print(f"{'✓' if cond else '✗'} {name}")

# BiChannel mechanics
ax, ay = th.randn(7,9), th.randn(7,9)
bic = BiChannel(ax, ay)
verify("BiChannel shape", bic.dimensions == (7,9))
mag, ang = bic.extract_polar()
rebuilt = BiChannel.build_polar(mag, ang)
verify("Polar roundtrip", th.allclose(bic.axis_x, rebuilt.axis_x, atol=1e-4))
p1, p2 = BiChannel(th.randn(4), th.randn(4)), BiChannel(th.randn(4), th.randn(4))
prod = p1.combine_with(p2)
verify("Multiplication output", prod.dimensions == (4,))
stacked = wave_merge([bic, rebuilt])
verify("Wave merge", stacked.dimensions == bic.dimensions)

# Phase transformations
transform = BiMatrix(11, 13)
tin = BiChannel(th.randn(5,11), th.randn(5,11))
tout = transform(tin)
verify("BiMatrix dims", tout.dimensions == (5,13))
normalizer = TwoPathNorm(16)
normalizer.train()
nin = BiChannel(th.randn(30,16), th.randn(30,16))
nout = normalizer(nin)
verify("Normalization", nout.dimensions == nin.dimensions)
dropper = MagPreserveDropout(0.4)
dropper.eval()
din = BiChannel(th.ones(10,8), th.ones(10,8))
dout = dropper(din)
verify("Dropout eval mode", th.equal(din.axis_x, dout.axis_x))
gtest = BiChannel(th.randn(6), th.randn(6))
gated = phase_gate(gtest, shift=0.5)
_, ang1 = gtest.extract_polar()
_, ang2 = gated.extract_polar()
verify("Phase gate angle", th.allclose(ang1, ang2, atol=1e-3))

# Circular binding
u, v = th.randn(88), th.randn(88)
uv = twist_merge(u, v)
vu = twist_merge(v, u)
verify("Twist commute", th.allclose(uv, vu, atol=1e-4))
unbound = twist_split(uv, v)
corr = (u*unbound).sum()/(th.norm(u)*th.norm(unbound)+1e-9)
verify("Unbind quality", corr > 0.75)
auv = asymm_twist(u, v)
avu = asymm_twist(v, u)
verify("Asymm non-commute", not th.allclose(auv, avu, atol=0.1))
membank = CircularMemoryBank(72)
k1, v1 = th.randn(72), th.randn(72)
membank.store(k1, v1)
r1 = membank.retrieve(k1)
msim = (v1*r1).sum()/(th.norm(v1)*th.norm(r1)+1e-9)
verify("Memory retrieval", msim > 0.6)

# Attention layers
scorer = BiPhaseScorer(40, head_count=5)
qin = BiChannel(th.randn(3,8,40), th.randn(3,8,40))
sout = scorer(qin, qin, qin)
verify("Attention shape", sout.dimensions == qin.dimensions)
jumper = UnexpectedJump(28)
jout, jmask = jumper(th.randn(4,10,28), enable_jump=False)
verify("Jump output", jout.shape == (4,10,28))
poser = GeometricPosition(35, 24)
penc = poser(18)
verify("Position encoding", penc.dimensions == (18,24))

# Broadcasting
hub = BroadcastHub(hub_size=56, specialist_count=6)
hin = th.randn(7,56)
hout, hlev, _ = hub(hin)
verify("Hub output", hout.shape == hin.shape)
verify("Hub level type", isinstance(hlev, BroadcastLevel))
router = SpecialistRouter(22, 33, 4)
rout, rwts = router(th.randn(6,22))
verify("Router output", rout.shape == (6,33))
wsum = rwts.sum(dim=1)
verify("Router weights", th.allclose(wsum, th.ones(6), atol=1e-4))

# Energy dynamics
predictor = DescentPredictor(14, 9, 44, 2)
xinp = th.randn(3,14)
yinf, hist = predictor.infer_output(xinp, iterations=12)
verify("Inference shape", yinf.shape == (3,9))
verify("Energy descent", hist[-1] < hist[0])
cascader = CascadeSystem(85)
cout, csize = cascader(th.randn(2,85), track=True)
verify("Cascade output", cout.shape == (2,85))
verify("Avalanche occurred", csize > 0)

# Full integration
for cfg in ["balanced", "compact", "large", "energy", "broadcast"]:
    fsys = build_system(cfg, input_size=19, output_size=7)
    verify(f"Config {cfg}", isinstance(fsys, FullHarmonicSystem))
model = build_system("compact", input_size=15, output_size=6)
sout, _ = model(th.randn(4,15))
verify("Model single", sout.shape == (4,6))
qout, _ = model(th.randn(2,9,15))
verify("Model sequence", qout.shape == (2,9,6))
model.train()
gx = th.randn(3,15, requires_grad=True)
model(gx)[0].sum().backward()
verify("Gradient flow", gx.grad is not None and gx.grad.abs().sum() > 0)

# Learning capacity
learner = build_system("compact", input_size=2, output_size=1)
optim = th.optim.Adam(learner.parameters(), lr=0.05)
xdat = th.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
ydat = th.tensor([[0.],[1.],[1.],[0.]])
loss0 = ((learner(xdat)[0]-ydat)**2).mean().item()
for _ in range(28):
    optim.zero_grad()
    ((learner(xdat)[0]-ydat)**2).mean().backward()
    optim.step()
lossf = ((learner(xdat)[0]-ydat)**2).mean().item()
# Check both loss reduction and prediction accuracy
preds = learner(xdat)[0]
correct = ((preds > 0.5).float() == ydat).float().mean().item()
verify("XOR learning", lossf < loss0*0.8 and correct >= 0.5)

passed = sum(1 for _,c in results if c)
print(f"\n{passed}/{len(results)} verifications passed")
sys.exit(0 if passed == len(results) else 1)
