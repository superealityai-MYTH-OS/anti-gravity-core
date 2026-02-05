"""Train harmonic field on annular pattern recognition"""
import torch as T, torch.nn as N, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harmonic_field import build_system

def make_rings(ct=450, bands=4, fuzz=0.09):
    coords, tags = [], []
    for b in range(bands):
        radius = 0.6 + b*0.45
        theta = T.rand(ct)*6.283
        r_actual = radius + T.randn(ct)*fuzz
        coords.append(T.stack([r_actual*T.cos(theta), r_actual*T.sin(theta)], 1))
        tags.append(T.full((ct,), b, dtype=T.long))
    shuffle = T.randperm(ct*bands)
    return T.cat(coords)[shuffle], T.cat(tags)[shuffle]

hw = T.device("cuda" if T.cuda.is_available() else "cpu")
print(f"Hardware: {hw}")

x_learn, y_learn = make_rings(380, 4, 0.085)
x_check, y_check = make_rings(95, 4, 0.085)
x_learn, y_learn = x_learn.to(hw), y_learn.to(hw)
x_check, y_check = x_check.to(hw), y_check.to(hw)

net = build_system("compact", input_size=2, output_size=4).to(hw)
updater = T.optim.Adam(net.parameters(), lr=0.006)
criterion = N.CrossEntropyLoss()

print(f"Learn: {len(x_learn)}, Check: {len(x_check)}, Params: {sum(p.numel() for p in net.parameters()):,}")

peak = 0
for round in range(1, 46):
    net.train()
    order = T.randperm(len(x_learn))
    batch_losses, batch_accs = [], []
    for i in range(0, len(x_learn), 36):
        indices = order[i:i+36]
        updater.zero_grad()
        forecast, _ = net(x_learn[indices])
        err = criterion(forecast, y_learn[indices])
        err.backward()
        updater.step()
        batch_losses.append(err.item())
        batch_accs.append((forecast.argmax(1)==y_learn[indices]).float().mean().item())
    
    net.eval()
    with T.no_grad():
        check_forecast, _ = net(x_check)
        check_err = criterion(check_forecast, y_check).item()
        check_acc = (check_forecast.argmax(1)==y_check).float().mean().item()*100
    
    if check_acc > peak:
        peak = check_acc
        flag = "★"
    else:
        flag = " "
    
    if round % 5 == 0 or round == 1:
        avg_loss = sum(batch_losses)/len(batch_losses)
        avg_acc = sum(batch_accs)/len(batch_accs)*100
        print(f"R{round:2} L:{avg_loss:.3f} A:{avg_acc:.1f}% | CL:{check_err:.3f} CA:{check_acc:.1f}% {flag}")

print(f"\nPeak: {peak:.1f}%")

net.eval()
with T.no_grad():
    _, info = net(x_check[:25], return_diagnostics=True)
    print(f"Broadcasts: {info['broadcast_count']}, Magnitude: {info['magnitude_mean']:.3f}")
