"""Studi kasus: klasifikasi driver deforestasi (Random Forest vs CNN) + uji ketahanan terhadap awan.

Jalankan:  python src/train.py --n 6000 --epochs 15
"""
import argparse, json, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from data import make_dataset, standardize, CLASSES
from features import handcrafted

BANDS = {"S1+S2": slice(0, 6), "S2 saja": slice(0, 4), "S1 saja": slice(4, 6)}


def evaluate(y, p):
    return {"accuracy": round(accuracy_score(y, p), 4),
            "macro_f1": round(f1_score(y, p, average="macro"), 4)}


def train_cnn(Xtr, ytr, Xva, yva, epochs, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    c = Xtr.shape[1]
    net = nn.Sequential(
        nn.Conv2d(c, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, len(CLASSES)))
    freq = np.bincount(ytr, minlength=len(CLASSES)) / len(ytr)
    w = torch.tensor(1 / np.sqrt(freq), dtype=torch.float32); w /= w.mean()  # bobot kelas (imbalance)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    best, best_state = -1, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); loss_fn(net(Xt[idx]), yt[idx]).backward(); opt.step()
        f1 = f1_score(yva, predict(net, Xva), average="macro")
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  epoch {ep + 1:2d}/{epochs}  val macro-F1 = {f1:.3f}")
    net.load_state_dict(best_state)
    return net


def predict(net, X):
    import torch
    net.eval()
    with torch.no_grad():
        return torch.cat([net(torch.tensor(X[i:i + 256])).argmax(1) for i in range(0, len(X), 256)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--skip-cnn", action="store_true")
    a = ap.parse_args()

    # train/val: 30% sampel berawan (cloud augmentation); test: bersih vs 100% berawan
    X, y = make_dataset(a.n, cloud_prob=0.3, seed=1)
    Xte_clear, yte = make_dataset(a.n // 3, cloud_prob=0.0, seed=2)
    Xte_cloud, _ = make_dataset(a.n // 3, cloud_prob=1.0, seed=2)  # seed sama -> label sama
    k = int(0.8 * len(X))
    Xtr, ytr, Xva, yva = X[:k], y[:k], X[k:], y[k:]
    Xtr_s, m, s = standardize(Xtr)
    Xva_s, Xc_s, Xk_s = standardize(Xva, m, s)[0], standardize(Xte_clear, m, s)[0], standardize(Xte_cloud, m, s)[0]

    results = {}
    print("== Baseline: Random Forest (fitur manual) ==")
    rf = RandomForestClassifier(300, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(handcrafted(Xtr), ytr)
    results["RF (S1+S2)"] = {"clear": evaluate(yte, rf.predict(handcrafted(Xte_clear))),
                             "cloudy": evaluate(yte, rf.predict(handcrafted(Xte_cloud)))}
    print(results["RF (S1+S2)"])

    if not a.skip_cnn:
        for name, sl in BANDS.items():
            print(f"== CNN ({name}) ==")
            net = train_cnn(Xtr_s[:, sl], ytr, Xva_s[:, sl], yva, a.epochs)
            pc, pk = predict(net, Xc_s[:, sl]), predict(net, Xk_s[:, sl])
            results[f"CNN ({name})"] = {"clear": evaluate(yte, pc), "cloudy": evaluate(yte, pk)}
            if name == "S1+S2":
                print(classification_report(yte, pc, target_names=CLASSES, digits=3))
                print("Confusion matrix:\n", confusion_matrix(yte, pc))

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/metrics.json", "w"), indent=2)
    print("\n== Ringkasan (accuracy | macro-F1) ==")
    print(f"{'Model':16s} {'Langit cerah':>20s} {'Berawan 100%':>20s}")
    for k_, v in results.items():
        print(f"{k_:16s} {v['clear']['accuracy']:>9.3f} | {v['clear']['macro_f1']:<8.3f} "
              f"{v['cloudy']['accuracy']:>9.3f} | {v['cloudy']['macro_f1']:<8.3f}")


if __name__ == "__main__":
    main()
