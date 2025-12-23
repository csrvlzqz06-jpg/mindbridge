import argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor
from lstm_model import LSTMRiskNet

CLASSES = ["LOW", "MEDIUM", "HIGH"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/windows_dataset.npz")
    parser.add_argument("--ckpt", type=str, default="model/mindbridge_lstm.ckpt")
    parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(np.float32)

    _, T, F = X.shape
    net = LSTMRiskNet(input_size=F, hidden_size=64, num_layers=1, num_classes=3)
    ms.load_checkpoint(args.ckpt, net=net)
    net.set_train(False)

    idx = np.random.choice(X.shape[0], size=args.n, replace=False)
    x = Tensor(X[idx])
    logits = net(x).asnumpy()
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    for i, p in zip(idx, probs):
        cls = int(p.argmax())
        score = float(p.max())
        print(f"sample={i}  class={CLASSES[cls]}  confidence={score:.3f}  probs={p}")

if __name__ == "__main__":
    main()
