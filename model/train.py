import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.dataset import NumpySlicesDataset

from lstm_model import LSTMRiskNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/windows_dataset.npz")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--save", type=str, default="model/mindbridge_lstm.ckpt")
    parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU"])
    args = parser.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)  # (N, T, F)
    y = data["y"].astype(np.int32)    # (N,)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = NumpySlicesDataset({"x": X_train, "y": y_train}, shuffle=True).batch(args.batch_size)
    test_ds  = NumpySlicesDataset({"x": X_test, "y": y_test}, shuffle=False).batch(args.batch_size)

    _, T, F = X.shape
    net = LSTMRiskNet(input_size=F, hidden_size=args.hidden, num_layers=1, num_classes=3)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr)

    model = ms.train.Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={"acc": nn.Accuracy()})

    print("[INFO] Training...")
    model.train(args.epochs, train_ds, dataset_sink_mode=False)
    print("[INFO] Evaluating...")
    metrics = model.eval(test_ds, dataset_sink_mode=False)
    print("[METRICS]", metrics)

    # Predicciones para reporte
    y_pred = []
    y_true = []
    for batch in test_ds.create_dict_iterator():
        logits = net(Tensor(batch["x"]))
        pred = logits.asnumpy().argmax(axis=1)
        y_pred.extend(pred.tolist())
        y_true.extend(batch["y"].asnumpy().tolist())

    print("\n[CONFUSION MATRIX]\n", confusion_matrix(y_true, y_pred))
    print("\n[CLASSIFICATION REPORT]\n", classification_report(y_true, y_pred, digits=4))

    ms.save_checkpoint(net, args.save)
    print(f"[OK] Saved checkpoint: {args.save}")

if __name__ == "__main__":
    main()
