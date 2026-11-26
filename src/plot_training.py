import argparse
import json
import os

import matplotlib.pyplot as plt


# 保存训练历史：train() 每个 batch 间隔调用一次
def save_history(
    path,
    loss_list,
    acc_list,
    lr_list,
    val_loss_list,
    val_acc_list,
    batch_loss_list,
    batch_acc_list,
):
    history = {
        "loss": loss_list,              # 每个 epoch 的训练 loss
        "acc": acc_list,                # 每个 epoch 的训练 accuracy
        "lr": lr_list,                  # 每个 batch 的学习率
        "val_loss": val_loss_list,      # 每个 epoch 的验证 loss
        "val_acc": val_acc_list,        # 每个 epoch 的验证 accuracy
        "batch_loss": batch_loss_list,  # 每个 batch 的 loss
        "batch_acc": batch_acc_list,    # 每个 batch 的 accuracy
    }

    # 先写临时文件再替换，训练中另开脚本读取时不会读到半个文件
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# 读取训练历史
def load_history(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 移动平均，用于平滑 batch 曲线
def moving_average(values, window=20):
    if not values or window <= 1:
        return list(values)

    result = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        result.append(total / min(i + 1, window))
    return result


# 数据可视化：
# 1. Epoch Loss  2. Epoch Accuracy  3. Learning Rate
# 4. Batch Loss  5. Batch Accuracy  6. 汇总信息
def plot_training_curve(
    loss_list,
    acc_list,
    lr_list=None,
    val_loss_list=None,
    val_acc_list=None,
    batch_loss_list=None,
    batch_acc_list=None,
    save_path="reports/classification_curve.png",
    show=False,
):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    loss_list = loss_list or []
    acc_list = acc_list or []
    lr_list = lr_list or []
    val_loss_list = val_loss_list or []
    val_acc_list = val_acc_list or []
    batch_loss_list = batch_loss_list or []
    batch_acc_list = batch_acc_list or []

    if not loss_list and not lr_list:
        print("没有训练数据可绘图")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # ---- 1. Epoch Loss ----
    ax = axes[0, 0]
    if loss_list:
        epoch_axis = range(1, len(loss_list) + 1)
        ax.plot(
            epoch_axis, loss_list,
            color="tab:red", marker="o", markersize=3, label="Train",
        )
        if val_loss_list:
            ax.plot(
                epoch_axis, val_loss_list[: len(loss_list)],
                color="tab:orange", linestyle="--",
                marker="o", markersize=3, label="Valid",
            )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "等待第一个 epoch 结束", ha="center", va="center")
    ax.set_title("Epoch Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ---- 2. Epoch Accuracy ----
    ax = axes[0, 1]
    if acc_list:
        epoch_axis = range(1, len(acc_list) + 1)
        ax.plot(
            epoch_axis, acc_list,
            color="tab:blue", marker="o", markersize=3, label="Train",
        )
        if val_acc_list:
            ax.plot(
                epoch_axis, val_acc_list[: len(acc_list)],
                color="tab:green", linestyle="--",
                marker="o", markersize=3, label="Valid",
            )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "等待第一个 epoch 结束", ha="center", va="center")
    ax.set_title("Epoch Accuracy")
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ---- 3. 每个 batch 的学习率 ----
    ax = axes[0, 2]
    if lr_list:
        ax.plot(
            range(1, len(lr_list) + 1), lr_list,
            color="tab:green", linewidth=1,
        )
        ax.set_yscale("log")
        ax.set_title("Learning Rate (per batch)")
        ax.set_xlabel("Batch Step")
        ax.grid(True, linestyle="--", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "no lr data", ha="center", va="center")
        ax.set_title("Learning Rate")

    # ---- 4. Batch Loss ----
    ax = axes[1, 0]
    if batch_loss_list:
        batch_axis = range(1, len(batch_loss_list) + 1)
        ax.plot(
            batch_axis, batch_loss_list,
            color="tab:red", alpha=0.25, linewidth=0.8, label="Raw",
        )
        smooth_loss = moving_average(batch_loss_list, window=20)
        ax.plot(
            batch_axis, smooth_loss,
            color="tab:red", linewidth=1.5, label="Smooth(20)",
        )
        ax.legend()
        ax.set_title("Batch Loss")
        ax.set_xlabel("Batch Step")
        ax.grid(True, linestyle="--", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "no batch loss data", ha="center", va="center")
        ax.set_title("Batch Loss")

    # ---- 5. Batch Accuracy ----
    ax = axes[1, 1]
    if batch_acc_list:
        batch_axis = range(1, len(batch_acc_list) + 1)
        ax.plot(
            batch_axis, batch_acc_list,
            color="tab:blue", alpha=0.25, linewidth=0.8, label="Raw",
        )
        smooth_acc = moving_average(batch_acc_list, window=20)
        ax.plot(
            batch_axis, smooth_acc,
            color="tab:blue", linewidth=1.5, label="Smooth(20)",
        )
        ax.legend()
        ax.set_title("Batch Accuracy")
        ax.set_xlabel("Batch Step")
        ax.grid(True, linestyle="--", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "no batch acc data", ha="center", va="center")
        ax.set_title("Batch Accuracy")

    # ---- 6. 汇总信息 ----
    ax = axes[1, 2]
    ax.axis("off")
    summary = []
    if loss_list and acc_list:
        summary.append(f"Epochs: {len(loss_list)}")
        summary.append(f"Train loss: {loss_list[-1]:.6f}")
        summary.append(f"Train acc: {acc_list[-1] * 100:.2f}%")
    if val_loss_list and val_acc_list:
        summary.append(f"Valid loss: {val_loss_list[-1]:.6f}")
        summary.append(f"Valid acc: {val_acc_list[-1] * 100:.2f}%")
    if lr_list:
        summary.append(f"LR: {lr_list[-1]:.2e}")
    if batch_loss_list and batch_acc_list:
        summary.append(f"Batch loss: {batch_loss_list[-1]:.6f}")
        summary.append(f"Batch acc: {batch_acc_list[-1] * 100:.2f}%")
    ax.text(0.05, 0.95, "\n".join(summary), va="top", fontsize=11)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="reports/classification_history.json")
    parser.add_argument(
        "--checkpoint",
        default="./model/TextClassificationModel_checkpoint.bin",
    )
    parser.add_argument("--output", default="reports/classification_curve.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.history):
        data = load_history(args.history)
        print("读取训练历史:", args.history)
        loss_list = data.get("loss", [])
        acc_list = data.get("acc", [])
        lr_list = data.get("lr", [])
        val_loss_list = data.get("val_loss", [])
        val_acc_list = data.get("val_acc", [])
        batch_loss_list = data.get("batch_loss", [])
        batch_acc_list = data.get("batch_acc", [])
    elif os.path.exists(args.checkpoint):
        import torch

        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        print("读取断点历史:", args.checkpoint)
        loss_list = checkpoint.get("loss_list", [])
        acc_list = checkpoint.get("acc_list", [])
        lr_list = checkpoint.get("lr_list", [])
        val_loss_list = checkpoint.get("val_loss_list", [])
        val_acc_list = checkpoint.get("val_acc_list", [])
        batch_loss_list = checkpoint.get("batch_loss_list", [])
        batch_acc_list = checkpoint.get("batch_acc_list", [])
    else:
        print("未找到训练历史或断点文件")
        return

    if not loss_list and not lr_list:
        print("没有可绘制的训练数据")
        return

    plot_training_curve(
        loss_list,
        acc_list,
        lr_list,
        val_loss_list,
        val_acc_list,
        batch_loss_list,
        batch_acc_list,
        save_path=args.output,
        show=not args.no_show,
    )
    print("图像已保存:", args.output)

    if loss_list and acc_list:
        print(
            f"已完成 {len(loss_list)} 个 epoch，"
            f"最新 train_loss={loss_list[-1]:.6f}, "
            f"train_acc={acc_list[-1] * 100:.2f}%"
        )
    if val_loss_list and val_acc_list:
        print(
            f"最新 valid_loss={val_loss_list[-1]:.6f}, "
            f"valid_acc={val_acc_list[-1] * 100:.2f}%"
        )
    if batch_loss_list and batch_acc_list:
        print(
            f"最新 batch_loss={batch_loss_list[-1]:.6f}, "
            f"batch_acc={batch_acc_list[-1] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()