import torch
from torch import nn
from torch.utils.data import dataloader, Dataset
from datasets import load_dataset  # HuggingFace 的数据集加载工具
from tqdm import tqdm
from transformers import BertTokenizer, BertModel  # Bert分词器与Bert模型
from torch.optim import AdamW  # 优化器
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
import csv


torch.manual_seed(42)
random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 模型路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"
model_path = os.environ.get("BERT_MODEL_PATH", str(PROJECT_ROOT / "models" / "bert-base-chinese"))

tokenizer = BertTokenizer.from_pretrained(model_path)
# 加载模型
bert_model = BertModel.from_pretrained(model_path).to(device)
bert_model.eval()


class TrainDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # 加载数据，并过滤出长度>44的句子
        self.dataset_item = load_dataset(
            "csv", data_files=data_path, split="train"
        ).filter(lambda x: len(x["text"]) > 44)

    def __len__(self):
        return len(self.dataset_item)

    def __getitem__(self, idx):
        # 1.假设label= 1有关系 0无关系
        label = 1
        # 2.根据索引获取句子
        sentence = self.dataset_item[idx]["text"]
        # 3.构建句子对   (统一在22做切割，句子长度截断为44)
        sentence1 = sentence[:22]
        sentence2 = sentence[22:44]
        # 4.构建负样本 0
        if random.randint(0, 1) == 0:
            pos = random.randint(0, len(self.dataset_item) - 1)
            if pos != idx:  #  当随机的索引与当前索引不一致的时候
                sentence2 = self.dataset_item[pos]["text"][22:44]
                label = 0
        # 5.返回数据
        return sentence1, sentence2, label


# 数据整理函数，用于处理批量数据
def collate_fn(batch_data):
    # batch_data -> [(s1, s2, label)]
    # 将标签与文本分离
    sent_pairs = [item[:2] for item in batch_data]
    labels = torch.LongTensor([item[-1] for item in batch_data])

    # 批量文本编码
    inputs = tokenizer(
        sent_pairs,
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors="pt",
    )

    return inputs.to(device), labels.to(device)


# 获取数据加载器
def get_dataloader(file_path=None):
    if file_path is None:
        file_path = str(DATA_DIR / "train.csv")
    # 获取数据集
    train_dataset = TrainDataset(file_path)

    # 训练集加载器
    train_dataloader = dataloader.DataLoader(
        train_dataset,  # 训练集
        batch_size=32,  # 批次大小
        shuffle=True,  # 是否打乱
        drop_last=True,  # 是否丢弃最后不足批次大小的一批
        collate_fn=collate_fn,  # 每批数据要经过该函数的处理
    )

    return train_dataloader


# 下游模型二分类头
class TextNSPModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        # BERT模型层 (上游模型)
        self.bert_model = bert_model
        # 全连接映射层，将BERT的768维映射成2维 (有关系/无关系)
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        前向传播
        :param input_ids: 文本的数字编码
        :param token_type_ids:  句子类型标记
        :param attention_mask:  注意力掩码
        :return:
        """
        # 不训练bert的参数 (冻结参数)
        with torch.no_grad():
            bert_outputs = self.bert_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        # 将BERT的池化输出映射成二维标签
        output = self.linear(bert_outputs.pooler_output)
        # 返回数据
        return output


# 模型训练
def train():
    train_dataloader = get_dataloader()
    # 训练参数
    epochs = 10
    total_steps = epochs * len(train_dataloader)

    # 加载下游模型
    model = TextNSPModel(bert_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.linear.parameters(), lr=1e-3)

    # 余弦退火：每个 batch 调一次 scheduler.step()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-5,
    )

    # epoch 级历史
    loss_list = []  # 每个 epoch 的训练 loss
    acc_list = []  # 每个 epoch 的训练 accuracy
    val_loss_list = []  # 每个 epoch 的验证 loss
    val_acc_list = []  # 每个 epoch 的验证 accuracy

    # batch 级历史
    lr_list = []  # 每个 batch 的学习率
    batch_loss_list = []  # 每个 batch 的 loss
    batch_acc_list = []  # 每个 batch 的 accuracy

    best_val_acc = -1.0
    start_epoch = 0
    global_step = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 轮次训练
    for epoch in range(start_epoch, epochs):
        model.train()
        model.bert_model.eval()

        total_loss = 0
        total_acc = 0
        total_samples = 0
        bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        # 批次训练
        for x, y in bar:
            optimizer.zero_grad()
            output = model(**x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            # batch 级统计
            batch_loss = loss.item()
            correct = (output.argmax(1) == y).sum().item()
            batch_acc = correct / y.size(0)

            lr_list.append(optimizer.param_groups[0]["lr"])
            batch_loss_list.append(batch_loss)
            batch_acc_list.append(batch_acc)

            total_loss += batch_loss * y.size(0)
            total_acc += correct
            total_samples += y.size(0)

            # 实时显示
            bar.set_postfix(
                lr=f"{lr_list[-1]:.2e}",
                loss=f"{total_loss / total_samples:.6f}",
                acc=f"{total_acc / total_samples * 100:.2f}%",
            )

        # 本轮训练指标
        train_loss = total_loss / total_samples
        train_acc = total_acc / total_samples
        loss_list.append(train_loss)
        acc_list.append(train_acc)

    # 保存最终模型
    torch.save(model.state_dict(), str(CHECKPOINT_DIR / "TextNSPModel.bin"))
    print("训练完成，模型已保存")

    # 数据可视化
    plot_training_curves(
        loss_list=loss_list,
        acc_list=acc_list,
        lr_list=lr_list,
        batch_loss_list=batch_loss_list,
        batch_acc_list=batch_acc_list,
        val_loss_list=val_loss_list,
        val_acc_list=val_acc_list,
        save_path=str(REPORT_DIR / "nsp_curve.png"),
        show=False,
    )


def _moving_average(values, window):
    """简单滑动平均，用于平滑 batch 级曲线。"""
    if not values:
        return []
    window = max(1, min(window, len(values)))
    result = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        result.append(running / min(i + 1, window))
    return result


def plot_training_curves(
    loss_list,
    acc_list,
    lr_list=None,
    batch_loss_list=None,
    batch_acc_list=None,
    val_loss_list=None,
    val_acc_list=None,
    save_path=str(REPORT_DIR / "nsp_curve.png"),
    smooth_window=50,
    show=False,
):
    """
    绘制 NSP 训练曲线：
    1. epoch 训练/验证 loss
    2. epoch 训练/验证 accuracy
    3. batch 级学习率
    4. batch 级 loss（原始 + 平滑）
    5. batch 级 accuracy（原始 + 平滑）
    6. 最终指标摘要
    """
    lr_list = lr_list or []
    batch_loss_list = batch_loss_list or []
    batch_acc_list = batch_acc_list or []
    val_loss_list = val_loss_list or []
    val_acc_list = val_acc_list or []

    if not loss_list:
        print("[plot] loss_list 为空，跳过绘图")
        return None

    # 中文字体，避免标题乱码
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("BERT NSP Training Curves", fontsize=15)

    # 1. epoch loss
    ax = axes[0, 0]
    epochs = list(range(1, len(loss_list) + 1))
    ax.plot(epochs, loss_list, marker="o", label="train loss")
    if val_loss_list:
        ax.plot(
            range(1, len(val_loss_list) + 1),
            val_loss_list,
            marker="s",
            label="val loss",
        )
    ax.set_title("Epoch Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # 2. epoch accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [v * 100 for v in acc_list], marker="o", label="train acc")
    if val_acc_list:
        ax.plot(
            range(1, len(val_acc_list) + 1),
            [v * 100 for v in val_acc_list],
            marker="s",
            label="val acc",
        )
    ax.set_title("Epoch Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # 3. batch learning rate
    ax = axes[0, 2]
    if lr_list:
        ax.plot(range(1, len(lr_list) + 1), lr_list, color="tab:green")
        ax.set_yscale("log")
        ax.set_title("Learning Rate per Batch")
        ax.set_xlabel("Batch")
        ax.set_ylabel("LR (log scale)")
        ax.grid(True, linestyle="--", alpha=0.4)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no lr history", ha="center", va="center")

    # 4. batch loss
    ax = axes[1, 0]
    if batch_loss_list:
        x = range(1, len(batch_loss_list) + 1)
        ax.plot(x, batch_loss_list, color="tab:red", alpha=0.2, label="raw")
        ax.plot(
            x,
            _moving_average(batch_loss_list, smooth_window),
            color="tab:red",
            label=f"smoothed({smooth_window})",
        )
        ax.set_title("Batch Loss")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no batch loss", ha="center", va="center")

    # 5. batch accuracy
    ax = axes[1, 1]
    if batch_acc_list:
        x = range(1, len(batch_acc_list) + 1)
        raw = [v * 100 for v in batch_acc_list]
        smooth = [v * 100 for v in _moving_average(batch_acc_list, smooth_window)]
        ax.plot(x, raw, color="tab:blue", alpha=0.2, label="raw")
        ax.plot(x, smooth, color="tab:blue", label=f"smoothed({smooth_window})")
        ax.set_title("Batch Accuracy")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no batch acc", ha="center", va="center")

    # 6. summary
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"epochs: {len(loss_list)}",
        f"final train loss: {loss_list[-1]:.4f}",
        f"final train acc: {acc_list[-1] * 100:.2f}%",
        f"best train acc: {max(acc_list) * 100:.2f}%",
    ]
    if val_acc_list:
        lines.append(f"best val acc: {max(val_acc_list) * 100:.2f}%")
    if lr_list:
        lines.append(f"final lr: {lr_list[-1]:.2e}")
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[plot] 训练曲线已保存: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path


# 模型评估
def evaluate(model_path=str(CHECKPOINT_DIR / "TextNSPModel.bin")):
    test_dataloader = get_dataloader(file_path=str(DATA_DIR / "test.csv"))
    model = TextNSPModel(bert_model).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    total_samples = 0
    acc_total = 0

    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            output = model(**x)
            total_samples += y.size(0)
            acc_total += (output.argmax(1) == y).sum().item()

    print(f"test acc: {acc_total / total_samples * 100:.2f}%")


if __name__ == "__main__":
    # train()
    evaluate()
