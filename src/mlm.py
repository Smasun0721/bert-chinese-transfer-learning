import torch
from torch import nn
from torch.utils.data import dataloader
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


# 获取数据集
def get_datasets():
    # 加载训练集
    train_dataset = load_dataset(
        "csv", data_files=str(DATA_DIR / "train.csv"), split="train"
    )
    # 加载训练集
    test_dataset = load_dataset("csv", data_files=str(DATA_DIR / "test.csv"), split="train")
    # 加载验证集
    valid_dataset = load_dataset(
        "csv", data_files=str(DATA_DIR / "validation.csv"), split="train"
    )

    return train_dataset, test_dataset, valid_dataset


# 随机掩码
def random_mask(inputs, mask_prob=0.15):
    """
    对 input_ids 中非 special token 的位置随机替换为 [MASK]
    返回 (inputs, labels)，labels 形状 [batch, seq_len]
    :param inputs: 输入张量
    :param mask_prob: 遮挡比例
    :return:
    """
    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"]
    special_ids = set(tokenizer.all_special_ids)  # CLS/SEP/PAD/UNK/MASK

    # 建标签矩阵，全填 -100（忽略位，loss时手动设置忽略该值）
    labels = torch.full_like(input_ids, -100)

    for i in range(input_ids.size(0)):
        # 1.找出这一条里"有效且非 special"的位置
        valid_pos = [
            j
            for j in range(input_ids.size(1))
            if attention_mask[i, j] == 1 and input_ids[i, j].item() not in special_ids
        ]
        if not valid_pos:
            continue

        # 2.按比例采样要遮挡的位置
        n_mask = max(1, int(len(valid_pos) * mask_prob))  # 遮挡数不能小于1
        n_mask = min(n_mask, len(valid_pos))  # 遮挡数不能大于有效位置数量
        # 从 valid_pos 中随机选出 n_mask 个位置
        mask_pos = random.sample(valid_pos, n_mask)

        # 3.替换为 [MASK]
        for p in mask_pos:
            labels[i, p] = input_ids[i, p]
            input_ids[i, p] = tokenizer.mask_token_id

    inputs["input_ids"] = input_ids
    return inputs, labels


# 数据整理函数，用于处理批量数据
def collate_fn(batch_data):
    # batch_data -> [{"text": "...", "label": 0}, {"text": "...", "label": 1}
    # 将标签与文本分离
    texts = [item["text"] for item in batch_data]

    # 批量文本编码
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # 将句子中随机位置的词替换成[MASK]
    inputs, labels = random_mask(inputs)

    return inputs.to(device), labels.to(device)


# 获取数据加载器
def get_dataloader():
    # 获取数据集
    train_dataset, test_dataset, valid_dataset = get_datasets()

    # 训练集加载器
    train_dataloader = dataloader.DataLoader(
        train_dataset,  # 训练集
        batch_size=32,  # 批次大小
        shuffle=True,  # 是否打乱
        drop_last=True,  # 是否丢弃最后不足批次大小的一批
        collate_fn=collate_fn,  # 每批数据要经过该函数的处理
    )

    # 测试集加载器
    test_dataloader = dataloader.DataLoader(
        test_dataset,  # 训练集
        batch_size=32,  # 批次大小
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,  # 每批数据要经过该函数的处理
    )

    # 验证集加载器
    valid_dataloader = dataloader.DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_dataloader, test_dataloader, valid_dataloader


# 获取数据加载器
train_dataloader, test_dataloader, valid_dataloader = get_dataloader()


# 下游模型
class TextForMaskModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        # 上游模型
        self.bert_model = bert_model
        # 冻结参数
        for p in bert_model.parameters():
            p.requires_grad = False
        # 多分类头
        self.linear = nn.Linear(bert_model.config.hidden_size, tokenizer.vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 不训练bert的参数 (冻结参数)
        with torch.no_grad():
            bert_outputs = self.bert_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        # 将BERT的池化输出映射成词典下标
        output = self.linear(bert_outputs.last_hidden_state)
        # 返回数据
        return output


# 模型训练
def train():
    # 加载模型
    model = TextForMaskModel(bert_model).to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # 优化器
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 训练参数
    epochs = 10
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        total_mask = 0
        bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for x, y in bar:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播 + 反向传播 + 参数更新
            logits = model(**x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            loss.backward()
            optimizer.step()

            # 统计
            mask_sel = y != -100  # [B, S]
            n_mask = mask_sel.sum().item()
            epoch_loss += loss.item() * n_mask
            epoch_correct += ((logits.argmax(-1) == y) & mask_sel).sum().item()
            total_mask += n_mask
            # 实时显示
            bar.set_postfix(
                loss=f"{epoch_loss / max(total_mask, 1):.4f}",
                acc=f"{epoch_correct / max(total_mask, 1) * 100:.2f}%",
            )

        # 统计
        loss_list.append(epoch_loss / max(total_mask, 1))
        acc_list.append(epoch_correct / max(total_mask, 1))
        print(f"epoch {epoch + 1} | loss={loss_list[-1]:.4f} | acc={acc_list[-1]:.2%}")

    # 保存最终模型
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    torch.save(model.state_dict(), str(CHECKPOINT_DIR / "TextForMaskModel.bin"))

    # 数据可视化
    plot_training_curves(loss_list, acc_list, save_dir=str(REPORT_DIR))


# 数据可视化
def plot_training_curves(loss_list, acc_list, save_dir=str(REPORT_DIR)):
    """
    绘制训练过程的 loss 与 accuracy 曲线，并保存图片和原始数据
    :param loss_list: 每个 epoch 的平均 loss
    :param acc_list:  每个 epoch 的 mask 位置准确率
    :param save_dir:  保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(loss_list) + 1))

    # ---------- 保存原始数据为 csv ----------
    csv_path = os.path.join(save_dir, "mlm_train_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "acc"])
        for e, l, a in zip(epochs, loss_list, acc_list):
            writer.writerow([e, f"{l:.6f}", f"{a:.6f}"])
    print(f"[plot] 训练日志已保存: {csv_path}")

    # ---------- 绘图 ----------
    # 中文字体（没有 SimHei 就退化为英文标签，避免乱码）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：loss
    axes[0].plot(epochs, loss_list, color="tab:red", label="train loss")
    axes[0].set_title("Training Loss (MLM)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    # 标注最小值
    best_loss_epoch = epochs[
        int(min(range(len(loss_list)), key=lambda i: loss_list[i]))
    ]
    best_loss = min(loss_list)
    axes[0].annotate(
        f"min={best_loss:.4f}",
        xy=(best_loss_epoch, best_loss),
        xytext=(best_loss_epoch, best_loss + (max(loss_list) - min(loss_list)) * 0.1),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
    )

    # 右图：accuracy
    acc_pct = [a * 100 for a in acc_list]
    axes[1].plot(epochs, acc_pct, color="tab:blue", label="mask acc")
    axes[1].set_title("Masked Token Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    # 标注最大值
    best_acc_epoch = epochs[int(max(range(len(acc_list)), key=lambda i: acc_list[i]))]
    best_acc = max(acc_pct)
    axes[1].annotate(
        f"max={best_acc:.2f}%",
        xy=(best_acc_epoch, best_acc),
        xytext=(best_acc_epoch, best_acc - (max(acc_pct) - min(acc_pct)) * 0.15 - 1),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
    )

    fig.suptitle("BERT MLM Training Curves", fontsize=13)
    fig.tight_layout()

    png_path = os.path.join(save_dir, "mlm_curve.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 训练曲线已保存: {png_path}")


def evaluate():
    # 加载模型
    model = TextForMaskModel(bert_model).to(device)
    model.load_state_dict(
        torch.load(
            str(CHECKPOINT_DIR / "TextForMaskModel.bin"), map_location=device, weights_only=True
        )
    )
    model.eval()

    # 模型预测
    correct = 0
    total = 0
    for x, y in tqdm(test_dataloader):
        with torch.no_grad():
            logits = model(**x)
            mask_sel = y != -100
            correct += ((logits.argmax(-1) == y) & mask_sel).sum().item()
            total += mask_sel.sum().item()

    print(f"准确率: {correct / total:.2%}")


from collections import Counter


def show_predictions(model, dataloader, tokenizer, n_batches=3, topk=3):
    """
    打印 [MASK] 位置的预测结果，并与"高频字先验"基线对比
    """
    model.eval()

    # ---------- 1. 统计训练集 token 频率，作为高频先验基线 ----------
    freq = Counter()
    for item in train_dataloader.dataset:
        ids = tokenizer(item["text"], truncation=True, max_length=128)["input_ids"]
        freq.update(ids)
    # 去掉 special token
    special = set(tokenizer.all_special_ids)
    for s in special:
        freq.pop(s, None)
    top1_token = freq.most_common(1)[0][0]
    print(
        f"[基线] 训练集最高频 token = {tokenizer.decode([top1_token])!r} "
        f"(id={top1_token}, 出现 {freq[top1_token]} 次)"
    )
    print("=" * 100)

    # ---------- 2. 逐 batch 展示预测 ----------
    model_correct = baseline_correct = total = 0
    shown = 0
    for b_idx, (x, y) in enumerate(dataloader):
        if b_idx >= n_batches:
            break
        with torch.no_grad():
            logits = model(**x)  # [B, S, V]
            preds = logits.argmax(-1)  # [B, S]
            topk_ids = logits.topk(topk, dim=-1).indices  # [B, S, k]

        input_ids = x["input_ids"].cpu()
        labels = y.cpu()
        preds_cpu = preds.cpu()
        topk_cpu = topk_ids.cpu()
        mask_id = tokenizer.mask_token_id

        for i in range(input_ids.size(0)):
            mask_pos = (labels[i] != -100).nonzero(as_tuple=True)[0].tolist()
            if not mask_pos:
                continue

            # 还原"原句"：把 labels 里的真值填回 input_ids 的 mask 位置
            orig_ids = input_ids[i].clone()
            for p in mask_pos:
                orig_ids[p] = labels[i, p]
            orig_text = tokenizer.decode(orig_ids, skip_special_tokens=True)

            # "mask 后的句子"
            masked_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)

            print(f"[样本 {shown}] 原句   : {orig_text}")
            print(f"          mask 后: {masked_text}")
            for p in mask_pos:
                true_id = labels[i, p].item()
                pred_id = preds_cpu[i, p].item()
                true_tok = tokenizer.decode([true_id])
                pred_tok = tokenizer.decode([pred_id])
                flag = "✓" if pred_id == true_id else "✗"
                topk_toks = [tokenizer.decode([t]) for t in topk_cpu[i, p].tolist()]
                print(
                    f"          pos={p:3d} | 真实={true_tok!r:>6} | "
                    f"预测={pred_tok!r:>6} {flag} | top{topk}={topk_toks}"
                )

                model_correct += pred_id == true_id
                baseline_correct += top1_token == true_id
                total += 1
            print("-" * 100)
            shown += 1

    # ---------- 3. 汇总 ----------
    print(f"展示样本数: {shown}, mask token 数: {total}")
    print(f"模型 top-1 准确率    : {model_correct / total:.2%}")
    print(f"高频先验 top-1 准确率: {baseline_correct / total:.2%}")


if __name__ == "__main__":
    # train()
    evaluate()
    # 展示实际预测
    model = TextForMaskModel(bert_model).to(device)
    model.load_state_dict(
        torch.load(
            str(CHECKPOINT_DIR / "TextForMaskModel.bin"), map_location=device, weights_only=True
        )
    )
    show_predictions(model, test_dataloader, tokenizer, n_batches=3, topk=3)
