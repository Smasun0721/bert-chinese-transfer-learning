import torch
from torch import nn
from torch.utils.data import dataloader
from datasets import load_dataset  # HuggingFace 的数据集加载工具
from tqdm import tqdm
from transformers import BertTokenizer, BertModel  # Bert分词器与Bert模型
from torch.optim import AdamW  # 优化器
import os
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
from .plot_training import plot_training_curve, save_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 模型路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"
model_path = os.environ.get("BERT_MODEL_PATH", str(PROJECT_ROOT / "models" / "bert-base-chinese"))

# 加载分词器
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


# 数据整理函数，用于处理批量数据
def collate_fn(batch_data):
    # batch_data -> [{"text": "...", "label": 0}, {"text": "...", "label": 1}
    # 将标签与文本分离
    texts = [item["text"] for item in batch_data]
    labels = torch.tensor(
        [item["label"] for item in batch_data], dtype=torch.long, device=device
    )

    # 批量文本编码
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    return inputs, labels


# 获取数据加载器
def get_dataloader():
    # 获取数据集
    train_dataset, test_dataset, valid_dataset = get_datasets()

    # 训练集加载器
    train_dataloader = dataloader.DataLoader(
        train_dataset,  # 训练集
        batch_size=64,  # 批次大小
        shuffle=True,  # 是否打乱
        drop_last=True,  # 是否丢弃最后不足批次大小的一批
        collate_fn=collate_fn,  # 每批数据要经过该函数的处理
    )

    # 测试集加载器
    test_dataloader = dataloader.DataLoader(
        test_dataset,  # 训练集
        batch_size=64,  # 批次大小
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,  # 每批数据要经过该函数的处理
    )

    # 验证集加载器
    valid_dataloader = dataloader.DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_dataloader, test_dataloader, valid_dataloader


# 好评分类头 (下游模型)
class TextClassificationModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        # BERT模型层 (上游模型)
        self.bert_model = bert_model
        # 全连接映射层，将BERT的768维映射成2维 (好评/差评)
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


# 获取数据加载器
train_dataloader, test_dataloader, valid_dataloader = get_dataloader()


# 通用评估函数
def evaluate_model(model, dataloader, criterion=None, desc="Evaluating"):
    model.eval()
    total_samples = 0
    total_loss = 0.0
    acc_total = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc):
            output = model(**x)
            total_samples += y.size(0)
            acc_total += (output.argmax(1) == y).sum().item()
            if criterion is not None:
                total_loss += criterion(output, y).item() * y.size(0)

    # 恢复训练状态：BERT 是冻结的，保持 eval
    model.train()
    model.bert_model.eval()

    if total_samples == 0:
        return float("nan"), 0.0

    avg_loss = total_loss / total_samples if criterion is not None else float("nan")
    return avg_loss, acc_total / total_samples


# 保存断点：模型、优化器、调度器、训练进度、batch 历史、随机数状态
def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_val_acc,
    loss_list,
    acc_list,
    lr_list,
    val_loss_list,
    val_acc_list,
    batch_loss_list,
    batch_acc_list,
):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "loss_list": loss_list,
        "acc_list": acc_list,
        "lr_list": lr_list,
        "val_loss_list": val_loss_list,
        "val_acc_list": val_acc_list,
        "batch_loss_list": batch_loss_list,
        "batch_acc_list": batch_acc_list,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }

    # 先写临时文件再替换，避免写一半时崩溃损坏断点
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


# 加载断点并恢复训练状态
def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 恢复随机数状态，保证 shuffle 结果可复现
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_state"]])

    return checkpoint


# 模型训练
def train(resume=True):
    # 训练参数
    epochs = 70
    base_lr = 1e-5
    eta_min = 1e-7
    total_steps = epochs * len(train_dataloader)

    # 加载下游模型
    model = TextClassificationModel(bert_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.linear.parameters(), lr=base_lr)

    # 余弦退火：每个 batch 调一次 scheduler.step()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=eta_min,
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

    checkpoint_path = str(CHECKPOINT_DIR / "TextClassificationModel_checkpoint.bin")
    best_model_path = str(CHECKPOINT_DIR / "TextClassificationModel_best.bin")
    history_path = str(REPORT_DIR / "classification_history.json")

    # 每 10 个 batch 写一次历史；如果要每个 batch 都写，改成 1
    history_interval = 10

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 存在断点时自动恢复
    if resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_val_acc = checkpoint.get("best_val_acc", -1.0)

        loss_list = checkpoint.get("loss_list", [])
        acc_list = checkpoint.get("acc_list", [])
        val_loss_list = checkpoint.get("val_loss_list", [])
        val_acc_list = checkpoint.get("val_acc_list", [])
        lr_list = checkpoint.get("lr_list", [])
        batch_loss_list = checkpoint.get("batch_loss_list", [])
        batch_acc_list = checkpoint.get("batch_acc_list", [])

        # 先恢复到基准学习率
        for group in optimizer.param_groups:
            group["lr"] = base_lr

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=eta_min,
        )

        # 定位到当前已完成的 batch 数
        scheduler.last_epoch = global_step
        scheduler._step_count = global_step + 1

        # 计算并写回当前步数对应的学习率
        lrs = scheduler._get_closed_form_lr()
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr
        scheduler._last_lr = lrs

        print(f"重建后当前 lr={lrs[0]:.2e}")

    if start_epoch >= epochs:
        print("训练已经完成。若要重新训练，请设置 resume=False 或删除断点文件。")
        return

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

            # 每 N 个 batch 写一次轻量历史，外部绘图脚本能实时看到 batch 级 lr
            if global_step % history_interval == 0:
                save_history(
                    history_path,
                    loss_list,
                    acc_list,
                    lr_list,
                    val_loss_list,
                    val_acc_list,
                    batch_loss_list,
                    batch_acc_list,
                )

        # 本轮训练指标
        train_loss = total_loss / total_samples
        train_acc = total_acc / total_samples
        loss_list.append(train_loss)
        acc_list.append(train_acc)

        # 每个 epoch 结束后评估验证集
        val_loss, val_acc = evaluate_model(
            model,
            valid_dataloader,
            criterion=criterion,
            desc=f"Valid {epoch + 1}/{epochs}",
        )
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # 验证集提升时保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Epoch {epoch + 1}: 验证集准确率 {val_acc * 100:.2f}%，已保存最佳模型"
            )

        # 每个 epoch 结束保存断点
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            best_val_acc,
            loss_list,
            acc_list,
            lr_list,
            val_loss_list,
            val_acc_list,
            batch_loss_list,
            batch_acc_list,
        )
        print(f"Epoch {epoch + 1} 断点已保存")

        # epoch 结束时补写一次完整历史
        save_history(
            history_path,
            loss_list,
            acc_list,
            lr_list,
            val_loss_list,
            val_acc_list,
            batch_loss_list,
            batch_acc_list,
        )

    # 保存最终模型
    torch.save(model.state_dict(), str(CHECKPOINT_DIR / "TextClassificationModel.bin"))
    print(f"训练完成，最佳验证集准确率: {best_val_acc * 100:.2f}%")

    # 数据可视化
    plot_training_curve(
        loss_list,
        acc_list,
        lr_list,
        val_loss_list,
        val_acc_list,
        batch_loss_list,
        batch_acc_list,
    )


# 模型评估
def evaluate(model_path=str(CHECKPOINT_DIR / "TextClassificationModel_best.bin")):
    # 没有最佳模型时回退到最终模型
    if not os.path.exists(model_path):
        model_path = str(CHECKPOINT_DIR / "TextClassificationModel.bin")

    model = TextClassificationModel(bert_model).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_loss, test_acc = evaluate_model(
        model,
        test_dataloader,
        criterion=nn.CrossEntropyLoss(),
        desc="Testing",
    )
    print(f"test loss: {test_loss:.6f}, test acc: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    train()
    evaluate()
