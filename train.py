from transformers import get_cosine_schedule_with_warmup

from Transformer import Transformer
from Tokenizer import Tokenizer, CHARS
import json
import torch
from torch import nn
from tqdm import tqdm


def load_data(json_path, tokenizer, max_len=None, device="cpu"):
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_formulas = json.load(f)

    pad_id = tokenizer.stoi['[PAD]']
    all_ids = []

    if max_len is None:
        raw_id_list = []
        max_physical_len = 0

        for formula_str in tqdm(raw_formulas, desc="Pre-scanning text to find max_len"):
            ids = tokenizer.encode(formula_str, add_special_tokens=True)
            raw_id_list.append(ids)
            if len(ids) > max_physical_len:
                max_physical_len = len(ids)
        max_len = ((max_physical_len + 7) // 8) * 8
        print(f"Auto-detected max len: {max_physical_len} -> Aligned max len = {max_len}")
        for ids in tqdm(raw_id_list, desc="Padding sequences"):
            if len(ids) < max_len:
                padded_ids = ids + [pad_id] * (max_len - len(ids))
            else:
                padded_ids = ids[:max_len]
            all_ids.append(padded_ids)

    else:
        print(f"Using fixed max_len = {max_len}. Skipping pre-scanning.")
        for formula_str in tqdm(raw_formulas, desc="Processing and padding sequences"):
            ids = tokenizer.encode(formula_str, add_special_tokens=True)

            if len(ids) < max_len:
                padded_ids = ids + [pad_id] * (max_len - len(ids))
            else:
                padded_ids = ids[:max_len]
            all_ids.append(padded_ids)

    dataset_tensor = torch.tensor(all_ids, dtype=torch.long).to(device)
    print(f"Data loaded successfully. Shape: {dataset_tensor.shape}\n")
    return dataset_tensor, max_len


def train(save_path, device="cpu"):
    tokenizer = Tokenizer()
    max_len = 192
    x, _ = load_data('dataset/dataset.json', tokenizer, device=device, max_len=max_len)
    data_size = x.size(0)
    train_size = int(data_size * 0.9)

    train_x = x[:train_size]
    val_x = x[train_size:]
    batch_size = 512
    epochs = 50

    model = Transformer(
        vocab_size=len(CHARS),
        max_len=max_len,
        d_model=256,
        ffn_dim=1024,
        num_heads=8,
        dropout=0.1,
        num_layers=4
    )
    model.to(device)

    pad_id = tokenizer.stoi['[PAD]']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    train_num = (train_x.size(0) + batch_size - 1) // batch_size
    num_training_steps = train_num * epochs
    num_warmup_steps = int(num_training_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using Mixed Precision Dtype: {amp_dtype}")
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        bar = tqdm(range(0, train_x.size(0), batch_size), desc=f"Epoch {epoch + 1}/{epochs} [Train]", total=train_num)

        for i in bar:
            batch_data = train_x[i: i + batch_size].to(device)
            x_batch = batch_data[:, :-1].contiguous()
            y_batch = batch_data[:, 1:].contiguous()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            bar.set_postfix({"Train Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})

        model.eval()
        val_loss = 0.0
        val_num = (val_x.size(0) + batch_size - 1) // batch_size

        total_correct_tokens = 0
        total_valid_tokens = 0

        with torch.no_grad():
            for i in range(0, val_x.size(0), batch_size):
                batch_data = val_x[i: i + batch_size].to(device)
                x_batch = batch_data[:, :-1].contiguous()
                y_batch = batch_data[:, 1:].contiguous()

                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                non_pad_mask = (y_batch != pad_id)
                correct = (preds == y_batch) & non_pad_mask
                total_correct_tokens += correct.sum().item()
                total_valid_tokens += non_pad_mask.sum().item()

        avg_val_loss = val_loss / val_num
        val_acc = (total_correct_tokens / total_valid_tokens) * 100 if total_valid_tokens > 0 else 0.0

        print(f"Epoch {epoch + 1} Val:")
        print(f"Val Avg Loss: {avg_val_loss:.4f}")
        print(f"Token Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Best Model Found! Saved to {save_path} (Best Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"EarlyStopping Counter: {patience_counter}/{patience}")
        print("-" * 30 + "\n")
        if patience_counter >= patience:
            print(
                f"[Early Stopping] Patience: {patience}")
            break

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train("./best_model.pth", device="cuda")