import numpy as np
import torch
from tqdm import tqdm

from utils import classifiction_metric


def train(epoch_num, n_gpu, model, train_dataloader, dev_dataloader,
          optimizer, criterion, gradient_accumulation_steps, device, label_list,
          output_model_file, output_config_file, early_stop):
    early_stop_times = 0

    best_auc = 0
    global_step = 0
    for epoch in range(int(epoch_num)):

        if early_stop_times >= early_stop:
            break

        print(f'---------------- Epoch: {epoch + 1:02} ----------')

        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) if t is not None else t for t in batch)
            input_ids, token_type_ids, c2w, w2s, input_mask, char_mask, word_mask, sentence_mask, label_ids = batch
            logits = model(input_ids, [c2w, w2s], input_mask, char_mask, word_mask, sentence_mask,
                           token_type_ids=token_type_ids)
            loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            train_steps += 1

            loss.backward()

            # 用于画图和分析的数据
            epoch_loss += loss.item()
            preds = logits.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)
            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)

            # if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            train_loss = epoch_loss / train_steps

        # if global_step % print_step == 0 and global_step != 0:

        train_acc, train_report, train_auc = classifiction_metric(all_preds, all_labels, label_list)
        dev_loss, dev_acc, dev_report, dev_auc = evaluate(model, dev_dataloader, criterion, device, label_list)

        if best_auc < dev_auc:
            best_auc = dev_auc
            model_to_save = model.module if hasattr(
                model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            early_stop_times = 0
        else:
            early_stop_times += 1


def evaluate(model, dataloader, criterion, device, label_list):
    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    epoch_loss = 0

    for batch in tqdm(dataloader, desc="Eval"):
        batch = tuple(t.to(device) if t is not None else t for t in batch)

        input_ids, token_type_ids, c2w, w2s, input_mask, char_mask, word_mask, sentence_mask, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, [c2w, w2s], input_mask, char_mask, word_mask, sentence_mask,
                           token_type_ids=token_type_ids)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        epoch_loss += loss.mean().item()

    acc, report, auc = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss / len(dataloader), acc, report, auc
