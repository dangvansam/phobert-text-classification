from vncorenlp import VnCoreNLP
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import tqdm
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW


rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
# text = "xin chào việt nam"

# word_segmented_text = rdrsegmenter.tokenize(text) 
# print(word_segmented_text)

parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="vinai/phobert-base/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
    )
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("vinai/phobert-base/vocab.txt")

classes  = ['__label__sống_trẻ', '__label__thời_sự', '__label__công_nghệ', '__label__sức_khỏe', '__label__giáo_dục', '__label__xe_360', '__label__thời_trang', '__label__du_lịch', '__label__âm_nhạc', '__label__xuất_bản', '__label__nhịp_sống', '__label__kinh_doanh', '__label__pháp_luật', '__label__ẩm_thực', '__label__thế_giới', '__label__thể_thao', '__label__giải_trí', '__label__phim_ảnh']

train_path = 'train.txt'
test_path = 'test.txt'

train_text, train_labels = [], []
test_text, test_labels = [], []

# print("LOADING TRAIN SET")
# with open(train_path, 'r') as f_r:
#     for sample in f_r:
#         splits = sample.strip().split(" ",1)
#         # id = splits[0]
#         label = classes.index(splits[0])
#         text = splits[1]
#         # print(text)
#         #text = rdrsegmenter.tokenize(text)
#         # text = ' '.join([' '.join(x) for x in text])
#         # train_id.append(id)
#         train_text.append(text)
#         train_labels.append(label)

# print("LOADDING TEST SET")
# with open(test_path, 'r') as f_r:
#     for sample in f_r:
#         splits = sample.strip().split(" ",1)
#         # id = splits[0]
#         label = classes.index(splits[0])
#         text = splits[1]
#         # text = rdrsegmenter.tokenize(text)
#         # text = ' '.join([' '.join(x) for x in text])
#         test_labels.append(label)
#         test_text.append(text)

# print("LOAD DATA DONE")

MAX_LEN = 256

# print("TRAIN TEXT TO IDS")
train_ids = []
val_ids = []

# for sent in train_text:
#     subwords = '<s> ' + bpe.encode(sent) + ' </s>'
#     encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
#     train_ids.append(encoded_sent)

# print("TEST TEXT TO IDS")
# for sent in test_text:
#     subwords = '<s> ' + bpe.encode(sent) + ' </s>'
#     encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
#     val_ids.append(encoded_sent)

# with open("train_ids", 'wb') as f:
#     pickle.dump(train_ids, f)
# with open("train_labels", 'wb') as f:
#     pickle.dump(train_labels, f)

# with open("val_ids", 'wb') as f:
#     pickle.dump(val_ids, f)
# with open("test_labels", 'wb') as f:
#     pickle.dump(test_labels, f)


with open("train_ids", 'rb') as f:
    train_ids = pickle.load(f)
with open("train_labels", 'rb') as f:
    train_labels = pickle.load(f)
with open("val_ids", 'rb') as f:
    val_ids = pickle.load(f)
with open("test_labels", 'rb') as f:
    test_labels = pickle.load(f)

print("PADDING TRAIN IDS")
train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
print("PADDING TEST IDS")
val_ids = pad_sequences(val_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")


print("CREATE TRAIN MAKS")
train_masks = []
for sent in train_ids:
    mask = [int(token_id > 0) for token_id in sent]
    train_masks.append(mask)

print("CREATE TEST MAKS")
val_masks = []
for sent in val_ids:
    mask = [int(token_id > 0) for token_id in sent]
    val_masks.append(mask)

print("CONVERT TO TORCH TENSOR")
train_inputs = torch.tensor(train_ids)
val_inputs = torch.tensor(val_ids)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(test_labels)
train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)

print("CREATE TRAIN DATALOADER")
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)

print("CREATE TEST DATALOADER")
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=16)

print("train_dataloader:", len(train_dataloader))
print("val_dataloader:", len(val_dataloader))


config = RobertaConfig.from_pretrained(
    "vinai/phobert-base/config.json",
    from_tf=False,
    num_labels = len(classes),
    output_hidden_states=False,
    )

print("LOAD BERT PRETRAIN MODEL")
BERT_SA = RobertaForSequenceClassification.from_pretrained(
    "vinai/phobert-base/pytorch_model.bin",
    config=config
    )

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    F1_score = f1_score(pred_flat, labels_flat, average='macro')
    return accuracy_score(pred_flat, labels_flat), F1_score


device = torch.device('cuda:1')
epochs = 10

BERT_SA.to(device)

# print(BERT_SA)

def save_checkpoint(save_path, model):
    if save_path is None:
        return
    
    state_dict = {
                     'model_state_dict': model.state_dict(),
                 }
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    if load_path is None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])

param_optimizer = list(BERT_SA.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)


for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    BERT_SA.train()
    train_accuracy = 0
    nb_train_steps = 0
    train_f1 = 0
    best_valid_loss = 999999
    best_eval_accuracy = 0
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # print(b_labels)
        
        BERT_SA.zero_grad()
        outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask, 
            labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        tmp_train_accuracy, tmp_train_f1 = flat_accuracy(logits, label_ids)
        train_accuracy += tmp_train_accuracy
        train_f1 += tmp_train_f1
        nb_train_steps += 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(BERT_SA.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print("[TRAIN] Epoch {}/{} | Batch {}/{} | Train Loss={} | Train Acc={}".format(epoch_i, epochs, step, len(train_dataloader), loss.item(), tmp_train_accuracy))
        
    avg_train_loss = total_loss / len(train_dataloader)
    print(" Train Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
    print(" Train F1 score: {0:.4f}".format(train_f1/nb_train_steps))
    print(" Train Loss: {0:.4f}".format(avg_train_loss))

    print("Running Validation...")
    BERT_SA.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_f1 = 0
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask,
            labels=b_labels
            )
            tmp_eval_loss, logits = outputs[0], outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()

            tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_loss += tmp_eval_loss
            eval_f1 += tmp_eval_f1
            nb_eval_steps += 1

    print(" Valid Loss: {0:.4f}".format(eval_loss/nb_eval_steps))
    print(" Valid Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print(" Valid F1 score: {0:.4f}".format(eval_f1/nb_eval_steps))

    if best_valid_loss > eval_loss:
        best_valid_loss = eval_loss
        save_checkpoint("checkpoints/model_{}_valloss{:.3f}.pt".format(epoch_i, best_valid_loss), model)
    if best_eval_accuracy > eval_accuracy:
        best_eval_accuracy = eval_accuracy
        save_checkpoint("checkpoints/model_{}_acc{:.2f}.pt".format(epoch_i, best_eval_accuracy*100), model)

print("Training complete!")
