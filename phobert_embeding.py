import torch
from transformers import AutoModel, AutoTokenizer
import tensorflow as tf

phobert = AutoModel.from_pretrained("./vinai/phobert-base", local_files_only=True).to(torch.device("cuda:0"))
# phobert.save_pretrained("vinai/phobert-base") #lưu model sau khi tải về ổ cứng
# For transformers v4.x+: 
tokenizer = AutoTokenizer.from_pretrained("./vinai/phobert-base", use_fast=False, local_files_only=True)
# tokenizer.save_pretrained("vinai/phobert-base") #lưu model sau khi tải về ổ cứng
MAX_LEN = 256
print("LOAD phoBERT DONE")

def get_emb_vector(input_ids):
    input_ids  = torch.tensor([input_ids]).to(torch.long)
    with torch.no_grad():
        features = phobert(input_ids.to(torch.device("cuda:0")))
    #print(features)
    emb_vecs = features[0].cpu().numpy()[0]#[1:-1]
    #print(emb_vecs)
    return emb_vecs

def text2ids(text):
    # print(tokenizer.encode("<pad> nhà <pad>"))
    tkz = tokenizer.encode(text)
    return tkz