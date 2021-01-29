import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
    
phobert = AutoModel.from_pretrained("vinai/phobert-base")

# For transformers v4.x+: 
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# For transformers v3.x: 
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
line = "Do vậy cử tri đề nghị Bộ nghiên cứu trình Quốc hội xem xét sửa đổi bổ sung đồng bộ giữa Luật giao thông đường bộ và Luật xây dựng đối với lĩnh vực giao thông đường bộ"

input_ids = torch.tensor([tokenizer.encode(line)])

print(input_ids.shape)

with torch.no_grad():
    features = phobert(input_ids)

emb_vecs = features[0].cpu().numpy()[0][1:-1]
labels = line.split(" ")

print(emb_vecs.shape)
print(len(labels))
#Visualize
tsne_model = TSNE(perplexity=100, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(emb_vecs)

x = []
y = []

for value in new_values:
    x.append(value[0])
    y.append(value[1])
    
plt.figure(figsize=(16, 16)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.show()

