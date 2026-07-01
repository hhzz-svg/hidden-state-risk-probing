import torch

PATH = "outputs/qwen05b/experiment1/experiment1_hidden_extraction/hidden_states_pilot.pt"

records = torch.load(PATH)

print("样本数:", len(records))

first = records[0]

print("第一条样本 keys:", first.keys())
print("第一条 prompt:", first["prompt"])
print("第一条 label:", first["label"])
print("第一条 type:", first["type"])
print("hidden shape:", first["hidden"].shape)

num_layers, hidden_dim = first["hidden"].shape

print("层数，包括 embedding 层:", num_layers)
print("hidden_dim:", hidden_dim)

for i in range(min(3, len(records))):
    print("-" * 50)
    print("id:", records[i]["id"])
    print("prompt:", records[i]["prompt"])
    print("label:", records[i]["label"])
    print("hidden norm of last layer:", records[i]["hidden"][-1].norm().item())