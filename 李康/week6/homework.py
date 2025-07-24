from transformers import BertModel
def cal_size(dict, para):
    total_size = 0
    total_para = 0
    for _, tensor in state_dict.items():
        total_size += tensor.nelement() * tensor.element_size()
    for p in para:
        total_para += p.numel()
    return total_size, total_para

if __name__ == "__main__":
    model = BertModel.from_pretrained(r"./bert-base-chinese/bert-base-chinese")
    state_dict = model.state_dict()
    para = model.parameters()
    total_size, total_para = cal_size(state_dict, para)
    print(f"total_size:{total_size}(b), total_para:{total_para}")