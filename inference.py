from Tokenizer import CHARS, Tokenizer
from Transformer import Transformer
import torch


def load_model(path, device='cuda'):
    model = Transformer(
        vocab_size=len(CHARS),
        max_len=192,
        d_model=256,
        ffn_dim=1024,
        num_heads=8,
        dropout=0.1,
        num_layers=4
    )
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def singe_inference(formula, model, tokenizer, device='cuda', max_gen_len=64):
    model.eval()
    tokens = tokenizer(formula, add_special_tokens=True, is_training=False)
    # print(tokens)
    tensors = torch.tensor([tokens], dtype=torch.long, device=device)
    eos_id = tokenizer.stoi['[EOS]']
    with torch.no_grad():
        for _ in range(max_gen_len):
            if tensors.size(1) >= 144:
                tensors = tensors[:, -143:]

            logits = model(tensors)
            next_token_logits = logits[:, -1, :]
            # print(next_token_logits)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            tensors = torch.cat([tensors, next_token_id], dim=-1)

            if next_token_id.item() == eos_id:
                break

    generated_ids = tensors[0].tolist()
    return tokenizer.decode(generated_ids)

if __name__ == '__main__':
    formula = '4321-4320'
    tokenizer = Tokenizer()
    model = load_model('./best_model.pth')
    print(singe_inference(formula, model, tokenizer))