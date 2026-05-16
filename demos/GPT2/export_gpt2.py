import torch
import struct
import numpy as np
from transformers import GPT2LMHeadModel

def save_tensor(f, tensor):
    # Flatten and write float data
    data = tensor.detach().cpu().numpy().astype(np.float32)
    # Force contiguous if not
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    
    # Write Rank
    f.write(struct.pack('I', len(data.shape)))
    # Write Shape
    for s in data.shape:
        f.write(struct.pack('I', s))
    # Write Data
    f.write(data.tobytes())

def export():
    print("Loading GPT-2 from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Axon Binary Format Header (ASCII 'AXON')
    MAGIC = 0x41584F4E
    
    with open("gpt2_axon.bin", "wb") as f:
        f.write(struct.pack('I', MAGIC))
        f.write(struct.pack('I', 0)) # Placeholder for count

        count = 0
        
        # 1. WTE
        save_tensor(f, model.transformer.wte.weight)
        count += 1

        # 2. WPE
        save_tensor(f, model.transformer.wpe.weight)
        count += 1

        # 3. Blocks
        for i, block in enumerate(model.transformer.h):
            # LN 1
            save_tensor(f, block.ln_1.weight) # Gamma
            save_tensor(f, block.ln_1.bias)   # Beta
            count += 2

            # ATTENTION (Split HF Conv1D to Q, K, V)
            qkv_w = block.attn.c_attn.weight # (768, 2304)
            qkv_b = block.attn.c_attn.bias   # (2304)
            
            # HF Conv1D weights are (Hidden, 3*Hidden). Axon Linear is (In, Out).
            # So no transpose needed for shapes, just split.
            q_w, k_w, v_w = torch.split(qkv_w, 768, dim=1)
            q_b, k_b, v_b = torch.split(qkv_b, 768, dim=0)

            save_tensor(f, q_w); save_tensor(f, q_b)
            save_tensor(f, k_w); save_tensor(f, k_b)
            save_tensor(f, v_w); save_tensor(f, v_b)
            count += 6

            # Attn Output Proj
            save_tensor(f, block.attn.c_proj.weight)
            save_tensor(f, block.attn.c_proj.bias)
            count += 2

            # LN 2
            save_tensor(f, block.ln_2.weight)
            save_tensor(f, block.ln_2.bias)
            count += 2

            # MLP
            save_tensor(f, block.mlp.c_fc.weight)
            save_tensor(f, block.mlp.c_fc.bias)
            
            save_tensor(f, block.mlp.c_proj.weight)
            save_tensor(f, block.mlp.c_proj.bias)
            count += 4
            
            print(f"Saved Block {i}")

        # 4. Final LN
        save_tensor(f, model.transformer.ln_f.weight)
        save_tensor(f, model.transformer.ln_f.bias)
        count += 2

        # 5. LM Head
        # HF ties weights (wte == lm_head). 
        # But HF Linear weight is (Out, In) = (50257, 768).
        # Axon Linear expects (In, Out) = (768, 50257).
        # We must Transpose!
        save_tensor(f, model.lm_head.weight.t()) 
        
        # Bias (Explicit Zeros as HF GPT2 has no bias on head)
        save_tensor(f, torch.zeros(50257))
        count += 2
        
        print("Saved Final Layers")

        # Go back and write count
        f.seek(4)
        f.write(struct.pack('I', count))
        print(f"Done. Total Tensors: {count}")

if __name__ == "__main__":
    export()