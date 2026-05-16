from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")

# REPLACE THIS LIST WITH YOUR C++ OUTPUT
output_ids = [464, 3225, 286, 4881, 318, 286, 318, 3225, 286, 8031, 318, 3225, 286, 8602, 318] 

print(tok.decode(output_ids))
