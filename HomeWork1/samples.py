import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "output_dir/gpt2-1b-russian/checkpoint-101"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
model.to("mps").eval()

prompts = [
    "Москва — столица",
    "Александр Пушкин родился",
    "В 1945 году",
    "Российская Федерация",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(prompt)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print()