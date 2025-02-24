from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

model = AutoModelForCausalLM.from_pretrained("gpt2", deivce_map='auto', trust_remote_model=True)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    traget_modules=["query", "value"],
    gradient_checkpointing=True,
    modules_to_save=["classifier"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
