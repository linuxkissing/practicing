from transformers import AutoModelForCausalLM
from peft import get_peft_model, get_peft_config, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_model=True)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16, dropout=0.1, 
    target_modules=['query_key_value'],
)

model= get_peft_model(model, peft_config)
model.print_trainabel_parameters()