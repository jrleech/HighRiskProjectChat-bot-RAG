from peft import LoraModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B")

# Load the model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")
lora_config = LoraConfig(target_modules=["q_proj", "v_proj"], r=16, lora_alpha=32)
model = get_peft_model(model, lora_config)

# Load your dataset
dataset = load_dataset("traintext", "traintext-2-raw-v1") 
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Tokenize  data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# set the format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

#Define training args
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save checkpoints
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch
    learning_rate=5e-5,             # Learning rate
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,   # Batch size for evaluation
    num_train_epochs=3,             # Number of epochs
    weight_decay=0.01,              # Weight decay for regularization
    save_strategy="epoch",          # Save model at the end of each epoch
    logging_dir="./logs",           # Directory for logs
    logging_steps=10,               # Log every 10 steps
    save_total_limit=2,             # Limit the number of saved checkpoints
    fp16=True,                      # Use mixed precision (if supported by hardware)
    push_to_hub=False               # Set to True if pushing to Hugging Face Hub
)

# Fine-tune the model with Trainer API
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    tokenizer=tokenizer, 
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("path_to_save_model")
tokenizer.save_pretrained("path_to_save_model")

# Load the model for inference
model = LoraModel.from_pretrained("path_to_save_model")
input_text = "Your input text here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)

# Decode the output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)

