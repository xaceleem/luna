import os
import json
import warnings
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)

warnings.filterwarnings("ignore")

# Set environment variable to help with CUDA memory fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ------------------------------
# 1. Create or verify training dataset
# ------------------------------
TRAINING_DATA_PATH = "conversation_data.txt"
if not os.path.exists(TRAINING_DATA_PATH) or os.stat(TRAINING_DATA_PATH).st_size == 0:
    sample_data = ""
    with open(TRAINING_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(sample_data)
    print(f"Sample training data created at {TRAINING_DATA_PATH}")

# ------------------------------
# 2. Set up model and tokenizer
# ------------------------------
MODEL_NAME = "gpt2"  # Using GPT-2 Small for lower VRAM usage.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

FINE_TUNED_MODEL_DIR = "fine_tuned_model"
MODEL_WEIGHTS_PATH = os.path.join(FINE_TUNED_MODEL_DIR, "pytorch_model.bin")

"""def convert_csv_to_txt(csv_file_path, txt_file_path):

    with open(csv_file_path, "r", encoding="utf-8") as csv_file:
        lines = csv_file.readlines()
    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        for line in lines:
            txt_file.write(line.strip() + "\n")
            
            """
# ------------------------------
# 3. Fine-tuning or loading the model
# ------------------------------
if not os.path.exists(FINE_TUNED_MODEL_DIR) or not os.path.exists(MODEL_WEIGHTS_PATH):
    print("No valid fine-tuned model found. Starting fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)

    def load_dataset(file_path, tokenizer, block_size=128):
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size,
            overwrite_cache=True
        )

    train_dataset = load_dataset(TRAINING_DATA_PATH, tokenizer, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=FINE_TUNED_MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Ensure the configuration contains the correct model_type.
    model.config.model_type = "gpt2"
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)

    # Update config.json explicitly
    config_path = os.path.join(FINE_TUNED_MODEL_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["model_type"] = "gpt2"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Fine-tuning complete. Model saved to {FINE_TUNED_MODEL_DIR}")
else:
    print(f"Loading fine-tuned model from {FINE_TUNED_MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)

# ------------------------------
# 4. Interactive Conversation Code
# ------------------------------
PERSONA = (
    "You are a friendly, engaging, and thoughtful conversational partner. "
    "Keep your responses very very short, natural, and supportive."
)

def clean_output(text):
    """Remove unwanted content such as URLs and offensive words."""
    text = re.sub(r"http\S+", "", text)
    banned_words = []
    for word in banned_words:
        text = re.sub(word, "[censored]", text, flags=re.IGNORECASE)
    return text.strip()

def generate_response(conversation_history, user_input, max_length=100):
    conversation_history += f"User: {user_input}\n"
    prompt = f"{PERSONA}\n{conversation_history}Friend:"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_length = inputs.input_ids.shape[1]

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()

    outputs = model.generate(
        **inputs,
        max_length=prompt_length + max_length,
        do_sample=True,
        temperature=0.4,
        top_k=25,
        top_p=0.96,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_response = generated_text[len(prompt):].strip()
    if "User:" in new_response:
        new_response = new_response.split("User:")[0].strip()

    new_response = clean_output(new_response)
    conversation_history += f"Friend: {new_response}\n"
    return new_response, conversation_history

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("Warning: GPU not available. Running on CPU.")
    
    conversation_history = ""
    print("Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        response, conversation_history = generate_response(conversation_history, user_input)
        print(f"Luna: {response}")
