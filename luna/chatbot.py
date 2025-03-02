import os
import json
import warnings
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

warnings.filterwarnings("ignore")

# Set environment variable to help with CUDA memory fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ------------------------------
# 1. Set the training dataset path
# ------------------------------
# Use the absolute path or the relative path as needed:
TRAINING_DATA_PATH = "/home/xleem/luna/luna/conversation_data.txt"
# Alternatively, if running from /home/xleem/luna/, you might use:
# TRAINING_DATA_PATH = "luna/conversation_data.txt"

if not os.path.exists(TRAINING_DATA_PATH) or os.stat(TRAINING_DATA_PATH).st_size == 0:
    # Create sample conversation data if file is missing or empty.
    # Ensure there is a blank line between conversation blocks.
    sample_data = (
        "User: Hi, I'm feeling really overwhelmed today.\n"
        "Friend: I'm really sorry to hear that. What's on your mind?\n"
        "User: Everything feels chaotic and unmanageable.\n"
        "Friend: It sounds really hard. Maybe we can break it down into smaller parts.\n\n"
        "User: I'm anxious about my future.\n"
        "Friend: That sounds tough. What worries you the most?\n"
        "User: I'm not sure if I'll find the right job.\n"
        "Friend: Your feelings are valid. Taking one step at a time can ease that worry.\n\n"
        "User: I feel so lonely these days.\n"
        "Friend: That must be really painful. I'm here to listen.\n"
        "User: I don't have many friends, and I feel isolated.\n"
        "Friend: Sometimes reaching out, even in small ways, can help you feel more connected.\n"
    )
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

# ------------------------------
# 3. Custom Conversation Dataset Loader with Debugging
# ------------------------------
def load_conversation_dataset(file_path, tokenizer, block_size=512):
    """
    Loads conversation data by splitting the file into samples using blank lines as delimiters.
    Debug prints are added to help inspect the raw text and the resulting samples.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    # Debug: print the raw text (with escape sequences visible).
    print("Raw text read from file:")
    print(repr(text))
    
    # Split text on one or more blank lines (handles Unix and Windows newlines).
    samples = [sample.strip() for sample in re.split(r'\r?\n\s*\r?\n', text) if sample.strip()]
    
    # Debug: print the samples found.
    print(f"Found {len(samples)} conversation sample(s):")
    for i, sample in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        print(repr(sample))
    
    # Create a Dataset from the conversation samples.
    dataset = Dataset.from_dict({"text": samples})
    
    # Tokenize each sample.
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=block_size)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_dataset

# ------------------------------
# 4. Fine-tuning or loading the model
# ------------------------------
if not os.path.exists(FINE_TUNED_MODEL_DIR) or not os.path.exists(MODEL_WEIGHTS_PATH):
    print("No valid fine-tuned model found. Starting fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
    
    # Use our conversation loader.
    train_dataset = load_conversation_dataset(TRAINING_DATA_PATH, tokenizer, block_size=512)
    print(f"Loaded dataset with {len(train_dataset)} samples.")

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

    # Remove unrecognized "loss_type" from config if it exists.
    if "loss_type" in model.config.__dict__:
        del model.config.__dict__["loss_type"]

    # Ensure the configuration contains the correct model_type.
    model.config.model_type = "gpt2"
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)

    # Update config.json to reflect changes.
    config_path = os.path.join(FINE_TUNED_MODEL_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["model_type"] = "gpt2"
    config.pop("loss_type", None)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Fine-tuning complete. Model saved to {FINE_TUNED_MODEL_DIR}")
else:
    print(f"Loading fine-tuned model from {FINE_TUNED_MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)

# ------------------------------
# 5. Interactive Conversation Code
# ------------------------------
PERSONA = (
    "You are a friendly, engaging, and thoughtful conversational partner. "
    "Keep your responses very very short, natural, and supportive."
)

def clean_output(text):
    """Remove unwanted content such as URLs and offensive words."""
    text = re.sub(r"http\S+", "", text)
    banned_words = []  # Add banned words if needed.
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
