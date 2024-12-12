import transformers
import datasets
import nltk
from rouge_score import rouge_scorer
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

print("Transformers version:", transformers.__version__)
print("Datasets version:", datasets.__version__)
print("NLTK version:", nltk.__version__)
print("Torch version:", torch.__version__)

# Load the dataset
data_file = "processed/_2_generated_prompts_FULL.json"
with open(data_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare the dataset
prompts = [entry["prompt"] for entry in data]
responses = [entry["response"] for entry in data]

# Split data into train/test sets
train_prompts, test_prompts, train_responses, test_responses = train_test_split(
    prompts, responses, test_size=0.2, random_state=42
)

train_dataset = Dataset.from_dict({"prompt": train_prompts, "response": train_responses})
test_dataset = Dataset.from_dict({"prompt": test_prompts, "response": test_responses})

# Tokenizer and Model
model_name = "meta-llama/Llama-2-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")

# Preprocess Dataset
def preprocess_data(examples):
    inputs = examples["prompt"]
    targets = examples["response"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_data, batched=True)
tokenized_test = test_dataset.map(preprocess_data, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./llama2-chatbot-spanish",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)

# Save the fine-tuned model
trainer.save_model("./llama2-chatbot-spanish")
tokenizer.save_pretrained("./llama2-chatbot-spanish")

# Define Evaluation Metrics
def evaluate_bleu(predictions, references):
    smooth = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
        for pred, ref in zip(predictions, references)
    ]
    return sum(bleu_scores) / len(bleu_scores)

def evaluate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    rouge_scores = {key: score / len(predictions) for key, score in rouge_scores.items()}
    return rouge_scores

# Test the Model
def generate_response(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Generate predictions for the test set
test_prompts = test_dataset["prompt"]
test_references = test_dataset["response"]
test_predictions = [generate_response(f"Pregunta: {prompt}", model, tokenizer) for prompt in test_prompts]

# Evaluate Predictions
bleu_score = evaluate_bleu(test_predictions, test_references)
rouge_scores = evaluate_rouge(test_predictions, test_references)

print("\nEvaluation Results:")
print(f"BLEU Score: {bleu_score:.4f}")
print(f"ROUGE Scores: {rouge_scores}")

# Save Evaluation Results
evaluation_results = {
    "metrics": metrics,
    "bleu_score": bleu_score,
    "rouge_scores": rouge_scores,
}
with open("evaluation_results.json", "w", encoding="utf-8") as file:
    json.dump(evaluation_results, file, ensure_ascii=False, indent=4)
