import os
os.environ["WANDB_DISABLED"] = "true"
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    pipeline
)
from nltk.translate.bleu_score import sentence_bleu
import torch

# Validate JSON structure
def validate_json(data):
    for entry in data:
        if "prompt" not in entry or "response" not in entry:
            raise ValueError("Each entry must contain 'prompt' and 'response' keys.")
        if not isinstance(entry["prompt"], str) or not isinstance(entry["response"], str):
            raise ValueError("Both 'prompt' and 'response' must be strings.")

# Load and validate JSON file
with open("processed/_2_generated_prompts_FULL.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    validate_json(data)

# Convert to Hugging Face Dataset
formatted_data = [{"input_text": entry["prompt"], "output_text": entry["response"]} for entry in data]
dataset = Dataset.from_list(formatted_data)

# Load model and tokenizer
model_name = "bigscience/bloom-560m"  # Replace with desired BLOOM variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize dataset
def preprocess_function(examples):
    inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    outputs = tokenizer(examples["output_text"], max_length=512, truncation=True, padding="max_length")
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Evaluation questions
evaluation_questions = [
    "¿Puedes calcularme las comisiones para una transferencia fuera de la Zona €, de 10.000€ con las comisiones a cargo del socio?",
    "¿Puedes indicarme el coste de recibir una transferencia de 5.000€ en yenes?",
    "¿Noruega tiene convenio con la UE?",
    "¿Qué nivel de riesgo tiene el fondo de inversión, CI Environment ISR?",
    "¿Qué exposición tiene el fondo de inversión, CI Environment ISR en RV?",
    "¿Qué Rating tiene el CI Environment ISR?",
    "¿Qué requisitos hay que cumplir para solicitar un préstamo postgrado?",
    "¿Importe máximo préstamo consumo?",
    "¿Cuál es el coste de apertura de un Préstamo ECO Rehabilita Comunidad de Propietarios a 10 años?",
    "¿Dónde puedo invertir 50000 € para obtener la máxima rentabilidad en 1 año? ¿Cuáles son los riesgos de esta inversión? ¿Cuáles son los intereses y tasas?",
]

# BLEU Evaluation
def calculate_bleu(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        score = sentence_bleu([ref.split()], pred.split())
        scores.append(score)
    return sum(scores) / len(scores)

# Debug Callback
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)

# Evaluation Callback
class EvaluationCallback(TrainerCallback):
    def __init__(self, questions, model, tokenizer, device, output_dir):
        self.questions = questions
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        # Trigger evaluation and save every 50 steps
        if state.global_step % 50 == 0:
            checkpoint_path = f"{self.output_dir}/checkpoint-{state.global_step}"
            print(f"\nSaving model checkpoint to {checkpoint_path} at step {state.global_step}")
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)

            print("\nEvaluating questions:")
            for i, question in enumerate(self.questions, start=1):
                inputs = self.tokenizer(
                    f"Pregunta: {question}",
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512
                ).to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5,
                    temperature=0.7,
                    repetition_penalty=2.0,
                    early_stopping=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Q{i}: {question}")
                print(f"Response: {response}")
                print("-" * 50)


# Training Arguments
training_args = TrainingArguments(
    output_dir="./m-bloom_560",
    evaluation_strategy="epoch",
    eval_steps=50,
    # save_steps=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    save_steps=50,
    save_total_limit=9,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./m-bloom_560/logs"
)

# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Trainer Initialization
# Include updated callback in the training loop
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    callbacks=[
        DebugCallback(),
        EvaluationCallback(
            evaluation_questions,
            model,
            tokenizer,
            device,
            output_dir="./m-bloom_560"
        ),
    ],
)

# Train the Model
trainer.train()

# Train the Model
trainer.train()

# Save the Model
trainer.save_model("./m-bloom_560")
tokenizer.save_pretrained("./m-bloom_560")
