#### Documentation for the Development of a Financial Chatbot with LLM Models

This project aims to build a chatbot based on LLM models in Spanish to answer queries related to financial products for a banking entity. The chatbot should process documentation in diverse PDF formats, convert it into structured data, and provide accurate responses to questions posed by employees and customers.

**[También en Español](https://github.com/Leci37/LLM-Financial-Chatbot-Spanish-Training-and-Development-from-Raw-PDFs/blob/main/README_ESPA%C3%91OL.md)**


## **Project Scope**

- **Chatbot Features:**
  - Ability to answer questions about: **International Transfers**, **Investment Funds**, **Active Loans** and **Investment Options**

- **Data Sources:**
  - Provided PDF documents in diverse formats, organized in the `raw/` folder:
    - `W1_Tarifas transferencias Extranjero.pdf`
    - `W2_Ficha Tecnica.pdf`
    - `W3_Catalogo de productos de activo vigentes.pdf`
    - `W4_2023_12_Posicionamiento_Environment-1.pdf`
    - `W5_2023_12_Posicionamiento_Environment-2.pdf`

- **Data Processing:**
  - Conversion of PDF documents into `.docx` format to facilitate extraction.
  - Transformation into JSON structures, with data categorized into:
    - Text Sections: titles and content.
    - Structured Tables: titles and organized data.

## **Example Questions Answered**
the answers to these questions are found inside the .pdf in various formats such as tables
1. **Transfers outside the Euro Zone:**
   - ¿Puedes calcularme las comisiones para una transferencia fuera de la Zona €, de 10.000€ con las comisiones a cargo del socio?
   _- Can you calculate the fees for a transfer outside the Euro Zone of €10,000 with the fees borne by the partner?_

3. **Investment Funds:**
   - ¿Qué nivel de riesgo tiene el fondo de inversión, CI Environment ISR?
   _- What is the risk level of the investment fund CI Environment ISR?_

5. **Active Loans:**
   - ¿Qué requisitos hay que cumplir para solicitar un préstamo postgrado?
   _- What are the requirements to apply for a postgraduate loan?_

This chatbot will be a valuable tool for employees and customers, simplifying access to financial information and enhancing user experience.


## **0. Preliminary Evaluation and Transformation (from .pdf to .docx)**

### **Initial Document Evaluation:**
- The downloaded `.pdf` documents contain:
  - Structured text sections (descriptive paragraphs).
  - Tables with various formats and financial data.
- The documents exhibit a wide variety of styles and formats, making uniform extraction challenging using a single method.

### **Conversion to .docx Format:**
- Multiple tools were evaluated, with **iLovePDF** selected for its high-quality conversion, particularly in preserving tables.

### **Manual Validation of the Conversion:**
- Verified that the tables were correctly represented.

### **Documents converted to `.docx` are:**
```python
WORD_LIST = [
    "raw/W1_Tarifas transferencias Extranjero.docx",
    "raw/W2_Ficha Tecnica.docx",
    "raw/W3_Catalogo de productos de activo vigentes.docx",
    "raw/W4_2023_12_Posicionamiento_Environment-1.docx",
    "raw/W5_2023_12_Posicionamiento_Environment-2.docx"
]
```

## **1. Data Extraction (from docx to .json)**

### **What is spaCy NLP and why is it used for classification?**
```python
import spacy 
nlp = spacy.load("es_core_news_md")
```
The **spaCy** model is useful for classifying content because it analyzes the text and assigns relevant semantic and grammatical information, facilitating tasks such as:

- Detects important elements in a text and classifies them into predefined categories:
  - **MONEY**: Monetary amounts.
  - **DATE**: Dates.
  - **ORG**: Organizations.
  - **PERCENT**: Percentages.

### **Objective**
- Convert the content of `.docx` files into a structured `.json` format.  
  The structure is divided into:
  - **Paragraphs:** Title and content.
  - **Tables:** Title and data (the first row of data corresponds to the column name).

Example of JSON structure:
```json
{
    "sections": [
        {
            "title": "Título de la sección",
            "content": "Contenido relacionado con el título."
        }
    ],
    "tables": [
        {
            "title": "Título de la tabla",
            "data": [
                ["Columna1", "Columna2", "Columna3"],
                ["Dato1", "Dato2", "Dato3"]
            ]
        }
    ]
}
```

## **2. Question and Answer Generation**

### **Tools Used**
- `_2_1_Generate Prompts_sections.py`: To generate questions from sections (paragraphs).
- `_2_2_Generate Prompts_table.py`: To generate questions based on tables.

### **Question Generation for Sections-Paragraphs**

#### **Method `generate_questions_by_NLP()`**
- Generation based on the section title:
  - These are generic, despite NLP-based thematic classification, and may result in "robotic" sounding questions:
    ```python
    f"¿Qué información general se describe en la sección '{title}'?"
    ```
- Customization based on detected entities (using NLP):
  - For money-related questions, a specific nuance is provided:
    ```python
    if ent.label_ == "MONEY":
        prompts.append(f"¿Cuáles son los costos asociados con {ent.text} mencionados en '{title}'?")
        # What are the costs associated with {ent.text} mentioned in '{title}'?
    ```
- List of topics with specific questions:
  ```python
  entity_labels = [
      "MONEY",    # Costs, fees, and financial calculations.
      "ORG",      # Organizations, services, and products.
      "PRODUCT",  # Products, features, and benefits.
      "LOC",      # Locations and regulations.
      "PERCENT",  # Percentages, calculations, and implications.
      "DATE",     # Dates, events, and temporal relevance.
      "TIME",     # Timelines and schedules.
      "EVENT",    # Events and associated details.
      "LAW",      # Regulations and legal requirements.
      "QUANTITY", # Quantities and calculations.
      "CARDINAL"  # Numbers and their contextual interpretation.
  ]
  ```

#### **Method `generate_questions_by_Model()`**
- Questions generated with the `mrm8488/bert2bert-spanish-question-generation` model:
  - Uses the section title and paragraph as input.
  - Automatically generates questions. **Note:** This model is suitable for educational purposes but not recommended for production.

### **Validation of Generated Questions**
- Questions whose NLP topic does not match the answer are discarded via:
  ```python
  validate_prompt_response_NLP()
  ```

### **Question Generation for Tables**

#### **1. `generate_General_table_questions`**
- Generates general questions about an entire table.
- **Note:** Discarded due to inefficiency in summaries using the `bert2bert_shared-spanish-finetuned-summarization` model.

#### **2. `generate_ROW_specific_questions`**
- Generates specific questions for a table row, combining:
  - Row data: Associated properties and values.
  - NLP-based questions: Detects relevant entities such as percentages, dates, locations.
  - Questions generated by models:
    ```python
    if ent.label_ == "MONEY":
        nlp_questions.append({
            "prompt": f"¿Qué costos están relacionados con el elemento '{row_reference}' en la tabla '{title}'?",
    # f"What costs are related to the element '{row_reference}' in table '{title}'?"
            "response": response_text
        })
    ```

#### **3. `generate_CELL_specific_questions`**
- Generates specific questions for a table cell, using:
  - **Cell Data:** Cell value with row and column headers.
  - **NLP-based questions:** Analyzes detected entities.
  - **Model-generated questions:** Crosses data between columns and rows:
    ```python
    if ent.label_ == "MONEY":
        nlp_questions.append({
            "prompt": f"¿Qué costos están relacionados con '{column_reference}' en relación con '{row_reference}'?",
           # f"What costs are related to '{column_reference}' relative to '{row_reference}'?"
            "response": response_text_CELL
        })
    ```

### **Validation `validate_description()`**
- Thematic verification between questions and answers using NLP.

### **Final Merge**
- The `_2_generated_prompts_FULL.json` file was created by combining:
  - `_2_generated_prompts_SECTIONS.json`
  - `_2_generated_prompts_TABLE.json`


## **3. Validation of the .json**

### **Validation of `_2_generated_prompts_FULL.json`**
- Verified the correspondence between questions and answers based on keywords and thematic alignment:
  ```python
  doc_prompt = nlp(prompt)
  doc_response = nlp(response)
  keywords_prompt = {token.text.lower() for token in doc_prompt if not token.is_stop and token.is_alpha}
  keywords_response = {token.text.lower() for token in doc_response if not token.is_stop and token.is_alpha}
  ```

#### **Observations**
- Requires manual improvements or the use of more advanced language models.

## **4. Training**

### **Models Used**
- `google/mt5-base`
- `google/mt5-small`
- `meta-llama/Llama-2-7b`
- `bigscience/bloom-560m`

### **Training Process for BLOOM**

#### **Data Validation and Preprocessing**

1. **JSON Validation**:
   - Ensures each entry in the JSON file contains the keys `prompt` and `response`.
   ```python
   def validate_json(data):
       for entry in data:
           if "prompt" not in entry or "response" not in entry:
               raise ValueError("Each entry must contain 'prompt' and 'response' keys.")
   ```

2. **Conversion to Dataset**:
   - The validated JSON is converted into a Hugging Face-compatible dataset using the `{input_text, output_text}` format.

3. **Tokenization**:
   - Input texts (prompt) and output texts (response) are tokenized with truncation and padding up to a maximum of 512 tokens.
   ```python
   def preprocess_function(examples):
       inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
       outputs = tokenizer(examples["output_text"], max_length=512, truncation=True, padding="max_length")
       inputs["labels"] = outputs["input_ids"]
       return inputs
   ```

##### **Metric Evaluation**

- Metrics calculated on validation and test sets:
  - **BLEU**: Measures precision in n-gram matches between generated and reference text.
  - **ROUGE**: Evaluates recall-based similarities between generated and reference sequences.
    - Calculated metrics: `rouge1`, `rouge2`, `rougeL`.
  - **Loss**: Based on cross-entropy loss between model predictions and true labels.
  ```python
  def compute_metrics(eval_pred):
      logits, labels = eval_pred 
      metrics = {
          "bleu": bleu["bleu"],
          "rouge1": rouge["rouge1"],
          "rouge2": rouge["rouge2"],
          "rougeL": rouge["rougeL"],
      }
      return metrics
  ```

#### **Model Training**

1. **Training Arguments**:
   - Key parameters like learning rate, batch size, number of epochs, use of FP16 (mixed precision), and steps for saving checkpoints.
   ```python
   training_args = TrainingArguments(
       output_dir="./m-bloom_560",
       evaluation_strategy="epoch",
       eval_steps=50,
       per_device_train_batch_size=4,
       learning_rate=2e-5,
       save_steps=50,
       num_train_epochs=3,
       fp16=True,
   )
   ```

2. **Device**:
   - The model is trained on GPU if available (`cuda`), otherwise on CPU.

3. **Callbacks**:
   - **DebugCallback**: Displays training logs to monitor metrics and progress.
   - **EvaluationCallback**: Every 50 steps:
     - Saves a model checkpoint.
     - Generates responses for evaluation questions and displays them in the console.

#### **Final Validation**
- **Script Used**: `_5_evaluate_mT5_small.py`.
- **Tasks Performed**:
  - Final tests on responses generated by the models.
  - Evaluation of the consistency and naturalness of the responses.

## **Final Observations**
- For educational purposes, the result is adequate. At an **industrial level, significant improvements** are needed in the creation and validation of synthetic questions/answers, which currently sound robotic.

## **5. Improvements**

-  **Increase Dataset Size**
Include greater linguistic variability to avoid repetitive and robotic responses.
Use tools like **ChatGPT API (GPT-4)** or **Llama Index** to generate more natural and diverse questions/answers.

-  **Implement Semantic Preprocessing Models**
**spaCy** or **Sentence-BERT**:
  Detect and eliminate redundancies in the responses.
  Refine the format of the questions/answers.

-  **Semantic Analysis Tools**
**AllenNLP** or **Hugging Face NLP Pipelines**:
  Evaluate the thematic correspondence between questions and answers.

-  **Human Evaluation**
Incorporate manual review cycles to ensure quality in the generated questions/answers.

-  **Migrate to Larger or Specialized Models**
Use models like **GPT-4**, **Llama-2**, or **Claude** for generating more natural responses.

-  **Adjust Generation Techniques**
Apply methods such as:
  **Top-k Sampling**
  **Nucleus Sampling (top-p)**
  **Temperature**
  Improve variability without losing coherence.

-  **Recommended Tools**
1. **AllenNLP**: For detailed analysis of semantic dependencies in questions/answers.
2. **Transformers Evaluate (Hugging Face)**: To implement advanced metrics like `BERTScore` and `METEOR`.


-  **Dynamic Calculations in the Chatbot**
The current model only returns generic information such as:
  _"The commission for France is 2%."_  
**Objective**: Allow dynamic calculations to generate processed results:
  ```
  Pregunta: "¿Cuál es la comisión de transferencias para Francia de 1000€?"
  Respuesta: "La comisión para Francia es 2%, lo que equivale a: 1000€ × 2% = 20€."
  # Question: "What is the transfer commission for France on €1,000?"
  # Answer: "The commission for France is 2%, which equals: €1,000 × 2% = €20."
  ```
-  **Implementation with LangChain**:
  Analyzes the question to identify if specific calculations are required.
  Redirects necessary calculations to a Python environment.
