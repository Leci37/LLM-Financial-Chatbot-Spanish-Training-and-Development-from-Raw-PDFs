### Documentación para el Desarrollo de un Chatbot Financiero con Modelos LLM

Este proyecto tiene como objetivo construir un chatbot basado en modelos de **lenguaje LLM en español** para responder consultas relacionadas con productos financieros de una entidad bancaria. El chatbot debe procesar documentación en formatos PDF diversos, convertirla en datos estructurados y ofrecer respuestas precisas a preguntas formuladas por empleados y clientes.

## **Alcance del Proyecto**

- **Funcionalidades del Chatbot:**
  - Capacidad para responder preguntas sobre:
    - **Transferencias internacionales** , **Fondos de inversión**, **Préstamos activos** y **Opciones de inversión**

- **Fuente de Datos:**
  - Documentos PDF proporcionados en formatos heterogéneos, organizados en la carpeta `raw/`:
    - `W1_Tarifas transferencias Extranjero.pdf`
    - `W2_Ficha Tecnica.pdf`
    - `W3_Catalogo de productos de activo vigentes.pdf`
    - `W4_2023_12_Posicionamiento_Environment-1.pdf`
    - `W5_2023_12_Posicionamiento_Environment-2.pdf`

- **Procesamiento de Datos:**
  - Conversión de los documentos PDF a formato `.docx` para facilitar su extracción.
  - Transformación a estructuras JSON, con clasificación de datos en:
    - Secciones de texto: títulos y contenido.
    - Tablas estructuradas: títulos y datos organizados.

## **Ejemplo de Preguntas Respondidas**

1. **Transferencias fuera de la Zona €:**
   - ¿Puedes calcularme las comisiones para una transferencia fuera de la Zona €, de 10.000€ con las comisiones a cargo del socio?

2. **Fondos de Inversión:**
   - ¿Qué nivel de riesgo tiene el fondo de inversión, CI Environment ISR?

3. **Préstamos Activos:**
   - ¿Qué requisitos hay que cumplir para solicitar un préstamo postgrado?

Este chatbot será una herramienta útil para empleados y clientes, simplificando la obtención de información financiera y mejorando la experiencia del usuario.

## **0. Evaluación y Transformación Previa (de .pdf a .docx)**

### **Evaluación Inicial de los Documentos:**
- Los documentos descargados en formato `.pdf` contienen:
  - Secciones de texto estructurado (párrafos descriptivos).
  - Tablas con formatos diversos y datos financieros.
- Los documentos presentan gran variedad de estilos y formatos, lo que dificulta la extracción uniforme mediante un solo método.

### **Conversión a Formato .docx:**
- Se evaluaron múltiples herramientas, seleccionándose **iLovePDF** por la alta calidad en la conversión, especialmente en la preservación de tablas.

### **Validación Manual de la Conversión:**
- Se verificó que las tablas estuvieran correctamente representadas.

### **Los documentos convertidos a `.docx` son:**
```python
WORD_LIST = [
    "raw/W1_Tarifas transferencias Extranjero.docx",
    "raw/W2_Ficha Tecnica.docx",
    "raw/W3_Catalogo de productos de activo vigentes.docx",
    "raw/W4_2023_12_Posicionamiento_Environment-1.docx",
    "raw/W5_2023_12_Posicionamiento_Environment-2.docx"
]
```

## **1. Extracción de Datos (de docx a .json)**

### **¿Qué es spaCy NLP y por qué se usa para clasificar?**
```python
import spacy 
nlp = spacy.load("es_core_news_md")
```
El modelo de **spaCy** es útil para clasificar contenido porque analiza el texto y asigna información semántica y gramatical relevante, facilitando tareas como:

- Detecta elementos importantes en un texto y los clasifica en categorías predefinidas:
  - **MONEY**: Cantidades monetarias.
  - **DATE**: Fechas.
  - **ORG**: Organizaciones.
  - **PERCENT**: Porcentajes.

### **Objetivo**
- Convertir el contenido de los `.docx` a un formato estructurado en `.json`.  
  La estructura se divide en:
  - **Párrafos:** Título y contenido.
  - **Tablas:** Título y datos (la primera fila de datos corresponde al nombre de la columna).

Ejemplo de estructura JSON:
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

## **2. Generación de Preguntas y Respuestas**

### **Herramientas Utilizadas**
- `_2_1_Generate Prompts_sections.py`: Para generar preguntas a partir de secciones (párrafos).
- `_2_2_Generate Prompts_table.py`: Para generar preguntas basadas en tablas.


### **Generación de Preguntas para Secciones-Párrafos**

#### **Método `generate_questions_by_NLP()`**
- Generación basada en el título de la sección:
  - Son genéricas, a pesar de la clasificación por tema a través de NLP, y pueden generar sensación de preguntas "robóticas":
    ```python
    f"¿Qué información general se describe en la sección '{title}'?"
    ```
- Personalización según entidades detectadas (usando NLP):
  - Para preguntas de dinero, se da un matiz particular:
    ```python
    if ent.label_ == "MONEY":
        prompts.append(f"¿Cuáles son los costos asociados con {ent.text} mencionados en '{title}'?")
    ```
- Listado de temas con preguntas específicas:
  ```python
  entity_labels = [
      "MONEY",    # Costos, tarifas y cálculos financieros.
      "ORG",      # Organizaciones, servicios y productos.
      "PRODUCT",  # Productos, características y beneficios.
      "LOC",      # Ubicaciones y regulaciones.
      "PERCENT",  # Porcentajes, cálculos y significados.
      "DATE",     # Fechas, eventos y relevancia temporal.
      "TIME",     # Horarios y cronogramas.
      "EVENT",    # Eventos y detalles asociados.
      "LAW",      # Regulaciones y requisitos legales.
      "QUANTITY", # Cantidades y cálculos.
      "CARDINAL"  # Números y su interpretación en contexto.
  ]
  ```

#### **Método `generate_questions_by_Model()`**
- Preguntas generadas con el modelo `mrm8488/bert2bert-spanish-question-generation`:
  - Se utiliza el título y el párrafo de la sección como entrada.
  - Genera preguntas automáticamente. **Nota:** Este modelo es adecuado para fines educativos pero no recomendado para producción.


### **Validación de Preguntas Generadas**
- Las preguntas cuyo tema NLP no coincide con la respuesta son descartadas mediante:
  ```python
  validate_prompt_response_NLP()
  ```

### **Generación de Preguntas para Tablas**

#### **1. `generate_General_table_questions`**
- Genera preguntas generales sobre una tabla completa.
- **Nota:** Se descartó por falta de efectividad en resúmenes con el modelo `bert2bert_shared-spanish-finetuned-summarization`.

#### **2. `generate_ROW_specific_questions`**
- Genera preguntas específicas para una fila de la tabla, combinando:
  - Datos de la fila: Propiedades y valores asociados.
  - Preguntas basadas en NLP: Detecta entidades relevantes como porcentajes, fechas, ubicaciones.
  - Preguntas generadas por modelos:
    ```python
    if ent.label_ == "MONEY":
        nlp_questions.append({
            "prompt": f"¿Qué costos están relacionados con el elemento '{row_reference}' en la tabla '{title}'?",
            "response": response_text
        })
    ```

#### **3. `generate_CELL_specific_questions`**
- Genera preguntas específicas para una celda de la tabla, utilizando:
  - **Datos de la celda:** Valor de la celda con encabezados de fila y columna.
  - **Preguntas basadas en NLP:** Analiza entidades detectadas.
  - **Preguntas generadas por modelos:** Cruza datos entre columnas y filas:
    ```python
    if ent.label_ == "MONEY":
        nlp_questions.append({
            "prompt": f"¿Qué costos están relacionados con '{column_reference}' en relación con '{row_reference}'?",
            "response": response_text_CELL
        })
    ```


### **Validación `validate_description()`**
- Verificación temática entre preguntas y respuestas con NLP.

### **Fusión Final**
- Se generó el archivo `_2_generated_prompts_FULL.json` combinando:
  - `_2_generated_prompts_SECTIONS.json`
  - `_2_generated_prompts_TABLE.json`

## **3. Validación del .json**

### **Validación del `_2_generated_prompts_FULL.json`**
- Se verificó la correspondencia entre preguntas y respuestas basadas en palabras clave y temática:
  ```python
  doc_prompt = nlp(prompt)
  doc_response = nlp(response)
  keywords_prompt = {token.text.lower() for token in doc_prompt if not token.is_stop and token.is_alpha}
  keywords_response = {token.text.lower() for token in doc_response if not token.is_stop and token.is_alpha}
  ```

#### **Observaciones**
- Requiere mejoras manuales o el uso de modelos de lenguaje más avanzados.

## **4. Entrenamiento**

### **Modelos Utilizados**
- `google/mt5-base`
- `google/mt5-small`
- `meta-llama/Llama-2-7b`
- `bigscience/bloom-560m`

### **Proceso de Entrenamiento para BLOOM**

#### **Validación y Preprocesamiento de Datos**

1. **Validación del JSON**:
   - Se asegura que cada entrada en el archivo JSON contenga las claves `prompt` y `response`.
   ```python
   def validate_json(data):
       for entry in data:
           if "prompt" not in entry or "response" not in entry:
               raise ValueError("Each entry must contain 'prompt' and 'response' keys.")
   ```

2. **Conversión a Dataset**:
   - El JSON validado se convierte a un dataset compatible con Hugging Face, usando el formato `{input_text, output_text}`.

3. **Tokenización**:
   - Los textos de entrada (prompt) y salida (response) se tokenizan con truncamiento y relleno hasta un máximo de 512 tokens.
   ```python
   def preprocess_function(examples):
       inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
       outputs = tokenizer(examples["output_text"], max_length=512, truncation=True, padding="max_length")
       inputs["labels"] = outputs["input_ids"]
       return inputs
   ```

##### **Evaluación de Métricas**

- Métricas calculadas en conjuntos de validación y prueba:
  - **BLEU**: Mide precisión en coincidencias de n-gramas entre texto generado y de referencia.
  - **ROUGE**: Evalúa similitudes basadas en recall entre las secuencias generadas y las de referencia.
    - Métricas calculadas: `rouge1`, `rouge2`, `rougeL`.
  - **Loss**: Basado en la pérdida de entropía cruzada entre predicciones del modelo y etiquetas reales.
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

#### **Entrenamiento del Modelo**

1. **Argumentos de Entrenamiento**:
   - Parámetros clave como tasa de aprendizaje, tamaño de batch, número de épocas, uso de FP16 (precisión mixta) y pasos para guardar checkpoints.
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

2. **Dispositivo**:
   - El modelo se entrena en GPU si está disponible (`cuda`), de lo contrario, en CPU.

3. **Callbacks**:
   - **DebugCallback**: Muestra logs del entrenamiento para monitorear métricas y progreso.
   - **EvaluationCallback**: Cada 50 pasos:
     - Guarda un checkpoint del modelo.
     - Genera respuestas para preguntas de evaluación y las muestra en consola.

#### **Validación Final**
- **Script Utilizado**: `_5_evaluate_mT5_small.py`.
- **Tareas Realizadas**:
  - Pruebas finales de las respuestas generadas por los modelos.
  - Evaluación de consistencia y naturalidad de las respuestas.


## **Observaciones Finales**
- A nivel educativo, el resultado es correcto. A **_nivel industrial, requiere mejoras significativas_** en la creación y validación de preguntas/respuestas sintéticas, que actualmente suenan robóticas.

## **5. Mejoras**

-  **Aumentar el tamaño del dataset**
Incluir mayor variabilidad lingüística para evitar respuestas repetitivas y robóticas.
Uso de herramientas como **ChatGPT API (GPT-4)** o **Llama Index** para generar preguntas/respuestas más naturales y diversificadas.

-  **Reentrenar modelos públicos de DeepSeek** https://huggingface.co/deepseek-ai/DeepSeek-V3 

-  **Implementar Modelos de Preprocesamiento Semántico**
**spaCy** o **Sentence-BERT**:
  Detectar y eliminar redundancias en las respuestas.
  Refinar el formato de las preguntas/respuestas.

-  **Herramientas de Análisis Semántico**
**AllenNLP** o **Hugging Face NLP Pipelines**:
  Evaluar la correspondencia temática entre preguntas y respuestas.

-  **Evaluación Humana**
Incorporar ciclos de revisión manual para garantizar la calidad en preguntas/respuestas generadas.

-  **Migrar a Modelos Más Grandes o Especializados**
Utilizar modelos como **GPT-4**, **Llama-2** o **Claude** para generación de respuestas más naturales.

-  **Ajustar Técnicas de Generación**
Aplicar métodos como:
  **Top-k Sampling**
  **Nucleus Sampling (top-p)**
  **Temperature**
  Mejorar la variabilidad sin perder coherencia.

-  **Herramientas Recomendadas**
1. **AllenNLP**: Para análisis detallado de dependencias semánticas en preguntas/respuestas.
2. **Transformers Evaluate (Hugging Face)**: Para implementar métricas avanzadas como `BERTScore` y `METEOR`.


-  **Cálculos Dinámicos en el Chatbot**
El modelo actual solo devuelve información genérica como:
  _"La comisión para Francia es 2%."_  
**Objetivo**: Permitir cálculos dinámicos para generar resultados procesados:
  ```
  Pregunta: "¿Cuál es la comisión de transferencias para Francia de 1000€?"
  Respuesta: "La comisión para Francia es 2%, lo que equivale a: 1000€ × 2% = 20€."
  ```
-  **Implementación con LangChain**:
  Analiza la pregunta para identificar si requiere cálculos específicos.
  Redirige los cálculos necesarios a un entorno Python.
