import os
from docx import Document
import pandas as pd
import re
import json
import unicodedata
import language_tool_python

# Initialize Spanish grammar checker
from _google_generate_ask import generate_question

tool = language_tool_python.LanguageTool("es")

# Cleaning Functions
def clean_text(text):
    """Clean and normalize text for Spanish."""
    if not text.strip():
        return ""

    # Remove extra whitespace
    text = text.strip()
    text = " ".join(text.split())

    # Normalize punctuation and remove unnecessary characters
    text = re.sub(r'\s+([?.!,¿])', r'\1', text)
    text = re.sub(r"[^\w\s¿?€.,‰]", "", text)

    # Normalize Unicode characters (e.g., accents)
    text = unicodedata.normalize("NFC", text)

    # Replace special symbols
    text = text.replace("‰", "por mil").replace("€", "euros")

    # Correct grammar issues
    text = tool.correct(text)

    return text

def clean_table_data(table_data):
    """Clean and normalize table data."""
    cleaned_data = []
    for row in table_data:
        cleaned_data.append([clean_text(cell) for cell in row])
    return cleaned_data




def asocited_label_with_ask_1(doc, prompts, title):
    # Generate prompts based on entities in the content
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            prompts.append(f"¿Cuáles son los costes relacionados con {ent.text}?")
            prompts.append(f"¿Qué tarifas aplican a las transferencias de {ent.text}?")
        elif ent.label_ == "ORG":
            prompts.append(f"¿Qué servicios ofrece {ent.text}?")
            prompts.append(f"¿Qué productos financieros tiene {ent.text}?")
        elif ent.label_ == "PRODUCT":
            prompts.append(f"¿Cuáles son las características principales de {ent.text}?")
            prompts.append(f"¿Qué nivel de riesgo tiene el producto {ent.text}?")
        elif ent.label_ == "LOC":
            prompts.append(f"¿Qué operaciones están disponibles en {ent.text}?")
            prompts.append(f"¿Qué impacto tienen las regulaciones de {ent.text}?")
        elif ent.label_ == "PERCENT":
            prompts.append(f"¿Qué significa un porcentaje de {ent.text} en este contexto?")
            prompts.append(f"¿Cómo se calcula una comisión del {ent.text}?")
        elif ent.label_ == "DATE":
            prompts.append(f"¿Qué fechas clave se mencionan en relación con {ent.text}?")
            prompts.append(f"¿Qué eventos importantes suceden el {ent.text}?")
        elif ent.label_ == "TIME":
            prompts.append(f"¿Cuál es el horario relacionado con {ent.text}?")
            prompts.append(f"¿Cómo afecta el horario de {ent.text} a las operaciones?")
        elif ent.label_ == "EVENT":
            prompts.append(f"¿Qué detalles están disponibles sobre el evento {ent.text}?")
            prompts.append(f"¿Cómo participar en el evento {ent.text}?")
        elif ent.label_ == "LAW":
            prompts.append(f"¿Qué implica la regulación {ent.text}?")
            prompts.append(f"¿Cuáles son los requisitos legales según {ent.text}?")
        elif ent.label_ == "QUANTITY":
            prompts.append(f"¿Cuántas unidades están involucradas en {ent.text}?")
            prompts.append(f"¿Cómo se calcula la cantidad mencionada de {ent.text}?")
        elif ent.label_ == "CARDINAL":
            prompts.append(f"¿Qué significa el número {ent.text} en este contexto?")
            prompts.append(f"¿Cómo se relaciona {ent.text} con los cálculos del documento?")
    # Add general prompts based on the section title
    prompts.append(f"¿Qué información contiene la sección '{title}'?")
    prompts.append(f"¿Puedes explicar los detalles sobre '{title}'?")
    prompts.append(f"¿Qué aspectos clave se describen en '{title}'?")
    return prompts


def generate_questions_by_NLP(doc, prompts, title):
    """
    Generate high-quality prompts based on entities in the content and the section title.
    :param doc: The spaCy doc object for the content.
    :param prompts: List to store generated prompts.
    :param title: The title of the section for context.
    :return: Updated list of prompts.
    """
    # Generate prompts based on entities in the content
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            prompts.append(f"¿Cuáles son los costos asociados con {ent.text} mencionados en '{title}'?")
            prompts.append(f"¿Cómo se calculan las tarifas aplicables a {ent.text} según la sección '{title}'?")
            prompts.append(f"¿Existen descuentos o excepciones en los costos relacionados con {ent.text}?")
        elif ent.label_ == "ORG":
            prompts.append(f"¿Qué servicios destacados ofrece la organización {ent.text} según '{title}'?")
            prompts.append(f"¿Cómo se describe el rol de {ent.text} en el contexto de esta sección?")
            prompts.append(f"¿Qué productos financieros exclusivos tiene {ent.text}?")
        elif ent.label_ == "PRODUCT":
            prompts.append(f"¿Qué beneficios clave se mencionan para el producto {ent.text}?")
            prompts.append(f"¿Qué detalles de inversión están asociados con el producto {ent.text}?")
            prompts.append(f"¿Cómo se compara {ent.text} con otros productos similares mencionados en '{title}'?")
        elif ent.label_ == "LOC":
            prompts.append(f"¿Qué operaciones específicas están disponibles en {ent.text}?")
            prompts.append(f"¿Cómo afectan las normativas de {ent.text} las acciones mencionadas en '{title}'?")
            prompts.append(f"¿Qué ventajas ofrece la ubicación {ent.text} según este documento?")
        elif ent.label_ == "PERCENT":
            prompts.append(f"¿Cómo se interpreta un porcentaje de {ent.text} en el contexto de '{title}'?")
            prompts.append(f"¿Qué implica un cambio en el porcentaje {ent.text} para los usuarios?")
            prompts.append(f"¿Qué ejemplos específicos están asociados con el porcentaje {ent.text}?")
        elif ent.label_ == "DATE":
            prompts.append(f"¿Qué eventos importantes están programados para la fecha {ent.text}?")
            prompts.append(f"¿Qué relevancia tiene {ent.text} en el marco de las operaciones descritas?")
            prompts.append(f"¿Cómo influye la fecha {ent.text} en los detalles expuestos en '{title}'?")
        elif ent.label_ == "TIME":
            prompts.append(f"¿Cuál es el horario específico relacionado con {ent.text}?")
            prompts.append(f"¿Qué actividades están previstas para la hora {ent.text}?")
            prompts.append(f"¿Cómo afecta el horario {ent.text} a las transacciones descritas en '{title}'?")
        elif ent.label_ == "EVENT":
            prompts.append(f"¿Qué información relevante se proporciona sobre el evento {ent.text}?")
            prompts.append(f"¿Cómo participar en el evento {ent.text} según esta sección?")
            prompts.append(f"¿Qué impacto se espera del evento {ent.text} en el contexto descrito?")
        elif ent.label_ == "LAW":
            prompts.append(f"¿Qué detalles clave se mencionan sobre la regulación {ent.text}?")
            prompts.append(f"¿Cómo afecta {ent.text} a las prácticas descritas en esta sección?")
            prompts.append(f"¿Qué acciones deben tomarse para cumplir con {ent.text}?")
        elif ent.label_ == "QUANTITY":
            prompts.append(f"¿Cómo se distribuyen las cantidades relacionadas con {ent.text}?")
            prompts.append(f"¿Qué cálculos específicos están relacionados con {ent.text}?")
            prompts.append(f"¿Cómo se utilizan las cantidades indicadas de {ent.text} en esta sección?")
        elif ent.label_ == "CARDINAL":
            prompts.append(f"¿Qué importancia tiene el número {ent.text} en el contexto de '{title}'?")
            prompts.append(f"¿Cómo afecta el número {ent.text} a los cálculos presentados?")
            prompts.append(f"¿Qué interpretaciones posibles se dan para el valor {ent.text}?")

    # Add general prompts based on the section title
    prompts.append(f"¿Qué información general se describe en la sección '{title}'?")
    prompts.append(f"¿Qué aspectos destacados puedes explicar sobre '{title}'?")
    prompts.append(f"¿Cómo se relaciona el contenido de '{title}' con otras secciones del documento?")
    prompts.append(f"¿Puedes proporcionar un resumen detallado de la sección '{title}'?")

    return prompts


import spacy
nlp = spacy.load("es_core_news_md")
def generate_entity_prompts_NLP_table(cell_value, row_reference, column_reference, title):
    """
    Generate entity-based prompts for a given cell value in the table.
    :param cell_value: The content of the table cell.
    :param row_reference: The reference (header) for the row.
    :param column_reference: The reference (header) for the column.
    :param title: The title of the table.
    :return: List of prompts and the cell value as responses.
    """
    prompts = []
    doc = nlp( f" la tabla {title}  valores {column_reference},  con tipo {row_reference} y  con valor {cell_value}")

    for ent in doc.ents:
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                prompts.append({
                    "prompt": f"¿Cuáles son los costos relacionados con '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "PERCENT":
                prompts.append({
                    "prompt": f"¿Qué representa el porcentaje '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "DATE":
                prompts.append({
                    "prompt": f"¿Qué eventos o actividades están relacionadas con la fecha '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "ORG":
                prompts.append({
                    "prompt": f"¿Qué rol desempeña la organización '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "PRODUCT":
                prompts.append({
                    "prompt": f"¿Qué detalles se mencionan sobre el producto '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "LOC":
                prompts.append({
                    "prompt": f"¿Qué información está relacionada con la ubicación '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "QUANTITY":
                prompts.append({
                    "prompt": f"¿Cómo se interpreta la cantidad '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "CARDINAL":
                prompts.append({
                    "prompt": f"¿Qué significa el número '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "TIME":
                prompts.append({
                    "prompt": f"¿Qué actividades o eventos están programados para la hora '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "EVENT":
                prompts.append({
                    "prompt": f"¿Qué detalles relevantes se proporcionan sobre el evento '{ent.text}' en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })
            elif ent.label_ == "LAW":
                prompts.append({
                    "prompt": f"¿Cómo afecta la regulación '{ent.text}' a la información en la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                    "response": cell_value
                })

        # Fallback for cells with no detected entities
        if not prompts:
            prompts.append({
                "prompt": f"¿Qué información describe la tabla '{title}', valores '{column_reference}', con tipo '{row_reference}' y con valor '{cell_value}'?",
                "response": cell_value
            })

        return prompts




import spacy
nlp = spacy.load("es_core_news_md")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Load the Spanish-compatible summarization and question-answering models
summarization_tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert2bert_shared-spanish-finetuned-summarization")
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/bert2bert_shared-spanish-finetuned-summarization")

qa_tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert2bert-spanish-question-generation")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/bert2bert-spanish-question-generation")


def validate_description(title, headers, rows, description, threshold=3):
    """
    Validate that the description is related to the title, headers, and rows provided.
    A description is valid if at least `threshold` terms from the description are present in the inputs.
    :param title: The title of the table.
    :param headers: The column headers of the table.
    :param rows: The data rows in the table.
    :param description: The generated description of the table.
    :param threshold: Minimum number of terms that must overlap for validation.
    :return: True if the description is valid, False otherwise.
    """
    # Combine inputs into a single text for comparison
    input_text = f"Título: {title}. Columnas: {', '.join(headers)}. "
    input_text += " Filas: " + " | ".join([", ".join(row) for row in rows])

    # Process the input and description using NLP
    doc_input = nlp(input_text)
    doc_description = nlp(description)

    # Extract entities and keywords from the input
    input_entities = {ent.text.lower() for ent in doc_input.ents}
    input_keywords = {chunk.text.lower() for chunk in doc_input.noun_chunks}

    # Extract entities and keywords from the description
    description_entities = {ent.text.lower() for ent in doc_description.ents}
    description_keywords = {chunk.text.lower() for chunk in doc_description.noun_chunks}

    # Combine entities and keywords for comparison
    input_terms = input_entities.union(input_keywords)
    description_terms = description_entities.union(description_keywords)

    # Calculate overlap
    overlap = description_terms.intersection(input_terms)

    # Validate based on the threshold
    if len(overlap) >= threshold:
        return True  # Sufficient overlap
    else:
        missing_terms = input_terms - description_terms
        print(f"Validation failed. Missing terms in description: {missing_terms}")
        return False

def extract_table_description(title, headers, rows, max_rows=3):
    """
    Generate a description of the table based on its data.
    :param title: The title of the table.
    :param headers: The column headers of the table.
    :param rows: The data rows in the table.
    :param max_rows: Maximum number of rows to include in the summarization.
    :return: A summarized description of the table content.
    """
    # Limit the number of rows included for summarization
    rows_to_include = rows[:max_rows]

    # Combine headers and rows into a structured input
    table_text = f"Título: {title}. Columnas: {', '.join(headers)}. "
    for row in rows_to_include:
        row_data = ", ".join(row)
        table_text += f" Fila: {row[0]}."

    # Summarize the table content
    inputs = summarization_tokenizer.encode(
        table_text, return_tensors="pt", truncation=True, max_length=512
    )
    outputs = summarization_model.generate(
        inputs, max_length=128, num_beams=5, early_stopping=True
    )
    description = summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return description

def generate_General_table_questions(title, headers, rows):
    """
    Generate general questions and answers about the table.
    :param title: The title of the table.
    :param headers: The column headers of the table.
    :param rows: The data rows in the table.
    :return: List of general questions with responses.
    """
    # Extract table description dynamically
    description = extract_table_description(title, headers, rows)

    is_valid = validate_description(title, headers, rows, description)

    if not is_valid:
        return None
    # Define general questions
    questions = [
        f"¿Qué información describe la tabla '{title}'?",
        f"¿Cuáles son los datos principales contenidos en la tabla '{title}'?"
        # f"¿Qué propósito tiene la tabla '{title}'?"
    ]

    # Generate responses for each question
    prompts_and_responses = []
    for question in questions:
        # input_text = f"{question} {description}"
        # inputs = qa_tokenizer.encode(
        #     input_text, return_tensors="pt", truncation=True, max_length=512  # Truncate to prevent overflow
        # )
        #
        # outputs = qa_model.generate(
        #     inputs, max_length=64, num_beams=5, early_stopping=True
        # )
        # response = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompts_and_responses.append({
            "prompt": question,
            "response": description
        })

    return prompts_and_responses


def generate_nlp_questions_table(title, row_reference, response_text):
    """
    Generate NLP-based questions based on detected entities in the response text.
    Each entity label can only have one set of questions added.
    :param title: The title of the table.
    :param row_reference: The row header providing context for the row.
    :param response_text: The formatted response text containing property-value pairs.
    :return: List of NLP-based questions with responses.
    """
    nlp_questions = []
    processed_labels = set()  # Track processed entity labels
    doc = nlp(f"En {title}, el elemento '{row_reference}' tiene los valores: {response_text}")

    for ent in doc.ents:
        if ent.label_ not in processed_labels:
            if ent.label_ == "MONEY":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué costos están relacionados con el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text},
                    {
                        "prompt": f"¿Cómo se calculan los costos asociados al elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text}
                ])
            elif ent.label_ == "PERCENT":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué implica el porcentaje mencionado en el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text},
                    {
                        "prompt": f"¿Cómo afecta el porcentaje relacionado al elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text}
                ])
            elif ent.label_ == "DATE":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué eventos están asociados con las fechas indicadas en el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text},
                    {
                        "prompt": f"¿Cómo influyen las fechas mencionadas en el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text}
                ])
            elif ent.label_ == "ORG":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué rol tiene la organización relacionada con el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text},
                    {
                        "prompt": f"¿Qué servicios ofrece la organización mencionada para el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text}
                ])
            elif ent.label_ == "PRODUCT":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué características se describen sobre el producto asociado al elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text},
                    {
                        "prompt": f"¿Cómo se diferencia el producto relacionado con el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text}
                ])
            elif ent.label_ == "LOC":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué detalles están asociados con la ubicación para el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text},
                    {
                        "prompt": f"¿Cómo afecta la ubicación relevante al elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text}
                ])
            elif ent.label_ == "LAW":
                nlp_questions.extend([
                    {"prompt": f"¿Qué regulaciones afectan al elemento '{row_reference}' en la tabla '{title}'?",
                     "response": response_text},
                    {
                        "prompt": f"¿Qué requisitos legales están asociados con el elemento '{row_reference}' en la tabla '{title}'?",
                        "response": response_text}
                ])

            # Mark the label as processed
            processed_labels.add(ent.label_)

    return nlp_questions


def generate_ROW_specific_questions(title, row_reference, headers, row_data):
    """
    Generate questions specific to each row in the table, with responses based on the row's data.
    Combines handcrafted, NLP, and model-generated questions.
    :param title: The title of the table.
    :param row_reference: The row header providing context for the row.
    :param headers: The column headers of the table.
    :param row_data: The full row data (including the row_reference and cell values).
    :return: List of row-specific questions with responses.
    """
    # Ensure all row values are non-empty, replacing missing values with "NA"
    if len(headers) !=  len(row_data) +1:
        raise "no corresponde el formato "


    row_values = [value.strip() if value.strip() else "NA" for value in row_data]

    # Format response as "Property: Value"
    response_data = [
        f"{header}: {value}" for header, value in zip(headers[1:], row_values)
    ]
    response_text = " | ".join(response_data)

    if not response_text:
        print("debug")

    # Generate diverse context strings for the model
    context_variations = [
        f"En {title}, el elemento '{row_reference}' tiene las siguientes propiedades: {response_text}",
        f"Analiza el elemento '{row_reference}' en la tabla '{title}', que tiene los valores: {response_text}",
        f"En el contexto del elemento '{row_reference}' en la tabla '{title}', se destacan los valores: {response_text}"
    ]

    # Generate model-based questions
    generated_questions = [
        {"prompt": generate_question(context), "response": response_text}
        for context in context_variations
    ]

    # Generate NLP-based questions
    nlp_questions = generate_nlp_questions_table(title, row_reference, response_text)

    # Combine generated questions
    return generated_questions + nlp_questions if nlp_questions else generated_questions


def generate_CELL_specific_questions(row_reference, column_reference, cell_value, title):
    """
    Generate questions specific to each cell in the table.
    Combines handcrafted, NLP, and model-generated questions.
    :param row_reference: The row header providing context for the cell.
    :param column_reference: The column header providing context for the cell.
    :param cell_value: The value of the cell.
    :return: List of cell-specific questions and responses.
    """
    generated_questions = []
    processed_labels = set()  # Track processed entity labels
    cell_value = cell_value.strip() if cell_value.strip() else "NA"

    # Format response text
    response_text_nlp = f"{column_reference}: {cell_value} | Elemento: {row_reference}"
    response_text = cell_value
    # Generate diverse context strings for the model
    context_variations = [
        # f"¿Cuál es el significado de '{column_reference}' con elemento '{row_reference}'  ? '{column_reference}' y '{row_reference}'",
        # f" la propiedad: '{column_reference}' con elemento: '{row_reference}'. '{column_reference}' y '{row_reference}'",
        f"En el elemento: '{row_reference}', con columna: '{column_reference}'? '{column_reference}' y '{row_reference}'"
    ]
    # Generate model-based questions
    for context in context_variations:
        generated_questions.append({"prompt": generate_question(context), "response": response_text})

    # Generate NLP-based questions
    nlp_questions = []
    # Generate NLP-based questions
    nlp_questions = []
    doc = nlp(response_text_nlp)
    for ent in doc.ents:
        if ent.label_ not in processed_labels:
            if ent.label_ == "MONEY":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué costos están relacionados con '{column_reference}' en relación con '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿De qué manera se calculan los costos mencionados para '{row_reference}' en '{column_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cómo afectan los costos especificados en '{column_reference}' a '{row_reference}' según '{title}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cuál es el valor asociado a los costos de '{column_reference}' para '{row_reference}' en la tabla '{title}'?",
                        "response": response_text
                    }
                ])
            elif ent.label_ == "PERCENT":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué significa el porcentaje indicado en '{column_reference}' relacionado con '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cómo influye el porcentaje de '{column_reference}' en las características de '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Qué interpretación tiene el porcentaje especificado en '{column_reference}' para '{row_reference}' según la tabla '{title}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cuál es el porcentaje mencionado en '{column_reference}' vinculado a '{row_reference}' en '{title}'?",
                        "response": response_text
                    }
                ])
            elif ent.label_ == "DATE":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué fechas relevantes están asociadas con '{column_reference}' en relación con '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cómo impactan las fechas indicadas en '{column_reference}' sobre las condiciones de '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Qué importancia tienen las fechas asociadas a '{column_reference}' respecto a '{row_reference}' según la tabla '{title}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Qué eventos están relacionados con las fechas indicadas en '{column_reference}' y su vínculo con '{row_reference}'?",
                        "response": response_text
                    }
                ])
            elif ent.label_ == "ORG":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué papel juega la organización mencionada en '{column_reference}' vinculada a '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Qué servicios destacan en la organización relacionada con '{column_reference}' para '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cómo contribuye la organización referida en '{column_reference}' al contexto de '{row_reference}' dentro de '{title}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Qué aspectos clave de la organización en '{column_reference}' se relacionan con '{row_reference}' en la tabla '{title}'?",
                        "response": response_text
                    }
                ])
            elif ent.label_ == "LOC":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué información clave está vinculada a la ubicación descrita en '{column_reference}' respecto a '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cómo influye la ubicación indicada en '{column_reference}' sobre las características de '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Qué relevancia tiene la ubicación mencionada en '{column_reference}' para '{row_reference}' en el contexto de la tabla '{title}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cuál es la ubicación señalada en '{column_reference}' vinculada con '{row_reference}' en '{title}'?",
                        "response": response_text
                    }
                ])
            elif ent.label_ == "LAW":
                nlp_questions.extend([
                    {
                        "prompt": f"¿Qué regulaciones se mencionan en '{column_reference}' relacionadas con '{row_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Qué requisitos legales afectan a '{row_reference}' en la propiedad '{column_reference}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Cómo influyen las regulaciones indicadas en '{column_reference}' para '{row_reference}' en la tabla '{title}'?",
                        "response": response_text
                    },
                    {
                        "prompt": f"¿Qué normativas específicas están vinculadas a '{column_reference}' en relación con '{row_reference}' en '{title}'?",
                        "response": response_text
                    }
                ])

            # Mark the label as processed
            processed_labels.add(ent.label_)

    # Add generic questions if no NLP questions were generated
    if not nlp_questions:
        nlp_questions.extend([
            {
                "prompt": f"¿Qué datos describe '{column_reference}' para '{row_reference}'?",
                "response": response_text
            },
            {
                "prompt": f"¿Qué significa el valor proporcionado en '{column_reference}' del elemento '{row_reference}'?",
                "response": response_text
            },
            {
                "prompt": f"¿Cómo se relaciona '{column_reference}' con '{row_reference}' en la tabla '{title}'?",
                "response": response_text
            },
            {
                "prompt": f"¿Cuál es el valor especificado en '{column_reference}' asociado a '{row_reference}' en el título '{title}'?",
                "response": response_text
            }
        ])
    nlp_questions.extend([ {
                "prompt": f"¿Cual es el valor de '{column_reference}' del tipo '{row_reference}'?",
                "response": response_text
            }])

    # Combine generated questions
    return generated_questions + nlp_questions





