from docx import Document
import json
import re
import openai  # Assuming you have the OpenAI library installed


def remove_uniform_columns(data):
    """
    Remove columns where all elements are identical across rows.
    """
    if not data or len(data) < 2:  # No data or no rows to compare
        return data

    headers = data[0]
    rows = data[1:]
    num_columns = len(headers)

    # Check for uniform columns
    to_remove = []
    for col_idx in range(num_columns):
        column_values = [row[col_idx] for row in rows if col_idx < len(row)]
        if all(value == column_values[0] for value in column_values):
            to_remove.append(col_idx)

    # Remove uniform columns
    filtered_data = []
    for row in data:
        filtered_row = [cell for idx, cell in enumerate(row) if idx not in to_remove]
        filtered_data.append(filtered_row)

    return filtered_data


from transformers import pipeline

# Load a multilingual summarization model
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")


def generate_title(content, max_length=15, language="es"):
    """
    Generate a concise title for the given paragraph content using a multilingual summarization model.
    Args:
        content (str): The paragraph content to summarize.
        max_length (int): Maximum length of the generated title.
        language (str): Language of the content, default is Spanish ("es").
    Returns:
        str: A generated title in the specified language.
    """
    try:
        # Add a language prefix for better context understanding
        prefixed_content = f"resumir en {language}: {content}"

        # Generate a title using the summarizer pipeline
        summary = summarizer(prefixed_content, max_length=max_length, min_length=3, do_sample=False)

        return summary[0]['summary_text'].strip()
    except Exception as e:
        print(f"Error generating title for content: {content[:50]}... -> {e}")
        return "Título generado automáticamente"