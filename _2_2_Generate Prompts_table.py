import json
import spacy
# Load the Spanish spaCymodel
from _utils import *
from _validation import validate_prompt_response_NLP
from _google_generate_ask import generate_questions_by_Model

nlp = spacy.load("es_core_news_md")

# File paths
input_file = "processed/_1_combined.json"
output_file = "processed/_2_generated_prompts_TABLE.json"

def generate_prompts_and_responses(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    prompts_and_responses = []

    for table in data.get("tables", []):
        title = table.get("title", "Sin título")
        rows = table.get("data", [])

        if not rows or len(rows[0]) == 0:
            continue  # Skip empty tables

        headers = rows[0]  # First row as headers
        data_rows = rows[1:]  # Remaining rows
        print("\t ", title)
        # TODO General table questions. generate_General_table_questions does not work
        # genral_prompts = generate_General_table_questions(title, headers, rows)
        # if  genral_prompts is not None:
        #     prompts_and_responses.extend(generate_General_table_questions(title, headers, rows))
        # TODO for the same row keys they must be together
        for row in data_rows:
            row_reference = row[0]  # First column as row reference
            cell_values = row[1:]  # Remaining cells

            # Row-specific questions
            prompts_and_responses.extend(generate_ROW_specific_questions(title, row_reference, headers, row[1:]))

            for col_idx, cell_value in enumerate(cell_values):
                if cell_value.strip():  # Skip empty cells
                    column_reference = headers[col_idx + 1]

                    # Cell-specific questions
                    prompts_and_responses.extend(
                        generate_CELL_specific_questions(row_reference, column_reference, cell_value, title)
                    )

                    String_defenition = f"en {title}  propiedad {column_reference}, elemento {row_reference} tiene valor  {cell_value}"

    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(prompts_and_responses, out_file, ensure_ascii=False, indent=4)

    print(f"Prompts and responses saved to {output_file}")


generate_prompts_and_responses(input_file, output_file)
