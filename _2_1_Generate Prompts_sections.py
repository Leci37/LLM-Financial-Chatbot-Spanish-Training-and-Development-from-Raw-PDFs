import json
import spacy

# Load the Spanish spaCy model
from _utils import generate_questions_by_NLP
from _validation import validate_prompt_response_NLP
from _google_generate_ask import generate_questions_by_Model

nlp = spacy.load("es_core_news_md")

# File paths
input_file = "processed/_1_combined.json"
output_file = "processed/_2_generated_prompts_FULL.json"

def generate_prompts_and_responses(input_file, output_file):
    # Load the input JSON file
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    prompts_and_responses = []

    # Process sections for prompts and responses
    for section in data.get("sections", []):
        title = section.get("title", "Sin título").strip()
        content = section.get("content", "")

        # Ensure content is a single string
        if isinstance(content, list):
            content = " ".join(content).strip()

        # Skip empty sections
        if not content:
            continue

        # Analyze content using spaCy NLP
        doc = nlp(content)

        prompts_nlp = []
        prompts_nlp = generate_questions_by_NLP(doc, prompts_nlp, title)
        prompts_google = generate_questions_by_Model(content, title)

        for prompt in prompts_nlp:
            if validate_prompt_response_NLP(prompt, content):  # Validate the pair
                prompts_and_responses.append({
                    "prompt": prompt,
                    "response": content
                })
            else:
                print("\t pregunta no casa con respuesta NLP ")
        for prompt in prompts_google:
            if validate_prompt_response_NLP(prompt, content):  # Validate the pair
                prompts_and_responses.append({
                    "prompt": prompt,
                    "response": content
                })

    # # Process tables for structured data prompts
    # for table in data.get("tables", []):
    #     title = table.get("title", "Sin título").strip()
    #     rows = table.get("data", [])
    #
    #     for row in rows:
    #         if len(row) > 1:
    #             # Construct response from table data
    #             response = " | ".join(row)
    #
    #             # Generate tailored prompts based on the table's title
    #             prompts = [
    #                 f"¿Cuáles son los detalles sobre '{title}'?",
    #                 f"¿Puedes describir la información de '{title}' en la tabla?",
    #                 f"¿Qué datos relevantes contiene la tabla '{title}'?"
    #             ]
    #
    #             for prompt in prompts:
    #                 prompts_and_responses.append({
    #                     "prompt": prompt,
    #                     "response": response
    #                 })

    # Save generated prompts and responses to output JSON file
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(prompts_and_responses, out_file, ensure_ascii=False, indent=4)

    print(f"Prompts and responses saved to {output_file}")


# Execute the function
generate_prompts_and_responses(input_file, output_file)
