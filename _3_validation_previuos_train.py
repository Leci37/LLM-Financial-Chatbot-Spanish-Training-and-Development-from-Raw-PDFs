import spacy
import json
from difflib import SequenceMatcher

# Load a Spanish NLP model
nlp = spacy.load("es_core_news_md")


def similarity_ratio(text1, text2):
    """
    Calculate the similarity ratio between two strings.
    """
    return SequenceMatcher(None, text1, text2).ratio()


def validate_prompt_response_pair(prompt, response):
    """
    Validate whether the prompt matches the response grammatically and contextually.
    """
    doc_prompt = nlp(prompt)
    doc_response = nlp(response)

    # Extract keywords and entities from the prompt and response
    keywords_prompt = {token.text.lower() for token in doc_prompt if not token.is_stop and token.is_alpha}
    keywords_response = {token.text.lower() for token in doc_response if not token.is_stop and token.is_alpha}

    # Extract named entities
    entities_prompt = {ent.text.lower() for ent in doc_prompt.ents}
    entities_response = {ent.text.lower() for ent in doc_response.ents}

    # Check keyword overlap
    keyword_overlap = keywords_prompt.intersection(keywords_response)
    entity_overlap = entities_prompt.intersection(entities_response)

    # Calculate similarity ratio for overall text comparison
    sim_ratio = similarity_ratio(prompt, response)

    # Validation criteria
    valid_context = len(keyword_overlap) > 0 or len(entity_overlap) > 0
    valid_similarity = sim_ratio > 0.4  # Adjust threshold based on requirement

    return {
        "valid_context": valid_context,
        "valid_similarity": valid_similarity,
        "keyword_overlap": list(keyword_overlap),
        "entity_overlap": list(entity_overlap),
        "similarity_ratio": sim_ratio
    }


def validate_and_filter_json(input_file, validation_output_file, filtered_output_file):
    """
    Validate a JSON file of prompt-response pairs and save the results,
    also create a filtered version of the prompts with specific criteria.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    validation_results = []
    filtered_prompts = []

    for i, pair in enumerate(data):
        prompt = pair.get("prompt", "").strip()
        response = pair.get("response", "").strip()

        if not prompt or not response:
            validation_results.append({
                "index": i,
                "prompt": prompt,
                "response": response,
                "valid": False,
                "reason": "Missing prompt or response"
            })
            continue

        # Validate the pair
        validation = validate_prompt_response_pair(prompt, response)
        is_valid = validation["valid_context"] and validation["valid_similarity"]

        validation_results.append({
            "index": i,
            "prompt": prompt,
            "response": response,
            "valid": is_valid,
            "details": validation
        })

        # Filtering conditions
        if validation["similarity_ratio"] >= 0.15 or prompt.startswith("¿Cuál es el valor de"):
            filtered_prompts.append(pair)

    # Save the validation results
    with open(validation_output_file, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=4)

    # Save the filtered prompts
    with open(filtered_output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_prompts, f, ensure_ascii=False, indent=4)

    print(f"Validation results saved to {validation_output_file}")
    print(f"Filtered prompts saved to {filtered_output_file}")


# Input and output file paths
input_file = "processed/_2_generated_prompts_FULL.json"
validation_output_file = "processed/_3_generated_prompts_clean.json"
filtered_output_file = "processed/_3_generated_prompts_FULL.json"

# Run validation and filtering
validate_and_filter_json(input_file, validation_output_file, filtered_output_file)
