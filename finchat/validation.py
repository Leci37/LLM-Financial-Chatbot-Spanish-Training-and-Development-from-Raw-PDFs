import spacy
nlp = spacy.load("es_core_news_md")


def validate_prompt_response_NLP(prompt, response, threshold=0.1):
    """
    Validate that the response corresponds to the prompt by checking entity and keyword overlap.
    :param prompt: The generated prompt.
    :param response: The corresponding response.
    :param threshold: Minimum ratio of terms that must match in the response.
    :return: True if validation passes, False otherwise.
    """
    doc_prompt = nlp(prompt)
    doc_response = nlp(response)

    # Extract entities and keywords from the prompt
    entities_prompt = {ent.text.lower() for ent in doc_prompt.ents}
    keywords_prompt = {chunk.text.lower() for chunk in doc_prompt.noun_chunks}

    # Combine entities and keywords
    important_terms = entities_prompt.union(keywords_prompt)

    # Check the overlap of important terms in the response
    matched_terms = [term for term in important_terms if term in response.lower()]
    match_ratio = len(matched_terms) / len(important_terms) if important_terms else 1.0

    # Logging or debugging output (optional)
    if match_ratio < threshold:
        print(f"Validation failed for:\nPrompt: {prompt}\nResponse: {response}")
        print(f"Missing terms: {important_terms - set(matched_terms)}")

    # Validation passes if the match ratio meets or exceeds the threshold
    return match_ratio >= threshold