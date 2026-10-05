# legacy/

Kept for reference; nothing in `pipeline/` or `finchat/` uses these files.

| File | Was | Why it is here |
|---|---|---|
| `extract_helpers_unused.py` | `_1_utils.py` | `remove_uniform_columns` and an OpenAI-based `generate_title`; no script imports it |
| `processed/_1_.json` | `processed/_1_.json` | an early extraction of W1 (1 section, 9 tables titled "Table 1" …), replaced by `data/processed/_1_w1.json` |
| `processed/_2_generated_prompts_TABLE_row.json` | same name in `processed/` | 1,699 pairs from an earlier run of the table questions; 1,189 of them are also in `_2_generated_prompts_TABLE.json`, which replaced it |
