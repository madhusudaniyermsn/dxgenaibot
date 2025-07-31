import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from fuzzywuzzy import fuzz
import time

# ===== CONFIGURATION =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './zephyr_local_model'  # Update to your Zephyr model folder
# MODEL_PATH = './fine_tuned_zephyr_lora_model' # Uncomment and update if using LoRA
CSV_PATH = './data/template.csv'

# ===== LOAD MODEL & DATA =====
def load_components():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        df = pd.read_csv(CSV_PATH).fillna("")
        knowledge_base = {}

        # Get actual column names from the CSV header
        actual_column_names = [col.strip().lower() for col in df.columns.tolist()]
        print(f"DEBUG: Actual column names from CSV: {actual_column_names}")


        for _, row in df.iterrows():
            extract_name = str(row.iloc[0]).strip().lower()
            # Use the cleaned actual column names as keys in the knowledge base
            knowledge_base[extract_name] = {actual_column_names[i]: str(row.iloc[i]).strip() for i in range(len(actual_column_names))}


        return tokenizer, model, knowledge_base, actual_column_names # Return actual column names too
    except Exception as e:
        raise Exception(f"Initialization failed: {str(e)}")


# ===== USE LLM TO PARSE QUERY =====
def analyze_query(message, tokenizer, model):
    system_prompt = (
        "You are an intelligent assistant for data extracts. "
        "Given a user query, predict the following as JSON:\n"
        "- intent: one of ['lookup', 'thank_you', 'exit', 'unknown']\n"
        "- identifier: (extract name if mentioned)\n"
        "- column: (column name if mentioned)\n"
        "- value: (any specific source code/table name value)\n"
        "Respond ONLY as valid JSON without extra text."
    )

    input_text = f"{system_prompt}\nUser Query: {message}\n"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # --- Improved JSON extraction and flexible parsing ---
    json_matches = re.findall(r'\{.*?\}', response, re.DOTALL)
    parsed = {'intent': 'unknown', 'identifier': None, 'column': None, 'value': None}

    for json_str in json_matches:
        try:
            temp_parsed = json.loads(json_str)
            parsed.update({k: v for k, v in temp_parsed.items() if k in parsed})
            if parsed.get('intent') in ['lookup', 'thank_you', 'exit', 'unknown']:
                 break
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"DEBUG: Error parsing potential JSON: {e} in string: {json_str}")
            continue

    valid_intents = ['lookup', 'thank_you', 'exit', 'unknown']
    if parsed.get('intent') not in valid_intents:
        parsed['intent'] = 'unknown'

    parsed['identifier'] = str(parsed.get('identifier')).strip().lower() if parsed.get('identifier') is not None else None
    parsed['column'] = str(parsed.get('column')).strip().lower() if parsed.get('column') is not None else None
    parsed['value'] = str(parsed.get('value')).strip() if parsed.get('value') is not None else None

    # Note: We will handle column name normalization and matching more robustly in generate_response

    print(f"DEBUG: Parsed prediction: {parsed}")

    return parsed


# ===== RESPONSE GENERATOR =====
def generate_response(message, tokenizer, model, knowledge_base, actual_column_names):
    if not message or not message.strip():
        return "⚠️ Please enter a valid question."

    prediction = analyze_query(message, tokenizer, model)
    print("DEBUG Zephyr Prediction:", json.dumps(prediction, indent=2))

    intent = prediction.get('intent')
    identifier = prediction.get('identifier')
    column = prediction.get('column')
    value = prediction.get('value')

    if intent == 'exit':
        return "Goodbye! 👋"
    if intent == 'thank_you':
        return "You're welcome! 😊"

    if intent == 'lookup':
        # --- New Logic: Identify potential column names and extract names in the original user message ---
        identified_column_from_message = None
        best_col_score_message = 0
        identified_identifier_from_message = None
        best_id_score_message = 0
        message_lower = message.lower()

        for actual_col in actual_column_names:
            # Use partial_ratio to find if the column name is part of the message
            score = fuzz.partial_ratio(actual_col, message_lower)
            # Also check if the actual column name is a substring in the message
            if actual_col in message_lower: # Prefer exact substring match
                identified_column_from_message = actual_col
                best_col_score_message = 100 # Assign high score for exact match
                break # Stop at the first exact match
            elif score > best_col_score_message:
                 best_col_score_message = score
                 identified_column_from_message = actual_col

        # Identify potential extract names from the message
        for extract_name_key in knowledge_base.keys():
             score = fuzz.partial_ratio(extract_name_key, message_lower)
             if extract_name_key in message_lower: # Prefer exact substring match
                  identified_identifier_from_message = extract_name_key
                  best_id_score_message = 100
                  break
             elif score > best_id_score_message:
                  best_id_score_message = score
                  identified_identifier_from_message = extract_name_key


        print(f"DEBUG: Identified column from message: '{identified_column_from_message}' with score {best_col_score_message}")
        print(f"DEBUG: Identified identifier from message: '{identified_identifier_from_message}' with score {best_id_score_message}")
        # --- End New Logic ---


        # --- Improved Logic: Determine search strategy based on combined information ---
        search_value = None
        search_column = None
        search_identifier = None
        requesting_all_details = False

        # Determine if the user is requesting all details based on the original message
        # Added condition for prompts starting with "search for value"
        if any(phrase in message_lower for phrase in ["tell me about", "details for", "show me", "everything on", "details"]) or message_lower.startswith("search for value"):
             requesting_all_details = True
             print("DEBUG: User is requesting all details.")

        # Strategy 1: Prioritize Value Search if a value is predicted by LLM or identified from message
        if value:
             search_value = value
             # If LLM predicted a column and it's valid, use it
             if column and column in actual_column_names:
                  search_column = column
                  print(f"DEBUG: Strategy 1: Using LLM predicted column '{search_column}' for value search.")
             # Otherwise, if a column was identified from the message with high confidence, use that
             elif identified_column_from_message and best_col_score_message > 80:
                  search_column = identified_column_from_message
                  print(f"DEBUG: Strategy 1: Using message identified column '{search_column}' for value search.")
             else:
                  print("DEBUG: Strategy 1: No specific column determined for value search, searching all.")

        # Strategy 2: If no value from LLM, but identifier and column are present, and identifier looks like a value
        # This heuristic is less likely to be needed with improved column determination, but kept as a fallback
        elif column and identifier and not value:
             print(f"DEBUG: Strategy 2: Applying heuristic - Using identifier '{identifier}' as search value.")
             search_value = identifier
             # Prioritize LLM's predicted column if valid, otherwise use message column if identified with high confidence
             if column and column in actual_column_names:
                  search_column = column
                  print(f"DEBUG: Strategy 2: Using LLM predicted column '{search_column}' for value search.")
             elif identified_column_from_message and best_col_score_message > 80:
                  search_column = identified_column_from_message
                  print(f"DEBUG: Strategy 2: Using message identified column '{search_column}' for value search.")
             else:
                  print("DEBUG: Strategy 2: No specific column determined for value search, searching all.")


        # Strategy 3: If no value search triggered, check if an identifier is predicted by LLM or identified from message
        if not search_value and (identifier or identified_identifier_from_message):
             # Prioritize identifier identified from message if the score is high
             if identified_identifier_from_message and best_id_score_message > 70: # Use message identifier if identified with decent score
                  search_identifier = identified_identifier_from_message
                  print(f"DEBUG: Strategy 3: Using message identified identifier: '{search_identifier}'")
             # Otherwise, use LLM's predicted identifier if it exists
             elif identifier:
                  search_identifier = identifier
                  print(f"DEBUG: Strategy 3: Using LLM predicted identifier: '{search_identifier}'")

             # If an identifier is determined, check if a specific column is requested
             if search_identifier:
                  # Prioritize LLM's predicted column if valid
                  if column and column in actual_column_names:
                       search_column = column
                       print(f"DEBUG: Strategy 3: User requested specific column '{search_column}' (LLM predicted).")
                  # Otherwise, check if a column was identified from the message with high confidence
                  elif identified_column_from_message and best_col_score_message > 80:
                       search_column = identified_column_from_message
                       print(f"DEBUG: Strategy 3: User requested specific column '{search_column}' (message identified).")
                  # If no specific column requested, the requesting_all_details flag will handle the response format

        # --- End Improved Logic ---


        # --- Perform the determined search ---
        if search_value:
            matches = []
            search_value_lower = search_value.lower()
            print(f"DEBUG: Performing value search for '{search_value_lower}' in column '{search_column}' (if specified)")

            # Strictly search only in the determined column if specified
            if search_column and search_column in knowledge_base.get(list(knowledge_base.keys())[0], {}):
                 for key, data in knowledge_base.items():
                     if search_column in data:
                          if search_value_lower in data[search_column].lower() or fuzz.partial_ratio(search_value_lower, data[search_column].lower()) > 80:
                               if requesting_all_details:
                                    matches.append(f"Details for {key.title()}:\n" + "\n".join([f"{k.title()}: {v}" for k, v in data.items()]))
                               else:
                                    matches.append(f"{key.title()} uses {search_column.title()}: {data[search_column]}")
            else: # If no specific column to search in, search all columns (fallback)
                 print("DEBUG: No specific search column determined, searching all columns.")
                 for key, data in knowledge_base.items():
                     for col, val in data.items():
                         if search_value_lower in val.lower() or fuzz.partial_ratio(search_value_lower, val.lower()) > 80:
                             if requesting_all_details:
                                  matches.append(f"Details for {key.title()}:\n" + "\n".join([f"{k.title()}: {v}" for k, v in data.items()]))
                             else:
                                  matches.append(f"{key.title()} uses {col.title()}: {val}")

            if matches:
                matches = list(set(matches)) # Remove duplicates
                return "Found matches:\n" + "\n".join(matches)
            else:
                print(f"DEBUG: No matches found for value '{search_value}'.")
                if search_column:
                     return f"Sorry, no matching extract found for '{search_value}' in the '{search_column.title()}' column."
                else:
                     return "Sorry, no matching extract found for that value."

        elif search_identifier:
            print(f"DEBUG: Performing identifier search for '{search_identifier}'")
            identifier = search_identifier.lower()
            best_match = max(knowledge_base.keys(), key=lambda k: fuzz.partial_ratio(k, identifier))
            if fuzz.partial_ratio(best_match, identifier) > 65: # Slightly increased identifier threshold
                extract_data = knowledge_base[best_match]
                print(f"DEBUG: Found extract '{best_match}' with score {fuzz.partial_ratio(best_match, identifier)}")

                # If a specific column was requested (either by LLM or message)
                if search_column and search_column in extract_data:
                     print(f"DEBUG: Returning specific column '{search_column}' for identifier '{best_match}'.")
                     return f"{search_column.title()} for {best_match.title()}: {extract_data.get(search_column, 'Not Found')}"
                else:
                    # If no specific column requested, return all details
                    print(f"DEBUG: Returning all details for identifier '{best_match}'.")
                    details = "\n".join([f"{k.title()}: {v}" for k, v in extract_data.items()])
                    return f"Details for {best_match.title()}:\n{details}"
            else:
                 print(f"DEBUG: No good extract match found for identifier '{identifier}'. Best score: {fuzz.partial_ratio(best_match, identifier)}")
                 return "Sorry, no matching extract found with that name."

        # If intent is lookup but no search value or search identifier was determined
        else:
             return "Sorry, I couldn't identify what you're looking for. Please specify an extract name or a value, or include a column name like 'source code' or 'table name'."


    # Default response for unknown intent
    return "Sorry, I couldn't understand your query. Please rephrase."


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    try:
        print("Loading components...")
        # Pass actual_column_names from load_components
        tokenizer, model, knowledge_base, actual_column_names = load_components()

        print("Ask a question about the extracts (type 'exit' to quit):")
        while True:
            user_input = input("You: ").strip()
            start_time = time.time()
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                break
            # Pass actual_column_names to generate_response
            response = generate_response(user_input, tokenizer, model, knowledge_base, actual_column_names)
            print(f"Bot: {response}\n")
            elapsed = time.time() - start_time
            print(f"(Response time: {elapsed:.2f} seconds)\n")
            # Check the response from generate_response to see if it's the exit message

            '''if response == "Goodbye! 👋":
                 break'''

    except Exception as e:
        print(f"Application failed to start: {str(e)}")