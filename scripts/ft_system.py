# scripts/ft_system.py

import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

def load_ft_model():
    """
    Loads the fine-tuned GPT-2 model and tokenizer from the local directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    model_dir = os.path.join(project_root, "models", "gpt2-finetuned")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Fine-tuned model not found at: {model_dir}")

    print("Loading fine-tuned GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print("Fine-tuned model loaded successfully.")
    
    return tokenizer, model

def ft_predict(query, tokenizer, model):
    """
    Generates a clean answer using the fine-tuned model.
    """
    prompt = f"Instruction: {query}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Use greedy decoding and stopping at EOS token
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,          # turn off sampling
            repetition_penalty=2.0,   # reduce repeated tokens
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    try:
        answer = response.split("Response:")[1].strip()
    except IndexError:
        answer = "Sorry, I could not generate a clear response."
    
    confidence = 0.95  # placeholder
    return answer, confidence


def input_guardrail(query):
    """
    Simple input-side guardrail to filter relevant queries.
    """
    financial_keywords = [
        "revenue", "sales", "income", "profit", "assets", "liabilities",
        "cash flow", "dividend", "financial", "margin"
    ]

    is_relevant = any(keyword in query.lower() for keyword in financial_keywords)

    if is_relevant:
        return True, "Query is relevant."
    else:
        return False, "This chatbot is designed for financial questions only."

def run_ft_system(query, components):
    """
    Orchestrates the fine-tuned model pipeline for a given query.
    """
    tokenizer, model = components

    # Guardrail check
    is_relevant, guardrail_message = input_guardrail(query)
    if not is_relevant:
        return {
            "answer": guardrail_message,
            "confidence": 0,
            "response_time": 0,
            "is_relevant": False,
        }

    start_time = time.time()

    # Generate prediction
    answer, confidence = ft_predict(query, tokenizer, model)

    end_time = time.time()
    response_time = end_time - start_time

    return {
        "answer": answer,
        "confidence": confidence,
        "response_time": response_time,
        "is_relevant": True,
    }
