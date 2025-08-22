# scripts/ft_dataset_prep.py

import json
import os
import re

def prepare_ft_dataset():
    """
    Loads Q/A pairs from a text file, formats them for supervised instruction fine-tuning,
    and saves the result to a JSON file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    qa_text_path = os.path.join(project_root, "data", "Q&A.txt")
    
    if not os.path.exists(qa_text_path):
        raise FileNotFoundError(f"Q/A text file not found at: {qa_text_path}")
    
    print(f"Reading and parsing Q/A pairs from: {qa_text_path}")
    
    with open(qa_text_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    # Use a regex pattern to find and extract Q&A pairs
    # This pattern looks for "Q:" followed by the question, and "A:" followed by the answer
    pattern = re.compile(r'\d+\.\s*Q:\s*(.*?)\s*A:\s*(.*?)(?=\n\d+\.|\Z)', re.DOTALL)
    matches = pattern.findall(text_content)
    
    if not matches:
        raise ValueError("Could not find any Q&A pairs in the provided text file.")

    # Format the data into an instruction-style dataset
    ft_dataset = []
    for question, answer in matches:
        # Clean up whitespace and newlines
        clean_question = " ".join(question.strip().split())
        clean_answer = " ".join(answer.strip().split())
        
        ft_dataset.append({
            "instruction": f"Please answer the following question based on the provided financial data: {clean_question}",
            "response": clean_answer
        })
        
    # Save the formatted dataset to a JSON file
    ft_dataset_path = os.path.join(project_root, "data", "ft_dataset.json")
    with open(ft_dataset_path, "w", encoding="utf-8") as f:
        json.dump(ft_dataset, f, indent=4)
        
    print(f"Fine-tuning dataset created with {len(ft_dataset)} pairs at {ft_dataset_path}")
    
    return ft_dataset

if __name__ == "__main__":
    prepare_ft_dataset()
