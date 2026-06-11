from datasets import load_dataset

def load_truthfulqa(max_samples=100, categories=None):
    ds = load_dataset('truthfulqa/truthful_qa', 'generation')
    data = ds['validation']
    
    if categories:
        data = data.filter(lambda x: x['category'] in categories)
    
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    prompts = []
    for row in data:
        prompts.append({
            "question": row["question"],
            "best_answer": row["best_answer"],
            "incorrect_answers": row["incorrect_answers"],
            "category": row["category"]
        })
    
    return prompts

if __name__ == "__main__":
    prompts = load_truthfulqa(max_samples=10)
    for p in prompts:
        print(f"Q: {p['question']}")
        print(f"Best: {p['best_answer']}")
        print()
