import os
import sys
import json
import random
import importlib
import concurrent.futures
from typing import Dict
from datasets import load_dataset

sys.path.append('/sandbox/workspace')

def execute_problem(problem_item, system_path):
    try:
        # Load the system
        system_module = importlib.import_module(system_path)
        workflow, _ = system_module.build_system()
        
        # Run the problem through the system
        input_state = {"messages": [], "question": problem_item["question"], "options": problem_item["options"]}
        output = workflow.invoke(input_state)
        
        predicted = output.get("solution", "")
        expected = problem_item["answer"]
        
        is_correct = predicted == expected
        
        return {
            "question_id": problem_item["question_id"],
            "question": problem_item["question"],
            "predicted": predicted,
            "expected": expected,
            "is_correct": is_correct,
            "category": problem_item["category"]
        }
    except Exception as e:
        return {
            "question_id": problem_item["question_id"],
            "question": problem_item["question"],
            "predicted": f"Exception: {repr(e)}",
            "expected": problem_item["answer"],
            "is_correct": False,
            "category": problem_item["category"]
        }

def run_benchmark_parallel(system_path, max_workers=4):
    """Run the MMLU-Pro benchmark with parallel execution."""
    print(f"Running benchmark for: {system_path}")
    
    # Load the dataset
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")['test']
        
        target_category = "computer science"
        filtered_data = []
    
        category_data = [item for item in dataset if item["category"].lower() == target_category.lower()]
        
        if len(category_data) > 120:
            random.seed(42)
            category_data = random.sample(category_data, 120)
        
        filtered_data.extend(category_data)
        print(f"Added {len(category_data)} questions from category: {target_category}")
        
        dataset = filtered_data
        print(f"Loaded dataset with {len(dataset)} problems from category: {target_category}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    results = {
        "system": system_path,
        "total_problems": len(dataset),
        "correct": 0,
        "incorrect": 0,
        "problem_results": {}
    }
    
    # Execute all problems in parallel
    print(f"Executing problems in parallel (max_workers={max_workers})...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(execute_problem, problem_item, system_path): i 
            for i, problem_item in enumerate(dataset)
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_problem), 1):
            problem_idx = future_to_problem[future]
            problem_item = dataset[problem_idx]
            category = problem_item["category"].lower()
            
            try:
                result_info = future.result()
                is_correct = result_info["is_correct"]
                
                if is_correct:
                    print(f"✓ Problem {problem_idx+1} ({category}): Correct")
                    results["correct"] += 1
                else:
                    print(f"✗ Problem {problem_idx+1} ({category}): Incorrect. Expected: {result_info['expected']}, Got: {result_info['predicted']}")
                    results["incorrect"] += 1
                
                results["problem_results"][str(problem_idx)] = result_info
                
                print(f"Progress: {i}/{len(dataset)} problems processed")
                
            except Exception as exc:
                print(f"Problem {problem_idx+1} generated an exception: {exc}")
                results["incorrect"] += 1
    
    # Calculate metrics
    total_attempted = results["correct"] + results["incorrect"]
    results["accuracy"] = results["correct"] / total_attempted if total_attempted > 0 else 0
    
    results_file = f"sandbox/workspace/benchmark/MMLUPro/results/benchmark_results_{system_path}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark Summary:")
    print(f"Total problems: {total_attempted}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Overall Accuracy: {results['accuracy'] * 100:.2f}%")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MMLU-Pro benchmark")
    parser.add_argument("--system", required=True, help="System path to benchmark")
    
    args = parser.parse_args()
    
    run_benchmark_parallel(args.system)