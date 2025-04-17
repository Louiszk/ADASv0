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
        input_state = {"messages": [], "problem": problem_item["input"]}
        output = workflow.invoke(input_state)
        
        # Get the predicted answer and compare with ground truth
        predicted = output.get("solution", "")
        expected = str(problem_item["target"])
        
        try:
            predicted_float = float(predicted)
            expected_float = float(expected)
            # Allow for small differences in floating point
            is_correct = abs(predicted_float - expected_float) < 1e-6
        except ValueError:
            is_correct = predicted == expected
        
        return {
            "question": problem_item["input"],
            "predicted": predicted,
            "expected": expected,
            "is_correct": is_correct,
        }
    except Exception as e:
        return {
            "question": problem_item["input"],
            "predicted": f"Exception: {repr(e)}",
            "expected": str(problem_item["target"]),
            "is_correct": False,
        }

def run_benchmark_parallel(system_path, max_workers=4):
    """Run the GSM-Hard benchmark with parallel execution."""
    print(f"Running benchmark for: {system_path}")
    
    # Load the dataset
    try:
        dataset = load_dataset("reasoning-machines/gsm-hard")['train']
        random.seed(42)
        indices = random.sample(range(len(dataset)), 120)
        dataset = dataset.select(indices)
        print(f"Loaded dataset with {len(dataset)} problems")
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
            executor.submit(execute_problem, problem_item, system_path): idx 
            for idx, problem_item in enumerate(dataset)
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_problem), 1):
            idx = future_to_problem[future]
            
            try:
                result_info = future.result()
                is_correct = result_info["is_correct"]
                
                if is_correct:
                    print(f"✓ Problem {idx}: Correct")
                    results["correct"] += 1
                else:
                    print(f"✗ Problem {idx}: Incorrect. Expected: {result_info['expected']}, Got: {result_info['predicted']}")
                    results["incorrect"] += 1
                
                results["problem_results"][idx] = result_info
                
                print(f"Progress: {i}/{len(dataset)} problems processed")
                
            except Exception as exc:
                print(f"Problem {idx} generated an exception: {exc}")
                results["incorrect"] += 1
    
    # Calculate metrics
    total_attempted = results["correct"] + results["incorrect"]
    results["accuracy"] = results["correct"] / total_attempted if total_attempted > 0 else 0
    
    results_file = f"sandbox/workspace/benchmark/GSMHard/results/benchmark_results_{system_path}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark Summary:")
    print(f"Total problems: {len(dataset)}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GSM-Hard benchmark")
    parser.add_argument("--system", required=True, help="System path to benchmark")
    
    args = parser.parse_args()
    
    run_benchmark_parallel(args.system)