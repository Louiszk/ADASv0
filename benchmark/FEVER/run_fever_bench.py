import os
import sys
import json
import random
import importlib
import concurrent.futures
from typing import Dict

sys.path.append('/sandbox/workspace')

def execute_problem(problem_item, system_path):
    try:
        # Load the system
        system_module = importlib.import_module(system_path)
        workflow, _ = system_module.build_system()
        
        # Run the problem through the system
        input_state = {"messages": [], "claim": problem_item["claim"]}
        output = workflow.invoke(input_state)
        
        # Get the predicted answer and compare with ground truth
        predicted = output.get("prediction", "")
        expected = problem_item["label"]
        
        is_correct = predicted == expected
        
        return {
            "id": problem_item["id"],
            "claim": problem_item["claim"],
            "predicted": predicted,
            "expected": expected,
            "is_correct": is_correct,
        }
    except Exception as e:
        return {
            "id": problem_item["id"],
            "claim": problem_item["claim"],
            "predicted": f"Exception: {repr(e)}",
            "expected": problem_item["label"],
            "is_correct": False,
        }

def run_benchmark_parallel(system_path, max_workers=4):
    """Run the FEVER benchmark with parallel execution."""
    print(f"Running benchmark for: {system_path}")
    
    # Load the dataset
    try:
        with open('sandbox/workspace/benchmark/FEVER/fever_subset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded dataset with {len(dataset)} problems")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    results = {
        "system": system_path,
        "total_problems": len(dataset),
        "correct": 0,
        "incorrect": 0,
        "problem_results": {},
        "label_metrics": {
            "SUPPORTS": {"true": 0, "false": 0, "total": 0},
            "REFUTES": {"true": 0, "false": 0, "total": 0},
            "NOT ENOUGH INFO": {"true": 0, "false": 0, "total": 0}
        }
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
                expected_label = result_info["expected"]
                
                # Update label-specific metrics
                if expected_label in results["label_metrics"]:
                    results["label_metrics"][expected_label]["total"] += 1
                    if is_correct:
                        results["label_metrics"][expected_label]["true"] += 1
                    else:
                        results["label_metrics"][expected_label]["false"] += 1
                
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
    
    # Calculate per-label accuracy
    for label in results["label_metrics"]:
        label_total = results["label_metrics"][label]["total"]
        if label_total > 0:
            results["label_metrics"][label]["accuracy"] = results["label_metrics"][label]["true"] / label_total
        else:
            results["label_metrics"][label]["accuracy"] = 0
    
    results_file = f"sandbox/workspace/benchmark/FEVER/results/benchmark_results_{system_path}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark Summary:")
    print(f"Total problems: {len(dataset)}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    
    print("\nPer-label Performance:")
    for label, metrics in results["label_metrics"].items():
        if metrics["total"] > 0:
            print(f"{label}: {metrics['accuracy'] * 100:.2f}% ({metrics['true']}/{metrics['total']})")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FEVER benchmark")
    parser.add_argument("--system", required=True, help="System path to benchmark")
    
    args = parser.parse_args()
    
    run_benchmark_parallel(args.system)