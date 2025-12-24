import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


def create_output_dir(base_dir: str = "outputs") -> Dict[str, str]:
    dirs = {
        "base": base_dir,
        "results": os.path.join(base_dir, "results"),
        "logs": os.path.join(base_dir, "logs"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def save_results(results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
    dirs = create_output_dir()
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crescendo_results_{timestamp}.json"
    
    filepath = os.path.join(dirs["results"], filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def print_result_summary(results: List[Dict[str, Any]]):
    if not results:
        print("No results to summarize.")
        return
    
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    failed = total - successful
    
    turns = [r.get("num_turns", 0) for r in results if "num_turns" in r]
    avg_turns = sum(turns) / len(turns) if turns else 0
    
    durations = [r.get("duration_seconds", 0) for r in results if "duration_seconds" in r]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total tasks:    {total}")
    print(f"Successful:     {successful} ({successful/total*100:.1f}%)")
    print(f"Failed:         {failed} ({failed/total*100:.1f}%)")
    print(f"Avg turns:      {avg_turns:.1f}")
    print(f"Avg duration:   {avg_duration:.1f}s")
    print(f"Total time:     {sum(durations):.1f}s ({sum(durations)/60:.1f}min)")
    print(f"{'='*70}\n")


def load_tasks_from_json(filepath: str) -> List[Dict[str, str]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Task file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    print(f"Loaded {len(tasks)} tasks from {filepath}")
    return tasks


def print_task_progress(current: int, total: int, task_name: str):
    progress = (current / total) * 100
    print(f"\n{'='*70}")
    print(f"Progress: [{current}/{total}] ({progress:.1f}%) - Task: {task_name}")
    print(f"{'='*70}")
