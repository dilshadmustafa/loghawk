
import psutil  # Add this import
import os
import time

# --- Memory Tracking ---
memory_log = []

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def log_memory(stage, details=""):
    """Log memory usage at a specific stage"""
    memory_mb = get_memory_usage()
    timestamp = time.time()
    memory_log.append({
        'stage': stage,
        'memory_mb': memory_mb,
        'timestamp': timestamp,
        'details': details
    })
    print(f"💾 {stage}: {memory_mb:.1f} MB {details}")


def print_memory_summary():
    """Print a summary of memory usage throughout the process"""
    if not memory_log:
        return

    print("\n" + "=" * 50)
    print("📊 MEMORY USAGE SUMMARY")
    print("=" * 50)

    start_memory = memory_log[0]['memory_mb']

    for i, entry in enumerate(memory_log):
        delta = entry['memory_mb'] - memory_log[i - 1]['memory_mb'] if i > 0 else 0
        print(f"{entry['stage']:<25} {entry['memory_mb']:>8.1f} MB  {delta:>+7.1f} MB  {entry['details']}")

    peak_memory = max(log['memory_mb'] for log in memory_log)
    current_memory = memory_log[-1]['memory_mb']

    print("-" * 50)
    print(f"{'Peak Memory Used':<25} {peak_memory:>8.1f} MB")
    print(f"{'Current Memory':<25} {current_memory:>8.1f} MB")
    print(f"{'Total Memory Delta':<25} {current_memory - start_memory:>+8.1f} MB")
    print("=" * 50 + "\n")






