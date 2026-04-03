import time


# Parallel Logger for handling print statements in parallel execution
class ParallelLogger:
    """Buffered logger for organizing print statements during parallel execution."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.logs = []
        self.start_time = time.time()

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = time.time() - self.start_time
        self.logs.append({
            "timestamp": timestamp,
            "level": level,
            "message": message
        })

    def get_logs(self) -> list[dict]:
        """Get all logged messages."""
        return self.logs

    def print_logs(self, prefix: str = ""):
        """Print all logged messages in order."""
        for entry in self.logs:
            print(f"{prefix}[{entry['timestamp']:.2f}s] {entry['message']}")


class ParallelLogManager:
    """Manager for multiple parallel loggers."""

    def __init__(self):
        self.loggers = {}
        self.execution_order = []

    def get_logger(self, task_id: str) -> ParallelLogger:
        """Get or create a logger for a task."""
        if task_id not in self.loggers:
            self.loggers[task_id] = ParallelLogger(task_id)
            self.execution_order.append(task_id)
        return self.loggers[task_id]

    def print_all_logs(self, title: str = "Parallel Execution Results"):
        """Print all logs in organized format."""
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print(f"{'=' * 80}")

        for task_id in self.execution_order:
            logger = self.loggers[task_id]
            print(f"\n--- Task: {task_id} ---")
            logger.print_logs("  ")

        print(f"{'=' * 80}")

    def clear(self):
        """Clear all loggers."""
        self.loggers.clear()
        self.execution_order.clear()
