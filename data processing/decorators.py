from time import perf_counter
import logging
from typing import Any, Callable

def benchmark(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()
        value = func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time
        print(f"Execution of {func.__name__} took {run_time:.2f} seconds.")
        return value

    return wrapper