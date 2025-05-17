import json
from pathlib import Path
from typing import List, Union

def save_tool_calling_output(
    response: Union[str, None],
    messages: List[dict],
    base_path: Path,
    timestamp: str,
    model: str,
    print_output: bool = True
) -> None:
    """
    Save the final response and message history from a tool calling loop.

    Args:
        response (str | None): The final text response to save.
        messages (List[dict]): The full message history to save as JSON.
        base_path (Path): Base directory where the results will be stored.
        timestamp (str): Timestamp string to organize result folders.
        model (str): The model name used (used in folder naming).
    """
    # Create timestamped directory
    results_dir = base_path / "results" / timestamp / model
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    txt_path = results_dir / "final_response.txt"
    json_path = results_dir / "message_history.json"

    # Save final response (if it exists)
    if response:
        txt_path.write_text(response, encoding="utf-8")

    # Save full message history as JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    # Optional: print where the files were saved
    if print_output:
        print(f"✅ Final response saved to: {txt_path}")
        print(f"✅ Message history saved to: {json_path}")