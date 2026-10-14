from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the root directory to PYTHONPATH so we can import torch_judge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_judge.tasks import TASKS, get_task
from torch_judge.web_engine import execute_code
from api.parser import get_all_templates

app = FastAPI(title="TorchCode UI Backend")

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load templates on startup
TEMPLATES = get_all_templates()

class SubmitRequest(BaseModel):
    code: str

@app.get("/api/tasks")
def list_tasks():
    tasks_list = []
    for task_id, task_data in TASKS.items():
        tasks_list.append({
            "id": task_id,
            "title": task_data["title"],
            "difficulty": task_data.get("difficulty", "Unknown")
        })
    return tasks_list

@app.get("/api/tasks/{task_id}")
def get_task_details(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    template = TEMPLATES.get(task_id, {})
    return {
        "id": task_id,
        "title": task["title"],
        "difficulty": task.get("difficulty", "Unknown"),
        "hint": task.get("hint", ""),
        "description": template.get("description", "Description not found."),
        "initial_code": template.get("initial_code", "# Write your code here.")
    }

@app.post("/api/submit/{task_id}")
def submit_code(task_id: str, request: SubmitRequest):
    result = execute_code(task_id, request.code)
    return result


# --- Save a solution and test results as a Jupyter Notebook ---

from pathlib import Path
from datetime import datetime, timezone
import nbformat


@app.post("/api/save/{task_id}")
def save_notebook(task_id: str, request: SubmitRequest):
    # 1. Validate the selected problem
    task = get_task(task_id)
    if task is None or not task_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=404, detail="Unknown task")

    # 2. Locate your personal GitHub repository
    repo = Path.home() / "TorchCode"
    if not (repo / ".git").exists():
        raise HTTPException(
            status_code=400,
            detail=f"Git repository not found: {repo}"
        )

    # 3. Run the official tests again
    result = execute_code(task_id, request.code)
    template = TEMPLATES.get(task_id, {})

    # 4. Create a Jupyter Notebook
    nb = nbformat.v4.new_notebook()

    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }

    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"# {task['title']}\n\n"
        f"{template.get('description', '')}\n\n"
        "Source: https://github.com/duoan/TorchCode"
    ))

    
    # Your implementation and its captured outputs
    solution_cell = nbformat.v4.new_code_cell(request.code)

    # Attach stdout and stderr captured by the TorchCode judge
    for stream_name in ("stdout", "stderr"):
        captured = result.get(stream_name, "")

        if captured:
            solution_cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="stream",
                    name=stream_name,
                    text=captured
                )
            )

    nb.cells.append(solution_cell)


    # Overall grading results
    status = "Accepted" if result["success"] else "Not Accepted"
    timestamp = datetime.now(timezone.utc).isoformat()

    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"## TorchCode Test Results\n\n"
        f"**{status}: {result['passed']}/{result['total']}**\n\n"
        f"Saved at (UTC): {timestamp}\n\n"
        "Results below are snapshots from the TorchCode judge."
    ))

    if result.get("error"):
        nb.cells.append(nbformat.v4.new_markdown_cell(
            f"### Execution Error\n\n"
            f"```text\n{result['error']}\n"
            f"{result.get('traceback', '')}\n```"
        ))

    # 5. Save individual test cases
    for i, test in enumerate(result.get("tests", []), 1):
        outcome = "PASS" if test["passed"] else "FAIL"
        info = (
            f"### Case {i}: {test['name']}\n\n"
            f"**{outcome}** — {test['time_ms']:.2f} ms"
        )

        if test.get("error_msg"):
            info += f"\n\nError: `{test['error_msg']}`"

        if test.get("stdout"):
            info += f"\n\nOutput:\n\n```text\n{test['stdout']}\n```"

        if test.get("stderr"):
            info += f"\n\nStderr:\n\n```text\n{test['stderr']}\n```"

        nb.cells.append(nbformat.v4.new_markdown_cell(info))

        test_code = test["code"].replace(
            "{fn}", task["function_name"]
        )
        
        # Create a test code cell
        test_cell = nbformat.v4.new_code_cell(test_code)

        # Attach the results recorded by the TorchCode judge
        output_lines = [
            f"TorchCode judge snapshot: {outcome}",
            f"Execution time: {test['time_ms']:.2f} ms"
        ]

        if test.get("stdout"):
            output_lines.append(test["stdout"].rstrip())

        if test.get("stderr"):
            output_lines.append(test["stderr"].rstrip())

        if not test["passed"] and test.get("error_msg"):
            output_lines.append(
                f"Error: {test['error_msg']}"
            )

        test_cell.outputs = [
            nbformat.v4.new_output(
                output_type="stream",
                name="stdout",
                text="\n".join(output_lines) + "\n"
            )
        ]

        nb.cells.append(test_cell)


    # 6. Save to your personal repository
    output_dir = repo / "solutions"
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # Find the original numbered notebook template
    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    matches = sorted(
        templates_dir.glob(f"[0-9][0-9]_{task_id}.ipynb")
    )

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Numbered template not found for {task_id}"
        )

    output_path = output_dir / matches[0].name


    nbformat.validate(nb)
    nbformat.write(nb, output_path)

    return {
        "saved": True,
        "path": str(output_path),
        "passed": result["passed"],
        "total": result["total"]
    }



# --- Publish a saved notebook to GitHub ---

import subprocess


@app.post("/api/publish/{task_id}")
def publish_notebook(task_id: str, request: SubmitRequest):

    # Save and grade the current code first
    saved = save_notebook(task_id, request)

    # Only publish accepted solutions
    if saved["passed"] != saved["total"]:
        raise HTTPException(
            status_code=409,
            detail="Notebook saved locally, but tests did not all pass."
        )

    repo = (Path.home() / "TorchCode").resolve()
    notebook = Path(saved["path"]).resolve()
    relative_path = notebook.relative_to(repo).as_posix()

    # Run Git commands inside the personal repository
    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise HTTPException(status_code=500, detail=str(e))

        if result.returncode != 0:
            raise HTTPException(
                status_code=409,
                detail=(result.stderr or result.stdout).strip()
            )

        return result.stdout.strip()

    # Verify the target repository and branch
    if git("branch", "--show-current") != "main":
        raise HTTPException(409, "Switch to the main branch first.")

    remote = git("remote", "get-url", "origin")
    allowed_remotes = {
        "https://github.com/jhmoon19/TorchCode.git",
        "https://github.com/jhmoon19/TorchCode",
        "git@github.com:jhmoon19/TorchCode.git"
    }

    if remote not in allowed_remotes:
        raise HTTPException(409, "Unexpected GitHub repository.")

    # Check for updates made on GitHub
    git("fetch", "origin", "main")

    check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"],
        cwd=repo
    )

    if check.returncode != 0:
        raise HTTPException(
            409,
            "Remote history has changed. Synchronize Git first."
        )

    # Avoid including unrelated staged changes
    if git("diff", "--cached", "--name-only"):
        raise HTTPException(
            409,
            "Other staged changes exist. Review Git status first."
        )

    # Stage only the saved notebook
    git("add", "--", relative_path)

    # Commit only when there is a change
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", relative_path],
        cwd=repo
    )

    if diff.returncode == 1:
        git(
            "commit", "-m",
            f"Save {notebook.stem} solution and test results",
            "--", relative_path
        )
    elif diff.returncode != 0:
        raise HTTPException(500, "Could not inspect staged changes.")

    # Upload to GitHub
    git("push", "origin", "main")

    return {
        **saved,
        "pushed": True,
        "commit": git("rev-parse", "--short", "HEAD")
    }
