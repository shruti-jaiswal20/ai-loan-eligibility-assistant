"""Builds the agent-workflow flowchart shown on the progress & agent
execution pages, using Streamlit's built-in graphviz support.

Shape reflects the actual LangGraph topology in workflow/graph.py:
document -> validate -> router -> [financial ‖ rag] (parallel) -> risk (join) ->
recommend -> human. The financial/rag pair is drawn side-by-side to make
the parallel execution visually obvious.
"""


def build_workflow_dot(completed_steps: set[str], active_step: str | None = None,
                        route: str | None = None) -> str:
    def color(node: str) -> str:
        if node == active_step:
            return '"#2f6fed"'
        if node in completed_steps:
            return '"#16a34a"'
        return '"#cbd5e1"'

    def fill(node: str) -> str:
        if node == active_step:
            return '"#eaf1ff"'
        if node in completed_steps:
            return '"#eafbf0"'
        return '"#f8fafc"'

    nodes = ["start", "document", "validate", "router", "financial", "rag", "risk", "recommend", "human"]
    labels = {
        "start": "Start",
        "document": "Document\\nIntake Agent",
        "validate": "Input\\nValidation",
        "router": "Router",
        "financial": "Financial\\nAgent",
        "rag": "Policy / RAG\\nAgent",
        "risk": "Risk Agent\\n(join)",
        "recommend": "Recommendation\\nAgent",
        "human": "Human\\nApproval",
    }

    lines = ["digraph G {", 'bgcolor="transparent"', "rankdir=TB",
             'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 margin="0.18,0.12"]',
             'edge [color="#94a3b8" arrowsize=0.7]',
             # keep the parallel pair visually level
             "{rank=same; financial; rag}"]

    for n in nodes:
        lines.append(
            f'{n} [label="{labels[n]}" color={color(n)} fillcolor={fill(n)} fontcolor="#0f172a"]'
        )

    lines += [
        "start -> document", "document -> validate", "validate -> router",
        "router -> financial", "router -> rag",
        "financial -> risk", "rag -> risk",
        "risk -> recommend",
        "recommend -> human",
    ]
    if route:
        lines.append(f'router [xlabel="{route}" fontsize=9 fontcolor="#2f6fed"]')

    lines.append("}")
    return "\n".join(lines)
