import json
import uuid
from langchain_core.messages import HumanMessage, SystemMessage
from app.common.llm import get_llm
from app.common.schemas import ObjectivePlan, FinalSynthesis
from app.agents.factory import build_researcher_agent
from app.clustering.clusterer import cluster_cards
from app.config.settings import settings


def objective_plan_node(state):
    llm = get_llm().with_structured_output(ObjectivePlan)

    prompt = [
        SystemMessage(
            content="""
You are the Objective Agent for Atlas-Workbench.

Given a mission:
- summarize it clearly
- define acceptance criteria
- propose a few seed research questions

Keep it practical and execution-focused.
"""
        ),
        HumanMessage(content=state["mission"]),
    ]

    plan = llm.invoke(prompt)

    return {
        "mission_summary": plan.mission_summary,
        "acceptance_criteria": plan.acceptance_criteria,
        "research_tasks": [
            {
                "task_id": str(uuid.uuid4()),
                "cluster_id": "unassigned",
                "expert_role": "general",
                "question": q,
                "rationale": "Seed task from objective planning",
            }
            for q in plan.seed_research_questions[: settings.max_initial_tasks]
        ],
        "sandbox_cards": [],
        "clusters": [],
        "round_num": 0,
        "max_rounds": settings.max_research_rounds,
        "current_solution": "",
        "final_report": None,
    }


def deploy_researchers_node(state):
    tasks = state.get("research_tasks", [])
    if not tasks:
        return {}

    new_cards = []
    researcher = build_researcher_agent("general")

    for task in tasks:
        response = researcher.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
Mission summary:
{state.get('mission_summary', '')}

Assigned research question:
{task['question']}

Please investigate and provide:
- short answer
- evidence bullets
- source URLs
- open questions
""",
                    }
                ]
            }
        )

        content = response["messages"][-1].content if response.get("messages") else str(response)

        new_cards.append(
            {
                "card_id": str(uuid.uuid4()),
                "title": task["question"][:80],
                "content": content,
                "source_urls": [],
                "agent_role": "researcher",
                "cluster_id": task.get("cluster_id"),
            }
        )

    sandbox_cards = state.get("sandbox_cards", []) + new_cards

    return {
        "sandbox_cards": sandbox_cards,
        "research_tasks": [],
        "round_num": state.get("round_num", 0) + 1,
    }


def butler_cluster_node(state):
    sandbox_cards = state.get("sandbox_cards", [])
    groups = cluster_cards(sandbox_cards)

    llm = get_llm()

    clusters = []
    for idx, cards in groups.items():
        joined = "\n\n".join(f"- {c['title']}: {c['content'][:1000]}" for c in cards[:5])

        response = llm.invoke(
            [
                SystemMessage(
                    content="""
You are the Butler Agent.

Given a small set of related research cards:
- produce a concise cluster title
- produce a 2-3 sentence cluster summary

Return strict JSON:
{"title": "...", "summary": "..."}
"""
                ),
                HumanMessage(content=joined),
            ]
        )

        raw = response.content
        try:
            parsed = json.loads(raw)
            title = parsed["title"]
            summary = parsed["summary"]
        except Exception:
            title = f"Cluster {idx + 1}"
            summary = raw[:400]

        cluster_id = f"cluster_{idx + 1}"

        for card in cards:
            card["cluster_id"] = cluster_id

        clusters.append(
            {
                "cluster_id": cluster_id,
                "title": title,
                "summary": summary,
                "card_ids": [c["card_id"] for c in cards],
            }
        )

    return {
        "sandbox_cards": sandbox_cards,
        "clusters": clusters,
    }


def expert_review_node(state):
    clusters = state.get("clusters", [])
    if not clusters:
        return {"research_tasks": []}

    llm = get_llm()
    tasks = []

    for cluster in clusters:
        response = llm.invoke(
            [
                SystemMessage(
                    content="""
You are an Expert Agent.

Read the cluster summary and propose up to 2 high-value follow-up research questions.
Return strict JSON:
{"expert_role":"general","tasks":[{"question":"...","rationale":"..."}]}
"""
                ),
                HumanMessage(
                    content=f"""
Mission summary:
{state.get('mission_summary', '')}

Cluster title:
{cluster['title']}

Cluster summary:
{cluster['summary']}
"""
                ),
            ]
        )

        try:
            parsed = json.loads(response.content)
            expert_role = parsed.get("expert_role", "general")
            for t in parsed.get("tasks", [])[: settings.max_tasks_per_cluster]:
                tasks.append(
                    {
                        "task_id": str(uuid.uuid4()),
                        "cluster_id": cluster["cluster_id"],
                        "expert_role": expert_role,
                        "question": t["question"],
                        "rationale": t["rationale"],
                    }
                )
        except Exception:
            continue

    return {"research_tasks": tasks}


def synthesize_node(state):
    llm = get_llm().with_structured_output(FinalSynthesis)

    cluster_blob = "\n\n".join(
        f"[{c['title']}]\n{c['summary']}" for c in state.get("clusters", [])
    )

    final = llm.invoke(
        [
            SystemMessage(
                content="""
You are the Objective Agent.

Synthesize the current workspace into:
- executive summary
- key findings
- unresolved risks
- final recommendation

Be honest about uncertainty.
"""
            ),
            HumanMessage(
                content=f"""
Mission:
{state.get('mission_summary', '')}

Acceptance criteria:
{state.get('acceptance_criteria', [])}

Clusters:
{cluster_blob}
"""
            ),
        ]
    )

    return {
        "final_report": final.model_dump(),
        "current_solution": final.executive_summary,
    }


def should_continue_after_expert(state):
    if state.get("round_num", 0) >= state.get("max_rounds", 2):
        return "synthesize"

    if not state.get("research_tasks"):
        return "synthesize"

    return "research"