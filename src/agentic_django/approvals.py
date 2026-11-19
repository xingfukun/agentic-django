"""Persist SDK approval checkpoints and reconstruct their interruption state."""

from __future__ import annotations

from collections.abc import Callable

from agents import Agent, RunState
from agents.result import RunResult, RunResultStreaming
from agents.run_context import RunContextWrapper
from django.db import transaction

from agentic_django.models import AgentApproval, AgentRun
from agentic_django.serializers import Serializer
from agentic_django.types import json_value


def pause_for_approval(
    run: AgentRun,
    result: RunResult | RunResultStreaming,
    serializer: Serializer,
    on_requested: Callable[[AgentRun, AgentApproval], None],
) -> None:
    """Atomically save an opaque SDK checkpoint and its reviewable requests."""
    snapshot = json_value(result.to_state().to_json())
    with transaction.atomic():
        run.approval_round += 1
        run.run_state = snapshot
        run.status = AgentRun.Status.WAITING_APPROVAL
        run.task_id = ""
        run.save(
            update_fields=[
                "approval_round",
                "run_state",
                "status",
                "task_id",
                "updated_at",
            ]
        )
        for index, interruption in enumerate(result.interruptions):
            approval = AgentApproval.objects.create(
                run=run,
                round=run.approval_round,
                interruption_index=index,
                tool_name=interruption.tool_name or "",
                request=json_value(serializer.serialize(interruption.raw_item)),
            )
            on_requested(run, approval)


async def restore_run_state(
    run: AgentRun,
    agent: Agent[object],
    context: object,
    decisions: list[AgentApproval],
) -> RunState[object, Agent[object]] | None:
    """Restore the SDK checkpoint and apply persisted decisions in order."""
    if run.run_state is None:
        return None
    if not isinstance(run.run_state, dict):
        raise ValueError("Stored run state is invalid")
    state = await RunState.from_json(
        agent,
        run.run_state,
        context_override=RunContextWrapper(context) if context is not None else None,
    )
    interruptions = state.get_interruptions()
    if len(interruptions) != len(decisions) or any(
        decision.decision == AgentApproval.Decision.PENDING for decision in decisions
    ):
        raise ValueError("Run approval decisions are incomplete")
    for approval in decisions:
        interruption = interruptions[approval.interruption_index]
        if approval.decision == AgentApproval.Decision.APPROVED:
            state.approve(interruption)
        else:
            state.reject(interruption)
    return state
