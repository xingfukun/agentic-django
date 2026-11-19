"""Persist SDK approval checkpoints and reconstruct their interruption state."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser

from agents import Agent, RunState
from agents.result import RunResult, RunResultStreaming
from agents.run_context import RunContextWrapper
from asgiref.sync import sync_to_async
from django.db import transaction
from django.utils import timezone

from agentic_django.models import AgentApproval, AgentRun
from agentic_django.serializers import Serializer
from agentic_django.types import json_value


def pause_for_approval(
    run: AgentRun,
    result: RunResult | RunResultStreaming,
    serializer: Serializer,
    on_event: Callable[[str, AgentRun, AgentApproval], None],
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
            on_event("approval_requested", run, approval)


async def restore_run_state(
    run: AgentRun,
    agent: Agent[object],
    context: object,
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
    decisions = await sync_to_async(
        lambda: list(
            AgentApproval.objects.filter(run=run, round=run.approval_round).order_by(
                "interruption_index"
            )
        ),
        thread_sensitive=True,
    )()
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


def decide_approval(
    *,
    run_id: UUID,
    approval_id: UUID,
    owner: AbstractBaseUser,
    decision: str,
    on_event: Callable[[str, AgentRun, AgentApproval], None],
) -> tuple[AgentApproval, bool]:
    """Consume one decision under the caller's dispatch lock; return enqueue flag."""
    if decision not in {
        AgentApproval.Decision.APPROVED,
        AgentApproval.Decision.REJECTED,
    }:
        raise ValueError("Invalid approval decision")
    run = AgentRun.objects.get(id=run_id, owner_id=owner.pk)
    approval = AgentApproval.objects.get(id=approval_id, run=run)
    if approval.decision != AgentApproval.Decision.PENDING:
        if approval.decision != decision:
            raise ValueError("Approval was already resolved differently")
        return approval, False
    if (
        run.status != AgentRun.Status.WAITING_APPROVAL
        or approval.round != run.approval_round
    ):
        raise ValueError("Approval is no longer pending")
    approval.decision = decision
    approval.resolved_at = timezone.now()
    approval.save(update_fields=["decision", "resolved_at"])
    on_event("approval_resolved", run, approval)
    if AgentApproval.objects.filter(
        run=run, round=run.approval_round, decision=AgentApproval.Decision.PENDING
    ).exists():
        return approval, False
    run.status = AgentRun.Status.PENDING
    run.save(update_fields=["status", "updated_at"])
    return approval, True
