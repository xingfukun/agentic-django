from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from agents import Agent, Model, RunConfig, function_tool
from agents.items import ModelResponse, TResponseOutputItem, TResponseStreamEvent
from agents.usage import Usage
from django.contrib.auth.models import User
from django.test import Client, override_settings
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agentic_django.models import AgentApproval, AgentEvent, AgentRun, AgentSession
from agentic_django.services import execute_run


class ApprovalModel(Model):
    def __init__(self, tool_calls: int = 1) -> None:
        self.calls = 0
        self.tool_calls = tool_calls

    async def get_response(self, *args: object, **kwargs: object) -> ModelResponse:
        self.calls += 1
        output: list[TResponseOutputItem]
        if self.calls == 1:
            output = [
                ResponseFunctionToolCall(
                    arguments=f'{{"value":"updated-{index}"}}',
                    call_id=f"call-{index}",
                    name="write_record",
                    type="function_call",
                )
                for index in range(self.tool_calls)
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id="message-1",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="done", type="output_text"
                        )
                    ],
                )
            ]
        return ModelResponse(output=output, usage=Usage(), response_id=str(self.calls))

    async def stream_response(
        self, *args: object, **kwargs: object
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError
        yield  # type: ignore[unreachable]


def _assert_write_outcome(writes: list[str], decision: str) -> None:
    assert writes == (["updated-0"] if decision == "approved" else [])


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_real_sdk_approval_resume_and_owner_scope(
    monkeypatch: pytest.MonkeyPatch, user: User, client: Client, decision: str
) -> None:
    writes: list[str] = []

    @function_tool(needs_approval=True)
    def write_record(value: str) -> str:
        """Update a record."""
        writes.append(value)
        return "updated"

    model = ApprovalModel()
    monkeypatch.setattr(
        "agentic_django.services.get_agent",
        lambda key: Agent(name="approval", model=model, tools=[write_record]),
    )
    # Keep queue delivery explicit so the checkpoint can be inspected first.
    monkeypatch.setattr("agentic_django.services.enqueue_agent_run", lambda id: None)
    session = AgentSession.objects.create(
        session_key="approval-integration", owner=user
    )
    run = AgentRun.objects.create(
        session=session, owner=user, agent_key="default", input_payload="update"
    )
    with override_settings(
        AGENTIC_DJANGO_ENABLE_EVENTS=False,
        AGENTIC_DJANGO_DEFAULT_RUN_OPTIONS={
            "run_config": RunConfig(tracing_disabled=True),
        },
    ):
        execute_run(str(run.id))
        run.refresh_from_db()
        assert run.status == AgentRun.Status.WAITING_APPROVAL
        assert isinstance(run.run_state, dict)
        approval = AgentApproval.objects.get(run=run)
        assert approval.tool_name == "write_record"
        client.force_login(user)
        assert client.get(f"/runs/{run.id}/events/").status_code == 404

        client.force_login(User.objects.create_user(username="another"))
        url = f"/runs/{run.id}/approvals/{approval.id}/"
        assert client.post(url, {"decision": decision}).status_code == 404

        client.force_login(user)
        assert client.post(url, {"decision": decision}).status_code == 200
        assert client.post(url, {"decision": decision}).status_code == 200
        assert (
            client.post(
                url, {"decision": "rejected" if decision == "approved" else "approved"}
            ).status_code
            == 409
        )
        run.refresh_from_db()
        assert run.status == AgentRun.Status.PENDING

        execute_run(str(run.id))
        run.refresh_from_db()
        assert run.status == AgentRun.Status.COMPLETED
        assert AgentRun.objects.get(pk=run.pk).run_state is None
        _assert_write_outcome(writes, decision)
        assert list(
            AgentEvent.objects.filter(run=run).values_list("event_type", flat=True)
        ) == ["approval_requested", "approval_resolved"]


@pytest.mark.django_db(transaction=True)
def test_multiple_approvals_resume_only_after_all_decisions(
    monkeypatch: pytest.MonkeyPatch, user: User, client: Client
) -> None:
    writes: list[str] = []

    @function_tool(needs_approval=True)
    def write_record(value: str) -> str:
        """Update a record."""
        writes.append(value)
        return "updated"

    model = ApprovalModel(tool_calls=2)
    monkeypatch.setattr(
        "agentic_django.services.get_agent",
        lambda key: Agent(name="approval", model=model, tools=[write_record]),
    )
    monkeypatch.setattr("agentic_django.services.enqueue_agent_run", lambda id: None)
    session = AgentSession.objects.create(session_key="two-calls", owner=user)
    run = AgentRun.objects.create(
        session=session, owner=user, agent_key="default", input_payload="update twice"
    )
    client.force_login(user)
    with override_settings(
        AGENTIC_DJANGO_ENABLE_EVENTS=False,
        AGENTIC_DJANGO_DEFAULT_RUN_OPTIONS={
            "run_config": RunConfig(tracing_disabled=True),
        },
    ):
        execute_run(str(run.id))
        approvals = list(run.approvals.order_by("interruption_index"))
        assert len(approvals) == 2
        first_url = f"/runs/{run.id}/approvals/{approvals[0].id}/"
        second_url = f"/runs/{run.id}/approvals/{approvals[1].id}/"
        assert (
            client.post(first_url, {"decision": "approved"}).json()["status"]
            == AgentRun.Status.WAITING_APPROVAL
        )
        assert model.calls == 1
        assert (
            client.post(second_url, {"decision": "rejected"}).json()["status"]
            == AgentRun.Status.PENDING
        )
        execute_run(str(run.id))
        assert AgentRun.objects.get(id=run.id).status == AgentRun.Status.COMPLETED
        assert writes == ["updated-0"]
