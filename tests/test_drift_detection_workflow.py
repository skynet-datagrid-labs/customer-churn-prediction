"""Tests for the drift-detection workflow configuration.

Scope: validates the `permissions: contents: read` addition made in this PR.
The workflow YAML is parsed with PyYAML and assertions confirm that the
permissions block is present, correctly scoped, and least-privilege.
"""

import os
import pytest
import yaml

WORKFLOW_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    ".github",
    "workflows",
    "drift-detection.yml",
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Load and parse the drift-detection workflow YAML."""
    with open(WORKFLOW_PATH, "r") as fh:
        return yaml.safe_load(fh)


# ─── YAML validity ────────────────────────────────────────────────────────────


def test_workflow_file_exists():
    assert os.path.isfile(WORKFLOW_PATH), (
        "drift-detection.yml not found at expected path"
    )


def test_workflow_yaml_is_parseable():
    """Workflow file must be valid YAML (no parse errors)."""
    with open(WORKFLOW_PATH, "r") as fh:
        parsed = yaml.safe_load(fh)
    assert parsed is not None
    assert isinstance(parsed, dict)


# ─── Permissions block present at workflow root ───────────────────────────────


def test_permissions_key_exists_at_root(workflow):
    """The `permissions` key must exist at the top-level of the workflow."""
    assert "permissions" in workflow, (
        "Expected a top-level 'permissions' key in drift-detection.yml"
    )


def test_permissions_is_a_mapping(workflow):
    """The permissions value must be a mapping (dict), not a string shorthand."""
    assert isinstance(workflow["permissions"], dict), (
        "'permissions' should be a mapping, e.g. {contents: read}"
    )


# ─── contents: read enforcement ───────────────────────────────────────────────


def test_permissions_contents_is_read(workflow):
    """contents permission must be explicitly set to 'read'."""
    permissions = workflow["permissions"]
    assert "contents" in permissions, (
        "Expected 'contents' key inside 'permissions'"
    )
    assert permissions["contents"] == "read", (
        f"Expected permissions.contents == 'read', got {permissions['contents']!r}"
    )


def test_permissions_contents_is_not_write(workflow):
    """contents permission must NOT be 'write' – regression guard."""
    permissions = workflow["permissions"]
    assert permissions.get("contents") != "write", (
        "permissions.contents must not be 'write'"
    )


def test_permissions_contents_is_not_write_all(workflow):
    """Workflow must not use the blanket 'write-all' shorthand."""
    # write-all is expressed as the string "write-all" instead of a mapping
    assert workflow.get("permissions") != "write-all", (
        "Workflow must not use the 'write-all' permissions shorthand"
    )


def test_permissions_contents_is_not_none(workflow):
    """contents permission must not be None/missing after the key is added."""
    permissions = workflow["permissions"]
    value = permissions.get("contents")
    assert value is not None, "permissions.contents must not be None"
    assert value != "", "permissions.contents must not be an empty string"


# ─── Permissions scoped at workflow level, not only inside jobs ───────────────


def test_permissions_not_only_inside_jobs(workflow):
    """The permissions block must live at the workflow level, not exclusively
    inside individual job definitions."""
    # Confirm root-level key exists (already checked above, here for clarity)
    assert "permissions" in workflow

    # Verify `jobs` is present and is a mapping
    jobs = workflow.get("jobs", {})
    assert isinstance(jobs, dict)


def test_permissions_defined_before_jobs(workflow):
    """The top-level 'permissions' key must be present independently of jobs."""
    # PyYAML dicts preserve insertion order (Python 3.7+), so we can check
    # that the 'permissions' key appears before 'jobs'
    keys = list(workflow.keys())
    assert "permissions" in keys
    if "jobs" in keys:
        assert keys.index("permissions") < keys.index("jobs"), (
            "'permissions' should appear before 'jobs' in the workflow file"
        )


# ─── Least-privilege: no unexpected extra permissions ─────────────────────────


def test_permissions_only_contains_contents(workflow):
    """The permissions block introduced in this PR should only contain
    'contents'; no additional permission scopes should have been added."""
    permissions = workflow["permissions"]
    extra = {k for k in permissions if k != "contents"}
    assert not extra, (
        f"Unexpected extra permission scopes found: {extra}. "
        "Only 'contents' was intended to be added in this PR."
    )


# ─── Regression: permissions value is the read-only string literal ────────────


def test_permissions_contents_exact_string(workflow):
    """Regression: the exact string 'read' (not 'Read', 'READ', etc.)."""
    assert workflow["permissions"]["contents"] == "read"


# ─── Broader workflow structural integrity ────────────────────────────────────


def test_workflow_has_required_top_level_keys(workflow):
    """Workflow must still contain its core structural keys after the PR."""
    for key in ("name", "on", "env", "jobs"):
        assert key in workflow, f"Expected top-level key '{key}' to be present"


def test_permissions_does_not_shadow_job_level_permissions(workflow):
    """If any individual job defines its own permissions mapping, it must not
    grant more than the workflow-level permissions for contents."""
    jobs = workflow.get("jobs", {})
    for job_name, job_def in jobs.items():
        if not isinstance(job_def, dict):
            continue
        job_perms = job_def.get("permissions", {})
        if isinstance(job_perms, dict):
            job_contents = job_perms.get("contents")
            assert job_contents != "write", (
                f"Job '{job_name}' grants contents:write, "
                "which conflicts with the workflow-level contents:read"
            )