"""Register and run a fresh frozen-base study with the audited single-card guardian.

Registration uses CPU only. Model/data content identities come from the root
scientific protocol; the scientific runner verifies those large files before
using them. Only small, immutable protocol/source files enter the guardian's
frequent binding checks. No CUDA loading, process signalling, retries or STOP
removal is implemented here.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import sparse_token_readout_queue as guardian

SCHEMA = "hanguard_frozen_base_readout_resource_plan_v1"
RESOURCE_SCHEMA = "hanguard_frozen_base_readout_single_gpu_queue_v1"
SELECTOR_SCHEMA = "hanguard_frozen_base_readout_cpu_selector_v1"
ATTEMPT = "resource_attempt_001"
CANDIDATES = (6, 2, 4, 5, 7, 3, 0, 1)


def validate_registration(study, stages, gpu, reserved_gpu):
    if (type(gpu) is not int or type(reserved_gpu) is not int or
            gpu not in range(8) or reserved_gpu not in range(8) or gpu == reserved_gpu):
        raise ValueError("Use exactly one physical GPU0–GPU7 and reserve a different card")
    # Reuse its stage/path validation without altering the audited module's
    # original 2/4 policy. This new study registers the actual physical card.
    guardian._validate_registration(study, stages, 2, 4)


def scientific_scope(study):
    """The scientific entrypoint owns sticky flags for the authorized barriers."""
    path = Path(study) / "state.json"
    state = guardian.legacy.read_json(path) if path.exists() else {}
    return dict(formal_validation_opened=state.get("formal_validation_opened") is True,
                formal_test_opened=state.get("formal_test_opened") is True,
                scientific_scope_state_path=str(path))


class ScopedSupervisor(guardian.SingleCardSupervisor):
    """Override scope metadata only; retain all audited ownership/STOP methods."""

    def __init__(self, *args, gpu=2, reserved_gpu=4,
                 inventory=guardian.inventory_module.gpu_inventory,
                 identity_reader=guardian.inventory_module.process_identity,
                 snapshot_reader=guardian.process_snapshot, **kwargs):
        # SingleCardSupervisor.__init__ pins its old study to GPU2/GPU4.
        # Initialize its generic CPU engine and exactly the same ownership
        # attributes, keeping observation/check/execute/termination inherited.
        guardian.legacy.SequentialSupervisor.__init__(self, *args,
                                                       environment=guardian.CPU_ENV, **kwargs)
        validate_registration(self.study_output, self.stages, gpu, reserved_gpu)
        self.gpu, self.reserved_gpu = gpu, reserved_gpu
        self.inventory, self.identity_reader = inventory, identity_reader
        self.snapshot_reader = snapshot_reader
        self.stage_by_name = {stage.name: stage for stage in self.stages}
        self.trigger = None
        self.stop_directories = tuple(dict.fromkeys(self.stop_directories + (self.output.parent,)))
        self.registration["resource_policy"] = dict(physical_gpu=gpu, reserved_gpu=reserved_gpu,
            max_cuda_workers=1, admission_seconds=0, strict_identity=True,
            never_terminate_foreign_processes=True, automatic_retry=False)
        self.registration.update(schema=RESOURCE_SCHEMA,
            evaluation_scope="Fresh frozen-base train/validation followed by selection-locked test",
            formal_test_enabled=True, formal_validation_enabled=True,
            scientific_scope_state_path=str(self.study_output / "state.json"))

    def _state(self, state, **details):
        value = dict(schema=RESOURCE_SCHEMA, state=state, supervisor_pid=os.getpid(),
            updated_unix=time.time(), queue_protocol_sha256=self.protocol_sha,
            stage=self.active_stage, physical_gpu=self.gpu, reserved_gpu=self.reserved_gpu,
            **scientific_scope(self.study_output), **details)
        if self.trigger is not None and state in ("failed", "stopped", "interrupted"):
            value["trigger_evidence"] = dict(self.trigger, termination_started=True,
                termination_completed=self.child is None or self.child.poll() is not None)
        guardian.legacy.dump(self.output / "queue_state.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
        return value

    def run(self):
        try:
            return super().run()
        finally:
            # The legacy engine publishes a failure summary after _state with
            # its old unopened-test literal. Preserve the scientific flags in
            # that advisory artifact as well, without changing execution.
            path = self.output / "diagnostic_summary.json"
            if path.exists():
                summary = guardian.legacy.read_json(path)
                summary.update(scientific_scope(self.study_output))
                guardian.legacy.dump(path, summary)


def _path_digest_map(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must map absolute file paths to SHA256 digests")
    result = {}
    for filename, digest in value.items():
        path = Path(filename)
        if (not path.is_absolute() or not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"Invalid {label} file identity: {filename}")
        name = str(path.resolve())
        if name in result and result[name] != digest:
            raise ValueError(f"Conflicting {label} identity: {filename}")
        result[name] = digest
    return result


def _root_identity(study):
    protocol_path = study / "protocol.json"
    registration_path = study / "protocol_registration.json"
    protocol = guardian.legacy.read_json(protocol_path)
    registration = guardian.legacy.read_json(registration_path)
    digest = guardian.legacy.sha(protocol_path)
    if registration.get("protocol_sha256") != digest:
        raise ValueError("Root protocol differs from its independent SHA registration")
    inputs = _path_digest_map(protocol.get("inputs"), "root inputs")
    sources = _path_digest_map(protocol.get("code_sha256"), "root sources")
    if not inputs or not sources:
        raise ValueError("Root protocol must register model/data inputs and source code")
    return protocol, digest, inputs, sources


def initialize(study, *, gpu=2, reserved_gpu=4):
    """Create one immutable plan/attempt without examining or reserving GPUs."""
    study = Path(study).resolve()
    attempt = study / ATTEMPT
    plan_path = study / "resource_plan.json"
    if attempt.exists() or plan_path.exists():
        raise FileExistsError("Resource plan/attempt is immutable; use a fresh study")
    protocol, protocol_sha, inputs, sources = _root_identity(study)
    python = str((ROOT / ".venv-hanguard/bin/python").absolute())
    runner = ROOT / "scripts/hanguard/frozen_base_readout_study.py"
    if not Path(python).is_file() or not runner.is_file():
        raise FileNotFoundError("Registered hanguard runtime and fresh-study runner are required")
    if str(runner.resolve()) not in sources:
        raise ValueError("Root protocol must bind the fresh-study runner source")
    # Root input digests are bound indirectly by protocol.json. Repeatedly
    # hashing multi-GB model shards inside the inherited guardian would delay
    # STOP polling; the scientific runner owns their full integrity gate.
    input_identities = {}
    for filename, digest in inputs.items():
        stat = Path(filename).stat()
        if not Path(filename).is_file():
            raise ValueError(f"Root input is not a regular file: {filename}")
        input_identities[filename] = dict(sha256=digest, bytes=stat.st_size,
                                          mtime_ns=stat.st_mtime_ns)
    bindings = dict(sources)
    for path in (study / "protocol.json", study / "protocol_registration.json", Path(__file__)):
        filename, digest = str(path.resolve()), guardian.legacy.sha(path)
        if filename in bindings and bindings[filename] != digest:
            raise ValueError(f"Root source binding differs at registration: {filename}")
        bindings[filename] = digest
    stage = guardian.RegisteredStage(
        "frozen_base_pipeline",
        (python, str(runner.resolve()), "run", "--output", str(study)),
        inputs=tuple(sorted(bindings)), outputs=(str(study / "analysis/results.json"),),
        validator="complete_json", gpu_required=True)
    # Validate before publishing the plan so invalid card choices leave no
    # partially registered study behind.
    validate_registration(study, [stage], gpu, reserved_gpu)
    for filename, digest in bindings.items():
        if guardian.legacy.sha(filename) != digest:
            raise ValueError(f"Frozen source binding differs at registration: {filename}")
    plan = dict(schema=SCHEMA, study=str(study), cwd=str(ROOT.resolve()),
        root_protocol_sha256=protocol_sha,
        stages=[stage.registration()], bindings=bindings,
        extra_input_files=sorted(bindings), input_identities=input_identities,
        large_input_verification="Scientific runner checks registered model/data before use",
        root_evaluation_scope=protocol.get("evaluation_scope", protocol.get("scope")),
        resource_policy=dict(physical_gpu=gpu, reserved_gpu=reserved_gpu, max_cuda_workers=1,
            automatic_retry=False, busy_before_launch="CPU-only wait for an empty card"),
        stop_directories=[str(study), str(attempt), str(attempt / "execution")],
        ready_command=[python, str(Path(__file__).resolve()), "run", "--study", str(study)])
    guardian.legacy.dump(plan_path, plan)
    bound_plan = dict(bindings, **{str(plan_path): guardian.legacy.sha(plan_path)})
    # The inherited resource init/run helpers hard-code an older study's
    # unopened-holdout metadata. Register this scope independently while reusing
    # its audited execution class unchanged for ownership, admission and STOP.
    for module in (guardian, guardian.legacy, guardian.inventory_module):
        path = Path(module.__file__).resolve()
        bound_plan[str(path)] = guardian.legacy.sha(path)
    resource = dict(schema=RESOURCE_SCHEMA, study=str(study), cwd=str(ROOT.resolve()),
        stages=plan["stages"], bindings=bound_plan,
        resource_policy=plan["resource_policy"], automatic_retries=False,
        evaluation_scope="Fresh frozen-base train/validation followed by selection-locked test",
        formal_test_enabled=True, formal_validation_enabled=True,
        scientific_scope_state_path=str(study / "state.json"), created_unix=time.time())
    attempt.mkdir()
    guardian.legacy.dump(attempt / "resource_protocol.json", resource)
    guardian.legacy.dump(attempt / "resource_registration.json",
        dict(protocol_sha256=guardian.legacy.sha(attempt / "resource_protocol.json")))
    return plan


def check_plan(study):
    study = Path(study).resolve()
    plan = guardian.legacy.read_json(study / "resource_plan.json")
    if plan.get("schema") != SCHEMA or plan.get("study") != str(study):
        raise ValueError("Fresh-study resource plan belongs to a different study")
    _, digest, inputs, sources = _root_identity(study)
    if plan.get("root_protocol_sha256") != digest:
        raise ValueError("Root protocol differs from the resource plan")
    if {name: identity["sha256"] for name, identity in plan["input_identities"].items()} != inputs:
        raise ValueError("Registered model/data identities differ from the root protocol")
    for filename, identity in plan["input_identities"].items():
        stat = Path(filename).stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"Registered model/data file metadata changed: {filename}")
    attempt = study / ATTEMPT
    registered = guardian.legacy.read_json(attempt / "resource_protocol.json")
    registration = guardian.legacy.read_json(attempt / "resource_registration.json")
    if (registered.get("schema") != RESOURCE_SCHEMA or
            registration.get("protocol_sha256") != guardian.legacy.sha(attempt / "resource_protocol.json")):
        raise ValueError("Resource protocol differs from its independent SHA registration")
    stages = [guardian.RegisteredStage.from_registration(value) for value in registered["stages"]]
    policy = registered["resource_policy"]
    validate_registration(study, stages, policy["physical_gpu"], policy["reserved_gpu"])
    if (policy != plan["resource_policy"] or policy["max_cuda_workers"] != 1 or
            policy["automatic_retry"] is not False or registered.get("automatic_retries") is not False):
        raise ValueError("Registered resource policy differs from the single-card plan")
    for filename, file_sha in registered["bindings"].items():
        if guardian.legacy.sha(filename) != file_sha:
            raise ValueError(f"Frozen queue binding changed: {filename}")
    if (registered["study"] != str(study) or registered["stages"] != plan["stages"] or
            registered["cwd"] != plan["cwd"] or len(plan["stages"]) != 1):
        raise ValueError("Guardian registration differs from the fresh-study plan")
    expected_command = [str((ROOT / ".venv-hanguard/bin/python").absolute()),
        str((ROOT / "scripts/hanguard/frozen_base_readout_study.py").resolve()),
        "run", "--output", str(study)]
    stage = plan["stages"][0]
    if (stage["command"] != expected_command or stage["validator"] != "complete_json" or
            stage["outputs"] != [str(study / "analysis/results.json")] or
            stage["gpu_required"] is not True):
        raise ValueError("Fresh-study pipeline command or completion contract changed")
    for filename, source_sha in sources.items():
        if registered["bindings"].get(filename) != source_sha:
            raise ValueError(f"Guardian does not bind the root source: {filename}")
    for filename, file_sha in plan["bindings"].items():
        if registered["bindings"].get(filename) != file_sha:
            raise ValueError(f"Guardian binding differs from the fresh-study plan: {filename}")
    return plan


def run(study):
    study = Path(study).resolve()
    check_plan(study)
    attempt = study / ATTEMPT
    with (attempt / "resource_supervisor.pid").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another guardian owns this resource attempt") from exc
        if (attempt / "resource_state.json").exists():
            raise RuntimeError("Resource attempt was already executed; no automatic retry")
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        # Recheck under the attempt lock immediately before publication/launch.
        check_plan(study)
        registered = guardian.legacy.read_json(attempt / "resource_protocol.json")
        bindings = dict(registered["bindings"])
        for path in (attempt / "resource_protocol.json", attempt / "resource_registration.json"):
            bindings[str(path)] = guardian.legacy.sha(path)
        policy = registered["resource_policy"]
        guardian.legacy.dump(attempt / "resource_state.json", dict(schema=RESOURCE_SCHEMA,
            state="running", supervisor_pid=os.getpid(), physical_gpu=policy["physical_gpu"],
            reserved_gpu=policy["reserved_gpu"], automatic_retry=False, **scientific_scope(study)))
        try:
            supervisor = ScopedSupervisor(study,
                [guardian.RegisteredStage.from_registration(value) for value in registered["stages"]],
                bindings, queue_state_dir=attempt / "execution", cwd=registered["cwd"],
                gpu=policy["physical_gpu"], reserved_gpu=policy["reserved_gpu"])
            result = supervisor.run()
            guardian.legacy.dump(attempt / "resource_state.json", dict(schema=RESOURCE_SCHEMA,
                state="completed", stages_completed=len(registered["stages"]),
                automatic_retry=False, physical_gpu=policy["physical_gpu"],
                reserved_gpu=policy["reserved_gpu"], **scientific_scope(study)))
            return result
        except BaseException as exc:
            state = "stopped" if isinstance(exc, guardian.legacy.StopRequested) else "failed"
            guardian.legacy.dump(attempt / "resource_state.json", dict(schema=RESOURCE_SCHEMA,
                state=state, error_type=type(exc).__name__, reason=str(exc), automatic_retry=False,
                scientific_conclusion=None, foreign_processes_signalled=False, **scientific_scope(study)))
            raise


def _check_stop(study):
    stop = study / "STOP"
    if stop.exists():
        raise guardian.legacy.StopRequested(stop.read_text().strip() or "STOP requested")


def _input_metadata(inputs):
    result = {}
    for filename, digest in inputs.items():
        path = Path(filename)
        stat = path.stat()
        if not path.is_file():
            raise ValueError(f"Root input is not a regular file: {filename}")
        result[filename] = dict(sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def _check_wait_bindings(study, bindings, input_identities):
    _check_stop(study)
    if (study / ATTEMPT).exists() or (study / "resource_plan.json").exists():
        raise RuntimeError("A resource attempt/plan already exists; no automatic retry")
    for filename, digest in bindings.items():
        if guardian.legacy.sha(filename, study) != digest:
            raise ValueError(f"Frozen selector binding changed: {filename}")
    for filename, identity in input_identities.items():
        stat = Path(filename).stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"Registered model/data file metadata changed while waiting: {filename}")
    _check_stop(study)


def _candidate_observations(inventory, identity_reader):
    rows = inventory()
    result = []
    for index in CANDIDATES:
        gpu = next((row for row in rows if row["index"] == index), None)
        if gpu is None:
            result.append(dict(index=index, available=False, eligible=False, pids=[],
                process_owners=[], reason="Candidate card unavailable"))
            continue
        owners = []
        for pid in gpu["pids"]:
            try:
                identity = identity_reader(pid)
                try:
                    owner = pwd.getpwuid(identity["uid"]).pw_name
                except KeyError:
                    owner = f'uid:{identity["uid"]}'
                owners.append(dict(pid=pid, uid=identity["uid"], owner=owner,
                    start_time_ticks=identity["start_time_ticks"],
                    command_sha256=identity["command_sha256"], classification="foreign",
                    reason="Every existing process blocks selector admission"))
            except (OSError, ValueError, KeyError) as exc:
                owners.append(dict(pid=pid, owner="unknown", classification="unknown",
                    error_type=type(exc).__name__, reason="Unverified ownership blocks admission"))
        result.append(dict(**gpu, available=True, process_owners=owners,
            eligible=not gpu["pids"] and gpu["free_gib"] >= 4.0,
            completely_idle=not gpu["pids"]))
    return result


def _sleep_wait(study, seconds):
    # Inventory/status refresh every five seconds; STOP is checked within .2s
    # during the interval, and again before immutable admission.
    deadline = time.monotonic() + seconds
    while True:
        _check_stop(study)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(.2, remaining))


def wait_and_run(study, *, inventory=guardian.inventory_module.gpu_inventory,
                 identity_reader=guardian.inventory_module.process_identity, poll_seconds=5.0):
    """CPU-select one empty public card, register once, then delegate run."""
    if not 0 < poll_seconds <= 5:
        raise ValueError("Selector polling must be positive and at most five seconds")
    study = Path(study).resolve()
    if (study / ATTEMPT).exists() or (study / "resource_plan.json").exists():
        raise RuntimeError("A resource attempt/plan already exists; no automatic retry")
    with (study / "resource_waiter.pid").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another CPU selector owns this study") from exc
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        protocol, protocol_sha, inputs, sources = _root_identity(study)
        bindings = dict(sources)
        input_identities = _input_metadata(inputs)
        began, candidates = time.time(), []
        base = dict(schema=SELECTOR_SCHEMA, study=str(study), supervisor_pid=os.getpid(),
            root_protocol_sha256=protocol_sha, started_unix=began,
            candidate_gpus=list(CANDIDATES), max_cuda_workers=1, selector_cuda_loaded=False,
            selector_started_GPU_child=False, automatic_retry=False,
            foreign_processes_signalled=False, input_identities=input_identities,
            root_evaluation_scope=protocol.get("evaluation_scope", protocol.get("scope")))
        def publish(state, **details):
            value = dict(base, state=state, updated_unix=time.time(),
                candidate_observations=candidates, **scientific_scope(study), **details)
            guardian.legacy.dump(study / "waiting_state.json", value)
            print(json.dumps(value, ensure_ascii=False), flush=True)
        try:
            for path in (study / "protocol.json", study / "protocol_registration.json", Path(__file__),
                         Path(guardian.__file__), Path(guardian.legacy.__file__),
                         Path(guardian.inventory_module.__file__)):
                filename, digest = str(path.resolve()), guardian.legacy.sha(path, study)
                if filename in bindings and bindings[filename] != digest:
                    raise ValueError(f"Frozen selector source binding changed: {filename}")
                bindings[filename] = digest
            while True:
                _check_wait_bindings(study, bindings, input_identities)
                candidates = _candidate_observations(inventory, identity_reader)
                # Close the inventory-to-registration interval with another
                # STOP and frozen-source/input-metadata check. The inherited
                # guardian will independently recheck ownership before launch.
                _check_wait_bindings(study, bindings, input_identities)
                chosen = next((row["index"] for row in candidates if row["eligible"]), None)
                if chosen is None:
                    publish("CPU_wait_for_empty_candidate", registered_cuda_workers=0)
                    _sleep_wait(study, poll_seconds)
                    continue
                reserved = 2 if chosen == 4 else 4
                publish("empty_candidate_selected", physical_gpu=chosen, reserved_gpu=reserved,
                    registered_cuda_workers=0)
                initialize(study, gpu=chosen, reserved_gpu=reserved)
                publish("registered_single_guardian", physical_gpu=chosen, reserved_gpu=reserved,
                    registered_cuda_workers=1,
                    GPU_child_state_owner=str(study / ATTEMPT / "execution/queue_state.json"))
                result = run(study)
                publish("completed", physical_gpu=chosen, reserved_gpu=reserved,
                    registered_cuda_workers=1)
                return result
        except BaseException as exc:
            state = "stopped" if isinstance(exc, guardian.legacy.StopRequested) else (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            publish(state, error_type=type(exc).__name__, reason=str(exc), scientific_conclusion=None)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "wait-and-run", "status"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), default=2)
    parser.add_argument("--reserved-gpu", type=int, choices=range(8), default=4)
    args = parser.parse_args(argv)
    if args.action == "init":
        result = initialize(args.study, gpu=args.gpu, reserved_gpu=args.reserved_gpu)
    elif args.action == "run":
        result = run(args.study)
    elif args.action == "wait-and-run":
        result = wait_and_run(args.study)
    else:
        if (args.study / "resource_plan.json").exists():
            plan = check_plan(args.study)
            state = args.study / ATTEMPT / "execution/queue_state.json"
            result = guardian.legacy.read_json(state) if state.exists() else plan
        else:
            _, protocol_sha, _, _ = _root_identity(args.study)
            result = guardian.legacy.read_json(args.study / "waiting_state.json")
            if result.get("root_protocol_sha256") != protocol_sha:
                raise ValueError("Waiting status differs from the root protocol")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
