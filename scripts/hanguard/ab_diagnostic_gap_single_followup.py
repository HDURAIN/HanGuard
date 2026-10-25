"""User-authorized immediate admission of one empty GPU, leaving GPU2 unused."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import ab_diagnostic_followup as follow
from scripts.hanguard import ab_diagnostic_queue as queue


class ImmediateExclusiveAdmission:
    def apply(self, decision, now):
        allowed = decision['completely_idle'] or (bool(decision['owned_pids']) and not decision['blocking_processes'])
        return dict(decision, eligible=decision['eligible'] and allowed,
                    idle_confirmation_required_seconds=0,
                    admission_authorized_by='User: immediately use one card during an empty interval; leave the other card',
                    idle_confirmed=allowed)


class SingleGPU(follow.StrictFollowup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idle = ImmediateExclusiveAdmission()

    def _guard_gpu_idle(self):
        if self.active_stage not in self.gpu_stages:
            return
        last_status = 0
        while True:
            queue.SequentialSupervisor._check(self)
            observation = self.observation()
            if observation['eligible']:
                self._state('user_authorized_empty_GPU4_admitted', resource_observation=observation)
                return
            if time.monotonic() - last_status >= 5:
                self._state('waiting_for_empty_GPU4', resource_observation=observation,
                            GPU2_reserved_for_other_user=True, GPU_child_started=False)
                last_status = time.monotonic()
            time.sleep(.5)


def run(output):
    output = Path(output).resolve()
    protocol = follow.check_registration(output)
    policy_path = output / 'user_resource_policy.json'
    policy = queue.read_json(policy_path)
    registration = queue.read_json(output / 'user_resource_registration.json')
    if (policy['GPU'] != 4 or policy['unused_GPU'] != 2 or policy['admission_seconds'] != 0 or
            registration['policy_sha256'] != queue.sha(policy_path) or
            policy['source_sha256'] != queue.sha(__file__)):
        raise ValueError('User-directed single-GPU resource registration changed')
    state_path = output / 'resource_state.json'
    if state_path.exists():
        raise RuntimeError('Single-GPU attempt already executed; no automatic retry')
    stages = follow.followup_stages(protocol['study'], protocol['python'])
    if [stage.registration() for stage in stages] != protocol['stages']:
        raise ValueError('Scientific stage commands changed')
    bindings = dict(protocol['bindings'])
    for path in (Path(__file__), policy_path, output / 'user_resource_registration.json',
                 output / 'continuation_protocol.json'):
        bindings[str(path)] = queue.sha(path)
    a = follow.verify_A_completion(protocol['study'], protocol['original_A_process_identity'])
    b = follow.verify_base_completion(protocol['study'], protocol['base_attempt'])
    if not a['ready'] or not b['ready']:
        raise RuntimeError('Original A/base prerequisites are no longer complete')
    env = dict(CUDA_VISIBLE_DEVICES='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    queue.dump(state_path, dict(state='running_registered_stages', GPU=4, unused_GPU=2,
                              automatic_retry=False, scientific_settings_changed=False))
    try:
        supervisor = SingleGPU(protocol['study'], stages, bindings,
                               queue_state_dir=output / 'execution', environment=env)
        supervisor.run()
        queue.summarize_results(protocol['study'], output)
        queue.dump(state_path, dict(state='completed', scientific_settings_changed=False,
                                  summary_path=str(output / 'diagnostic_summary.json')))
    except BaseException as exc:
        queue.dump(state_path, dict(state='stopped' if isinstance(exc, queue.StopRequested) else 'failed',
                                  reason=str(exc), automatic_retry=False, scientific_conclusion=None))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
