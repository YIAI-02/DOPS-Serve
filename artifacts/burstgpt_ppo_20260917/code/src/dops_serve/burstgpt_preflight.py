"""Compute-node integration checks for the agreed BurstGPT PPO contract."""
import json
import math
import time

import torch

from .burstgpt_ppo import EPOCHS, MODELS, make_evaluator
from .placement_policy import StaticPlacementActor
from .placement_training import static_tensors
from .replay import BurstPlacementReplay
from .io import read, write


def request(i, prompt, output, arrival=0.):
    return dict(request_id=f"check_{i}", arrival_time=arrival, prompt_len=prompt,
                max_tokens=7315, output_len=output)


def constant_plan(compiler, choice):
    actions = []
    for i, parent in enumerate(compiler.fixed_from):
        actions.append(0 if parent == -2 else actions[parent] if parent >= 0
                       else choice if compiler.masks[i][choice] else compiler.masks[i].index(True))
    return compiler.compile(actions)


def preflight(root):
    protocol = read(root / "protocol.json")
    started = time.perf_counter()
    checks = []
    for model in MODELS:
        evaluator = make_evaluator(root, model)
        decode_feature = protocol["decode_features"][model]
        rows = [request(0, 128, 1), request(1, 256, 4), request(2, 19, 2, .3), request(3, 100, 1, 1.)]
        env = BurstPlacementReplay(rows, evaluator, decode_feature)
        changed_future = BurstPlacementReplay([{**r, "output_len": r["output_len"]+2} for r in rows], evaluator, decode_feature)
        assert env.inputs() == changed_future.inputs(), "future output leaked into policy inputs"
        inputs = env.inputs()
        assert len(inputs["state"]) == 56 and all(len(n) == 35 for n in inputs["nodes"])
        assert inputs["state"][2] == decode_feature
        assert math.isclose(inputs["nodes"][0][18], math.log1p(128+256)/10)
        for choice in range(3):
            plan = constant_plan(evaluator.compiler, choice)
            before = env.snapshot.clone()
            regular = evaluator.evaluate(before, plan, "prefill", env.shapes())
            inplace = evaluator.evaluate(before.clone(), plan, "prefill", env.shapes(), inplace=True)
            assert regular.duration_s == inplace.duration_s
            assert [(t.node_id, t.device, t.start, t.finish) for t in regular.schedule] == [
                (t.node_id, t.device, t.start, t.finish) for t in inplace.schedule]
            assert regular.snapshot.kv_written_tokens == inplace.snapshot.kv_written_tokens
        total = 0.
        phases = []
        while not env.done:
            phases.append(env.pending[0])
            plan = evaluator.pd_plan(env.snapshot, env.pending[0], env.shapes())
            total += env.step(plan)
        assert env.output_tokens == 8 and len(env.completed) == 4
        assert not env.snapshot.kv_reserved_tokens and not env.snapshot.kv_written_tokens
        assert math.isclose(total, env.snapshot.now-env.start_time, rel_tol=1e-12)
        assert "prefill" in phases and "decode" in phases
        for r in env.completed:
            assert len(r.token_times) == r.output_len
        checks.append(dict(model=model, test="lifecycle_time_three_devices_no_future", **env.metrics()))

        # Full-length capacity pressure, on every physical-device choice.
        for choice in range(3):
            extreme = BurstPlacementReplay([request(i, 29665, 2) for i in range(16)], evaluator, decode_feature)
            admitted = len(extreme.pending[1])
            capacity = (evaluator.initial.scheduler.buffer.pim_state["PIM0"].limit_bytes
                        - protocol["kv_headroom_bytes"])
            expected = min(16, capacity // (36980*evaluator.kv_bytes_per_token_per_pim))
            assert admitted == expected
            plan = constant_plan(evaluator.compiler, choice)
            while not extreme.done:
                extreme.step(plan)
            assert extreme.output_tokens == 32
            assert all(v <= evaluator.initial.scheduler.buffer.pim_state[p].limit_bytes
                       for p, v in extreme.peak_runtime_bytes.items())
            checks.append(dict(model=model, test="max_prompt_capacity", choice=choice,
                               first_admitted=admitted, **extreme.metrics()))

        last = BurstPlacementReplay([request(0, 29665, 7315)], evaluator, decode_feature)
        last._execute(constant_plan(evaluator.compiler, 0))
        r = last.active[0]
        r.token_times = [last.snapshot.now]*7314
        last.snapshot.kv_written_tokens[r.request_id] = r.prompt_len+7313
        last.pending = ("decode", [r])
        assert math.isclose(last.inputs()["state"][3], math.log1p(7313)/10)
        last.step(constant_plan(evaluator.compiler, 1))
        assert last.done and len(r.token_times) == 7315 and not last.snapshot.kv_reserved_tokens
        checks.append(dict(model=model, test="max_output_final_token_and_release", passed=True))

    # Verify actual selected checkpoint provenance and autonomous E0 on ragged input.
    for route in protocol["models"]:
        epoch = EPOCHS[route]
        checkpoint = torch.load(root / "initial" / route / "e0.pt", map_location="cpu", weights_only=False)
        assert checkpoint["bc_epoch"] == epoch
        actor = StaticPlacementActor(**checkpoint["dimensions"]).eval()
        actor.load_state_dict(checkpoint["actor"])
        assert torch.count_nonzero(actor.state_encoder[0].weight[:, 47:]) == 0
        compiled = torch.jit.script(actor)
        for model in (MODELS if route == "shared" else [route]):
            evaluator = make_evaluator(root, model)
            env = BurstPlacementReplay([request(0, 128, 4), request(1, 512, 2)], evaluator,
                                       protocol["decode_features"][route])
            while not env.done:
                state, nodes, masks, fixed, learned = static_tensors(env.inputs())
                with torch.no_grad():
                    actions = compiled.generate(state, nodes, masks, fixed)
                env.step(evaluator.compiler.compile(actions[0].tolist()))
            checks.append(dict(route=route, bc_epoch=epoch, model=model, test="e0_autonomous", **env.metrics()))
    write(root / "preflight_passed.json", dict(passed=True, wall_s=time.perf_counter()-started,
        kv_headroom_bytes=protocol["kv_headroom_bytes"], checks=checks,
        evidence="analytical fast DOPS execution and PPO unit checks; no tensor inference"))
    print(json.dumps(dict(stage="preflight_passed", checks=len(checks), wall_s=time.perf_counter()-started)), flush=True)


def orchestration_check(root):
    """真实 12 环境的整批排空、保存、待验证恢复；短夹具不代表正式训练质量。"""
    import tempfile
    from pathlib import Path
    from .burstgpt_ppo import train

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ppo_orchestration_", dir=root / "runtime") as temporary:
        test_root = Path(temporary)
        for name in ("backend", "configs", "initial"):
            (test_root / name).symlink_to(root / name, target_is_directory=True)
        protocol = read(root / "protocol.json")
        protocol["train_indices"] = [0, 1, 2, 3]
        protocol["quick_val_indices"] = [0, 1, 2, 3]
        protocol["validation_every_epochs"] = 1
        write(test_root / "protocol.json", protocol)
        write(test_root / "preflight_passed.json", read(root / "preflight_passed.json"))
        for index in range(4):
            rows = [request(0, 128, 2+index), request(1, 256, 3)]
            write(test_root / "data/train" / f"{index:06d}.json", rows)
            write(test_root / "data/val" / f"{index:06d}.json", rows)
        route = "qwen_1.8b"
        write(test_root / "calibration/check" / f"{route}.json",
              dict(original_arrival_output_tokens_per_s=1., reference_output_tokens_per_s=1., reward_scale_s=1.))
        train(test_root, route, 4, 1, 4, "check", False, "cpu")
        base = test_root / "pilot/training" / route / "seed_4"
        saved = torch.load(base / "last.pt", map_location="cpu", weights_only=False)
        assert saved["epoch"] == 1 and saved["completed_sources"] == 4 and saved["updates"] > 0
        assert saved["cursor"] == 0 and len(saved["worker_rng"]) == 12
        expected = {k: v.clone() for k, v in saved["actor"].items()}
        # 从已排空且等待验证的 checkpoint 恢复，不能重复训练或丢失验证。
        pending = torch.load(base / "checkpoints/epoch_001.pt", map_location="cpu", weights_only=False)
        assert pending["pending_validation"] is not None
        torch.save(pending, base / "last.pt")
        train(test_root, route, 4, 1, 4, "check", True, "cpu")
        restored = torch.load(base / "last.pt", map_location="cpu", weights_only=False)
        assert restored["epoch"] == 1 and restored["pending_validation"] is None
        assert restored["updates"] == saved["updates"] and len(restored["history"]) == 2
        for name, value in expected.items():
            torch.testing.assert_close(value, restored["actor"][name], rtol=0, atol=0)
        assert restored["history"] == saved["history"]
        assert not (base / "test").exists()
        # 模拟中断后丢弃尚未提交的整个 cohort，恢复 E0 边界并重做。
        boundary = torch.load(base / "checkpoints/epoch_000.pt", map_location="cpu", weights_only=False)
        torch.save(boundary, base / "last.pt")
        train(test_root, route, 4, 1, 4, "check", True, "cpu")
        repeated = torch.load(base / "last.pt", map_location="cpu", weights_only=False)
        assert repeated["updates"] == saved["updates"]
        for network in ("actor", "critic"):
            for name, value in saved[network].items():
                torch.testing.assert_close(value, repeated[network][name], rtol=0, atol=0)
        for expected_rng, repeated_rng in zip(saved["worker_rng"], repeated["worker_rng"]):
            assert torch.equal(expected_rng, repeated_rng)
        result = dict(passed=True, environments=12, updates=saved["updates"], device="cpu",
            checkpoint_and_pending_validation_resume=True, no_repeated_updates=True, test_locked=True,
            interrupted_cohort_replay_exact=True,
            fixtures="four two-request traces; all test checkpoints discarded", wall_s=time.perf_counter()-started)
    write(root / "orchestration_check_passed.json", result)
    print(json.dumps(dict(stage="orchestration_check_passed", **result)), flush=True)
