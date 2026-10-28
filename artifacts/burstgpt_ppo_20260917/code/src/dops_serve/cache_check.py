"""Compare exact replay against the unchanged cost implementation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time


def paired_replay(root, reference, output, model, load, trace, repeat, quick=False):
    import torch
    from .burstgpt_ppo import make_evaluator
    from .burstgpt_preflight import request
    from .io import read, write
    from .placement_policy import StaticPlacementActor
    from .placement_training import static_tensors
    from .replay import BurstPlacementReplay
    from .workloads import load_full_trace

    root, reference, output = Path(root), Path(reference), Path(output)
    torch.set_num_threads(1)
    torch.manual_seed(4)
    protocol = read(root / "protocol.json")
    evaluators = [make_evaluator(root, model), make_evaluator(root, model)]
    # 对照组只使用旧成本类；其余源码已逐文件核对，网络和模拟器状态各自独立。
    spec = importlib.util.spec_from_file_location("dops_serve.reference_cost",
        reference / "code/src/dops_serve/placement_cost.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    evaluators[0].initial.scheduler.cost.__class__ = old.RaggedCostModel
    if quick:
        rows = [request(0, 64, 9), request(1, 128, 7, .01), request(2, 29665, 2, .03)]
    else:
        calibration = read(root / "calibration/pilot" / f"{model}.json")
        rows = load_full_trace(root / "data", "val", trace,
            calibration["original_arrival_output_tokens_per_s"] /
            (load * calibration["reference_output_tokens_per_s"]))
        assert len(rows) == 1000
    envs = [BurstPlacementReplay(rows, evaluator, protocol["decode_features"][model])
            for evaluator in evaluators]
    initial = torch.load(root / "initial" / model / "e0.pt", map_location="cpu", weights_only=False)
    actor = StaticPlacementActor(**initial["dimensions"])
    actor.load_state_dict(initial["actor"])
    actor = torch.jit.script(actor.eval())
    state, nodes, masks, fixed, _ = static_tensors(envs[0].inputs())
    with torch.no_grad():
        for _ in range(3):
            actor.generate(state, nodes, masks, fixed)
    elapsed = [0., 0.]
    name = f"{model}_load{load:g}_trace{trace}_repeat{repeat}"
    started = last_report = time.perf_counter()
    order = (0, 1) if repeat == 0 else (1, 0)
    while not envs[0].done:
        rng = torch.get_rng_state()
        observed = [None, None]
        for i in order:
            torch.set_rng_state(rng)
            env = envs[i]
            begin = time.perf_counter()
            inputs = env.inputs()
            state, nodes, masks, fixed, _ = static_tensors(inputs)
            with torch.no_grad():
                actions = actor.sample(state, nodes, masks, fixed, not quick)[0].tolist()
            plan = env.evaluator.compiler.compile(actions)
            env.decision_s += time.perf_counter() - begin
            dt = env.step(plan)
            elapsed[i] += time.perf_counter() - begin
            observed[i] = (inputs, actions, dt, env.done, env.output_tokens,
                           env.snapshot.now, env.snapshot.kv_written_tokens.copy(),
                           env.snapshot.kv_reserved_tokens.copy(),
                           env.snapshot.scheduler.avail.copy(),
                           env.snapshot.scheduler.comm.timeline_end.copy())
        assert observed[0] == observed[1], (name, envs[0].plan_count, "decision/state/duration mismatch")
        now = time.perf_counter()
        if now-last_report >= 60:
            write(output / "progress" / f"{name}.json", dict(model=model, load=load,
                trace=trace, repeat=repeat, completed=len(envs[0].completed),
                plans=envs[0].plan_count, wall_s=now-started))
            last_report = now
    metrics = [env.metrics() for env in envs]
    exact = [{k: v for k, v in value.items() if k not in ("decision_s", "simulator_s")}
             for value in metrics]
    assert exact[0] == exact[1], (name, "final metrics mismatch")
    for before, after in zip(envs[0].requests, envs[1].requests):
        assert before == after, (name, before.request_id, "request lifecycle mismatch")
    for device, before in envs[0].snapshot.scheduler.buffer.device_cache.items():
        after = envs[1].snapshot.scheduler.buffer.device_cache[device]
        for attr in ("items", "order", "used", "pinned", "meta"):
            assert getattr(before, attr) == getattr(after, attr), (name, device, attr)
    assert metrics[0]["completed"] == len(rows)
    assert metrics[0]["output_tokens"] == sum(row["output_len"] for row in rows)
    result = dict(model=model, load=load, trace=trace, repeat=repeat, quick=quick,
        exact_parity=True, original_wall_s=elapsed[0], cached_wall_s=elapsed[1],
        speedup=elapsed[0]/elapsed[1], pair_wall_s=time.perf_counter()-started,
        original=metrics[0], cached=metrics[1])
    write(output / "results" / f"{name}.json", result)
    return result


def main():
    from .io import read, write
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    models = read(args.root / "protocol.json")["models"]
    tasks = [(m, load, trace, repeat) for trace in ([0] if args.quick else [184, 1])
             for repeat in ([0] if args.quick else [0, 1])
             for m in models for load in ([1.] if args.quick else [.5, 1., 1.5])]
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = []
        for m, load, trace, repeat in tasks:
            path = args.output / "results" / f"{m}_load{load:g}_trace{trace}_repeat{repeat}.json"
            if path.exists():
                results.append(read(path))
            else:
                pending.append(pool.submit(paired_replay, args.root, args.reference,
                    args.output, m, load, trace, repeat, args.quick))
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("original", "cached")}), flush=True)
    rounds = []
    for repeat in sorted({r["repeat"] for r in results}):
        selected = [r for r in results if r["repeat"] == repeat]
        before = sum(r["original_wall_s"] for r in selected)
        after = sum(r["cached_wall_s"] for r in selected)
        rounds.append(dict(repeat=repeat, original_wall_s=before, cached_wall_s=after,
                           speedup=before/after))
    summary = dict(cases=len(results), exact_parity=all(r["exact_parity"] for r in results),
                   rounds=rounds, quick=args.quick)
    summary["passed"] = summary["exact_parity"] and (args.quick or all(r["speedup"] > 1.05 for r in rounds))
    write(args.output / "summary.json", summary)
    if not args.quick and summary["passed"]:
        write(args.root / "exact_cache_check_passed.json", summary)
    print(json.dumps(summary), flush=True)
    assert summary["passed"], "Cache parity or measured speed gate failed"


if __name__ == "__main__":
    main()
