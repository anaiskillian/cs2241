#!/usr/bin/env python3
"""
main.py  ─ Real cluster experiments
===================================
Run baseline, workload or hyper-parameter experiments on a *live* PostgreSQL
cluster through the `RealServerCluster` Gym environment.

Differences vs. main_example.py
-------------------------------
1. Uses ``RealServerCluster`` (async, talks to real hosts) instead of the
   in-memory ``ServerCluster`` simulator.
2. Drops the PPO path - this script is *exclusively* about the MAB agents.
3. Derives per-step latency / throughput from the env's own bookkeeping
   because `RealServerCluster.step()` does not return them directly.
4. Adds CLI flags (or env-vars) for DB credentials and host list.

Everything else - argument parsing, experiment orchestration, plotting,
metrics - is kept as close as possible to the original template so you can
copy-paste without touching the rest of your code-base.
"""

# ───────────────────────────────── standard libs ───────────────────────────
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────── project imports ─────────────────────────
from src.environment.cluster import RealServerCluster

# from src.environment.request import RequestType            # one-hot length
from src.agents.round_robin import RoundRobinAgent
from src.agents.random_agent import RandomAgent
from src.agents.least_loaded import LeastLoadedAgent
from src.agents.mab_agent import MultiArmedBanditAgent, BanditStrategy
from src.utils.metrics import PerformanceMetrics
from src.environment.request import make_query_generator


# from src.utils.workload_gen import WorkloadGenerator
from src.utils.visualization import (
    plot_latency_comparison,
    plot_throughput_comparison,
    plot_server_utilization,
    plot_comparative_metrics,
)
from src.config import MAB_CONFIG  # ← optimised defaults
from src.config import DB_TABLES_CONFIG


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              CLI helpers                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAB-based request routing on a real PostgreSQL cluster"
    )

    # ── generic selection ────────────────────────────────────────────────
    parser.add_argument(
        "--experiment",
        type=str,
        default="delayed_baseline",
        choices=["baseline", "workload", "hyperparameter", "delayed_baseline"],
        help="Which experiment pipeline to run",
    )
    parser.add_argument("--episodes", type=int, default=10, help="# episodes")
    parser.add_argument("--steps", type=int, default=1000, help="# steps / ep")
    parser.add_argument(
        "--test_steps", type=int, default=10000, help="# steps for final test episode"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--history",
        type=int,
        default=10000,
        help="History length exposed in the RL state",
    )

    # ── database connection details ──────────────────────────────────────
    # parser.add_argument(
    #     "--hosts",
    #     type=str,
    #     default=HOSTS,
    #     help="Comma-separated list of PostgreSQL hosts "
    #     "(overrides ec2_tests.HOSTS if given)",
    # )
    parser.add_argument("--db_user", type=str, default=os.getenv("DB_USER", "postgres"))
    parser.add_argument(
        "--db_password", type=str, default=os.getenv("DB_PASSWORD", "postgres")
    )
    parser.add_argument("--db_name", type=str, default=os.getenv("DB_NAME", "postgres"))
    parser.add_argument(
        "--db_port", type=int, default=int(os.getenv("DB_PORT", "5432"))
    )

    # ── agent choice / tuning ────────────────────────────────────────────
    parser.add_argument(
        "--mab_strategy",
        type=str,
        default="ucb",
        choices=["epsilon_greedy", "ucb", "thompson"],
        help="Which MAB flavour to *default* to (all run in baseline)",
    )

    # ── workload generator ───────────────────────────────────────────────
    parser.add_argument(
        "--workload_type",
        type=str,
        default="all",
        choices=["uniform", "bursty", "diurnal", "skewed", "all"],
        help="Request arrival pattern for `workload` experiment",
    )
    parser.add_argument(
        "--workload_size",
        type=int,
        default=500,
        help="# synthetic requests in each workload",
    )

    return parser.parse_args()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           Environment factory                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def create_environment(args: argparse.Namespace) -> RealServerCluster:
    # ── 1) host list ─────────────────────────────────────────────────────
    from src.config import HOSTS

    # ── 2) DB credential bundle ──────────────────────────────────────────
    db_cfg = {
        "user": args.db_user,
        "password": args.db_password,
        "database": args.db_name,
        "port": args.db_port,
        # pool size is set internally by RealServerCluster
    }

    query_generator = make_query_generator(DB_TABLES_CONFIG, seed=14)

    # ── 3) instantiate env ───────────────────────────────────────────────
    env = RealServerCluster(
        hosts=HOSTS,
        db_cfg=db_cfg,
        history_length=args.history,
        max_steps=args.steps,
        query_generator=query_generator,
    )
    return env


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                             Agent factory                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def create_agents(env: RealServerCluster, args: argparse.Namespace):
    agents = {
        "Random": RandomAgent(num_servers=env.num_hosts),
        "Round Robin": RoundRobinAgent(num_servers=env.num_hosts),
        # "Least Loaded": LeastLoadedAgent(num_servers=env.num_hosts),
    }

    def _mab(name: str, strat: BanditStrategy, cfg: dict):
        agents[name] = MultiArmedBanditAgent(
            num_servers=env.num_hosts, strategy=strat, **cfg
        )

    # single selection or *all three* for baseline
    if args.mab_strategy == "epsilon_greedy":
        _mab(
            "MAB (ϵ-Greedy)",
            BanditStrategy.EPSILON_GREEDY,
            MAB_CONFIG["epsilon_greedy"],
        )
    elif args.mab_strategy == "ucb":
        _mab("MAB (UCB)", BanditStrategy.UCB, MAB_CONFIG["ucb"])
    elif args.mab_strategy == "thompson":
        _mab("MAB (Thompson)", BanditStrategy.THOMPSON_SAMPLING, MAB_CONFIG["thompson"])
    else:  # pragma: no cover
        raise ValueError(f"Unknown strategy {args.mab_strategy}")

    return agents


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         Episode execution loop                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
from typing import Tuple


def run_episode_delayed_train(
    env: RealServerCluster, agent, max_steps: int = 1000, render: bool = False
) -> Tuple[PerformanceMetrics, List[Tuple[int, int, float]]]:
    obs, _ = env.reset()
    agent.reset()
    metrics = PerformanceMetrics()

    done, step = False, 0
    while not done and step < max_steps:
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        # -- update metrics like before --
        server_utils = [cpu for cpu, _ram in env.server_utils]
        metrics.update(info, server_utils)

        obs, step = next_obs, step + 1

        if render and step % 1000 == 0:
            env.render()

    # at this point, all requests have been drained and rewards known
    training_data = env.get_episode_data()
    return metrics, training_data


def run_episode(
    env: RealServerCluster, agent, max_steps: int = 1000, render: bool = False
) -> PerformanceMetrics:
    obs, _ = env.reset()
    agent.reset()
    metrics = PerformanceMetrics()

    done, step = False, 0
    while not done and step < max_steps:
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, next_obs, done)

        # ── craft a *simulator-like* info dict so our old Metrics helper keeps working

        server_utils = [cpu for cpu, _ram in env.server_utils]  # just CPU %
        metrics.update(info, server_utils)

        obs, step = next_obs, step + 1

        if render and step % 1000 == 0:
            env.render()

    return metrics


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     Experiment orchestration helpers                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def baseline_experiment(env, agents, args):
    print("▶ Running *baseline* MAB comparison …")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(f"results/baseline_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    comparative = {}

    for name, ag in agents.items():
        print(f"  ↳ {name}")
        agent_metrics = PerformanceMetrics()
        for ep in range(args.episodes):
            print(f"    episode {ep+1}/{args.episodes}")
            ep_metrics = run_episode(env, ag, max_steps=args.steps, render=(ep == 0))

            # merge
            agent_metrics.latencies.extend(ep_metrics.latencies)
            agent_metrics.throughputs.extend(ep_metrics.throughputs)
            agent_metrics.server_utilizations.extend(ep_metrics.server_utilizations)

        avg = agent_metrics.get_average_metrics()
        comparative[name] = avg
        all_metrics[name] = agent_metrics

        # per-agent plot
        # plot_server_utilization(agent_metrics.server_utilizations, name)

        print(
            f"    avg-latency {avg['avg_latency']:.4f}  "
            f"avg-throughput {avg['avg_throughput']:.4f}"
        )

    # cross-agent comparison plots
    plot_latency_comparison(
        {n: m.latencies for n, m in all_metrics.items()},
        title="Latency comparison - real cluster",
    )
    plot_throughput_comparison(
        {n: m.throughputs for n, m in all_metrics.items()},
        title="Throughput comparison - real cluster",
    )
    plot_comparative_metrics(comparative)

    # JSON dump
    with (outdir / "results.json").open("w") as fp:
        json.dump(
            {n: m.get_average_metrics() for n, m in all_metrics.items()}, fp, indent=2
        )

    # move *.png
    os.system(f"mv *.png {outdir}/")
    print(f"✔ Baseline complete → {outdir}")


from src.agents.base_agent import BaseAgent
from collections import Counter


def delayed_baseline_experiment(
    env: RealServerCluster, agents: dict[str, BaseAgent], args
):
    print("▶ Running *delayed baseline* MAB experiment …")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(f"results/delayed_baseline_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    results_summary = {}
    env.turn_on_train_mode(args.steps)
    for name, agent in agents.items():
        print(f"  ↳ {name}")

        # ───── Training Phase (skip metrics collection) ─────
        if getattr(agent, "trainable", False):
            for ep in range(args.episodes):
                print(f"    training episode {ep + 1}/{args.episodes}")
                _, training_data = run_episode_delayed_train(
                    env, agent, max_steps=args.steps, render=False
                )
                print(f"    training data: {len(training_data)}")
                print(Counter([x[1] for x in training_data]))
                agent.batch_update(training_data)
        else:
            print("    ⏩ not trainable — skipping training phase")

        # ───── Testing Phase (final eval) ─────
        print("    ➤ running final test episode")
        env.turn_on_test_mode(args.test_steps)
        # env.reset()
        agent.reset()
        print("\n\n----------------------------")
        print("Verify that the environment is reset:")
        print(env.pending_requests)
        print(env.latency_hist)
        print(env.steps_taken)
        print(env.average_latency)
        print("Verification complete.")
        print("----------------------------")

        # ignoring both metrics and training data because data
        # is stored in environment
        start_time = time.perf_counter()
        metrics, training_data = run_episode_delayed_train(
            env, agent, max_steps=args.test_steps, render=False
        )
        end_time = time.perf_counter()
        print(f"    test episode took {end_time - start_time:.4f} s")

        assert env.total_latency == sum(env.latency_hist)
        assert env.completed_cnt == len(env.latency_hist)
        total_latency = sum(env.latency_hist)
        num_requests = len(env.latency_hist)

        assert (
            env.steps_taken == args.test_steps
        ), f"[{name}] Mismatch in steps: expected {args.test_steps}, got {env.steps_taken}"
        assert num_requests > 0, f"[{name}] No completed requests in final episode"
        assert (
            num_requests == args.test_steps
        ), f"[{name}] Mismatch in completed requests: expected {args.test_steps}, got {num_requests}"

        wall_time = end_time - start_time
        avg_latency = total_latency / num_requests
        throughput = num_requests / wall_time

        results_summary[name] = {
            "total_latency": total_latency,
            "avg_latency": avg_latency,
            "throughput": throughput,
            "completed_requests": num_requests,
        }

        print(
            f"    [Test] avg-latency {avg_latency:.4f} s  "
            f"throughput {throughput:.4f} req/s  "
            f"completed {num_requests}"
        )
        env.turn_on_train_mode(args.steps)

    # Save to JSON
    with (outdir / "results.json").open("w") as fp:
        json.dump(results_summary, fp, indent=2)

    print(f"✔ Delayed experiment complete → {outdir}")


def workload_experiment(env, agents, args):
    print("▶ Workload sensitivity experiment …")
    outdir = Path("results/workload")
    outdir.mkdir(parents=True, exist_ok=True)

    wg = WorkloadGenerator(seed=args.seed)
    if args.workload_type == "all":
        workloads = {
            "Uniform": wg.uniform_workload(args.workload_size),
            "Bursty": wg.bursty_workload(args.workload_size),
            "Diurnal": wg.diurnal_workload(args.workload_size),
            "Skewed": wg.skewed_workload(args.workload_size),
        }
    else:
        fn = getattr(wg, f"{args.workload_type}_workload")
        workloads = {args.workload_type.capitalize(): fn(args.workload_size)}

    summary = {}
    for wl_name, wl_reqs in workloads.items():
        print(f"  ↳ {wl_name}")
        wl_metrics = {}
        for agent_name, ag in agents.items():
            print(f"    with {agent_name}")
            env.reset()
            ag.reset()
            env.pending_requests = wl_reqs.copy()  # plug workload
            m = run_episode(env, ag, steps=args.steps, render=False)
            wl_metrics[agent_name] = m

        # plotting
        plot_latency_comparison(
            {n: m.latencies for n, m in wl_metrics.items()},
            title=f"Latency - {wl_name}",
        )
        plot_throughput_comparison(
            {n: m.throughputs for n, m in wl_metrics.items()},
            title=f"Throughput - {wl_name}",
        )
        summary[wl_name] = {n: m.get_average_metrics() for n, m in wl_metrics.items()}

    with (outdir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)
    os.system("mv *.png results/workload/")
    print("✔ Workload experiment complete")


def hyperparameter_experiment(env, args):
    print("▶ Hyper-parameter tuning …")
    outdir = Path("results/hyperparameter")
    outdir.mkdir(parents=True, exist_ok=True)

    # search space (same as example but adjustable via --mab_strategy)
    search = {
        BanditStrategy.EPSILON_GREEDY: {
            "epsilon": [0.05, 0.1, 0.2],
            "alpha": [0.05, 0.2],
        },
        BanditStrategy.UCB: {
            "ucb_c": [0.5, 1.0, 2.0],
            "alpha": [0.05, 0.2],
        },
        BanditStrategy.THOMPSON_SAMPLING: {
            "alpha": [0.05, 0.2],
        },
    }

    strategies = {
        "epsilon_greedy": BanditStrategy.EPSILON_GREEDY,
        "ucb": BanditStrategy.UCB,
        "thompson": BanditStrategy.THOMPSON_SAMPLING,
    }
    strat = strategies[args.mab_strategy]
    candidates = search[strat]

    from itertools import product

    param_names, param_vals = list(candidates.keys()), list(candidates.values())

    best_latency, best_cfg = float("inf"), None
    results = {}

    for combo in product(*param_vals):
        cfg = dict(zip(param_names, combo))
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"  testing {cfg_str}")

        agent = MultiArmedBanditAgent(num_servers=env.num_hosts, strategy=strat, **cfg)
        ag_metrics = PerformanceMetrics()
        for _ in range(args.episodes):
            epm = run_episode(env, agent, max_steps=args.steps, render=False)
            ag_metrics.latencies.extend(epm.latencies)
            ag_metrics.throughputs.extend(epm.throughputs)

        avg_lat = np.mean(ag_metrics.latencies) if ag_metrics.latencies else 0.0
        results[cfg_str] = avg_lat
        if avg_lat < best_latency:
            best_latency, best_cfg = avg_lat, cfg_str

        print(f"    → avg latency {avg_lat:.4f}")

    with (outdir / "results.json").open("w") as fp:
        json.dump(
            {"results": results, "best_cfg": best_cfg, "best_latency": best_latency},
            fp,
            indent=2,
        )

    print(f"✔ Best configuration: {best_cfg}  ({best_latency:.4f} s)")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                   main                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    Path("results").mkdir(exist_ok=True)

    env = create_environment(args)
    agents = create_agents(env, args)

    print(args)

    if args.experiment == "baseline":
        baseline_experiment(env, agents, args)
    elif args.experiment == "delayed_baseline":
        delayed_baseline_experiment(env, agents, args)
    elif args.experiment == "workload":
        workload_experiment(env, agents, args)
    elif args.experiment == "hyperparameter":
        hyperparameter_experiment(env, args)
    else:
        print(f"Unknown experiment type: {args.experiment}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
