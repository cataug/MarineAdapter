import os
import time
import json
import shlex
import subprocess
from pathlib import Path
from datetime import datetime


ROOT = Path("/home/tahiti/MARINE_DATASETS")
SCRIPT = ROOT / "run_marine_adapter_experiments.py"

OUT_ROOT = ROOT / "MARINE_RESULTS_DYNAMIC"
LOG_DIR = OUT_ROOT / "logs"
STATUS_DIR = OUT_ROOT / "status"

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = "/home/tahiti/Malashin_Projects/.venv_a100/bin/python"

GPU_ID = 0
MAX_RUNNING = 4

CHECK_EVERY_SEC = 20
MIN_FREE_MEM_MB = 3000
MAX_GPU_UTIL = 100

EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = 224
MAX_CLASSES = 30
MAX_IMAGES_PER_CLASS = 300

METHODS = [
    "scratch_cnn",
    "linear_probe",
    "marine_adapter",
    "finetune_resnet18",
]

BUDGETS = [
    0.01,
    0.05,
    0.10,
    1.0,
]

SEEDS = [
    42,
    43,
    44,
]

ADAPTER_DIMS = [
    64,
    128,
    256,
]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd):
    p = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def get_gpu_state(gpu_id=0):
    cmd = (
        "nvidia-smi "
        f"--query-gpu=index,memory.used,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    code, out, err = run_cmd(cmd)

    if code != 0:
        return None

    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue

        idx = int(parts[0])
        if idx != gpu_id:
            continue

        return {
            "index": idx,
            "mem_used": int(parts[1]),
            "mem_free": int(parts[2]),
            "gpu_util": int(parts[3]),
        }

    return None


def gpu_is_free():
    st = get_gpu_state(GPU_ID)

    if st is None:
        print(f"[{now()}] WARN: cannot read nvidia-smi")
        return False

    free = st["mem_free"] >= MIN_FREE_MEM_MB

    print(
        f"[{now()}] GPU{GPU_ID}: "
        f"free={st['mem_free']}MB used={st['mem_used']}MB util={st['gpu_util']}% "
        f"=> {'CAN_LAUNCH' if free else 'NO_MEMORY'}"
    )

    return free


def make_jobs():
    jobs = []

    for seed in SEEDS:
        for budget in BUDGETS:
            for method in METHODS:
                if method == "marine_adapter":
                    for ad in ADAPTER_DIMS:
                        jobs.append({
                            "method": method,
                            "budget": budget,
                            "seed": seed,
                            "adapter_dim": ad,
                        })
                else:
                    jobs.append({
                        "method": method,
                        "budget": budget,
                        "seed": seed,
                        "adapter_dim": 128,
                    })

    return jobs


def job_name(job):
    return (
        f"{job['method']}"
        f"_b{int(round(job['budget'] * 100)):03d}"
        f"_s{job['seed']}"
        f"_ad{job['adapter_dim']}"
    )


def is_done(job):
    name = job_name(job)
    done_file = STATUS_DIR / f"{name}.done"
    fail_file = STATUS_DIR / f"{name}.fail"

    if done_file.exists():
        return True

    # also check inner result
    expected_run_dir = OUT_ROOT / "runs" / (
        f"{job['method']}"
        f"_budget{int(round(job['budget'] * 100)):03d}"
        f"_seed{job['seed']}"
        f"_ad{job['adapter_dim']}"
    )
    if (expected_run_dir / "result.json").exists():
        done_file.write_text("done\n")
        return True

    return False


def build_command(job):
    method = job["method"]
    budget = job["budget"]
    seed = job["seed"]
    adapter_dim = job["adapter_dim"]

    cmd = [
        PYTHON,
        str(SCRIPT),
        "--out_dir", str(OUT_ROOT),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--img_size", str(IMG_SIZE),
        "--num_workers", "0",
        "--max_classes", str(MAX_CLASSES),
        "--max_images_per_class", str(MAX_IMAGES_PER_CLASS),
        "--methods", method,
        "--budgets", str(budget),
        "--seeds", str(seed),
        "--adapter_dim", str(adapter_dim),
    ]

    return " ".join(shlex.quote(x) for x in cmd)


def launch_job(job):
    name = job_name(job)
    log_path = LOG_DIR / f"{name}.log"
    done_path = STATUS_DIR / f"{name}.done"
    fail_path = STATUS_DIR / f"{name}.fail"

    cmd = build_command(job)

    wrapped = f"""
set -e
source /home/tahiti/Malashin_Projects/.venv_a100/bin/activate
cd /home/tahiti/MARINE_DATASETS

export CUDA_VISIBLE_DEVICES={GPU_ID}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[START] {now()}"
echo "JOB: {name}"
echo "CMD: {cmd}"
echo

{cmd}

echo
echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S')"
"""

    sh_path = STATUS_DIR / f"{name}.sh"
    sh_path.write_text(wrapped)

    print(f"[{now()}] LAUNCH: {name}")
    print(f"[{now()}] LOG: {log_path}")

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            ["bash", str(sh_path)],
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    return {
        "name": name,
        "job": job,
        "proc": proc,
        "log_path": str(log_path),
        "done_path": str(done_path),
        "fail_path": str(fail_path),
        "start_time": time.time(),
    }


def main():
    print("=" * 100)
    print("MARINE DYNAMIC GPU LAUNCHER")
    print("=" * 100)
    print("ROOT:", ROOT)
    print("SCRIPT:", SCRIPT)
    print("OUT_ROOT:", OUT_ROOT)
    print("GPU_ID:", GPU_ID)
    print("MAX_RUNNING:", MAX_RUNNING)

    if not SCRIPT.exists():
        raise FileNotFoundError(SCRIPT)

    code, out, err = run_cmd(
        f"{PYTHON} - <<'PY'\n"
        "import torch\n"
        "print(torch.__version__)\n"
        "print(torch.cuda.is_available())\n"
        "print(torch.version.cuda)\n"
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')\n"
        "PY"
    )
    print("\nTorch check:")
    print(out)
    if err:
        print(err)

    jobs = make_jobs()
    print("\nTotal jobs:", len(jobs))

    remaining = [j for j in jobs if not is_done(j)]
    print("Remaining jobs:", len(remaining))

    running = []

    while remaining or running:
        # check finished
        still_running = []

        for r in running:
            ret = r["proc"].poll()

            if ret is None:
                still_running.append(r)
                continue

            name = r["name"]

            if ret == 0:
                Path(r["done_path"]).write_text("done\n")
                print(f"[{now()}] FINISHED OK: {name}")
            else:
                Path(r["fail_path"]).write_text(f"failed ret={ret}\nlog={r['log_path']}\n")
                print(f"[{now()}] FAILED: {name} ret={ret} log={r['log_path']}")

        running = still_running

        # launch new jobs if possible
        while len(running) < MAX_RUNNING and remaining:
            if not gpu_is_free():
                break

            job = remaining.pop(0)

            if is_done(job):
                continue

            r = launch_job(job)
            running.append(r)

            time.sleep(5)

        print(
            f"[{now()}] STATUS: "
            f"running={len(running)} remaining={len(remaining)}"
        )

        time.sleep(CHECK_EVERY_SEC)

    print("\nALL JOBS FINISHED")


if __name__ == "__main__":
    main()