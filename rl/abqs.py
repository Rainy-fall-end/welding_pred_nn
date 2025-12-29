# batch_runner.py
import argparse
import time
import json
from typing import List, Optional, Tuple, Any

from abaqus_data.pipeline import run_pipeline


def _normalize_time(data, t):
    """
    data: list of (timestamp, current, voltage)
    t: target max time after normalization
    """
    if not data:
        return data

    # 原始最大时间戳
    t_max = max(x[0] for x in data)
    if t_max == 0:
        raise ValueError("原始时间戳最大值为 0，无法正则化")

    normalized_data = [
        (x[0] / t_max * t, x[1], x[2])
        for x in data
    ]

    return normalized_data


def _parse_voltage_seq(seq_str: str) -> List[float]:
    """
    支持两种输入：
    1) JSON list: "[0.1, 0.2, 0.3]"
    2) 逗号分隔: "0.1,0.2,0.3"
    """
    s = (seq_str or "").strip()
    if not s:
        return []
    # JSON list
    if s.startswith("[") and s.endswith("]"):
        arr = json.loads(s)
        if not isinstance(arr, list):
            raise ValueError("voltage_seq JSON 必须是 list")
        return [float(x) for x in arr]
    # comma-separated
    return [float(x) for x in s.split(",") if x.strip()]


def _load_voltage_seq_from_file(path: str) -> List[float]:
    """
    读取电压序列文件：
    - .json: 期望为 list，如 [0.1,0.2,...]
    - .txt: 每行一个数，或逗号分隔一行
    """
    if not path:
        return []
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            raise ValueError("voltage_seq_file 的 JSON 内容必须是 list")
        return [float(x) for x in arr]

    # txt / csv-like
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    # 若是多行，每行一个数
    if "\n" in content and "," not in content:
        return [float(line.strip()) for line in content.splitlines() if line.strip()]
    # 否则按逗号分割
    return [float(x.strip()) for x in content.split(",") if x.strip()]


def run_one_job(
    run_id: int,
    args: argparse.Namespace,
    current_voltage_seq: Optional[List[float]] = None,
) -> Tuple[bool, str, Any]:
    """
    运行一次有限元 pipeline。
    - current_voltage_seq: 强化学习传入的连续动作（电压序列）
    返回：(success, output_dir, reward)
    """
    jobname = f"{args.job_prefix}{run_id}"

    # 若 RL 没传，则从命令行提供的 voltage_seq / voltage_seq_file 获取
    if current_voltage_seq is None:
        if args.voltage_seq_file:
            current_voltage_seq = _load_voltage_seq_from_file(args.voltage_seq_file)
        elif args.voltage_seq:
            current_voltage_seq = _parse_voltage_seq(args.voltage_seq)
        else:
            current_voltage_seq = []  # 不传也允许，由 pipeline 决定默认策略/默认加载
    
    current_voltage_seq = _normalize_time(current_voltage_seq,args.t)
    config = {
        "inp_path": args.inp,
        "part": args.part,
        "delta": args.delta,
        "wu_range": (args.wu_min, args.wu_max),
        "wi_range": (args.wi_min, args.wi_max),
        "jobname": jobname,
        "fortran_template": args.fortran,
        "cpus": int(args.cpus),
        "gpus": int(args.gpus),
        "export_script": args.export_script,
        "abaqus_cmd": args.cmd,

        # ✅ 强化学习动作入口：把电压序列传给 run_pipeline
        "current_voltage_seq": current_voltage_seq,
    }
    print(f"▶▶▶ [第 {run_id + 1} 次] 正在运行: {jobname}")
    print(f"    current_voltage_seq_len={len(current_voltage_seq)}")

    success: bool = False
    output_dir: str = ""
    reward = None

    try:
        # ✅ run_pipeline 返回三个值：success, output_dir, reward
        success, output_dir, reward = run_pipeline(**config)

        if success:
            print(f"✅ 第 {run_id + 1} 次运行完成，结果保存在 {output_dir}")
            print(f"   reward = {reward}")
        else:
            print(f"⚠️ 第 {run_id + 1} 次运行失败")
            print(f"   reward = {reward}")

    except Exception as e:
        print(f"❌ 第 {run_id + 1} 次运行出错: {e}")

    print("=" * 60)
    time.sleep(args.delay)

    return success, output_dir, reward


def main():
    parser = argparse.ArgumentParser(description="批量运行 Abaqus pipeline 流程（支持RL传入current_voltage_seq）")
    parser.add_argument("--max_runs", type=int, default=10, help="运行次数")
    parser.add_argument("--delay", type=int, default=5, help="每次运行间隔秒数")
    parser.add_argument("--job_prefix", default="Plate4Job_", help="Job名前缀")

    # pipeline 参数
    parser.add_argument("--inp", default="abaqus_data/workdir_pipeline/Job-4.inp")
    parser.add_argument("--part", default="P4_19")
    parser.add_argument("--delta", default="0.0004")
    parser.add_argument("--wu_min", type=float, default=10)
    parser.add_argument("--wu_max", type=float, default=15)
    parser.add_argument("--wi_min", type=float, default=35)
    parser.add_argument("--wi_max", type=float, default=50)
    parser.add_argument("--fortran", default="scripts/Plate4.for")
    parser.add_argument("--cpus", default="10")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--export_script", default="export_all_data.py")
    parser.add_argument("--cmd", default="abaqus2025", help="abaqus路径/命令")

    # ✅ 允许命令行直接指定电压序列（非RL方式也能跑）
    parser.add_argument(
        "--voltage_seq",
        default="",
        help='电压序列（JSON list 或逗号分隔）。例: "[0.1,0.2,0.3]" 或 "0.1,0.2,0.3"',
    )
    parser.add_argument(
        "--voltage_seq_file",
        default="",
        help="电压序列文件路径（.json 为 list；.txt 每行一个数或逗号分隔）",
    )

    args = parser.parse_args()

    # 批跑：每次都用同一份 voltage_seq（如果你希望每次不同，可在这里随机/生成）
    for run_id in range(args.max_runs):
        run_one_job(run_id, args)


if __name__ == "__main__":
    main()
