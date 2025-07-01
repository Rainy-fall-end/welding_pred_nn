# batch_runner.py
import argparse
import time
from pipeline import run_pipeline  # ✅ 直接导入函数

def run_one_job(run_id, args):
    jobname = f"{args.job_prefix}{run_id}"
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
    }

    print(f"▶▶▶ [第 {run_id+1} 次] 正在运行: {jobname}")
    try:
        success, output_dir = run_pipeline(**config)
        if success:
            print(f"✅ 第 {run_id+1} 次运行完成，结果保存在 {output_dir}")
        else:
            print(f"⚠️ 第 {run_id+1} 次运行失败")
    except Exception as e:
        print(f"❌ 第 {run_id+1} 次运行出错: {e}")
    print("=" * 60)
    time.sleep(args.delay)


def main():
    parser = argparse.ArgumentParser(description="批量运行 Abaqus pipeline 流程")
    parser.add_argument('--max_runs', type=int, default=10, help='运行次数')
    parser.add_argument('--delay', type=int, default=5, help='每次运行间隔秒数')
    parser.add_argument('--job_prefix', default="Plate4Job_", help='Job名前缀')

    # pipeline 参数
    parser.add_argument('--inp', default="workdir_pipeline/Job-4.inp")
    parser.add_argument('--part', default="P4_19")
    parser.add_argument('--delta', default="0.0004")
    parser.add_argument('--wu_min', type=float, default=10)
    parser.add_argument('--wu_max', type=float, default=15)
    parser.add_argument('--wi_min', type=float, default=35)
    parser.add_argument('--wi_max', type=float, default=50)
    parser.add_argument('--fortran', default="scripts/Plate4.for")
    parser.add_argument('--cpus', default="4")
    parser.add_argument('--gpus', default="1")
    parser.add_argument('--export_script', default="export_all_data.py")
    parser.add_argument('--cmd', default="abaqus", help="abaqus路径")
    
    args = parser.parse_args()

    for run_id in range(args.max_runs):
        run_one_job(run_id, args)


if __name__ == "__main__":
    main()
