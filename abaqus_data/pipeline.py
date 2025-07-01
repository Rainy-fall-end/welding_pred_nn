# pipeline.py
import subprocess
import os
import shutil
import sys
import numpy as np
from scripts.modify_for import modify_fortran_current_voltage
from scripts.utils import (
    create_folder,
    get_side_nodes,
    check_abaqus_completion,
    delete_all,
    wait_and_move,
)
from scripts.create_inp import perturb_inp_z
from postprocess.csv2npz import convert_matrix
def run_pipeline(
    inp_path: str,
    part: str,
    delta: str,
    wu_range: tuple,
    wi_range: tuple,
    jobname: str,
    fortran_template: str,
    cpus: int,
    gpus: int,
    export_script: str,
    abaqus_cmd: str = "abaqus"
):
    """
    自动执行 Abaqus Job 流程，包括 INP/FOR 扰动、作业提交、导出结果等。
    参数：
        inp_path         - 原始 INP 文件路径
        part             - 零件名称
        delta            - 扰动量 (str 类型)
        wu_range         - 电压范围 (tuple of float)
        wi_range         - 电流范围 (tuple of float)
        jobname          - 作业名称
        fortran_template - 原始 Fortran 文件路径
        cpus             - CPU 数量
        gpus             - GPU 数量
        export_script    - 结果导出脚本路径
        abaqus_cmd       - Abaqus 命令调用（默认 "abaqus"）
    返回：
        success (bool), output_dir (str)
    """

    # 文件路径准备
    cur_dir = create_folder()
    inp_output = os.path.join(cur_dir, "Job.inp")
    user_subroutine = os.path.join(cur_dir, "plate.for")
    log_path = os.path.join(cur_dir, f"{jobname}_abaqus.log")

    # Step 1: 生成扰动后的 INP
    print("▶ 正在生成扰动后的 .inp 文件 ...")
    node_ids = get_side_nodes()

    perturb_inp_z(
        inp_path=inp_path,
        out_path=inp_output,
        part_name=part,
        target_nodes=node_ids,
        delta=float(delta)
    )

    # Step 2: 生成扰动后的 .for
    print("▶ 正在生成扰动后的 .for 文件 ...")
    new_wu, new_wi, _ = modify_fortran_current_voltage(
        fortran_path=fortran_template,
        output_path=user_subroutine,
        wu_range=wu_range,
        wi_range=wi_range,
    )

    # Step 3: 提交 Abaqus Job
    print("▶ 正在提交 Abaqus 作业 ...")
    abaqus_cmd_str = (
        f'{abaqus_cmd} interactive job={jobname} '
        f'input="{inp_output}" user="{user_subroutine}" '
        f'cpus={cpus} gpus={gpus} double=both'
    )
    print(f"  ⤷ 运行命令: {abaqus_cmd_str}")
    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                abaqus_cmd_str,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            proc.communicate()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, abaqus_cmd_str)
    except subprocess.CalledProcessError:
        print("❌ Abaqus 作业失败，清理临时目录")
        shutil.rmtree(cur_dir)
        return False, None

    print(f"✅ Abaqus 作业已结束，输出日志 → {log_path}")

    # Step 4: 检查作业完成
    if check_abaqus_completion(jobname):
        print("✅ 检测到 .sta 指示作业成功完成")
    else:
        print("❌ 未检测到成功完成字样")

    # Step 5: 导出结果
    print("▶ 正在导出 Abaqus 结果 ...")
    try:
        subprocess.run(
            [
                abaqus_cmd, "python", export_script,
                f"{jobname}.odb", "out.csv",
            ],
            check=True,
            shell=True,
        )
        print("✔️ 结果导出成功")
    except subprocess.CalledProcessError as e:
        print("❌ 导出失败:", e)

    # Step 6: 等待并移动结果文件
    for file_name in ("out.csv_results.csv", "out.csv_steps.csv"):
        src = file_name
        dst = os.path.join(cur_dir, file_name)
        wait_and_move(src, dst, timeout=60, retry_interval=1)
    res = convert_matrix(os.path.join(cur_dir, "out.csv_results.csv"),os.path.join(cur_dir, "out.csv_steps.csv"))
    res["para"] = {}
    res["para"]["ui"] = new_wu
    res["para"]["vi"] = new_wi
    npz_path = os.path.join(cur_dir, "weld_data.npz")
    np.savez_compressed(npz_path,**res)
    # Step 7: 清理临时文件
    delete_all(jobname)
    print(f"✅ 全流程结束，输出文件保存在: {cur_dir}")
    
    return True, cur_dir
