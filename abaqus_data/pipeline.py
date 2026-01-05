# pipeline.py
import subprocess
import os
import shutil
import math
import sys
import numpy as np
from scripts.modify_for import modify_fortran_current_voltage,generate_dflux_stepwise_fortran
from scripts.utils import (
    create_folder,
    get_side_nodes,
    check_abaqus_completion,
    delete_all,
    wait_and_move,
)
from scripts.create_inp import perturb_inp_z
from postprocess.csv2npz import convert_matrix
from typing import Optional
from pathlib import Path

def run_abaqus_bat_python_export(
    abaqus_bat: str,
    export_script: str,
    jobname: str,
    out_csv: str = "out.csv",
    workdir: Optional[str] = None,
    log_path: Optional[str] = None,
):
    """
    Windows: 使用 cmd.exe /c 调用 abaqus.bat，执行：
      abaqus python export_script jobname.odb out.csv

    - shell=False（推荐）
    - 兼容路径含空格：在 shell=False + list args 下，不要手动加引号
    """
    # 不要 strip('"') 去“修正”路径：用户传入带引号/不带引号都能处理
    # 这里做一个温和的去引号（仅去首尾一对引号）
    def dequote(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    abaqus_bat_p = Path(dequote(abaqus_bat)).resolve()
    if not abaqus_bat_p.exists():
        raise FileNotFoundError(f"找不到 abaqus.bat：{abaqus_bat_p}")

    export_p = Path(dequote(export_script)).resolve()
    if not export_p.exists():
        raise FileNotFoundError(f"找不到 export_script：{export_p}")

    odb = f"{jobname}.odb"
    cwd = workdir if workdir is not None else os.getcwd()

    # 关键：shell=False + list 形式时，不要给任何参数加引号
    # cmd.exe 只负责解释/执行 bat；参数原样传递即可（含空格也没问题）
    cmd = [
        "cmd.exe", "/c",
        str(abaqus_bat_p),
        "python",
        str(export_p),
        odb,
        out_csv,
    ]

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            subprocess.run(
                cmd,
                check=True,
                shell=False,
                cwd=cwd,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
    else:
        subprocess.run(
            cmd,
            check=True,
            shell=False,
            cwd=cwd,
        )
        
def cal_E(data_dict,workers,E0 = 680.0):
    instances = ["P4_19-2", "P4_19-1"]
    E = np.zeros((71, len(instances)), dtype=np.float64)

    # 时间积分：workers 遍历时间片
    for w in workers:
        dt = float(w["timePeriod"])
        step_name = w["step_name"]

        for j, ins in enumerate(instances):
            nt = data_dict[step_name][ins]["nt"]       # (71,N,3) or (71,N) or (71,)
            s_vec = nt.mean(axis=1)     # (71,)
            E[:, j] += s_vec * dt                      # 对时间累加（积分）

    # 阈值惩罚：低于 E0 才扣分
    reward_matrix = np.minimum(E - E0, 0.0)            # (71,2)，元素<=0
    reward = float(reward_matrix.mean())               # scalar，<=0
    return reward

def cal_U(data_dict):
    instances = ["P4_19-1", "P4_19-2"]

    u_scores = []
    for ins in instances:
        u = data_dict["unflatten"][ins]["u"]          # (71,49,3)
        u_mag = np.linalg.norm(u, axis=-1)            # (71,49) 位移幅值
        u_scores.append(u_mag.mean() - 2e-4)          # 原来的 -0.0002 偏置保留

    reward = -float(np.mean(u_scores))
    return reward

def cal_S(data_dict):
    instances = ["P4_19-1", "P4_19-2"]

    s_scores = []
    for ins in instances:
        s = data_dict["unflatten"][ins]["s"]        
        s_mag = np.linalg.norm(s, axis=-1)           
        s_scores.append(s_mag.mean())          

    reward = -float(np.mean(s_scores))
    return reward

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
    abaqus_cmd: str = "abaqus",
    timeout = 60*40,
    current_voltage_seq = [
        (0,12,42),
        (1,4,32),
        (2,120,32),
        (4,10,200)
    ],
    cmd_path = None
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
    # new_wu, new_wi, _ = modify_fortran_current_voltage(
    #     fortran_path=fortran_template,
    #     output_path=user_subroutine,
    #     wu_range=wu_range,
    #     wi_range=wi_range,
    # )
    w_iv = generate_dflux_stepwise_fortran(
        output_path = user_subroutine,
        current_voltage_seq =  current_voltage_seq
    )
    # Step 3: 提交 Abaqus Job
    print("▶ 正在提交 Abaqus 作业 ...")
    abaqus_cmd_str = (
        f'abaqus job={jobname} input="{inp_output}" '
        f'user="{user_subroutine}" cpus={cpus} gpus={gpus} double=both'
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
    except subprocess.CalledProcessError:
        print("❌ Abaqus 作业失败，清理临时目录")
        shutil.rmtree(cur_dir)
        return False, None, None
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        delete_all(jobname)
        return True, None, None
        # raise RuntimeError(f"Abaqus run timed out: {abaqus_cmd_str}")
    # print(f"✅ Abaqus 作业已结束，输出日志 → {log_path}")

    # Step 4: 检查作业完成
    if check_abaqus_completion(jobname):
        print("✅ 检测到 .sta 指示作业成功完成")
    else:
        print("❌ 未检测到成功完成字样")

    # Step 5: 导出结果
    print("▶ 正在导出 Abaqus 结果 ...")
    try:
        if cmd_path:
            run_abaqus_bat_python_export(
                abaqus_bat=cmd_path,
                export_script=export_script,
                jobname=jobname,
                out_csv="out.csv",
                workdir=None,                 
                log_path="export.log",         
        )
        else:
            subprocess.run(
            [
                abaqus_cmd, "python", export_script,
                f"{jobname}.odb", "out.csv",
            ],
            check=True,
            shell=False,
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
    res["para"]["ui"] = w_iv
    res["para"]["vi"] = w_iv
    npz_path = os.path.join(cur_dir, "weld_data.npz")
    np.savez_compressed(npz_path,**res)
    # Step 7: 清理临时文件
    delete_all(jobname)
    print(f"✅ 全流程结束，输出文件保存在: {cur_dir}")
    reward_e = cal_E(data_dict=res["data"],workers=res["worker"],E0=55)
    reward_u = cal_U(data_dict=res["data"])
    reward_s = cal_S(data_dict=res["data"])
    return True, cur_dir, [reward_e,reward_u,reward_s]
