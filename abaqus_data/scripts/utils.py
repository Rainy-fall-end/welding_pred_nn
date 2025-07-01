from datetime import datetime
import os
import time
import subprocess
import glob
import random
import shutil
import os

def create_folder(save_dir_base="data"):
    now = datetime.now()
    time_folder = now.strftime('%H%M')  # 如 "1430"
    save_dir = os.path.join(save_dir_base, time_folder)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def check_abaqus_completion(jobname, timeout=600*6*5):
    sta_file = f"{jobname}.sta"
    start_time = time.time()

    # 等待 .sta 文件生成
    while time.time() - start_time < timeout:
        cur_time = time.time() - start_time
        print(f"⏳ 等待 Abaqus 作业完成... {cur_time:.1f}s", flush=True)

        if os.path.exists(sta_file):
            try:
                with open(sta_file, 'r') as f:
                    content = f.read()
                    if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in content:
                        print("✅ 分析已完成")
                        return True
            except IOError:
                # 文件正在被占用/写入，跳过这一轮
                print("⚠️ .sta 文件当前无法读取，可能正在被写入...", flush=True)
        else:
            print("🕒 尚未生成 .sta 文件", flush=True)
        time.sleep(60)
    return False

def get_side_nodes(n=3):
    nodes = ['1', '50', '99', '148', '197', '246', '295', '344', '393', '442', '491', '540', '589', '638', '687', '736', '785', '834', '883', '932', '981', '1030', '1079', '1128', '1177', '1226', '1275', '1324', '1373', '1422', '1471', '1520', '1569', '1618', '1667', '1716', '1765', '1814', '1863', '1912', '1961', '2010', '2059', '2108', '2157', '2206', '2255', '2304', '2353', '2402', '2451', '2500', '2549', '2598', '2647', '2696', '2745', '2794', '2843', '2892', '2941', '2990', '3039', '3088', '3137', '3186', '3235', '3284', '3333', '3382', '3431', '3480', '3529', '3578', '3627', '3676', '3725', '3774', '3823', '3872', '3921', '3970', '4019', '4068', '4117', '4166', '4215', '4264', '4313', '4362', '4411', '4460', '4509', '4558', '4607', '4656', '4705', '4754', '4803', '4852', '4901', '4950', '4999', '5048', '5097', '5146', '5195', '5244', '5293', '5342', '5391', '5440', '5489', '5538', '5587', '5636', '5685', '5734', '5783', '5832', '5881', '5930', '5979', '6028', '6077', '6126', '6175', '6224', '6273', '6322', '6371', '6420', '6469', '6518', '6567', '6616', '6665', '6714', '6763', '6812', '6861', '6910', '1', '50', '99', '148', '197', '246', '295', '344', '393', '442', '491', '540', '589', '638', '687', '736', '785', '834', '883', '932', '981', '1030', '1079', '1128', '1177', '1226', '1275', '1324', '1373', '1422', '1471', '1520', '1569', '1618', '1667', '1716', '1765', '1814', '1863', '1912', '1961', '2010', '2059', '2108', '2157', '2206', '2255', '2304', '2353', '2402', '2451', '2500', '2549', '2598', '2647', '2696', '2745', '2794', '2843', '2892', '2941', '2990', '3039', '3088', '3137', '3186', '3235', '3284', '3333', '3382', '3431', '3480', '3529', '3578', '3627', '3676', '3725', '3774', '3823', '3872', '3921', '3970', '4019', '4068', '4117', '4166', '4215', '4264', '4313', '4362', '4460', '4558', '4656']
    return random.sample(nodes,n)

def export_last_frame(odb,csv):
    try:
        # 保存当前路径
        original_dir = os.getcwd()

        # 如果输出文件已存在，删除
        csv_path = os.path.join(original_dir, csv)
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"已删除文件：{csv}")
        else:
            print(f"文件不存在：{csv}")
        import pdb
        pdb.set_trace()
        # 构造 Abaqus 命令并执行
        abaqus_cmd = ['abaqus', 'python', 'export_last_frame.py', odb, csv]
        subprocess.run(abaqus_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ 命令执行失败：", e)
    except Exception as e:
        print("❌ 出现异常：", e)
    finally:
        # 确保无论如何都回到原目录
        os.chdir(original_dir)

def delete_all(name):
    # 当前目录
    current_dir = os.getcwd()

    # 匹配所有 Plate.* 的文件（不递归子目录）
    pattern = os.path.join(current_dir, f"{name}.*")
    files_to_delete = glob.glob(pattern)

    # 删除文件
    for file_path in files_to_delete:
        if os.path.isfile(file_path):  # 确保不是目录
            os.remove(file_path)
            print(f"删除：{file_path}")

def wait_and_move(src, dst, timeout=60, retry_interval=1.0):
    """
    等待 src 文件可读 & 未被占用，然后移动到 dst。
    - timeout: 最多等待秒数
    - retry_interval: 每次重试间隔
    """
    start = time.time()
    while True:
        try:
            # 1⃣ 文件必须存在
            if os.path.exists(src):
                # 2⃣ 尝试以独占方式打开验证不被占用（Windows）
                with open(src, "rb"):
                    pass
                # 3⃣ 可移动
                shutil.move(src, dst)
                print(f"✔ 已移动 {os.path.basename(src)} -> {dst}")
                return
        except (PermissionError, OSError):
            # 文件还在被占用，继续等待
            pass

        if time.time() - start > timeout:
            raise RuntimeError(f"等待移动 {src} 超时（{timeout}s）")
        time.sleep(retry_interval)
