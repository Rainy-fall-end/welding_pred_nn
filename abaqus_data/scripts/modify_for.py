import re
import random
from typing import List, Tuple

def modify_fortran_current_voltage(
    fortran_path,
    output_path=None,
    wu_range=(10, 15),
    wi_range=(35, 50)
):
    """
    修改 Fortran 文件中的电压（wu）和电流（wi）为随机值，并返回结果。

    参数:
        fortran_path (str): 原始 Fortran 文件路径
        output_path (str): 修改后保存的文件路径（如果为 None，不保存）
        wu_range (tuple): 电压随机范围，例如 (10, 15)
        wi_range (tuple): 电流随机范围，例如 (35, 50)

    返回:
        new_wu (float): 修改后的电压值
        new_wi (float): 修改后的电流值
        modified_lines (List[str]): 修改后的 Fortran 文件内容行
    """
    # 读取 Fortran 文件
    with open(fortran_path, 'r') as f:
        lines = f.readlines()

    # 生成随机新值
    new_wu = round(random.uniform(*wu_range), 2)
    new_wi = round(random.uniform(*wi_range), 2)

    # 正则匹配替换
    pattern_wu = re.compile(r'^\s*wu\s*=\s*[\d.Ee+-]+', re.IGNORECASE)
    pattern_wi = re.compile(r'^\s*wi\s*=\s*[\d.Ee+-]+', re.IGNORECASE)

    modified_lines = []
    for line in lines:
        if pattern_wu.match(line):
            line = f"      wu={new_wu}\n"
        elif pattern_wi.match(line):
            line = f"      wi={new_wi}\n"
        modified_lines.append(line)

    # 写入新文件（如指定）
    if output_path:
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)

    return new_wu, new_wi, modified_lines

import os
from typing import List, Tuple, Optional

def generate_dflux_stepwise_fortran(
    output_path: str,
    current_voltage_seq: List[Tuple[float, float, float]],
    effi: float = 0.85,
    v: float = 0.04,
    power_factor: float = 25.0,
    per_line: int = 4,                  # 强烈建议 3~4，避免 72 列截断
    active_window: Optional[Tuple[float, float]] = None,  # (t_on, t_off)，如 (1.0, 5.0)
):
    """
    更稳健的 F77 固定格式 DFLUX 生成器（v2）：
    - 控制每行长度（<=72）以避免 DATA 被截断
    - 禁用 tab：全部空格
    - TKEY/WUKEY/WIKEY、wu/wi、T 显式 DOUBLE PRECISION，避免隐式类型坑
    - 常量用 D 指数（5.0D-4, 1.8675D-3），与 Abaqus 双精度更一致
    - 仍保持 stepwise（左端点常数）选择，不插值
    - 可选 active_window：热源仅在 [t_on, t_off] 输出，否则 FLUX=0

    参数:
        output_path: 输出 .for
        current_voltage_seq: [(t, wu, wi), ...] t 单位与 Abaqus TIME(2) 一致
        effi, v, power_factor: 热源参数
        per_line: DATA 每行数字个数（建议 3~4）
        active_window: (t_on, t_off)，若不为 None，则在窗外强制 FLUX=0
    """
    if not current_voltage_seq:
        raise ValueError("current_voltage_seq 不能为空")
    if abs(float(current_voltage_seq[0][0]) - 0.0) > 1e-12:
        raise ValueError("current_voltage_seq 首个时间必须是 t=0")
    for i in range(1, len(current_voltage_seq)):
        if float(current_voltage_seq[i][0]) < float(current_voltage_seq[i - 1][0]):
            raise ValueError("时间序列必须单调非降")

    if per_line < 1:
        raise ValueError("per_line 必须 >= 1")

    # ---------- Fortran number formatting (short + stable) ----------
    def fD(x: float, sig: int = 8) -> str:
        """
        生成尽量短的 DOUBLE PRECISION 字面量：
        - 用 D 指数
        - 不要太长，避免固定格式行长超限
        """
        # 用一般格式，保留 sig 位有效数字
        s = f"{float(x):.{sig}g}"
        s = s.replace("e", "D").replace("E", "D")
        if "D" not in s:
            # 非指数形式补 D0
            if "." not in s:
                s += ".0D0"
            else:
                s += "D0"
        return s

    TKEY  = [fD(t)  for (t, _, _) in current_voltage_seq]
    WUKEY = [fD(wu) for (_, wu, _) in current_voltage_seq]
    WIKEY = [fD(wi) for (_, _, wi) in current_voltage_seq]
    NKEY = len(TKEY)

    # ---------- fixed-format helpers ----------
    # 固定格式：列 1-5 为空格，列 6 为续行标志（用 $），列 7-72 为代码
    # 这里强制用空格，避免 tab 引发列错位
    def f77_line(code: str, cont: bool = False) -> str:
        """
        生成一行固定格式 Fortran。
        cont=False: '      ' + code
        cont=True : '     $' + code
        """
        prefix = "     $" if cont else "      "
        line = prefix + code
        # 安全：截断到 72 列（不建议依赖截断，但可以防止意外超长）
        return line[:72].rstrip()

    def data_block_dp(name: str, arr: List[str], per_line: int) -> str:
        """
        生成：
          DATA name / a,b,c,
         $ d,e,f,
         $ ... /
        且每行尽量短；每行只放 per_line 个数。
        """
        lines = []
        for i in range(0, len(arr), per_line):
            chunk = ", ".join(arr[i:i + per_line])
            if i == 0:
                # 第一行
                lines.append(f77_line(f"DATA {name} / {chunk},", cont=False))
            else:
                # 续行
                lines.append(f77_line(f"{chunk},", cont=True))
        # 收尾：把最后一行末尾逗号改成 ' /'
        last = lines[-1]
        if last.endswith(","):
            last = last[:-1]
        lines[-1] = (last + " /")[:72].rstrip()
        return "\n".join(lines)

    t_data  = data_block_dp("TKEY",  TKEY,  per_line)
    wu_data = data_block_dp("WUKEY", WUKEY, per_line)
    wi_data = data_block_dp("WIKEY", WIKEY, per_line)

    effi_s = fD(effi)
    v_s    = fD(v)
    pf_s   = fD(power_factor)

    # time window gating
    if active_window is not None:
        t_on, t_off = active_window
        t_on_s, t_off_s = fD(t_on), fD(t_off)
        gating_code = f"""
C     Active window gating
      IF (TIME(2) .LT. {t_on_s} .OR. TIME(2) .GT. {t_off_s}) THEN
        JLTYP=1
        FLUX(1)=0.0D0
        FLUX(2)=0.0D0
        RETURN
      ENDIF
"""
    else:
        gating_code = ""

    # ---------- source (fixed-format) ----------
    src = f"""      SUBROUTINE DFLUX(FLUX,SOL,JSTEP,JINC,TIME,NOEL,NPT,COORDS,JLTYP,
     $                 TEMP,PRESS,SNAME)
C
      INCLUDE 'ABA_PARAM.INC'
C
      DIMENSION COORDS(3),FLUX(2),TIME(2),SOL(2)
      CHARACTER*80 SNAME
C
C     Keyframes (stepwise, left-constant)
      INTEGER NKEY
      PARAMETER (NKEY={NKEY})
      DOUBLE PRECISION TKEY(NKEY), WUKEY(NKEY), WIKEY(NKEY)
      DOUBLE PRECISION wu, wi, effi, v, q, d
      DOUBLE PRECISION x, y, z, x0, y0, z0
      DOUBLE PRECISION a1, a2, b, c, f1, PI
      DOUBLE PRECISION heat1, heat2, shape1, shape2
{t_data}
{wu_data}
{wi_data}
C
      effi={effi_s}
      v={v_s}
{gating_code}
C
C     Stepwise pick (wu, wi) by TIME(2)
      CALL GET_WU_WI_STEP(TIME(2),wu,wi,TKEY,WUKEY,WIKEY,NKEY)
C
      q=wu*wi*effi*{pf_s}
      d=v*TIME(2)
C
      x=COORDS(1)
      y=COORDS(2)
      z=COORDS(3)
C
      x0=-v
      y0=0.0D0
      z0=5.0D-4
C
      a1=2.0D0*1.8675D-3
      a2=2.0D0*3.735D-3
      b =2.0D0*1.8675D-3
      c =2.0D0*2.763D-3
C
      f1=0.6D0
      PI=3.1415926D0
C
      heat1=6.0D0*sqrt(3.0D0)*q/(a1*b*c*PI*sqrt(PI))*f1
      heat2=6.0D0*sqrt(3.0D0)*q/(a2*b*c*PI*sqrt(PI))*(2.0D0-f1)
C
      shape1=exp(-3.0D0*(x-x0-d)**2/(a1)**2-3.0D0*(y-y0)**2/b**2
     $  -3.0D0*(z-z0)**2/c**2)
      shape2=exp(-3.0D0*(x-x0-d)**2/(a2)**2-3.0D0*(y-y0)**2/b**2
     $  -3.0D0*(z-z0)**2/c**2)
C
      JLTYP=1
      FLUX(2)=0.0D0
      IF (x .GE. (x0+d)) THEN
        FLUX(1)=heat1*shape1
      ELSE
        FLUX(1)=heat2*shape2
      ENDIF
      RETURN
      END
C=====================================================================
      SUBROUTINE GET_WU_WI_STEP(T, WU, WI, TKEY, WUKEY, WIKEY, NKEY)
C     Stepwise selection: pick last keyframe with T >= TKEY(i).
      INTEGER NKEY, I, IDX
      DOUBLE PRECISION T, WU, WI
      DOUBLE PRECISION TKEY(*), WUKEY(*), WIKEY(*)
C
      IF (NKEY .LE. 0) THEN
         WU = 0.0D0
         WI = 0.0D0
         RETURN
      ENDIF
C
      IF (T .LT. TKEY(1)) THEN
         WU = WUKEY(1)
         WI = WIKEY(1)
         RETURN
      ENDIF
C
      IDX = 1
      DO 10 I = 1, NKEY
         IF (T .GE. TKEY(I)) IDX = I
   10 CONTINUE
C
      WU = WUKEY(IDX)
      WI = WIKEY(IDX)
      RETURN
      END
"""

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 强制使用 \n，且不写 tab
    src = src.replace("\t", " ")

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)

    return output_path
