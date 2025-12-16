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
from typing import List, Tuple

def generate_dflux_stepwise_fortran(
    output_path: str,
    current_voltage_seq: List[Tuple[float, float, float]],
    effi: float = 0.85,
    v: float = 0.04,
    power_factor: float = 25.0,
):
    """
    生成一个分段常数（不插值）的 DFLUX.for 文件：
    - (t, wu, wi) 三元组定义在 Fortran 的 DATA 中
    - 运行时按 TIME(2) 选择最近的“左端点” (wu, wi)，不会插值
    - 要求: current_voltage_seq[0][0] == 0 且 t 单调非降

    参数:
        output_path: 生成的 .for 文件路径
        current_voltage_seq: [(t, wu, wi), ...]，t 单位须与模型一致（秒）
        effi: 热效率
        v: 热源行进速度 (m/s)
        power_factor: 你的单位/标定系数（若不需要可设为 1.0）
    """
    if not current_voltage_seq:
        raise ValueError("current_voltage_seq 不能为空")
    # 校验时间序列
    t0 = current_voltage_seq[0][0]
    if abs(t0 - 0.0) > 1e-12:
        raise ValueError("current_voltage_seq 首个时间必须是 t=0")
    for i in range(1, len(current_voltage_seq)):
        if current_voltage_seq[i][0] < current_voltage_seq[i-1][0]:
            raise ValueError("时间序列必须单调非降")
    # 格式化为 Fortran 双精度常量（D0 结尾）
    def f64(x: float) -> str:
        s = f"{x:.12g}"
        # 转为 Fortran 双精度指数形式（e->D）
        s = s.replace('e', 'D').replace('E', 'D')
        if 'D' not in s and '.' not in s:
            s += '.0D0'
        elif 'D' not in s:
            s += 'D0'
        return s

    TKEY = [f64(t)  for (t, _, _) in current_voltage_seq]
    WUKEY = [f64(wu) for (_, wu, _) in current_voltage_seq]
    WIKEY = [f64(wi) for (_, _, wi) in current_voltage_seq]
    NKEY  = len(TKEY)

    # 把 DATA 分块成多行，避免过长
    def data_lines(name: str, arr: List[str], per_line: int = 6) -> str:
        lines = []
        for i in range(0, len(arr), per_line):
            chunk = ", ".join(arr[i:i+per_line])
            if i == 0:
                lines.append(f"      DATA {name} / {chunk} /")
            else:
                lines.append(f"     &     {chunk} /")
        # 合并行：只有第一行以 DATA 开头，后续行用续行标记（列 6 有 &）
        if len(lines) <= 1:
            return lines[0]
        first = lines[0]
        rest = [lines[1].replace("      ", "     &", 1)]
        return "\n".join([first] + rest)

    t_data  = data_lines("TKEY",  TKEY)
    wu_data = data_lines("WUKEY", WUKEY)
    wi_data = data_lines("WIKEY", WIKEY)

    effi_s = f64(effi)
    v_s    = f64(v)
    pf_s   = f64(power_factor)

    # —— 生成完整 Fortran 源码（Goldak 双椭球 + 分段常数 GET_WU_WI_STEP）——
    src = f"""      SUBROUTINE DFLUX(FLUX,SOL,JSTEP,JINC,TIME,NOEL,NPT,COORDS,JLTYP,
     &                 TEMP,PRESS,SNAME)
C
C     DFLUX with stepwise (wu, wi) defined by keyframes (t, wu, wi).
C     No interpolation: uses left-constant value for [t_i, t_{ {NKEY} }).
C
      INCLUDE 'ABA_PARAM.INC'
      CHARACTER*80 SNAME
      DOUBLE PRECISION FLUX, SOL, TIME, COORDS, TEMP, PRESS
      DIMENSION FLUX(2), TIME(2), COORDS(3)

C     Keyframes
      INTEGER NKEY
      PARAMETER (NKEY={NKEY})
      DOUBLE PRECISION TKEY(NKEY), WUKEY(NKEY), WIKEY(NKEY)
{t_data}
{wu_data}
{wi_data}

C     Other parameters
      DOUBLE PRECISION wu, wi, effi, v, q, d
      DOUBLE PRECISION x, y, z, x0, y0, z0, xc
      DOUBLE PRECISION a1, a2, b, c, f1, PI
      DOUBLE PRECISION heat1, heat2, shape1, shape2
      INTEGER JLTYP

C     Efficiency and travel speed
      effi = {effi_s}
      v    = {v_s}

C     Get stepwise (wu, wi)
      CALL GET_WU_WI_STEP(TIME(2), wu, wi, TKEY, WUKEY, WIKEY, NKEY)

C     Effective input power q = U * I * effi * factor
      q = wu * wi * effi * {pf_s}

C     Moving source position
      d  = v * TIME(2)
      x  = COORDS(1)
      y  = COORDS(2)
      z  = COORDS(3)
      x0 = -v
      y0 = 0.D0
      z0 = 5.0D-4
      xc = x0 + d

C     Double-ellipsoid parameters (meters)
      a1 = 2.D0*1.8675D-3
      a2 = 2.D0*3.7350D-3
      b  = 2.D0*1.8675D-3
      c  = 2.D0*2.7630D-3

      f1 = 0.6D0
      PI = 3.141592653589793D0

C     Normalization
      heat1 = 6.D0*DSQRT(3.D0)*q/(a1*b*c*PI*DSQRT(PI)) * f1
      heat2 = 6.D0*DSQRT(3.D0)*q/(a2*b*c*PI*DSQRT(PI)) * (2.D0 - f1)

C     Shapes
      shape1 = DEXP( -3.D0*( (x-xc)*(x-xc)/(a1*a1)
     &                      + (y-y0)*(y-y0)/(b*b)
     &                      + (z-z0)*(z-z0)/(c*c) ) )
      shape2 = DEXP( -3.D0*( (x-xc)*(x-xc)/(a2*a2)
     &                      + (y-y0)*(y-y0)/(b*b)
     &                      + (z-z0)*(z-z0)/(c*c) ) )

C     Assign flux
      JLTYP = 1
      IF ( x .GE. xc ) THEN
         FLUX(1) = heat1 * shape1
      ELSE
         FLUX(1) = heat2 * shape2
      ENDIF

      RETURN
      END
C=====================================================================
      SUBROUTINE GET_WU_WI_STEP(T, WU, WI, TKEY, WUKEY, WIKEY, NKEY)
C     Stepwise selection: pick index i = max{{ i | T >= TKEY(i) }}.
      DOUBLE PRECISION T, WU, WI
      DOUBLE PRECISION TKEY(*), WUKEY(*), WIKEY(*)
      INTEGER NKEY, I, IDX

      IF (NKEY .LE. 0) THEN
         WU = 0.D0
         WI = 0.D0
         RETURN
      ENDIF

C     Clamp below first
      IF (T .LT. TKEY(1)) THEN
         WU = WUKEY(1)
         WI = WIKEY(1)
         RETURN
      ENDIF

C     Find the last key not exceeding T
      IDX = 1
      DO I = 1, NKEY
         IF (T .GE. TKEY(I)) IDX = I
      END DO

      WU = WUKEY(IDX)
      WI = WIKEY(IDX)
      RETURN
      END
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(src)
    return output_path
