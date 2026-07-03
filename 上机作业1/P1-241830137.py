import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

# ------------------------------
# 1. bit_breakdown.png
# ------------------------------
def plot_bit_breakdown():
    # 获取 3.14 的 IEEE 754 单精度二进制表示
    num = 3.14
    # 将浮点数打包为4字节，按大端序解析为32位整数
    packed = struct.pack('>f', num)
    bits = struct.unpack('>I', packed)[0]
    bin_str = f'{bits:032b}'  # 32位二进制字符串

    # 指数位 (bits 30..23) 和 尾数位 (bits 22..0)
    sign_bit = bin_str[0]
    exponent_bits = bin_str[1:9]
    mantissa_bits = bin_str[9:32]

    # 创建表格数据
    # 第一行：指数位 (8 bits)
    exp_row = list(exponent_bits)
    # 第二行：尾数位 (23 bits)
    man_row = list(mantissa_bits)

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_axis_off()

    tb = Table(ax, bbox=[0, 0, 1, 1])

    n_exp = len(exp_row)   # 8
    n_man = len(man_row)   # 23

    # 添加表头
    tb.add_cell(0, 0, width=n_man/31, height=0.4, text='Mantissa (23 bits)',
                loc='center', facecolor='lightgray')
    tb.add_cell(0, n_man, width=n_exp/31, height=0.4, text='Exponent (8 bits)',
                loc='center', facecolor='lightgray')

    # 填充尾数位 (从高位到低位)
    for i, bit in enumerate(man_row):
        tb.add_cell(1, i, width=1/31, height=0.4, text=bit, loc='center',
                    edgecolor='black', linewidth=0.5)

    # 填充指数位 (从高位到低位)
    for i, bit in enumerate(exp_row):
        tb.add_cell(1, n_man + i, width=1/31, height=0.4, text=bit, loc='center',
                    edgecolor='black', linewidth=0.5)

    # 添加 MSB 和 LSB 标注
    # MSB 在最左边 (bit 31)
    ax.text(0.02, 0.1, 'Bit 31 (MSB)', transform=ax.transAxes, fontsize=10, ha='left')
    # LSB 在最右边 (bit 0)
    ax.text(0.98, 0.1, 'Bit 0 (LSB)', transform=ax.transAxes, fontsize=10, ha='right')

    ax.add_table(tb)
    plt.title('Binary Representation of 3.14 (IEEE 754 single-precision)', pad=20)
    plt.tight_layout()
    plt.savefig('bit_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('bit_breakdown.png saved.')


# ------------------------------
# 2. error_comparison.png
# ------------------------------
def plot_error_comparison():
    data = [
        ['Method', 'Absolute Error (log scale)'],
        ['Direct', '9.4e-20'],
        ['Denom Rationalize', '1.0e-10'],
        ['Numer Rationalize', '1.3e-20']
    ]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')

    tb = Table(ax, bbox=[0, 0, 1, 1])

    n_rows = len(data)
    n_cols = len(data[0])

    # 单元格尺寸
    col_widths = [0.45, 0.55]

    for i in range(n_rows):
        for j in range(n_cols):
            text = data[i][j]
            if i == 0:
                facecolor = 'lightgray'
                fontweight = 'bold'
            else:
                facecolor = 'white'
                fontweight = 'normal'

            tb.add_cell(i, j, width=col_widths[j], height=0.3, text=text,
                        loc='center', edgecolor='black', linewidth=0.8,
                        facecolor=facecolor)

    ax.add_table(tb)
    plt.title('Error Comparison', pad=15)
    plt.tight_layout()
    plt.savefig('error_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('error_comparison.png saved.')


# ------------------------------
# 3. float_structure.png
# ------------------------------
def plot_float_structure():
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # 绘制两个矩形区域
    # 尾数区域 (23 bits)
    rect_man = plt.Rectangle((0, 0.5), 23, 1, edgecolor='black', facecolor='skyblue', lw=2)
    ax.add_patch(rect_man)
    ax.text(11.5, 1.0, 'Mantissa (23 bits)', ha='center', va='center', fontsize=12, fontweight='bold')

    # 指数区域 (8 bits)
    rect_exp = plt.Rectangle((23, 0.5), 8, 1, edgecolor='black', facecolor='salmon', lw=2)
    ax.add_patch(rect_exp)
    ax.text(27, 1.0, 'Exponent (8 bits)', ha='center', va='center', fontsize=12, fontweight='bold')

    # 下方标注位范围
    ax.text(11.5, 0.2, 'bits 22..0', ha='center', va='center', fontsize=10)
    ax.text(27, 0.2, 'bits 30..23', ha='center', va='center', fontsize=10)

    # 添加符号位示意
    ax.text(-1, 1.0, 'Sign\n(1 bit)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))

    plt.title('Single-Precision Floating-Point Format (32 bits)', pad=15)
    plt.tight_layout()
    plt.savefig('float_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('float_structure.png saved.')


# ------------------------------
if __name__ == '__main__':
    plot_bit_breakdown()
    plot_error_comparison()
    plot_float_structure()