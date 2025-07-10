import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


def split_csv_to_train_valid(csv_path, output_dir, target_column=None, valid_size=0.2,
                             random_state=42, stratify=False, output_format='json'):
    """
    将CSV文件分割为训练集和验证集，可指定输出格式

    参数:
    csv_path (str): CSV文件路径
    output_dir (str): 输出目录
    target_column (str): 用于分层抽样的目标列名
    valid_size (float): 验证集比例 (0.0-1.0)
    random_state (int): 随机种子
    stratify (bool): 是否使用分层抽样
    output_format (str): 输出格式 ('csv' 或 'json')

    返回:
    train_path, valid_path: 分割后的文件路径
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"原始数据集大小: {len(df)} 条记录")

    # 检查目标列是否存在（如果使用分层抽样）
    if stratify and target_column and target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于数据集中")

    # 设置分层抽样参数
    stratify_col = df[target_column] if stratify and target_column else None

    # 分割数据集
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        stratify=stratify_col
    )

    print(f"训练集大小: {len(train_df)} 条记录 ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"验证集大小: {len(valid_df)} 条记录 ({len(valid_df) / len(df) * 100:.1f}%)")

    # 生成输出文件路径
    base_name = os.path.basename(csv_path).replace(".csv", "")

    # 根据输出格式确定文件扩展名
    ext = "json" if output_format.lower() == "json" else "csv"
    train_path = os.path.join(output_dir, f"{base_name}_train.{ext}")
    valid_path = os.path.join(output_dir, f"{base_name}_valid.{ext}")

    # 保存数据集（根据格式选择）
    if output_format.lower() == "json":
        # 保存为JSON
        # train_df.to_json(train_path, orient="records", force_ascii=False, indent=2)
        # valid_df.to_json(valid_path, orient="records", force_ascii=False, indent=2)
        # 保存为JSON Lines
        with open(train_path, 'w', encoding='utf-8') as f:
            for _, row in train_df.iterrows():
                json_obj = {"label": row["label"], "review": row["review"]}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

        with open(valid_path, 'w', encoding='utf-8') as f:
            for _, row in valid_df.iterrows():
                json_obj = {"label": row["label"], "review": row["review"]}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        print(f"训练集已保存为JSON: {train_path}")
        print(f"验证集已保存为JSON: {valid_path}")
    else:
        # 保存为CSV
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        print(f"训练集已保存为CSV: {train_path}")
        print(f"验证集已保存为CSV: {valid_path}")

    return train_path, valid_path


# ==================== 使用示例 ====================

# 示例1：全部输出为JSON
csv_path = "./data/文本分类练习.csv"
output_dir = "data"
train_path, valid_path = split_csv_to_train_valid(
    csv_path,
    output_dir,
    # target_column="sentiment",
    valid_size=0.2,
    random_state=42,
    # stratify=True,
    output_format="json"  # 控制所有输出格式
)
'''
# 示例2：全部输出为CSV
train_path, valid_path = split_csv_to_train_valid(
    csv_path,
    "csv_data",
    target_column="sentiment",
    output_format="csv"  # CSV格式
)


# 示例3：混合格式（需要修改函数支持）
def split_csv_mixed_format(csv_path, output_dir, target_column=None,
                           train_format="csv", valid_format="json",
                           valid_size=0.2, random_state=42, stratify=False):
    """
    支持训练集和验证集不同输出格式的分割函数
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 分割数据集
    stratify_col = df[target_column] if stratify and target_column else None
    train_df, valid_df = train_test_split(
        df, test_size=valid_size, random_state=random_state, stratify=stratify_col
    )

    base_name = os.path.basename(csv_path).replace(".csv", "")

    # 训练集路径
    train_path = os.path.join(output_dir, f"{base_name}_train.{train_format}")
    # 验证集路径
    valid_path = os.path.join(output_dir, f"{base_name}_valid.{valid_format}")

    # 保存训练集
    if train_format.lower() == "json":
        train_df.to_json(train_path, orient="records", force_ascii=False, indent=2)
    else:
        train_df.to_csv(train_path, index=False)

    # 保存验证集
    if valid_format.lower() == "json":
        valid_df.to_json(valid_path, orient="records", force_ascii=False, indent=2)
    else:
        valid_df.to_csv(valid_path, index=False)

    print(f"训练集保存为: {train_path} ({train_format})")
    print(f"验证集保存为: {valid_path} ({valid_format})")

    return train_path, valid_path


# 使用混合格式：训练集CSV，验证集JSON
train_path, valid_path = split_csv_mixed_format(
    "sales_data.csv",
    "mixed_format_data",
    target_column="product_category",
    train_format="csv",
    valid_format="json",
    valid_size=0.15
)
'''
