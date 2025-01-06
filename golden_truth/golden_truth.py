# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:12:58 2025

@author: yuyue
"""

import pandas as pd
import json

# 定义文件路径
tsv_file_path = r"C:\Users\yuyue\Downloads\AdMIRe Subtask A Train (1)\train\subtask_a_train.tsv"  
idiomatic_json_path = "golden_truth_idiomatic.json"
literal_json_path = "golden_truth_literal.json"

# 读取 TSV 文件
data = pd.read_csv(tsv_file_path, sep="\t")

# 根据 sentence_type 分组
idiomatic_data = data[data["sentence_type"] == "idiomatic"]
literal_data = data[data["sentence_type"] == "literal"]

# 创建字典
idiomatic_dict = {row["compound"]: eval(row["expected_order"]) for _, row in idiomatic_data.iterrows()}
literal_dict = {row["compound"]: eval(row["expected_order"]) for _, row in literal_data.iterrows()}

# 保存为 JSON 文件
with open(idiomatic_json_path, "w", encoding="utf-8") as idiomatic_file:
    json.dump(idiomatic_dict, idiomatic_file, indent=4)

with open(literal_json_path, "w", encoding="utf-8") as literal_file:
    json.dump(literal_dict, literal_file, indent=4)

print(f"JSON files saved:\n- {idiomatic_json_path}\n- {literal_json_path}")

combined_json_path = "golden_truth.json"

# 读取 TSV 文件
data = pd.read_csv(tsv_file_path, sep="\t")

# 创建字典
combined_dict = {row["compound"]: eval(row["expected_order"]) for _, row in data.iterrows()}

# 保存为 JSON 文件
with open(combined_json_path, "w", encoding="utf-8") as combined_file:
    json.dump(combined_dict, combined_file, indent=4)

print(f"Combined JSON file saved: {combined_json_path}")
