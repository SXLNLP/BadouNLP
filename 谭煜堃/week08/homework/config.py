# -*- coding: utf-8 -*-

"""
配置参数信息
"""

# 默认配置
default_config = {
    "model_path": "model_output",
    "schema_path": "./data/schema.json",
    "train_data_path": "./data/train.json",
    "valid_data_path": "./data/valid.json",
    "vocab_path":"./chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "optimizer": "adam",
    "learning_rate": 1e-3,
}




class GlobalConfigManager(object):
    def __init__(self):
        self.config = default_config

    # 单例模式
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

    def reload_config(self, config_index):
        import csv
        with open("./homework/hyper_param.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            config_rows = list(reader)

        header = config_rows[0]
        # config_index 是数据行的从1开始的索引, 所以自动跳过表头了
        data_row_index = config_index
        selected_params = dict(zip(header, config_rows[data_row_index]))

        for key_from_csv, value_str in selected_params.items():
            if key_from_csv in self.config:
                # 从默认配置中推断目标类型
                target_type = type(self.config[key_from_csv])
                try:
                    # 将从CSV中读取的字符串转换为正确的类型
                    self.config[key_from_csv] = target_type(value_str)
                except ValueError:
                    # 如果类型转换失败 (例如, optimizer="adam"), 直接使用字符串值
                    self.config[key_from_csv] = value_str
    def get_config(self):
        return self.config



if __name__ == "__main__":
    config_manager = GlobalConfigManager()
    print("Default config:", config_manager.config)
    config_manager.reload_config(1)
    print("Reloaded config:", config_manager.config)