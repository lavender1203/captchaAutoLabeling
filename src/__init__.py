# 自动注册所有模型
from src.model import ModelFactory

import inspect
import importlib

def get_all_models(module_name):
    # 动态导入模块
    module = importlib.import_module(module_name)
    # 获取模块中的所有类
    classes = [name for name, obj in inspect.getmembers(module, inspect.isclass)]
    return classes

# 获取所有模型
models = get_all_models("src.model")
# 命名规范: 模型名 + Model
models = [model for model in models if model.endswith("Model") and model != "BaseModel"]
ModelFactory.register_all_models(models)
