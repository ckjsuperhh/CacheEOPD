"""
Unified registry utilities and simple JSON-based save/load helpers.
统一的注册表工具函数及基于 JSON 的简单序列化/反序列化辅助函数。

This module provides:
本模块提供以下功能：

- create_registry: factory to create (registry dict, register decorator, get_class)
  创建注册表系统的工厂函数，返回（注册表字典, 注册装饰器, 获取类函数）三元组。

- capture_init_args: decorator to record __init__ kwargs on instances as _init_args
  类装饰器，用于在实例化时捕获 __init__ 的参数，并存储到实例的 _init_args 属性中，
  方便后续序列化。

- save_object / load_object: serialize/deserialize object configs via registry
  将对象的构造配置（类名 + 初始化参数）序列化到 JSON 文件 / 从 JSON 文件反序列化并重建对象。

- dumps_object_config / loads_object_config: 与上面类似的字符串版本，
  用于将配置序列化为 JSON 字符串 / 从 JSON 字符串反序列化。

在 C2C 框架中的作用：
  本模块是 C2C（Cache-to-Cache）框架中投影器（projector）等组件的注册与管理基础设施。
  通过 create_registry 创建的注册表允许用户以装饰器方式注册自定义的投影器类，
  并通过名称动态获取。配合 capture_init_args 和 save/load 函数，
  可以将模型配置持久化为 JSON 文件，实现训练/推理配置的保存与恢复。

与其他模块的关系：
  - rosetta.utils: 本模块位于 utils 包中，作为底层工具被其他模块调用。
  - 投影器模块（如 projectors/）使用 register_model 装饰器注册自身。
  - 训练/评估脚本使用 get_projector_class 按名称获取投影器类，
    并使用 load_object 从配置文件加载投影器实例。
"""

from __future__ import annotations

import inspect  # 用于检查函数签名，获取 __init__ 的参数名称和默认值
import json     # JSON 序列化/反序列化，用于配置文件的读写
from typing import Dict, Type, Callable, Optional, Tuple, TypeVar, Any
import torch    # PyTorch 框架，用于处理 torch.dtype、torch.device 等特殊类型的序列化

# 泛型类型变量，用于注册表中存储的类类型
# 在 create_registry 的返回类型注解中使用，使类型检查器能推断注册/获取的类类型
T = TypeVar("T")


def create_registry(
    registry_name: str,
    case_insensitive: bool = False,
) -> Tuple[Dict[str, Type[T]], Callable[..., Type[T]], Callable[[str], Type[T]]]:
    """
    Create a registry system with register and get functions.
    创建一个注册表系统，包含注册装饰器和获取函数。

    这是 C2C 框架中的核心注册机制。通过此工厂函数，不同的组件类型
    （如投影器 projector）可以拥有各自独立的注册表，实现松耦合的组件管理。

    Args:
        registry_name: Name used in error messages (e.g., "projector")
                       用于错误消息中的注册表名称，例如 "projector"
        case_insensitive: Whether to store lowercase versions of names
                          是否同时存储名称的小写版本，实现大小写不敏感查找

    Returns:
        (registry_dict, register_function, get_function)
        三元组：(注册表字典, 注册装饰器函数, 按名称获取类函数)

    典型用法:
        # 创建投影器注册表
        REGISTRY, register_projector, get_projector = create_registry("projector")

        # 使用装饰器注册类
        @register_projector
        class LinearProjector:
            ...

        # 按名称获取类
        cls = get_projector("LinearProjector")
    """

    # 注册表字典：存储 名称 -> 类 的映射关系
    # 如果 case_insensitive=True，同一个类会同时以原始名称和小写名称存储
    registry: Dict[str, Type[T]] = {}

    def register(cls_or_name=None, name: Optional[str] = None):
        """Register a class in the registry. Supports multiple usage patterns.
        将类注册到注册表中。支持多种使用模式。

        Usage / 用法:
            # 模式1: 直接使用装饰器，以类名作为注册名
            @register
            class Foo: ...

            # 模式2: 传入字符串参数作为注册名
            @register("foo")
            class Foo: ...

            # 模式3: 通过关键字参数指定注册名
            @register(name="foo")
            class Foo: ...
        """

        def _register(c: Type[T]) -> Type[T]:
            # Determine the name to use
            # 确定注册名称的优先级：
            # 1. 如果 cls_or_name 是字符串，则用它作为名称（模式2: @register("foo")）
            # 2. 如果 name 关键字参数不为空，则用它（模式3: @register(name="foo")）
            # 3. 否则使用类本身的 __name__（模式1: @register）
            if isinstance(cls_or_name, str):
                class_name = cls_or_name
            elif name is not None:
                class_name = name
            else:
                class_name = c.__name__

            # 将类存入注册表
            registry[class_name] = c
            # 如果需要大小写不敏感，同时存入小写版本
            # 这样 "MyProjector" 和 "myprojector" 都能查找到同一个类
            if case_insensitive:
                registry[class_name.lower()] = c
            return c

        # 判断调用方式，以支持不同的装饰器语法
        if cls_or_name is not None and not isinstance(cls_or_name, str):
            # Called as @register or register(cls)
            # 直接作为装饰器使用（无参数），cls_or_name 就是被装饰的类
            return _register(cls_or_name)
        else:
            # Called as @register("name") or @register(name="name")
            # 带参数调用，返回内部装饰器函数 _register
            return _register

    def get_class(name: str) -> Type[T]:
        """Get class by name from registry.
        按名称从注册表中获取类。

        Args:
            name: 注册时使用的类名称

        Returns:
            对应的类对象

        Raises:
            ValueError: 如果名称未在注册表中注册，会列出所有可用的类名
        """
        if name not in registry:
            # Build readable available list without duplicates when case_insensitive
            # 构建可用类名列表，当 case_insensitive=True 时去除小写重复项
            seen = set()
            available = []
            for k in registry.keys():
                if k.lower() in seen:
                    continue
                seen.add(k.lower())
                available.append(k)
            raise ValueError(
                f"Unknown {registry_name} class: {name}. Available: {available}"
            )
        return registry[name]

    return registry, register, get_class


def capture_init_args(cls):
    """
    Decorator to capture initialization arguments of a class.
    装饰器：捕获类的初始化参数。

    Stores the mapping of the constructor's parameters to the values supplied
    at instantiation time into `self._init_args` for later serialization.
    将构造函数的参数名与实例化时传入的值之间的映射关系存储到 `self._init_args` 中，
    以便后续通过 save_object 序列化为 JSON 配置。

    工作原理：
      1. 保存原始 __init__ 方法的引用
      2. 创建一个新的 __init__ 方法，在其中：
         - 通过 inspect.signature 解析原始构造函数的参数列表
         - 将位置参数映射到对应的参数名
         - 合并关键字参数
         - 将结果存入 self._init_args
         - 调用原始 __init__ 完成实际初始化
      3. 替换类的 __init__ 方法并返回修饰后的类

    典型用法:
        @capture_init_args
        class LinearProjector:
            def __init__(self, input_dim, output_dim, bias=True):
                ...

        proj = LinearProjector(128, 64, bias=False)
        # proj._init_args == {"input_dim": 128, "output_dim": 64, "bias": False}
        # 之后可通过 save_object(proj, "config.json") 持久化配置

    Args:
        cls: 需要被装饰的类

    Returns:
        修饰后的类（__init__ 被替换为带参数捕获逻辑的版本）
    """
    # 保存原始 __init__ 方法的引用，后续会在新 __init__ 中调用
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        # Store all initialization arguments
        # 存储所有初始化参数：参数名 -> 值的映射
        init_args: Dict[str, Any] = {}

        # Get parameter names from the original __init__ method
        # 通过 inspect 解析原始 __init__ 的参数签名
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self' / 跳过 self 参数

        # Map positional args to parameter names
        # 将位置参数按顺序映射到对应的参数名
        # 例如: __init__(self, a, b, c) 调用 MyClass(1, 2) -> {"a": 1, "b": 2}
        for i, arg in enumerate(args):
            if i < len(param_names):
                init_args[param_names[i]] = arg

        # Add keyword args / 合并关键字参数
        # 关键字参数可能覆盖上面的位置参数映射（不过 Python 中这本身会报错）
        init_args.update(kwargs)

        # 将参数映射存储到实例属性中，供 save_object 使用
        self._init_args = init_args

        # Call the original __init__ / 调用原始构造函数完成实际初始化
        original_init(self, *args, **kwargs)

    # 替换类的 __init__ 方法
    cls.__init__ = new_init
    return cls


# -------------------------
# Serialization utilities
# 序列化/反序列化工具函数
# -------------------------

def _encode_value(value: Any) -> Any:
    """Best-effort JSON encoding for common ML types.
    尽力将常见的 ML 相关类型编码为 JSON 可序列化的格式。

    支持以下类型的编码：
      - 基本类型 (None, bool, int, float, str): 直接返回
      - tuple: 递归转换为 list（JSON 不支持 tuple）
      - list: 递归编码每个元素
      - dict: 递归编码每个值
      - torch.dtype: 编码为 {"__type__": "torch.dtype", "value": "float32"} 格式
      - torch.device: 编码为 {"__type__": "torch.device", "value": "cuda:0"} 格式
      - 其他类型: 回退为字符串表示

    Args:
        value: 需要编码的值，可以是任意 Python 对象

    Returns:
        JSON 可序列化的值（基本类型、list 或 dict）
    """
    # Primitives and None / 基本类型和 None，直接返回
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Tuples -> lists / 元组转为列表（JSON 不支持元组类型）
    if isinstance(value, tuple):
        return [
            _encode_value(v) for v in value  # 递归编码每个元素
        ]

    # Lists / 列表：递归编码每个元素
    if isinstance(value, list):
        return [
            _encode_value(v) for v in value
        ]

    # Dicts / 字典：递归编码每个值
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}

    # torch-specific types / PyTorch 特有类型的处理
    if torch is not None:
        # torch.dtype / 数据类型（如 torch.float32, torch.int64 等）
        if isinstance(value, type(getattr(torch, "float32", object))):
            # Guard: torch.dtype is not a class; rely on str(value) format
            # 注意：torch.dtype 不是一个普通类，需要通过字符串表示来解析
            # str(value) 格式为 "torch.float32"，提取最后一段作为类型名
            s = str(value)
            if s.startswith("torch."):
                return {"__type__": "torch.dtype", "value": s.split(".")[-1]}

        # torch.device / 设备类型（如 torch.device("cuda:0")）
        if isinstance(value, getattr(torch, "device", ())):
            return {"__type__": "torch.device", "value": str(value)}

    # Fallback to string representation
    # 无法识别的类型，回退为字符串表示，并标记 __type__ 为 "str"
    return {"__type__": "str", "value": str(value)}


def _decode_value(value: Any) -> Any:
    """Decode values produced by _encode_value, recursively for containers.
    解码由 _encode_value 编码的值，对容器类型递归处理。

    与 _encode_value 互逆，将带有 __type__ 标记的特殊字典还原为对应的 Python/PyTorch 对象：
      - torch.dtype 标记 -> torch.float32 等实际类型
      - torch.device 标记 -> torch.device 对象
      - str 标记 -> 普通字符串
      - list: 递归解码每个元素
      - dict（无 __type__ 标记）: 递归解码每个值
      - 基本类型: 直接返回

    Args:
        value: 需要解码的值

    Returns:
        还原后的 Python 对象
    """
    # Lists: decode each element / 列表：递归解码每个元素
    if isinstance(value, list):
        return [_decode_value(v) for v in value]

    # Dicts: either a typed-marker dict or a regular mapping that needs recursive decoding
    # 字典：可能是带 __type__ 标记的特殊字典（需要还原类型），也可能是普通字典（需要递归解码值）
    if isinstance(value, dict):
        if "__type__" in value:
            # 提取类型标记和值
            t = value.get("__type__")
            v = value.get("value")

            # 还原 torch.dtype：通过 getattr 从 torch 模块获取对应的 dtype 对象
            # 例如 {"__type__": "torch.dtype", "value": "float32"} -> torch.float32
            if t == "torch.dtype" and torch is not None:
                dtype = getattr(torch, str(v), None)
                if dtype is None:
                    raise ValueError(f"Unknown torch.dtype: {v}")
                return dtype

            # 还原 torch.device：构造 torch.device 对象
            # 例如 {"__type__": "torch.device", "value": "cuda:0"} -> torch.device("cuda:0")
            if t == "torch.device" and torch is not None:
                return torch.device(v)

            # 还原普通字符串（之前通过 str() 回退编码的值）
            if t == "str":
                return str(v)

            # Unknown type marker; return raw as-is
            # 未知类型标记，原样返回
            return value

        # Regular dict: decode values recursively / 普通字典：递归解码每个值
        return {k: _decode_value(v) for k, v in value.items()}

    # Primitives and anything else: return as-is / 基本类型和其他类型：直接返回
    return value


def save_object(obj: Any, file_path: str) -> None:
    """
    Save an object's construction config to a JSON file.
    将对象的构造配置保存到 JSON 文件。

    The object is expected to have been decorated with capture_init_args,
    so that `obj._init_args` exists.
    预期对象已被 @capture_init_args 装饰器修饰，因此 obj._init_args 属性存在。

    保存的 JSON 格式示例:
    {
      "class": "LinearProjector",
      "init_args": {
        "input_dim": 128,
        "output_dim": 64,
        "dtype": {"__type__": "torch.dtype", "value": "float32"}
      }
    }

    工作流程：
      1. 获取对象的类名（obj.__class__.__name__）
      2. 获取 _init_args（由 capture_init_args 在初始化时捕获）
      3. 通过 _encode_value 递归序列化参数（处理 torch.dtype 等特殊类型）
      4. 构建 payload 字典并写入 JSON 文件

    Args:
        obj: 需要保存配置的对象实例（应被 @capture_init_args 修饰）
        file_path: 目标 JSON 文件路径
    """
    # 获取对象的类名，用于后续反序列化时从注册表查找对应的类
    class_name = obj.__class__.__name__
    # 获取由 capture_init_args 捕获的初始化参数映射
    init_args = getattr(obj, "_init_args", {})

    # 递归编码参数，将 torch.dtype、torch.device 等特殊类型转换为 JSON 友好格式
    serializable_args = _encode_value(init_args)
    # 构建最终的 JSON payload：包含类名和初始化参数
    payload = {
        "class": class_name,
        "init_args": serializable_args,
    }

    # 以 UTF-8 编码写入 JSON 文件，缩进为 2 空格以保持良好的可读性
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_object(
    file_path: str,
    get_class_fn: Callable[[str], Type[T]],
    override_args: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Load an object from a JSON config file previously saved by save_object.
    从 save_object 保存的 JSON 配置文件中加载对象实例。

    工作流程：
      1. 读取 JSON 文件，解析出类名和编码后的初始化参数
      2. 通过 _decode_value 递归解码参数（还原 torch.dtype 等特殊类型）
      3. 可选地用 override_args 覆盖部分参数（例如更换 device）
      4. 通过 get_class_fn 从注册表查找对应的类
      5. 使用解码后的参数实例化类并返回

    Args:
        file_path: JSON 配置文件的路径
        get_class_fn: 从注册表中按名称解析类的函数，
                      通常是 create_registry 返回的 get_class
        override_args: 可选的参数覆盖字典，用于在加载时替换配置文件中的部分参数
                       （例如将训练时的 device="cuda:0" 覆盖为推理时的 "cpu"）

    Returns:
        根据配置文件实例化的对象

    Raises:
        ValueError: 如果类名未在注册表中注册
    """
    # 读取 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # 从 payload 中提取类名和编码后的初始化参数
    class_name = payload["class"]
    encoded_args = payload.get("init_args", {})
    # 递归解码参数：将 {"__type__": "torch.dtype", ...} 等标记还原为 Python/PyTorch 对象
    init_args = _decode_value(encoded_args)

    # 如果提供了覆盖参数，合并到 init_args 中（覆盖同名参数）
    # 典型场景：加载模型时更换设备，如 override_args={"device": torch.device("cpu")}
    if override_args:
        init_args.update(override_args)

    # 通过注册表查找类
    cls = get_class_fn(class_name)
    # 使用解码后的参数实例化并返回
    return cls(**init_args)


def dumps_object_config(obj: Any) -> str:
    """Return a JSON string with the object's class and init args.
    将对象的类名和初始化参数序列化为 JSON 字符串。

    与 save_object 类似，但输出为字符串而非写入文件。
    适用于将配置嵌入日志、消息传递或网络传输等场景。

    Args:
        obj: 需要序列化的对象实例（应被 @capture_init_args 修饰）

    Returns:
        包含 class 和 init_args 的 JSON 字符串
    """
    class_name = obj.__class__.__name__
    init_args = getattr(obj, "_init_args", {})
    serializable_args = _encode_value(init_args)
    return json.dumps({"class": class_name, "init_args": serializable_args}, indent=2)


def loads_object_config(
    s: str,
    get_class_fn: Callable[[str], Type[T]],
    override_args: Optional[Dict[str, Any]] = None,
) -> T:
    """Instantiate an object from a JSON string produced by dumps_object_config.
    从 dumps_object_config 生成的 JSON 字符串中实例化对象。

    与 load_object 类似，但输入为 JSON 字符串而非文件路径。

    Args:
        s: dumps_object_config 生成的 JSON 字符串
        get_class_fn: 从注册表中按名称解析类的函数
        override_args: 可选的参数覆盖字典

    Returns:
        根据配置实例化的对象
    """
    # 解析 JSON 字符串
    payload = json.loads(s)
    # 提取类名和编码后的初始化参数
    class_name = payload["class"]
    encoded_args = payload.get("init_args", {})
    # 递归解码参数
    init_args = _decode_value(encoded_args)
    # 可选的参数覆盖
    if override_args:
        init_args.update(override_args)
    # 从注册表查找类并实例化
    cls = get_class_fn(class_name)
    return cls(**init_args)


# Model Registry System (case-insensitive for backward compatibility)
# 投影器（Projector）注册表系统
# 使用大小写不敏感模式，以便向后兼容不同命名风格（如 "LinearProjector" vs "linearprojector"）
# PROJECTOR_REGISTRY: 存储所有注册的投影器类，键为类名（含小写副本），值为类对象
# register_model: 注册装饰器，投影器类通过 @register_model 注册到此注册表中
# get_projector_class: 按名称查找投影器类的函数
PROJECTOR_REGISTRY, register_model, get_projector_class = create_registry(
    "projector", case_insensitive=True
)