import inspect
from importlib import import_module

from nequip import data
from nequip.data.transforms import TypeMapper
from nequip.data import AtomicDataset, register_fields
from nequip.utils import instantiate, get_w_prefix


def dataset_from_config(config, prefix: str = "dataset") -> AtomicDataset:
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py

    Args:

    config (dict, nequip.utils.Config): dict/object that store all the parameters
    prefix (str): Optional. The prefix of all dataset parameters

    Return:

    dataset (nequip.data.AtomicDataset)
    """

    
    # 설정 파일에서 주어진 prefix에 대한 설정을 읽어옴. 없으면 None을 반환
    config_dataset = config.get(prefix, None)

    # 만약 None이라면 KeyError 발생!
    if config_dataset is None:
        raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")
        
    # config_dataset이 class인지 확인함
    if inspect.isclass(config_dataset):
        # user define class
        # 맞으면 class_name 변수에 할당함
        class_name = config_dataset

    # class가 아니라면
    else:
        try:
            # 문자열 마지막을 빼고 모듈 이름을 추출함
            module_name = ".".join(config_dataset.split(".")[:-1])
            # 문자열 마지막을 빼고 클래스 이름을 추출함
            class_name = ".".join(config_dataset.split(".")[-1:])
            # 먼저 import_module(module_name) 실행
            # 모듈 이름을 사용해서 모듈을 가져옴
            # 가져온 모듈에서 클래스 이름에 해당하는 클래스를 져옴
            class_name = getattr(import_module(module_name), class_name)
        except Exception:
            # 실패한다면?  기본 데이터셋 클래스를 찾아봄
            # ^ TODO: don't catch all Exception
            # default class defined in nequip.data or nequip.dataset
            dataset_name = config_dataset.lower()

            class_name = None
            for k, v in inspect.getmembers(data, inspect.isclass):
                if k.endswith("Dataset"):
                    if k.lower() == dataset_name:
                        class_name = v
                    if k[:-7].lower() == dataset_name:
                        class_name = v
                elif k.lower() == dataset_name:
                    class_name = v
    # 클래스 이름을 찾지 못한 경우 에러를 발생시킴
    if class_name is None:
        raise NameError(f"dataset type {dataset_name} does not exists")

    # if dataset r_max is not found, use the universal r_max

    # eff_key와 prefixed_eff_key 설정
    eff_key = "extra_fixed_fields"
    prefixed_eff_key = f"{prefix}_{eff_key}"

    # get_w_prefix를 이용해서 config에서 "prefix_extra_fixed_field"에 해당하는 설정을 가져옴
    config[prefixed_eff_key] = get_w_prefix(
        eff_key, {}, prefix=prefix, arg_dicts=config
    )

    # 그 설정에다가 "r_max" 필드를 추가하고 설정값을 가져와서 설정함
    config[prefixed_eff_key]["r_max"] = get_w_prefix(
        "r_max",
        prefix=prefix,
        arg_dicts=[config[prefixed_eff_key], config],
    )

    # Build a TypeMapper from the config
    # type_mapper 인스턴스를 생성함. 
    # instantiate 함수를 사용해서 TypeMapper 클래스를 인스턴스화
    # TypeMapper객체는 데이터셋의 필드 유형을 매핑할 떄 사용되는듯
    type_mapper, _ = instantiate(TypeMapper, prefix=prefix, optional_args=config)

    # Register fields:
    # This might reregister fields, but that's OK:
    instantiate(register_fields, all_args=config)

    
    # 데이터셋 인스턴스를 만들어줌.
    # class_name을 사용하여 데이터셋 클래스를 인스턴스화 함
    # 아마 여기서 dataset은 "~~~~.xyz" 이런 식일 것임
    instance, _ = instantiate(
        class_name,
        prefix=prefix,
        positional_args={"type_mapper": type_mapper},
        optional_args=config,
    )

    return instance
