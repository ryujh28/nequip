""" Train a network."""
import logging
import argparse
import warnings

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from os.path import isdir
from pathlib import Path

import torch

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.utils import load_file
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.utils.versions import check_code_version
from nequip.utils._global_options import _set_global_options
from nequip.scripts._logger import set_up_script_logger

default_config = dict(
    root="./",
    run_name="NequIP",
    wandb=False,
    wandb_project="NequIP",
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "ForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
)


def main(args=None, running_as_script: bool = True):
    """
    변수는 두개
        args는 명령행 인수
        running_as_script는 bool
        프로그램이 스크립트로 실행되는지, 프로그램 안에서 실행되는지 확인하는 변수임
    """

    # parse_command_line 함수를 이용해서 명령행 인수를 분석함. 
    config = parse_command_line(args)
    
    # 만약 running_as_script가 True라면 (default)
    # 프로그램이 스크립트라면 logger 를 붙여줌
    if running_as_script:
        # 
        set_up_script_logger(config.get("log", None), config.verbose)

    # restart file이 있는지 확인, 파일은 있는데 config에 append하지 말라고 써있으면 에러 발생
    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not config.append:
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )

    # for fresh new train
    # 새로운 train이라면
    if not found_restart_file:
        trainer = fresh_start(config)
    else:
        trainer = restart(config)

    # Train
    trainer.save()
    trainer.train()

    return


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Train (or restart training of) a NequIP model."
    )
    parser.add_argument(
        "config", help="YAML file configuring the model, dataset, and other options"
    )
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training on n (default 1) random frames from the dataset",
        const=1,
        type=int,
        nargs="?",
    )
    parser.add_argument(
        "--model-debug-mode",
        help="enable model debug mode, which can sometimes give much more useful error messages at the cost of some speed. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "--grad-anomaly-mode",
        help="enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "--log",
        help="log file to store all the screen logging",
        type=Path,
        default=None,
    )
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    for flag in ("model_debug_mode", "equivariance_test", "grad_anomaly_mode"):
        config[flag] = getattr(args, flag) or config[flag]

    return config


def fresh_start(config):
    # we use add_to_config cause it's a fresh start and need to record it

    # 버전 관리
    check_code_version(config, add_to_config=True)

    # 전역 옵션 초기화 함수
    _set_global_options(config)

    # = Make the trainer =
    # wandb 관련 옵션
    if config.wandb:
        import wandb  # noqa: F401
        from nequip.train.trainer_wandb import TrainerWandB

        # download parameters from wandb in case of sweeping
        from nequip.utils.wandb import init_n_update

        config = init_n_update(config)

        trainer = TrainerWandB(model=None, **dict(config))
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer(model=None, **dict(config))

    # what is this
    # to update wandb data?

    # trainer 객체의 매개변수를 설정 정보에 업데이트함 (왜?)
    config.update(trainer.params)

    # = Load the dataset =
    # dataset_from_config 부터 dataset instance를 불러옴.
    # class이름은 아마 AtomicData 일듯?
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None

    # = Train/test split =
    # trainer에 dataset을 붙여줌
    trainer.set_dataset(dataset, validation_dataset)

    # = Build model =
    # 모델을 설정 정보 기반으로 빌드하고 초기화함
    final_model = model_from_config(
        config=config, initialize=True, dataset=trainer.dataset_train
    )
    logging.info("Successfully built the network...")

    # by doing this here we check also any keys custom builders may have added
    # 이전 버전의 설정 키를 확인함
    _check_old_keys(config)

    # Equivar test
    # Equivariance를 테스트함
    if config.equivariance_test > 0:
        n_train: int = len(trainer.dataset_train)
        assert config.equivariance_test <= n_train
        final_model.eval()
        indexes = torch.randperm(n_train)[: config.equivariance_test]
        errstr = assert_AtomicData_equivariant(
            final_model, [trainer.dataset_train[i] for i in indexes]
        )
        final_model.train()
        logging.info(
            "Equivariance test passed; equivariance errors:\n"
            "   Errors are in real units, where relevant.\n"
            "   Please note that the large scale of the typical\n"
            "   shifts to the (atomic) energy can cause\n"
            "   catastrophic cancellation and give incorrectly\n"
            "   the equivariance error as zero for those fields.\n"
            f"{errstr}"
        )
        del errstr, indexes, n_train

    # Set the trainer
    # 트레이너 객체에 최종 모델을 할당함
    trainer.model = final_model

    # Store any updated config information in the trainer
    trainer.update_kwargs(config)

    return trainer


def restart(config):
    # load the dictionary
    restart_file = f"{config.root}/{config.run_name}/trainer.pth"
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=restart_file,
        enforced_format="torch",
    )

    # compare dictionary to config and update stop condition related arguments
    for k in config.keys():
        if config[k] != dictionary.get(k, ""):
            if k == "max_epochs":
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif k.startswith("early_stop"):
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif isinstance(config[k], type(dictionary.get(k, ""))):
                raise ValueError(
                    f'Key "{k}" is different in config and the result trainer.pth file. Please double check'
                )

    # note, "trainer.pth"/dictionary also store code versions,
    # which will not be stored in config and thus not checked here
    check_code_version(config)

    # recursive loop, if same type but different value
    # raise error

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    # dtype, etc.
    _set_global_options(config)

    # note, the from_dict method will check whether the code version
    # in trainer.pth is consistent and issue warnings
    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB
        from nequip.utils.wandb import resume

        resume(config)
        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer.from_dict(dictionary)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully re-loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully re-loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None
    trainer.set_dataset(dataset, validation_dataset)

    return trainer


def _check_old_keys(config) -> None:
    """check ``config`` for old/depricated keys and emit corresponding errors/warnings"""
    # compile_model
    k = "compile_model"
    if k in config:
        if config[k]:
            raise ValueError("the `compile_model` option has been removed")
        else:
            warnings.warn("the `compile_model` option has been removed")


if __name__ == "__main__":
    main(running_as_script=True)
