#!/usr/bin/env python3
import errno
import logging
import os
import sys

import aws_cdk as cdk
import yaml
from lsmresources_stack import LSMResourcesStack


def configure_logging(verbose: bool) -> logging.Logger:
    """
    Setup logger.
    :param verbose: setup verbose level
    :return: logger
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s",
        level=log_level,
        stream=sys.stdout,
    )
    return logging.getLogger(__name__)


def get_proj_config(cfg_file) -> dict:
    """
    Load configuration file.
    :param cfg_file: location of the cfg file
    :return: dict of all the keys in yaml file
    """
    environs = {}
    try:
        logger.info("Reading configuration file : %s" % cfg_file)
        with open(cfg_file, "rb") as f:
            environs = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        logger.error("Failed to read : %s" % cfg_file)
        raise Exception(e)

    return environs


def validate_cfg_file_path(config_file):
    """
    Load configuration file.
    :param cfg_file: location of the cfg file to validate
    """
    if not config_file:
        logger.error(
            "Please provide the missing configuration file, check readme for help."
        )
        sys.exit(-1)

    logger.info("Validating yaml file %s", config_file)
    if not os.path.exists(config_file):
        logger.error("YAML file not found")
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)


if __name__ == "__main__":

    logger = configure_logging(verbose=False)

    app = cdk.App()

    cfg_file = app.node.try_get_context("cfg")

    # if this argument not found in cdk context, inform user to provide valid configuration file.
    validate_cfg_file_path(cfg_file)

    # Get Project Configuration
    proj_config = get_proj_config(cfg_file)
    environs = proj_config.get("environments")

    env = app.node.try_get_context("env")
    prefix = proj_config.get("prefix")
    repo = proj_config.get("repository")

    if env not in environs.keys():
        logger.error(
            "Inappropriate environment selection. Allowed values are : %s"
            % list(environs.keys())
        )
        sys.exit(-1)

    logger.info(
        "Preparing a dict to be passed as additional arguments to the CDK pipeline"
    )

    selected_environ = environs.get(env)
    del proj_config["environments"]
    proj_config.update(selected_environ)
    proj_config.update({"env": env})

    # retriving environment specific tags
    tag_env = proj_config.get("tag_env")

    logger.info(proj_config)

    LSMResourcesStack(
        app,
        f"{prefix}-{repo}-{env}",
        proj_config,
        tags={
            "Product": "New Solutions - Data Science",
            "Environment": f"{tag_env}",
            "Cost Center": "Engineering",
        },
    )

    app.synth()
