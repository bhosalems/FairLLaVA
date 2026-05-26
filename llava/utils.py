import datetime
import logging
import logging.handlers
import os
import sys
import json

import requests

from llava.constants import LOGDIR

handler = None

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def data_loader_default(data_path):
    logging.info("using the default loader.")
    dataset = json.load(open(data_path, "r"))
    logging.info(f"loaded {len(dataset)} samples.")
    return dataset


def data_loader_mimic_cxr_all_frontal_findings(data_path):
    logging.info("using the MIMIC-CXR loader: all frontal findings.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d['view'] in ('AP', 'PA'):
            ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


def data_loader_mimic_cxr_all_views_findings(data_path):
    logging.info("using the MIMIC-CXR loader: all views findings.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        view = d["view"] if isinstance(d["view"], str) else "Unknown"
        d["conversations"][0]["value"] = f"<image>\nGiven the chest X-ray image from {view} view, describe the findings in the image: "
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


def data_loader_mimic_reason_findings(data_args=None):
    data_path = data_args.data_path
    split = data_args.split
    logging.info(f"using the MIMIC-CXR loader: MIMIC {split}.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        if split == 'test' and d['generate_method'] != 'rule-based':
            continue
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d['view'] not in ('AP', 'PA'):
            continue
        if data_args.fairness_labels:
            if d['anchor_age_group'] is None or d['gender'] is None or d['race_major'] is None:
                continue
        if d['image'].startswith("mimic/"):
            d['image'] = d['image'][len('mimic/'):]
        if d['reason'] is not None:
            reason = d['reason'].replace('\n', ' ')
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image given the following indication: {reason}"
        else:
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image."
        ret.append(d) 
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret

def data_loader_iam(data_args=None):
    # Using stanford AIMI's PadChest dataset which only considers PA images
    data_path = data_args.data_path
    split = data_args.split
    logging.info(f"using the IAM loader: IAM {split}.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        if data_args.fairness_labels:
            if d['WritingType'] is None or d['Gender'] is None:
                continue
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret

def data_loader_padchest_reason_findings(data_args=None):
    # Using stanford AIMI's PadChest dataset which only considers PA images
    data_path = data_args.data_path
    split = data_args.split
    logging.info(f"using the PadChest loader: PadChest {split}.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not d['FINDINGS']:
            continue
        if data_args.fairness_labels:
            if d['anchor_age_group'] is None or d['gender'] is None:
                continue
        if d['INDICATION'] is not None:
            indication = d['INDICATION'].replace('\n', ' ')
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image given the following indication: {indication}"
        else:
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image."
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret

def set_split_inplace(data_args, split: str):
    data_args.split = split
    return data_args


def data_loader_ham10000_qa(data_args=None):
    """Loader for HAM10000 QA-style JSON converted to LLaVA conversations.

    Expects each sample to have:
      - image: relative path under data_args.image_folder
      - conversations: list[{from,value}] with first human message containing <image>
      - gender (optional), anchor_age_group (optional), race_major (optional)
    """
    data_path = data_args.data_path
    split = getattr(data_args, "split", "train")
    logging.info(f"using the HAM10000 QA loader: HAM10000 {split}.")
    with open(data_path) as f:
        dataset = json.load(f)

    ret = []
    for d in dataset:
        if "conversations" not in d or not d["conversations"]:
            continue
        # Ensure image token exists in human prompt
        try:
            if d["conversations"][0].get("from") == "human" and "<image>" not in d["conversations"][0].get("value", ""):
                d["conversations"][0]["value"] = "<image>\n" + str(d["conversations"][0].get("value", ""))
        except Exception:
            pass

        # Normalize image path: allow either 'HAM10000/...' or relative under image_folder
        if "image" in d and isinstance(d["image"], str) and d["image"].startswith("HAM10000/"):
            d["image"] = d["image"][len("HAM10000/"):]

        if data_args.fairness_labels:
            if d.get("anchor_age_group") is None or d.get("gender") is None:
                continue

        ret.append(d)

    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret
    
data_loaders = {
    "default": data_loader_default,
    "mimic_train_findings": lambda data_args: data_loader_mimic_reason_findings(set_split_inplace(data_args, "train")),
    "mimic_test_findings": lambda data_args: data_loader_mimic_reason_findings(set_split_inplace(data_args, "test")),
    "mimic_cxr_all_frontal_findings": data_loader_mimic_cxr_all_frontal_findings,
    "mimic_cxr_all_views_findings": data_loader_mimic_cxr_all_views_findings,
    "padchest_train_findings": lambda data_args: data_loader_padchest_reason_findings(set_split_inplace(data_args, "train")),
    "padchest_test_findings": lambda data_args: data_loader_padchest_reason_findings(set_split_inplace(data_args, "test")),
    "ham10000_train_qa": lambda data_args: data_loader_ham10000_qa(set_split_inplace(data_args, "train")),
    "ham10000_test_qa": lambda data_args: data_loader_ham10000_qa(set_split_inplace(data_args, "test")),
    "iam_train": lambda data_args: data_loader_iam(set_split_inplace(data_args, "train")),
    "iam_test": lambda data_args: data_loader_iam(set_split_inplace(data_args, "test")),
}