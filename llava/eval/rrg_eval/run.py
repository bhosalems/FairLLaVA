from typing import List
import os
import json 
import random
import shutil

import evaluate
import pandas as pd
import numpy as np
from pyparsing import line
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import sys

if __package__ in (None, ""):
    # Running as a script: add repo root to sys.path and use absolute imports
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, REPO_ROOT)
    import llava.eval.rrg_eval.rrg_eval as rrgeval
    from llava.eval.rrg_eval.rrg_eval.f1radgraph import F1RadGraphv2, bootstrap_confidence_interval
    from llava.eval.rrg_eval.rrg_eval.factuality_utils import CONDITIONS
    from llava.eval.rrg_eval.rrg_eval.green_scorev2 import Green
else:
    # Running as a module: relative imports work
    from . import rrg_eval as rrgeval
    from .rrg_eval.f1radgraph import F1RadGraphv2, bootstrap_confidence_interval
    from .rrg_eval.factuality_utils import CONDITIONS
    from .rrg_eval.green_scorev2 import Green

try:
    import wandb as wandb_lib
except ImportError:
    wandb_lib = None


random.seed(3)
np.random.seed(3)


def bleu4(predictions, references, bootstrap_ci: bool = False):
    if len(predictions) == 0:
        if bootstrap_ci:
            return {"median": float("nan"), "ci_l": float("nan"), "ci_h": float("nan")}
        return float("nan")
    if bootstrap_ci:
        if len(predictions) < 2:
            # Too few samples for bootstrap; fall back to a degenerate CI.
            score = BLEU().corpus_score(hypotheses=predictions, references=[references]).score
            return {"median": score, "ci_l": score, "ci_h": score}
        ret = BLEU().corpus_score(hypotheses=predictions, references=[references], n_bootstrap=500)
        return {"median": ret.score, "ci_l": ret._mean - ret._ci, "ci_h": ret._mean + ret._ci}
    else:
        return evaluate.load("bleu").compute(predictions=predictions, references=references)["bleu"]


def bleu1(predictions, references, bootstrap_ci: bool = False):
    if len(predictions) == 0:
        if bootstrap_ci:
            return {"median": float("nan"), "ci_l": float("nan"), "ci_h": float("nan")}
        return float("nan")
    if bootstrap_ci:
        if len(predictions) < 2:
            # Too few samples for bootstrap; fall back to a degenerate CI.
            score = BLEU(max_ngram_order=1).corpus_score(hypotheses=predictions, references=[references]).score
            return {"median": score, "ci_l": score, "ci_h": score}
        ret = BLEU(max_ngram_order=1).corpus_score(hypotheses=predictions, references=[references], n_bootstrap=500)
        return {"median": ret.score, "ci_l": ret._mean - ret._ci, "ci_h": ret._mean + ret._ci}
    else:
        return evaluate.load("bleu").compute(predictions=predictions, references=references, max_order=1)["bleu"]


def rougel(predictions, references, bootstrap_ci: bool = False):
    if len(predictions) == 0:
        if bootstrap_ci:
            return {"median": float("nan"), "ci_l": float("nan"), "ci_h": float("nan")}
        return float("nan")
    if bootstrap_ci:
        return rrgeval.rouge.compute(predictions, references, ["rougeL"])["rougeL"]
    else:
        return evaluate.load("rouge").compute(predictions=predictions, references=references)["rougeL"]


def rouge2(predictions, references, bootstrap_ci: bool = False):
    if len(predictions) == 0:
        if bootstrap_ci:
            return {"median": float("nan"), "ci_l": float("nan"), "ci_h": float("nan")}
        return float("nan")
    if bootstrap_ci:
        return rrgeval.rouge.compute(predictions, references, ["rouge2"])["rouge2"]
    else:
        return evaluate.load("rouge").compute(predictions=predictions, references=references)["rouge2"]


def bertscore(predictions, references):
    if len(predictions) == 0:
        return float("nan")
    return evaluate.load("bertscore").compute(predictions=predictions, references=references)["f1"]


def radgraph(predictions, references, bootstrap_ci: bool = False):
    if len(predictions) == 0:
        if bootstrap_ci:
            return {"median": float("nan"), "ci_l": float("nan"), "ci_h": float("nan")}
        return float("nan")
    if bootstrap_ci:
        reward_list = F1RadGraphv2(reward_level="partial", batch_size=1)(hyps=predictions, refs=references)[1]
        if len(reward_list) < 2:
            # SciPy bootstrap requires >=2 observations. For tiny groups, report a degenerate CI.
            med = float(np.median(reward_list)) if len(reward_list) else float("nan")
            return {"median": med, "ci_l": med, "ci_h": med}
        bs = bootstrap_confidence_interval(reward_list, n_resamples=500)
        return {
            "median": np.median(bs.bootstrap_distribution),
            "ci_l": bs.confidence_interval.low,
            "ci_h": bs.confidence_interval.high,
        } 
    else:
        return F1RadGraphv2(reward_level="partial", batch_size=1)(hyps=predictions, refs=references)[0]


def chexbert(predictions, references, bootstrap_ci: bool = False):
    if len(predictions) == 0:
        if bootstrap_ci:
            return []
        return float("nan")
    return rrgeval.chexbert.evaluate(predictions, references, include_original=False, bootstrap_ci=bootstrap_ci)

def greenscore(predictions, references, bootstrap_ci: bool = False, results_file=None):
    if len(predictions) == 0:
        if bootstrap_ci:
            return {"median": float("nan"), "ci_l": float("nan"), "ci_h": float("nan"), "greenscore": []}
        return {"mean": float("nan"), "greenscore": []}
    reuse_results = False
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(results_file)
    if results_file is not None:
        with open(results_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)    
                except json.JSONDecodeError:
                    continue
                if "greenscore" in rec:        
                    reuse_results = True
                else:
                    break
    if not reuse_results:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29510" # might needto change if running multiple jobs on one machine
        os.environ["RANK"]         = "0"
        os.environ["WORLD_SIZE"]   = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # SciPy bootstrap (used inside GREEN) requires >=2 observations.
        use_agg = bool(bootstrap_ci and len(predictions) >= 2)
        scorer = Green(use_aggregator=use_agg)
        result = scorer.compute(predictions, references, results_file=results_file)
    else:
        print(f"===========Reuse existing results: {reuse_results} from {results_file}===============")
        with open(results_file, "r") as f:
            green_score_list = [
                float(json.loads(line)["greenscore"])
                for line in f if line.strip()
            ]
        bs = None
        if bootstrap_ci and len(green_score_list) >= 2:
            bs = Green.bootstrap_confidence_interval(green_score_list)
        mean = np.mean(green_score_list)
        std = np.std(green_score_list)
        summary, result_df = None, None
        result = {
            'mean': mean,
            'std': std,
            'summary': summary,
            'result_df': result_df,
            'bs': bs if bootstrap_ci else None,
            'greenscore':green_score_list
        }
        
    if bootstrap_ci:
        # If we couldn't bootstrap (e.g., tiny subgroup), fall back to a degenerate CI.
        gs_list = result.get('greenscore') or []
        med = float(np.median(gs_list)) if len(gs_list) else float('nan')
        bs_obj = result.get('bs')
        if bs_obj is None:
            return {"median": med, "ci_l": med, "ci_h": med, "greenscore": gs_list}
        return {"median": np.median(bs_obj.bootstrap_distribution), "ci_l": bs_obj.confidence_interval.low, "ci_h": bs_obj.confidence_interval.high, "greenscore": gs_list}
    else:
        return {"mean": result['mean'], "greenscore": result['greenscore']}

SCORER_NAME_TO_CLASS = {
    "ROUGE-L": rougel,
    "ROUGE-2": rouge2,
    "BLEU-4": bleu4,
    "BLEU-1": bleu1,
    "BERTScore": bertscore,
    "F1-RadGraph": radgraph,
    "CheXbert": chexbert,
    "GreenScore": greenscore,
}


class ReportGenerationEvaluator:
    def __init__(self, scorers=['CheXbert'], bootstrap_ci: bool = False):
        self.bootstrap_ci = bootstrap_ci
        self.scorers = {}
        
        for scorer_name in scorers:
            if scorer_name in SCORER_NAME_TO_CLASS:
                if scorer_name in SCORER_NAME_TO_CLASS: 
                    self.scorers[scorer_name] = SCORER_NAME_TO_CLASS[scorer_name]  
                else:
                    raise NotImplementedError(f'scorer of type {scorer_name} not implemented')

    def evaluate(self, predictions, references, results_file=None):
        assert len(predictions) == len(references), f'Length of predictions (i.e. generations) {len(predictions)} and references (i.e. ground truths) {len(references)} must match.'
        
        scores = {}
        
        for scorer_name, scorer in (pbar := tqdm(self.scorers.items())):
            pbar.set_description(scorer_name)
            if scorer_name == "GreenScore":
                scorer_scores = scorer(predictions, references, self.bootstrap_ci, results_file=results_file)
            else:
                scorer_scores = scorer(predictions, references, self.bootstrap_ci)
            scores[scorer_name] = scorer_scores
            
        self.postprocess_eval(scores)
        return scores

    def postprocess_eval(self, scores):
        if self.bootstrap_ci:
            keys = ("median", "ci_l", "ci_h")
            for name in list(scores.keys()):
                if name == "CheXbert":
                    metrics = scores.pop(name)
                    if not isinstance(metrics, list) or len(metrics) < 4:
                        nan_ci = {k: float("nan") for k in keys}
                        scores["Micro-F1-14"] = dict(nan_ci)
                        scores["Macro-F1-14"] = dict(nan_ci)
                        scores["Micro-F1-5"] = dict(nan_ci)
                        scores["Macro-F1-5"] = dict(nan_ci)
                        scores["Micro-F1-14+"] = dict(nan_ci)
                        scores["Macro-F1-14+"] = dict(nan_ci)
                        scores["Micro-F1-5+"] = dict(nan_ci)
                        scores["Macro-F1-5+"] = dict(nan_ci)
                        continue
                    scores["Micro-F1-14"] = {k: metrics[0]["micro avg"][k] for k in keys}
                    scores["Macro-F1-14"] = {k: metrics[0]["macro avg"][k] for k in keys}
                    scores["Micro-F1-5"] = {k: metrics[1]["micro avg"][k] for k in keys}
                    scores["Macro-F1-5"] = {k: metrics[1]["macro avg"][k] for k in keys}
                    scores["Micro-F1-14+"] = {k: metrics[2]["micro avg"][k] for k in keys}
                    scores["Macro-F1-14+"] = {k: metrics[2]["macro avg"][k] for k in keys}
                    scores["Micro-F1-5+"] = {k: metrics[3]["micro avg"][k] for k in keys}
                    scores["Macro-F1-5+"] = {k: metrics[3]["macro avg"][k] for k in keys}
                    scores["breakdown-"] = metrics[0]
                    scores["breakdown+"] = metrics[2]
                    scores["chexbert_metrics"] = metrics[-1]
                elif name == "F1-RadGraph":
                    scores["F1-RadGraph"] = scores.pop(name)
        else:
            for name in list(scores.keys()):
                if name == "CheXbert":
                    metrics = scores.pop(name)
                    if not isinstance(metrics, list) or len(metrics) < 4:
                        nan = float("nan")
                        scores["Micro-F1-14"] = nan
                        scores["Macro-F1-14"] = nan
                        scores["Micro-F1-5"] = nan
                        scores["Macro-F1-5"] = nan
                        scores["Micro-F1-14+"] = nan
                        scores["Macro-F1-14+"] = nan
                        scores["Micro-F1-5+"] = nan
                        scores["Macro-F1-5+"] = nan
                        continue
                    scores["Micro-F1-14"] = metrics[0]["micro avg"]["f1-score"]
                    scores["Macro-F1-14"] = metrics[0]["macro avg"]["f1-score"]
                    scores["Micro-F1-5"] = metrics[1]["micro avg"]["f1-score"]
                    scores["Macro-F1-5"] = metrics[1]["macro avg"]["f1-score"]
                    scores["Micro-F1-14+"] = metrics[2]["micro avg"]["f1-score"]
                    scores["Macro-F1-14+"] = metrics[2]["macro avg"]["f1-score"]
                    scores["Micro-F1-5+"] = metrics[3]["micro avg"]["f1-score"]
                    scores["Macro-F1-5+"] = metrics[3]["macro avg"]["f1-score"]
                    scores["breakdown-"] = metrics[0]
                    scores["breakdown+"] = metrics[2]
                    scores["chexbert_metrics"] = metrics[-1]
                elif name == "F1-RadGraph":
                    scores["F1-RadGraph"] = scores.pop(name)["f1-radgraph"]


def test_evaluator():
    generations = [
        "Totally unrelated.",
        'Lungs and pleural spaces are clear. Cardiomediastinal contour is normal.',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour'
    ]

    ground_truths = [
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour'
    ]
    
    evaluator = ReportGenerationEvaluator()
    print(evaluator.evaluate(generations, ground_truths))

    return


def main(
        filepath: str,
        scorers: List = None,
        report_chexbert_f1: bool = False,
        bootstrap_ci: bool = True,
        wandb_log: bool = False,
        output_dir: str = "./",
        run_name: str = "mimic_cxr_eval",
    ):
    with open(filepath) as f:
        preds, refs, rows = [], [], []
        for line_no, l in enumerate(f, start=1):
            s = l.strip()
            if not s:
                # Some pipelines may accidentally write blank lines into JSONL.
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Malformed JSONL in {filepath} at line {line_no}: {e}. "
                    f"Line startswith={s[:80]!r}"
                ) from e
            rows.append(d)
            preds.append(d["prediction"])
            refs.append(d["reference"])

    if scorers is None:
        scorers = [
            'CheXbert',
            'F1-RadGraph',
            'BLEU-1',
            'BLEU-4',
            'ROUGE-L',
            'GreenScore'
        ]

    evaluator = ReportGenerationEvaluator(scorers=scorers, bootstrap_ci=bootstrap_ci)
    results = evaluator.evaluate(preds, refs, filepath)
    
    print("\n")
    print(f"Total reports: {len(preds)}\n")

    print("========== Main Results ==========")
    
    # If input JSONL already contains per-sample greenscore, skip rewriting.
    # Also guard the empty-file case (some subgroup splits can be empty).
    if "GreenScore" in results and rows and "greenscore" not in rows[0]:
        if "greenscore" not in results['GreenScore']:
            raise KeyError("Expected results['GreenScore'] to contain per-sample scores.")

        greenscores = results["GreenScore"]["greenscore"]
        if len(greenscores) != len(rows):
            raise ValueError("Length mismatch between GreenScore list and input rows.")
        
        for d, gs in zip(rows, greenscores):
            d["greenscore"] = gs  # add field; leaves all other previously computed fields untouched

        tmp_path = f"{filepath}.tmp"
        bak_path = f"{filepath}.bak"
        with open(tmp_path, "w") as f:
            for d in rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # keep a backup of the original file, then replace
        shutil.copyfile(filepath, bak_path)
        os.replace(tmp_path, filepath)
        print(f"Added per-sample GreenScore to original file.\nBackup saved at: {bak_path}")
    
    if bootstrap_ci:
        main_results = pd.DataFrame.from_dict({
            k:v for k,v in results.items() if k not in ("breakdown+", "breakdown-", "chexbert_metrics")
        })
        keys = [
            "Micro-F1-14", "Micro-F1-5", "Macro-F1-14", "Macro-F1-5",
            "Micro-F1-14+", "Micro-F1-5+", "Macro-F1-14+", "Macro-F1-5+",
            "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"
        ]
        if "GreenScore" in main_results:
            keys.append("GreenScore")
        print(main_results)
        # print(main_results[keys])
    else:
        main_results = pd.DataFrame.from_dict({k:v for k,v in results.items() if type(v)!= dict}, 'index')
        keys = [
            "Micro-F1-14", "Micro-F1-5", "Macro-F1-14", "Macro-F1-5",
            "Micro-F1-14+", "Micro-F1-5+", "Macro-F1-14+", "Macro-F1-5+",
            "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"
        ]
        if "GreenScore" in main_results:
            keys.append("GreenScore")
        print(main_results)
        # print(main_results.T[keys])
        
    print("")
    if output_dir == "./":
        output_dir = os.path.dirname(filepath)
        
    os.makedirs(output_dir, exist_ok=True)
    main_results.to_csv(os.path.join(output_dir, "main.csv"))
    
    if "GreenScore" in main_results.columns and "greenscore" in main_results.index:
        main_results = main_results.drop(index="greenscore")  # <- remove the extra row, we dont need it as we save greenscore already.
    
    # print("")
    # os.makedirs(output_dir, exist_ok=True)
    # main_results.to_csv(os.path.join(output_dir, "main.csv"))

    if wandb_log:
        if wandb_lib is None:
            raise ImportError("wandb_log=True but wandb is not installed")
        wandb_results = {}
        for metric in main_results.columns:
            for index in main_results.index:
                key = metric
                if isinstance(index, str):
                    key += f"-{index}"
                wandb_results[key] = main_results[metric][index]

        wandb_lib.init(name=run_name, id=run_name)
        wandb_lib.log(wandb_results)
    
    # CheXbert breakdown tables are only available when chexbert metrics were computed.
    # When callers pass scorers that exclude chexbert (e.g., BLEU/RadGraph/GreenScore only),
    # `results` will not contain these keys.
    if "breakdown+" in results and "breakdown-" in results:
        print("========== CheXbert F1 (uncertain as positive) ==========")
        breakdown_p = pd.DataFrame(results["breakdown+"])[sorted(CONDITIONS) + ["micro avg", "macro avg"]].T[
            ["f1-score", "precision", "recall", "support"]
        ]
        print(breakdown_p)
        print("")
        breakdown_p.to_csv(os.path.join(output_dir, "breakdown_p.csv"))

        print("========== CheXbert F1 (uncertain as negative) ==========")
        breakdown_n = pd.DataFrame(results["breakdown-"])[sorted(CONDITIONS) + ["micro avg", "macro avg"]].T[
            ["f1-score", "precision", "recall", "support"]
        ]
        print(breakdown_n)
        print("")
        breakdown_n.to_csv(os.path.join(output_dir, "breakdown_n.csv"))
    else:
        print("(CheXbert breakdown skipped: chexbert not computed for this run)")

    if report_chexbert_f1 and "chexbert_metrics" in results:
        print("========== CheXbert F1 ==========")
        chexbert_df = pd.DataFrame(results["chexbert_metrics"])[sorted(CONDITIONS) + ["avg"]].T[
            ["positive f1", "negation f1", "uncertain f1", "blank f1", "weighted f1", "kappas"]
        ]
        print(chexbert_df)
        print("")
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)