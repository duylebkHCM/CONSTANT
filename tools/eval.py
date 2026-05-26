import torch
from pathlib import Path
from loguru import logger
import argparse
from src.metrics.fid import FIDScore
from src.metrics.hwd import compute_hwd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--real_dir",
    required=True,
    help="Path to groundtruth data, located at <ckpt_path_parent>/eval/<eval_name>/evaluate_images/real",
)
parser.add_argument(
    "--fake_dir",
    required=True,
    help="Path to groundtruth data, located at <ckpt_path_parent>/eval/<eval_name>/evaluate_images/gen",
)
parser.add_argument("--batchsize", default=64, help="Reduce evaluation time by increasing batch size")
parser.add_argument("--metrics", nargs="+", default=["fid"], choices=["fid", "hwd"])
parser.add_argument(
    "--mode",
    choices=["global", "by_writer"],
    default="global",
    help="Either evaluate on style-agnostic (global) or on style-specific (by_writer)",
)
parser.add_argument("--device", default="cuda", help="Running device")
args = parser.parse_args()

wids = list(Path(args.real_dir).glob("*"))
wids = [p.stem for p in wids]

logger.add(
    Path(args.fake_dir).parent.joinpath(args.log_name),
    format="{time} {level} {message}",
    level="INFO",
)
device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

if args.mode == "global":
    for metric in args.metrics:
        if metric == "fid":
            fid_scorer = FIDScore(batch_size=args.batchsize, dims=2048, device="cuda")
            global_fid = fid_scorer.compute_scores(Path(args.real_dir).as_posix(), Path(args.fake_dir).as_posix())

            logger.info("Global FID: {}", global_fid)

else:
    for metric in args.metrics:
        if metric == "fid":
            fid_scorer = FIDScore(batch_size=args.batchsize, dims=2048, device="cuda")
            mean_fid = 0
            for wid in wids:
                logger.info("-" * 30)
                logger.info("WID: {}", wid)
                fid = fid_scorer.compute_scores(
                    Path(args.real_dir).joinpath(wid).as_posix(),
                    Path(args.fake_dir).joinpath(wid).as_posix(),
                )
                logger.info("wid {} has FID {}", wid, fid)
                mean_fid += fid

            logger.info("Average FID score: {}", mean_fid / len(wids))
        elif metric == "hwd":
            result = compute_hwd(args.fake_dir, args.real_dir, args.batchsize)
            final_hwd = result.pop("final")
            for wid in result:
                logger.info("-" * 30)
                logger.info("WID: {}", wid)
                logger.info("wid {} has HWD {}", wid, result[wid])
            logger.info("Average HWD score: {}", final_hwd)
