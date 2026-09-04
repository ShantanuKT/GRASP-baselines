"""
Run IRCoT inference with chunks and save predictions to a directory.

Usage:
    python run_questions_with_chunks_pred_dir.py \
    --config base_configs/ircot_qa_gpt5mini_openrouter_qwen2_5_3b_instruct_hotpotqa.jsonnet \
    --questions-json data_new/questions_500.json \
    --chunks-json data_new/chunks.json \
    --index-name hqa \
    --force-reindex \
    --force-predict \
    --pred-dir results_7b/hqa
"""

import argparse
import os
import sys

import run_questions_with_chunks as base


def _extract_pred_dir(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="predictions",
        help="Base directory where prediction artifacts are written.",
    )
    parsed, remaining = parser.parse_known_args(argv)
    return parsed.pred_dir, remaining


def _get_prediction_paths_with_dir(pred_dir: str, config_path: str, evaluation_path: str, prediction_suffix: str):
    from lib import get_config_file_path_from_name_or_path, infer_source_target_prefix

    config_filepath = get_config_file_path_from_name_or_path(config_path)
    experiment_name = os.path.splitext(os.path.basename(config_filepath))[0]
    prediction_directory = os.path.join(pred_dir, experiment_name + prediction_suffix)
    os.makedirs(prediction_directory, exist_ok=True)

    prediction_filename = os.path.splitext(os.path.basename(evaluation_path))[0]
    prediction_filename = infer_source_target_prefix(config_filepath, evaluation_path) + prediction_filename
    prediction_filepath = os.path.join(prediction_directory, "prediction__" + prediction_filename + ".json")

    return {
        "config_filepath": config_filepath,
        "prediction_directory": prediction_directory,
        "prediction_filename": prediction_filename,
        "prediction_filepath": prediction_filepath,
        "chains_filepath": os.path.splitext(prediction_filepath)[0] + "_chains.txt",
        "reasoning_steps_filepath": os.path.splitext(prediction_filepath)[0] + "_reasoning_steps.json",
        "time_taken_filepath": os.path.splitext(prediction_filepath)[0] + "_time_taken.txt",
        "metadata_filepath": os.path.splitext(prediction_filepath)[0] + "_run_metadata.json",
    }


def main() -> None:
    pred_dir, remaining_argv = _extract_pred_dir(sys.argv[1:])

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining_argv]
        args = base.parse_args()
    finally:
        sys.argv = original_argv

    if not os.path.exists(args.questions_json):
        raise FileNotFoundError(f"Questions file not found: {args.questions_json}")
    if not os.path.exists(args.chunks_json):
        raise FileNotFoundError(f"Chunks file not found: {args.chunks_json}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    count = base.convert_questions_to_processed_jsonl(args.questions_json, args.processed_output)
    print(f"Wrote {count} examples to {args.processed_output}")

    base.build_or_reuse_index(
        chunks_json=args.chunks_json,
        index_name=args.index_name,
        elasticsearch_host=args.elasticsearch_host,
        elasticsearch_port=args.elasticsearch_port,
        force_reindex=args.force_reindex,
    )

    if args.skip_predict:
        print("Skipping predict.py as requested (--skip-predict).")
        return

    base.ensure_nltk_dependencies()

    original_get_prediction_paths = base.get_prediction_paths
    base.get_prediction_paths = lambda config_path, evaluation_path, prediction_suffix: _get_prediction_paths_with_dir(
        pred_dir=pred_dir,
        config_path=config_path,
        evaluation_path=evaluation_path,
        prediction_suffix=prediction_suffix,
    )
    try:
        base.run_predict_incremental(
            config_path=args.config,
            evaluation_path=args.processed_output,
            index_name=args.index_name,
            chunks_json=args.chunks_json,
            prediction_suffix=args.prediction_suffix,
            force_predict=args.force_predict,
            silent=args.silent,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            batch_size=args.batch_size,
        )
    finally:
        base.get_prediction_paths = original_get_prediction_paths


if __name__ == "__main__":
    main()
