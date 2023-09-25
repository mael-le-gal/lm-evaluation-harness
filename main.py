import argparse
import json
import logging
import os
import pandas as pd
from datetime import datetime
from lm_eval import tasks, evaluator, utils

logging.getLogger("openai").setLevel(logging.WARNING)


def save_parquet(args, results, task_names):
    all_records = []
    for task in task_names:
        config = results['config']['description_dict']
        version = results['versions'][task]
        for metric_name, metric_value in results['results'][task].items():
            record = {
                'run_id': args.run_id,
                'task': task,
                'task_version': version,
                'task_few_shot': results['config']['num_fewshot'],
                'task_limit': results['config']['limit'],
                'metric_name': metric_name,
                'metric_value': metric_value,
            }
            llmevha_sha = config.get('llmevha_sha')
            if llmevha_sha is not None:
                record['llmevha_sha'] = llmevha_sha
            if args.model == 'tgi':
                record.update({
                    'tgi_model_id': config.get('model_id'),
                    'tgi_model_dtype': config.get('model_dtype'),
                    'tgi_max_input_length': config.get('max_input_length'),
                    'tgi_max_total_tokens': config.get('max_total_tokens'),
                    'tgi_version': config.get('version'),
                })
            all_records.append(record)
    df = pd.DataFrame.from_records(all_records)
    # Save into parquet
    import pathlib
    output_base_path = args.output_base_path
    output_base_path = (pathlib.Path(output_base_path) if output_base_path is not None else pathlib.Path("."))
    try:
        output_base_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    df.to_parquet(os.path.join(output_base_path, f'{args.run_id}.parquet'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--save_parquet", action="store_true", default=False)
    parser.add_argument("--run_id", type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"))

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))

    if args.save_parquet:
        save_parquet(args, results, task_names)


if __name__ == "__main__":
    main()
