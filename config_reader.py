import copy
import multiprocessing as mp


def process_configs(target, arg_parser):
    args, _ = arg_parser.parse_known_args()
    ctx = mp.get_context("spawn")

    for run_args, _run_config, _run_repeat in _yield_configs(arg_parser, args):
        process = ctx.Process(target=target, args=(run_args,))
        process.start()
        process.join()


def _read_config(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    runs = []
    run = [1, dict()]

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("#"):
            continue

        if not stripped_line:
            if run[1]:
                runs.append(run)
            run = [1, dict()]
            continue

        if stripped_line.startswith("[") and stripped_line.endswith("]"):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split("=", maxsplit=1)
            key, value = key.strip(), value.strip()
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs


def _convert_config(config):
    config_list = []

    for key, value in config.items():
        value_lower = value.lower()

        if value_lower == "true":
            config_list.append("--" + key)
        elif value_lower != "false":
            config_list.extend(["--" + key] + value.split(" "))

    return config_list


def _yield_configs(arg_parser, args, verbose=True):
    _print = print if verbose else (lambda x: x)

    if args.config:
        config = _read_config(args.config)

        for run_repeat, run_config in config:
            print("-" * 50)
            print("Config:")
            print(run_config)

            args_copy = copy.deepcopy(args)
            config_list = _convert_config(run_config)
            run_args = arg_parser.parse_args(config_list, namespace=args_copy)
            run_args_dict = vars(run_args)

            # Explicitly set false boolean values from configuration files.
            for key, value in run_config.items():
                if value.lower() == "false":
                    run_args_dict[key] = False

            print(f"Repeat {run_repeat} times")
            print("-" * 50)

            for iteration in range(run_repeat):
                _print(f"Iteration {iteration}")
                _print("-" * 50)
                yield run_args, run_config, run_repeat
    else:
        yield args, None, None
