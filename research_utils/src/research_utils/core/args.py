def print_args(args):
    print(f"\n===========[INFO] ARGS:===========")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")
    print(f"===================================\n")


def print_args_color(args):
    COLOR = "\033[92m"
    RESET = "\033[0m"
    print(f"\n{COLOR}===========[INFO] ARGS:=========== {RESET}")
    for arg in vars(args):
        print(f"{COLOR}- {arg}{RESET}: {getattr(args, arg)}")
    print(f"{COLOR}==================================={RESET}\n")

