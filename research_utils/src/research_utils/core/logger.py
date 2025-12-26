def print_args(args):
    print(f"\n===========[INFO] ARGS:===========")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")
    print(f"===================================\n")