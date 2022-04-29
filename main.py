import sys
from termcolor import colored

from scripts import predict, train, update, compare


def main(argv):
    try:
        if argv == "-h" or argv == "help" or argv == "--help":
            print("\nUse a command line argument to execute the specified action:\n"
                  "    -p: Make predictions on assignments without a categoryId\n"
                  "    -t: Retrain AI with updated dataset (the new ai wont has to be activated with -u)\n"
                  "    -c: Compare newly trained AI performance with performance of AI currently in use\n"
                  "    -u: Use newly trained AI\n"
                  "    -r: Restore old AI and discard AI currently in use)\n")
        elif argv == "-p" or argv == "predict":
            predict.predict_main()
        elif argv == "-t" or argv == "train":
            train.train_main()
        elif argv == "-c" or argv == "compare":
            compare.compare_main()
        elif argv == "-u" or argv == "update":
            update.update_main()
        elif argv == "-r" or argv == "restore":
            update.restore()
        else:
            print(colored(f"Unknown argument: {argv}\n--help for command line arguments\n", "yellow"))
    except ModuleNotFoundError as err:
        print(colored(f"error raised: {err}", "red"))
    finally:
        return


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            print(colored(f"{len(sys.argv) - 1} command line argument found, "
                          "use --help for command line arguments", "yellow"))
        else:
            main(sys.argv[1])
    except Exception as e:
        print(colored("An unexpected error occurred", "red"))
        print(e)
        print(colored("application crashed", "red"))
