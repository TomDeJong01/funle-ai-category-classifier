import sys
from scripts import predict, train, update, compare


def main(argv):
    try:
        if argv == "-h" or argv == "help" or argv == "--help":
            print("""-p: Make predictions on assignments without a categoryId
            -t: Retrain AI with updated dataset (the new ai wont has to be activated with -u)
            -c: Compare newly trained AI performance with performance of AI currently in use
            -u: Use newly trained AI""")
        elif argv == "-p" or argv == "predict":
            predict.predict_main()
        elif argv == "-t" or argv == "train":
            train.train_main()
        elif argv == "-c" or argv == "compare":
            compare.compare_main()
        elif argv == "-u" or argv == "update":
            update.update_main()
        else:
            print(f"argv: {argv}\n--help for command line arguments\n")
    except ModuleNotFoundError as err:
        print(f"error raised: {err}")


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            print(f"{len(sys.argv) -1} command line argument found, use --help for command line arguments")
        else:
            main(sys.argv[1])
    except Exception as e:
        print("An unexpected error occurred")
        print(e)
        print("application crashed")
