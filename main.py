import sys
from scripts import predict, train, update, compare


def main(argv):
    try:
        if argv == "-h" or argv == "help" or argv == "--help":
            print("todo, help")
        elif argv == "-p" or argv == "predict":
            predict.predict_main()
            print("predict_complete")
        elif argv == "-t" or argv == "train":
            train.train_main()
            print("training complete")
        elif argv == "-c" or argv == "compare":
            compare.compare_main()
            print("compare")
        elif argv == "-u" or argv == "update":
            update.update_main()
            print("update")
        else:
            print(f"argv: {argv}\n--help for command line arguments\n")
    except ModuleNotFoundError as err:
        print(f"error raised: {err}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # try:
        if len(sys.argv) != 2:
            print(f"{len(sys.argv) -1} command line argument found, use --help for command line arguments")
        else:
            main(sys.argv[1])
    # except Exception as e:
    #     print(e)
    #     print("application crashed")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
