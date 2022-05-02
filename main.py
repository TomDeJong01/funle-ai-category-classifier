from termcolor import colored
from scripts import predict, train, update, compare


def main_menu():
    help_text = "\nUse a command line argument to execute the specified action:\n"\
                "    -p: Make predictions on assignments without a categoryId\n"\
                "    -t: Retrain AI with updated dataset (the new ai wont has to be activated with -u)\n"\
                "    -c: Compare newly trained AI performance with performance of AI currently in use\n"\
                "    -u: Use newly trained AI\n"\
                "    -r: Restore old AI and discard AI currently in use)\n"
    while True:
        ai_command = input(help_text)
        if ai_command == "-h" or ai_command == "help" or ai_command == "--help":
            print(help_text)
        elif ai_command == "-p" or ai_command == "predict":
            predict.predict_main()
        elif ai_command == "-t" or ai_command == "train":
            train.train_main()
        elif ai_command == "-c" or ai_command == "compare":
            compare.compare_main()
        elif ai_command == "-u" or ai_command == "update":
            update.update_main()
        elif ai_command == "-r" or ai_command == "restore":
            update.restore()
        elif ai_command == "--test":
            print(colored("Dit is een testje, om de app iets te laten doen", "green"))
        else:
            print(colored("wrong input", "red"), f"\n{help_text}")
            continue


if __name__ == '__main__':
    main_menu()
