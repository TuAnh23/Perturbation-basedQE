import argparse


def main():
    parser = argparse.ArgumentParser('Format so that the vocab contains one word per line')
    parser.add_argument('--data_root_dir', type=str, default="data")
    parser.add_argument('--dataname', type=str, default="MuST-SHE-en2fr",
                        help="[Country_Adjective]")
    args = parser.parse_args()
    print(args)

    if args.dataname == "Country_Adjective":
        with open(f'{args.data_root_dir}/Country_Adjective.csv', 'r') as f:
            vocab = f.read()
        vocab = vocab.replace(',', '\n')

        with open(f'{args.data_root_dir}/Country_Adjective_vocab.txt', 'w') as f:
            f.write(vocab)


if __name__ == "__main__":
    main()
