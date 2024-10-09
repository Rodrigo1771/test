import os
import regex
import argparse
import humanize
from tqdm import tqdm
import lxml.etree as et


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='ISO 639-1 code of target language.')
    parser.add_argument('--max_corpus_size', type=int, default=1000000000, help='the maximum size of the corpus.')
    args = parser.parse_args()
    return args


def get_file_size(file_path):
    stat_info = os.stat(file_path)
    file_size = stat_info.st_size
    file_size_human_readable = humanize.naturalsize(file_size)
    return file_size_human_readable


def clean_text(text):
    text = regex.sub("(?s)<ref>.+?</ref>", "", text)  # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text)  # remove html tags
    text = regex.sub("&[a-z]+;", "", text)  # remove html entities
    text = regex.sub("\{\{(?:[^{}]++|(?R))*+\}\}", "", text)  # remove markup tags
    text = regex.sub("(?s){.+?}", "", text)  # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text)  # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text)  # remove media links

    text = regex.sub("[']{5}", "", text)  # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text)  # remove bold symbols
    text = regex.sub("[']{2}", "", text)  # remove italic symbols

    text = regex.sub(u"[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)  # mostly european languages
    text = text.lower()

    text = regex.sub("[ ]{2,}", " ", text)  # Squeeze spaces.

    return text


def build_corpus(args):
    corpus_file_path = f"data/{args.lang}/{args.lang}_corpus.txt"
    with open(corpus_file_path, 'w', encoding='utf-8') as fout:
        fname = f"{args.lang}_wikidump.xml"
        ns = "{http://www.mediawiki.org/xml/export-0.11/}"  # namespace
        for _, elem in tqdm(et.iterparse(f"data/{args.lang}/{fname}", tag=ns+"text"), desc="Building corpus"):
            running_text = elem.text
            try:
                running_text = clean_text(running_text)
                sents = regex.split("([.?!])?[\n]+|[.?!] ", running_text)
                for sent in sents:
                    if sent is not None:
                        words = sent.split()
                        if len(words) > 10:
                            fout.write(" ".join(words) + "\n")
            except:
                continue  # it's okay as we have a pretty big corpus!
            elem.clear()  # We need to save memory!
            fsize = os.path.getsize(corpus_file_path)
            if fsize > args.max_corpus_size:
                break
    print(f'Corpus size: {get_file_size(corpus_file_path)}')


if __name__ == "__main__":
    args = parse_args()
    build_corpus(args)
