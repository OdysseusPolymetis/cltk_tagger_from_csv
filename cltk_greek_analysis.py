#!/usr/bin/env python
# coding: utf-8
# author: Marianne Reboul


from cltk import NLP
from cltk.alphabet.processes import LatinNormalizeProcess
from cltk.ner.processes import LatinNERProcess
from cltk.embeddings.processes import LatinEmbeddingsProcess
from cltk.stops.processes import StopsProcess
from cltk.alphabet.processes import GreekNormalizeProcess
from cltk.ner.processes import GreekNERProcess
from cltk.embeddings.processes import GreekEmbeddingsProcess
from cltk.lexicon.processes import LatinLexiconProcess
from nltk.tokenize import sent_tokenize
from cltk.morphology.morphosyntax import CLTKException
import glob
import pandas as pd
from pathlib import Path
import csv
import argparse
import sys
import os


print('This software analyzes a csv file input and generates a csv output with metadata, using cltk \nusage: --src path/to/your/source/directory')
parser = argparse.ArgumentParser()
parser.add_argument('--src', help= '/your/directory/to/csv/files/')
args = parser.parse_args()


if len(sys.argv) == 1:
    sys.exit()


argssrc= os.path.join(args.src, '')

author_grc=argssrc


cltk_nlp_greek = NLP(language="grc")
cltk_nlp_greek.pipeline.processes.remove(GreekEmbeddingsProcess)
cltk_nlp_greek.pipeline.processes.remove(StopsProcess)
cltk_nlp_greek.pipeline.processes.remove(GreekNERProcess)


files= glob.iglob(author_grc+'/**/*.csv', recursive=True)
headers=['Word','Lemma','POS','Case','Gender','Number','Aspect','Tense','VerbForm','Voice']
features=['Case','Gender','Number','Aspect','Tense','VerbForm','Voice']
for filename in files :
    path=Path(filename).stem
    if os.path.exists('./results/')==False:
        os.mkdir('./results/')
    with open('./results/'+path+'.csv', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(headers)
        print(filename)
        df = pd.read_csv(filename, sep='\t')
        words = df['word'].tolist()
        gktxt = " ".join(words)
        for sent in sent_tokenize(gktxt):
            cltk_doc = cltk_nlp_greek.analyze(text=sent)
            for word in cltk_doc.words:
                row=list()
                row.append(word.string)
                row.append(word.lemma)
                row.append(word.upos)
                for feature in features:
                    try:
                        row.append(word.features[feature])
                    except CLTKException:
                        row.append("")
            writer.writerow(row)
