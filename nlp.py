# import packages
import sys
import os
import csv
import re
import time
import pandas as pd
import spacy
import gc
nlp = spacy.load('en_core_web_sm')

from eHostess.PyConTextInterface.SentenceSplitters import SpacySplitter
from eHostess.PyConTextInterface import PyConText
import targetsandmodifiers as tm

# input files
METADATA_FILE = "./data/TrainingNotes/training_metadata_1.csv"
TRAINING_FILE = "./data/oac_trainingset_csv.csv"

# output files
TARGETS_MODIFIERS = './results/'
TARGETS_FILE = 'file:///C:/Users/kevin.wood/Documents/OAC NLP Project/results/targets.tsv'
# 'file:///' + os.path.dirname(__file__) + '/TargetsAndModifiers/targets.tsv'
MODIFIERS_FILE = 'file:///C:/Users/kevin.wood/Documents/OAC NLP Project/results/modifiers.tsv'
RESULTS_FILE = "./results/results_training_classification.csv"
PHRASE_FILE = "./results/phrases_training_classification.csv"

# load data
metadata_frame = pd.read_csv(METADATA_FILE)
training_frame = pd.read_csv(TRAINING_FILE)

# load targets and modifiers
afib_targets_and_mods = tm.ModifiersAndTargets()

# targets
afib_targets_and_mods.addTarget("warfarin", r"(?i)\warf[a-z]+\b")
afib_targets_and_mods.addTarget("coumadin", r"(?i)\bcoum[a-z]+\b")
afib_targets_and_mods.addTarget("dabigatran", r"(?i)\bdabi[a-z]+\b|\bdabi\b")
afib_targets_and_mods.addTarget("pradaxa", r"(?i)\bprad[a-z]+\b")
afib_targets_and_mods.addTarget("rivaroxaban", r"(?i)\briva[a-z]+\b|\briva\b")
afib_targets_and_mods.addTarget("xarelto", r"(?i)\bxar[a-z]+\b")
afib_targets_and_mods.addTarget("eliquis", r"(?i)\beliq[a-z]+\b|\belliq[a-z]+\b")
afib_targets_and_mods.addTarget("apixaban", r"(?i)\bapix[a-z]+\b|\bapixa\b|\bapix\b")
afib_targets_and_mods.addTarget("savaysa", r"(?i)\bsavay[a-z+]\b")
afib_targets_and_mods.addTarget("edoxaban", r"(?i)\bedox[a-z]+\b|\bedoxa\b|\bedox\b")

# modifiers
afib_targets_and_mods.addModifier("start", "AFFIRMED_EXISTENCE" , r"(?i)\bstart\b|\binitiate\b|\bbegin\b|\btake\b", direction = 'forward')
afib_targets_and_mods.addModifier("stop", "NEGATED_EXISTENCE" , r"(?i)\bstop\b|\bdo not treat\b|\bdon't treat\b", direction = "forward")

afib_targets_and_mods.writeTargetsAndModifiers(TARGETS_MODIFIERS,
                                               targets_name="targets.tsv",
                                               modifiers_name="modifiers.tsv")

# create patient objects
mrns = metadata_frame['mrn'].unique()
patient_objs = []
for mrn in mrns:
    records = metadata_frame[metadata_frame['mrn'] == mrn]
    notes = []
    for row in records.itertuples():
        if not isinstance(row.text, str):    # remove empty notes
            continue
        notes.append((row.noteid, row.text))
    obj = {
        'mrn' : mrn,
        'positive_notes' : [],
        'notes' : notes
    }
    patient_objs.append(obj)

# process patient notes
def processDocuments(notes, positive_list):
    for note_tuple in notes:
        noteid = note_tuple[0]
        note_text = note_tuple[1]
        if note_text == None:
            continue
        input_obj = SpacySplitter.splitSentencesRawString(note_text, noteid)    #tuple contains text and noteid
        document = PyConText.PyConTextInterface.PerformAnnotation(input_obj,
                                                       targetFilePath=TARGETS_FILE,
                                                      modifiersFilePath=MODIFIERS_FILE,
                                                        modifierToClassMap={
                                                            "NEGATED_EXISTENCE" : "negative",
                                                            "AFFIRMED_EXISTENCE" : "positive"})
        for annotation in document.annotations:
            if annotation.annotationClass == 'positive':
                positive_list.append(noteid)
                break

# annotate patient notes
print('Starting annotation at: ', time.ctime())
num_patients = len(patient_objs)
count = 1
for patient_obj in patient_objs:
    processDocuments(patient_obj['notes'], patient_obj['positive_notes'])
    sys.stdout.write(f'\rCompleted {count} of {num_patients}. ({count / num_patients * 100:.2f}%)')
    count += 1
    gc.collect()
print('\nEnding annotation at: ', time.ctime())

# predict each mrn for atrial fibrillation
trimmed_objects = []
for patient_obj in patient_objs:
    trimmed_objects.append({'mrn': patient_obj['mrn'], 'positive_notes' : patient_obj['positive_notes']})
mrns = []
predictions = []
for patient_obj in trimmed_objects:
    if len(patient_obj['positive_notes']) > 0:
        mrns.append(patient_obj['mrn'])
        predictions.append(1)
    else:
        mrns.append(patient_obj['mrn'])
        predictions.append(0)
predictions_frame = pd.DataFrame({'mrn' : mrns, 'predicted_class': predictions})
predictions_frame.to_csv(RESULTS_FILE)

# write results
# combined_frame = predictions_frame.merge(training_frame, 'left', on='mrn')
# combined_frame.to_csv(RESULTS_FILE)
#
# write phrase prediction results
out_fieldnames = ['mrn',
                 'note_id',
                 'note_date',
                 # 'binary_adj_goldstd',
                 'mrn_predicted_class',
                 'phrase_predicted_class',
                 'phrase',
                 'targets',
                 'modifiers']

target_pattern = r"(?i)\bwarfarin\b|\bcoumadin|\bdabigatran|\bdabi\b|\bpradaxa\b|\brivaroxaban\b|\briva\b|\bxarelto\b|\beliquis\b|\belliquis\b|\bapixaban\b|\bapixa\b|\bapix\b|\bsavaysa\b|\bedoxaban\b|\bedoxa\b|\bedox\b"
modifier_pattern1 = r"(?i)\bstart\b|\binitiate\b|\bbegin\b|\btake\b"
# r"(?i)\bstop\b|\bdo not treat\b|\bdon't treat\b

with open (PHRASE_FILE, 'w') as resultsfile:
    writer = csv.writer(resultsfile)
    writer.writerow(out_fieldnames)
    for index, row in metadata_frame.iterrows():
        mrn = row['mrn']
        note_id = row['noteid']
        note_date = row['note_date']
        # binary_adj_goldstd = list(combined_frame[combined_frame['mrn'] == mrn]['binary_adj_goldstd'])[0]
        mrn_predicted_class = list(predictions_frame[predictions_frame['mrn'] == mrn]['predicted_class'])[0]
        # if (binary_adj_goldstd != mrn_predicted_class):
        if not isinstance(row['text'], str):    # remove empty notes
            continue
        doc = nlp(row['text'])
        for sent in doc.sents:
            phrase = str(sent)
            target_matches, modifier_matches = afib_targets_and_mods.testText(phrase)
            if len(target_matches) > 0:
                targets = re.findall(target_pattern, phrase)
                modifiers = re.findall(modifier_pattern1, phrase)
                if len(modifiers) > 0:
                    phrase_predicted_class = 0
                else:
                    phrase_predicted_class = 1
                # line = [mrn, note_id, note_date, binary_adj_goldstd, mrn_predicted_class, phrase_predicted_class,
                #        phrase, targets, modifiers]
                line = [mrn, note_id, note_date, mrn_predicted_class, phrase_predicted_class,
                       phrase, targets, modifiers]
                writer.writerow(line)
            else:
                continue

print ('Done!')
