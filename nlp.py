# import packages
import sys
import os
import csv
import re
import time
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

from eHostess.PyConTextInterface.SentenceSplitters import SpacySplitter
from eHostess.PyConTextInterface import PyConText
import targetsandmodifiers as tm

# input files
METADATA_FILE = "./data/TrainingNotes/oac_trainingnotes_metadata_expanded.csv"
TRAINING_FILE = "./data/oac_trainingset_csv.csv"

# output files
TARGETS_MODIFIERS = './results/'
TARGETS_FILE = 'file:///C:/Users/kevin.wood/Documents/OAC NLP Project/results/targets.tsv'
# 'file:///' + os.path.dirname(__file__) + '/TargetsAndModifiers/targets.tsv'
MODIFIERS_FILE = 'file:///C:/Users/kevin.wood/Documents/OAC NLP Project/results/modifiers.tsv'
RESULTS_FILE = "./results/results4.csv"
PHRASE_FILE = "./results/phrases_round4.csv"

# load data
metadata_frame = pd.read_csv(METADATA_FILE)
training_frame = pd.read_csv(TRAINING_FILE)

# load targets and modifiers
afib_targets_and_mods = tm.ModifiersAndTargets()

# targets
# afib_targets_and_mods.addTarget("afib", r"(?i)\bafib\b|\batrial\sfib|a-fib|a\.\sfib|a\.fib|\ba\sfib\b")
afib_targets_and_mods.addTarget("warfarin", r"(?i)\bwarfarin\b")
afib_targets_and_mods.addTarget("coumadin", r"(?i)\bcoumadin\b")
afib_targets_and_mods.addTarget("dabigatran", r"(?i)\bdabigatran\b|\bdabi\b")
afib_targets_and_mods.addTarget("pradaxa", r"(?i)\bpradaxa\b")
afib_targets_and_mods.addTarget("rivaroxaban", r"(?i)\brivaroxaban\b|\briva\b")
afib_targets_and_mods.addTarget("xarelto", r"(?i)\bxarelto\b")
afib_targets_and_mods.addTarget("eliquis", r"(?i)\beliquis\b|\belliquis\b")
afib_targets_and_mods.addTarget("apixaban", r"(?i)\bapixaban\b|\bapixa\b|\bapix\b")
afib_targets_and_mods.addTarget("savaysa", r"(?i)\bsavaysa\b")
afib_targets_and_mods.addTarget("edoxaban", r"(?i)\bedoxaban\b|\bedoxa\b|\bedox\b")
afib_targets_and_mods.addTarget("lovenox", r"(?i)\blovenox\b")
afib_targets_and_mods.addTarget("enoxaparin", r"(?i)\benoxaparin\b")
afib_targets_and_mods.addTarget("oac", r"(?i)\boral\santicoagulants?\b|\banticoags?\b|\banticoagulants?\b|\banticoagulation\b|\b(n|d|ts)?oacs?\b")

# modifiers
afib_targets_and_mods.addModifier("start", "AFFIRMED_EXISTENCE" , r"(?i)\bstart\b|\binitiate\b|\bbegin\b|\btake\b")
# afib_targets_and_mods.addModifier("none", r"(?i)\bnone\b", direction='backwards')
# afib_targets_and_mods.addModifier("negative", r"(?i)\bnegative\b")
# afib_targets_and_mods.addModifier("denies", r"(?i)denies|denied|denying")
# afib_targets_and_mods.addModifier("family", r"(?i)\bmother\b|\bfather\b|\bsister\b|\bbrother\b|\bdaughter\b|\bson\b|\baunt\b|\buncle\b|\bgranddaughter\b|\bgrandson\b", direction='bidirectional')
# afib_targets_and_mods.addModifier("rule_out", r"(?i)r/o|r\\o|\brule\s+out\b|\brules\s+out\b|\bruled\s+out\b")
# afib_targets_and_mods.addModifier("unlikely", r"(?i)\bunlikely\b", direction='bidirectional')
# afib_targets_and_mods.addModifier("without", r"(?i)\bunlikely\b")
# afib_targets_and_mods.addModifier("investigate", r"(?i)\binvestigate\b|\binvestigating\b")
# afib_targets_and_mods.addModifier("look_for", r"(?i)\blook\s+for\b\b")
# afib_targets_and_mods.addModifier("differential", r"(?i)\bdifferential\b\b|ddx", direction="bidirectional")
# afib_targets_and_mods.addModifier("possible", r"(?i)\bpossible\b", direction="bidirectional")
# afib_targets_and_mods.addModifier("holter", r"(?i)\b(holter|event)\s+(monitor(ing)?\s+)?ordered\s+for\b\b|ddx")
# afib_targets_and_mods.addModifier("etc", r"(?i)\betc\b", direction='backwards')
# afib_targets_and_mods.addModifier("screen_for", r"(?i)\bscreen\s+for\b")
# afib_targets_and_mods.addModifier("risk_of", r"(?i)\brisk\s+(of|for)\b")
# afib_targets_and_mods.addModifier("suspicious", r"(?i)\bsuspicious\b")
# afib_targets_and_mods.addModifier("question_of", r"(?i)\bquestion\s+of\b")

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

# write results
combined_frame = predictions_frame.merge(training_frame, 'left', on='mrn')
combined_frame.to_csv(RESULTS_FILE)

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

target_pattern = r"(?i)\bwarfarin\b|\bcoumadin|\bdabigatran|\bdabi\b|\bpradaxa\b|\brivaroxaban\b|\briva\b|\bxarelto\b|\beliquis\b|\belliquis\b|\bapixaban\b|\bapixa\b|\bapix\b|\bsavaysa\b|\bedoxaban\b|\bedoxa\b|\bedox\b|\blovenox\b|\boral\santicoagulants?\b|\banticoags?\b|\banticoagulants?\b|\banticoagulation\b|\b(n|d|ts)?oacs?\b"
modifier_pattern1 = r"(?i)\bstart\b|\binitiate\b|\bbegin\b"
modifier_pattern2 = r"(?i)\btake\b"

with open (PHRASE_FILE, 'w') as resultsfile:
    writer = csv.writer(resultsfile)
    writer.writerow(out_fieldnames)
    for index, row in metadata_frame.iterrows():
        mrn = row['mrn']
        note_id = row['noteid']
        note_date = row['note_date']
        # binary_adj_goldstd = list(combined_frame[combined_frame['mrn'] == mrn]['binary_adj_goldstd'])[0]
        mrn_predicted_class = list(combined_frame[combined_frame['mrn'] == mrn]['predicted_class'])[0]
        # if (binary_adj_goldstd != mrn_predicted_class):
        if not isinstance(row['text'], str):    # remove empty notes
            continue
        doc = nlp(row['text'])
        for sent in doc.sents:
            phrase = str(sent)
            target_matches, modifier_matches = afib_targets_and_mods.testText(phrase)
            if len(target_matches) > 0:
                targets = re.findall(target_pattern, phrase)
                modifiers = re.findall(modifier_pattern1, phrase) + re.findall(modifier_pattern2, phrase)
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
