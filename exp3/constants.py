from os.path import dirname, join, realpath, isfile

TRAIN, DEV, TEST = 'train', 'dev', 'test'

SHOULD_SHUFFLE_DURING_INFERENCE = False

# Datasets
COMETA = 'cometa'
MEDMENTIONS = 'medmentions'
BC5CDR_C = 'bc5cdr-chemical'
BC5CDR_D = 'bc5cdr-disease'
NCBI_D = 'ncbi-disease'
DATASETS = [BC5CDR_C, BC5CDR_D, NCBI_D, COMETA, MEDMENTIONS]
USE_TRAINDEV = True

# Cometa Dataset
STRATIFIED_SPECIFIC = 'stratified_specific'
STRATIFIED_GENERAL = 'stratified_general'
ZEROSHOT_GENERAL = 'zeroshot_general'
COMETA_REMOVE_EASY_CASES = False
COMETA_SETTING = STRATIFIED_GENERAL

# Nametypes
NAME_PRIMARY = 'primary'
NAME_SECONDARY = 'synonym/secondary'
NAMETYPES = [NAME_PRIMARY, NAME_SECONDARY]

# Ontologies File Path
BASE_ONTOLOGY_DIR = 'resources/ontologies'
SNOMEDCT_FP = 'resources/ontologies/snomedct.json'