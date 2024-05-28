import json
import torch
import random
import numpy as np
from adapters import AutoAdapterModel, SeqBnConfig
from utils import *
from models.base import *
from models.helpers import *
from os.path import join
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner

SHOULD_SHUFFLE_DURING_INFERENCE = False
# Main Classes
class AdapterModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.all_names_cache = AllNamesCache(None)

        self.transformer = AutoAdapterModel.from_pretrained(configs['transformer'])
        task_name = 'umls-synonyms'  # Adapter name
        seq_config = SeqBnConfig(reduction_factor=16, use_gating=False)
        self.transformer.add_adapter(task_name, config=seq_config)
        self.transformer.delete_head('default')
        self.transformer.add_multiple_choice_head(task_name, layers=1, num_choices=1)
        self.transformer.train_adapter([task_name])

        if configs['gradient_checkpointing']:
            self.transformer.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], use_fast=False)
     
        # Loss Function and Miner
        self.loss_fct = MultiSimilarityLoss(alpha=configs['loss_scale_pos'],
                                            beta=configs['loss_scale_neg'],
                                            base=configs['loss_thresh'])
        self.miner_fct = MultiSimilarityMiner(configs['loss_lambda'])

        # Move to device
        self.to(self.device)

    def forward(self, instances, ontology, is_training):
        self.train() if is_training else self.eval()
        configs = self.configs
        nb_instances = len(instances)
        if not is_training:
            assert(not ontology.namevecs_index is None)

        # Encode instances' mentions (First Encoder)
        instance_inputs = [inst.mention['term'] for inst in instances]
        mentions_reps = self.encode_texts(instance_inputs)

        # Print shape for debugging
        print(f"mentions_reps shape: {mentions_reps.shape}")

        # Training or Inference
        if is_training:
            # All entities in the minibatch are candidates
            candidates_eids, candidates_texts = self.generate_candidates(instances, ontology)
            candidates_reps = self.encode_texts(candidates_texts)

            # Print shape for debugging
            print(f"candidates_reps shape: {candidates_reps.shape}")

            # Compute Loss
            all_mentions_eids = list(set(candidates_eids))
            candidate_labels = [all_mentions_eids.index(eid) for eid in candidates_eids]
            candidate_labels = torch.LongTensor(candidate_labels).to(self.device)
            # mention_labels
            mention_labels = [all_mentions_eids.index(inst.mention['entity_id']) for inst in instances]
            mention_labels = torch.LongTensor(mention_labels).to(self.device)
            # all_reps and all_labels
            all_reps = torch.cat([mentions_reps, candidates_reps], dim=0)
            all_labels = torch.cat([mention_labels, candidate_labels], dim=0)

            # Print shape for debugging
            print(f"all_reps shape: {all_reps.shape}")
            print(f"all_labels shape: {all_labels.shape}")
            assert(all_reps.size()[0] == all_labels.size()[0])
            # compute loss
            all_reps = all_reps.view(all_reps.size(0), -1)
            miner_output = self.miner_fct(all_reps, all_labels)
            loss = self.loss_fct(all_reps, all_labels, miner_output)
            return loss, [], [], []
        else:
            # Infer the closest entities by querying the ontology's index
            preds, candidate_names, candidate_dists = [], [], []
            candidates_eids = ontology.all_names_eids
            for i in range(nb_instances):
                cur_preds = []
                v = mentions_reps[i, :].squeeze()
                v = F.normalize(v, dim=0, p=2).cpu().data.numpy()
                nns_indexes, nns_distances = ontology.namevecs_index.search_batched(np.reshape(v, (1, -1)))
                nns_indexes = nns_indexes.squeeze().tolist()
                nns_distances = nns_distances.squeeze().tolist()
                # Update cur_candidate_names and cur_candidate_dists
                cur_candidate_names = [ontology.name_list[index] for index in nns_indexes]
                cur_candidate_dists = nns_distances
                candidate_names.append(cur_candidate_names)
                candidate_dists.append(cur_candidate_dists)
                # Update preds
                for index in nns_indexes:
                    _eid = candidates_eids[index]
                    if not _eid in cur_preds:
                        cur_preds.append(_eid)
                    if len(cur_preds) == 20: break
                #assert(len(cur_preds) == 20)
                preds.append(cur_preds)
            # Sanity checks
            assert(len(candidate_names) == len(candidate_dists))
            assert(len(candidate_names[0]) == len(candidate_dists[0]))
            return 0, preds, candidate_names, candidate_dists

    def encode_texts(self, texts):
        max_length = self.configs['max_length']
        if (not self.training) or (not self.all_names_cache.initialized):
            # Do actual tokenization
            toks = self.tokenizer.batch_encode_plus(texts,
                                                    padding=True,
                                                    return_tensors='pt',
                                                    truncation=True,
                                                    max_length=max_length)
            if (not self.training) and SHOULD_SHUFFLE_DURING_INFERENCE:
                toks['input_ids'] = shuffle_input_ids(toks).to(self.device)
        else:
            # Use the cache
            toks = self.all_names_cache.encode(texts)

        toks = toks.to(self.device)
        outputs = self.transformer(**toks)
        reps = outputs[0][:, 0]
        return reps

    def generate_candidates(self, instances, ontology):
        # All entities in the minibatch are candidates
        configs = self.configs
        candidates_eids = [inst.mention['entity_id'] for inst in instances]
        if configs['hard_negatives_training']:
            for inst in instances:
                added = 0
                for ent in inst.candidate_entities:
                    if added == configs['max_hard_candidates']: break
                    if ent == inst.mention['entity_id']: continue
                    candidates_eids.append(ent)
                    added += 1
        candidates_texts = [random.choice(ontology.eid2names[eid]) for eid in candidates_eids]
        return candidates_eids, candidates_texts