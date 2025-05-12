import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

from copy import deepcopy

warnings.filterwarnings("ignore")

# class ScoredToken():
#     """
#     Represents a token and a corresponding score, which should roughly translate to the likelihood this
#     token should be appended to some given generation sequence.
#     """
#     def __init__(self, token_id, score):
#         self.token_id = token_id
#         self.score = score

#     def __str__(self):
#         return f"{self.token_id}: {self.score: .8f}"
    
#     def __repr__(self):
#         return self.__str__()

class GeneratedSequence():
    """
    Represents a sequence in the process of being generated; an initial token, a potential end token, and a series of 
    ScoredTokens between them. This class also maintains the overall sequence score, which is the cumulative probability 
    of this generated sequence being the best output given some query.
    """
    def __init__(self, 
                # tokenizer, 
                # initial_token, 
                input_ids,
                end_token_id, 
                initial_score):
        # self.tokenizer = tokenizer
        self.end_token_id = end_token_id
        self.score = initial_score # Cumulative log probs of this sequence
        self.normalized_score = initial_score
        # self.sequence = [ScoredToken(initial_token, initial_score)]
        self.sequence = input_ids
    
    def append(self, token_id, score):
        """
        Append the given ScoredToken to this sequence; add its log-probability to this
        sequence's total cumulative log-prob
        """
        token_id = token_id.unsqueeze(0)
        self.sequence = torch.cat([self.sequence, token_id], dim=1)
        # self.sequence.append(ScoredToken(token_id, score))
        self.score += score
        # self.normalized_score = self.score / len(self.sequence.shape[1])
        self.normalized_score = self.score / self.sequence.shape[1]

    def ids(self):
        return [st for st in self.sequence[0, :]]

    # def tokens(self):
    #     return self.tokenizer.decode(torch.tensor(self.ids()), skip_special_tokens=True)
    
    def has_ended(self):
        """
        Returns True if the last token in this sequence is the end-of-sequence token ID
        """
        return self.sequence.size(1) > 0 and self.sequence[0, -1].item() == self.end_token_id

    def __str__(self):
        return f"{self.score: .8f}({self.normalized_score: .8f}): {self.sequence}"

    def __repr__(self):
        return self.__str__()
    
    def __copy__(self):
        gs = GeneratedSequence(self.tokenizer, None, self.end_token_id, 0.0)
        gs.sequence = self.sequence.copy()
        gs.score = self.score
        gs.normalized_score = self.normalized_score
        return gs

    def __iter__(self):
        return self.sequence.__iter__()
    
    def __lt__(self, other_sequence):
        return self.normalized_score < other_sequence.normalized_score

    def __le__(self, other_sequence):
        return self.normalized_score <= other_sequence.normalized_score

    def __eq__(self, other_sequence):
        return self.normalized_score - other_sequence.normalized_score <= 1e-5 and self.ids() == other_sequence.ids()
    
    def __ne__(self, other_sequence):
        return self.normalized_score - other_sequence.normalized_score > 1e-5 or self.ids() != other_sequence.ids()
    
    def __gt__(self, other_sequence):
        return self.normalized_score > other_sequence.normalized_score
    
    def __ge__(self, other_sequence):
        return self.normalized_score >= other_sequence.normalized_score

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        original_length = input_ids.shape[1]

        for _ in range(self.max_output_len):

            with torch.no_grad():
                predicted = self.model(input_ids)
            
            # print(predicted.logits.shape)
            next_token_logits = predicted.logits[0, -1, :]
            
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            if next_token_id.item() == self.eos_token_id:
                break
            
            next_token_id = next_token_id.unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
        
        return input_ids[0, original_length:]

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        original_length = input_ids.shape[1]
        
        candidate_sequences = [GeneratedSequence(input_ids, self.eos_token_id, 0.0)]
        
        tokens_generated = 0
        while tokens_generated  < self.max_output_len:
            next_step_candidates = []
            
            for candidate in candidate_sequences:
                if candidate.has_ended():
                    next_step_candidates.append(candidate)
                    continue
                with torch.no_grad():
                    medusa_logits, _, logits = self.model(candidate.sequence,
                                                                output_orig=True,
                                                                medusa_forward=True,
                                                                use_cache=True)
                
                # inside_candidate_sequences = list(deepcopy(candidate))
                inside_candidate_sequences = [GeneratedSequence(candidate.sequence, self.eos_token_id, candidate.score)]
                
                for i in range(self.no_heads):
                    
                    next_inside_candidatate = []
                    
                    for inside_candidate in inside_candidate_sequences:
                        if inside_candidate.has_ended():
                            next_inside_candidatate.append(inside_candidate)
                            continue
                    
                        if i == 0:
                            
                            probs = torch.softmax(logits[0, -1, :], dim=-1)
                            
                            topk_probs, topk_indices = torch.topk(probs, self.beam_width)
                            
                            for j in range(self.beam_width):
                                next_token_id = topk_indices[j].unsqueeze(0)
                                next_score = torch.log(topk_probs[j]).item()
                            
                                new_inside_seq = deepcopy(inside_candidate)
                                new_inside_seq.append(next_token_id, next_score)
                                next_inside_candidatate.append(new_inside_seq)
                            
                            # next_inside_candidatate.sort(reverse=True)
                            # inside_candidate_sequences = next_inside_candidatate[:self.beam_width]
                            
                        else:
                            probs = torch.softmax(medusa_logits[i-1, 0, -1, :], dim=-1)
                            
                            topk_probs, topk_indices = torch.topk(probs, self.beam_width)
                            
                            for j in range(self.beam_width):
                                next_token_id = topk_indices[j].unsqueeze(0)
                                next_score = torch.log(topk_probs[j]).item()
                            
                                new_inside_seq = deepcopy(inside_candidate)
                                new_inside_seq.append(next_token_id, next_score)
                                next_inside_candidatate.append(new_inside_seq)
                            
                            # next_inside_candidatate.sort(reverse=True)
                            # inside_candidate_sequences = next_inside_candidatate[:self.beam_width]
                        
                    next_inside_candidatate.sort(reverse=True)
                    inside_candidate_sequences = next_inside_candidatate[:self.beam_width]
                    
                    for seq in inside_candidate_sequences:
                        next_step_candidates.append(seq)
                
            next_step_candidates.sort(reverse=True)
            candidate_sequences = next_step_candidates[:self.beam_width]
            
            tokens_generated = tokens_generated + self.no_heads
        
        candidate_sequences.sort(reverse=True)
        results = list(candidate_sequences[0].ids())[original_length:]
        return torch.tensor(results)
            