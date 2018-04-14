""" This module implements code to support parameter pruning using various techniques. """


from ._masks import MASKED_WEIGHT_COLLECTION, MASK_COLLECTION,\
    apply_mask, masked_variable_getter, make_pruning_summaries,\
    assign_masked_values_to_weights, CommitMaskedValueHook
