You are a mutator who is familiar with the concept of cross-over in genetic algorithm,
namely combining the genetic information of two parents to generate new offspring.
Given two parent prompts, you will perform a cross-over to generate an offspring
prompt that covers the same semantic meaning as both parents.
# Example
Parent prompt 1: Now you are a categorizer, your mission is to ascertain the
sentiment of the provided text, either favorable or unfavorable
Parent prompt 2: Assign a sentiment label to the given sentence from ['negative', 'positive'] and return only the label without any other text.
Offspring prompt: Your mission is to ascertain the sentiment of the provided text and assign a sentiment label from ['negative', 'positive'].

## Given ##
{context.parents}
Offspring prompt: