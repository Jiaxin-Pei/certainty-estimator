from certainty_estimator.predict_certainty import CertaintyEstimator
from tqdm import tqdm

findings = ['Dopamine and serotonin are important for different forms of flexibility associated with receiving reward and punishment.',
 'The reason for this might be that the fetal central nervous system, which controls movements in general and facial movements in particular did not develop at the same rate and in the same manner as in fetuses of mothers who did not smoke during pregnancy.',
 'Mice lacking the tet1 gene were able to learn to navigate a water maze, but were unable to extinguish the memory.',
 'Tet1 exerts its effects on memory by altering the levels of dna methylation, a modification that controls access to genes.',
 'Mice lacking tet1 had much lower levels of hydroxymethylation -- an intermediate step in the removal of methylation -- in the hippocampus and the cortex, which are both key to learning and memory.',
 'A threshold level of methylation is necessary for gene expression to take place, and that the job of tet1 is to maintain low methylation, ensuring that the genes necessary for memory formation are poised and ready to turn on at the moment they are needed.',
 'Many teens ages 14 to 19 are actually addicted to the internet, particularly social media.',
 'Kids who spend more time on social media feel less fulfilled socially.',
 'Psychedelics, or substances like them, may be quite effective in treating depression, anxiety, and post-traumatic stress disorder.',
 'Using biotechnology was not unnatural, and would simply correct a change in the ecosystem created by the human hunting of rhinos.']

#construct a CertaintyEstimator for aspect-level certainty
estimator = CertaintyEstimator(task ='aspect-level',use_auth_token=False)
d = estimator.predict(findings)

for i in range(len(findings)):
    
    print(d[i], findings[i])
    
#construct a CertaintyEstimator for sentence-level certainty
sentence_estimator = CertaintyEstimator(task ='sentence-level',use_auth_token=False)
a = sentence_estimator.predict(findings,tqdm=tqdm)
for i in range(len(findings)):
    print(a[i], findings[i])