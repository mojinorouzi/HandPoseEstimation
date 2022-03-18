
import numpy as np 
""" 
 Function Args
 index: the hand --- result # i.e 0 , 1 
 hand: the actual hand landmark  
 result: all detection from model  
"""
def get_label(index , hand , result , mp_hands):
    output = None
    for idx , classification in enumerate(result.multi_handedness):
        if classification.classification[0].index == index:
            # process result 
            label = classification.classification[0].label 
            score = classification.classification[0].score
            text = "{} {}".format(label, round(score ,2))
            # extract coordinates
            coords = tuple(np.multiply(np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x , hand.landmark[mp_hands.HandLandmark.WRIST].y)) , [640,480]).astype(int))
                           
            output = text , coords 
    return output